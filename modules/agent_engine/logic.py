"""
agent_engine/logic.py — Central agent loop and orchestration
============================================================
Owns the model-facing turn loop, tool registry, and client event protocol.
LLM calls are strictly routed through gpt_server's HTTP endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from core.base_module import BaseModule, ModuleStatus
from core.port_system import ConnectionMode, Payload, Port, SignalType

log = logging.getLogger(__name__)

_TOOL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9]*_[a-z0-9][a-z0-9_]*$")
_LOG_PREVIEW_CHARS = 400
_LLM_OPTIONAL_KEYS = frozenset(
    {
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "repeat_penalty",
        "grammar",
        "json_schema",
        "logit_bias",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "tfs_z",
        "typical_p",
        "penalize_nl",
        "n_predict",
        "n_keep",
        "cache_prompt",
        "moe_enabled",
    }
)


class AgentEngineModule(BaseModule):
    MODULE_NAME = "agent_engine"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Centralized LLM + tools orchestration engine"
    UI_COL_SPAN = 2

    def get_default_params(self) -> dict[str, Any]:
        return {
            "tools_count": 0,
            "tools_discovered": False,
            "llm_status": "disconnected",
            "llm_host": "127.0.0.1",
            "llm_port": 5000,
            "llm_protocol": "http",
            "llm_model": "local",
            "ctx_length": 4096,
            "active_requests": 0,
            "missing_links": ["API_IN"],
            "active_preset": "",
            "system_prompt": "",
            "max_tool_rounds": 3,
            "tool_timeout": 30.0,
        }

    def define_ports(self) -> None:
        self.add_input(
            "API_IN",
            accepted_signals={
                SignalType.SERVER_CONFIG_PAYLOAD,
                SignalType.BRAIN_STREAM_PAYLOAD,
                SignalType.CHAT_REQUEST,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "API provider channel. Wired to gpt_server.LOCAL_IP_OUT. "
                "Consumes SERVER_CONFIG_PAYLOAD; ignores BRAIN_STREAM_PAYLOAD "
                "and CHAT_REQUEST which transit the same CHANNEL."
            ),
        )
        self.add_output(
            "AGENT_OUT",
            accepted_signals={SignalType.AGENT_PAYLOAD},
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=8,
            description=(
                "Bidirectional client channel. Receives chat requests/cancel from "
                "clients and emits agent stream/status events."
            ),
        )

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        self._tool_registry: dict[str, dict[str, Any]] = {}
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._active_turns: dict[str, asyncio.Task[Any]] = {}

        self._llm_host = str(self.params.get("llm_host", "127.0.0.1"))
        self._llm_port = int(self.params.get("llm_port", 5000))
        self._llm_protocol = str(self.params.get("llm_protocol", "http")).strip().lower()
        self._llm_model = str(self.params.get("llm_model", "local"))
        self._llm_ctx_length = int(self.params.get("ctx_length", 4096))
        self._llm_linked = False
        self._llm_reachable = False
        self._llm_peer_module_id = ""
        self._active_preset: dict[str, Any] = {}

        self.outputs["AGENT_OUT"]._listeners.clear()
        self.outputs["AGENT_OUT"].on_receive(self._on_agent_payload)

        self.status = ModuleStatus.RUNNING
        await self.refresh_wiring_state()

    def load_preset(self, preset: dict[str, Any]) -> None:
        """Apply a preset to the agent's runtime configuration."""
        if not isinstance(preset, dict):
            return
        normalized = self._normalize_preset(preset)
        self._active_preset = normalized
        self.params["active_preset"] = str(normalized.get("name", "")).strip()
        self.params["system_prompt"] = str(normalized.get("system_prompt", "")).strip()
        tool_cfg = normalized.get("tool_config", {})
        if isinstance(tool_cfg, dict):
            max_rounds = tool_cfg.get("max_rounds")
            timeout_secs = tool_cfg.get("timeout_secs")
            if max_rounds is not None:
                self.params["max_tool_rounds"] = self._coerce_positive_int(max_rounds, fallback=3)
            if timeout_secs is not None:
                self.params["tool_timeout"] = self._coerce_positive_float(timeout_secs, fallback=30.0)

    async def init(self) -> None:
        await self.refresh_wiring_state()

    async def process(self, payload: Payload, source_port: Port) -> None:
        data = payload.data if isinstance(payload.data, dict) else {}

        if source_port.name == "API_IN" and payload.signal_type == SignalType.SERVER_CONFIG_PAYLOAD:
            source = self._hypervisor.active_modules.get(payload.source_module) if self._hypervisor else None
            if source is None or source.MODULE_NAME != "gpt_server":
                return
            protocol = str(data.get("protocol", self._llm_protocol)).strip().lower()
            if protocol in {"http", "https"}:
                self._llm_protocol = protocol
            self._llm_host = str(data.get("host", self._llm_host))
            self._llm_port = int(data.get("port", self._llm_port))
            self._llm_model = str(data.get("model", self._llm_model) or self._llm_model)
            self._llm_ctx_length = int(data.get("ctx_length", self._llm_ctx_length))
            self._llm_peer_module_id = payload.source_module
            self._llm_linked = True
            self.params["llm_status"] = "linked"
            asyncio.create_task(self._probe_llm_health())
            await self._emit_status()
            return

    async def check_ready(self) -> bool:
        if not hasattr(self, "_llm_reachable"):
            return False
        linked = self._llm_wire_active()
        self._llm_linked = linked
        if not linked:
            self._llm_reachable = False
            self.params["llm_status"] = "disconnected"
            return False
        if self._llm_reachable:
            return True
        return await self._probe_llm_health()

    async def refresh_wiring_state(self) -> None:
        self._llm_linked = self._llm_wire_active()
        if not self._llm_linked:
            self._llm_reachable = False
        elif not self._llm_reachable:
            await self._probe_llm_health()
        self.params["llm_status"] = "ready" if self._llm_reachable else ("linked" if self._llm_linked else "disconnected")
        await self._emit_status()

    async def shutdown(self) -> None:
        active_turns = getattr(self, "_active_turns", {})
        for task in list(active_turns.values()):
            task.cancel()
        for task in list(active_turns.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        active_turns.clear()
        self.params["active_requests"] = 0
        self.status = ModuleStatus.STOPPED

    async def _on_agent_payload(self, payload: Payload) -> None:
        if payload.signal_type != SignalType.AGENT_PAYLOAD:
            return
        data = payload.data if isinstance(payload.data, dict) else {}
        action = str(data.get("action", "")).strip().lower()
        request_id = str(data.get("request_id", "")).strip()

        if action == "chat_request":
            active_turns = getattr(self, "_active_turns", {})
            if not request_id:
                request_id = f"agent_{int(time.time() * 1000)}"
                data["request_id"] = request_id
            input_method = str(data.get("input_method", "")).strip().lower()
            log.info(
                "agent_engine: received chat_request request_id=%s input_method=%s messages=%d llm_linked=%s llm_reachable=%s active_turns=%d",
                request_id,
                input_method or "keyboard",
                len(data.get("messages", [])),
                self._llm_linked,
                self._llm_reachable,
                len(active_turns),
            )
            if request_id in active_turns:
                await self._emit_agent(
                    action="error",
                    request_id=request_id,
                    error="request already active",
                )
                return
            task = asyncio.create_task(self._handle_turn(data))
            active_turns[request_id] = task
            self.params["active_requests"] = len(active_turns)
            task.add_done_callback(lambda _: active_turns.pop(request_id, None))
            task.add_done_callback(lambda _: self.params.__setitem__("active_requests", len(active_turns)))
            return

        if action == "cancel" and request_id:
            active_turns = getattr(self, "_active_turns", {})
            active = active_turns.pop(request_id, None)
            if active is not None:
                active.cancel()
                log.info("agent_engine: cancelled active turn request_id=%s", request_id)
            else:
                log.info("agent_engine: cancel for unknown/completed request_id=%s (no-op)", request_id)
            self.params["active_requests"] = len(active_turns)
            return

        if action == "tools_available":
            await self._handle_tools_available(data)
            await self._emit_status()
            return

        if action == "tool_result":
            await self._handle_tool_result_data(data, source_module=payload.source_module)
            return

    async def _handle_turn(self, request: dict[str, Any]) -> None:
        request_id = str(request.get("request_id", "")).strip()
        messages = request.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        input_method = str(request.get("input_method", "")).strip().lower()
        thread_id = str(request.get("thread_id", "")).strip()
        user_text = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict) and str(last.get("role", "")).strip().lower() == "user":
                content = last.get("content", "")
                if isinstance(content, str):
                    user_text = content.strip()
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and str(part.get("type", "")).strip() == "text":
                            text_parts.append(str(part.get("text", "")))
                    user_text = " ".join(text_parts).strip()
        if not request_id:
            return

        if not self._llm_linked:
            log.warning("agent_engine: turn blocked — API_IN not wired request_id=%s", request_id)
            await self._emit_agent(action="error", request_id=request_id, error="API_IN is not wired to gpt_server")
            return

        if not self._llm_reachable and not await self._probe_llm_health():
            log.warning(
                "agent_engine: turn blocked — gpt_server unreachable request_id=%s host=%s port=%s",
                request_id,
                self._llm_host,
                self._llm_port,
            )
            await self._emit_agent(
                action="error",
                request_id=request_id,
                error=f"gpt_server unreachable at {self._llm_host}:{self._llm_port}",
            )
            return

        log.info(
            "agent_engine: emitting stream_start request_id=%s input_method=%s user_text_chars=%d",
            request_id,
            input_method or "keyboard",
            len(user_text),
        )
        await self._emit_agent(
            action="stream_start",
            request_id=request_id,
            input_method=input_method,
            thread_id=thread_id,
            user_text=user_text,
        )

        preset = self._active_preset if isinstance(self._active_preset, dict) else {}
        preset_sampling = preset.get("sampling", {})
        if not isinstance(preset_sampling, dict):
            preset_sampling = {}
        model_override = str(preset.get("model_override", "")).strip()
        resolved_model = (
            request.get("model")
            or model_override
            or self._llm_model
            or "local"
        )
        resolved_temperature = request.get(
            "temperature",
            preset_sampling.get("temperature", 0.7),
        )
        base_payload: dict[str, Any] = {
            "request_id": request_id,
            "thread_id": request.get("thread_id"),
            "model": resolved_model,
            "temperature": float(resolved_temperature),
        }
        for key in _LLM_OPTIONAL_KEYS:
            if key in preset_sampling and preset_sampling[key] is not None:
                base_payload[key] = preset_sampling[key]
        for key in _LLM_OPTIONAL_KEYS:
            if key in request and request[key] is not None:
                base_payload[key] = request[key]

        tools = self._llm_tools_payload()
        has_tools = bool(tools)
        log.info(
            "agent_engine: turn start request_id=%s messages=%d tools_enabled=%s tools_count=%d",
            request_id,
            len(messages),
            has_tools,
            len(tools),
        )
        if has_tools:
            base_payload["tools"] = tools
            base_payload["tool_choice"] = "auto"

        llm_url = f"{self._llm_base_url()}/v1/chat/completions"
        full_text_parts: list[str] = []
        finish_reason = "stop"
        working_messages = self._messages_with_system_prompt(
            messages,
            request=request,
            preset=preset,
        )

        client_kwargs = self._local_httpx_kwargs(self._llm_protocol)
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                if has_tools:
                    seen_tool_signatures: set[tuple[str, str]] = set()
                    max_tool_rounds = self._max_tool_rounds()
                    for round_idx in range(max_tool_rounds):
                        log.info(
                            "agent_engine: probe round start request_id=%s round=%d working_messages=%d",
                            request_id,
                            round_idx,
                            len(working_messages),
                        )
                        probe_payload = dict(base_payload)
                        probe_payload["messages"] = working_messages
                        probe_payload["stream"] = False
                        probe_resp = await client.post(
                            llm_url,
                            json=probe_payload,
                            headers=self._llm_auth_headers(),
                            timeout=None,
                        )
                        if probe_resp.status_code != 200:
                            await self._emit_agent(
                                action="error",
                                request_id=request_id,
                                error=f"LLM {probe_resp.status_code}: {probe_resp.text}",
                            )
                            log.error(
                                "agent_engine: probe failed request_id=%s round=%d status=%d body=%s",
                                request_id,
                                round_idx,
                                probe_resp.status_code,
                                self._log_preview(probe_resp.text),
                            )
                            return

                        probe_data = probe_resp.json()
                        choice0 = (probe_data.get("choices") or [{}])[0]
                        assistant_msg = choice0.get("message") or {}
                        tool_calls = list(assistant_msg.get("tool_calls") or [])
                        finish_reason_probe = str(choice0.get("finish_reason") or "")
                        assistant_content = str(assistant_msg.get("content") or "")
                        content_preview = re.sub(r"\s+", " ", assistant_content).strip()
                        if len(content_preview) > 280:
                            content_preview = content_preview[:280] + "..."
                        log.info(
                            "agent_engine: probe round=%d request_id=%s "
                            "finish_reason=%s tool_calls=%d content_len=%d",
                            round_idx,
                            request_id,
                            finish_reason_probe,
                            len(tool_calls),
                            len(assistant_content),
                        )
                        if not tool_calls:
                            log.info(
                                "agent_engine: probe no-tool-calls request_id=%s "
                                "content_preview=%r",
                                request_id,
                                content_preview,
                            )
                        if not tool_calls:
                            break

                        duplicate_round = False
                        for call in tool_calls:
                            fn_chk = str(
                                ((call.get("function") or {}).get("name")) or ""
                            ).strip().lower()
                            args_chk = self._decode_tool_args(
                                (call.get("function") or {}).get("arguments")
                            )
                            sig_key = (
                                fn_chk,
                                json.dumps(args_chk, sort_keys=True, default=str),
                            )
                            if sig_key in seen_tool_signatures:
                                log.warning(
                                    "agent_engine: duplicate tool call suppressed "
                                    "request_id=%s round=%d tool=%s args=%s",
                                    request_id,
                                    round_idx,
                                    fn_chk,
                                    self._log_preview(args_chk),
                                )
                                duplicate_round = True
                                break
                            seen_tool_signatures.add(sig_key)
                        if duplicate_round:
                            break

                        working_messages.append(
                            {
                                "role": "assistant",
                                "content": assistant_msg.get("content", "") or "",
                                "tool_calls": tool_calls,
                            }
                        )

                        for call in tool_calls:
                            fn = str(((call.get("function") or {}).get("name")) or "").strip().lower()
                            call_id = str(call.get("id") or f"tool_{request_id}_{round_idx}")
                            args = self._decode_tool_args((call.get("function") or {}).get("arguments"))
                            log.info(
                                "agent_engine: tool dispatch request_id=%s round=%d call_id=%s tool=%s args=%s",
                                request_id,
                                round_idx,
                                call_id,
                                fn,
                                self._log_preview(args),
                            )

                            await self._emit_agent(
                                action="tool_call",
                                request_id=request_id,
                                tool_call_id=call_id,
                                tool_name=fn,
                                status="pending",
                                arguments=args,
                            )

                            tool_result = await self._execute_registered_tool(
                                tool_name=fn,
                                arguments=args,
                                tool_call_id=call_id,
                                request_id=request_id,
                                timeout_secs=self._tool_timeout_secs(),
                            )
                            log.info(
                                "agent_engine: tool result request_id=%s round=%d call_id=%s tool=%s success=%s error=%s",
                                request_id,
                                round_idx,
                                call_id,
                                fn,
                                bool(tool_result.get("success", False)),
                                self._log_preview(tool_result.get("error")),
                            )

                            await self._emit_agent(
                                action="tool_call",
                                request_id=request_id,
                                tool_call_id=call_id,
                                tool_name=fn,
                                status="completed",
                                result=tool_result,
                            )

                            working_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "name": fn,
                                    "content": self._tool_result_content_for_llm(
                                        tool_result
                                    ),
                                }
                            )
                    else:
                        log.warning(
                            "agent_engine: max tool rounds exceeded request_id=%s rounds=%d",
                            request_id,
                            max_tool_rounds,
                        )
                        await self._emit_agent(
                            action="error",
                            request_id=request_id,
                            error=f"Exceeded max tool rounds ({max_tool_rounds})",
                        )
                        return

                stream_payload = dict(base_payload)
                stream_payload["messages"] = working_messages
                stream_payload["stream"] = True
                async with client.stream(
                    "POST",
                    llm_url,
                    json=stream_payload,
                    headers=self._llm_auth_headers(),
                    timeout=None,
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="replace")
                        await self._emit_agent(
                            action="error",
                            request_id=request_id,
                            error=f"LLM {resp.status_code}: {body}",
                        )
                        log.error(
                            "agent_engine: stream request failed request_id=%s status=%d body=%s",
                            request_id,
                            resp.status_code,
                            self._log_preview(body),
                        )
                        return

                    line_buffer = ""
                    async for chunk in resp.aiter_text():
                        if not chunk:
                            continue
                        line_buffer += chunk
                        while "\n" in line_buffer:
                            line, line_buffer = line_buffer.split("\n", 1)
                            token, maybe_finish = self._parse_sse_line(line)
                            if token:
                                full_text_parts.append(token)
                                await self._emit_agent(
                                    action="token",
                                    request_id=request_id,
                                    content=token,
                                )
                            if maybe_finish:
                                finish_reason = maybe_finish
                    if line_buffer:
                        token, maybe_finish = self._parse_sse_line(line_buffer)
                        if token:
                            full_text_parts.append(token)
                            await self._emit_agent(action="token", request_id=request_id, content=token)
                        if maybe_finish:
                            finish_reason = maybe_finish

        except asyncio.CancelledError:
            await self._emit_agent(action="error", request_id=request_id, error="request cancelled")
            log.info("agent_engine: turn cancelled request_id=%s", request_id)
            raise
        except Exception as exc:
            await self._emit_agent(action="error", request_id=request_id, error=str(exc))
            log.exception("agent_engine: turn exception request_id=%s", request_id)
            return

        log.info(
            "agent_engine: turn end request_id=%s finish_reason=%s chars=%d",
            request_id,
            finish_reason,
            len("".join(full_text_parts)),
        )
        await self._emit_agent(
            action="stream_end",
            request_id=request_id,
            full_text="".join(full_text_parts),
            finish_reason=finish_reason,
            input_method=input_method,
            thread_id=thread_id,
            user_text=user_text,
        )

    async def _handle_tools_available(self, data: dict[str, Any]) -> None:
        tools = data.get("tools", [])
        if not isinstance(tools, list):
            return

        registry: dict[str, dict[str, Any]] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip().lower()
            if not self._is_namespaced_tool_name(name):
                continue
            registry[name] = {
                "tool_name": name,
                "provider": name.split("_", 1)[0],
                "description": str(function.get("description", "")),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                "updated_at": time.time(),
            }

        self._tool_registry = registry
        self.params["tools_count"] = len(self._tool_registry)
        self.params["tools_discovered"] = bool(self._tool_registry)
        log.info("agent_engine: tools synchronized from terminal total=%d", len(self._tool_registry))

    async def _handle_tool_result_data(self, data: dict[str, Any], *, source_module: str) -> None:
        call_id = str(data.get("tool_call_id", "")).strip()
        if not call_id:
            return
        pending = self._pending_tool_calls.get(call_id)
        if pending is None:
            log.warning(
                "agent_engine: tool result with unknown call_id=%s source=%s",
                call_id,
                source_module,
            )
            return
        pending["result"] = data
        log.info(
            "agent_engine: tool result received call_id=%s source=%s success=%s error=%s",
            call_id,
            source_module,
            bool(data.get("success", False)),
            self._log_preview(data.get("error")),
        )
        event = pending.get("event")
        if isinstance(event, asyncio.Event):
            event.set()

    def _llm_tools_payload(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        disabled_tools = set()
        preset_cfg = self._active_preset if isinstance(self._active_preset, dict) else {}
        tool_cfg = preset_cfg.get("tool_config", {})
        if isinstance(tool_cfg, dict):
            raw_disabled = tool_cfg.get("disabled_tools", [])
            if isinstance(raw_disabled, list):
                disabled_tools = {
                    str(tool_name).strip().lower()
                    for tool_name in raw_disabled
                    if str(tool_name).strip()
                }
        for name in sorted(self._tool_registry):
            if name in disabled_tools:
                continue
            entry = self._tool_registry.get(name, {})
            if not isinstance(entry, dict):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.get("tool_name", name),
                        "description": entry.get("description", ""),
                        "parameters": entry.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return tools

    async def _execute_registered_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        request_id: str,
        timeout_secs: float,
    ) -> dict[str, Any]:
        if tool_name not in self._tool_registry:
            log.warning(
                "agent_engine: execute tool not found request_id=%s call_id=%s tool=%s",
                request_id,
                tool_call_id,
                tool_name,
            )
            return {
                "success": False,
                "error": {"code": "TOOL_NOT_FOUND", "message": f"'{tool_name}' not in registry"},
            }

        event = asyncio.Event()
        started_at = time.time()
        self._pending_tool_calls[tool_call_id] = {
            "event": event,
            "tool_name": tool_name,
            "requested_at": started_at,
            "requested_by": "chat:" + request_id,
        }
        log.info(
            "agent_engine: execute tool request_id=%s call_id=%s tool=%s args=%s",
            request_id,
            tool_call_id,
            tool_name,
            self._log_preview(arguments),
        )

        await self._emit_agent(
            action="tool_execute",
            request_id=request_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_secs)
        except asyncio.TimeoutError:
            self._pending_tool_calls.pop(tool_call_id, None)
            log.warning(
                "agent_engine: execute tool timeout request_id=%s call_id=%s tool=%s wait_s=%.2f",
                request_id,
                tool_call_id,
                tool_name,
                time.time() - started_at,
            )
            return {
                "success": False,
                "error": {
                    "code": "TOOL_TIMEOUT",
                    "message": f"Timed out after {timeout_secs:.0f}s",
                },
            }

        pending = self._pending_tool_calls.pop(tool_call_id, None)
        if not isinstance(pending, dict):
            log.warning(
                "agent_engine: execute tool missing pending record request_id=%s call_id=%s tool=%s",
                request_id,
                tool_call_id,
                tool_name,
            )
            return {"success": False, "error": {"code": "TOOL_RESULT_MISSING", "message": "No result"}}
        result_payload = pending.get("result", {})
        if not isinstance(result_payload, dict):
            log.warning(
                "agent_engine: execute tool invalid result payload request_id=%s call_id=%s tool=%s payload_type=%s",
                request_id,
                tool_call_id,
                tool_name,
                type(result_payload).__name__,
            )
            return {"success": False, "error": {"code": "TOOL_RESULT_INVALID", "message": "Invalid result"}}
        log.info(
            "agent_engine: execute tool complete request_id=%s call_id=%s tool=%s success=%s elapsed_s=%.2f",
            request_id,
            tool_call_id,
            tool_name,
            bool(result_payload.get("success", False)),
            time.time() - started_at,
        )
        return {
            "success": bool(result_payload.get("success", False)),
            "result": result_payload.get("result"),
            "error": result_payload.get("error"),
            "tool_name": tool_name,
        }

    async def _emit_agent(self, *, action: str, request_id: str = "", **extra: Any) -> None:
        payload_data: dict[str, Any] = {"action": action}
        if request_id:
            payload_data["request_id"] = request_id
        payload_data.update(extra)
        await self.outputs["AGENT_OUT"].emit(
            Payload(
                signal_type=SignalType.AGENT_PAYLOAD,
                source_module=self.module_id,
                data=payload_data,
            )
        )

    async def _emit_status(self) -> None:
        if not hasattr(self, "_llm_linked"):
            return

        missing_links: list[str] = []
        if not self._llm_wire_active():
            missing_links.append("API_IN")

        payload = {
            "llm_linked": self._llm_linked,
            "llm_ready": self._llm_reachable,
            "llm_host": self._llm_host,
            "llm_port": self._llm_port,
            "llm_protocol": self._llm_protocol,
            "llm_model": self._llm_model or "local",
            "ctx_length": self._llm_ctx_length,
            "tools_count": len(self._tool_registry),
            "tools_providers": sorted({str(v.get("provider", "")) for v in self._tool_registry.values() if isinstance(v, dict)}),
            "tools_registry": self._tools_registry_by_provider(),
            "missing_links": missing_links,
            "active_requests": len(getattr(self, "_active_turns", {})),
            "active_preset": str(self.params.get("active_preset", "")).strip(),
        }
        self.params["missing_links"] = missing_links
        self.params["tools_count"] = len(self._tool_registry)
        self.params["tools_registry"] = payload["tools_registry"]
        await self._emit_agent(action="status", **payload)

    async def _probe_llm_health(self) -> bool:
        base = self._llm_base_url()
        headers = self._llm_auth_headers()
        client_kwargs = self._local_httpx_kwargs(self._llm_protocol)
        async with httpx.AsyncClient(**client_kwargs) as client:
            for endpoint in (f"{base}/v1/health", f"{base}/health"):
                try:
                    r = await client.get(endpoint, timeout=1.5, headers=headers)
                    if r.is_success:
                        self._llm_reachable = True
                        self.params["llm_status"] = "ready"
                        return True
                except Exception:
                    continue
        self._llm_reachable = False
        self.params["llm_status"] = "linked" if self._llm_linked else "disconnected"
        return False

    def _llm_wire_active(self) -> bool:
        if self._hypervisor is None:
            return False
        for wire in self._hypervisor.active_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name in {"API_IN", "LLM_IN"}:
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name in {"API_IN", "LLM_IN"}:
                peer_id = src.owner_module_id
            else:
                continue
            peer = self._hypervisor.active_modules.get(peer_id)
            if peer is None or not peer.enabled:
                continue
            if peer.MODULE_NAME != "gpt_server":
                continue
            if peer.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            self._llm_peer_module_id = peer_id
            return True
        return False

    def _llm_base_url(self) -> str:
        return f"{self._llm_protocol}://{self._llm_host}:{self._llm_port}"

    def _llm_auth_headers(self) -> dict[str, str]:
        if self._vault is not None:
            token = self._vault.get_or_create_secret(self._vault.DEFAULT_GPT_SERVER_SECRET_KEY)
            return {"Authorization": f"Bearer {token}"}
        return {}

    @staticmethod
    def _local_httpx_kwargs(protocol: str) -> dict[str, Any]:
        if protocol == "https":
            return {"verify": False}
        return {}

    @staticmethod
    def _decode_tool_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    @staticmethod
    def _tool_result_content_for_llm(tool_result: dict[str, Any]) -> str:
        """Serialize tool output for the model: inner payload on success, clear
        error JSON on failure (avoid nesting success/result/tool_name twice)."""
        if tool_result.get("success"):
            inner = tool_result.get("result")
            try:
                return json.dumps(inner, ensure_ascii=False, default=str)
            except TypeError:
                return json.dumps({"data": str(inner)}, ensure_ascii=False)
        err = tool_result.get("error")
        if isinstance(err, dict):
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        return json.dumps(
            {"success": False, "error": str(err or "unknown")},
            ensure_ascii=False,
        )

    @staticmethod
    def _log_preview(value: Any, max_chars: int = _LOG_PREVIEW_CHARS) -> str:
        """Compact JSON-safe preview for logs."""
        try:
            if isinstance(value, str):
                rendered = value
            else:
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            rendered = repr(value)
        rendered = re.sub(r"\s+", " ", str(rendered)).strip()
        if len(rendered) > max_chars:
            return rendered[:max_chars] + "..."
        return rendered

    @staticmethod
    def _parse_sse_line(line: str) -> tuple[str, str | None]:
        cleaned = line.strip()
        if not cleaned.startswith("data:"):
            return "", None
        raw = cleaned[5:].strip()
        if not raw or raw == "[DONE]":
            return "", None
        try:
            payload = json.loads(raw)
        except Exception:
            return "", None
        choices = payload.get("choices") or []
        if not choices:
            return "", None
        choice0 = choices[0] or {}
        delta = choice0.get("delta") or {}
        token = str(delta.get("content") or "")
        finish = choice0.get("finish_reason")
        if finish is not None:
            return token, str(finish)
        return token, None

    def _is_namespaced_tool_name(self, name: str) -> bool:
        return bool(_TOOL_NAME_RE.match(name))

    def _tools_registry_by_provider(self) -> dict[str, list[dict[str, Any]]]:
        """Return tool metadata grouped by provider for frontend display."""
        by_provider: dict[str, list[dict[str, Any]]] = {}
        for name, entry in sorted(self._tool_registry.items()):
            if not isinstance(entry, dict):
                continue
            provider = str(entry.get("provider", "unknown"))
            by_provider.setdefault(provider, []).append(
                {
                    "name": name,
                    "short_name": name.split("_", 1)[1] if "_" in name else name,
                    "description": str(entry.get("description", "")),
                }
            )
        return by_provider

    def _max_tool_rounds(self) -> int:
        return self._coerce_positive_int(self.params.get("max_tool_rounds", 3), fallback=3)

    def _tool_timeout_secs(self) -> float:
        return self._coerce_positive_float(self.params.get("tool_timeout", 30.0), fallback=30.0)

    def _messages_with_system_prompt(
        self,
        messages: list[Any],
        *,
        request: dict[str, Any],
        preset: dict[str, Any],
    ) -> list[Any]:
        prompt = self._resolved_system_prompt(request=request, preset=preset)
        if not prompt:
            return list(messages)
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role == "system":
                return list(messages)
        return [{"role": "system", "content": prompt}, *list(messages)]

    def _resolved_system_prompt(self, *, request: dict[str, Any], preset: dict[str, Any]) -> str:
        prompt = str(
            request.get("system_prompt")
            or preset.get("system_prompt")
            or self.params.get("system_prompt", "")
        ).strip()
        if not prompt:
            return ""
        user_name = str(request.get("user_name", "")).strip() or "user"
        replacements = {
            "{{date}}": time.strftime("%Y-%m-%d"),
            "{{user_name}}": user_name,
        }
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def _normalize_preset(self, preset: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "name": str(preset.get("name", "")).strip(),
            "description": str(preset.get("description", "")).strip(),
            "system_prompt": str(preset.get("system_prompt", "")).strip(),
            "model_override": preset.get("model_override"),
            "sampling": {},
            "tool_config": {
                "max_rounds": 3,
                "timeout_secs": 30.0,
                "disabled_tools": [],
            },
        }
        sampling = preset.get("sampling", {})
        if isinstance(sampling, dict):
            for key, value in sampling.items():
                if key == "temperature" or key in _LLM_OPTIONAL_KEYS:
                    normalized["sampling"][key] = value
        tool_cfg = preset.get("tool_config", {})
        if isinstance(tool_cfg, dict):
            if "max_rounds" in tool_cfg:
                normalized["tool_config"]["max_rounds"] = self._coerce_positive_int(
                    tool_cfg.get("max_rounds"),
                    fallback=3,
                )
            if "timeout_secs" in tool_cfg:
                normalized["tool_config"]["timeout_secs"] = self._coerce_positive_float(
                    tool_cfg.get("timeout_secs"),
                    fallback=30.0,
                )
            disabled_tools = tool_cfg.get("disabled_tools", [])
            if isinstance(disabled_tools, list):
                normalized["tool_config"]["disabled_tools"] = [
                    str(name).strip().lower()
                    for name in disabled_tools
                    if str(name).strip()
                ]
        return normalized

    @staticmethod
    def _coerce_positive_int(value: Any, *, fallback: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return fallback
        return parsed if parsed > 0 else fallback

    @staticmethod
    def _coerce_positive_float(value: Any, *, fallback: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return fallback
        return parsed if parsed > 0 else fallback


def register(hypervisor: Any) -> None:
    module = AgentEngineModule()
    hypervisor.register_module(module)

