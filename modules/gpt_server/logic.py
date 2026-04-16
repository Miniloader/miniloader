"""
gpt_server/logic.py — OpenAI-Compatible HTTP API Server
=========================================================
Runs a lightweight FastAPI / uvicorn server that exposes an
OpenAI-compatible ``/v1/chat/completions`` endpoint.  Incoming HTTP
requests are translated into ``CHAT_REQUEST`` payloads emitted to
``BRAIN_IN``.  Streamed ``BRAIN_STREAM_PAYLOAD`` tokens from the brain
are collected and returned as SSE or JSON responses.

The server only starts when a brain module is wired to ``BRAIN_IN``.
Without a wired brain the module stays in RUNNING (not-ready) state
and no HTTP process is spawned.

Thread-safety note
------------------
uvicorn runs its own asyncio event loop in a daemon thread.
Token queues are ``queue.Queue`` (stdlib, thread-safe) so that
``_on_brain_payload`` (main loop) can ``put_nowait`` and the uvicorn
handlers can ``run_in_executor(None, q.get, timeout)`` to await them
without any cross-loop asyncio issues.  The CHAT_REQUEST emit from
uvicorn's context uses ``asyncio.run_coroutine_threadsafe`` to schedule
on the main loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as stdlib_queue
import socket
import threading
import time
import uuid
from typing import Any

from core.base_module import BaseModule, ModuleStatus
from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    SignalType,
)

log = logging.getLogger(__name__)

_PORT_RELEASE_DELAY = 1.5  # Seconds to wait after shutdown for OS to release port (Windows)


class GptServerModule(BaseModule):
    MODULE_NAME = "gpt_server"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "OpenAI-compatible HTTP API server (facade over brain)"
    SENSITIVE_PARAMS = {"endpoint_password"}
    _server_thread: threading.Thread | None = None
    _uvicorn_server_ref: list = []

    def get_default_params(self) -> dict[str, Any]:
        return {
            "server_port": 5000,
            "cors_policy": "*",
            "server_active": False,
            "ctx_length": 4096,
            "endpoint_password": "",
        }

    def _auth_token(self) -> str | None:
        """Return active bearer token (UI-set password overrides vault token)."""
        custom = str(self.params.get("endpoint_password", "")).strip()
        if custom:
            return custom
        if self._vault is not None:
            try:
                return self._vault.get_or_create_secret(
                    self._vault.DEFAULT_GPT_SERVER_SECRET_KEY
                )
            except Exception:
                return self._vault.derive_api_token()
        return None

    # ── Ports ───────────────────────────────────────────────────

    def define_ports(self) -> None:
        self.add_input(
            "BRAIN_IN",
            accepted_signals={
                SignalType.CHAT_REQUEST,
                SignalType.BRAIN_STREAM_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional AI channel (AI_IN label). Sends assembled "
                "prompts (CHAT_REQUEST) to the inference engine "
                "and receives streamed tokens "
                "(BRAIN_STREAM_PAYLOAD) back."
            ),
        )
        self.add_output(
            "LOCAL_IP_OUT",
            accepted_signals={
                SignalType.SERVER_CONFIG_PAYLOAD,
                SignalType.BRAIN_STREAM_PAYLOAD,
                SignalType.CHAT_REQUEST,
            },
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=8,
            description=(
                "Bidirectional API channel (API_OUT label). Announces the "
                "server address (SERVER_CONFIG_PAYLOAD) and "
                "streams inference tokens to downstream clients "
                "(terminal, voice). Receives chat requests back "
                "through the same wire. Up to 8 clients."
            ),
        )
        self.add_output(
            "WEB_OUT",
            accepted_signals={
                SignalType.ROUTING_CONFIG_PAYLOAD,
                SignalType.TUNNEL_STATUS_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=1,
            description=(
                "Bidirectional web channel for tunnel integration. Emits local "
                "routing config and receives tunnel status updates."
            ),
        )

    # ── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        # Thread-safe token queues keyed by request_id
        self._pending_requests: dict[str, stdlib_queue.Queue] = {}
        self._request_stats: dict[str, dict[str, Any]] = {}
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server_ref: list = []  # [Server] for shutdown from main thread
        # Store the main event loop so uvicorn thread can schedule back onto it
        self._main_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.params["server_active"] = False
        if self._vault is not None:
            # Keep card display and server auth in sync with the vault-backed endpoint key.
            self.params["endpoint_password"] = self._auth_token() or ""
        self.status = ModuleStatus.RUNNING

    async def check_ready(self) -> bool:
        return self._brain_is_wired()

    def _brain_is_wired(self) -> bool:
        brain_port = self.inputs.get("BRAIN_IN")
        if brain_port is None:
            return False
        return len(brain_port.connected_wires) > 0

    async def init(self) -> None:
        if not self._brain_is_wired():
            log.info("gpt_server: no brain wired to BRAIN_IN — server not started")
            return
        if self._server_thread is not None and self._server_thread.is_alive():
            # Re-broadcast config on INIT so downstream modules recover after rewiring.
            await self.broadcast_server_config()
            await self.broadcast_routing_config()
            return
        await self._start_server()

    async def broadcast_server_config(self) -> None:
        """Emit the current server config to downstream LOCAL_IP consumers."""
        if not self.params.get("server_active", False):
            return
        port = int(self.params.get("server_port", 5000))
        ctx_length = int(self.params.get("ctx_length", 4096))
        await self.outputs["LOCAL_IP_OUT"].emit(
            Payload(
                signal_type=SignalType.SERVER_CONFIG_PAYLOAD,
                source_module=self.module_id,
                data={
                    "server_status": "ready",
                    "protocol": "http",
                    "host": "127.0.0.1",
                    "port": port,
                    "api_base": "/v1",
                    "ctx_length": ctx_length,
                    "supported_endpoints": [
                        "/v1/chat/completions",
                        "/v1/models",
                        "/v1/health",
                    ],
                    "capabilities": {"supports_streaming": True},
                },
            )
        )

    async def broadcast_routing_config(self) -> None:
        """Emit current HTTP routing config for tunnel modules."""
        if not self.params.get("server_active", False):
            return
        port = int(self.params.get("server_port", 5000))
        await self.outputs["WEB_OUT"].emit(
            Payload(
                signal_type=SignalType.ROUTING_CONFIG_PAYLOAD,
                source_module=self.module_id,
                data={
                    "target_protocol": "http",
                    "target_host": "127.0.0.1",
                    "target_port": port,
                    "health_check_endpoint": "/v1/health",
                    "auth_required": bool(self._auth_token()),
                },
            )
        )

    def _port_in_use(self, port: int) -> bool:
        """Check whether something is already listening on the port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0

    async def _start_server(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        # Clear any stale listeners from prior power-off/on cycles to avoid duplicate tokens
        self.inputs["BRAIN_IN"]._listeners.clear()
        self.inputs["BRAIN_IN"].on_receive(self._on_brain_payload)

        port = int(self.params.get("server_port", 5000))
        if self._port_in_use(port):
            log.error(
                "gpt_server: port %d already in use — kill the process holding it "
                "or change server_port in module settings",
                port,
            )
            self.status = ModuleStatus.ERROR
            await self.report_state(
                severity="ERROR",
                message=f"Port {port} in use — change server_port or kill orphan process",
            )
            return

        cors = str(self.params.get("cors_policy", "*"))
        module_ref = self
        main_loop = self._main_loop

        def _run() -> None:
            import uvicorn
            from fastapi import FastAPI, Body
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse, JSONResponse

            app = FastAPI(title="Miniloader GPT Server", version="0.1.0")

            if cors == "*":
                origins = ["*"]
            elif cors == "localhost":
                origins = ["http://localhost", "http://127.0.0.1"]
            else:
                origins = []

            if origins:
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=origins,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

            from core.auth_middleware import BearerAuthMiddleware
            app.add_middleware(
                BearerAuthMiddleware,
                token_provider=lambda: module_ref._auth_token() or "",
                enabled_check=lambda: bool(module_ref._auth_token()),
            )

            @app.get("/v1/models")
            async def list_models() -> JSONResponse:
                return JSONResponse({
                    "object": "list",
                    "data": [{
                        "id": "miniloader-brain",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "miniloader",
                    }],
                })

            @app.post("/v1/chat/completions")
            async def chat_completions(
                request: dict = Body(...),
            ) -> StreamingResponse | JSONResponse:
                def _content_preview(content: Any) -> str:
                    """Return safe text preview for string or multimodal list content."""
                    if isinstance(content, list):
                        parts: list[str] = []
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "text":
                                parts.append(str(part.get("text", "")))
                        return " ".join(parts)[:50].replace("\n", " ")
                    return str(content or "")[:50].replace("\n", " ")

                messages = request.get("messages", [])
                do_stream = request.get("stream", False)
                request_id = request.get("request_id") or str(uuid.uuid4())
                thread_id = request.get("thread_id")
                tools = request.get("tools")
                tool_choice = request.get("tool_choice")
                if not isinstance(tools, list):
                    tools = None
                else:
                    tools = [row for row in tools if isinstance(row, dict)]
                if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
                    tool_choice = None
                created = int(time.time())
                log.info(
                    "gpt_server: HTTP request received request_id=%s stream=%s messages=%d",
                    request_id, do_stream, len(messages),
                )

                # Thread-safe queue: main loop puts tokens, we get them here
                token_q: stdlib_queue.Queue = stdlib_queue.Queue()
                module_ref._pending_requests[request_id] = token_q
                last_user = next(
                    (m for m in reversed(messages) if m.get("role") == "user"),
                    None,
                )
                prompt_preview = _content_preview(last_user.get("content", "")) if last_user else "request"
                module_ref._request_stats[request_id] = {
                    "tokens": 0,
                    "first_logged": False,
                    "prompt_preview": prompt_preview,
                }

                # Emit CHAT_REQUEST on the main event loop (cross-loop safe)
                payload = Payload(
                    signal_type=SignalType.CHAT_REQUEST,
                    source_module=module_ref.module_id,
                    data={
                        "messages": messages,
                        "request_id": request_id,
                        "thread_id": thread_id,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    },
                )
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        module_ref.inputs["BRAIN_IN"].emit(payload),
                        main_loop,
                    )
                    # Wait in a thread executor so we don't block uvicorn's loop
                    uvicorn_loop = asyncio.get_running_loop()
                    await uvicorn_loop.run_in_executor(None, future.result, 10.0)
                    log.info("gpt_server: emitted CHAT_REQUEST request_id=%s", request_id)
                except Exception as exc:
                    module_ref._pending_requests.pop(request_id, None)
                    module_ref._request_stats.pop(request_id, None)
                    log.error("gpt_server: emit failed request_id=%s error=%s", request_id, exc)
                    return JSONResponse(
                        {"error": {"message": str(exc), "type": "server_error"}},
                        status_code=502,
                    )

                model_name = "miniloader-brain"

                if do_stream:
                    return StreamingResponse(
                        _stream_sse(request_id, token_q, created, model_name),
                        media_type="text/event-stream",
                    )
                else:
                    return await _collect_response(
                        request_id, token_q, created, model_name,
                    )

            async def _stream_sse(
                request_id: str,
                q: stdlib_queue.Queue,
                created: int,
                model_name: str,
            ):
                """Yield SSE chunks. Blocks a thread-pool thread per token."""
                completion_id = f"chatcmpl-{request_id[:8]}"
                loop = asyncio.get_running_loop()
                try:
                    while True:
                        try:
                            data = await loop.run_in_executor(
                                None, lambda: q.get(timeout=120.0)
                            )
                        except stdlib_queue.Empty:
                            log.warning(
                                "gpt_server: token wait timeout request_id=%s (120s)",
                                request_id,
                            )
                            break

                        if data.get("error"):
                            log.error(
                                "gpt_server: stream error payload request_id=%s error=%s",
                                request_id, data.get("error"),
                            )
                            # Signal error then stop
                            err_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }],
                            }
                            yield f"data: {json.dumps(err_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        if data.get("done"):
                            log.info("gpt_server: stream done request_id=%s", request_id)
                            final_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }],
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        token = data.get("token", "")
                        if token:
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": token,
                                    },
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                finally:
                    module_ref._pending_requests.pop(request_id, None)
                    module_ref._request_stats.pop(request_id, None)

            async def _collect_response(
                request_id: str,
                q: stdlib_queue.Queue,
                created: int,
                model_name: str,
            ) -> JSONResponse:
                tokens: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                finish_reason = "stop"
                prompt_tokens = 0
                completion_tokens = 0
                loop = asyncio.get_running_loop()
                try:
                    while True:
                        try:
                            data = await loop.run_in_executor(
                                None, lambda: q.get(timeout=120.0)
                            )
                        except stdlib_queue.Empty:
                            log.warning(
                                "gpt_server: non-stream timeout request_id=%s (120s)",
                                request_id,
                            )
                            break

                        if data.get("error"):
                            return JSONResponse(
                                {"error": {
                                    "message": data["error"],
                                    "type": "server_error",
                                }},
                                status_code=502,
                            )
                        if data.get("done"):
                            prompt_tokens = data.get("prompt_tokens", 0)
                            completion_tokens = data.get(
                                "completion_tokens", len(tokens)
                            )
                            if data.get("tool_calls"):
                                tc = data.get("tool_calls")
                                if isinstance(tc, list):
                                    tool_calls = [row for row in tc if isinstance(row, dict)]
                                    finish_reason = "tool_calls"
                            break
                        if data.get("tool_calls"):
                            tc = data.get("tool_calls")
                            if isinstance(tc, list):
                                tool_calls = [row for row in tc if isinstance(row, dict)]
                                finish_reason = "tool_calls"
                        token = data.get("token", "")
                        if token:
                            tokens.append(token)
                finally:
                    module_ref._pending_requests.pop(request_id, None)
                    module_ref._request_stats.pop(request_id, None)

                response: dict[str, Any] = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "".join(tokens),
                        },
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
                if tool_calls:
                    response["choices"][0]["message"]["tool_calls"] = tool_calls
                return JSONResponse(response)

            @app.get("/health")
            @app.get("/v1/health")
            async def health() -> JSONResponse:
                return JSONResponse({"status": "ok"})

            ssl_kwargs: dict = {}
            if module_ref.params.get("tls_enabled", False):
                from core.cert_manager import CertManager
                cert_path, key_path = CertManager().ensure_certs()
                ssl_kwargs = {"ssl_keyfile": str(key_path), "ssl_certfile": str(cert_path)}

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                **ssl_kwargs,
            )
            server = uvicorn.Server(config)
            module_ref._uvicorn_server_ref.append(server)
            server.run()

        self._server_thread = threading.Thread(target=_run, daemon=True, name="gpt-server")
        self._server_thread.start()
        self.params["server_active"] = True
        log.info("gpt_server: HTTP API started on 127.0.0.1:%d", port)

        await self.broadcast_server_config()
        await self.broadcast_routing_config()

    async def _on_brain_payload(self, payload: Payload) -> None:
        """
        Listener on BRAIN_IN — runs in the main event loop.
        Routes BRAIN_STREAM_PAYLOAD tokens to the correct HTTP request's
        thread-safe queue (``put_nowait`` is safe from any thread/loop).
        """
        if payload.signal_type == SignalType.BRAIN_STREAM_PAYLOAD:
            request_id = payload.data.get("request_id", "")
            q = self._pending_requests.get(request_id)
            stats = self._request_stats.get(request_id)
            if q is not None:
                q.put_nowait(payload.data)  # thread-safe stdlib Queue
            if stats is not None:
                if payload.data.get("token"):
                    stats["tokens"] += 1
                    if not stats["first_logged"]:
                        stats["first_logged"] = True
                        log.info(
                            "gpt_server: first token from brain request_id=%s",
                            request_id,
                        )
                if payload.data.get("done"):
                    log.info(
                        "gpt_server: brain done request_id=%s tokens=%d error=%s",
                        request_id,
                        int(stats.get("tokens", 0)),
                        bool(payload.data.get("error")),
                    )
                    # Feed inference log for brain card (concise previews)
                    brain_id = payload.source_module
                    if brain_id and self._hypervisor is not None:
                        resp = (payload.data.get("full_response") or "")[:50].replace(
                            "\n", " "
                        )
                        if payload.data.get("error"):
                            resp = f"err: {payload.data.get('error', '')[:30]}"
                        self._hypervisor.ingest_inference(
                            brain_id,
                            stats.get("prompt_preview", "request"),
                            resp,
                            int(payload.data.get("prompt_tokens", 0)),
                            int(payload.data.get("completion_tokens", 0)),
                            float(payload.data.get("ttft_s", 0.0)),
                            float(payload.data.get("total_s", 0.0)),
                        )

            # Also forward to LOCAL_IP_OUT for downstream rack modules
            await self.outputs["LOCAL_IP_OUT"].emit(payload)

        elif payload.signal_type == SignalType.CHAT_REQUEST:
            # Forward rack-originated requests to the brain
            await self.inputs["BRAIN_IN"].emit(payload)

    async def process(self, payload: Payload, source_port: Port) -> None:
        if payload.signal_type == SignalType.CHAT_REQUEST:
            await self.inputs["BRAIN_IN"].emit(payload)
        elif payload.signal_type == SignalType.BRAIN_STREAM_PAYLOAD:
            await self.outputs["LOCAL_IP_OUT"].emit(payload)
        elif payload.signal_type == SignalType.TUNNEL_STATUS_PAYLOAD:
            # Tunnel module reports public URL/status through WEB_OUT channel.
            if payload.data.get("public_url"):
                self.params["public_url"] = payload.data.get("public_url", "")
            if payload.data.get("tunnel_status"):
                self.params["tunnel_status"] = payload.data.get("tunnel_status", "")

    async def shutdown(self) -> None:
        self.params["server_active"] = False
        uvicorn_refs = getattr(self, "_uvicorn_server_ref", [])
        if uvicorn_refs:
            server = uvicorn_refs[0]
            if server is not None:
                server.should_exit = True
        server_thread = getattr(self, "_server_thread", None)
        if server_thread is not None and server_thread.is_alive():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: server_thread.join(timeout=5.0)
            )
        uvicorn_refs.clear()
        self._server_thread = None
        # Give OS time to release port (especially on Windows)
        await asyncio.sleep(_PORT_RELEASE_DELAY)
        self.status = ModuleStatus.STOPPED
        log.info("gpt_server: HTTP API stopped, port released")


def register(hypervisor: Any) -> None:
    module = GptServerModule()
    hypervisor.register_module(module)
