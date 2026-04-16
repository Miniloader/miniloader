"""
discord_terminal/logic.py — Discord Chat Relay
==============================================
Discord-based chat terminal that mirrors gpt_terminal rack wiring and forwards
messages to an OpenAI-compatible backend (typically gpt_server).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any

import httpx

from core.base_module import BaseModule, ModuleStatus
from core.port_system import ConnectionMode, Payload, Port, SignalType

try:
    import discord
except Exception:  # pragma: no cover - import failure is handled at runtime
    discord = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_MAX_CHANNEL_HISTORY = 14
_DISCORD_MAX_MESSAGE = 2000
_RAG_AUTO_INJECT_THRESHOLD = 0.28
_DEFAULT_HISTORY_WINDOW_MESSAGES = 10
_DEFAULT_HISTORY_TOKEN_RATIO = 0.35
_CHANNEL_ID_RE = re.compile(r"\d{15,25}")
_CHANNEL_URL_RE = re.compile(r"discord(?:app)?\.com/channels/(?:@me|\d{15,25})/(\d{15,25})")
_MAX_TOOL_ROUNDS = 3


class DiscordTerminalModule(BaseModule):
    MODULE_NAME = "discord_terminal"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Discord bot terminal for chat requests and LLM forwarding"
    UI_COL_SPAN = 2
    SENSITIVE_PARAMS = {"discord_bot_token", "companion_secret"}
    _TOKEN_OWNERS: dict[str, str] = {}

    def get_default_params(self) -> dict[str, Any]:
        return {
            "discord_bot_token": "",
            "channel_ids": "",
            "require_mention": True,
            "system_prompt": "You are a helpful assistant.",
            "use_channel_history": True,
            "history_window_messages": _DEFAULT_HISTORY_WINDOW_MESSAGES,
            "history_token_ratio": _DEFAULT_HISTORY_TOKEN_RATIO,
            "save_history": True,
            "rag_enabled": True,
            "max_history_messages": _MAX_CHANNEL_HISTORY,
            "bot_connected": False,
            "bot_user": "",
            "llm_status": "disconnected",
            "tools_count": 0,
            "tools_discovered": False,
            "public_url": "",
            "tunnel_status": "inactive",
            "companion_url": "",
            "companion_secret": "",
            "companion_lease_seconds": 35,
            "companion_heartbeat_interval_sec": 10,
        }

    def define_ports(self) -> None:
        self.add_input(
            "AGENT_IN",
            accepted_signals={SignalType.AGENT_PAYLOAD},
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional agent channel. Sends chat_request/cancel and receives "
                "stream/status/tool events from agent_engine."
            ),
        )
        self.add_input(
            "LOCAL_IP_IN",
            accepted_signals={
                SignalType.SERVER_CONFIG_PAYLOAD,
                SignalType.BRAIN_STREAM_PAYLOAD,
                SignalType.CHAT_REQUEST,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional API channel. Receives inference server config and "
                "stream payloads while allowing upstream chat request forwarding."
            ),
        )
        self.add_input(
            "DB_IN_OUT",
            accepted_signals={
                SignalType.DB_QUERY_PAYLOAD,
                SignalType.DB_TRANSACTION_PAYLOAD,
                SignalType.DB_RESPONSE_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description="Bidirectional storage link for compatibility with gpt_terminal wiring.",
        )
        self.add_input(
            "CONTEXT_IN",
            accepted_signals={
                SignalType.QUERY_PAYLOAD,
                SignalType.CONTEXT_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional RAG link. Sends query payloads and receives semantic context."
            ),
        )
        self.add_input(
            "MCP_IN",
            accepted_signals={
                SignalType.TOOL_SCHEMA_PAYLOAD,
                SignalType.TOOL_EXECUTION_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=4,
            description="Bidirectional tools channel for schema and execution payloads.",
        )
        self.add_output(
            "WEB_OUT",
            accepted_signals={
                SignalType.ROUTING_CONFIG_PAYLOAD,
                SignalType.TUNNEL_STATUS_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description="Bidirectional tunnel channel retained for rack-level compatibility.",
        )

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING

        # Kill any prior bot instance before resetting references so we never
        # orphan a running discord.py client that keeps receiving messages.
        old_client = getattr(self, "_discord_client", None)
        old_task = getattr(self, "_bot_task", None)
        if old_client is not None or old_task is not None:
            await self._stop_bot()

        self._discord_client: discord.Client | None = None
        self._bot_task: asyncio.Task | None = None
        self._active_bot_token = ""

        self._llm_host = "127.0.0.1"
        self._llm_port = 5000
        self._llm_protocol = "http"
        self._llm_model: str | None = None
        self._llm_ctx_length = 4096
        self._llm_linked = False
        self._llm_reachable = False

        self._token_buffer: list[dict[str, Any]] = []
        self._pending_rag: dict[str, tuple[asyncio.Event, dict[str, Any]]] = {}
        self._tool_registry: dict[str, dict[str, Any]] = {}
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._channel_locks: dict[int, asyncio.Lock] = {}
        self._pending_agent_streams: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._agent_status: dict[str, Any] = {}
        self._companion_attached = False
        self._companion_heartbeat_task: asyncio.Task | None = None

        self.params["bot_connected"] = False
        self.params["bot_user"] = ""
        self.params["tools_count"] = 0
        self.params["tools_discovered"] = False
        self.params["llm_status"] = "disconnected"

        if discord is None:
            self.status = ModuleStatus.ERROR
            await self.report_state(
                severity="ERROR",
                message="discord.py is not installed. Run pip install discord.py.",
            )
            return

        await self.refresh_wiring_state()
        await self._attach_to_companion()
        await self._maybe_start_bot()
        await self._ensure_companion_heartbeat()
        self.status = ModuleStatus.RUNNING

    async def init(self) -> None:
        await self.refresh_wiring_state()
        await self._attach_to_companion()
        await self._maybe_start_bot()
        await self._ensure_companion_heartbeat()

    async def check_ready(self) -> bool:
        """READY only when a usable Discord token is configured."""
        return bool(self._bot_token().strip())

    async def process(self, payload: Payload, source_port: Port) -> None:
        if payload.signal_type == SignalType.SERVER_CONFIG_PAYLOAD:
            data = payload.data if isinstance(payload.data, dict) else {}
            protocol = str(data.get("protocol", self._llm_protocol)).strip().lower()
            if protocol in {"http", "https"}:
                self._llm_protocol = protocol
            self._llm_host = str(data.get("host", self._llm_host))
            self._llm_port = int(data.get("port", self._llm_port))
            # If config arrives, treat LOCAL_IP wiring as present even when wire
            # topology heuristics don't match every provider name.
            self._llm_linked = True
            if "model" in data:
                self._llm_model = str(data.get("model", "")).strip() or None
            if "ctx_length" in data:
                self._llm_ctx_length = int(data["ctx_length"])
            self.params["llm_status"] = "linked"
            asyncio.create_task(self._probe_llm_health())
            return

        if payload.signal_type == SignalType.BRAIN_STREAM_PAYLOAD:
            data = payload.data if isinstance(payload.data, dict) else {}
            if data.get("done"):
                self._token_buffer.clear()
            elif data.get("token"):
                self._token_buffer.append(data)
            return

        if payload.signal_type == SignalType.TUNNEL_STATUS_PAYLOAD:
            data = payload.data if isinstance(payload.data, dict) else {}
            self.params["public_url"] = str(data.get("public_url", "") or "")
            self.params["tunnel_status"] = str(
                data.get("tunnel_status", data.get("status", "unknown"))
            )
            return

        if payload.signal_type == SignalType.CONTEXT_PAYLOAD:
            rid = str(payload.data.get("request_id", "") or "")
            pending = self._pending_rag.get(rid)
            if pending is not None:
                event, holder = pending
                holder.update(payload.data if isinstance(payload.data, dict) else {})
                event.set()
            return

        if payload.signal_type == SignalType.TOOL_SCHEMA_PAYLOAD:
            await self._register_tools(payload)
            return

        if payload.signal_type == SignalType.TOOL_EXECUTION_PAYLOAD:
            action = str(payload.data.get("action", "")).strip().lower()
            if action == "result":
                await self._handle_tool_result(payload)
            return

        if payload.signal_type == SignalType.CHAT_REQUEST:
            # Kept to mirror gpt_terminal port contract. Discord-side requests use HTTP.
            return
        if payload.signal_type == SignalType.AGENT_PAYLOAD:
            await self._handle_agent_payload(payload)
            return

    async def _handle_agent_payload(self, payload: Payload) -> None:
        data = payload.data if isinstance(payload.data, dict) else {}
        action = str(data.get("action", "")).strip().lower()
        request_id = str(data.get("request_id", "")).strip()
        if action == "status":
            self._agent_status = dict(data)
            self._llm_linked = bool(data.get("llm_linked", self._llm_linked))
            self._llm_reachable = bool(data.get("llm_ready", self._llm_reachable))
            self.params["llm_status"] = "ready" if self._llm_reachable else ("linked" if self._llm_linked else "disconnected")
            return
        q = self._pending_agent_streams.get(request_id)
        if q is not None:
            try:
                q.put_nowait(dict(data))
            except Exception:
                pass

    async def shutdown(self) -> None:
        await self._stop_companion_heartbeat()
        await self._stop_bot()
        await self._detach_from_companion()
        getattr(self, "_pending_rag", {}).clear()
        getattr(self, "_pending_tool_calls", {}).clear()
        self.status = ModuleStatus.STOPPED

    async def refresh_wiring_state(self) -> None:
        linked = self._agent_wire_active() or self._llm_wire_active()
        self._llm_linked = linked
        self.params["llm_status"] = "linked" if linked else "disconnected"

    def _agent_wire_active(self) -> bool:
        if self._hypervisor is None:
            return False
        for wire in self._hypervisor.active_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name == "AGENT_IN":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name == "AGENT_IN":
                peer_id = src.owner_module_id
            else:
                continue
            peer = self._hypervisor.active_modules.get(peer_id)
            if (
                peer is not None
                and peer.enabled
                and peer.MODULE_NAME == "agent_engine"
                and peer.status in {ModuleStatus.RUNNING, ModuleStatus.READY}
            ):
                return True
        return False

    def _has_local_ip_wire(self) -> bool:
        local_ip_in = self.inputs.get("LOCAL_IP_IN")
        if local_ip_in is None:
            return False
        return len(local_ip_in.connected_wires) > 0

    def _llm_wire_active(self) -> bool:
        if self._hypervisor is None:
            return False
        for wire in self._hypervisor.active_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name == "LOCAL_IP_IN":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name == "LOCAL_IP_IN":
                peer_id = src.owner_module_id
            else:
                continue
            peer = self._hypervisor.active_modules.get(peer_id)
            if peer is None or not peer.enabled:
                continue
            if peer.MODULE_NAME in {"gpt_server", "llamafile"} and peer.status in {
                ModuleStatus.RUNNING,
                ModuleStatus.READY,
            }:
                return True
        return False

    async def _probe_llm_health(self) -> bool:
        base = self._llm_base_url()
        headers = self._llm_auth_headers()
        kwargs = self._local_httpx_kwargs(self._llm_protocol)
        async with httpx.AsyncClient(**kwargs) as client:
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
        if self._llm_linked:
            self.params["llm_status"] = "unreachable"
        return False

    async def _maybe_start_bot(self) -> None:
        token = self._bot_token().strip()
        if not token:
            if self._bot_task is not None:
                await self._stop_bot()
            return

        running = self._bot_task is not None and not self._bot_task.done()
        if running and token == self._active_bot_token:
            return

        await self._stop_bot()
        self._active_bot_token = token
        # Last-started module owns dispatch for this token. This prevents
        # duplicate replies when multiple discord_terminal instances share one bot.
        DiscordTerminalModule._TOKEN_OWNERS[token] = self.module_id
        self._bot_task = asyncio.create_task(self._run_discord_bot(token))
        log.info(
            "discord_terminal: bot task started module_id=%s local_ip_wired=%s",
            self.module_id,
            self._has_local_ip_wire(),
        )

    async def _stop_bot(self) -> None:
        client = getattr(self, "_discord_client", None)
        task = getattr(self, "_bot_task", None)
        active_token = str(getattr(self, "_active_bot_token", "") or "").strip()

        # Clear references first so no in-flight message handler can use them.
        self._discord_client = None
        self._bot_task = None
        self._active_bot_token = ""
        self.params["bot_connected"] = False
        self.params["bot_user"] = ""

        if client is not None:
            try:
                await self._set_presence(client=client, connected=False)
                if not client.is_closed():
                    await asyncio.wait_for(client.close(), timeout=5.0)
            except asyncio.TimeoutError:
                log.warning("discord_terminal: client.close() timed out")
            except Exception:
                pass

        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                pass

        owner = DiscordTerminalModule._TOKEN_OWNERS.get(active_token)
        if active_token and owner == self.module_id:
            DiscordTerminalModule._TOKEN_OWNERS.pop(active_token, None)

    async def _run_discord_bot(self, token: str) -> None:
        if discord is None:
            return
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        client = discord.Client(intents=intents)
        self._discord_client = client

        # Capture the client reference so handlers silently exit if the
        # module swapped to a new client (prevents orphan double-replies).
        expected_client = client

        @client.event
        async def on_ready() -> None:
            if self._discord_client is not expected_client:
                return
            user = str(client.user) if client.user is not None else ""
            self.params["bot_connected"] = True
            self.params["bot_user"] = user
            await self._set_presence(client=client, connected=True)
            log.info("discord_terminal: connected as %s", user)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if self._discord_client is not expected_client:
                return
            await self._handle_discord_message(message)

        try:
            await client.start(token)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.params["bot_connected"] = False
            self.params["bot_user"] = ""
            await self.report_state(
                severity="ERROR",
                message=f"discord_terminal bot failed: {exc}",
            )
            log.error("discord_terminal: bot start failed: %s", exc)
        finally:
            self.params["bot_connected"] = False
            self.params["bot_user"] = ""
            await self._set_presence(client=client, connected=False)

    async def _set_presence(self, *, client: Any, connected: bool) -> None:
        if not hasattr(client, "change_presence"):
            return
        try:
            if connected and hasattr(discord, "Game"):
                await client.change_presence(
                    activity=discord.Game(name="Miniloader online"),
                )
            else:
                await client.change_presence(activity=None)
        except Exception:
            log.debug("discord_terminal: unable to set Discord presence", exc_info=True)

    def _companion_url(self) -> str:
        return str(self.params.get("companion_url", "") or "").strip().rstrip("/")

    def _companion_secret(self) -> str:
        if self._vault is not None:
            try:
                secret = str(self._vault.get_secret("discord_terminal.companion_secret") or "").strip()
                if secret:
                    return secret
            except Exception:
                pass
        return str(self.params.get("companion_secret", "") or "").strip()

    def _companion_enabled(self) -> bool:
        return bool(self._companion_url()) and bool(self._bot_token().strip())

    async def _attach_to_companion(self) -> bool:
        if not self._companion_enabled():
            self._companion_attached = False
            return False
        payload = {
            "module_id": self.module_id,
            "token": self._bot_token().strip(),
            "secret": self._companion_secret(),
            "lease_seconds": int(self.params.get("companion_lease_seconds", 35) or 35),
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._companion_url()}/attach",
                    json=payload,
                    timeout=2.5,
                )
            if resp.is_success:
                self._companion_attached = bool((resp.json() or {}).get("attached", True))
                return self._companion_attached
            log.warning("discord_terminal: companion attach failed status=%s", resp.status_code)
        except Exception as exc:
            log.warning("discord_terminal: companion attach failed: %s", exc)
        self._companion_attached = False
        return False

    async def _detach_from_companion(self) -> None:
        if not self._companion_url():
            self._companion_attached = False
            return
        payload = {
            "module_id": self.module_id,
            "secret": self._companion_secret(),
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self._companion_url()}/detach",
                    json=payload,
                    timeout=2.5,
                )
        except Exception:
            log.debug("discord_terminal: companion detach failed", exc_info=True)
        self._companion_attached = False

    async def _ensure_companion_heartbeat(self) -> None:
        if not self._companion_enabled():
            await self._stop_companion_heartbeat()
            return
        if self._companion_heartbeat_task is not None and not self._companion_heartbeat_task.done():
            return
        self._companion_heartbeat_task = asyncio.create_task(self._companion_heartbeat_loop())

    async def _stop_companion_heartbeat(self) -> None:
        task = getattr(self, "_companion_heartbeat_task", None)
        self._companion_heartbeat_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _companion_heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(
                max(3, int(self.params.get("companion_heartbeat_interval_sec", 10) or 10))
            )
            if not self._companion_enabled():
                self._companion_attached = False
                continue
            payload = {
                "module_id": self.module_id,
                "secret": self._companion_secret(),
                "lease_seconds": int(self.params.get("companion_lease_seconds", 35) or 35),
            }
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self._companion_url()}/heartbeat",
                        json=payload,
                        timeout=2.5,
                    )
                if resp.is_success:
                    self._companion_attached = bool((resp.json() or {}).get("attached", True))
                    continue
            except Exception:
                pass
            self._companion_attached = False
            await self._attach_to_companion()

    def _bot_was_mentioned(self, message: Any) -> bool:
        """Return True if the bot user was @-mentioned in the message."""
        if self._discord_client is None or self._discord_client.user is None:
            return False
        return self._discord_client.user in (message.mentions or [])

    def _strip_bot_mention(self, content: str) -> str:
        """Remove the bot's @mention from the message text."""
        if self._discord_client is None or self._discord_client.user is None:
            return content
        bot_id = self._discord_client.user.id
        # Discord encodes mentions as <@BOT_ID> or <@!BOT_ID>
        cleaned = re.sub(rf"<@!?{bot_id}>", "", content).strip()
        return cleaned

    @staticmethod
    def _attachment_placeholder(attachment: Any) -> str:
        filename = str(getattr(attachment, "filename", "") or "attachment")
        lower_name = filename.lower()
        content_type = str(getattr(attachment, "content_type", "") or "").lower()
        if lower_name.endswith(".gif") or "gif" in content_type:
            kind = "gif"
        elif content_type.startswith("image/"):
            kind = "image"
        elif content_type.startswith("video/"):
            kind = "video"
        else:
            kind = "file"
        return f"[{kind} attachment: {filename}]"

    def _message_text_with_attachments(self, message: Any, text: str) -> str:
        base = str(text or "").strip()
        attachments = list(getattr(message, "attachments", []) or [])
        if not attachments:
            return base
        placeholders = " ".join(self._attachment_placeholder(a) for a in attachments).strip()
        if not placeholders:
            return base
        if base:
            return f"{base}\n{placeholders}"
        return placeholders

    async def _handle_discord_message(self, message: Any) -> None:
        if self._discord_client is None:
            return
        if not self._is_active_token_owner():
            return
        if message.author == self._discord_client.user:
            return
        if getattr(message.author, "bot", False):
            return

        channel_id = int(message.channel.id)
        raw_content = str(message.content or "")
        content = raw_content.strip()

        # !debug and !reset work without @mention.
        if content.lower() == "!debug":
            raw_param = str(self.params.get("channel_ids", "") or "")
            allowed = self._allowed_channel_ids()
            require_mention = bool(self.params.get("require_mention", True))
            await message.channel.send(
                f"**discord_terminal debug**\n"
                f"this channel: `{channel_id}`\n"
                f"raw param: `{raw_param!r}`\n"
                f"parsed allowlist: `{sorted(allowed)}`\n"
                f"filter active: `{bool(allowed)}`\n"
                f"this channel allowed: `{channel_id in allowed if allowed else 'yes (all)'}`\n"
                f"require @mention: `{require_mention}`"
            )
            return

        if content.lower() in {"!reset", "/reset"}:
            await message.channel.send(
                "History source is Discord channel messages; there is no local buffer to reset."
            )
            return

        if not self._is_allowed_channel(channel_id):
            log.debug(
                "discord_terminal: blocked message in channel %s "
                "(allowlist=%s, raw_param=%r)",
                channel_id,
                sorted(self._allowed_channel_ids()),
                str(self.params.get("channel_ids", "")),
            )
            return

        # Only respond when the bot is @mentioned (unless disabled).
        if bool(self.params.get("require_mention", True)):
            if not self._bot_was_mentioned(message):
                return
            content = self._strip_bot_mention(raw_content).strip()

        if not content and not list(getattr(message, "attachments", []) or []):
            return

        lock = self._channel_locks.setdefault(channel_id, asyncio.Lock())
        async with lock:
            try:
                if not self._has_local_ip_wire() and not self._agent_wire_active():
                    await message.channel.send("LLM link unplugged (LOCAL_IP wire missing).")
                    return
                await self.refresh_wiring_state()
                reachable = True
                if not self._agent_wire_active():
                    reachable = self._llm_reachable or await self._probe_llm_health()
                # Some providers can be reachable even when static wire-name checks
                # fail; only emit unplugged when both link and health are absent.
                if not self._llm_linked and not reachable:
                    await message.channel.send("LLM link unplugged (LOCAL_IP wire missing).")
                    return
                if not reachable:
                    await message.channel.send(
                        f"LLM backend unreachable at {self._llm_host}:{self._llm_port}."
                    )
                    return

                response_text = await self._run_chat_turn(
                    message=message,
                    user_text=content,
                )
                await self._send_discord_chunks(message.channel, response_text)
            except Exception as exc:
                log.exception("discord_terminal: message processing failed: %s", exc)
                await message.channel.send(f"Error: {exc}")

    async def _run_chat_turn(self, *, message: Any, user_text: str) -> str:
        messages: list[dict[str, Any]] = []
        system_prompt = str(self.params.get("system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(await self._build_channel_context(message=message, user_text=user_text))

        if self._agent_wire_active():
            return await self._run_chat_turn_via_agent(messages=messages)

        if bool(self.params.get("rag_enabled", True)):
            await self._maybe_inject_rag(messages)

        messages = self._coalesce_consecutive_roles(messages)

        request_id = f"discord_{int(time.time() * 1000):x}_{uuid.uuid4().hex[:8]}"
        text = await self._chat_completion_sse(messages=messages, request_id=request_id)

        return text

    async def _run_chat_turn_via_agent(self, *, messages: list[dict[str, Any]]) -> str:
        request_id = f"discord_{int(time.time() * 1000):x}_{uuid.uuid4().hex[:8]}"
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_agent_streams[request_id] = q
        text_parts: list[str] = []
        try:
            await self.inputs["AGENT_IN"].emit(
                Payload(
                    signal_type=SignalType.AGENT_PAYLOAD,
                    source_module=self.module_id,
                    data={
                        "action": "chat_request",
                        "request_id": request_id,
                        "messages": messages,
                        "temperature": 0.7,
                    },
                )
            )
            while True:
                evt = await q.get()
                action = str(evt.get("action", "")).strip().lower()
                if action == "token":
                    tok = str(evt.get("content", "") or "")
                    if tok:
                        text_parts.append(tok)
                elif action == "error":
                    raise RuntimeError(str(evt.get("error", "agent error")))
                elif action == "stream_end":
                    break
        finally:
            self._pending_agent_streams.pop(request_id, None)
        text = "".join(text_parts).strip()
        return text or "(no response)"

    async def _build_channel_context(self, *, message: Any, user_text: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        use_channel_history = bool(self.params.get("use_channel_history", True))
        if use_channel_history:
            out.extend(await self._fetch_channel_history_messages(message=message))

        display_name = self._author_name(message.author)
        base_text = f"{display_name}: {user_text}".strip() if user_text else ""
        attachments = list(getattr(message, "attachments", []) or [])
        image_parts: list[dict[str, Any]] = []
        non_image_notes: list[str] = []
        for attachment in attachments:
            content_type = str(getattr(attachment, "content_type", "") or "").lower()
            filename = str(getattr(attachment, "filename", "") or "")
            if content_type.startswith("image/"):
                url = str(getattr(attachment, "url", "") or "").strip()
                if url:
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )
                else:
                    non_image_notes.append(self._attachment_placeholder(attachment))
                continue
            # Unknown/missing MIME: treat common static image extensions as vision-capable.
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")):
                url = str(getattr(attachment, "url", "") or "").strip()
                if url:
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )
                else:
                    non_image_notes.append(self._attachment_placeholder(attachment))
                continue
            non_image_notes.append(self._attachment_placeholder(attachment))

        if image_parts:
            text_chunks: list[str] = []
            if base_text:
                text_chunks.append(base_text)
            if non_image_notes:
                text_chunks.append(" ".join(non_image_notes))
            text_part = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            multimodal: list[dict[str, Any]] = []
            if text_part:
                multimodal.append({"type": "text", "text": text_part})
            multimodal.extend(image_parts)
            out.append({"role": "user", "content": multimodal})
        else:
            text_chunks: list[str] = []
            if base_text:
                text_chunks.append(base_text)
            if non_image_notes:
                text_chunks.append(" ".join(non_image_notes))
            fallback_text = "\n".join(text_chunks).strip()
            if fallback_text:
                out.append({"role": "user", "content": fallback_text})
        return self._trim_messages_to_budget(out)

    async def _fetch_channel_history_messages(self, *, message: Any) -> list[dict[str, Any]]:
        limit = int(self.params.get("history_window_messages", _DEFAULT_HISTORY_WINDOW_MESSAGES))
        limit = max(1, min(100, limit))
        rows: list[dict[str, Any]] = []
        try:
            async for prev in message.channel.history(limit=limit + 1, oldest_first=False):
                if int(getattr(prev, "id", 0) or 0) == int(getattr(message, "id", 0) or 0):
                    continue
                text = self._message_text_with_attachments(
                    prev,
                    str(getattr(prev, "content", "") or "").strip(),
                )
                if not text:
                    continue
                if self._is_bot_author(getattr(prev, "author", None)):
                    rows.append({"role": "assistant", "content": text})
                else:
                    rows.append(
                        {
                            "role": "user",
                            "content": f"{self._author_name(prev.author)}: {text}",
                        }
                    )
        except Exception:
            log.debug("discord_terminal: failed to fetch channel history", exc_info=True)
            return []
        rows.reverse()
        return rows

    @staticmethod
    def _coalesce_consecutive_roles(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive messages that share the same role.

        Discord channels can produce runs of user messages (multiple humans
        chatting in a row) or back-to-back assistant messages.  Strict-
        alternation chat templates (Mistral, Llama-Instruct, etc.) reject such
        sequences, so we merge them here before the request leaves the module.
        """
        if not messages:
            return messages

        def _join_content(a: Any, b: Any) -> Any:
            if isinstance(a, list) and isinstance(b, list):
                return a + b
            if isinstance(a, list):
                return a + [{"type": "text", "text": str(b or "")}]
            if isinstance(b, list):
                return [{"type": "text", "text": str(a or "")}] + b
            return f"{a}\n\n{b}" if a else b

        result: list[dict[str, Any]] = [dict(messages[0])]
        for msg in messages[1:]:
            if msg.get("role") == result[-1].get("role"):
                result[-1] = dict(result[-1])
                result[-1]["content"] = _join_content(
                    result[-1].get("content", ""),
                    msg.get("content", ""),
                )
            else:
                result.append(dict(msg))
        return result

    def _trim_messages_to_budget(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return rows
        token_budget = self._history_token_budget()
        if token_budget <= 0:
            return rows

        kept: list[dict[str, Any]] = []
        used = 0
        for row in reversed(rows):
            content = row.get("content", "")
            estimated = 0
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        estimated += max(1, len(str(part or "")) // 4)
                        continue
                    ptype = str(part.get("type", "") or "")
                    if ptype == "text":
                        estimated += max(1, len(str(part.get("text", "") or "")) // 4)
                    elif ptype == "image_url":
                        estimated += 600
                    else:
                        estimated += max(1, len(str(part)) // 4)
            else:
                estimated = max(1, len(str(content or "")) // 4)
            estimated += 6
            if kept and (used + estimated) > token_budget:
                break
            kept.append(row)
            used += estimated
        kept.reverse()
        return kept

    def _history_token_budget(self) -> int:
        ctx = max(1024, int(self._llm_ctx_length or 4096))
        ratio = float(self.params.get("history_token_ratio", _DEFAULT_HISTORY_TOKEN_RATIO))
        ratio = max(0.05, min(0.9, ratio))
        return max(128, int(ctx * ratio))

    def _is_bot_author(self, author: Any) -> bool:
        if author is None:
            return False
        if self._discord_client is not None and self._discord_client.user is not None:
            try:
                return int(author.id) == int(self._discord_client.user.id)
            except Exception:
                pass
        return bool(getattr(author, "bot", False))

    def _is_active_token_owner(self) -> bool:
        token = str(getattr(self, "_active_bot_token", "") or "").strip()
        if not token:
            return True
        return DiscordTerminalModule._TOKEN_OWNERS.get(token) == self.module_id

    @staticmethod
    def _author_name(author: Any) -> str:
        if author is None:
            return "user"
        name = str(getattr(author, "display_name", "") or getattr(author, "name", "") or "").strip()
        return name or "user"

    async def _chat_completion_sse(
        self,
        *,
        messages: list[dict[str, Any]],
        request_id: str,
    ) -> str:
        url = f"{self._llm_base_url()}/v1/chat/completions"
        base_payload: dict[str, Any] = {
            "request_id": request_id,
            "model": self._llm_model or "local",
            "temperature": 0.7,
        }
        headers = self._llm_auth_headers()
        kwargs = self._local_httpx_kwargs(self._llm_protocol)
        fragments: list[str] = []
        line_buffer = ""
        tools = self._llm_tools_payload()
        working_messages = list(messages)

        async with httpx.AsyncClient(**kwargs) as client:
            if tools:
                for round_idx in range(_MAX_TOOL_ROUNDS):
                    probe_payload = dict(base_payload)
                    probe_payload["messages"] = working_messages
                    probe_payload["stream"] = False
                    probe_payload["tools"] = tools
                    probe_payload["tool_choice"] = "auto"
                    probe = await client.post(
                        url,
                        json=probe_payload,
                        headers=headers,
                        timeout=None,
                    )
                    if probe.status_code != 200:
                        body = (await probe.aread()).decode(errors="replace")
                        raise RuntimeError(f"LLM {probe.status_code}: {body}")
                    probe_data = probe.json()
                    choice = (probe_data.get("choices") or [{}])[0]
                    assistant_msg = choice.get("message") or {}
                    tool_calls = list(assistant_msg.get("tool_calls") or [])
                    if not tool_calls:
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
                        call_args = self._decode_tool_arguments((call.get("function") or {}).get("arguments"))
                        tool_result = await self._execute_registered_tool(
                            tool_name=fn,
                            arguments=call_args,
                            tool_call_id=call_id,
                        )
                        tool_content = json.dumps(tool_result, ensure_ascii=False)
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": fn,
                                "content": tool_content,
                            }
                        )

            async with client.stream(
                "POST",
                url,
                json={
                    **base_payload,
                    "messages": working_messages,
                    "stream": True,
                    **({"tools": tools, "tool_choice": "auto"} if tools else {}),
                },
                headers=headers,
                timeout=None,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise RuntimeError(f"LLM {resp.status_code}: {body}")
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue
                    line_buffer += chunk
                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", 1)
                        part = self._parse_sse_data_line(line)
                        if part is not None:
                            fragments.append(part)
                if line_buffer:
                    part = self._parse_sse_data_line(line_buffer)
                    if part is not None:
                        fragments.append(part)

        text = "".join(fragments).strip()
        return text or "(no response)"

    def _llm_tools_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for tool_name in sorted(self._tool_registry.keys()):
            entry = self._tool_registry.get(tool_name)
            if not isinstance(entry, dict):
                continue
            raw_tool = entry.get("tool")
            if isinstance(raw_tool, dict):
                payload.append(raw_tool)
        return payload

    @staticmethod
    def _decode_tool_arguments(raw: Any) -> dict[str, Any]:
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

    async def _execute_registered_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        if not tool_name:
            return {
                "success": False,
                "error": {"code": "TOOL_NAME_MISSING", "message": "Tool name is missing"},
            }
        if tool_name not in self._tool_registry:
            return {
                "success": False,
                "error": {"code": "TOOL_NOT_FOUND", "message": f"Tool '{tool_name}' is not registered"},
            }

        event = asyncio.Event()
        self._pending_tool_calls[tool_call_id] = {
            "event": event,
            "requested_at": time.time(),
            "tool_name": tool_name,
        }

        execute_payload = Payload(
            signal_type=SignalType.TOOL_EXECUTION_PAYLOAD,
            source_module=self.module_id,
            data={
                "action": "execute",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )
        entry = self._tool_registry.get(tool_name, {})
        target_module_id = str(entry.get("source_module", "")).strip() if isinstance(entry, dict) else ""
        emitted = False
        if target_module_id:
            emitted = await self.inputs["MCP_IN"].emit_to(
                execute_payload,
                target_module_id=target_module_id,
            )
            if not emitted:
                log.warning(
                    "discord_terminal: tool provider for '%s' not wired (target=%s); using broadcast fallback",
                    tool_name,
                    target_module_id,
                )
        if not emitted:
            await self.inputs["MCP_IN"].emit(execute_payload)

        try:
            await asyncio.wait_for(event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_tool_calls.pop(tool_call_id, None)
            return {
                "success": False,
                "error": {
                    "code": "TOOL_TIMEOUT",
                    "message": f"Tool call timed out for '{tool_name}'",
                },
            }

        pending = self._pending_tool_calls.pop(tool_call_id, None)
        if not isinstance(pending, dict):
            return {
                "success": False,
                "error": {"code": "TOOL_RESULT_MISSING", "message": "Tool result missing"},
            }
        result_payload = pending.get("result")
        if not isinstance(result_payload, dict):
            return {
                "success": False,
                "error": {"code": "TOOL_RESULT_INVALID", "message": "Tool result payload invalid"},
            }
        return {
            "success": bool(result_payload.get("success", False)),
            "result": result_payload.get("result"),
            "error": result_payload.get("error"),
            "tool_name": tool_name,
        }

    @staticmethod
    def _parse_sse_data_line(line: str) -> str | None:
        cleaned = line.strip()
        if not cleaned.startswith("data:"):
            return None
        data = cleaned[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            obj = json.loads(data)
        except Exception:
            return None
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = (choices[0] or {}).get("delta") or {}
        content = delta.get("content")
        return str(content) if content is not None else None

    async def _maybe_inject_rag(self, messages: list[dict[str, Any]]) -> None:
        user_msg = ""
        for row in reversed(messages):
            if row.get("role") == "user":
                content = row.get("content", "")
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(str(part.get("text", "") or ""))
                    user_msg = " ".join(t for t in text_parts if t).strip()
                else:
                    user_msg = str(content or "").strip()
                break
        if not user_msg:
            return

        rag = await self._execute_rag_query(
            query_text=user_msg,
            top_k=5,
            request_id=f"rag_{int(time.time() * 1000):x}_{uuid.uuid4().hex[:6]}",
        )
        chunks = list(rag.get("chunks") or [])
        if not chunks:
            return
        top_score = max((float(c.get("score", 0.0)) for c in chunks), default=0.0)
        if top_score < _RAG_AUTO_INJECT_THRESHOLD:
            return

        ctx = "\n\n".join(
            f"[source:{c.get('source') or 'unknown'}] {str(c.get('text') or '')}"
            for c in chunks[:4]
        ).strip()
        if not ctx:
            return

        rag_block = (
            "Use only the provided context when relevant. If the context is "
            "insufficient, explicitly say so.\n\n"
            f"Relevant context:\n\n{ctx}"
        )
        sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), -1)
        if sys_idx >= 0:
            # Append RAG context into the existing system message rather than
            # inserting a second system entry (breaks strict-alternation templates).
            messages[sys_idx] = dict(messages[sys_idx])
            messages[sys_idx]["content"] = (
                str(messages[sys_idx].get("content", "")).rstrip()
                + "\n\n"
                + rag_block
            )
        else:
            messages.insert(0, {"role": "system", "content": rag_block})

    async def _execute_rag_query(
        self,
        *,
        query_text: str,
        top_k: int,
        request_id: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        connected = self._rag_link_active()
        if not connected or not query_text:
            return {"chunks": [], "count": 0, "total_vectors_searched": 0}

        event = asyncio.Event()
        holder: dict[str, Any] = {}
        self._pending_rag[request_id] = (event, holder)
        await self.inputs["CONTEXT_IN"].emit(
            Payload(
                signal_type=SignalType.QUERY_PAYLOAD,
                source_module=self.module_id,
                data={
                    "query_text": query_text,
                    "top_k": int(top_k),
                    "request_id": request_id,
                    "filters": filters or {},
                },
            )
        )

        try:
            await asyncio.wait_for(event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._pending_rag.pop(request_id, None)
            log.warning("discord_terminal: RAG query timed out request_id=%s", request_id)
            return {"chunks": [], "count": 0, "total_vectors_searched": 0}

        result = self._pending_rag.pop(request_id, (None, {}))[1]
        return {
            "chunks": result.get("chunks", []),
            "count": result.get("count", 0),
            "total_vectors_searched": result.get("total_vectors_searched", 0),
            "latency_ms": float(result.get("latency_ms", 0.0)),
            "retrieval_method": str(result.get("retrieval_method", "")),
        }

    def _rag_link_active(self) -> bool:
        if self._hypervisor is None:
            return False
        for wire in self._hypervisor.active_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name == "CONTEXT_IN":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name == "CONTEXT_IN":
                peer_id = src.owner_module_id
            else:
                continue
            peer = self._hypervisor.active_modules.get(peer_id)
            if (
                peer is not None
                and peer.enabled
                and peer.MODULE_NAME == "rag_engine"
                and peer.status in {ModuleStatus.RUNNING, ModuleStatus.READY}
            ):
                return True
        return False

    async def _register_tools(self, payload: Payload) -> None:
        tools = payload.data.get("tools", [])
        if not isinstance(tools, list):
            return
        source = payload.source_module
        stale = [k for k, v in self._tool_registry.items() if v.get("source_module") == source]
        for key in stale:
            del self._tool_registry[key]
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "")).strip().lower()
            if not name:
                continue
            self._tool_registry[name] = {
                "tool": tool,
                "source_module": source,
                "provider": str(payload.data.get("provider", "")).strip().lower(),
            }
        self.params["tools_count"] = len(self._tool_registry)
        self.params["tools_discovered"] = bool(self._tool_registry)

    async def _handle_tool_result(self, payload: Payload) -> None:
        call_id = str(payload.data.get("tool_call_id", "")).strip()
        if not call_id:
            return
        pending = self._pending_tool_calls.get(call_id)
        if pending is None:
            return
        pending["result"] = payload.data
        event = pending.get("event")
        if isinstance(event, asyncio.Event):
            event.set()

    async def _send_discord_chunks(self, channel: Any, text: str) -> None:
        msg = str(text or "").strip() or "(empty response)"
        while msg:
            part = msg[:_DISCORD_MAX_MESSAGE]
            await channel.send(part)
            msg = msg[_DISCORD_MAX_MESSAGE:]

    def _is_allowed_channel(self, channel_id: int) -> bool:
        allowed = self._allowed_channel_ids()
        if not allowed:
            return True
        return channel_id in allowed

    def _allowed_channel_ids(self) -> set[int]:
        raw = str(self.params.get("channel_ids", "") or "").strip()
        if not raw:
            return set()
        return set(self._extract_channel_ids(raw))

    @staticmethod
    def _extract_channel_ids(raw: str) -> list[int]:
        out: list[int] = []
        for token in re.split(r"[\s,]+", str(raw or "").strip()):
            token = token.strip()
            if not token:
                continue

            # URL form: keep only the channel segment, not guild/server ID.
            m = _CHANNEL_URL_RE.search(token)
            if m:
                try:
                    out.append(int(m.group(1)))
                except ValueError:
                    pass
                continue

            # Mention/plain-id form.
            if token.startswith("<#") and token.endswith(">"):
                token = token[2:-1]
            for match in _CHANNEL_ID_RE.findall(token):
                try:
                    out.append(int(match))
                except ValueError:
                    continue

        # Preserve order while de-duping.
        return list(dict.fromkeys(out))

    def _bot_token(self) -> str:
        # Vault first (encrypted, survives restarts); fall back to in-memory param.
        if self._vault is not None:
            try:
                vt = str(self._vault.get_secret("discord_terminal.discord_bot_token") or "").strip()
                if vt:
                    return vt
            except Exception:
                pass
        token = str(self.params.get("discord_bot_token", "") or "").strip()
        if token and token != "__vault__":
            return token
        return ""

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

    def _llm_base_url(self) -> str:
        return f"{self._llm_protocol}://{self._llm_host}:{self._llm_port}"


def register(hypervisor: Any) -> None:
    module = DiscordTerminalModule()
    hypervisor.register_module(module)
