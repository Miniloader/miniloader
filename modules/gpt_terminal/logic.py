"""
gpt_terminal/logic.py — Application Server
=============================================
React + Vite frontend served by a pure-Python backend.  Acts as the
application layer: manages conversation threads, prompt assembly, tool
orchestration, and feature detection based on which modules are wired
and running.

Port map
--------
LOCAL_IP_IN     ◄─► gpt_server.LOCAL_IP_OUT     Bidirectional inference: sends
                    (or llamafile.LOCAL_IP_OUT)  chat requests, receives tokens
                                                 and server config.
DB_IN_OUT       ◄─► database.DB_IN_OUT          Persistent conversation
                                                 history and app state.
CONTEXT_IN      ◄─► rag_engine.CONTEXT_OUT       Semantic search: sends
                                                 queries, receives context.
MCP_IN          ◄─► mcp_bus.MCP_DOWNSTREAM       Tool execution: dispatches
                                                   calls, receives results.
                    (or file_access.TOOLS_OUT)
WEB_OUT         ◄─► ngrok_tunnel.WEB_IN          Bidirectional tunnel mgmt:
                                                 sends local address, receives
                                                 public URL and tunnel status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from core import async_sqlcipher
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from core.base_module import BaseModule, ModuleStatus
from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    SignalType,
)
from modules.database.logic import _DDL_SQLITE

log = logging.getLogger(__name__)

# Optional fields forwarded from the browser to the OpenAI-compatible LLM backend
# (llama.cpp server, cloud APIs, etc.). Unknown keys are ignored for safety.
_LLM_OPTIONAL_KEYS = frozenset({
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
    # MoE / expert-style backends may honor this; ignored by servers that do not.
    "moe_enabled",
})

_RAG_AUTO_INJECT_THRESHOLD = 0.55
_MAX_TOOL_ROUNDS = 3
_RAG_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the user's indexed documents for information relevant to a query. "
            "Use this when the conversation requires facts from the user's files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document chunks",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                },
            },
            "required": ["query"],
        },
    },
}


def _text_from_message_content(content: Any) -> str:
    """Plain text for RAG query when user content is string or multimodal list."""
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "\n".join(texts).strip()
    if content is None:
        return ""
    return str(content).strip()


_APP_DIR    = Path(__file__).parent / "app"
_DIST_INDEX = _APP_DIR / "dist" / "index.html"
_DIST_DIR   = _APP_DIR / "dist"

_HEALTH_TIMEOUT  = 15.0
_HEALTH_INTERVAL = 0.5
_TOKEN_BATCH_SIZE = 8
_PORT_RELEASE_DELAY = 1.5
_RESTART_HEALTH_GRACE_S = 4.0
_TOOL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9]*_[a-z0-9][a-z0-9_]*$")

_REQUIRED_TABLES = (
    "system_state",
    "threads",
    "messages",
    "templates",
    "settings",
    "consent_flags",
)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def _wait_for_port_free(port: int, timeout_s: float = 2.0) -> bool:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if not _port_in_use(port):
            return True
        await asyncio.sleep(0.2)
    return not _port_in_use(port)


class GptTerminalModule(BaseModule):
    MODULE_NAME = "gpt_terminal"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Browser-based chat UI with conversation state management"
    UI_COL_SPAN = 2

    _api_server: uvicorn.Server    | None = None
    _api_task:   asyncio.Task      | None = None
    _tool_timeout_task: asyncio.Task | None = None

    # Runtime LLM state (replaces Node's config object)
    _llm_host: str = "127.0.0.1"
    _llm_port: int = 5000
    _llm_protocol: str = "http"
    _llm_model: str | None = None
    _llm_ctx_length: int = 4096
    _llm_linked: bool = False
    _llm_reachable: bool = False
    _agent_linked: bool = False
    _rag_debug_events: list[dict[str, Any]] = []

    def get_default_params(self) -> dict[str, Any]:
        return {
            "system_prompt": "You are a helpful assistant.",
            "save_history": True,
            "web_port": 3000,
            "allow_remote_session_token": True,
            "tunnel_status": "inactive",
            "public_url": "",
            "tool_call_timeout_s": 30,
            "tools_count": 0,
            "tools_discovered": False,
        }

    # ── Ports ───────────────────────────────────────────────

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
            "VOICE_LINK",
            accepted_signals={SignalType.SERVER_CONFIG_PAYLOAD},
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Voice bridge config channel from livekit_voice. Receives LiveKit "
                "runtime settings used to mint browser join tokens."
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
                "Bidirectional API channel (API_IN label). Receives the "
                "inference server address (SERVER_CONFIG_PAYLOAD) "
                "and streamed LLM tokens; sends assembled chat "
                "requests back up the wire."
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
            description=(
                "Bidirectional storage link (STORAGE_LINK label). Sends queries "
                "and transactions to the database; receives "
                "results back through the same wire."
            ),
        )
        self.add_input(
            "CONTEXT_IN",
            accepted_signals={
                SignalType.QUERY_PAYLOAD,
                SignalType.CONTEXT_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional document channel (DOCUMENT_IN label). Sends semantic "
                "search queries to the RAG engine; receives the "
                "top-k relevant text chunks back through the "
                "same wire for prompt injection."
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
            description=(
                "Bidirectional tools channel (TOOLS_IN label). Receives tool schemas from "
                "connected MCP hosts and file_access; dispatches "
                "tool execution requests and receives results back "
                "through the same wire."
            ),
        )
        self.add_output(
            "WEB_OUT",
            accepted_signals={
                SignalType.ROUTING_CONFIG_PAYLOAD,
                SignalType.TUNNEL_STATUS_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional web channel. Sends this module's "
                "local address to the tunnel module; receives the "
                "active public ngrok URL and tunnel status back "
                "through the same wire."
            ),
        )

    # ── RAG + DB discovery ─────────────────────────────────

    _pending_rag: dict[str, tuple[asyncio.Event, dict[str, Any]]]

    def _rag_link_active(self) -> tuple[bool, int]:
        if self._hypervisor is None:
            return False, 0
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
            if peer is None or not peer.enabled:
                continue
            if peer.MODULE_NAME != "rag_engine":
                continue
            if peer.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            count = int(peer.params.get("vector_count", 0))
            return True, count
        return False, 0

    def _find_rag_peer(self) -> Any | None:
        if self._hypervisor is None:
            return None
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
                and peer.status in (ModuleStatus.RUNNING, ModuleStatus.READY)
            ):
                return peer
        return None

    def _voice_link_active(self) -> bool:
        if self._hypervisor is None:
            return False
        for wire in self._hypervisor.active_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name == "VOICE_LINK":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name == "VOICE_LINK":
                peer_id = src.owner_module_id
            else:
                continue
            peer = self._hypervisor.active_modules.get(peer_id)
            if (
                peer is not None
                and peer.enabled
                and peer.MODULE_NAME == "livekit_voice"
                and peer.status in (ModuleStatus.RUNNING, ModuleStatus.READY)
            ):
                return True
        return False

    def _voice_available(self) -> bool:
        key = str(self._voice_config.get("livekit_api_key", "")).strip()
        secret = str(self._voice_config.get("livekit_api_secret", "")).strip()
        ws_url = str(self._voice_config.get("livekit_ws_url", "")).strip()
        return bool(self._voice_link_active() and key and secret and ws_url)

    def _active_file_acl(self) -> list[str]:
        if self._hypervisor is None:
            return []
        allowed: set[str] = set()
        for module in self._hypervisor.active_modules.values():
            if module.MODULE_NAME != "file_access":
                continue
            active_files = module.params.get("active_files", [])
            if not isinstance(active_files, list):
                continue
            for item in active_files:
                name = Path(str(item)).name.strip().lower()
                if name:
                    allowed.add(name)
        return sorted(allowed)

    def _model_tier(self) -> str:
        if int(self._llm_ctx_length or 0) <= 2048:
            return "small"
        return "large"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 3.5))

    def _select_snippets(
        self,
        chunks: list[dict[str, Any]],
        *,
        token_budget: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used = 0
        for chunk in sorted(chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True):
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            est = self._estimate_tokens(text)
            if used + est <= token_budget:
                selected.append(chunk)
                used += est
                continue
            if used >= token_budget:
                break
            remain_chars = int((token_budget - used) * 3.5)
            if remain_chars < 48:
                break
            trimmed = text[:remain_chars]
            split_idx = max(trimmed.rfind(". "), trimmed.rfind(".\n"), trimmed.rfind("\n"))
            if split_idx > remain_chars // 2:
                trimmed = trimmed[: split_idx + 1]
            trimmed = trimmed.strip()
            if not trimmed:
                break
            copied = dict(chunk)
            copied["text"] = trimmed
            selected.append(copied)
            break
        return selected

    async def _rewrite_query_large(self, messages: list[dict[str, Any]]) -> str:
        llm_url = f"{self._llm_base_url()}/v1/chat/completions"
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the user's latest request into a standalone retrieval query. "
                    "Return only the rewritten query text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(messages[-6:], ensure_ascii=False),
            },
        ]
        body = {
            "model": self._llm_model or "local",
            "messages": prompt_messages,
            "stream": False,
            "max_tokens": 80,
            "temperature": 0.0,
        }
        client_kwargs = self._local_httpx_kwargs(self._llm_protocol)
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(
                llm_url,
                json=body,
                headers=self._llm_auth_headers(),
                timeout=15.0,
            )
            if not resp.is_success:
                return ""
            data = resp.json()
            return str(
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            ).strip()

    async def _build_rag_query(self, messages: list[dict[str, Any]]) -> str:
        user_turns = [
            _text_from_message_content(m.get("content"))
            for m in messages
            if m.get("role") == "user"
        ]
        user_turns = [u for u in user_turns if u]
        if not user_turns:
            return ""
        tier = self._model_tier()
        if tier == "large":
            rewritten = await self._rewrite_query_large(messages)
            if rewritten:
                return rewritten
        return " ".join(user_turns[-3:]).strip()

    def _get_db_path(self) -> Path | None:
        if self._hypervisor is None:
            return None
        for wire in self._hypervisor.active_wires:
            src_mid = wire.source_port.owner_module_id
            tgt_mid = wire.target_port.owner_module_id
            src_name = wire.source_port.name
            tgt_name = wire.target_port.name

            if src_mid == self.module_id and src_name == "DB_IN_OUT":
                peer = self._hypervisor.active_modules.get(tgt_mid)
            elif tgt_mid == self.module_id and tgt_name == "DB_IN_OUT":
                peer = self._hypervisor.active_modules.get(src_mid)
            else:
                continue

            if peer is not None and peer.enabled and peer.MODULE_NAME == "database":
                resolve_fn = getattr(peer, "_resolve_db_path", None)
                if callable(resolve_fn):
                    return resolve_fn()
                raw = str(peer.params.get("db_filepath", "miniloader_data.db")).strip()
                p = Path(raw)
                if not p.is_absolute():
                    p = Path.cwd() / p
                return p
        return None

    async def _open_db(self, db_path: Path) -> Any:
        """Open a DB connection using SQLCipher only."""
        if self._vault is None:
            raise RuntimeError("gpt_terminal: vault is required for DB access")
        if not async_sqlcipher.is_available():
            raise RuntimeError(
                "gpt_terminal: sqlcipher3 is required for DB access. "
                "Run: pip install sqlcipher3-binary"
            )
        key_hex = self._vault.derive_db_key().hex()
        return await async_sqlcipher.connect(str(db_path), key_hex=key_hex)

    def _llm_local_ip_link_active(self) -> bool:
        if self._hypervisor is None:
            return False
        allowed_server_types = {"gpt_server", "cloud_brain"}
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
            if peer.MODULE_NAME not in allowed_server_types:
                continue
            if peer.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            return True
        return False

    # ── WebSocket state broadcast ───────────────────────────

    def _config_payload(self) -> dict:
        rag_connected, rag_vector_count = self._rag_link_active()
        return {
            "type":       "config",
            "llm_linked": self._llm_linked,
            "llm_ready":  self._llm_linked and self._llm_reachable,
            "protocol":   self._llm_protocol,
            "host":       self._llm_host,
            "port":       self._llm_port,
            "llm_model":  self._llm_model or "local",
            "ctx_length": self._llm_ctx_length,
            "model_tier": self._model_tier(),
            "rag_connected": rag_connected,
            "rag_vector_count": rag_vector_count,
            "agent_connected": self._agent_linked,
            "voice_available": self._voice_available(),
        }

    async def _broadcast_ws(self, data: dict) -> None:
        """Send a JSON message to all connected browser WebSocket clients."""
        if not self._ws_clients:
            return
        disconnected: set = set()
        msg = json.dumps(data)
        for ws in set(self._ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.add(ws)
        self._ws_clients -= disconnected

    async def _broadcast_config_ws(self) -> None:
        await self._broadcast_ws(self._config_payload())

    def _llm_auth_headers(self) -> dict[str, str]:
        """Return Authorization header for internal LLM server calls."""
        if self._vault is not None:
            token = self._vault.get_or_create_secret(
                self._vault.DEFAULT_GPT_SERVER_SECRET_KEY
            )
            return {"Authorization": f"Bearer {token}"}
        return {}

    @staticmethod
    def _is_loopback_request(request: Request) -> bool:
        client = request.client
        if client is None:
            return False
        host = (client.host or "").strip().lower()
        return host in {"127.0.0.1", "::1", "localhost"}

    @staticmethod
    def _is_forwarded_request(request: Request) -> bool:
        """Detect reverse-proxy forwarded browser requests (e.g. ngrok)."""
        forwarded_host = str(request.headers.get("x-forwarded-host", "")).strip().lower()
        forwarded_proto = str(request.headers.get("x-forwarded-proto", "")).strip().lower()
        return bool(forwarded_host and forwarded_proto in {"http", "https"})

    def _allow_remote_session_token(self, request: Request) -> bool:
        if not bool(self.params.get("allow_remote_session_token", True)):
            return False
        if str(self.params.get("tunnel_status", "inactive")).strip().lower() != "active":
            return False
        if not self._is_forwarded_request(request):
            return False
        return True

    def _local_httpx_kwargs(self, protocol: str) -> dict[str, Any]:
        """Use relaxed TLS verification for local self-signed certs."""
        if protocol == "https":
            return {"verify": False}
        return {}

    def _llm_base_url(self) -> str:
        return f"{self._llm_protocol}://{self._llm_host}:{self._llm_port}"

    async def _probe_llm_health(self) -> bool:
        base = self._llm_base_url()
        headers = self._llm_auth_headers()
        client_kwargs = self._local_httpx_kwargs(self._llm_protocol)
        async with httpx.AsyncClient(**client_kwargs) as client:
            for endpoint in [f"{base}/v1/health", f"{base}/health"]:
                try:
                    r = await client.get(endpoint, timeout=1.5, headers=headers)
                    if r.is_success:
                        return True
                except Exception:
                    pass
        return False

    async def _probe_and_broadcast_llm_state(self) -> None:
        self._llm_reachable = await self._probe_llm_health()
        await self._broadcast_config_ws()

    # ── RAG query helper (used by /rag/query route and /api/chat) ──

    async def _execute_rag_query(
        self,
        query_text: str,
        top_k: int,
        request_id: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> dict:
        connected, _ = self._rag_link_active()
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
                    "top_k": top_k,
                    "request_id": request_id,
                    "filters": filters or {},
                    "acl_allowed_sources": self._active_file_acl(),
                },
            )
        )
        try:
            await asyncio.wait_for(event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._pending_rag.pop(request_id, None)
            log.warning("gpt_terminal: RAG query timed out request_id=%s", request_id)
            return {"chunks": [], "count": 0, "total_vectors_searched": 0}

        result = self._pending_rag.pop(request_id, (None, {}))[1]
        return {
            "chunks": result.get("chunks", []),
            "count":  result.get("count", 0),
            "total_vectors_searched": result.get("total_vectors_searched", 0),
            "latency_ms": float(result.get("latency_ms", 0.0)),
            "retrieval_method": str(result.get("retrieval_method", "")),
        }

    # ── Unified Python web app (replaces server.js + internal API) ──

    def _build_app(self) -> FastAPI:
        """Build the unified Starlette/FastAPI app that serves the React frontend,
        handles WebSocket broadcast, SSE chat streaming, and all DB/RAG routes.
        Previously split across Node's server.js and an internal FastAPI bridge."""
        app = FastAPI(docs_url=None, redoc_url=None)
        module = self

        if self._vault is not None:
            from core.auth_middleware import BearerAuthMiddleware
            app.add_middleware(
                BearerAuthMiddleware,
                token_provider=lambda: self._vault.get_or_create_secret(
                    self._vault.DEFAULT_GPT_SERVER_SECRET_KEY
                ),
            )

        # ── DB endpoints ───────────────────────────────────────────

        @app.get("/db/status")
        async def db_status():
            db_path = module._get_db_path()
            if db_path is None:
                return {"connected": False, "db_exists": False,
                        "tables_ready": False, "tables": []}
            exists = db_path.exists()
            tables: list[str] = []
            if exists:
                async with (await module._open_db(db_path)) as db:
                    cur = await db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in await cur.fetchall()]
            ready = all(t in tables for t in _REQUIRED_TABLES)
            return {
                "connected": True,
                "db_exists": exists,
                "tables_ready": ready,
                "tables": tables,
            }

        @app.post("/db/init")
        async def db_init():
            db_path = module._get_db_path()
            if db_path is None:
                return JSONResponse(
                    {"error": "No database module wired"}, status_code=400
                )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            async with (await module._open_db(db_path)) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA foreign_keys=ON")
                await db.executescript(_DDL_SQLITE)
                await db.commit()
            log.info("gpt_terminal: initialized tables in %s", db_path)
            return {"ok": True, "db_path": str(db_path)}

        @app.get("/db/threads")
        async def list_threads():
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            async with (await module._open_db(db_path)) as db:
                db.row_factory = async_sqlcipher.dict_row_factory
                cur = await db.execute(
                    "SELECT id, name, created_at, updated_at "
                    "FROM threads ORDER BY updated_at DESC"
                )
                rows = await cur.fetchall()
            return [dict(r) for r in rows]

        @app.post("/db/threads")
        async def create_thread(body: dict):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            tid = body.get("id", "")
            name = body.get("name", "New Thread")
            if not tid:
                return JSONResponse({"error": "id required"}, status_code=400)
            async with (await module._open_db(db_path)) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    "INSERT OR IGNORE INTO threads (id, name) VALUES (?, ?)",
                    (tid, name),
                )
                await db.commit()
            return {"ok": True, "id": tid}

        @app.put("/db/threads/{tid}")
        async def rename_thread(tid: str, body: dict):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            name = body.get("name", "")
            if not name:
                return JSONResponse({"error": "name required"}, status_code=400)
            async with (await module._open_db(db_path)) as db:
                await db.execute(
                    "UPDATE threads SET name=?, updated_at=datetime('now') WHERE id=?",
                    (name, tid),
                )
                await db.commit()
            return {"ok": True}

        @app.delete("/db/threads/{tid}")
        async def delete_thread(tid: str):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            async with (await module._open_db(db_path)) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute("DELETE FROM threads WHERE id=?", (tid,))
                await db.commit()
            return {"ok": True}

        @app.get("/db/threads/{tid}/messages")
        async def list_messages(tid: str):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            async with (await module._open_db(db_path)) as db:
                db.row_factory = async_sqlcipher.dict_row_factory
                cur = await db.execute(
                    "SELECT id, thread_id, role, content, created_at "
                    "FROM messages WHERE thread_id=? ORDER BY created_at ASC",
                    (tid,),
                )
                rows = await cur.fetchall()
            return [dict(r) for r in rows]

        @app.post("/db/threads/{tid}/messages")
        async def create_message(tid: str, body: dict):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            mid = body.get("id", "")
            role = body.get("role", "")
            content = body.get("content", "")
            if not mid or not role:
                return JSONResponse(
                    {"error": "id and role required"}, status_code=400
                )
            async with (await module._open_db(db_path)) as db:
                await db.execute("PRAGMA foreign_keys=ON")
                await db.execute(
                    "INSERT OR IGNORE INTO messages (id, thread_id, role, content) "
                    "VALUES (?, ?, ?, ?)",
                    (mid, tid, role, content),
                )
                await db.execute(
                    "UPDATE threads SET updated_at=datetime('now') WHERE id=?",
                    (tid,),
                )
                await db.commit()
            return {"ok": True}

        @app.get("/db/settings")
        async def list_settings():
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            async with (await module._open_db(db_path)) as db:
                db.row_factory = async_sqlcipher.dict_row_factory
                cur = await db.execute(
                    "SELECT key, value FROM settings WHERE machine_id=?",
                    ("",),
                )
                rows = await cur.fetchall()
            return {str(r["key"]): str(r["value"]) for r in rows}

        @app.put("/db/settings/{setting_key:path}")
        async def upsert_setting(setting_key: str, body: dict):
            db_path = module._get_db_path()
            if db_path is None or not db_path.exists():
                return JSONResponse({"error": "DB not available"}, status_code=503)
            key = str(setting_key or "").strip()
            if not key:
                return JSONResponse({"error": "key required"}, status_code=400)
            value = str(body.get("value", ""))
            async with (await module._open_db(db_path)) as db:
                await db.execute(
                    "INSERT INTO settings (key, machine_id, value) VALUES (?, ?, ?) "
                    "ON CONFLICT(key, machine_id) DO UPDATE SET "
                    "value=excluded.value, updated_at=datetime('now')",
                    (key, "", value),
                )
                await db.commit()
            return {"ok": True, "key": key, "value": value}

        # ── RAG endpoints ──────────────────────────────────────────

        @app.get("/rag/status")
        async def rag_status():
            connected, vector_count = module._rag_link_active()
            return {"connected": connected, "vector_count": vector_count}

        @app.post("/rag/query")
        async def rag_query(body: dict):
            query_text = str(body.get("query_text", "")).strip()
            top_k      = max(1, int(body.get("top_k", 5)))
            request_id = str(body.get("request_id", "")).strip() or \
                         f"rag_{int(time.time() * 1000)}"
            filters = body.get("filters", {})
            if not isinstance(filters, dict):
                filters = {}
            result = await module._execute_rag_query(
                query_text,
                top_k,
                request_id,
                filters=filters,
            )
            return {**result, "request_id": request_id, "query_text": query_text}

        @app.get("/rag/metrics")
        async def rag_metrics():
            rag_peer = module._find_rag_peer()
            if rag_peer is None:
                return JSONResponse({"error": "RAG not connected"}, status_code=503)
            hv = module._hypervisor
            if hv is None:
                return JSONResponse({"error": "Hypervisor unavailable"}, status_code=503)
            try:
                result = await hv.call_module_method(rag_peer.module_id, "get_metrics")
                return result if isinstance(result, dict) else {"error": "metrics unavailable"}
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)

        # ── Health ─────────────────────────────────────────────────

        @app.get("/api/health")
        async def health():
            return {
                "status":     "ok",
                "llm_linked": module._llm_linked,
                "llm_ready":  module._llm_linked and module._llm_reachable,
                "clients":    len(module._ws_clients),
            }

        @app.get("/api/llm/models")
        async def llm_models():
            if not module._llm_linked:
                return JSONResponse({"error": "LLM not linked"}, status_code=503)
            reachable = module._llm_reachable or await module._probe_llm_health()
            if not reachable:
                return JSONResponse({"error": "LLM backend unreachable"}, status_code=503)
            llm_url = f"{module._llm_base_url()}/v1/models"
            client_kwargs = module._local_httpx_kwargs(module._llm_protocol)
            async with httpx.AsyncClient(**client_kwargs) as client:
                try:
                    resp = await client.get(
                        llm_url,
                        timeout=5.0,
                        headers=module._llm_auth_headers(),
                    )
                    if resp.status_code != 200:
                        return JSONResponse(
                            {"error": f"LLM {resp.status_code}: {resp.text}"},
                            status_code=resp.status_code,
                        )
                    payload = resp.json()
                    if not isinstance(payload, dict):
                        return {"data": []}
                    return payload
                except Exception as exc:
                    return JSONResponse({"error": str(exc)}, status_code=500)

        @app.get("/auth/session")
        async def auth_session(request: Request):
            token = None
            if (
                module._vault is not None
                and (
                    module._is_loopback_request(request)
                    or module._allow_remote_session_token(request)
                )
            ):
                token = module._vault.get_or_create_secret(
                    module._vault.DEFAULT_GPT_SERVER_SECRET_KEY
                )
            return {
                "auth_enabled": module._vault is not None,
                "token": token,
            }

        @app.get("/voice/token")
        async def voice_token(identity: str = "", room: str = ""):
            if not module._voice_available():
                return JSONResponse({"error": "Voice bridge is not available"}, status_code=503)

            cfg = dict(module._voice_config)
            ws_url = str(cfg.get("livekit_ws_url", "")).strip()
            api_key = str(cfg.get("livekit_api_key", "")).strip()
            api_secret = str(cfg.get("livekit_api_secret", "")).strip()
            room_name = str(room or cfg.get("livekit_room", "voice")).strip() or "voice"
            identity_name = str(identity).strip() or f"web-{int(time.time() * 1000)}"
            if not (ws_url and api_key and api_secret):
                return JSONResponse({"error": "Voice config missing credentials"}, status_code=503)
            try:
                from livekit import api  # type: ignore

                token = (
                    api.AccessToken(api_key, api_secret)
                    .with_identity(identity_name)
                    .with_name(identity_name)
                    .with_grants(api.VideoGrants(room_join=True, room=room_name))
                    .to_jwt()
                )
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)
            return {
                "token": token,
                "url": ws_url,
                "room": room_name,
                "identity": identity_name,
            }

        # ── Internal: Python updates LLM config ────────────────────
        # These endpoints are still exposed so any external tooling can call
        # them; internally Python now calls the state fields directly.

        @app.post("/api/internal/config")
        async def internal_config(body: dict):
            if "llm_host"  in body: module._llm_host       = str(body["llm_host"])
            if "llm_port"  in body: module._llm_port       = int(body["llm_port"])
            if "llm_model" in body: module._llm_model      = body["llm_model"]
            if "ctx_length" in body: module._llm_ctx_length = int(body["ctx_length"])
            asyncio.create_task(module._probe_and_broadcast_llm_state())
            return {"ok": True}

        @app.post("/api/internal/wiring")
        async def internal_wiring(body: dict):
            module._llm_linked = bool(body.get("llm_linked", False))
            await module._broadcast_config_ws()
            return {"ok": True, "llm_linked": module._llm_linked}

        @app.post("/api/internal/notify")
        async def internal_notify(body: dict):
            await module._broadcast_ws(body)
            return {"ok": True, "sent_to": len(module._ws_clients)}

        # ── Chat: SSE streaming from LLM ───────────────────────────

        @app.post("/api/chat")
        async def chat(body: dict):
            messages   = list(body.get("messages", []))
            request_id = str(body.get("request_id") or
                             f"py_{int(time.time() * 1000):x}")
            thread_id  = body.get("thread_id")
            rag_enabled = bool(body.get("rag_enabled", True))

            async def stream():
                if module._agent_wire_active():
                    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
                    module._pending_agent_streams[request_id] = q
                    try:
                        agent_payload = {
                            "action": "chat_request",
                            "request_id": request_id,
                            "thread_id": thread_id,
                            "messages": messages,
                            "temperature": float(body.get("temperature", 0.7)),
                            "rag_enabled": rag_enabled,
                        }
                        for key in _LLM_OPTIONAL_KEYS:
                            if key in body and body[key] is not None:
                                agent_payload[key] = body[key]

                        await module.inputs["AGENT_IN"].emit(
                            Payload(
                                signal_type=SignalType.AGENT_PAYLOAD,
                                source_module=module.module_id,
                                data=agent_payload,
                            )
                        )

                        completion_id = f"chatcmpl-{request_id[:8]}"
                        created = int(time.time())
                        model_name = str(module._agent_status.get("llm_model") or module._llm_model or "local")
                        while True:
                            evt = await q.get()
                            action = str(evt.get("action", "")).strip().lower()
                            if action == "token":
                                token = str(evt.get("content", "") or "")
                                if token:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": token},
                                            "finish_reason": None,
                                        }],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            elif action == "error":
                                yield f"data: {json.dumps({'error': str(evt.get('error', 'agent error'))})}\n\n"
                                return
                            elif action == "stream_end":
                                final_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": str(evt.get("finish_reason", "stop")),
                                    }],
                                }
                                yield f"data: {json.dumps(final_chunk)}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                    finally:
                        module._pending_agent_streams.pop(request_id, None)

                if not module._llm_linked:
                    err = "LLM link unplugged (LOCAL_IP wire missing)"
                    log.warning("gpt_terminal: /api/chat unavailable: %s", err)
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return

                reachable = module._llm_reachable or await module._probe_llm_health()
                if not reachable:
                    err = (f"LLM backend unreachable at "
                           f"{module._llm_host}:{module._llm_port}")
                    log.warning("gpt_terminal: /api/chat unavailable: %s", err)
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return

                rag_connected, _ = module._rag_link_active()
                tool_mode_enabled = bool(rag_enabled and rag_connected)
                rag_mode = "disabled" if not rag_enabled else ("tool" if tool_mode_enabled else "disabled")
                if rag_enabled and any(m.get("role") == "user" for m in messages):
                    try:
                        rag_query_text = await module._build_rag_query(messages)
                        rag_filters: dict[str, Any] = {}
                        rag = await module._execute_rag_query(
                            rag_query_text,
                            top_k=5,
                            request_id=request_id,
                            filters=rag_filters,
                        )
                        rag_chunks = list(rag.get("chunks") or [])
                        top1_score = max(
                            (float(c.get("score", 0.0)) for c in rag_chunks),
                            default=0.0,
                        )
                        if rag_chunks and top1_score >= _RAG_AUTO_INJECT_THRESHOLD:
                            tier = module._model_tier()
                            token_budget = max(
                                128,
                                int(module._llm_ctx_length * 0.25) if module._llm_ctx_length else 512,
                            )
                            selected = module._select_snippets(
                                rag_chunks,
                                token_budget=token_budget,
                            )
                            if selected:
                                ctx = "\n\n".join(
                                    (
                                        f"[source:{c.get('source') or 'unknown'}] "
                                        f"{c.get('text') or ''}"
                                    )
                                    for c in selected
                                )
                                grounding = (
                                    "Answer using ONLY the context below. Say 'I don't know' if unsure."
                                    if tier == "small"
                                    else (
                                        "IMPORTANT: Base your answer ONLY on the provided document context below.\n"
                                        "If the context does not contain enough information, say so explicitly.\n"
                                        "Do not fabricate facts. Cite sources using [source:filename]."
                                    )
                                )
                                ctx_msg = {
                                    "role": "system",
                                    "content": (
                                        f"{grounding}\n\nRelevant context from your documents:"
                                        f"\n\n{ctx}"
                                    ),
                                }
                                sys_idx = next(
                                    (i for i, m in enumerate(messages)
                                     if m.get("role") == "system"),
                                    -1,
                                )
                                if sys_idx >= 0:
                                    messages.insert(sys_idx + 1, ctx_msg)
                                else:
                                    messages.insert(0, ctx_msg)
                                rag_mode = "auto"
                                rag_debug = {
                                    "type": "rag_debug",
                                    "mode": rag_mode,
                                    "request_id": request_id,
                                    "query_text": rag_query_text,
                                    "retrieved_count": int(rag.get("count", 0)),
                                    "injected_count": len(selected),
                                    "top1_score": top1_score,
                                    "latency_ms": float(rag.get("latency_ms", 0.0)),
                                    "retrieval_method": rag.get("retrieval_method", ""),
                                    "model_tier": tier,
                                    "token_budget": token_budget,
                                    "token_used_context": sum(
                                        module._estimate_tokens(str(c.get("text", "")))
                                        for c in selected
                                    ),
                                    "sources": [
                                        {
                                            "source": c.get("source"),
                                            "score": c.get("score"),
                                            "file_type": c.get("file_type", ""),
                                        }
                                        for c in selected
                                    ],
                                }
                                log.info("rag_turn: %s", json.dumps(rag_debug))
                                await module._broadcast_ws(rag_debug)
                                log.debug(
                                    "gpt_terminal: RAG auto-injected %d chunks request_id=%s",
                                    len(selected), request_id,
                                )
                        if rag_mode != "auto":
                            await module._broadcast_ws({
                                "type": "rag_debug",
                                "mode": "tool" if tool_mode_enabled else "disabled",
                                "request_id": request_id,
                                "query_text": rag_query_text,
                                "retrieved_count": int(rag.get("count", 0)),
                                "injected_count": 0,
                                "top1_score": top1_score,
                                "latency_ms": float(rag.get("latency_ms", 0.0)),
                                "retrieval_method": rag.get("retrieval_method", ""),
                                "sources": [],
                            })
                    except Exception as rag_err:
                        log.debug(
                            "gpt_terminal: RAG injection skipped request_id=%s: %s",
                            request_id, rag_err,
                        )
                        await module._broadcast_ws({
                            "type": "rag_debug",
                            "mode": "tool" if tool_mode_enabled else "disabled",
                            "request_id": request_id,
                            "query_text": "",
                            "retrieved_count": 0,
                            "injected_count": 0,
                            "top1_score": 0.0,
                            "latency_ms": 0.0,
                            "retrieval_method": "",
                            "sources": [],
                        })
                elif not rag_enabled:
                    await module._broadcast_ws({
                        "type": "rag_debug",
                        "mode": "disabled",
                        "request_id": request_id,
                        "query_text": "",
                        "retrieved_count": 0,
                        "injected_count": 0,
                        "top1_score": 0.0,
                        "latency_ms": 0.0,
                        "retrieval_method": "",
                        "sources": [],
                    })

                llm_url = f"{module._llm_base_url()}/v1/chat/completions"
                base_payload: dict[str, Any] = {
                    "request_id":  request_id,
                    "thread_id":   thread_id,
                    "model":       module._llm_model or "local",
                    "temperature": float(body.get("temperature", 0.7)),
                }
                for key in _LLM_OPTIONAL_KEYS:
                    if key in body and body[key] is not None:
                        base_payload[key] = body[key]

                all_tools: list[dict[str, Any]] = []
                if tool_mode_enabled:
                    all_tools.append(_RAG_TOOL_SCHEMA)
                all_tools.extend(module._llm_tools_payload())
                has_tools = bool(all_tools)

                if has_tools:
                    base_payload["tools"] = all_tools
                    base_payload["tool_choice"] = "auto"

                def _decode_tool_args(raw: Any) -> dict[str, Any]:
                    if isinstance(raw, dict):
                        return raw
                    if isinstance(raw, str) and raw.strip():
                        try:
                            parsed = json.loads(raw)
                            return parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            return {}
                    return {}

                def _rag_tool_result_content(
                    *,
                    query_text: str,
                    rag_result: dict[str, Any],
                    top_k: int,
                ) -> str:
                    rows = list(rag_result.get("chunks") or [])[:max(1, top_k)]
                    return json.dumps({
                        "query": query_text,
                        "count": int(rag_result.get("count", 0)),
                        "latency_ms": float(rag_result.get("latency_ms", 0.0)),
                        "retrieval_method": str(rag_result.get("retrieval_method", "")),
                        "results": [
                            {
                                "source": r.get("source"),
                                "score": float(r.get("score", 0.0)),
                                "text": str(r.get("text", "")),
                                "file_type": str(r.get("file_type", "")),
                            }
                            for r in rows
                        ],
                    })

                client_kwargs = module._local_httpx_kwargs(module._llm_protocol)
                async with httpx.AsyncClient(**client_kwargs) as client:
                    try:
                        working_messages = list(messages)
                        if has_tools:
                            should_stream_final = False
                            for round_idx in range(_MAX_TOOL_ROUNDS):
                                probe_payload = dict(base_payload)
                                probe_payload["messages"] = working_messages
                                probe_payload["stream"] = False
                                probe_resp = await client.post(
                                    llm_url,
                                    json=probe_payload,
                                    timeout=None,
                                    headers=module._llm_auth_headers(),
                                )
                                if probe_resp.status_code != 200:
                                    err = (
                                        f"LLM {probe_resp.status_code}: "
                                        f"{probe_resp.text}"
                                    )
                                    log.error(
                                        "gpt_terminal: LLM probe error request_id=%s: %s",
                                        request_id, err,
                                    )
                                    yield f"data: {json.dumps({'error': err})}\n\n"
                                    return
                                probe_data = probe_resp.json()
                                choice0 = (probe_data.get("choices") or [{}])[0]
                                assistant_msg = choice0.get("message") or {}
                                tool_calls = list(assistant_msg.get("tool_calls") or [])
                                if not tool_calls:
                                    should_stream_final = True
                                    break
                                working_messages.append({
                                    "role": "assistant",
                                    "content": assistant_msg.get("content", "") or "",
                                    "tool_calls": tool_calls,
                                })
                                for call in tool_calls:
                                    fn = ((call.get("function") or {}).get("name") or "").strip().lower()
                                    call_id = str(call.get("id") or f"tool_{request_id}_{round_idx}")
                                    args = _decode_tool_args((call.get("function") or {}).get("arguments"))

                                    await module._broadcast_ws({
                                        "type": "tool_call",
                                        "request_id": request_id,
                                        "tool_call_id": call_id,
                                        "tool_name": fn,
                                        "status": "pending",
                                        "args": args,
                                    })

                                    if fn == "search_documents" and tool_mode_enabled:
                                        tool_query = str(args.get("query", "")).strip()
                                        tool_top_k = min(max(int(args.get("top_k", 5) or 5), 1), 10)
                                        rag_result = await module._execute_rag_query(
                                            tool_query,
                                            top_k=tool_top_k,
                                            request_id=f"{request_id}_tool_{round_idx}_{call_id}",
                                            filters={},
                                        )
                                        rag_tool_chunks = list(rag_result.get("chunks") or [])
                                        rag_tool_top1 = max(
                                            (float(c.get("score", 0.0)) for c in rag_tool_chunks),
                                            default=0.0,
                                        )
                                        await module._broadcast_ws({
                                            "type": "rag_debug",
                                            "mode": "tool",
                                            "request_id": request_id,
                                            "query_text": tool_query,
                                            "retrieved_count": int(rag_result.get("count", 0)),
                                            "injected_count": 0,
                                            "top1_score": rag_tool_top1,
                                            "latency_ms": float(rag_result.get("latency_ms", 0.0)),
                                            "retrieval_method": rag_result.get("retrieval_method", ""),
                                            "sources": [],
                                        })
                                        tool_content = _rag_tool_result_content(
                                            query_text=tool_query,
                                            rag_result=rag_result,
                                            top_k=tool_top_k,
                                        )
                                    else:
                                        ext_result = await module._execute_registered_tool(
                                            tool_name=fn,
                                            arguments=args,
                                            tool_call_id=call_id,
                                            request_id=request_id,
                                        )
                                        tool_content = json.dumps(ext_result, ensure_ascii=False)

                                    working_messages.append({
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "name": fn,
                                        "content": tool_content,
                                    })
                                    await module._broadcast_ws({
                                        "type": "tool_call",
                                        "request_id": request_id,
                                        "tool_call_id": call_id,
                                        "tool_name": fn,
                                        "status": "result",
                                    })
                            if not should_stream_final:
                                err = f"Exceeded max tool rounds ({_MAX_TOOL_ROUNDS})"
                                log.warning(
                                    "gpt_terminal: request_id=%s %s",
                                    request_id, err,
                                )
                                yield f"data: {json.dumps({'error': err})}\n\n"
                                return
                        stream_payload = dict(base_payload)
                        stream_payload["messages"] = working_messages
                        stream_payload["stream"] = True
                        log.debug(
                            "gpt_terminal: /api/chat streaming request_id=%s -> %s",
                            request_id, llm_url,
                        )
                        async with client.stream(
                            "POST", llm_url, json=stream_payload, timeout=None,
                            headers=module._llm_auth_headers(),
                        ) as resp:
                            if resp.status_code != 200:
                                body_bytes = await resp.aread()
                                err = (f"LLM {resp.status_code}: "
                                       f"{body_bytes.decode(errors='replace')}")
                                log.error(
                                    "gpt_terminal: LLM error request_id=%s: %s",
                                    request_id, err,
                                )
                                yield f"data: {json.dumps({'error': err})}\n\n"
                                return
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                    except Exception as exc:
                        log.error(
                            "gpt_terminal: /api/chat exception request_id=%s: %s",
                            request_id, exc,
                        )
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # ── WebSocket: browser real-time push ──────────────────────

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            module._ws_clients.add(ws)
            log.debug(
                "gpt_terminal: browser WS connected (%d total)",
                len(module._ws_clients),
            )
            try:
                # Immediately send current LLM state so the browser knows
                await ws.send_text(json.dumps(module._config_payload()))
                # Push latest tool snapshot for late-connecting browser clients.
                await ws.send_text(
                    json.dumps(module._build_tool_registry_notify_payload())
                )
                # When agent is linked, also push agent-side tool registry
                # (the local _tool_registry may be empty when tools are
                # wired to the agent rather than this terminal directly).
                agent_tools = module._agent_tool_registry_ws_payload()
                if agent_tools is not None:
                    await ws.send_text(json.dumps(agent_tools))
                while True:
                    await ws.receive_text()  # keep-alive; browser messages ignored
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                module._ws_clients.discard(ws)
                log.debug(
                    "gpt_terminal: browser WS disconnected (%d remaining)",
                    len(module._ws_clients),
                )

        # ── Static files + SPA fallback ────────────────────────────

        if _DIST_DIR.exists():
            # Mount /assets separately so API routes take precedence
            assets_dir = _DIST_DIR / "assets"
            if assets_dir.exists():
                app.mount(
                    "/assets",
                    StaticFiles(directory=str(assets_dir)),
                    name="static-assets",
                )

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            # Serve real static files first (favicon.ico, manifest.json, etc.)
            candidate = _DIST_DIR / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(_DIST_INDEX))

        return app

    # ── Lifecycle ───────────────────────────────────────────

    async def _start_services(self, port: int) -> bool:
        """Start the unified Python web server on *port*."""
        if _port_in_use(port):
            if not await _wait_for_port_free(port, timeout_s=2.0):
                log.error(
                    "gpt_terminal[%s]: port %d already in use",
                    self.module_id, port,
                )
                await self.report_state(
                    severity="ERROR",
                    message=f"Port {port} is already occupied. "
                            "Free the port or change gpt_terminal web_port.",
                )
                self.status = ModuleStatus.ERROR
                return False
        if not _DIST_INDEX.exists():
            log.error(
                "gpt_terminal: pre-built frontend not found at %s\n"
                "  The dist/ folder should ship with the repository.\n"
                "  To rebuild from source:\n"
                "    cd modules/gpt_terminal/app && npm install && npm run build",
                _DIST_INDEX,
            )
            await self.report_state(
                severity="ERROR",
                message="Pre-built frontend (dist/) missing — see log for details",
            )
            self.status = ModuleStatus.ERROR
            return False

        api = self._build_app()
        ssl_kwargs: dict = {}
        if self.params.get("tls_enabled", False):
            from core.cert_manager import CertManager
            cert_path, key_path = CertManager().ensure_certs()
            ssl_kwargs = {"ssl_keyfile": str(key_path), "ssl_certfile": str(cert_path)}
        cfg = uvicorn.Config(
            api,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
            **ssl_kwargs,
        )
        self._api_server = uvicorn.Server(cfg)
        self._api_task = asyncio.create_task(self._api_server.serve())

        scheme = "https" if self.params.get("tls_enabled", False) else "http"
        health_url = f"{scheme}://127.0.0.1:{port}/api/health"
        headers = self._llm_auth_headers()
        loop     = asyncio.get_event_loop()
        deadline = loop.time() + _HEALTH_TIMEOUT
        started  = False

        probe_kwargs = self._local_httpx_kwargs(scheme)
        async with httpx.AsyncClient(**probe_kwargs) as probe:
            while loop.time() < deadline:
                await asyncio.sleep(_HEALTH_INTERVAL)
                if self._api_task.done():
                    exc = self._api_task.exception()
                    log.error("gpt_terminal: server failed to start: %s", exc)
                    self._api_server = None
                    self._api_task = None
                    self.status = ModuleStatus.ERROR
                    return False
                try:
                    r = await probe.get(health_url, timeout=1.0, headers=headers)
                    if r.status_code == 200:
                        started = True
                        break
                except Exception:
                    pass

        if not started:
            log.error(
                "gpt_terminal: server did not respond within %.0f s",
                _HEALTH_TIMEOUT,
            )
            await self._stop_api()
            self.status = ModuleStatus.ERROR
            return False

        self.outputs["WEB_OUT"]._listeners.clear()
        self.outputs["WEB_OUT"].on_receive(self._on_web_payload)
        self._runtime_web_port = port
        self._last_services_start_ts = time.monotonic()

        if self._tool_timeout_task is None or self._tool_timeout_task.done():
            self._tool_timeout_task = asyncio.create_task(self._tool_timeout_loop())

        log.info(
            "gpt_terminal[%s]: Python web server ready -> %s://127.0.0.1:%d",
            self.module_id, scheme, port,
        )
        return True

    async def _stop_services(self) -> None:
        await self._stop_api()
        self._runtime_web_port = None
        await asyncio.sleep(_PORT_RELEASE_DELAY)

    async def _restart_reason(self, desired_port: int) -> str | None:
        if self._runtime_web_port != desired_port:
            return "requested web_port changed"
        if self._api_task is None or self._api_task.done():
            return "server task missing or exited"
        start_ts = getattr(self, "_last_services_start_ts", None)
        if start_ts is not None and (time.monotonic() - float(start_ts)) < _RESTART_HEALTH_GRACE_S:
            return None
        try:
            scheme = "https" if self.params.get("tls_enabled", False) else "http"
            headers = self._llm_auth_headers()
            probe_kwargs = self._local_httpx_kwargs(scheme)
            async with httpx.AsyncClient(**probe_kwargs) as probe:
                for _ in range(3):
                    try:
                        r = await probe.get(
                            f"{scheme}://127.0.0.1:{self._runtime_web_port}/api/health",
                            timeout=2.0,
                            headers=headers,
                        )
                        if r.status_code == 200:
                            return None
                    except Exception:
                        pass
                    await asyncio.sleep(0.2)
                return "web health probe failed"
        except Exception:
            return "web health probe exception"

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        if self._vault is None:
            msg = "Vault is required for gpt_terminal."
            log.error("gpt_terminal[%s]: %s", self.module_id, msg)
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=msg)
            return
        if not async_sqlcipher.is_available():
            msg = (
                "sqlcipher3 is required for gpt_terminal DB access. "
                "Run: pip install sqlcipher3-binary"
            )
            log.error("gpt_terminal[%s]: %s", self.module_id, msg)
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=msg)
            return
        desired_port = int(self.params.get("web_port", 3000))
        running_now = self._api_task is not None and not self._api_task.done()
        if running_now:
            if self._runtime_web_port == desired_port:
                self.status = ModuleStatus.RUNNING
                await self.refresh_wiring_state()
                await self.broadcast_routing_config()
                return
            await self._stop_services()

        self._ws_clients: set = set()
        self._llm_host     = "127.0.0.1"
        self._llm_port     = 5000
        self._llm_protocol = "http"
        self._llm_model    = None
        self._llm_ctx_length = 4096
        self._llm_linked   = False
        self._llm_reachable = False
        self._agent_linked = False
        self._agent_status: dict[str, Any] = {}
        self._voice_config: dict[str, Any] = {}
        self._voice_turns: dict[str, dict[str, Any]] = {}
        self._token_buffer: list = []
        self._last_services_start_ts: float | None = None
        self._runtime_web_port: int | None = None
        self._tool_registry: dict[str, dict[str, Any]] = {}
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._pending_rag: dict[str, tuple[asyncio.Event, dict[str, Any]]] = {}
        self._pending_agent_streams: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self.params["tools_count"] = 0
        self.params["tools_discovered"] = False

        if not await self._start_services(desired_port):
            return

        self.status = ModuleStatus.RUNNING
        await self.refresh_wiring_state()
        await self.broadcast_routing_config()

    async def init(self) -> None:
        desired_port = int(self.params.get("web_port", 3000))

        reason = await self._restart_reason(desired_port)
        if reason is not None:
            log.info(
                "gpt_terminal: init reconcile — restarting services "
                "(runtime port %s -> requested %d, reason=%s)",
                self._runtime_web_port, desired_port, reason,
            )
            await self._stop_services()
            if not await self._start_services(desired_port):
                return
            self.status = ModuleStatus.RUNNING

        await self.refresh_wiring_state()
        await self.broadcast_routing_config()

    async def process(self, payload: Payload, source_port: Port) -> None:
        if not hasattr(self, "_ws_clients"):
            # Module has not finished initialize(); drop the payload to avoid
            # AttributeError on any of the runtime-state fields.
            log.debug(
                "gpt_terminal[%s]: process() called before initialize(); "
                "dropping signal=%s port=%s",
                self.module_id,
                payload.signal_type,
                source_port.name,
            )
            return
        data = payload.data if isinstance(payload.data, dict) else {}
        if payload.signal_type == SignalType.SERVER_CONFIG_PAYLOAD:
            if source_port.name == "VOICE_LINK" or str(data.get("kind", "")).strip().lower() == "voice_config":
                self._voice_config = dict(data)
                await self.refresh_wiring_state()
                return
            await self.refresh_wiring_state()
            protocol = str(data.get("protocol", self._llm_protocol)).strip().lower()
            if protocol in {"http", "https"}:
                self._llm_protocol = protocol
            self._llm_host = data.get("host", self._llm_host)
            self._llm_port = int(data.get("port", self._llm_port))
            if "model" in data:
                self._llm_model = data["model"]
            if "ctx_length" in data:
                self._llm_ctx_length = int(data["ctx_length"])
            asyncio.create_task(self._probe_and_broadcast_llm_state())

        elif payload.signal_type == SignalType.BRAIN_STREAM_PAYLOAD:
            if data.get("done"):
                await self._flush_token_batch()
                await self._broadcast_ws({"type": "token", "data": data})
            elif data.get("token"):
                self._token_buffer.append(data)
                if len(self._token_buffer) >= _TOKEN_BATCH_SIZE:
                    await self._flush_token_batch()

        elif payload.signal_type == SignalType.TUNNEL_STATUS_PAYLOAD:
            self.params["public_url"] = str(data.get("public_url", "") or "")
            self.params["tunnel_status"] = str(
                data.get("tunnel_status", data.get("status", "unknown"))
            )
            await self._broadcast_ws({
                "type":   "tunnel_url",
                "url":    data.get("public_url", ""),
                "status": data.get("status", "unknown"),
            })

        elif payload.signal_type == SignalType.CONTEXT_PAYLOAD:
            rid = str(data.get("request_id", ""))
            pending = self._pending_rag.get(rid)
            if pending is not None:
                event, holder = pending
                holder.update(data)
                event.set()

        elif payload.signal_type == SignalType.DB_RESPONSE_PAYLOAD:
            data = payload.data if isinstance(payload.data, dict) else {}
            await self._broadcast_ws({
                "type": "db_response",
                "request_id": str(data.get("request_id", "") or ""),
                "success": bool(data.get("success", False)),
                "operation": str(data.get("operation", "") or ""),
                "rowcount": int(data.get("rowcount", 0) or 0),
                "rows": data.get("rows", []),
                "error": data.get("error"),
            })

        elif payload.signal_type == SignalType.TOOL_SCHEMA_PAYLOAD:
            await self._register_tools(payload)

        elif payload.signal_type == SignalType.TOOL_EXECUTION_PAYLOAD:
            action = str(data.get("action", "")).strip().lower()
            if action == "execute":
                await self._dispatch_tool_execute(payload)
            elif action == "result":
                await self._handle_tool_result(payload)
            else:
                log.debug("gpt_terminal: ignored TOOL_EXECUTION_PAYLOAD action=%s", action)
        elif payload.signal_type == SignalType.AGENT_PAYLOAD:
            await self._handle_agent_payload(payload)

    async def _handle_agent_payload(self, payload: Payload) -> None:
        data = payload.data if isinstance(payload.data, dict) else {}
        action = str(data.get("action", "")).strip().lower()
        request_id = str(data.get("request_id", "")).strip()

        if action == "status":
            self._agent_status = dict(data)
            self.params["tools_count"] = int(data.get("tools_count", self.params.get("tools_count", 0)) or 0)
            self.params["tools_discovered"] = bool(self.params.get("tools_count", 0))
            llm_ready = bool(data.get("llm_ready", False))
            self._llm_linked = bool(data.get("llm_linked", False))
            self._llm_reachable = llm_ready
            if "llm_host" in data:
                self._llm_host = str(data.get("llm_host", self._llm_host))
            if "llm_port" in data:
                self._llm_port = int(data.get("llm_port", self._llm_port))
            if "llm_protocol" in data:
                proto = str(data.get("llm_protocol", self._llm_protocol)).strip().lower()
                if proto in {"http", "https"}:
                    self._llm_protocol = proto
            if "llm_model" in data:
                self._llm_model = str(data.get("llm_model") or self._llm_model)
            if "ctx_length" in data:
                self._llm_ctx_length = int(data.get("ctx_length", self._llm_ctx_length))
            await self._broadcast_ws({"type": "config", **self._config_payload()})
            # Populate the frontend tools panel from agent's tool registry.
            tools_registry = data.get("tools_registry")
            if isinstance(tools_registry, dict) and tools_registry:
                await self._broadcast_ws({
                    "type": "tool_registry",
                    "providers": tools_registry,
                    "active_tool_calls": [],
                })
            return

        if action == "tool_execute":
            await self._handle_agent_tool_execute(data)
            return

        if action == "tool_call":
            await self._broadcast_ws(
                {
                    "type": "tool_call",
                    "request_id": request_id,
                    "tool_call_id": str(data.get("tool_call_id", "")),
                    "tool_name": str(data.get("tool_name", "")),
                    "status": str(data.get("status", "")),
                    "result": data.get("result"),
                    "args": data.get("arguments"),
                }
            )

        if request_id:
            await self._handle_voice_agent_event(action=action, request_id=request_id, data=data)

        q = self._pending_agent_streams.get(request_id)
        if q is not None:
            try:
                q.put_nowait(dict(data))
            except Exception:
                pass

    async def _handle_agent_tool_execute(self, data: dict[str, Any]) -> None:
        tool_name = str(data.get("tool_name", "")).strip().lower()
        tool_call_id = (
            str(data.get("tool_call_id", "")).strip()
            or f"agent_call_{int(time.time() * 1000)}"
        )
        request_id = str(data.get("request_id", "")).strip()
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        if not self._is_namespaced_tool_name(tool_name):
            await self._emit_agent_tool_result(
                request_id=request_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name or "<missing>",
                success=False,
                result=None,
                error={
                    "code": "TOOL_NOT_FOUND_OR_INVALID_NAME",
                    "message": "Tool name must be namespaced as provider_tool",
                },
            )
            return

        entry = self._tool_registry.get(tool_name)
        if entry is None:
            await self._emit_agent_tool_result(
                request_id=request_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error={
                    "code": "TOOL_NOT_FOUND_OR_INVALID_NAME",
                    "message": f"Tool '{tool_name}' not found in registry",
                },
            )
            return

        self._pending_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "requested_at": time.time(),
            "requested_by": f"agent:{request_id}" if request_id else "agent:",
        }
        await self._broadcast_ws(
            {
                "type": "tool_call",
                "status": "pending",
                "request_id": request_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            }
        )
        await self._dispatch_tool_execute_to_provider(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            entry=entry,
        )

    async def _handle_voice_agent_event(self, *, action: str, request_id: str, data: dict[str, Any]) -> None:
        if action == "stream_start":
            input_method = str(data.get("input_method", "")).strip().lower()
            if input_method != "voice_stt":
                return
            thread_id = str(data.get("thread_id", "voice")).strip() or "voice"
            user_text = str(data.get("user_text", "") or "").strip()
            self._voice_turns[request_id] = {
                "thread_id": thread_id,
                "user_text": user_text,
                "assistant_text": "",
            }
            await self._broadcast_ws(
                {
                    "type": "voice_turn",
                    "phase": "start",
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "user_text": user_text,
                }
            )
            return

        turn = self._voice_turns.get(request_id)
        if turn is None:
            return

        if action == "token":
            token = str(data.get("content", "") or "")
            if not token:
                return
            turn["assistant_text"] = str(turn.get("assistant_text", "")) + token
            await self._broadcast_ws(
                {
                    "type": "voice_turn",
                    "phase": "token",
                    "request_id": request_id,
                    "thread_id": str(turn.get("thread_id", "voice")),
                    "content": token,
                }
            )
            return

        if action == "stream_end":
            full_text = str(data.get("full_text", "") or turn.get("assistant_text", "")).strip()
            await self._broadcast_ws(
                {
                    "type": "voice_turn",
                    "phase": "end",
                    "request_id": request_id,
                    "thread_id": str(turn.get("thread_id", "voice")),
                    "user_text": str(turn.get("user_text", "")),
                    "assistant_text": full_text,
                    "finish_reason": str(data.get("finish_reason", "stop")),
                }
            )
            self._voice_turns.pop(request_id, None)
            return

        if action == "error":
            await self._broadcast_ws(
                {
                    "type": "voice_turn",
                    "phase": "error",
                    "request_id": request_id,
                    "thread_id": str(turn.get("thread_id", "voice")),
                    "error": str(data.get("error", "voice turn failed")),
                }
            )
            self._voice_turns.pop(request_id, None)

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
                and peer.status in (ModuleStatus.RUNNING, ModuleStatus.READY)
            ):
                return True
        return False

    async def _register_tools(self, payload: Payload) -> None:
        if not hasattr(self, "_tool_registry"):
            self._tool_registry = {}
        if not hasattr(self, "_pending_tool_calls"):
            self._pending_tool_calls = {}
        data = payload.data if isinstance(payload.data, dict) else {}
        provider = str(data.get("provider", "")).strip().lower()
        tools = data.get("tools", [])
        if not isinstance(tools, list):
            return

        source = payload.source_module
        stale_keys = [
            k for k, v in self._tool_registry.items()
            if v.get("source_module") == source
        ]
        for k in stale_keys:
            del self._tool_registry[k]

        updated = 0
        ignored = 0
        for tool in tools:
            if not isinstance(tool, dict):
                ignored += 1
                continue

            function = tool.get("function", {})
            if not isinstance(function, dict):
                ignored += 1
                continue

            canonical = str(function.get("name", "")).strip().lower()
            if not self._is_namespaced_tool_name(canonical):
                ignored += 1
                await self.report_state(
                    severity="WARN",
                    message=(
                        "gpt_terminal: rejected non-namespaced tool "
                        f"'{canonical or '<missing>'}' from {payload.source_module}"
                    ),
                )
                continue

            existing = self._tool_registry.get(canonical)
            if existing is not None and existing.get("source_module") != payload.source_module:
                await self.report_state(
                    severity="WARN",
                    message=(
                        f"gpt_terminal: tool collision on {canonical}; "
                        f"replacing source {existing.get('source_module')} -> {payload.source_module}"
                    ),
                )

            self._tool_registry[canonical] = {
                "tool_name":     canonical,
                "raw_tool_name": str(tool.get("raw_tool_name", "")).strip().lower(),
                "provider":      provider or canonical.split("_", 1)[0],
                "description":   str(function.get("description", "")),
                "parameters":    function.get("parameters", {"type": "object", "properties": {}}),
                "source_module": payload.source_module,
                "updated_at":    time.time(),
            }
            updated += 1

        if updated > 0 or ignored > 0:
            self.params["tools_count"] = len(self._tool_registry)
            self.params["tools_discovered"] = bool(self._tool_registry)
            await self._broadcast_ws(
                self._build_tool_registry_notify_payload(updated=updated, ignored=ignored)
            )
            await self._sync_tools_to_agent()
        if updated == 0 and tools:
            log.warning(
                "gpt_terminal: received TOOL_SCHEMA_PAYLOAD from %s but registered 0 tools (ignored=%d)",
                payload.source_module,
                ignored,
            )

    async def _dispatch_tool_execute(self, payload: Payload) -> None:
        tool_name    = str(payload.data.get("tool_name", "")).strip().lower()
        tool_call_id = (str(payload.data.get("tool_call_id", "")).strip()
                        or f"call_{int(time.time() * 1000)}")
        arguments = payload.data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        if not self._is_namespaced_tool_name(tool_name):
            await self._emit_tool_error_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name or "<missing>",
                code="TOOL_NOT_FOUND_OR_INVALID_NAME",
                message="Tool name must be namespaced as provider_tool",
            )
            return

        entry = self._tool_registry.get(tool_name)
        if entry is None:
            await self._emit_tool_error_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                code="TOOL_NOT_FOUND_OR_INVALID_NAME",
                message=f"Tool '{tool_name}' not found in registry",
            )
            return

        self._pending_tool_calls[tool_call_id] = {
            "tool_name":    tool_name,
            "requested_at": time.time(),
            "requested_by": payload.source_module,
        }
        await self._broadcast_ws({
            "type":         "tool_call",
            "status":       "pending",
            "tool_call_id": tool_call_id,
            "tool_name":    tool_name,
        })

        await self._dispatch_tool_execute_to_provider(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            entry=entry,
        )

    async def _handle_tool_result(self, payload: Payload) -> None:
        tool_call_id = str(payload.data.get("tool_call_id", "")).strip()
        tool_name    = str(payload.data.get("tool_name", "")).strip().lower()
        if not tool_call_id:
            return

        pending = self._pending_tool_calls.get(tool_call_id)
        requested_by = ""
        if pending is None:
            log.debug(
                "gpt_terminal: received untracked tool result tool_call_id=%s",
                tool_call_id,
            )
        else:
            requested_by = str(pending.get("requested_by", ""))
            if pending.get("tool_name") != tool_name:
                await self.report_state(
                    severity="WARN",
                    message=(
                        f"gpt_terminal: tool result mismatch call={tool_call_id} "
                        f"expected={pending.get('tool_name')} got={tool_name}"
                    ),
                )
            event = pending.get("event")
            if isinstance(event, asyncio.Event):
                pending["result"] = payload.data
                event.set()
            else:
                # Calls initiated by _dispatch_tool_execute are not awaited via event.
                # Preserve legacy behavior by consuming the pending entry immediately.
                self._pending_tool_calls.pop(tool_call_id, None)

        await self._broadcast_ws({
            "type":         "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name":    tool_name,
            "success":      bool(payload.data.get("success", False)),
            "result":       payload.data.get("result"),
            "error":        payload.data.get("error"),
        })
        await self._broadcast_ws({
            "type":         "tool_call",
            "status":       "result",
            "tool_call_id": tool_call_id,
            "tool_name":    tool_name,
            "success":      bool(payload.data.get("success", False)),
        })
        if requested_by.startswith("agent:"):
            request_id = requested_by.split(":", 1)[1]
            await self._emit_agent_tool_result(
                request_id=request_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=bool(payload.data.get("success", False)),
                result=payload.data.get("result"),
                error=payload.data.get("error"),
            )

    async def _emit_tool_error_result(
        self,
        tool_call_id: str,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        await self.inputs["MCP_IN"].emit(
            Payload(
                signal_type=SignalType.TOOL_EXECUTION_PAYLOAD,
                source_module=self.module_id,
                data={
                    "action":       "result",
                    "tool_call_id": tool_call_id,
                    "tool_name":    tool_name,
                    "success":      False,
                    "result":       None,
                    "error":        {"code": code, "message": message},
                },
            )
        )

    async def _dispatch_tool_execute_to_provider(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        entry: dict[str, Any],
    ) -> None:
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
        target_module_id = str(entry.get("source_module", "")).strip()
        emitted = False
        if target_module_id:
            emitted = await self.inputs["MCP_IN"].emit_to(
                execute_payload,
                target_module_id=target_module_id,
            )
            if not emitted:
                log.warning(
                    "gpt_terminal: tool provider for '%s' not wired (target=%s); using broadcast fallback",
                    tool_name,
                    target_module_id,
                )
        if not emitted:
            await self.inputs["MCP_IN"].emit(execute_payload)

    async def _emit_agent_tool_result(
        self,
        *,
        request_id: str,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any,
        error: Any,
    ) -> None:
        payload_data: dict[str, Any] = {
            "action": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "success": success,
            "result": result,
            "error": error,
        }
        if request_id:
            payload_data["request_id"] = request_id
        await self.inputs["AGENT_IN"].emit(
            Payload(
                signal_type=SignalType.AGENT_PAYLOAD,
                source_module=self.module_id,
                data=payload_data,
            )
        )

    def _agent_tool_registry_ws_payload(self) -> dict[str, Any] | None:
        tools_registry = self._agent_status.get("tools_registry")
        if isinstance(tools_registry, dict) and tools_registry:
            return {
                "type": "tool_registry",
                "providers": tools_registry,
                "active_tool_calls": [],
            }
        return None

    async def _sync_tools_to_agent(self) -> None:
        if not self._agent_wire_active():
            return
        await self.inputs["AGENT_IN"].emit(
            Payload(
                signal_type=SignalType.AGENT_PAYLOAD,
                source_module=self.module_id,
                data={
                    "action": "tools_available",
                    "tools": self._llm_tools_payload(),
                    "tools_registry": self._build_tool_registry_notify_payload().get("providers", {}),
                },
            )
        )

    async def _tool_timeout_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                timeout_s = max(5, int(self.params.get("tool_call_timeout_s", 30)))
                now = time.time()
                stale_ids = [
                    call_id
                    for call_id, rec in self._pending_tool_calls.items()
                    if (now - float(rec.get("requested_at", now))) > timeout_s
                ]
                for call_id in stale_ids:
                    rec = self._pending_tool_calls.pop(call_id, None)
                    if rec is None:
                        continue
                    requested_by = str(rec.get("requested_by", ""))
                    tool_name = str(rec.get("tool_name", ""))
                    if requested_by.startswith("agent:"):
                        request_id = requested_by.split(":", 1)[1]
                        await self._emit_agent_tool_result(
                            request_id=request_id,
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            success=False,
                            result=None,
                            error={
                                "code": "TOOL_TIMEOUT",
                                "message": f"Tool call timed out after {timeout_s}s",
                            },
                        )
                    else:
                        await self._emit_tool_error_result(
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            code="TOOL_TIMEOUT",
                            message=f"Tool call timed out after {timeout_s}s",
                        )
        except asyncio.CancelledError:
            return

    def _build_tool_registry_notify_payload(
        self,
        *,
        updated: int = 0,
        ignored: int = 0,
    ) -> dict[str, Any]:
        providers: dict[str, list[dict[str, Any]]] = {}
        for tool_name, entry in sorted(self._tool_registry.items()):
            provider = (str(entry.get("provider", "")).strip().lower()
                        or tool_name.split("_", 1)[0])
            providers.setdefault(provider, []).append(
                {
                    "name":       tool_name,
                    "short_name": (tool_name[len(provider) + 1:]
                                   if tool_name.startswith(f"{provider}_")
                                   else tool_name),
                    "description": str(entry.get("description", "")),
                }
            )
        active_tool_calls = [
            {"tool_call_id": call_id, "tool_name": str(rec.get("tool_name", ""))}
            for call_id, rec in self._pending_tool_calls.items()
        ]
        return {
            "type":             "tool_registry",
            "count":            len(self._tool_registry),
            "updated":          updated,
            "ignored":          ignored,
            "providers":        providers,
            "active_tool_calls": active_tool_calls,
        }

    def _is_namespaced_tool_name(self, name: str) -> bool:
        return bool(_TOOL_NAME_RE.match(name))

    async def _request_tool_schema_sync(self) -> None:
        """Ask connected tool providers to re-announce schemas on CHANNEL links."""
        mcp_in = self.inputs.get("MCP_IN")
        if mcp_in is None or not mcp_in.connected_wires:
            return
        await mcp_in.emit(
            Payload(
                signal_type=SignalType.TOOL_SCHEMA_PAYLOAD,
                source_module=self.module_id,
                data={"request": "sync"},
            )
        )

    def _llm_tools_payload(self) -> list[dict[str, Any]]:
        """Build OpenAI-format tool list from the external tool registry."""
        tools: list[dict[str, Any]] = []
        for name in sorted(self._tool_registry):
            entry = self._tool_registry[name]
            if not isinstance(entry, dict):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": entry.get("tool_name", name),
                    "description": entry.get("description", ""),
                    "parameters": entry.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return tools

    async def _execute_registered_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Dispatch a tool call via MCP_IN and wait for the result."""
        if tool_name not in self._tool_registry:
            return {"success": False, "error": {"code": "TOOL_NOT_FOUND", "message": f"'{tool_name}' not in registry"}}

        event = asyncio.Event()
        self._pending_tool_calls[tool_call_id] = {
            "event": event,
            "tool_name": tool_name,
            "requested_at": time.time(),
            "requested_by": "chat:" + request_id,
        }

        entry = self._tool_registry.get(tool_name, {})
        if not isinstance(entry, dict):
            self._pending_tool_calls.pop(tool_call_id, None)
            return {"success": False, "error": {"code": "TOOL_NOT_FOUND", "message": f"'{tool_name}' not in registry"}}
        await self._dispatch_tool_execute_to_provider(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            entry=entry,
        )

        timeout_s = max(5, int(self.params.get("tool_call_timeout_s", 30)))
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending_tool_calls.pop(tool_call_id, None)
            return {"success": False, "error": {"code": "TOOL_TIMEOUT", "message": f"Timed out after {timeout_s}s"}}

        pending = self._pending_tool_calls.pop(tool_call_id, None)
        if not isinstance(pending, dict):
            return {"success": False, "error": {"code": "TOOL_RESULT_MISSING", "message": "No result"}}
        result_payload = pending.get("result", {})
        if not isinstance(result_payload, dict):
            return {"success": False, "error": {"code": "TOOL_RESULT_INVALID", "message": "Invalid result"}}
        return {
            "success": bool(result_payload.get("success", False)),
            "result": result_payload.get("result"),
            "error": result_payload.get("error"),
            "tool_name": tool_name,
        }

    async def _flush_token_batch(self) -> None:
        if not self._token_buffer:
            return
        batch = self._token_buffer[:]
        self._token_buffer.clear()
        await self._broadcast_ws({"type": "tokens_batch", "batch": batch})

    async def _on_web_payload(self, payload: Payload) -> None:
        if payload.signal_type != SignalType.TUNNEL_STATUS_PAYLOAD:
            return
        data = payload.data
        self.params["public_url"] = str(data.get("public_url", "") or "")
        self.params["tunnel_status"] = str(
            data.get("tunnel_status", data.get("status", "unknown"))
        )
        await self._broadcast_ws({
            "type":   "tunnel_url",
            "url":    data.get("public_url", ""),
            "status": data.get("tunnel_status", data.get("status", "unknown")),
        })

    async def refresh_wiring_state(self) -> None:
        """Update LOCAL_IP wiring state and push to all browser WebSocket clients."""
        was_agent_linked = self._agent_linked
        self._agent_linked = self._agent_wire_active()
        self._llm_linked = self._llm_local_ip_link_active()
        if self._agent_linked:
            self._llm_linked = True
            if not was_agent_linked:
                await self._sync_tools_to_agent()
        await self._broadcast_config_ws()

    async def broadcast_routing_config(self) -> None:
        port = self._runtime_web_port or int(self.params.get("web_port", 3000))
        await self.outputs["WEB_OUT"].emit(
            Payload(
                signal_type=SignalType.ROUTING_CONFIG_PAYLOAD,
                source_module=self.module_id,
                data={
                    "target_protocol":      (
                        "https" if self.params.get("tls_enabled", False) else "http"
                    ),
                    "target_host":          "127.0.0.1",
                    "target_port":          port,
                    "health_check_endpoint": "/api/health",
                    "auth_required":        bool(self._vault is not None),
                },
            )
        )

    def _web_base_url(self) -> str:
        port = self._runtime_web_port or int(self.params.get("web_port", 3000))
        scheme = "https" if self.params.get("tls_enabled", False) else "http"
        return f"{scheme}://127.0.0.1:{port}"

    # ── Readiness ───────────────────────────────────────────

    async def check_ready(self) -> bool:
        return (
            self._runtime_web_port is not None
            and self._api_task is not None
            and not self._api_task.done()
        )

    # ── Shutdown ─────────────────────────────────────────────

    async def _stop_api(self) -> None:
        if self._api_server:
            self._api_server.should_exit = True
        if self._api_task:
            try:
                await asyncio.wait_for(self._api_task, timeout=3.0)
            except asyncio.TimeoutError:
                if self._api_server:
                    self._api_server.force_exit = True
                try:
                    await asyncio.wait_for(self._api_task, timeout=1.5)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    self._api_task.cancel()
                    try:
                        await self._api_task
                    except (asyncio.CancelledError, Exception):
                        pass
            except (asyncio.CancelledError, Exception):
                self._api_task.cancel()
                try:
                    await self._api_task
                except (asyncio.CancelledError, Exception):
                    pass
        self._api_server = None
        self._api_task = None

    async def shutdown(self) -> None:
        if self._tool_timeout_task is not None:
            self._tool_timeout_task.cancel()
            try:
                await self._tool_timeout_task
            except asyncio.CancelledError:
                pass
            self._tool_timeout_task = None
        await self._stop_services()
        self.status = ModuleStatus.STOPPED


def register(hypervisor: Any) -> None:
    module = GptTerminalModule()
    hypervisor.register_module(module)
