# Chat Terminal (`gpt_terminal`) — Local Web UI

## Current Status

Implemented and operational.

> **Frontend build note (dev environment):**
> The React/Vite frontend must be built once in a development checkout:
> ```
> cd modules/gpt_terminal/app
> npm install && npm run build
> ```
> The build artefact (`app/dist/`) is included in release downloads.
> End users never run `npm install` — the pre-built `dist/` ships with the
> application. Node.js and npm are **not required at runtime**.

- `initialize()` starts a single Python web server (Starlette/FastAPI + uvicorn) on `web_port`.
- `process()` handles all inbound signals: `SERVER_CONFIG_PAYLOAD`, `BRAIN_STREAM_PAYLOAD`, `TUNNEL_STATUS_PAYLOAD`, `CONTEXT_PAYLOAD`, `TOOL_SCHEMA_PAYLOAD`, `TOOL_EXECUTION_PAYLOAD`.
- Inference endpoint protocol now follows `SERVER_CONFIG_PAYLOAD.protocol` (`http` or `https`) for health checks and `/v1/chat/completions` calls.
- RAG injection gate fully wired: `QUERY_PAYLOAD` emitted on
  `DOCUMENT_IN` (runtime: `CONTEXT_IN`) before each chat request; context
  chunks injected into the prompt inside the Python SSE handler.
- Tool discovery and dispatch implemented: schema registry, `tool_call_id` correlation, 30 s timeout loop.
- Python server exposes `/db/*`, `/rag/*`, `/api/chat` (SSE), `/ws` (WebSocket), `/api/internal/*`, and serves the Vite `dist/` as static files.

DB access note: `gpt_terminal` reads and writes DB directly via `_open_db()`.
When vault + SQLCipher are available it uses `core.async_sqlcipher`; otherwise
it falls back to `aiosqlite`. It resolves DB paths through the wired `database`
module, so terminal and database module target the same storage location.

---

## Architecture

```
Browser (React/Vite)
        │  WebSocket (/ws)  +  HTTP (/api/*)
        ▼
Python web server  (web_port, default 3000)
   ├── Starlette StaticFiles  →  app/dist/
   ├── FastAPI /api/chat      →  SSE streaming to LLM
   ├── FastAPI /ws            →  WebSocket broadcast hub
   ├── FastAPI /db/*          →  aiosqlite (direct)
   ├── FastAPI /rag/*         →  CONTEXT_IN wire
   └── FastAPI /api/internal/*→  in-memory state update
        │  asyncio payloads
        ▼
gpt_terminal/logic.py  ←──── Hypervisor event loop
```

- **Python `logic.py`** manages the module lifecycle, port wiring, tool registry, RAG correlation gate, LLM state, and WebSocket broadcast.
- **Browser `App.jsx`** is the React chat UI — it POSTs to `/api/chat` and receives streamed tokens. It connects to `/ws` for push notifications.

---

## Process Model

**In-process** — runs inside the Hypervisor process using `LocalTransport`.
A single uvicorn server starts on `web_port`. No child processes; no Node.js dependency.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | string | `"You are a helpful assistant."` | Default system prompt injected at the top of every chat request |
| `save_history` | bool | `true` | Whether to persist chat threads to DB |
| `web_port` | int | `3000` | Port the Python server binds to |
| `tool_call_timeout_s` | int | `30` | Seconds before a pending tool call is auto-timed-out |
| `tools_count` | int | `0` | Number of registered tools (live count, shown in widget) |
| `tools_discovered` | bool | `false` | True once any tool schema has been received |

---

## Ports

| Port | Direction | Mode | Accepted Signals | Notes |
|---|---|---|---|---|
| `AGENT_IN` | IN | CHANNEL | `AGENT_PAYLOAD` | Primary agent relay channel to `agent_engine` (chat_request -> token/tool/status events) |
| `API_IN` (runtime: `LOCAL_IP_IN`) | IN | CHANNEL | `SERVER_CONFIG_PAYLOAD`, `BRAIN_STREAM_PAYLOAD`, `CHAT_REQUEST` | Receives LLM address on boot; receives streamed tokens |
| `STORAGE_LINK` (runtime: `DB_IN_OUT`) | IN | CHANNEL | `DB_QUERY_PAYLOAD`, `DB_TRANSACTION_PAYLOAD`, `DB_RESPONSE_PAYLOAD` | Persistence/app-data attachment; not the agent-facing DB query path. `DB_RESPONSE_PAYLOAD` is the response signal on this same bidirectional port (no separate response port). |
| `DOCUMENT_IN` (runtime: `CONTEXT_IN`) | IN | CHANNEL | `QUERY_PAYLOAD`, `CONTEXT_PAYLOAD` | chat_terminal emits QUERY_PAYLOAD; receives CONTEXT_PAYLOAD back |
| `TOOLS_IN` (runtime: `MCP_IN`) | IN | CHANNEL | `TOOL_SCHEMA_PAYLOAD`, `TOOL_EXECUTION_PAYLOAD` | Schema ingestion + tool dispatch |
| `WEB_OUT` | OUT | CHANNEL | `ROUTING_CONFIG_PAYLOAD`, `TUNNEL_STATUS_PAYLOAD` | Emits local address on boot; receives public URL via listener |
| `VOICE_LINK` | IN | CHANNEL | `SERVER_CONFIG_PAYLOAD` | LiveKit voice config from `livekit_voice` (`kind: voice_config`); enables browser mic when available |

When **`AGENT_IN`** is wired to **`agent_engine`**, chat turns for the browser
are delegated over `AGENT_PAYLOAD`: the engine calls `gpt_server`, runs the
multi-round tool loop (unwrapped tool results, duplicate-call guard), and
streams tokens back. The terminal still supplies RAG-injected messages, tool
schemas, and execution path via existing ports; see `modules/agent_engine/MODULE.md`.

### LiveKit voice (`VOICE_LINK`)

When **`VOICE_LINK`** is wired from **`livekit_voice.VOICE_CONFIG`**, the web UI receives LiveKit join settings (`SERVER_CONFIG_PAYLOAD` / `voice_config`) and can enable the in-browser mic path for voice chat. Setup and rack controls for the voice module are documented in [`modules/livekit_voice/MODULE.md`](../livekit_voice/MODULE.md). Typical preset wiring: `core/presets.py` (`RTC_VOICE_AGENT`, `REMOTE_SECRETARY`).

---

## API Endpoints (single Python server on `web_port`)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Server health + LLM link/reachability state |
| `GET`  | `/auth/session` | Loopback-only auth bootstrap for the web client (returns bearer token when vault-auth is active) |
| `POST` | `/api/chat`   | SSE streaming chat (injects RAG context, streams LLM response) |
| `WS`   | `/ws`         | WebSocket push hub — tokens, tool events, tunnel URL, config |
| `GET`  | `/db/status`  | DB connection and table status |
| `POST` | `/db/init`    | Bootstrap schema (threads, messages, system_state) |
| `GET`  | `/db/threads` | List all threads |
| `POST` | `/db/threads` | Create thread |
| `PUT`  | `/db/threads/{tid}` | Rename thread |
| `DELETE` | `/db/threads/{tid}` | Delete thread (cascades messages) |
| `GET`  | `/db/threads/{tid}/messages` | List messages in thread |
| `POST` | `/db/threads/{tid}/messages` | Append message to thread |
| `GET`  | `/rag/status` | `{connected, vector_count}` for wired rag_engine |
| `POST` | `/rag/query`  | Emit QUERY_PAYLOAD; await + return CONTEXT_PAYLOAD chunks |
| `POST` | `/api/internal/config`  | Update LLM host/port/model/ctx_length |
| `POST` | `/api/internal/wiring`  | Update `llm_linked` state |
| `POST` | `/api/internal/notify`  | Broadcast arbitrary JSON to all WebSocket clients |
| `GET`  | `/*`          | SPA fallback — serves `app/dist/index.html` |

## Security (vault-enabled sessions)

When a vault is active, `BearerAuthMiddleware` is enabled.

- Protected prefixes: `/api/*`, `/v1/*`, `/db/*`, `/rag/*`, `/ws`
- Exempt health endpoints: `/health`, `/v1/health`
- Important: `/api/health` is under `/api/*`, so it requires bearer auth in secured sessions
- Outbound calls to local LLM providers include `Authorization: Bearer <gpt_server.endpoint_password>` from the user's encrypted vault profile via `_llm_auth_headers()`
- Internal startup/reconcile probes to `gpt_terminal` `/api/health` also include the bearer token when vault is active
- For local TLS with self-signed certs, internal probes/use of local inference endpoints allow local certificate verification bypass
- Browser UI obtains the same profile-backed bearer token from `/auth/session` and attaches it to `/db/*` and `/api/chat` requests (with auto-refresh on `401`)
- This ensures the web app can call `ai_server` (`gpt_server`) through
  `chat_terminal` (`gpt_terminal`) whenever `API_IN` (runtime: `LOCAL_IP_IN`)
  is wired to `ai_server`.

---

## RAG Gate

When `DOCUMENT_IN` (runtime: `CONTEXT_IN`) is wired to a running
`knowledge_engine` (`rag_engine`), the Python `/api/chat`
SSE handler queries RAG before forwarding each request to the LLM. Python emits
`QUERY_PAYLOAD`, waits up to 5 seconds for the matching `CONTEXT_PAYLOAD`, then
injects the chunks as a system message immediately after any existing system prompt.

If the wire is disconnected, the store is empty, or the call times out, chat
proceeds without context — zero latency penalty. Note that adding or removing
the `DOCUMENT_IN` wire (runtime: `CONTEXT_IN`) causes the Hypervisor to demote
both `chat_terminal` and `knowledge_engine` from `READY` to `RUNNING`
(standby) until they re-establish
readiness.

```
User message arrives at /api/chat
        │
        ├─ _execute_rag_query(query_text, request_id)
        │     │
        │     ├─ emit QUERY_PAYLOAD on CONTEXT_IN wire
        │     ├─ rag_engine embeds + searches
        │     ├─ rag_engine emits CONTEXT_PAYLOAD on wire
        │     ├─ process() resolves asyncio.Event
        │     └─ returns {chunks}
        │
        ├─ inject context system message into messages list
        └─ stream POST to LLM /v1/chat/completions via httpx
```
