# Agent Engine (`agent_engine`) — Central Agent Orchestrator

## Purpose

`agent_engine` centralizes the chat turn loop and tool orchestration for client
adapters (`gpt_terminal`, `discord_terminal`). It receives client requests over
`AGENT_OUT`, executes tool rounds, and streams model tokens back as
`AGENT_PAYLOAD` events.

## Critical Routing Guarantee

LLM requests are sent to `gpt_server` HTTP endpoints only. `agent_engine` does
not call `basic_brain` directly and does not bypass server wiring.

## Ports

| Port | Direction | Mode | Accepted Signals |
|---|---|---|---|
| `API_IN` | IN | CHANNEL | `SERVER_CONFIG_PAYLOAD`, `BRAIN_STREAM_PAYLOAD`, `CHAT_REQUEST` (accepts full `LOCAL_IP_OUT` signal set; only consumes config) |
| `TOOLS_IN` | IN | CHANNEL | `TOOL_SCHEMA_PAYLOAD`, `TOOL_EXECUTION_PAYLOAD` |
| `DB_IN` | IN | CHANNEL | `DB_QUERY_PAYLOAD`, `DB_RESPONSE_PAYLOAD`, `DB_TRANSACTION_PAYLOAD` |
| `CONTEXT_IN` | IN | CHANNEL | `QUERY_PAYLOAD`, `CONTEXT_PAYLOAD` |
| `AGENT_OUT` | OUT | CHANNEL | `AGENT_PAYLOAD` |

## AGENT_PAYLOAD Actions

Client -> engine:

- `chat_request`: `messages`, `request_id`, optional model/sampling fields.
- `cancel`: stop an in-flight request by `request_id`.

Engine -> client:

- `stream_start`
- `token`
- `tool_call` (`pending` / `completed`)
- `stream_end`
- `status`
- `error`

## Tool loop (HTTP → local brain)

`agent_engine` implements the multi-round OpenAI-style tool loop in-process:

1. **Request** — For each `chat_request`, it POSTs to `gpt_server`
   `/v1/chat/completions` with `messages`, merged tool schemas from `TOOLS_IN`,
   and streaming enabled where supported.
2. **Tool rounds** — When the completion finishes with `tool_calls`, the
   engine executes each call via `TOOLS_IN` (`TOOL_EXECUTION_PAYLOAD`), appends
   assistant + `role: "tool"` messages, and POSTs again until the model returns
   text without tools or a round limit is hit.
3. **Result content for the model** — On success, each `role: "tool"` message’s
   `content` is JSON for the **inner payload only** (the value returned as
   `result` from execution), not a wrapper object with `success` /
   `tool_name`. On failure, `content` is a small JSON object with
   `success: false` and an `error` field. This keeps the next completion
   focused on the actual calendar/email/etc. data instead of opaque metadata.
4. **Duplicate call guard** — If the model proposes the same tool name with
   the same arguments (canonical JSON, sorted keys) as in an earlier round of
   the same request, the engine **stops** the tool loop for that request, logs a
   warning, and does not execute or append that duplicate round. This avoids
   burning the round budget on repeated identical calls (for example listing
   calendar events twice).
5. **Round limit** — The maximum number of tool rounds is defined in code as
   `_MAX_TOOL_ROUNDS` (currently `3`). Exceeding it yields an `error` action on
   `AGENT_OUT` with a clear message.

Downstream, `basic_brain` may still rewrite tool-related messages for GGUF chat
templates (see `modules/basic_brain/MODULE.md`: tool prompt injection and
flattened tool-result user messages).

## Notes

- Tool names must be namespaced (`provider_tool`) to be registered.
- Readiness requires an active `API_IN` wire to `gpt_server` and successful
  health probing.
- **`livekit_voice`:** Wire `AGENT_OUT` ↔ `livekit_voice.AGENT_IO` so voice
  STT can emit `chat_request` and receive the same `AGENT_PAYLOAD` stream as
  the chat terminal. See [`modules/livekit_voice/MODULE.md`](../livekit_voice/MODULE.md).

