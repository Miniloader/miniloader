# Discord Terminal (`discord_terminal`) — Discord Chat Bridge

## Purpose

`discord_terminal` mirrors the `gpt_terminal` rack port contract, but uses a
Discord bot as the user interface instead of a web frontend. Messages received
in configured Discord channels are forwarded to an OpenAI-compatible backend
(usually `gpt_server`) and replies are posted back to Discord.

## Process Model

In-process module that runs a `discord.py` client on the main asyncio loop and
communicates with peer modules over transport ports.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `discord_bot_token` | string | `""` | Discord bot token (sensitive) |
| `channel_ids` | string | `""` | Comma-separated allowlist of Discord channel IDs; blank allows all channels |
| `require_mention` | bool | `true` | Only respond when the bot is @mentioned; `!debug` and `!reset` always work |
| `system_prompt` | string | `"You are a helpful assistant."` | Base system prompt for each chat turn |
| `use_channel_history` | bool | `true` | Use Discord channel history as conversation context source |
| `history_window_messages` | int | `10` | Number of recent channel messages to pull before the current turn |
| `history_token_ratio` | float | `0.35` | Fraction of model context window reserved for message history |
| `save_history` | bool | `true` | Keep per-channel in-memory history |
| `rag_enabled` | bool | `true` | Enable optional RAG context injection |
| `max_history_messages` | int | `14` | Maximum stored message entries per channel |
| `companion_url` | string | `""` | Optional companion control URL (for example `http://127.0.0.1:7892`) |
| `companion_secret` | string | `""` | Optional shared secret used for companion attach/detach/heartbeat |
| `companion_lease_seconds` | int | `35` | Companion handoff lease timeout in seconds |
| `companion_heartbeat_interval_sec` | int | `10` | How often to renew companion lease while module is running |

## Ports

| Port Name | Direction | Mode | Accepted Signals | Max Connections |
|---|---|---|---|---|
| `AGENT_IN` | IN | CHANNEL | `AGENT_PAYLOAD` | 1 |
| `API_IN` (runtime: `LOCAL_IP_IN`) | IN | CHANNEL | `SERVER_CONFIG_PAYLOAD`, `BRAIN_STREAM_PAYLOAD`, `CHAT_REQUEST` | 1 |
| `STORAGE_LINK` (runtime: `DB_IN_OUT`) | IN | CHANNEL | `DB_QUERY_PAYLOAD`, `DB_TRANSACTION_PAYLOAD`, `DB_RESPONSE_PAYLOAD` | 1 |
| `DOCUMENT_IN` (runtime: `CONTEXT_IN`) | IN | CHANNEL | `QUERY_PAYLOAD`, `CONTEXT_PAYLOAD` | 1 |
| `TOOLS_IN` (runtime: `MCP_IN`) | IN | CHANNEL | `TOOL_SCHEMA_PAYLOAD`, `TOOL_EXECUTION_PAYLOAD` | 1 |
| `WEB_OUT` | OUT | CHANNEL | `ROUTING_CONFIG_PAYLOAD`, `TUNNEL_STATUS_PAYLOAD` | 8 |

`AGENT_IN` is the preferred path for centralized orchestration via `agent_engine`
(multi-round `/v1/chat/completions` tool loop, unwrapped tool result JSON to the
model, duplicate tool+args suppression). When wired, Discord turns flow as
`AGENT_PAYLOAD` like `gpt_terminal`; see `modules/agent_engine/MODULE.md`.

**LLM HTTP endpoint:** the module calls the OpenAI-compatible backend at `<protocol>://<host>:<port>/v1/chat/completions` using values from `SERVER_CONFIG_PAYLOAD` on `LOCAL_IP_IN` (defaults in `logic.py` are `http` / `127.0.0.1` / `5000` until config arrives). That is an **outbound client** target, not a listening port on this module.

## Chat Flow

1. A user posts a Discord message in an allowed channel.
2. The module validates LLM link and backend reachability.
3. The module reads recent Discord channel messages (`history_window_messages`) as the source-of-truth context and trims to a token budget (`history_token_ratio * ctx_length`).
4. If `AGENT_IN` is wired to `agent_engine`, it emits `AGENT_PAYLOAD` (`chat_request`) and waits for streamed `token/error/stream_end` events.
5. Without an active agent wire, optional RAG query is sent on `CONTEXT_IN` (`QUERY_PAYLOAD`) and consumed via `CONTEXT_PAYLOAD`.
6. Direct mode sends a streaming request to `/v1/chat/completions`.
7. SSE deltas are accumulated into final text.
8. The reply is sent back to Discord (split to 2000-char chunks when needed).

## Companion Handoff Mode (Optional)

When `companion_url` is configured, `discord_terminal` attempts a single-token handoff:

- On module initialize: `POST /attach` so the always-on companion releases the token.
- While running: periodic `POST /heartbeat` renews the handoff lease.
- On module shutdown: `POST /detach` so companion reclaims the token and returns to basic mode.

## UI Controls

The `discord_terminal` rack card includes direct setup controls:

- `TOKEN` button: opens a masked input dialog for `discord_bot_token`.
  - Token is saved to module params.
  - When a vault is active, token is also written to `discord_terminal.bot_token`.
- `CHAN` button: sets the comma-separated `channel_ids` allowlist.
  - Leave blank to allow all channels.
  - Accepts plain IDs or mention-like values (for example `<#123...>`); digits are extracted automatically.
- Status area shows:
  - Bot online/offline state
  - Token present/missing
  - Number of configured channels
  - LLM/DB/RAG link indicators

## Notes

- Bot only responds when @mentioned (default). Set `require_mention` to `false` to respond to all messages.
- Module reports `READY` only when a Discord bot token is configured.
- `!reset` is informational in channel-history mode (`use_channel_history=true`) because context comes from Discord messages.
- `!debug` prints filter/allowlist diagnostics (works without @mention).
- The @mention is stripped from the message text before sending to the LLM.
- Module includes `discord_bot_token` in `SENSITIVE_PARAMS` for masked UI storage.
- Optional Discord voice dependency: if startup logs show `discord.client: davey is not installed, voice will NOT be supported`, text chat is unaffected; only Discord voice features are unavailable until `davey` is installed.
