# AI Server (`gpt_server`) — Local OpenAI API Bridge

## Purpose

`ai_server` (`gpt_server` runtime) converts local `local_brain` (`basic_brain`)
output into an OpenAI-compatible local HTTP API. It provides
`/v1/chat/completions` and `/v1/models` to downstream consumers such as
`chat_terminal` (`gpt_terminal` runtime).

## Process Model

In-process module that starts a local FastAPI/Uvicorn service and communicates with other modules over transport ports.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `server_port` | int | `5000` | Local HTTP port for the OpenAI-compatible API |
| `cors_policy` | string | `"*"` | CORS policy for local browser clients |
| `endpoint_password` | string | vault default | Bearer password for `/v1/*` endpoints. Auto-generated in vault on first account creation. |

## Ports

| Port Name | Direction | Mode | Accepted Signals | Max Connections |
|---|---|---|---|---|
| `AI_IN` (runtime: `BRAIN_IN`) | IN | PUSH | `BRAIN_STREAM_PAYLOAD` | 1 |
| `API_OUT` (runtime: `LOCAL_IP_OUT`) | OUT | CHANNEL | `SERVER_CONFIG_PAYLOAD`, `BRAIN_STREAM_PAYLOAD`, `CHAT_REQUEST` | 8 |
| `WEB_OUT` | OUT | CHANNEL | `ROUTING_CONFIG_PAYLOAD`, `TUNNEL_STATUS_PAYLOAD` | 1 |

`API_OUT` (runtime: `LOCAL_IP_OUT`) is bidirectional channel transport:

- emits `SERVER_CONFIG_PAYLOAD` on startup
- receives `CHAT_REQUEST` from clients
- emits `BRAIN_STREAM_PAYLOAD` streaming chunks back

## Security

`gpt_server` uses loopback binding (`127.0.0.1`) and enforces bearer auth when vault is active.

- On first account creation, a random key is stored in vault under
  `gpt_server.endpoint_password`.
- `gpt_server` loads that key into `endpoint_password` and requires it on
  `/v1/*` requests.
- `gpt_terminal` and `cloud_brain` use the same vault key automatically for
  internal endpoint calls.
- If no vault is loaded (headless/no account), endpoint remains loopback-local and open.
- `/v1/health` is exempt.
- `/v1/*` inference endpoints require `Authorization: Bearer <token>` whenever auth is active.

## UI Controls

The `ai_server` (`gpt_server`) card includes endpoint auth controls:

- `SET` prompts for a bearer password.
- With vault enabled, blank passwords are rejected.
- `EYE` toggles masked/revealed display of the configured password.
- Screen still shows endpoint IP:port (`127.0.0.1:<port>`).

## Relationship to Other Inference Providers

`chat_terminal.API_IN` (runtime: `gpt_terminal.LOCAL_IP_IN`) can be wired to
exactly one local provider path:

- `ai_server` (`gpt_server`) for local `local_brain` (`basic_brain`) inference
- `cloud_brain` (cloud provider proxy path)

All three publish `SERVER_CONFIG_PAYLOAD` on `API_OUT` (runtime:
`LOCAL_IP_OUT`), so terminal behavior stays consistent.
