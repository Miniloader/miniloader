# File Vault (`file_access`) — Context Manager

## Current Status

Implemented and operational for non-voice project path:

- `FILES_OUT` document emission implemented via `emit_active_files()`.
- Text extraction implemented for `.txt`, `.md`, `.csv`, `.json`, `.py`, `.yaml`, `.yml`, `.pdf`, `.docx`.
- MCP provider behavior implemented on `TOOLS_OUT`:
  - emits tool schema on init / schema heartbeat
  - executes `file_list`, `file_read_text`, `file_create_text`, `file_write_text`, `file_mkdir`, `file_delete`, `file_stat`
  - returns `TOOL_EXECUTION_PAYLOAD` results (`action="result"`).
- `access_mode` `read_only` blocks all mutating tools (`file_create_text`, `file_write_text`, `file_mkdir`, `file_delete`).
- Expanded card UI (`modules/file_access/widget.py`): **R/W** toggles `access_mode`; **SEND** calls `emit_active_files()`; **BROWSE** sets `root_path`; **FILES** panel lists immediate children of `root_path` (files only, sorted), with **^** / **v** scrolling and per-row checkboxes backed by `file_access_map`; **CHECK ALL** enables every file in that listing (disabled when the folder is empty or missing). Checkbox changes and CHECK ALL run `auto_ingest_if_ready` when configured.
- Integration tests passing in `tests/test_file_access_logic.py`.

## Purpose

Provides local document ingestion for RAG and exposes file tools through MCP.

## Migration Note

`file_access` is now a provider-style MCP module:

- old wiring: `file_access.MCP_IN -> mcp_bus.MCP_UPSTREAM`
- new wiring: `file_access.TOOLS_OUT -> mcp_bus.MCP_UPSTREAM`

Saved racks using the old port must be rewired.

## Process Model

**In-process**. Runs inside Hypervisor process using local memory transport.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `active_files` | list[string] | `[]` | Files selected for extraction (relative to `root_path` or absolute) |
| `root_path` | string | `"."` | Base directory for browsing and tool path resolution |
| `access_mode` | string | `"read_write"` | Write policy (`read_write` / `read_only`) |
| `file_access_map` | dict | `{}` | Map of filename (basename under `root_path`) → enabled; UI and **CHECK ALL** update this; `active_files` is derived on refresh as keys where value is true |

## Ports

| Port Name | Direction | Mode | Accepted Signals | Max Connections |
|---|---|---|---|---|
| `FILES_OUT` | OUT | PUSH | `DOCS_PAYLOAD` | 8 |
| `TOOLS_OUT` | OUT | CHANNEL | `TOOL_SCHEMA_PAYLOAD`, `TOOL_EXECUTION_PAYLOAD` | 8 |

## Tool Behavior Notes

- `file_list` supports scoped path listing with optional recursion and hidden-file filtering.
- `file_create_text` fails when the target already exists.
- `file_write_text` supports create/parent-directory flags and reports whether the file was created or updated.
- `file_delete` refuses deletion of configured `root_path`; directory deletion requires `recursive=true` for non-empty trees.
- All tool paths are sandboxed under `root_path` via `_resolve_under_root`; path-escape attempts return errors.

## Implemented Data Contracts

### `DOCS_PAYLOAD` (from `FILES_OUT`)

```json
{
  "signal_type": "DOCS_PAYLOAD",
  "source_module": "file_access_01",
  "data": {
    "documents": [
      {
        "path": "C:/docs/notes.txt",
        "name": "notes.txt",
        "mime_type": "text/plain",
        "text": "example content",
        "char_count": 15
      }
    ],
    "count": 1,
    "errors": []
  }
}
```

### `TOOL_SCHEMA_PAYLOAD` (from `TOOLS_OUT`)

```json
{
  "signal_type": "TOOL_SCHEMA_PAYLOAD",
  "source_module": "file_access_01",
  "data": {
    "provider": "file",
    "connection_status": "connected",
    "tools": [
      {"type": "function", "function": {"name": "file_list"}},
      {"type": "function", "function": {"name": "file_read_text"}},
      {"type": "function", "function": {"name": "file_create_text"}},
      {"type": "function", "function": {"name": "file_write_text"}},
      {"type": "function", "function": {"name": "file_mkdir"}},
      {"type": "function", "function": {"name": "file_delete"}},
      {"type": "function", "function": {"name": "file_stat"}}
    ]
  }
}
```

### `TOOL_EXECUTION_PAYLOAD` execute -> result

Request:

```json
{
  "signal_type": "TOOL_EXECUTION_PAYLOAD",
  "source_module": "gpt_terminal_01",
  "data": {
    "action": "execute",
    "tool_call_id": "call_123",
    "tool_name": "file_read_text",
    "arguments": {"path": "notes.txt"}
  }
}
```

Response:

```json
{
  "signal_type": "TOOL_EXECUTION_PAYLOAD",
  "source_module": "file_access_01",
  "data": {
    "action": "result",
    "tool_call_id": "call_123",
    "tool_name": "file_read_text",
    "success": true,
    "result": {"path": "C:/docs/notes.txt", "text": "example content"},
    "error": null
  }
}
```
