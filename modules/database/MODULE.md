# Database — Persistence Backend

## Current Status

Implemented and active for both SQLite and Postgres modes.

- SQLite path supports encrypted-at-rest operation (SQLCipher) when a vault is available.
- Postgres path is implemented with `asyncpg` (`db_type = "postgres"`).
- Query/transaction routing over `STORAGE_LINK` (runtime: `DB_IN_OUT`) is fully implemented.
- Sensitive Postgres password field is marked via `SENSITIVE_PARAMS = {"pg_password"}` and serializes as `__vault__` in snapshots.

## Purpose

Persistent relational storage for chat history, templates, machine settings, and app state.
This module is the shared data backend used by `gpt_terminal`, template storage, and machine settings.

## Process Model

In-process module.

- SQLite mode: uses `aiosqlite` fallback or SQLCipher wrapper (`core/async_sqlcipher.py`) when vault + sqlcipher are available.
- Postgres mode: async pool via `asyncpg`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_type` | string | `"sqlite"` | Backend engine: `sqlite` or `postgres` |
| `db_filepath` | string | `"miniloader_data.db"` | SQLite DB file path (relative paths resolve under `vault.get_user_data_dir()` when vault exists) |
| `pg_host` | string | `"localhost"` | Postgres host |
| `pg_port` | int | `5432` | Postgres port |
| `pg_user` | string | `"postgres"` | Postgres username |
| `pg_password` | string | `""` | Postgres password (vault key: `database.pg_password`) |
| `pg_database` | string | `"miniloader"` | Postgres database name |
| `access_mode` | string | `"read_write"` | `read_write` or `read_only` |
| `pg_connected` | bool | `false` | Runtime connection status (display/debug) |

## Port Contract

| Port Name | Direction | Mode | Accepted Signals | Max Connections |
|---|---|---|---|---|
| `STORAGE_LINK` (runtime: `DB_IN_OUT`) | IN | CHANNEL | `DB_QUERY_PAYLOAD`, `DB_TRANSACTION_PAYLOAD`, `DB_RESPONSE_PAYLOAD` | 4 |

`STORAGE_LINK` (runtime: `DB_IN_OUT`) is bidirectional; `DB_RESPONSE_PAYLOAD` is the response signal
on that same port, not a dedicated response port.

## SQLite Schema (current)

`_DDL_SQLITE` currently creates:

- `system_state`
- `threads`
- `messages`
- `templates`
- `settings`

See `docs/DATA_LAYOUT.md` for canonical field-level ownership (vault vs database).

## Query Rules

- Query payloads accept: `SELECT`, `WITH`, and `PRAGMA` (SQLite mode).
- Postgres mode rejects `PRAGMA` with `PRAGMA_NOT_SUPPORTED`.
- Transaction payloads are blocked when `access_mode = "read_only"`.

## Security Notes

- `pg_password` uses vault secret key `database.pg_password`.
- Snapshot persistence replaces sensitive params with `__vault__` and hydrates on load.
- SQLite file encryption key is derived per-user/per-vault via HKDF from the vault master key.
