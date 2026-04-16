"""
database/logic.py — SQLite / PostgreSQL Manager
=================================================
Persistent storage for chat histories, conversation threads, and
application state.  Executes database queries and returns results
back through the same bidirectional DB_IN_OUT wire.

Supported backends
------------------
SQLite  (default) — async via SQLCipher.  File-based; no server required.
PostgreSQL        — async via asyncpg.   Connects to a pre-existing Postgres
                    server; the module never hosts one.  Enable by setting
                    db_type = "postgres" and filling pg_* params.

All payload contracts (DB_QUERY_PAYLOAD, DB_TRANSACTION_PAYLOAD,
DB_RESPONSE_PAYLOAD) are identical across both backends.  SQL written
with SQLite-style ? positional placeholders is converted to $1/$2/… style
automatically when in postgres mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core import async_sqlcipher
from core.base_module import BaseModule, ModuleStatus
from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    SignalType,
)

log = logging.getLogger(__name__)

# ── Optional asyncpg (postgres backend) ─────────────────────────────────────
try:
    import asyncpg as _asyncpg  # type: ignore[import-untyped]
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _asyncpg = None  # type: ignore[assignment]
    _ASYNCPG_AVAILABLE = False

# ── Schema DDL ───────────────────────────────────────────────────────────────

_DDL_SQLITE = """\
CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
);
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
);
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only',
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS templates (
    name          TEXT PRIMARY KEY,
    is_default    INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now')),
    updated_at    TEXT DEFAULT (datetime('now')),
    sync_status   TEXT NOT NULL DEFAULT 'local_only'
);
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT NOT NULL,
    machine_id TEXT NOT NULL DEFAULT '',
    value      TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only',
    PRIMARY KEY (key, machine_id)
);
CREATE TABLE IF NOT EXISTS consent_flags (
    feature     TEXT PRIMARY KEY,
    granted     INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
);
"""

# Postgres DDL — one statement per entry (asyncpg executes them individually)
_DDL_PG = [
    """\
CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)""",
    """\
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)""",
    """\
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    thread_id  TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)""",
    """\
CREATE TABLE IF NOT EXISTS templates (
    name          TEXT PRIMARY KEY,
    is_default    INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    sync_status   TEXT NOT NULL DEFAULT 'local_only'
)""",
    """\
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT NOT NULL,
    machine_id  TEXT NOT NULL DEFAULT '',
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT NOT NULL DEFAULT 'local_only',
    PRIMARY KEY (key, machine_id)
)""",
    """\
CREATE TABLE IF NOT EXISTS consent_flags (
    feature     TEXT PRIMARY KEY,
    granted     INTEGER NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)""",
]


class DatabaseModule(BaseModule):
    MODULE_NAME = "database"
    MODULE_VERSION = "0.2.0"
    MODULE_DESCRIPTION = "SQLite / PostgreSQL persistent storage manager"
    SENSITIVE_PARAMS = {"pg_password"}

    _conn: Any = None
    _pool: Any = None  # asyncpg.Pool when in postgres mode

    def get_default_params(self) -> dict[str, Any]:
        return {
            # ── Backend selection ──────────────────────────
            "db_type": "sqlite",          # "sqlite" | "postgres"
            # ── SQLite params ──────────────────────────────
            "db_filepath": "miniloader_data.db",
            # ── PostgreSQL params ──────────────────────────
            "pg_host": "localhost",
            "pg_port": 5432,
            "pg_user": "postgres",
            "pg_password": "",            # vault-stored in production
            "pg_database": "miniloader",
            # ── Shared ────────────────────────────────────
            "access_mode": "read_write",  # "read_write" | "read_only"
            "pg_connected": False,        # display-only; set by logic
        }

    # ── Ports ────────────────────────────────────────────────────────────────

    def define_ports(self) -> None:
        self.add_input(
            "DB_IN_OUT",
            accepted_signals={
                SignalType.DB_QUERY_PAYLOAD,
                SignalType.DB_TRANSACTION_PAYLOAD,
                SignalType.DB_RESPONSE_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=4,
            description=(
                "Bidirectional storage link (STORAGE_LINK label). Accepts queries "
                "and transactions; returns DB_RESPONSE_PAYLOAD "
                "results back through the same wire. "
                "Up to 4 clients."
            ),
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        db_type = str(self.params.get("db_type", "sqlite")).strip().lower()
        if db_type == "postgres":
            await self._init_postgres()
        else:
            await self._init_sqlite()

    async def _init_sqlite(self) -> None:
        if self._vault is None:
            msg = "Vault is required for sqlite mode."
            log.error("database[%s]: %s", self.module_id, msg)
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=msg)
            return
        if not async_sqlcipher.is_available():
            msg = (
                "sqlcipher3 is required for sqlite mode. "
                "Run: pip install sqlcipher3-binary"
            )
            log.error("database[%s]: %s", self.module_id, msg)
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=msg)
            return
        db_path = self._resolve_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        key_hex = self._vault.derive_db_key().hex()
        self._conn = await async_sqlcipher.connect(str(db_path), key_hex=key_hex)
        log.info("database[%s]: opened encrypted sqlite %s", self.module_id, db_path)
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_DDL_SQLITE)
        await self._conn.commit()
        log.info("database[%s]: connected to sqlite %s", self.module_id, db_path)
        self.status = ModuleStatus.RUNNING

    async def _init_postgres(self) -> None:
        if not _ASYNCPG_AVAILABLE:
            msg = (
                "asyncpg is not installed — postgres mode unavailable. "
                "Run: pip install asyncpg"
            )
            log.error("database[%s]: %s", self.module_id, msg)
            self.params["pg_connected"] = False
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=msg)
            return

        host = str(self.params.get("pg_host", "localhost")).strip() or "localhost"
        port = int(self.params.get("pg_port", 5432))
        user = str(self.params.get("pg_user", "postgres")).strip() or "postgres"
        password = (
            self._vault.get_secret("database.pg_password")
            if self._vault is not None else None
        ) or str(self.params.get("pg_password", ""))
        database = str(self.params.get("pg_database", "miniloader")).strip() or "miniloader"

        try:
            self._pool = await _asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                min_size=1,
                max_size=4,
                command_timeout=30,
            )
            async with self._pool.acquire() as conn:
                for stmt in _DDL_PG:
                    await conn.execute(stmt)
            self.params["pg_connected"] = True
            log.info(
                "database[%s]: connected to postgres %s:%s/%s",
                self.module_id, host, port, database,
            )
            self.status = ModuleStatus.RUNNING
        except Exception as exc:
            self.params["pg_connected"] = False
            log.error("database[%s]: postgres connect failed — %s", self.module_id, exc)
            self.status = ModuleStatus.ERROR
            await self.report_state(severity="ERROR", message=f"Postgres connect failed: {exc}")

    async def check_ready(self) -> bool:
        db_type = str(self.params.get("db_type", "sqlite")).strip().lower()
        if db_type == "postgres":
            return self._pool is not None and not self._pool.is_closing()
        db_path = self.params.get("db_filepath", "")
        return bool(db_path) and Path(db_path).exists()

    async def init(self) -> None:
        """Prepare DB path early; full schema setup occurs in initialize()."""
        db_type = str(self.params.get("db_type", "sqlite")).strip().lower()
        if db_type != "sqlite":
            return
        p = self._resolve_db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        self.params["db_filepath"] = str(p)
        log.debug("database[%s]: prepared sqlite path %s", self.module_id, p)

    async def shutdown(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self.params["pg_connected"] = False
        self.status = ModuleStatus.STOPPED

    # ── Request routing ──────────────────────────────────────────────────────

    async def process(self, payload: Payload, source_port: Port) -> None:
        if payload.signal_type == SignalType.DB_QUERY_PAYLOAD:
            await self._handle_query(payload)
        elif payload.signal_type == SignalType.DB_TRANSACTION_PAYLOAD:
            await self._handle_transaction(payload)
        elif payload.signal_type == SignalType.DB_RESPONSE_PAYLOAD:
            return  # ignore echoed responses on the bidirectional channel

    async def _handle_query(self, payload: Payload) -> None:
        db_type = str(self.params.get("db_type", "sqlite")).strip().lower()
        sql = str(payload.data.get("sql", "")).strip()
        request_id = str(payload.data.get("request_id", "")).strip()
        params = self._normalize_params(payload.data.get("params", []))

        if not sql:
            await self._emit_error(payload, "MISSING_SQL", "DB query is missing 'sql'")
            return

        sql_head = sql.lstrip().upper()

        if db_type == "postgres":
            if sql_head.startswith("PRAGMA"):
                await self._emit_error(
                    payload,
                    "PRAGMA_NOT_SUPPORTED",
                    "PRAGMA statements are not valid in PostgreSQL mode",
                )
                return
            if not (sql_head.startswith("SELECT") or sql_head.startswith("WITH")):
                await self._emit_error(
                    payload,
                    "INVALID_QUERY",
                    "DB_QUERY_PAYLOAD only accepts SELECT / WITH in postgres mode",
                )
                return
            await self._query_pg(request_id, sql, params)
        else:
            if not (
                sql_head.startswith("SELECT")
                or sql_head.startswith("WITH")
                or sql_head.startswith("PRAGMA")
            ):
                await self._emit_error(
                    payload,
                    "INVALID_QUERY",
                    "DB_QUERY_PAYLOAD only accepts SELECT / WITH / PRAGMA",
                )
                return
            await self._query_sqlite(request_id, sql, params, payload)

    async def _query_sqlite(
        self,
        request_id: str,
        sql: str,
        params: Any,
        orig_payload: Payload,
    ) -> None:
        if self._conn is None:
            await self._emit_error(orig_payload, "DB_UNAVAILABLE", "SQLite connection is not open")
            return
        try:
            self._conn.row_factory = async_sqlcipher.dict_row_factory
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()
            await cursor.close()
            row_dicts = [dict(r) for r in rows]
            await self._emit_response(
                request_id=request_id,
                success=True,
                operation="query",
                rowcount=len(row_dicts),
                rows=row_dicts,
                error=None,
            )
        except Exception as exc:
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="query",
                rowcount=0,
                rows=[],
                error={"code": "QUERY_FAILED", "message": str(exc)},
            )

    async def _query_pg(self, request_id: str, sql: str, params: Any) -> None:
        if self._pool is None:
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="query",
                rowcount=0,
                rows=[],
                error={"code": "DB_UNAVAILABLE", "message": "PostgreSQL pool is not connected"},
            )
            return
        try:
            pg_sql = _qmark_to_numbered(sql)
            pg_params = _to_pg_params(params)
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(pg_sql, *pg_params)
            row_dicts = [dict(r) for r in rows]
            await self._emit_response(
                request_id=request_id,
                success=True,
                operation="query",
                rowcount=len(row_dicts),
                rows=row_dicts,
                error=None,
            )
        except Exception as exc:
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="query",
                rowcount=0,
                rows=[],
                error={"code": "QUERY_FAILED", "message": str(exc)},
            )

    async def _handle_transaction(self, payload: Payload) -> None:
        access_mode = str(self.params.get("access_mode", "read_write")).strip().lower()
        if access_mode == "read_only":
            await self._emit_error(payload, "READ_ONLY", "Database is in read-only mode")
            return

        db_type = str(self.params.get("db_type", "sqlite")).strip().lower()
        request_id = str(payload.data.get("request_id", "")).strip()
        sql = str(payload.data.get("sql", "")).strip()
        params = self._normalize_params(payload.data.get("params", []))

        if not sql:
            await self._emit_error(payload, "MISSING_SQL", "DB transaction is missing 'sql'")
            return

        if db_type == "postgres":
            await self._transact_pg(request_id, sql, params)
        else:
            await self._transact_sqlite(request_id, sql, params, payload)

    async def _transact_sqlite(
        self,
        request_id: str,
        sql: str,
        params: Any,
        orig_payload: Payload,
    ) -> None:
        if self._conn is None:
            await self._emit_error(orig_payload, "DB_UNAVAILABLE", "SQLite connection is not open")
            return
        try:
            cursor = await self._conn.execute(sql, params)
            rowcount = int(cursor.rowcount or 0)
            await cursor.close()
            await self._conn.commit()
            await self._emit_response(
                request_id=request_id,
                success=True,
                operation="transaction",
                rowcount=rowcount,
                rows=[],
                error=None,
            )
        except Exception as exc:
            try:
                await self._conn.rollback()
            except Exception:
                pass
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="transaction",
                rowcount=0,
                rows=[],
                error={"code": "TRANSACTION_FAILED", "message": str(exc)},
            )

    async def _transact_pg(self, request_id: str, sql: str, params: Any) -> None:
        if self._pool is None:
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="transaction",
                rowcount=0,
                rows=[],
                error={"code": "DB_UNAVAILABLE", "message": "PostgreSQL pool is not connected"},
            )
            return
        try:
            pg_sql = _qmark_to_numbered(sql)
            pg_params = _to_pg_params(params)
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    status = await conn.execute(pg_sql, *pg_params)
            # Parse rowcount from status tag like "INSERT 0 3" or "UPDATE 2"
            rowcount = 0
            if status:
                last = status.split()[-1]
                if last.isdigit():
                    rowcount = int(last)
            await self._emit_response(
                request_id=request_id,
                success=True,
                operation="transaction",
                rowcount=rowcount,
                rows=[],
                error=None,
            )
        except Exception as exc:
            await self._emit_response(
                request_id=request_id,
                success=False,
                operation="transaction",
                rowcount=0,
                rows=[],
                error={"code": "TRANSACTION_FAILED", "message": str(exc)},
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_db_path(self) -> Path:
        raw = str(self.params.get("db_filepath", "miniloader_data.db")).strip()
        if not raw:
            raw = "miniloader_data.db"
            self.params["db_filepath"] = raw
        p = Path(raw)
        if not p.is_absolute():
            if self._vault is None:
                raise RuntimeError("database: vault is required to resolve sqlite path")
            p = self._vault.get_user_data_dir() / p
        return p

    def _normalize_params(self, params: Any) -> Any:
        if isinstance(params, dict):
            return params
        if isinstance(params, (list, tuple)):
            return tuple(params)
        if params is None:
            return ()
        return (params,)

    async def _emit_response(
        self,
        *,
        request_id: str,
        success: bool,
        operation: str,
        rowcount: int,
        rows: list[dict[str, Any]],
        error: dict[str, Any] | None,
    ) -> None:
        await self.inputs["DB_IN_OUT"].emit(
            Payload(
                signal_type=SignalType.DB_RESPONSE_PAYLOAD,
                source_module=self.module_id,
                data={
                    "request_id": request_id,
                    "success": success,
                    "operation": operation,
                    "rowcount": rowcount,
                    "rows": rows,
                    "error": error,
                },
            )
        )

    async def _emit_error(self, request_payload: Payload, code: str, message: str) -> None:
        await self._emit_response(
            request_id=str(request_payload.data.get("request_id", "")).strip(),
            success=False,
            operation=(
                "query"
                if request_payload.signal_type == SignalType.DB_QUERY_PAYLOAD
                else "transaction"
            ),
            rowcount=0,
            rows=[],
            error={"code": code, "message": message},
        )


# ── Module-level utilities ───────────────────────────────────────────────────

def _qmark_to_numbered(sql: str) -> str:
    """Replace SQLite ? placeholders with PostgreSQL $1, $2, … placeholders."""
    result: list[str] = []
    counter = 0
    for ch in sql:
        if ch == "?":
            counter += 1
            result.append(f"${counter}")
        else:
            result.append(ch)
    return "".join(result)


def _to_pg_params(params: Any) -> list[Any]:
    """Normalise params to a flat list suitable for asyncpg *args expansion."""
    if isinstance(params, dict):
        return list(params.values())
    if isinstance(params, (list, tuple)):
        return list(params)
    if params is None:
        return []
    return [params]


def register(hypervisor: Any) -> None:
    module = DatabaseModule()
    hypervisor.register_module(module)
