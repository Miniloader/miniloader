"""
async_sqlcipher.py — Async wrapper around sqlcipher3
=====================================================
``aiosqlite`` is hard-wired to stdlib ``sqlite3`` and cannot use a
different backend.  This module provides a thin async adapter that
runs ``sqlcipher3`` operations in a dedicated thread, exposing the
same surface the database module and gpt_terminal rely on.

Usage::

    conn = await connect("/path/to/db.sqlite", key_hex="ab01cd...")
    await conn.execute("SELECT 1")
    rows = await conn.fetchall()
    await conn.close()
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import sqlcipher3 as _sqlcipher  # type: ignore[import-untyped]
except ImportError:
    _sqlcipher = None


def is_available() -> bool:
    """True when sqlcipher3 is importable."""
    return _sqlcipher is not None


class AsyncSqlCipherCursor:
    """Async wrapper around a sqlcipher3 cursor.

    All operations are dispatched through the same single-thread executor
    used by the parent connection, ensuring SQLCipher's thread-affinity
    requirement is always satisfied.
    """

    def __init__(
        self,
        cursor: Any,
        executor: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._cursor = cursor
        self._executor = executor
        self._loop = loop

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        bound = functools.partial(fn, *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, bound)

    async def fetchall(self) -> list:
        return await self._run(self._cursor.fetchall)

    async def fetchone(self) -> Any:
        return await self._run(self._cursor.fetchone)

    async def close(self) -> None:
        await self._run(self._cursor.close)

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def lastrowid(self) -> Any:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class AsyncSqlCipherConnection:
    """Async wrapper around a sqlcipher3 ``Connection``."""

    def __init__(
        self,
        connection: Any,
        executor: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._conn = connection
        self._executor = executor
        self._loop = loop

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._conn.row_factory = value

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        bound = functools.partial(fn, *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, bound)

    async def execute(self, sql: str, params: Any = ()) -> AsyncSqlCipherCursor:
        raw = await self._run(self._conn.execute, sql, params)
        return AsyncSqlCipherCursor(raw, self._executor, self._loop)

    async def executescript(self, sql: str) -> None:
        await self._run(self._conn.executescript, sql)

    async def commit(self) -> None:
        await self._run(self._conn.commit)

    async def rollback(self) -> None:
        await self._run(self._conn.rollback)

    async def close(self) -> None:
        await self._run(self._conn.close)
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> AsyncSqlCipherConnection:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class _Row:
    """sqlite3.Row-compatible wrapper for sqlcipher3 cursors."""

    def __init__(self, cursor: Any, row: tuple) -> None:
        self._keys = [d[0] for d in cursor.description]
        self._values = row

    def keys(self) -> list[str]:
        return list(self._keys)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._values[self._keys.index(key)]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def dict_row_factory(cursor: Any, row: tuple) -> _Row:
    """Row factory compatible with ``dict(row)``."""
    return _Row(cursor, row)


async def connect(
    db_path: str | Path,
    key_hex: str,
) -> AsyncSqlCipherConnection:
    """Open an encrypted SQLCipher database and return an async connection.

    ``key_hex`` is the hex-encoded 32-byte key passed via ``PRAGMA key``.
    """
    if _sqlcipher is None:
        raise ImportError(
            "sqlcipher3 is not installed. Run: pip install sqlcipher3-binary"
        )

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlcipher")

    def _open() -> Any:
        conn = _sqlcipher.connect(str(db_path))
        conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
        conn.execute("PRAGMA cipher_compatibility = 4")
        return conn

    raw_conn = await loop.run_in_executor(executor, _open)
    return AsyncSqlCipherConnection(raw_conn, executor, loop)
