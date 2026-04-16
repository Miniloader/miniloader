"""
download_store.py — Managed download persistence.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import async_sqlcipher

if TYPE_CHECKING:
    from core.vault import VaultManager

_DDL = """\
CREATE TABLE IF NOT EXISTS downloads (
    id            TEXT PRIMARY KEY,
    repo_id       TEXT NOT NULL,
    filename      TEXT NOT NULL,
    variant       TEXT NOT NULL,
    size          TEXT,
    url           TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',
    kind          TEXT NOT NULL DEFAULT 'file',
    progress      REAL NOT NULL DEFAULT 0,
    local_path    TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT
)
"""


class DownloadStore:
    """Async download record store backed by the user's encrypted SQLite DB."""

    def __init__(self, vault: VaultManager) -> None:
        self._vault = vault
        self._table_ready = False
        self._table_lock = threading.Lock()

    def _db_path(self) -> Path:
        return self._vault.get_user_data_dir() / "miniloader_data.db"

    async def _connect(self) -> Any:
        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if async_sqlcipher.is_available():
            return await async_sqlcipher.connect(
                str(db_path),
                key_hex=self._vault.derive_db_key().hex(),
            )
        import aiosqlite

        return await aiosqlite.connect(str(db_path))

    async def ensure_table(self) -> None:
        with self._table_lock:
            if self._table_ready:
                return
        async with await self._connect() as db:
            await db.executescript(_DDL)
            cur = await db.execute("PRAGMA table_info(downloads)")
            cols = [str(row[1]) for row in await cur.fetchall()]
            await cur.close()
            if "kind" not in cols:
                await db.execute("ALTER TABLE downloads ADD COLUMN kind TEXT NOT NULL DEFAULT 'file'")
            await db.commit()
        with self._table_lock:
            self._table_ready = True

    async def insert(
        self,
        *,
        download_id: str,
        repo_id: str,
        filename: str,
        variant: str,
        size: str,
        url: str,
        status: str,
        kind: str,
        progress: float,
        local_path: str | None,
        error: str | None,
        created_at: str,
        completed_at: str | None,
    ) -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute(
                "INSERT OR REPLACE INTO downloads ("
                "  id, repo_id, filename, variant, size, url, status, kind, progress,"
                "  local_path, error, created_at, completed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    download_id,
                    repo_id,
                    filename,
                    variant,
                    size,
                    url,
                    status,
                    kind,
                    float(progress),
                    local_path,
                    error,
                    created_at,
                    completed_at,
                ),
            )
            await db.commit()

    async def update_status(
        self,
        download_id: str,
        *,
        status: str,
        error: str | None = None,
        completed_at: str | None = None,
        local_path: str | None = None,
    ) -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute(
                "UPDATE downloads "
                "SET status = ?, error = ?, completed_at = COALESCE(?, completed_at), "
                "    local_path = COALESCE(?, local_path) "
                "WHERE id = ?",
                (status, error, completed_at, local_path, download_id),
            )
            await db.commit()

    async def update_progress(
        self,
        download_id: str,
        *,
        status: str,
        progress: float,
        error: str | None = None,
    ) -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute(
                "UPDATE downloads SET status = ?, progress = ?, error = ? WHERE id = ?",
                (status, float(progress), error, download_id),
            )
            await db.commit()

    async def get(self, download_id: str) -> dict[str, Any] | None:
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT id, repo_id, filename, variant, size, url, status, kind, progress, "
                "       local_path, error, created_at, completed_at "
                "FROM downloads WHERE id = ?",
                (download_id,),
            )
            row = await cur.fetchone()
            await cur.close()
        if row is None:
            return None
        return {
            "id": row[0],
            "repo_id": row[1],
            "filename": row[2],
            "variant": row[3],
            "size": row[4] or "",
            "url": row[5],
            "status": row[6],
            "kind": row[7] or "file",
            "progress": float(row[8] or 0.0),
            "local_path": row[9],
            "error": row[10],
            "created_at": row[11],
            "completed_at": row[12],
        }

    async def get_all(self) -> list[dict[str, Any]]:
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT id, repo_id, filename, variant, size, url, status, kind, progress, "
                "       local_path, error, created_at, completed_at "
                "FROM downloads ORDER BY datetime(created_at) DESC",
            )
            rows = await cur.fetchall()
            await cur.close()
        return [
            {
                "id": row[0],
                "repo_id": row[1],
                "filename": row[2],
                "variant": row[3],
                "size": row[4] or "",
                "url": row[5],
                "status": row[6],
                "kind": row[7] or "file",
                "progress": float(row[8] or 0.0),
                "local_path": row[9],
                "error": row[10],
                "created_at": row[11],
                "completed_at": row[12],
            }
            for row in rows
        ]

    async def delete(self, download_id: str) -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute("DELETE FROM downloads WHERE id = ?", (download_id,))
            await db.commit()

    async def reset_stale_downloading(self) -> int:
        """Reset rows left in downloading state after an unclean shutdown."""
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "UPDATE downloads SET status = 'queued' WHERE status = 'downloading'",
            )
            changed = int(getattr(cur, "rowcount", 0) or 0)
            await cur.close()
            await db.commit()
        return changed
