"""
settings_store.py — Per-machine settings in the encrypted database
===================================================================
All user and machine-specific settings are stored in a ``settings``
table inside the user's encrypted SQLite database.  This replaces
the previous ``vault.profile["app_settings"]`` approach.

Settings are key-value pairs scoped by an optional ``machine_id``.
Global settings (shared across all machines for this account) use
``machine_id = ''``.  Machine-specific settings use the hash
derived from ``hostname|gpu_name``.

Usage::

    store = SettingsStore(vault)
    mid = SettingsStore.get_machine_id()

    await store.set("hf_token", "hf_abc123")                      # global
    await store.set("selected_backend", "cuda", machine_id=mid)    # per-machine
    val = await store.get("selected_backend", machine_id=mid)
    machine = await store.get_machine_settings(mid)
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import async_sqlcipher

if TYPE_CHECKING:
    from core.vault import VaultManager

log = logging.getLogger(__name__)

_DDL = """\
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT NOT NULL,
    machine_id TEXT NOT NULL DEFAULT '',
    value      TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only',
    PRIMARY KEY (key, machine_id)
)
"""


class SettingsStore:
    """Async key-value store backed by the user's encrypted SQLite DB."""

    def __init__(self, vault: VaultManager) -> None:
        self._vault = vault

    def _db_path(self) -> Path:
        return self._vault.get_user_data_dir() / "miniloader_data.db"

    async def _connect(self) -> Any:
        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not async_sqlcipher.is_available():
            raise RuntimeError(
                "settings_store: sqlcipher3 is required. "
                "Run: pip install sqlcipher3-binary"
            )
        return await async_sqlcipher.connect(
            str(db_path),
            key_hex=self._vault.derive_db_key().hex(),
        )

    async def ensure_table(self) -> None:
        async with await self._connect() as db:
            await db.executescript(_DDL)
            await db.commit()

    # ── Scalar get/set ───────────────────────────────────────────

    async def get(self, key: str, machine_id: str = "") -> str | None:
        """Return a setting value, or None if not found."""
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT value FROM settings WHERE key = ? AND machine_id = ?",
                (key, machine_id),
            )
            row = await cur.fetchall()
            await cur.close()
        if not row:
            return None
        return row[0][0]

    async def get_json(self, key: str, machine_id: str = "") -> Any:
        """Return a setting parsed as JSON, or None."""
        raw = await self.get(key, machine_id)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(self, key: str, value: Any, machine_id: str = "") -> None:
        """Upsert a setting.  Non-string values are JSON-encoded."""
        await self.ensure_table()
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        async with await self._connect() as db:
            await db.execute(
                "INSERT INTO settings (key, machine_id, value, sync_status) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(key, machine_id) DO UPDATE SET"
                "   value = excluded.value,"
                "   updated_at = datetime('now'),"
                "   sync_status = 'pending_update'",
                (key, machine_id, value, "local_only"),
            )
            await db.commit()

    async def delete(self, key: str, machine_id: str = "") -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute(
                "DELETE FROM settings WHERE key = ? AND machine_id = ?",
                (key, machine_id),
            )
            await db.commit()

    # ── Bulk machine settings ────────────────────────────────────

    async def get_machine_settings(self, machine_id: str) -> dict[str, Any]:
        """Return all settings for a specific machine as a dict."""
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT key, value FROM settings WHERE machine_id = ?",
                (machine_id,),
            )
            rows = await cur.fetchall()
            await cur.close()
        result: dict[str, Any] = {}
        for row in rows:
            k, v = row[0], row[1]
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                result[k] = v
        return result

    async def save_machine_settings(
        self, machine_id: str, settings: dict[str, Any],
    ) -> None:
        """Upsert all key-value pairs for a machine in one transaction."""
        await self.ensure_table()
        async with await self._connect() as db:
            for key, value in settings.items():
                encoded = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=True, separators=(",", ":"),
                )
                await db.execute(
                    "INSERT INTO settings (key, machine_id, value, sync_status) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(key, machine_id) DO UPDATE SET"
                    "   value = excluded.value,"
                    "   updated_at = datetime('now'),"
                    "   sync_status = 'pending_update'",
                    (key, machine_id, encoded, "local_only"),
                )
            await db.commit()

    async def machine_exists(self, machine_id: str) -> bool:
        """True if any settings exist for this machine_id."""
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT 1 FROM settings WHERE machine_id = ? LIMIT 1",
                (machine_id,),
            )
            row = await cur.fetchall()
            await cur.close()
        return bool(row)

    # ── Machine identity ─────────────────────────────────────────

    @staticmethod
    def get_machine_id() -> str:
        """Derive a per-PC key from hostname + detected GPU.

        Changes when the GPU is swapped (intentional -- different GPU
        means different backend capabilities).
        """
        try:
            from core.hardware_probe import get_hardware_snapshot
            gpu = get_hardware_snapshot().gpu_name or ""
        except Exception:
            gpu = ""
        raw = f"{socket.gethostname()}|{gpu}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
