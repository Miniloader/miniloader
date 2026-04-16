"""
template_store.py — Encrypted per-user template storage
=========================================================
All user rack templates (both shipped defaults and user-created ones)
are stored in the user's encrypted SQLite database.

Templates are ``RackSnapshot`` JSON blobs stored in a ``templates``
table alongside threads and messages.  The DB is encrypted via
SQLCipher, so template contents -- including module topology and
``__vault__`` sentinel markers -- are protected at rest.

Usage::

    store = TemplateStore(vault)
    await store.ensure_table()
    await store.seed_defaults()          # idempotent; skips existing
    rows = await store.list_templates()  # [(name, is_default, updated_at), ...]
    snapshot = await store.get_template("blank_rack")
    await store.save_template("my_config", snapshot)
    await store.delete_template("my_config")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import async_sqlcipher

if TYPE_CHECKING:
    from core.vault import VaultManager

log = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent.parent / "templates"

_DDL = """\
CREATE TABLE IF NOT EXISTS templates (
    name          TEXT PRIMARY KEY,
    is_default    INTEGER NOT NULL DEFAULT 0,
    snapshot_json TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now')),
    updated_at    TEXT DEFAULT (datetime('now')),
    sync_status   TEXT NOT NULL DEFAULT 'local_only'
)
"""


class TemplateStore:
    """Async CRUD over the ``templates`` table in the user's encrypted DB."""

    def __init__(self, vault: VaultManager) -> None:
        self._vault = vault

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
        """Create the templates table if it does not yet exist."""
        async with await self._connect() as db:
            await db.executescript(_DDL)
            await db.commit()

    async def list_templates(self) -> list[dict[str, Any]]:
        """Return all templates as list of dicts with name/is_default/updated_at."""
        await self.ensure_table()
        async with await self._connect() as db:
            db.row_factory = async_sqlcipher.dict_row_factory if async_sqlcipher.is_available() else None
            if db.row_factory is None:
                import aiosqlite
                db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT name, is_default, updated_at "
                "FROM templates ORDER BY is_default DESC, name ASC"
            )
            rows = await cur.fetchall()
            await cur.close()
        return [
            {
                "name": row["name"],
                "is_default": bool(row["is_default"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def get_template(self, name: str) -> dict[str, Any] | None:
        """Return the raw snapshot dict for *name*, or None if not found."""
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT snapshot_json FROM templates WHERE name = ?", (name,)
            )
            row = await cur.fetchall()
            await cur.close()
        if not row:
            return None
        try:
            return json.loads(row[0][0])
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    async def save_template(
        self,
        name: str,
        snapshot: dict[str, Any],
        is_default: bool = False,
    ) -> None:
        """Upsert a template.  Existing templates keep their ``is_default`` flag."""
        await self.ensure_table()
        snapshot_json = json.dumps(snapshot, ensure_ascii=True, separators=(",", ":"))
        async with await self._connect() as db:
            await db.execute(
                "INSERT INTO templates (name, is_default, snapshot_json, sync_status) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET"
                "   snapshot_json = excluded.snapshot_json,"
                "   updated_at = datetime('now'),"
                "   sync_status = 'pending_update'",
                (name, int(is_default), snapshot_json, "local_only"),
            )
            await db.commit()

    async def delete_template(self, name: str) -> None:
        """Delete a template by name."""
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute("DELETE FROM templates WHERE name = ?", (name,))
            await db.commit()

    async def seed_defaults(self) -> int:
        """
        Seed shipped default templates from ``templates/*.json`` into the DB.

        Idempotent -- skips any template whose name already exists in the DB.
        Returns the number of templates actually inserted.
        """
        await self.ensure_table()
        if not _DEFAULTS_DIR.exists():
            return 0

        inserted = 0
        existing = {row["name"] for row in await self.list_templates()}

        for path in sorted(_DEFAULTS_DIR.glob("*.json")):
            name = path.stem
            if name in existing:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("template_store: skipping %s — %s", path.name, exc)
                continue
            await self.save_template(name, payload, is_default=True)
            inserted += 1
            log.info("template_store: seeded default template '%s'", name)

        return inserted
