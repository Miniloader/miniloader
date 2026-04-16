"""
agent_preset_store.py - Encrypted per-user agent preset storage.
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

_DDL = """\
CREATE TABLE IF NOT EXISTS agent_presets (
    name        TEXT PRIMARY KEY,
    is_builtin  INTEGER NOT NULL DEFAULT 0,
    preset_json TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)
"""

_DEFAULT_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "name": "Precise",
        "description": "Deterministic answers with minimal creativity.",
        "system_prompt": "",
        "sampling": {"temperature": 0.1, "top_p": 0.9},
        "tool_config": {"max_rounds": 3, "timeout_secs": 30, "disabled_tools": []},
        "model_override": None,
    },
    {
        "name": "Balanced",
        "description": "General-purpose default preset.",
        "system_prompt": "",
        "sampling": {"temperature": 0.7},
        "tool_config": {"max_rounds": 3, "timeout_secs": 30, "disabled_tools": []},
        "model_override": None,
    },
    {
        "name": "Creative",
        "description": "More varied and exploratory responses.",
        "system_prompt": "Offer imaginative options before converging on an answer.",
        "sampling": {"temperature": 1.1, "top_p": 0.98},
        "tool_config": {"max_rounds": 3, "timeout_secs": 30, "disabled_tools": []},
        "model_override": None,
    },
    {
        "name": "Coder",
        "description": "Code-focused reasoning with stable output.",
        "system_prompt": (
            "You are a software engineering assistant. Prefer concise, testable, "
            "stepwise answers and structured output."
        ),
        "sampling": {"temperature": 0.2, "top_p": 0.92, "max_tokens": 4096},
        "tool_config": {"max_rounds": 5, "timeout_secs": 45, "disabled_tools": []},
        "model_override": None,
    },
    {
        "name": "Research Assistant",
        "description": "Methodical synthesis with tool-first investigation.",
        "system_prompt": (
            "You are a meticulous research assistant. Gather evidence, cite "
            "uncertainty, and provide clear action items."
        ),
        "sampling": {"temperature": 0.25, "top_p": 0.95, "max_tokens": 4096},
        "tool_config": {"max_rounds": 6, "timeout_secs": 60, "disabled_tools": []},
        "model_override": None,
    },
)


class AgentPresetStore:
    """Async CRUD over ``agent_presets`` in the encrypted user DB."""

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
        async with await self._connect() as db:
            await db.executescript(_DDL)
            await db.commit()

    async def list_presets(self) -> list[dict[str, Any]]:
        await self.ensure_table()
        async with await self._connect() as db:
            db.row_factory = (
                async_sqlcipher.dict_row_factory if async_sqlcipher.is_available() else None
            )
            if db.row_factory is None:
                import aiosqlite

                db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT name, is_builtin, preset_json, updated_at "
                "FROM agent_presets ORDER BY is_builtin DESC, name ASC"
            )
            rows = await cur.fetchall()
            await cur.close()

        result: list[dict[str, Any]] = []
        for row in rows:
            raw = row["preset_json"]
            payload: dict[str, Any] = {}
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}
            result.append(
                {
                    "name": row["name"],
                    "is_builtin": bool(row["is_builtin"]),
                    "description": str(payload.get("description", "")),
                    "updated_at": row["updated_at"],
                }
            )
        return result

    async def get_preset(self, name: str) -> dict[str, Any] | None:
        await self.ensure_table()
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT preset_json, is_builtin FROM agent_presets WHERE name = ?",
                (name,),
            )
            rows = await cur.fetchall()
            await cur.close()
        if not rows:
            return None
        try:
            payload = json.loads(rows[0][0])
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("name", name)
        payload["is_builtin"] = bool(rows[0][1])
        return payload

    async def save_preset(
        self,
        name: str,
        preset: dict[str, Any],
        *,
        is_builtin: bool = False,
    ) -> None:
        await self.ensure_table()
        payload = dict(preset)
        payload["name"] = name
        preset_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        async with await self._connect() as db:
            await db.execute(
                "INSERT INTO agent_presets (name, is_builtin, preset_json, sync_status) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "preset_json = excluded.preset_json, "
                "updated_at = datetime('now'), "
                "is_builtin = excluded.is_builtin, "
                "sync_status = 'pending_update'",
                (name, int(is_builtin), preset_json, "local_only"),
            )
            await db.commit()

    async def delete_preset(self, name: str) -> None:
        await self.ensure_table()
        async with await self._connect() as db:
            await db.execute("DELETE FROM agent_presets WHERE name = ?", (name,))
            await db.commit()

    async def seed_defaults(self) -> int:
        await self.ensure_table()
        existing = {row["name"] for row in await self.list_presets()}
        inserted = 0
        for preset in _DEFAULT_PRESETS:
            name = str(preset.get("name", "")).strip()
            if not name or name in existing:
                continue
            await self.save_preset(name, preset, is_builtin=True)
            inserted += 1
        if inserted:
            log.info("agent_preset_store: seeded %d default preset(s)", inserted)
        return inserted
