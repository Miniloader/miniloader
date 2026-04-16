"""
entitlement_store.py - Local entitlement persistence for portal bridge.
"""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EntitlementStore:
    """Thread-safe JSON-backed storage for module entitlements."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = Path(store_path)
        self._lock = threading.Lock()

    def get_all(self) -> list[dict[str, str]]:
        with self._lock:
            data = self._read_locked()
        return [
            {
                "itemId": str(item.get("itemId") or ""),
                "licenseKey": str(item.get("licenseKey") or ""),
            }
            for item in data
            if str(item.get("itemId") or "").strip() and str(item.get("licenseKey") or "").strip()
        ]

    def upsert(self, items: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not isinstance(items, list):
            return self.get_all()

        normalized_incoming: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("itemId") or "").strip()
            license_key = str(item.get("licenseKey") or "").strip()
            if not item_id or not license_key:
                continue
            normalized_incoming.append(
                {
                    "itemId": item_id,
                    "licenseKey": license_key,
                    "acquiredAt": str(item.get("acquiredAt") or "").strip() or _now_iso(),
                }
            )

        with self._lock:
            existing = self._read_locked()
            by_item_id: dict[str, dict[str, str]] = {}
            ordered_ids: list[str] = []

            for item in existing:
                item_id = str(item.get("itemId") or "").strip()
                license_key = str(item.get("licenseKey") or "").strip()
                if not item_id or not license_key:
                    continue
                if item_id not in by_item_id:
                    ordered_ids.append(item_id)
                by_item_id[item_id] = {
                    "itemId": item_id,
                    "licenseKey": license_key,
                    "acquiredAt": str(item.get("acquiredAt") or "").strip() or _now_iso(),
                }

            for incoming in normalized_incoming:
                item_id = incoming["itemId"]
                if item_id not in by_item_id:
                    ordered_ids.append(item_id)
                by_item_id[item_id] = incoming

            merged = [by_item_id[item_id] for item_id in ordered_ids if item_id in by_item_id]
            self._write_locked(merged)

        return [
            {"itemId": item["itemId"], "licenseKey": item["licenseKey"]}
            for item in merged
        ]

    def assign_item_ids(self, item_ids: list[str]) -> list[dict[str, str]]:
        if not isinstance(item_ids, list):
            return []

        normalized_ids: list[str] = []
        seen: set[str] = set()
        for raw in item_ids:
            item_id = str(raw or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            normalized_ids.append(item_id)
        if not normalized_ids:
            return []

        with self._lock:
            existing = self._read_locked()
            by_item_id: dict[str, dict[str, str]] = {}
            ordered_ids: list[str] = []

            for item in existing:
                item_id = str(item.get("itemId") or "").strip()
                license_key = str(item.get("licenseKey") or "").strip()
                if not item_id or not license_key:
                    continue
                if item_id not in by_item_id:
                    ordered_ids.append(item_id)
                by_item_id[item_id] = {
                    "itemId": item_id,
                    "licenseKey": license_key,
                    "acquiredAt": str(item.get("acquiredAt") or "").strip() or _now_iso(),
                }

            for item_id in normalized_ids:
                if item_id not in by_item_id:
                    by_item_id[item_id] = {
                        "itemId": item_id,
                        "licenseKey": self._make_license_key(item_id),
                        "acquiredAt": _now_iso(),
                    }
                    ordered_ids.append(item_id)

            merged = [by_item_id[item_id] for item_id in ordered_ids if item_id in by_item_id]
            self._write_locked(merged)

        return [
            {"itemId": item_id, "licenseKey": by_item_id[item_id]["licenseKey"]}
            for item_id in normalized_ids
            if item_id in by_item_id
        ]

    def _read_locked(self) -> list[dict[str, Any]]:
        if not self._store_path.exists():
            return []
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [row for row in raw if isinstance(row, dict)]

    def _write_locked(self, data: list[dict[str, str]]) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._store_path)

    @staticmethod
    def _make_license_key(item_id: str) -> str:
        prefix = "".join(ch for ch in item_id.upper() if ch.isalnum())[:6] or "MODKEY"
        suffix = secrets.token_hex(6).upper()
        return f"{prefix}-{suffix}"
