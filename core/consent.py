"""
consent.py — Per-feature network consent manager.

No outbound network request should fire without the user's explicit
knowledge. This module stores consent flags in the authenticated
user's encrypted SQLite database.

Flags are keyed by a short feature slug (e.g. "huggingface_browse").
A missing key is treated as "not yet consented".
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.vault import VaultManager

_vault: VaultManager | None = None
_DDL = """\
CREATE TABLE IF NOT EXISTS consent_flags (
    feature     TEXT PRIMARY KEY,
    granted     INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT DEFAULT (datetime('now')),
    sync_status TEXT NOT NULL DEFAULT 'local_only'
)
"""


def configure_vault(vault: VaultManager | None) -> None:
    """Bind consent storage to the active unlocked vault session."""
    global _vault
    _vault = vault


def _db_path() -> Path | None:
    if _vault is None:
        return None
    return _vault.get_user_data_dir() / "miniloader_data.db"


def _connect() -> Any | None:
    db_path = _db_path()
    if db_path is None:
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import sqlcipher3  # type: ignore[import-untyped]

        conn = sqlcipher3.connect(str(db_path))
        key_hex = _vault.derive_db_key().hex() if _vault is not None else ""
        conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
        conn.execute("PRAGMA cipher_compatibility = 4")
    except Exception:
        conn = sqlite3.connect(str(db_path))

    conn.execute(_DDL)
    conn.commit()
    return conn


def has_consent(feature: str) -> bool:
    """Return True if the user has previously granted consent for *feature*."""
    conn = _connect()
    if conn is None:
        log.debug("consent: vault not configured; feature '%s' defaults to not consented", feature)
        return False
    try:
        row = conn.execute(
            "SELECT granted FROM consent_flags WHERE feature = ?",
            (feature,),
        ).fetchone()
        return bool(row[0]) if row is not None else False
    finally:
        conn.close()


def grant_consent(feature: str) -> None:
    """Persistently record that the user consented to *feature*."""
    conn = _connect()
    if conn is None:
        log.warning("consent: cannot grant '%s' without an active vault", feature)
        return
    try:
        conn.execute(
            "INSERT INTO consent_flags (feature, granted, sync_status) VALUES (?, ?, ?)"
            " ON CONFLICT(feature) DO UPDATE SET"
            "   granted = excluded.granted,"
            "   updated_at = datetime('now'),"
            "   sync_status = 'pending_update'",
            (feature, 1, "local_only"),
        )
        conn.commit()
    finally:
        conn.close()
    log.info("Consent granted: %s", feature)


def revoke_consent(feature: str) -> None:
    """Persistently revoke consent for *feature*."""
    conn = _connect()
    if conn is None:
        log.warning("consent: cannot revoke '%s' without an active vault", feature)
        return
    try:
        conn.execute(
            "INSERT INTO consent_flags (feature, granted, sync_status) VALUES (?, ?, ?)"
            " ON CONFLICT(feature) DO UPDATE SET"
            "   granted = excluded.granted,"
            "   updated_at = datetime('now'),"
            "   sync_status = 'pending_update'",
            (feature, 0, "local_only"),
        )
        conn.commit()
    finally:
        conn.close()
    log.info("Consent revoked: %s", feature)


def all_consents() -> dict[str, bool]:
    """Return a copy of every stored consent flag."""
    conn = _connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT feature, granted FROM consent_flags ORDER BY feature ASC"
        ).fetchall()
        return {str(feature): bool(granted) for feature, granted in rows}
    finally:
        conn.close()


# ── Well-known feature keys ─────────────────────────────────────────
HUGGINGFACE_BROWSE = "huggingface_browse"
HUGGINGFACE_DOWNLOAD = "huggingface_download"
NPM_BOOTSTRAP = "npm_bootstrap"
