"""
gpt_terminal/widget.py — GPT Terminal Card Controls
=====================================================
Builds and manages the expanded UI controls on the GPT Terminal
module card: IP/port display, DB status LED, conversation stats,
and action buttons (reset, open, explore).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

_DB_CACHE_TTL = 2.0  # Seconds to cache SQLite queries (avoids blocking UI)

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from core.base_module import ModuleStatus
from modules.database.logic import _DDL_SQLITE
from ui.rack_items import CardButtonItem, _LCD_FONT


def _open_db_sync(db_path: Path, module: BaseModule) -> sqlite3.Connection:
    """Open a SQLite DB, using SQLCipher when the module has a vault."""
    if getattr(module, "has_vault", False) and module._vault is not None:
        try:
            import sqlcipher3  # type: ignore[import-untyped]
            key_hex = module._vault.derive_db_key().hex()
            conn = sqlcipher3.connect(str(db_path))
            conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
            conn.execute("PRAGMA cipher_compatibility = 4")
            return conn
        except ImportError:
            pass
    return sqlite3.connect(str(db_path))


class GptTerminalCardBuilder:
    """Builds and manages the GPT Terminal module's expanded card controls.

    Owns the IP/port display, DB connectivity LED, conversation stats
    (threads + data size), and action buttons.  Dims stat readouts when
    no database module is wired and running.
    """

    # Vertical space consumed by the controls zone.
    # Referenced by RackWindow for card height maths.
    CONTROLS_HEIGHT: float = 96.0

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._stat_items: dict[str, dict[str, object]] = {}
        self._db_indicators: dict[str, dict[str, object]] = {}
        self._db_tables_cache: dict[str, tuple[bool, float]] = {}

    # ── Lifecycle helpers ─────────────────────────────────────────

    def clear(self) -> None:
        """Reset internal state (called when the rack layout is rebuilt)."""
        self._stat_items.clear()
        self._db_indicators.clear()
        self._db_tables_cache.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._stat_items.keys())

    def refresh_all(self) -> None:
        """Refresh all tracked GPT terminal modules."""
        for module_id in list(self._stat_items):
            self._refresh_stats(module_id)

    # ── Card construction ─────────────────────────────────────────

    def build_controls(
        self,
        card: ModuleCardItem,
        module: BaseModule,
        width: float,
        top_y: float,
        P: float = 0.0,
    ) -> None:
        """Build GPT-terminal-specific controls onto *card*."""
        module_id = module.module_id
        lx = 18 + P
        usable_w = width - 2 * (18 + P)

        _sm = _LCD_FONT
        _btn_font = QFont("Consolas", 7)
        _btn_font.setStyleHint(QFont.StyleHint.Monospace)

        # ── Row 1: LED-screen style HOST / IP + inline port +/- ───
        # Half-width window and left-aligned text.
        screen_w = usable_w * 0.50
        screen_h = 22.0
        screen_x = lx
        screen_y = top_y + 1

        screen_bg = QGraphicsRectItem(screen_x, screen_y, screen_w, screen_h, card)
        grad = QLinearGradient(screen_x, screen_y, screen_x, screen_y + screen_h)
        grad.setColorAt(0.0,  QColor("#0a100a"))
        grad.setColorAt(0.15, QColor("#0d150d"))
        grad.setColorAt(0.85, QColor("#0a120a"))
        grad.setColorAt(1.0,  QColor("#070c07"))
        screen_bg.setBrush(QBrush(grad))
        screen_bg.setPen(QPen(QColor("#2a3a2a"), 1.2))
        screen_bg.setZValue(2)

        screen_font = QFont("Consolas", 10, QFont.Weight.Bold)
        screen_font.setStyleHint(QFont.StyleHint.Monospace)
        port = int(module.params.get("web_port", 3000))
        ip_value = QGraphicsSimpleTextItem(f"127.0.0.1:{port}", card)
        ip_value.setFont(screen_font)
        ip_value.setBrush(QBrush(QColor("#30e848")))
        ip_value.setPos(screen_x + 8, screen_y + 2)
        ip_value.setZValue(3)

        CardButtonItem(
            screen_x + screen_w + 4, screen_y + 4, 16, 12, "-",
            lambda: self._adjust_port(module_id, -1), card, label_font=_btn_font,
        )
        CardButtonItem(
            screen_x + screen_w + 22, screen_y + 4, 16, 12, "+",
            lambda: self._adjust_port(module_id, +1), card, label_font=_btn_font,
        )

        # ── Row 2: Database + RAG status LEDs ──────────────────────
        db_led = QGraphicsEllipseItem(lx, top_y + 32, 8, 8, card)
        db_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        db_led.setBrush(QBrush(QColor("#e24c4c")))

        db_phrase = QGraphicsSimpleTextItem("DATABASE OFFLINE", card)
        db_phrase.setFont(_sm)
        db_phrase.setBrush(QBrush(QColor("#e24c4c")))
        db_phrase.setPos(lx + 12, top_y + 30)

        rag_x = min(lx + 210, lx + max(120.0, usable_w - 96.0))
        rag_led = QGraphicsEllipseItem(rag_x, top_y + 32, 8, 8, card)
        rag_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        rag_led.setBrush(QBrush(QColor("#3a3f45")))

        rag_phrase = QGraphicsSimpleTextItem("RAG OFFLINE", card)
        rag_phrase.setFont(_sm)
        rag_phrase.setBrush(QBrush(QColor("#5c636e")))
        rag_phrase.setPos(rag_x + 12, top_y + 30)

        # ── Row 3: Action buttons ─────────────────────────────────
        btn_h = 16.0
        db_init_btn = CardButtonItem(
            lx,          top_y + 54, 52, btn_h, "DB INIT",
            lambda: self._init_db(module_id), card, label_font=_btn_font,
        )
        CardButtonItem(
            lx + 56,     top_y + 54, 36, btn_h, "OPEN",
            lambda: self._open_browser(module_id), card, label_font=_btn_font,
        )
        CardButtonItem(
            lx + 96,     top_y + 54, 54, btn_h, "EXPLORE",
            lambda: self._explore_folder(module_id), card, label_font=_btn_font,
        )

        self._stat_items[module_id] = {
            "ip_value":      ip_value,
            "db_init_btn":   db_init_btn,
        }
        self._db_indicators[module_id] = {
            "led":       db_led,
            "db_phrase": db_phrase,
            "rag_led":   rag_led,
            "rag_phrase": rag_phrase,
        }
        self._refresh_stats(module_id)

    # ── Refresh helpers ───────────────────────────────────────────

    def _connected_db_module(self, module_id: str) -> BaseModule | None:
        """Return a connected and enabled database module, if any."""
        for wire in self._controller.hypervisor.active_wires:
            src_mid = wire.source_port.owner_module_id
            tgt_mid = wire.target_port.owner_module_id
            src_name = wire.source_port.name
            tgt_name = wire.target_port.name

            if src_mid == module_id and src_name == "DB_IN_OUT":
                peer = self._controller.hypervisor.active_modules.get(tgt_mid)
            elif tgt_mid == module_id and tgt_name == "DB_IN_OUT":
                peer = self._controller.hypervisor.active_modules.get(src_mid)
            else:
                continue

            if peer is not None and peer.enabled and peer.MODULE_NAME == "database":
                return peer
        return None

    def _db_has_required_tables(self, db_module: BaseModule) -> bool:
        resolve_fn = getattr(db_module, "_resolve_db_path", None)
        if callable(resolve_fn):
            db_path = Path(resolve_fn())
        else:
            db_path_raw = str(db_module.params.get("db_filepath", "miniloader_data.db")).strip()
            db_path = Path(db_path_raw)
            if not db_path.is_absolute() and db_module._vault is not None:
                db_path = db_module._vault.get_user_data_dir() / db_path
            elif not db_path.is_absolute():
                db_path = Path.cwd() / db_path
        if not db_path.exists():
            return False
        key = f"{db_module.module_id}:{db_path}"
        now = time.monotonic()
        if key in self._db_tables_cache:
            val, ts = self._db_tables_cache[key]
            if now - ts < _DB_CACHE_TTL:
                return val
        try:
            conn = _open_db_sync(db_path, db_module)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('system_state', 'threads', 'messages', 'templates', 'settings', 'consent_flags')"
            )
            names = {str(row[0]) for row in cur.fetchall()}
            conn.close()
            result = names >= {
                "system_state",
                "threads",
                "messages",
                "templates",
                "settings",
                "consent_flags",
            }
            self._db_tables_cache[key] = (result, now)
            return result
        except Exception:
            return False

    def _connected_rag_module(self, module_id: str) -> BaseModule | None:
        """Return a connected and enabled rag_engine module, if any."""
        for wire in self._controller.hypervisor.active_wires:
            src_mid = wire.source_port.owner_module_id
            tgt_mid = wire.target_port.owner_module_id
            src_name = wire.source_port.name
            tgt_name = wire.target_port.name

            if src_mid == module_id and src_name == "CONTEXT_IN":
                peer = self._controller.hypervisor.active_modules.get(tgt_mid)
            elif tgt_mid == module_id and tgt_name == "CONTEXT_IN":
                peer = self._controller.hypervisor.active_modules.get(src_mid)
            else:
                continue

            if (
                peer is not None
                and peer.enabled
                and peer.MODULE_NAME == "rag_engine"
                and peer.status in (ModuleStatus.RUNNING, ModuleStatus.READY)
            ):
                return peer
        return None

    def _style_db_init_button(self, btn: CardButtonItem, enabled: bool) -> None:
        if enabled:
            btn.setBrush(QBrush(QColor("#2a2e36")))
            btn.setPen(QPen(QColor("#626a77"), 1.0))
            txt = QColor("#d3d8df")
        else:
            btn.setBrush(QBrush(QColor("#1b1d21")))
            btn.setPen(QPen(QColor("#3a3f47"), 1.0))
            txt = QColor("#5d646f")
        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setBrush(QBrush(txt))

    def _refresh_stats(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._stat_items.get(module_id)
        indicator = self._db_indicators.get(module_id)
        if module is None or items is None or indicator is None:
            return

        db_module = self._connected_db_module(module_id)
        db_online = db_module is not None

        led = indicator.get("led")
        db_phrase = indicator.get("db_phrase")
        rag_led = indicator.get("rag_led")
        rag_phrase = indicator.get("rag_phrase")

        if isinstance(led, QGraphicsEllipseItem):
            led.setBrush(QBrush(QColor("#39d353" if db_online else "#e24c4c")))
        if isinstance(db_phrase, QGraphicsSimpleTextItem):
            db_phrase.setText("DATABASE ONLINE" if db_online else "DATABASE OFFLINE")
            db_phrase.setBrush(QBrush(QColor("#39d353" if db_online else "#e24c4c")))

        ready = bool(db_module is not None and self._db_has_required_tables(db_module))

        rag_module = self._connected_rag_module(module_id)
        rag_enabled = rag_module is not None
        if isinstance(rag_led, QGraphicsEllipseItem):
            rag_led.setBrush(QBrush(QColor("#39d353" if rag_enabled else "#3a3f45")))
        if isinstance(rag_phrase, QGraphicsSimpleTextItem):
            rag_phrase.setText("RAG ENABLED" if rag_enabled else "RAG OFFLINE")
            # Offline must be grey (not red).
            rag_phrase.setBrush(QBrush(QColor("#39d353" if rag_enabled else "#5c636e")))

        module.params["db_ready"] = ready
        module.params["db_init_enabled"] = bool(db_online and not ready)

        btn = items.get("db_init_btn")
        if isinstance(btn, CardButtonItem):
            self._style_db_init_button(btn, bool(module.params["db_init_enabled"]))

        ip_item = items.get("ip_value")
        if isinstance(ip_item, QGraphicsSimpleTextItem):
            ip_item.setText(f"127.0.0.1:{int(module.params.get('web_port', 3000))}")

    # ── Port adjustment ───────────────────────────────────────────

    def _adjust_port(self, module_id: str, direction: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = int(module.params.get("web_port", 3000))
        module.params["web_port"] = max(1024, min(65535, cur + direction))
        self._refresh_stats(module_id)
        self._controller.hv_log.info(
            f"{module_id}: web_port={module.params['web_port']}"
        )

    # ── Action buttons ────────────────────────────────────────────

    def _init_db(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        if not bool(module.params.get("db_init_enabled", False)):
            self._controller.statusBar().showMessage("DB init not required", 2000)
            self._refresh_stats(module_id)
            return

        db_module = self._connected_db_module(module_id)
        if db_module is None:
            self._controller.statusBar().showMessage("No compatible database connected", 2500)
            self._refresh_stats(module_id)
            return

        resolve_fn = getattr(db_module, "_resolve_db_path", None)
        if callable(resolve_fn):
            db_path = Path(resolve_fn())
        else:
            db_path_raw = str(db_module.params.get("db_filepath", "miniloader_data.db")).strip()
            db_path = Path(db_path_raw)
            if not db_path.is_absolute() and db_module._vault is not None:
                db_path = db_module._vault.get_user_data_dir() / db_path
            elif not db_path.is_absolute():
                db_path = Path.cwd() / db_path

        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = _open_db_sync(db_path, db_module)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            cur = conn.cursor()
            cur.executescript(_DDL_SQLITE)
            conn.commit()
            conn.close()
            self._controller.statusBar().showMessage("Database initialized for GPT Terminal", 2500)
            self._controller.hv_log.info(f"{module_id}: initialized DB tables on connected database")
        except Exception as exc:
            self._controller.hv_log.error(f"{module_id}: DB init failed: {exc}")
            self._controller.statusBar().showMessage(f"DB init failed: {exc}", 5000)
        finally:
            self._refresh_stats(module_id)

    def _open_browser(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        port = module.params.get("web_port", 3000)
        url = f"http://127.0.0.1:{port}"
        webbrowser.open(url)
        self._controller.hv_log.info(f"{module_id}: opened {url} in browser")
        self._controller.statusBar().showMessage(f"Opened {url}", 2000)

    def _explore_folder(self, module_id: str) -> None:
        app_dir = Path(__file__).parent
        self._controller.hv_log.info(f"{module_id}: exploring {app_dir}")
        self._controller.statusBar().showMessage(f"Opening {app_dir}", 2000)
        if os.name == "nt":
            subprocess.Popen(["explorer", str(app_dir)])
        else:
            import platform
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(app_dir)])
            else:
                subprocess.Popen(["xdg-open", str(app_dir)])
