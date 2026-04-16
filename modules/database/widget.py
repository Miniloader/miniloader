"""
database/widget.py — Database Card Controls
============================================
Dual-mode card builder.  In SQLite mode the existing file-browser and stats
panel are shown.  In PostgreSQL mode a set of connection-field rows is shown
with inline EDIT buttons that open QInputDialog prompts.

The DB TYPE toggle button switches between modes.  Any change takes effect
after the module is power-cycled (standard lifecycle behaviour).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QInputDialog,
    QLineEdit,
)

from modules.database.logic import _DDL_SQLITE
from ui.rack_items import CardButtonItem, _LCD_FONT

_DB_CACHE_TTL = 2.0


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


class DatabaseCardBuilder:
    """Builds and manages the Database module's expanded card controls."""

    SQLITE_CONTROLS_HEIGHT: float = 90.0
    POSTGRES_CONTROLS_HEIGHT: float = 128.0
    CONTROLS_HEIGHT: float = POSTGRES_CONTROLS_HEIGHT

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._items: dict[str, dict[str, object]] = {}
        self._count_cache: dict[str, tuple[int, float]] = {}

    def clear(self) -> None:
        self._items.clear()
        self._count_cache.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._items.keys())

    def get_controls_height(self, module_id: str | None = None) -> float:
        if module_id is None:
            return self.POSTGRES_CONTROLS_HEIGHT
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return self.POSTGRES_CONTROLS_HEIGHT
        db_type = str(module.params.get("db_type", "sqlite")).strip().lower()
        return self.POSTGRES_CONTROLS_HEIGHT if db_type == "postgres" else self.SQLITE_CONTROLS_HEIGHT

    def refresh_all(self) -> None:
        for module_id in list(self._items):
            self._refresh(module_id)

    # ── Card construction ────────────────────────────────────────────────────

    def build_controls(
        self,
        card: ModuleCardItem,
        module: BaseModule,
        width: float,
        top_y: float,
        P: float = 0.0,
    ) -> None:
        module_id = module.module_id
        lx = 18 + P
        rw = width - 2 * (18 + P)   # usable inner row width
        _sm = _LCD_FONT
        _btn = QFont("Consolas", 7)
        _btn.setStyleHint(QFont.StyleHint.Monospace)

        sqlite_items: list[QGraphicsItem] = []
        pg_items: list[QGraphicsItem] = []

        # ── Row 0: DB TYPE TOGGLE (full width, always visible) ────────────
        type_toggle = CardButtonItem(
            lx, top_y, rw, 18, "SQLITE",
            lambda: self._toggle_db_type(module_id),
            card, label_font=_btn,
        )

        # ── SQLITE GROUP (top_y + 22 …) ───────────────────────────────────

        def _sl(txt: str, x: float, y: float) -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_sm)
            it.setBrush(QBrush(QColor("#506050")))
            it.setPos(x, y)
            it.setZValue(3)
            sqlite_items.append(it)
            return it

        def _sv(txt: str, x: float, y: float) -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_sm)
            it.setBrush(QBrush(QColor("#8ea58e")))
            it.setPos(x, y)
            it.setZValue(3)
            sqlite_items.append(it)
            return it

        sq_y = top_y + 22  # start of sqlite group
        panel_h = 24.0

        # Stats panel bezel
        bezel = QGraphicsRectItem(lx - 2, sq_y - 2, rw + 4, panel_h + 4, card)
        bezel.setBrush(QBrush(QColor("#161a16")))
        bezel.setPen(QPen(QColor("#2a3a2a"), 0.8))
        bezel.setZValue(2)
        sqlite_items.append(bezel)

        grad = QLinearGradient(lx, sq_y, lx, sq_y + panel_h)
        grad.setColorAt(0.0, QColor("#0a100a"))
        grad.setColorAt(1.0, QColor("#070c07"))
        panel_bg = QGraphicsRectItem(lx, sq_y, rw, panel_h, card)
        panel_bg.setBrush(QBrush(grad))
        panel_bg.setPen(QPen(Qt.PenStyle.NoPen))
        panel_bg.setZValue(2)
        sqlite_items.append(panel_bg)

        _sl("TBLS", lx + 4, sq_y + 5)
        tables_val = _sv("0", lx + 30, sq_y + 5)
        _sl("SIZE", lx + rw // 2, sq_y + 5)
        size_val = _sv("0.00 MB", lx + rw // 2 + 28, sq_y + 5)

        # ── POSTGRES GROUP (top_y + 22 …) ─────────────────────────────────

        def _pl(txt: str, x: float, y: float) -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_sm)
            it.setBrush(QBrush(QColor("#4a5868")))
            it.setPos(x, y)
            it.setZValue(3)
            pg_items.append(it)
            return it

        def _pv(txt: str, x: float, y: float) -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_sm)
            it.setBrush(QBrush(QColor("#8eabc5")))
            it.setPos(x, y)
            it.setZValue(3)
            pg_items.append(it)
            return it

        edit_w = 32
        edit_x = lx + rw - edit_w  # right-align all EDIT buttons

        pg_y = top_y + 22

        # HOST row
        _pl("HOST", lx, pg_y + 2)
        host_val = _pv("localhost", lx + 30, pg_y + 2)
        host_edit = CardButtonItem(
            edit_x, pg_y, edit_w, 16, "EDIT",
            lambda: self._edit_field(module_id, "pg_host", "PostgreSQL Host"),
            card, label_font=_btn,
        )
        pg_items.append(host_edit)

        # PORT + DATABASE row
        pg_y2 = pg_y + 22
        _pl("PORT", lx, pg_y2 + 2)
        port_minus = CardButtonItem(
            lx + 30, pg_y2, 14, 14, "-",
            lambda: self._adjust_port(module_id, -1),
            card, label_font=_btn,
        )
        pg_items.append(port_minus)
        port_val = _pv("5432", lx + 47, pg_y2 + 2)
        port_plus = CardButtonItem(
            lx + 66, pg_y2, 14, 14, "+",
            lambda: self._adjust_port(module_id, +1),
            card, label_font=_btn,
        )
        pg_items.append(port_plus)

        mid_x = lx + rw // 2 - 10
        _pl("DB", mid_x, pg_y2 + 2)
        db_val = _pv("miniloader", mid_x + 16, pg_y2 + 2)
        db_edit = CardButtonItem(
            edit_x, pg_y2, edit_w, 14, "EDIT",
            lambda: self._edit_field(module_id, "pg_database", "Database Name"),
            card, label_font=_btn,
        )
        pg_items.append(db_edit)

        # USER row
        pg_y3 = pg_y + 44
        _pl("USER", lx, pg_y3 + 2)
        user_val = _pv("postgres", lx + 32, pg_y3 + 2)
        user_edit = CardButtonItem(
            edit_x, pg_y3, edit_w, 16, "EDIT",
            lambda: self._edit_field(module_id, "pg_user", "Username"),
            card, label_font=_btn,
        )
        pg_items.append(user_edit)

        # PASSWORD row
        pg_y4 = pg_y + 66
        _pl("PASS", lx, pg_y4 + 2)
        pass_val = _pv("NONE", lx + 32, pg_y4 + 2)
        pass_set = CardButtonItem(
            edit_x, pg_y4, edit_w, 16, "SET",
            lambda: self._set_password(module_id),
            card, label_font=_btn,
        )
        pg_items.append(pass_set)

        # ── ALWAYS-VISIBLE BOTTOM ROW ─────────────────────────────────────
        sqlite_rw_y = sq_y + panel_h + 6
        postgres_rw_y = top_y + 98
        bot_y = postgres_rw_y

        conn_led = QGraphicsEllipseItem(lx, bot_y + 4, 8, 8, card)
        conn_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        conn_led.setBrush(QBrush(QColor("#3a3f45")))
        conn_led.setZValue(3)

        rw_btn = CardButtonItem(
            lx + rw - 52, bot_y, 52, 16, "R/W",
            lambda: self._toggle_access_mode(module_id),
            card, label_font=_btn,
        )

        # ── Store refs ────────────────────────────────────────────────────
        self._items[module_id] = {
            "type_toggle": type_toggle,
            "sqlite_items": sqlite_items,
            "pg_items": pg_items,
            # SQLite individual refs
            "tables_val": tables_val,
            "size_val": size_val,
            # Postgres individual refs
            "host_val": host_val,
            "port_val": port_val,
            "db_val": db_val,
            "user_val": user_val,
            "pass_val": pass_val,
            # Always-visible
            "conn_led": conn_led,
            "rw_btn": rw_btn,
            "sqlite_rw_y": sqlite_rw_y,
            "postgres_rw_y": postgres_rw_y,
        }
        self._refresh(module_id)

    # ── Refresh ──────────────────────────────────────────────────────────────

    def _refresh(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._items.get(module_id)
        if module is None or items is None:
            return

        db_type = str(module.params.get("db_type", "sqlite")).strip().lower()
        is_sqlite = db_type != "postgres"

        # Show/hide groups
        for item in items.get("sqlite_items", []):  # type: ignore[union-attr]
            if isinstance(item, QGraphicsItem):
                item.setVisible(is_sqlite)
        for item in items.get("pg_items", []):  # type: ignore[union-attr]
            if isinstance(item, QGraphicsItem):
                item.setVisible(not is_sqlite)

        # Update type toggle label
        toggle = items.get("type_toggle")
        if isinstance(toggle, CardButtonItem):
            label = "SQLITE" if is_sqlite else "POSTGRES"
            for child in toggle.childItems():
                if isinstance(child, QGraphicsSimpleTextItem):
                    child.setText(label)
                    child.setBrush(
                        QBrush(QColor("#8ea58e" if is_sqlite else "#8eabc5"))
                    )

        access_mode = str(module.params.get("access_mode", "read_write"))
        if access_mode not in {"read_write", "read_only"}:
            access_mode = "read_write"
            module.params["access_mode"] = access_mode

        if is_sqlite:
            self._refresh_sqlite(module_id, module, items, access_mode)
        else:
            self._refresh_postgres(module_id, module, items, access_mode)

        # R/W button (always visible)
        rw_btn = items.get("rw_btn")
        if isinstance(rw_btn, CardButtonItem):
            sqlite_rw_y = float(items.get("sqlite_rw_y", rw_btn.pos().y()))
            postgres_rw_y = float(items.get("postgres_rw_y", rw_btn.pos().y()))
            rw_btn.setPos(rw_btn.pos().x(), sqlite_rw_y if is_sqlite else postgres_rw_y)
            self._style_mode_btn(rw_btn, access_mode)

        # Connection LED is only shown in Postgres mode.
        conn_led = items.get("conn_led")
        if isinstance(conn_led, QGraphicsEllipseItem):
            conn_led.setVisible(not is_sqlite)

    def _refresh_sqlite(
        self,
        module_id: str,
        module: Any,
        items: dict[str, object],
        access_mode: str,
    ) -> None:
        db_path = self._resolve_db_path(module)
        initialized = db_path.exists()
        tables = self._count_tables(db_path, module)
        size_mb = (db_path.stat().st_size / (1024 * 1024)) if initialized else 0.0

        tables_val = items.get("tables_val")
        if isinstance(tables_val, QGraphicsSimpleTextItem):
            tables_val.setText(str(tables))

        size_val = items.get("size_val")
        if isinstance(size_val, QGraphicsSimpleTextItem):
            size_val.setText(f"{size_mb:.2f} MB")

        # conn_led mirrors file-exists for sqlite
        conn_led = items.get("conn_led")
        if isinstance(conn_led, QGraphicsEllipseItem):
            conn_led.setBrush(QBrush(QColor("#39d353" if initialized else "#3a3f45")))

    def _refresh_postgres(
        self,
        module_id: str,
        module: Any,
        items: dict[str, object],
        access_mode: str,
    ) -> None:
        host = str(module.params.get("pg_host", "localhost"))
        port = int(module.params.get("pg_port", 5432))
        database = str(module.params.get("pg_database", "miniloader"))
        user = str(module.params.get("pg_user", "postgres"))
        has_password = bool(str(module.params.get("pg_password", "")).strip())
        connected = bool(module.params.get("pg_connected", False))

        def tv(key: str) -> QGraphicsSimpleTextItem | None:
            obj = items.get(key)
            return obj if isinstance(obj, QGraphicsSimpleTextItem) else None

        if (v := tv("host_val")) is not None:
            v.setText(_tail(host, 20))
            v.setToolTip(host)
        if (v := tv("port_val")) is not None:
            v.setText(str(port))
        if (v := tv("db_val")) is not None:
            v.setText(_tail(database, 10))
            v.setToolTip(database)
        if (v := tv("user_val")) is not None:
            v.setText(_tail(user, 18))
        if (v := tv("pass_val")) is not None:
            v.setText("••••••" if has_password else "NONE")
            v.setBrush(
                QBrush(QColor("#8eabc5" if has_password else "#5c636e"))
            )

        conn_led = items.get("conn_led")
        if isinstance(conn_led, QGraphicsEllipseItem):
            conn_led.setBrush(QBrush(QColor("#39d353" if connected else "#3a3f45")))

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_db_path(module: Any) -> Path:
        resolve_fn = getattr(module, "_resolve_db_path", None)
        if callable(resolve_fn):
            try:
                return Path(resolve_fn())
            except Exception:
                pass
        raw = str(module.params.get("db_filepath", "miniloader_data.db")).strip()
        p = Path(raw)
        if not p.is_absolute() and getattr(module, "_vault", None) is not None:
            p = module._vault.get_user_data_dir() / p
        elif not p.is_absolute():
            p = Path.cwd() / p
        return p

    def _count_tables(self, db_path: Path, module: Any = None) -> int:
        if not db_path.exists():
            return 0
        key = str(db_path)
        now = time.monotonic()
        if key in self._count_cache:
            val, ts = self._count_cache[key]
            if now - ts < _DB_CACHE_TTL:
                return val
        try:
            conn = _open_db_sync(db_path, module) if module is not None else sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            row = cur.fetchone()
            conn.close()
            count = int(row[0]) if row else 0
            self._count_cache[key] = (count, now)
            return count
        except Exception:
            return 0

    def _style_mode_btn(self, btn: CardButtonItem, access_mode: str) -> None:
        if access_mode == "read_write":
            btn.setBrush(QBrush(QColor("#1a2e1a")))
            btn.setPen(QPen(QColor("#39d353"), 1.0))
            label, color = "R/W", QColor("#b6d9b6")
        else:
            btn.setBrush(QBrush(QColor("#2e2a1a")))
            btn.setPen(QPen(QColor("#e8a838"), 1.0))
            label, color = "READ", QColor("#e8c878")
        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setText(label)
                child.setBrush(QBrush(color))

    # ── Button handlers ──────────────────────────────────────────────────────

    def _toggle_db_type(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = str(module.params.get("db_type", "sqlite")).strip().lower()
        module.params["db_type"] = "postgres" if cur == "sqlite" else "sqlite"
        self._refresh(module_id)
        self._controller._rebuild_layout()
        self._controller.hv_log.info(
            f"{module_id}: db_type={module.params['db_type']} — power-cycle to reconnect"
        )

    def _toggle_access_mode(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = str(module.params.get("access_mode", "read_write"))
        module.params["access_mode"] = "read_only" if cur == "read_write" else "read_write"
        self._refresh(module_id)
        mode_text = "READ" if module.params["access_mode"] == "read_only" else "READ/WRITE"
        self._controller.hv_log.info(f"{module_id}: access_mode={mode_text}")

    def _edit_field(self, module_id: str, param: str, label: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = str(module.params.get(param, ""))
        text, ok = QInputDialog.getText(
            self._controller,
            f"Database — {label}",
            f"{label}:",
            QLineEdit.EchoMode.Normal,
            current,
        )
        if ok:
            module.params[param] = text.strip()
            self._refresh(module_id)
            self._controller.hv_log.info(f"{module_id}: {param} updated")

    def _set_password(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        text, ok = QInputDialog.getText(
            self._controller,
            "Database — PostgreSQL Password",
            "Password:",
            QLineEdit.EchoMode.Password,
            "",  # never pre-fill passwords
        )
        if ok:
            module.params["pg_password"] = text
            if module._vault is not None:
                module._vault.set_secret("database.pg_password", text)
            self._refresh(module_id)
            self._controller.hv_log.info(f"{module_id}: pg_password updated")

    def _adjust_port(self, module_id: str, direction: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = int(module.params.get("pg_port", 5432))
        module.params["pg_port"] = max(1, min(65535, cur + direction))
        self._refresh(module_id)

    def _initialize_db(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        db_path = self._resolve_db_path(module)
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = _open_db_sync(db_path, module)
            cur = conn.cursor()
            cur.executescript(_DDL_SQLITE)
            conn.commit()
            conn.close()
            self._controller.hv_log.info(f"{module_id}: initialized database at {db_path}")
            self._controller.statusBar().showMessage("Database initialized", 2500)
        except Exception as exc:
            self._controller.hv_log.error(f"{module_id}: initialize failed: {exc}")
            self._controller.statusBar().showMessage(f"Database initialize failed: {exc}", 5000)
        finally:
            self._refresh(module_id)

    def _browse_db_location(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        db_path = self._resolve_db_path(module)
        path, _ = QFileDialog.getSaveFileName(
            self._controller,
            "Select database file",
            str(db_path),
            "SQLite databases (*.db);;All files (*)",
        )
        if path:
            module.params["db_filepath"] = path
            self._refresh(module_id)
            self._controller.hv_log.info(f"{module_id}: database location set to {path}")


# ── Utility ──────────────────────────────────────────────────────────────────

def _tail(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return "…" + s[-(max_len - 1):]
