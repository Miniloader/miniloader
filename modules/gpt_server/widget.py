"""
gpt_server/widget.py — GPT Server Card Controls
=================================================
Builds and manages the expanded UI controls on the GPT Server
module card: LED-screen style IP:port display, activity LED,
port +/- dial, CORS policy readout, and API / SDK reference buttons.
"""

from __future__ import annotations

import asyncio
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QInputDialog,
    QLineEdit,
)

from core.base_module import ModuleStatus
from ui.api_tooltip import ApiTooltipWidget
from ui.rack_items import CardButtonItem, _LCD_FONT

_SDK_URL = "https://github.com/openai/openai-python"

_SCREEN_FONT = QFont("Consolas", 10, QFont.Weight.Bold)
_SCREEN_FONT.setStyleHint(QFont.StyleHint.Monospace)

_CORS_OPTIONS = ["*", "localhost", "none"]


class GptServerCardBuilder:
    """Builds and manages the GPT Server module's expanded card controls.

    Draws a miniature LED screen showing the listening address, an
    activity indicator, port +/- buttons, and a CORS policy readout.
    """

    CONTROLS_HEIGHT: float = 116.0

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._items: dict[str, dict] = {}
        self._pw_visible: dict[str, bool] = {}
        self._api_tooltip = ApiTooltipWidget(controller.view.viewport())

    # ── Lifecycle ─────────────────────────────────────────────────

    def clear(self) -> None:
        self._items.clear()
        self._pw_visible.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._items.keys())

    def refresh_all(self) -> None:
        for module_id in list(self._items):
            self._refresh(module_id)

    # ── Card construction ─────────────────────────────────────────

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
        btn_w = 16.0
        _sm = _LCD_FONT
        _btn_font = QFont("Consolas", 7)
        _btn_font.setStyleHint(QFont.StyleHint.Monospace)

        # ── Row 1: LED screen ─────────────────────────────────────
        screen_w = width - 2 * (18 + P) - 38
        screen_h = 22.0
        screen_x = lx
        screen_y = top_y + 2

        screen_bg = QGraphicsRectItem(screen_x, screen_y, screen_w, screen_h, card)
        grad = QLinearGradient(screen_x, screen_y, screen_x, screen_y + screen_h)
        grad.setColorAt(0.0,  QColor("#0a100a"))
        grad.setColorAt(0.15, QColor("#0d150d"))
        grad.setColorAt(0.85, QColor("#0a120a"))
        grad.setColorAt(1.0,  QColor("#070c07"))
        screen_bg.setBrush(QBrush(grad))
        screen_bg.setPen(QPen(QColor("#2a3a2a"), 1.2))
        screen_bg.setZValue(2)

        # Subtle inner bezel highlight
        bezel = QGraphicsRectItem(
            screen_x + 1, screen_y + 1, screen_w - 2, screen_h - 2, card,
        )
        bezel.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        bezel.setPen(QPen(QColor(40, 70, 40, 50), 0.5))
        bezel.setZValue(2)

        port = module.params.get("server_port", 5000)
        addr_text = QGraphicsSimpleTextItem(f"127.0.0.1:{port}", card)
        addr_text.setFont(_SCREEN_FONT)
        addr_text.setBrush(QBrush(QColor("#30e848")))
        text_rect = addr_text.boundingRect()
        addr_text.setPos(
            screen_x + (screen_w - text_rect.width()) / 2,
            screen_y + (screen_h - text_rect.height()) / 2,
        )
        addr_text.setZValue(3)

        CardButtonItem(
            screen_x + screen_w + 4, screen_y + 4, 16, 12, "-",
            lambda: self._adjust_port(module_id, -1), card, label_font=_btn_font,
        )
        CardButtonItem(
            screen_x + screen_w + 22, screen_y + 4, 16, 12, "+",
            lambda: self._adjust_port(module_id, +1), card, label_font=_btn_font,
        )

        # ── Row 2: CORS ───────────────────────────────────────────
        row3_y = top_y + 30
        cors_x = lx
        cors_lbl = QGraphicsSimpleTextItem("CORS", card)
        cors_lbl.setFont(_sm)
        cors_lbl.setBrush(QBrush(QColor("#8ea58e")))
        cors_lbl.setPos(cors_x, row3_y)

        cors_value = QGraphicsSimpleTextItem(
            str(module.params.get("cors_policy", "*")), card,
        )
        cors_value.setFont(_sm)
        cors_value.setBrush(QBrush(QColor("#d8d8d8")))
        cors_value.setPos(cors_x + 36, row3_y)

        CardButtonItem(
            cors_x + 36 + 68,              row3_y - 2, 24, btn_w, "\u21bb",
            lambda: self._cycle_cors(module_id), card, label_font=_btn_font,
        )

        # ── Row 3: Endpoint password controls ──────────────────────
        row4_y = top_y + 52

        auth_lbl = QGraphicsSimpleTextItem("AUTH", card)
        auth_lbl.setFont(_sm)
        auth_lbl.setBrush(QBrush(QColor("#8ea58e")))
        auth_lbl.setPos(lx, row4_y - 2)

        pw_value = QGraphicsSimpleTextItem("OPEN", card)
        pw_value.setFont(_sm)
        pw_value.setBrush(QBrush(QColor("#d8d8d8")))
        pw_value.setPos(lx + 36, row4_y - 2)

        CardButtonItem(
            width - P - 58.0, row4_y - 1, 30.0, 14.0, "SET",
            lambda: self._set_password(module_id), card, label_font=_btn_font,
        )
        CardButtonItem(
            width - P - 26.0, row4_y - 1, 24.0, 14.0, "EYE",
            lambda: self._toggle_password_visibility(module_id), card, label_font=_btn_font,
        )

        # ── Row 4: Activity LED + API / SDK buttons ───────────────
        row5_y = top_y + 72

        act_led = QGraphicsEllipseItem(lx, row5_y, 8, 8, card)
        act_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        act_led.setBrush(QBrush(QColor("#3a3f45")))

        act_status = QGraphicsSimpleTextItem("IDLE", card)
        act_status.setFont(_sm)
        act_status.setBrush(QBrush(QColor("#5c636e")))
        act_status.setPos(lx + 12, row5_y - 2)

        # API reference button — positioned near the right edge of the card
        _sbtn_h = 14.0
        sdk_x   = width - P - 26.0
        api_x   = sdk_x - 4.0 - 28.0
        CardButtonItem(
            api_x, row5_y - 1, 28.0, _sbtn_h, "API",
            lambda: self._show_api_tooltip(module_id), card, label_font=_btn_font,
        )
        CardButtonItem(
            sdk_x, row5_y - 1, 26.0, _sbtn_h, "SDK",
            lambda: self._open_sdk(module_id), card, label_font=_btn_font,
        )

        self._items[module_id] = {
            "addr_text":  addr_text,
            "cors_value": cors_value,
            "pw_value":   pw_value,
            "act_led":    act_led,
            "act_status": act_status,
        }
        self._pw_visible.setdefault(module_id, False)
        self._refresh(module_id)

    # ── Refresh ───────────────────────────────────────────────────

    def _refresh(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._items.get(module_id)
        if module is None or items is None:
            return

        port = module.params.get("server_port", 5000)
        items["addr_text"].setText(f"127.0.0.1:{port}")
        items["cors_value"].setText(str(module.params.get("cors_policy", "*")))
        items["pw_value"].setText(self._password_label(module_id))

        led: QGraphicsEllipseItem = items["act_led"]
        status: QGraphicsSimpleTextItem = items["act_status"]
        server_active = module.params.get("server_active", False)

        if module.status == ModuleStatus.READY and server_active:
            led.setBrush(QBrush(QColor("#43b4e5")))
            status.setText("ACTIVE")
            status.setBrush(QBrush(QColor("#a3d4f0")))
        elif module.status in (ModuleStatus.RUNNING, ModuleStatus.READY):
            led.setBrush(QBrush(QColor("#d68c1a")))
            status.setText("NO BRAIN" if not server_active else "STANDBY")
            status.setBrush(QBrush(QColor("#d68c1a")))
        else:
            led.setBrush(QBrush(QColor("#3a3f45")))
            status.setText("IDLE")
            status.setBrush(QBrush(QColor("#5c636e")))

    def _password_label(self, module_id: str) -> str:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return "LOCKED"
        pw = str(module.params.get("endpoint_password", "")).strip()
        if not pw and module._vault is not None:
            pw = module._vault.get_or_create_secret(
                module._vault.DEFAULT_GPT_SERVER_SECRET_KEY
            )
            module.params["endpoint_password"] = pw
        if not pw:
            return "OPEN"
        if self._pw_visible.get(module_id, False):
            return pw[:22]
        return "*" * min(len(pw), 12)

    # ── Controls ──────────────────────────────────────────────────

    def _adjust_port(self, module_id: str, direction: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = int(module.params.get("server_port", 5000))
        module.params["server_port"] = max(1024, min(65535, cur + direction))
        self._refresh(module_id)
        self._controller.hv_log.info(
            f"{module_id}: server_port={module.params['server_port']}"
        )

    def _show_api_tooltip(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        port = int(module.params.get("server_port", 5000)) if module else 5000
        viewport = self._controller.view.viewport()
        global_pos = QCursor.pos()
        local_pos  = viewport.mapFromGlobal(global_pos)
        self._api_tooltip.show_at(port, local_pos)

    def _open_sdk(self, module_id: str) -> None:
        webbrowser.open(_SDK_URL)
        self._controller.hv_log.info(f"{module_id}: opened openai SDK docs")
        self._controller.statusBar().showMessage("Opened openai SDK on GitHub", 2000)

    def _toggle_password_visibility(self, module_id: str) -> None:
        self._pw_visible[module_id] = not self._pw_visible.get(module_id, False)
        self._refresh(module_id)

    def _set_password(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = str(module.params.get("endpoint_password", "")).strip()
        if not current and module._vault is not None:
            current = module._vault.get_or_create_secret(
                module._vault.DEFAULT_GPT_SERVER_SECRET_KEY
            )
        text, ok = QInputDialog.getText(
            self._controller,
            "GPT Server - Endpoint Password",
            "Set bearer password (required for endpoint access):",
            QLineEdit.EchoMode.Password,
            current,
        )
        if not ok:
            return
        updated = text.strip()
        if not updated and module._vault is not None:
            self._controller.statusBar().showMessage(
                "Password cannot be blank while vault security is enabled",
                3000,
            )
            return
        module.params["endpoint_password"] = updated
        if module._vault is not None:
            module._vault.set_secret(
                module._vault.DEFAULT_GPT_SERVER_SECRET_KEY,
                updated,
            )
        # Push updated auth_required metadata for tunnel/consumers.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(module.broadcast_routing_config())
        except RuntimeError:
            pass
        self._controller.hv_log.info(f"{module_id}: endpoint password updated")
        self._controller.statusBar().showMessage("Endpoint password updated", 2200)
        self._refresh(module_id)

    def _cycle_cors(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = str(module.params.get("cors_policy", "*"))
        try:
            idx = _CORS_OPTIONS.index(cur)
        except ValueError:
            idx = -1
        nxt = _CORS_OPTIONS[(idx + 1) % len(_CORS_OPTIONS)]
        module.params["cors_policy"] = nxt
        self._refresh(module_id)
        self._controller.hv_log.info(f"{module_id}: cors_policy={nxt}")
