"""agent_engine/widget.py — Agent Engine card controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsSimpleTextItem

from ui.rack_items import _LCD_FONT


class AgentEngineCardBuilder:
    CONTROLS_HEIGHT: float = 120.0

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._items: dict[str, dict[str, object]] = {}

    def clear(self) -> None:
        self._items.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._items.keys())

    def refresh_all(self) -> None:
        for module_id in list(self._items):
            self._refresh(module_id)

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
        panel_w = width - 2 * (18 + P)

        # ── Status LED row ────────────────────────────────────────
        llm_led = QGraphicsEllipseItem(lx, top_y + 2, 8, 8, card)
        llm_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        llm_led.setBrush(QBrush(QColor("#3a3f45")))

        llm_lbl = QGraphicsSimpleTextItem("LLM", card)
        llm_lbl.setFont(_LCD_FONT)
        llm_lbl.setBrush(QBrush(QColor("#3a4a3a")))
        llm_lbl.setPos(lx + 12, top_y)

        # ── LCD panel ─────────────────────────────────────────────
        panel_y = top_y + 18
        panel_h = 92.0

        panel = QGraphicsRectItem(lx, panel_y, panel_w, panel_h, card)
        grad = QLinearGradient(lx, panel_y, lx, panel_y + panel_h)
        grad.setColorAt(0.0,  QColor("#0a100a"))
        grad.setColorAt(0.15, QColor("#0d150d"))
        grad.setColorAt(0.85, QColor("#0a120a"))
        grad.setColorAt(1.0,  QColor("#070c07"))
        panel.setBrush(QBrush(grad))
        panel.setPen(QPen(QColor("#2a3a2a"), 1.2))

        row_x = lx + 8
        r0 = panel_y + 5
        r1 = r0 + 18
        r2 = r1 + 18
        r3 = r2 + 18
        r4 = r3 + 18

        def _key(txt: str, y: float) -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_LCD_FONT)
            it.setBrush(QBrush(QColor("#8ea58e")))
            it.setPos(row_x, y)
            return it

        def _val(txt: str, y: float, color: str = "#30e848") -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(txt, card)
            it.setFont(_LCD_FONT)
            it.setBrush(QBrush(QColor(color)))
            it.setPos(row_x + 64, y)
            return it

        _key("TOOLS",   r0)
        tools_val   = _val("0",             r0)
        _key("MISSING", r1)
        links_val   = _val("API_IN",        r1, "#e2c14c")
        _key("ACTIVE",  r2)
        active_val  = _val("0",             r2)
        _key("PRESET",  r3)
        preset_val  = _val("default agent", r3)

        self._items[module_id] = {
            "llm_led":   llm_led,
            "llm_lbl":   llm_lbl,
            "tools_val": tools_val,
            "links_val": links_val,
            "active_val": active_val,
            "preset_val": preset_val,
        }
        self._refresh(module_id)

    def _refresh(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._items.get(module_id)
        if module is None or items is None:
            return

        llm_ready   = module.params.get("llm_status", "") == "ready"
        llm_linked  = module.params.get("llm_status", "") in {"ready", "linked"}
        tools_count = int(module.params.get("tools_count", 0) or 0)
        missing     = list(module.params.get("missing_links", []) or [])
        active      = int(module.params.get("active_requests", 0) or 0)
        preset      = str(module.params.get("active_preset", "")).strip()

        # LED: bright green = ready, amber = linked, off = disconnected
        if llm_ready:
            led_color, lbl_color = "#39d353", "#39d353"
        elif llm_linked:
            led_color, lbl_color = "#e2c14c", "#e2c14c"
        else:
            led_color, lbl_color = "#3a3f45", "#3a4a3a"

        self._set_led(items.get("llm_led"), led_color)
        self._set_text(items.get("llm_lbl"), "LLM", lbl_color)

        tools_color = "#30e848" if tools_count > 0 else "#5c636e"
        self._set_text(items.get("tools_val"), str(tools_count), tools_color)

        if missing:
            self._set_text(items.get("links_val"), ",".join(missing), "#e2c14c")
        else:
            self._set_text(items.get("links_val"), "none", "#30e848")

        active_color = "#e2c14c" if active > 0 else "#5c636e"
        self._set_text(items.get("active_val"), str(active), active_color)

        self._set_text(
            items.get("preset_val"),
            preset or "default agent",
            "#30e848",
        )

    @staticmethod
    def _set_led(item: object | None, color: str) -> None:
        if isinstance(item, QGraphicsEllipseItem):
            item.setBrush(QBrush(QColor(color)))

    @staticmethod
    def _set_text(item: object | None, text: str, color: str) -> None:
        if isinstance(item, QGraphicsSimpleTextItem):
            item.setText(text)
            item.setBrush(QBrush(QColor(color)))
