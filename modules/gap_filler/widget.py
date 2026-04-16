"""
gap_filler/widget.py — Gap-Filler Card Controls
=================================================
Builds a decorative blank-panel module card containing the
miniloader logo, copyright text, website URL, and two 50%-larger
spinning exhaust fans.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from ui.fan_item import SpinningFanItem
from ui.rack_items import _LCD_FONT


class GapFillerCardBuilder:
    """Builds the decorative gap-filler panel content."""

    CONTROLS_HEIGHT: float = 140.0

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller

    def clear(self) -> None:
        pass

    @property
    def tracked_module_ids(self) -> list[str]:
        return []

    def refresh_all(self) -> None:
        pass

    def build_controls(
        self,
        card: ModuleCardItem,
        module: BaseModule,
        width: float,
        top_y: float,
        P: float = 0.0,
    ) -> list[SpinningFanItem]:
        lx = 18 + P
        usable_w = width - 2 * (18 + P)
        cx = lx + usable_w / 2

        # ── Logo LCD bezel ────────────────────────────────────────
        lcd_w = usable_w
        lcd_h = 38.0
        lcd_x = lx
        lcd_y = top_y + 2

        grad_lcd = QLinearGradient(lcd_x, lcd_y, lcd_x, lcd_y + lcd_h)
        grad_lcd.setColorAt(0.0, QColor("#0a100a"))
        grad_lcd.setColorAt(0.15, QColor("#0d150d"))
        grad_lcd.setColorAt(0.85, QColor("#0a120a"))
        grad_lcd.setColorAt(1.0, QColor("#070c07"))
        bezel = QGraphicsRectItem(lcd_x, lcd_y, lcd_w, lcd_h, card)
        bezel.setBrush(QBrush(grad_lcd))
        bezel.setPen(QPen(QColor("#2a3a2a"), 1.0))
        bezel.setZValue(2)

        logo_path = Path(__file__).resolve().parent.parent.parent / "ui" / "assets" / "miniloader-logo.png"
        logo_pix = QPixmap(str(logo_path))
        if not logo_pix.isNull():
            logo_pix = logo_pix.scaledToHeight(
                16, Qt.TransformationMode.SmoothTransformation,
            )
            logo_item = QGraphicsPixmapItem(logo_pix, card)
            logo_item.setPos(
                lcd_x + (lcd_w - logo_pix.width()) / 2,
                lcd_y + 4,
            )
            logo_item.setZValue(3)
            glow = QGraphicsDropShadowEffect()
            glow.setColor(QColor("#30e848"))
            glow.setBlurRadius(18)
            glow.setOffset(0, 0)
            logo_item.setGraphicsEffect(glow)

        _crt = QFont("Consolas", 9, QFont.Weight.Bold)
        _crt.setStyleHint(QFont.StyleHint.Monospace)
        title = QGraphicsSimpleTextItem("Local Intelligence Rack", card)
        title.setFont(_crt)
        title.setBrush(QBrush(QColor("#30e848")))
        tw = title.boundingRect().width()
        title.setPos(lcd_x + (lcd_w - tw) / 2, lcd_y + 20)
        title.setZValue(3)

        # ── Copyright / info text ─────────────────────────────────
        _tiny = QFont("Consolas", 6)
        _tiny.setStyleHint(QFont.StyleHint.Monospace)
        _dim = QColor("#607060")

        info_y = lcd_y + lcd_h + 8

        def _center_text(y: float, text: str, color: QColor = _dim) -> None:
            it = QGraphicsSimpleTextItem(text, card)
            it.setFont(_tiny)
            it.setBrush(QBrush(color))
            it.setPos(cx - it.boundingRect().width() / 2, y)
            it.setZValue(3)

        _center_text(info_y,      "\u00a9 2026 Miniloader LLC")
        _center_text(info_y + 11, "miniloader.ai", QColor("#4a8a5a"))
        _center_text(info_y + 22, "Source-Available \u2022 v0.1.0")

        # ── Spinning fans — bottom, 2× scale ─────────────────────
        fan_scale = 1.7
        fan_y = top_y + self.CONTROLS_HEIGHT - 2
        fan_spacing = 110.0

        fan_l = SpinningFanItem(parent=card)
        fan_l.setScale(fan_scale)
        fan_l.setPos(cx - fan_spacing / 2, fan_y)
        fan_l.setZValue(3)

        fan_r = SpinningFanItem(parent=card)
        fan_r.setScale(fan_scale)
        fan_r.setPos(cx + fan_spacing / 2, fan_y)
        fan_r.setZValue(3)

        return [fan_l, fan_r]
