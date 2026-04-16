"""
hypervisor_panel.py — Top-of-Rack Management Panel
====================================================
HypervisorLog (ring-buffer event log) and HypervisorPanelItem
(the green-phosphor display at the top of the rack chassis).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hardware_probe import HardwareSnapshot
    from core.hypervisor import Hypervisor

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
)

from core.base_module import ModuleStatus
from core.hardware_probe import AiBackend
from core.probe_service import get_probe_service
from ui.rack_items import (
    _LCD_FONT,
    _LOG_AMBER,
    _LOG_CYAN,
    _LOG_DIM,
    _LOG_GREEN,
    _LOG_RED,
    _MONO_FONT,
    _TITLE_FONT,
    _draw_screw,
    _status_color,
)


# ── Hypervisor event log ─────────────────────────────────────────

class HypervisorLog:
    """Ring buffer that stores timestamped log lines for the UI."""

    def __init__(self, max_lines: int = 200) -> None:
        self._lines: deque[tuple[str, str, QColor]] = deque(maxlen=max_lines)
        self._boot_time = time.monotonic()

    def _ts(self) -> str:
        elapsed = time.monotonic() - self._boot_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def info(self, msg: str) -> None:
        self._lines.append((self._ts(), msg, _LOG_GREEN))

    def warn(self, msg: str) -> None:
        self._lines.append((self._ts(), msg, _LOG_AMBER))

    def error(self, msg: str) -> None:
        self._lines.append((self._ts(), msg, _LOG_RED))

    def worker(self, msg: str) -> None:
        self._lines.append((self._ts(), msg, _LOG_CYAN))

    @property
    def lines(self) -> list[tuple[str, str, QColor]]:
        return list(self._lines)


# ── Hypervisor panel (top-of-rack switch) ────────────────────────

class HypervisorPanelItem(QGraphicsRectItem):
    """
    The top-of-rack management unit.  Contains a title LCD, a
    power LED, per-module status LEDs, live stats, and a scrolling
    green-phosphor log screen.
    """

    PANEL_HEIGHT = 270.0
    LOG_VISIBLE_LINES = 8
    LOG_WRAP_LOOKBACK = 32
    LOG_ENTRY_GAP = 2.0

    def __init__(
        self,
        width: float,
        hypervisor: Hypervisor,
        hv_log: HypervisorLog,
    ) -> None:
        super().__init__(0, 0, width, self.PANEL_HEIGHT)
        self.hypervisor = hypervisor
        self.hv_log = hv_log
        self._width = width

        grad = QLinearGradient(0, 0, 0, self.PANEL_HEIGHT)
        grad.setColorAt(0.0, QColor("#2a2d34"))
        grad.setColorAt(0.15, QColor("#1e2028"))
        grad.setColorAt(1.0, QColor("#191b20"))
        self.setBrush(QBrush(grad))
        self.setPen(QPen(QColor("#5c636e"), 1.5))
        self.setZValue(0)

        self._power_transition_task: asyncio.Task | None = None
        self._power_transition_phase: str = ""
        self._init_task: asyncio.Task | None = None
        self._auto_init_enabled: bool = True
        self._build_faceplate()
        self._hardware: HardwareSnapshot | None = None
        self._active_tab: str = "LOG"
        # Scroll log history: skip N newest deque entries (wheel up = larger).
        self._log_skip_tail: int = 0

    def _build_faceplate(self) -> None:
        w = self._width

        for cx, cy in [(12, 12), (w - 12, 12), (12, self.PANEL_HEIGHT - 12), (w - 12, self.PANEL_HEIGHT - 12)]:
            _draw_screw(None, cx, cy, parent=self)

        # Title LCD bezel — logo + HYPERVISOR inside one CRT screen
        # lcd_x offset to leave room for the power toggle on its left
        lcd_x, lcd_y, lcd_w, lcd_h = 74.0, 8.0, 220.0, 44.0
        grad_lcd = QLinearGradient(lcd_x, lcd_y, lcd_x, lcd_y + lcd_h)
        grad_lcd.setColorAt(0.0,  QColor("#0a100a"))
        grad_lcd.setColorAt(0.15, QColor("#0d150d"))
        grad_lcd.setColorAt(0.85, QColor("#0a120a"))
        grad_lcd.setColorAt(1.0,  QColor("#070c07"))
        bezel = QGraphicsRectItem(lcd_x, lcd_y, lcd_w, lcd_h, self)
        bezel.setBrush(QBrush(grad_lcd))
        bezel.setPen(QPen(QColor("#2a3a2a"), 1.0))
        bezel.setZValue(2)

        _logo_path = Path(__file__).parent / "assets" / "miniloader-logo.png"
        _logo_pix = QPixmap(str(_logo_path))
        if not _logo_pix.isNull():
            _logo_pix = _logo_pix.scaledToWidth(
                int(lcd_w - 8),
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_item = QGraphicsPixmapItem(_logo_pix, self)
            logo_item.setPos(
                lcd_x + (lcd_w - _logo_pix.width()) / 2,
                lcd_y + 2,
            )
            logo_item.setZValue(3)
            _logo_glow = QGraphicsDropShadowEffect()
            _logo_glow.setColor(QColor("#30e848"))
            _logo_glow.setBlurRadius(18)
            _logo_glow.setOffset(0, 0)
            logo_item.setGraphicsEffect(_logo_glow)

        _crt_font = QFont("Consolas", 10, QFont.Weight.ExtraBold)
        _crt_font.setStyleHint(QFont.StyleHint.Monospace)
        title = QGraphicsSimpleTextItem("Local Intelligence Rack", self)
        title.setFont(_crt_font)
        title.setBrush(QBrush(QColor("#30e848")))
        _title_w = title.boundingRect().width()
        title.setPos(lcd_x + (lcd_w - _title_w) / 2, lcd_y + 22)
        title.setZValue(3)
        _title_glow = QGraphicsDropShadowEffect()
        _title_glow.setColor(QColor("#30e848"))
        _title_glow.setBlurRadius(16)
        _title_glow.setOffset(0, 0)
        title.setGraphicsEffect(_title_glow)

        # Tabs (vertically centered with LCD bezel)
        tab_y = lcd_y + (lcd_h - 18) / 2
        vent_x = w - 50.0
        top_btn_gap = 4.0
        auto_btn_w, auto_btn_h = 52.0, 18.0
        init_btn_w, init_btn_h = 72.0, 18.0
        init_btn_x = vent_x - 10.0 - init_btn_w
        auto_btn_x = init_btn_x - top_btn_gap - auto_btn_w

        self._tab_log = self._make_tab_button(
            x=lcd_x + lcd_w + 8,
            y=tab_y,
            label="LOG",
            is_active=True,
            on_click=lambda: self.set_tab("LOG"),
        )
        self._tab_sys = self._make_tab_button(
            x=lcd_x + lcd_w + 88,
            y=tab_y,
            label="SYSTEM",
            is_active=False,
            on_click=lambda: self.set_tab("SYSTEM"),
        )
        self._tab_proc = self._make_tab_button(
            x=lcd_x + lcd_w + 168,
            y=tab_y,
            label="PROC",
            is_active=False,
            on_click=lambda: self.set_tab("PROCESSES"),
        )

        # Stats cluster
        stats_x = max(w - 400, lcd_x + lcd_w + 330)
        self._stats_items: dict[str, QGraphicsTextItem] = {}
        stat_defs = [
            ("MODULES", stats_x),
            ("WIRES", stats_x + 90),
            ("WORKERS", stats_x + 180),
        ]
        for label_text, sx in stat_defs:
            lbl = QGraphicsSimpleTextItem(label_text, self)
            lbl.setFont(_LCD_FONT)
            lbl.setBrush(QBrush(QColor("#808890")))
            lbl.setPos(sx, 10)
            lbl.setZValue(3)

            val_bezel = QGraphicsRectItem(sx, 22, 60, 18, self)
            val_bezel.setBrush(QBrush(QColor("#0a0f0a")))
            val_bezel.setPen(QPen(QColor("#2a3a2a"), 0.8))
            val_bezel.setZValue(2)

            val = QGraphicsTextItem("--", self)
            val.setFont(_LCD_FONT)
            val.setDefaultTextColor(QColor("#30b848"))
            val.setPos(sx + 4, 21)
            val.setZValue(3)
            self._stats_items[label_text] = val

        # READY status indicator
        ready_x = stats_x + 270
        ready_lbl = QGraphicsSimpleTextItem("STATUS", self)
        ready_lbl.setFont(_LCD_FONT)
        ready_lbl.setBrush(QBrush(QColor("#808890")))
        ready_lbl.setPos(ready_x, 10)
        ready_lbl.setZValue(3)

        ready_bezel = QGraphicsRectItem(ready_x, 22, 60, 18, self)
        ready_bezel.setBrush(QBrush(QColor("#0a0f0a")))
        ready_bezel.setPen(QPen(QColor("#2a3a2a"), 0.8))
        ready_bezel.setZValue(2)

        self._ready_value = QGraphicsTextItem("OFF", self)
        self._ready_value.setFont(_LCD_FONT)
        self._ready_value.setDefaultTextColor(QColor("#6f6f6f"))
        self._ready_value.setPos(ready_x + 4, 21)
        self._ready_value.setZValue(3)

        # Security shield indicator
        sec_x = ready_x + 70
        sec_lbl = QGraphicsSimpleTextItem("SEC", self)
        sec_lbl.setFont(_LCD_FONT)
        sec_lbl.setBrush(QBrush(QColor("#808890")))
        sec_lbl.setPos(sec_x, 10)
        sec_lbl.setZValue(3)

        self._sec_led = QGraphicsEllipseItem(sec_x + 8, 28, 8, 8, self)
        self._sec_led.setBrush(QBrush(QColor("#6f6f6f")))
        self._sec_led.setPen(QPen(Qt.PenStyle.NoPen))
        self._sec_led.setZValue(3)

        # Power toggle (round icon): left aligned to MODULE STATUS/log bezel.
        # Initial state mirrors hypervisor.system_powered.
        # Keeps power UI correct after template-driven layout rebuilds.
        self._power_on = self.hypervisor.system_powered
        pwr_r    = 17.0                          # outer button radius
        pwr_arc  = 10.0                          # power-symbol arc radius
        pwr_cx   = 30.0 + pwr_r                  # left-align with items below
        pwr_cy   = lcd_y + lcd_h / 2             # vertically centred on LCD bezel

        _on_fill  = QColor("#0a1a0a")
        _on_ring  = QColor("#30e848")
        _on_icon  = QColor("#30e848")
        _on_glow  = QColor("#30e848")
        _off_fill = QColor("#1a0808")
        _off_ring = QColor("#cc1111")
        _off_icon = QColor("#ff3333")
        _off_glow = QColor("#ff2020")

        self._pwr_btn = QGraphicsEllipseItem(
            pwr_cx - pwr_r, pwr_cy - pwr_r,
            pwr_r * 2, pwr_r * 2, self,
        )
        self._pwr_btn.setBrush(QBrush(_on_fill))
        self._pwr_btn.setPen(QPen(_on_ring, 1.8))
        self._pwr_btn.setZValue(4)

        self._pwr_btn_glow = QGraphicsDropShadowEffect()
        self._pwr_btn_glow.setColor(_on_glow)
        self._pwr_btn_glow.setBlurRadius(16)
        self._pwr_btn_glow.setOffset(0, 0)
        self._pwr_btn.setGraphicsEffect(self._pwr_btn_glow)

        _arc_rect = QRectF(pwr_cx - pwr_arc, pwr_cy - pwr_arc, pwr_arc * 2, pwr_arc * 2)
        _pwr_path = QPainterPath()
        _pwr_path.arcMoveTo(_arc_rect, 125)
        _pwr_path.arcTo(_arc_rect, 125, 290)
        _pwr_path.moveTo(pwr_cx, pwr_cy - 2.5)
        _pwr_path.lineTo(pwr_cx, pwr_cy - pwr_arc - 1.2)

        self._pwr_icon = QGraphicsPathItem(_pwr_path, self)
        _icon_pen = QPen(_on_icon, 2.0)
        _icon_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._pwr_icon.setPen(_icon_pen)
        self._pwr_icon.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._pwr_icon.setZValue(5)

        self._pwr_icon_glow = QGraphicsDropShadowEffect()
        self._pwr_icon_glow.setColor(_on_glow)
        self._pwr_icon_glow.setBlurRadius(10)
        self._pwr_icon_glow.setOffset(0, 0)
        self._pwr_icon.setGraphicsEffect(self._pwr_icon_glow)

        def _apply_power_visuals() -> None:
            if self._power_on:
                self._pwr_btn.setBrush(QBrush(_on_fill))
                self._pwr_btn.setPen(QPen(_on_ring, 1.8))
                self._pwr_btn_glow.setColor(_on_glow)
                p = QPen(_on_icon, 2.0)
                p.setCapStyle(Qt.PenCapStyle.RoundCap)
                self._pwr_icon.setPen(p)
                self._pwr_icon_glow.setColor(_on_glow)
            else:
                self._pwr_btn.setBrush(QBrush(_off_fill))
                self._pwr_btn.setPen(QPen(_off_ring, 1.8))
                self._pwr_btn_glow.setColor(_off_glow)
                p = QPen(_off_icon, 2.0)
                p.setCapStyle(Qt.PenCapStyle.RoundCap)
                self._pwr_icon.setPen(p)
                self._pwr_icon_glow.setColor(_off_glow)

        self._apply_power_visuals = _apply_power_visuals
        # Apply correct initial visuals based on the actual power state
        _apply_power_visuals()

        async def _run_manual_init() -> None:
            try:
                await self.hypervisor.init_all()
                self.hv_log.info("INIT complete")
            except Exception as exc:
                self.hv_log.error(f"INIT failed: {exc}")
            finally:
                self._init_task = None

        async def _power_off_sequence() -> None:
            self.hv_log.warn("Rack power OFF")
            try:
                await self.hypervisor.graceful_stop_all()
            except Exception as exc:
                self.hv_log.error(f"Power-off failed: {exc}")
            finally:
                self._power_on = self.hypervisor.system_powered
                _apply_power_visuals()
                self._power_transition_phase = ""
                self._power_transition_task = None

        async def _power_on_sequence() -> None:
            self.hv_log.info("Rack power ON")
            try:
                await self.hypervisor.boot_all()
                if self._auto_init_enabled:
                    self.hv_log.info("AUTO-INIT triggered")
                    self._init_task = asyncio.create_task(_run_manual_init())
                    await self._init_task
            except Exception as exc:
                self.hv_log.error(f"Power-on failed: {exc}")
            finally:
                self._power_on = self.hypervisor.system_powered
                _apply_power_visuals()
                self._power_transition_phase = ""
                self._power_transition_task = None

        def _on_power_toggle() -> None:
            if self._power_transition_task is not None and not self._power_transition_task.done():
                return
            if self._power_on:
                self._power_transition_phase = "off"
                self._power_on = False
                _apply_power_visuals()
                self._power_transition_task = asyncio.create_task(_power_off_sequence())
            else:
                self._power_transition_phase = "on"
                self._power_on = True
                _apply_power_visuals()
                self._power_transition_task = asyncio.create_task(_power_on_sequence())

        class _PowerProxy(QGraphicsEllipseItem):
            def __init__(proxy_self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                proxy_self.setAcceptHoverEvents(True)
                proxy_self.setCursor(Qt.CursorShape.PointingHandCursor)

            def mousePressEvent(proxy_self, event) -> None:  # type: ignore[override]
                if event.button() == Qt.MouseButton.LeftButton:
                    _on_power_toggle()
                    event.accept()

            def hoverEnterEvent(proxy_self, event) -> None:  # type: ignore[override]
                if self._power_on:
                    self._pwr_btn.setPen(QPen(_on_ring.lighter(130), 2.2))
                else:
                    self._pwr_btn.setPen(QPen(_off_ring.lighter(130), 2.2))
                super().hoverEnterEvent(event)

            def hoverLeaveEvent(proxy_self, event) -> None:  # type: ignore[override]
                if self._power_on:
                    self._pwr_btn.setPen(QPen(_on_ring, 1.8))
                else:
                    self._pwr_btn.setPen(QPen(_off_ring, 1.8))
                super().hoverLeaveEvent(event)

        _pwr_proxy = _PowerProxy(
            pwr_cx - pwr_r, pwr_cy - pwr_r,
            pwr_r * 2, pwr_r * 2, self,
        )
        _pwr_proxy.setPen(QPen(Qt.PenStyle.NoPen))
        _pwr_proxy.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        _pwr_proxy.setZValue(6)

        # ── INIT button — right of AUTO, left of vents ───────────
        init_btn_y = 58.0

        self._init_btn_bg = QGraphicsRectItem(
            init_btn_x, init_btn_y, init_btn_w, init_btn_h, self,
        )
        self._init_btn_bg.setBrush(QBrush(QColor("#0c1218")))
        self._init_btn_bg.setPen(QPen(QColor("#2a4060"), 1.0))
        self._init_btn_bg.setZValue(3)

        _init_font = QFont("Consolas", 7, QFont.Weight.Bold)
        self._init_label = QGraphicsSimpleTextItem("INIT", self)
        self._init_label.setFont(_init_font)
        self._init_label.setBrush(QBrush(QColor("#6090c0")))
        _ilr = self._init_label.boundingRect()
        self._init_label.setPos(
            init_btn_x + (init_btn_w - _ilr.width()) / 2,
            init_btn_y + (init_btn_h - _ilr.height()) / 2 - 1,
        )
        self._init_label.setZValue(4)

        class _InitProxy(QGraphicsRectItem):
            def __init__(proxy_self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                proxy_self.setAcceptHoverEvents(True)
                proxy_self.setCursor(Qt.CursorShape.PointingHandCursor)

            def mousePressEvent(proxy_self, event) -> None:  # type: ignore[override]
                if event.button() == Qt.MouseButton.LeftButton:
                    power_busy = self._power_transition_task is not None and not self._power_transition_task.done()
                    init_busy = self._init_task is not None and not self._init_task.done()
                    if self.hypervisor.system_powered and not power_busy and not init_busy:
                        self.hv_log.info("INIT triggered — checking module readiness")
                        self._init_task = asyncio.create_task(_run_manual_init())
                event.accept()

            def hoverEnterEvent(proxy_self, event) -> None:  # type: ignore[override]
                power_busy = self._power_transition_task is not None and not self._power_transition_task.done()
                init_busy = self._init_task is not None and not self._init_task.done()
                if self.hypervisor.system_powered and not power_busy and not init_busy:
                    self._init_btn_bg.setPen(QPen(QColor("#4080c0"), 1.4))
                super().hoverEnterEvent(event)

            def hoverLeaveEvent(proxy_self, event) -> None:  # type: ignore[override]
                self._init_btn_bg.setPen(QPen(QColor("#2a4060"), 1.0))
                super().hoverLeaveEvent(event)

        _init_proxy = _InitProxy(
            init_btn_x, init_btn_y, init_btn_w, init_btn_h, self,
        )
        _init_proxy.setPen(QPen(Qt.PenStyle.NoPen))
        _init_proxy.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        _init_proxy.setZValue(5)
        _init_proxy.setToolTip("Run init scripts and check readiness for all modules")

        # ── AUTO-INIT toggle (AUTO=green / OFF=orange) ───────────
        auto_btn_y = 58.0
        auto_lbl = QGraphicsSimpleTextItem("AUTO-INIT", self)
        auto_lbl.setFont(QFont("Consolas", 7))
        auto_lbl.setBrush(QBrush(QColor("#6f7680")))
        auto_lbl.setPos(auto_btn_x - 64.0, auto_btn_y + 5.0)
        auto_lbl.setZValue(4)

        self._auto_mode_btn = QGraphicsRectItem(auto_btn_x, auto_btn_y, auto_btn_w, auto_btn_h, self)
        self._auto_mode_btn.setZValue(3)
        self._auto_mode_label = QGraphicsSimpleTextItem("", self)
        self._auto_mode_label.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        self._auto_mode_label.setZValue(4)

        def _apply_auto_mode_visuals() -> None:
            if self._auto_init_enabled:
                self._auto_mode_btn.setBrush(QBrush(QColor("#1a2e1a")))
                self._auto_mode_btn.setPen(QPen(QColor("#39d353"), 1.0))
                self._auto_mode_label.setBrush(QBrush(QColor("#b6d9b6")))
                self._auto_mode_label.setText("AUTO")
            else:
                self._auto_mode_btn.setBrush(QBrush(QColor("#2e2a1a")))
                self._auto_mode_btn.setPen(QPen(QColor("#e8a838"), 1.0))
                self._auto_mode_label.setBrush(QBrush(QColor("#e8c878")))
                self._auto_mode_label.setText("OFF")
            rect = self._auto_mode_label.boundingRect()
            self._auto_mode_label.setPos(
                self._auto_mode_btn.rect().x() + (self._auto_mode_btn.rect().width() - rect.width()) / 2,
                self._auto_mode_btn.rect().y() + (self._auto_mode_btn.rect().height() - rect.height()) / 2 - 1,
            )

        self._apply_auto_mode_visuals = _apply_auto_mode_visuals
        _apply_auto_mode_visuals()

        class _AutoModeProxy(QGraphicsRectItem):
            def __init__(proxy_self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                proxy_self.setAcceptHoverEvents(True)
                proxy_self.setCursor(Qt.CursorShape.PointingHandCursor)

            def mousePressEvent(proxy_self, event) -> None:  # type: ignore[override]
                if event.button() == Qt.MouseButton.LeftButton:
                    self._auto_init_enabled = not self._auto_init_enabled
                    _apply_auto_mode_visuals()
                    state = "AUTO" if self._auto_init_enabled else "OFF"
                    self.hv_log.info(f"AUTO-INIT mode: {state}")
                event.accept()

            def hoverEnterEvent(proxy_self, event) -> None:  # type: ignore[override]
                if self._auto_init_enabled:
                    self._auto_mode_btn.setPen(QPen(QColor("#5be06f"), 1.4))
                else:
                    self._auto_mode_btn.setPen(QPen(QColor("#f0b448"), 1.4))
                super().hoverEnterEvent(event)

            def hoverLeaveEvent(proxy_self, event) -> None:  # type: ignore[override]
                _apply_auto_mode_visuals()
                super().hoverLeaveEvent(event)

        _auto_proxy = _AutoModeProxy(auto_btn_x, auto_btn_y, auto_btn_w, auto_btn_h, self)
        _auto_proxy.setPen(QPen(Qt.PenStyle.NoPen))
        _auto_proxy.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        _auto_proxy.setZValue(5)
        _auto_proxy.setToolTip("When AUTO is on, run INIT after power-on boot")

        # Per-module status LEDs row
        led_row_y = 60.0
        led_label = QGraphicsSimpleTextItem("MODULE STATUS", self)
        led_label.setFont(_LCD_FONT)
        led_label.setBrush(QBrush(QColor("#808890")))
        _lbl_h = led_label.boundingRect().height()
        led_label.setPos(30, led_row_y + 11.0 - _lbl_h / 2)
        led_label.setZValue(3)

        self._module_leds: dict[str, QGraphicsEllipseItem] = {}
        self._module_led_labels: dict[str, QGraphicsSimpleTextItem] = {}

        modules = list(self.hypervisor.active_modules.values())
        led_start_x = 150.0
        led_spacing = 50.0
        for i, mod in enumerate(modules):
            lx = led_start_x + i * led_spacing
            led = QGraphicsEllipseItem(lx, led_row_y + 2, 8, 8, self)
            led.setPen(QPen(QColor("#1a1a1a"), 0.6))
            led.setBrush(QBrush(_status_color(mod.status)))
            led.setZValue(3)
            self._module_leds[mod.module_id] = led

            abbr = mod.MODULE_NAME[:4].upper()
            lbl = QGraphicsSimpleTextItem(abbr, self)
            lbl.setFont(QFont("Consolas", 6))
            lbl.setBrush(QBrush(QColor("#606870")))
            lbl.setPos(lx - 4, led_row_y + 12)
            lbl.setZValue(3)
            self._module_led_labels[mod.module_id] = lbl

        # Decorative ventilation slots
        vent_x = w - 50
        for vy in range(56, 80, 6):
            slot = QGraphicsRectItem(vent_x, vy, 20, 2, self)
            slot.setBrush(QBrush(QColor("#0e1014")))
            slot.setPen(QPen(Qt.PenStyle.NoPen))
            slot.setZValue(2)

        # Screen bezel + background
        log_x, log_y = 30.0, 88.0
        log_w = self._width - 60.0
        log_h = self.PANEL_HEIGHT - log_y - 16.0
        log_bezel = QGraphicsRectItem(log_x - 2, log_y - 2, log_w + 4, log_h + 4, self)
        log_bezel.setBrush(QBrush(QColor("#161a16")))
        log_bezel.setPen(QPen(QColor("#2a3a2a"), 1.0))
        log_bezel.setZValue(2)

        log_bg = QGraphicsRectItem(log_x, log_y, log_w, log_h, self)
        log_bg.setBrush(QBrush(QColor("#0a100a")))
        log_bg.setPen(QPen(Qt.PenStyle.NoPen))
        log_bg.setZValue(2)
        log_bg.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True
        )

        self._log_x = log_x
        self._log_y = log_y
        self._log_w = log_w
        self._log_h = log_h
        self._log_text_items: list[QGraphicsTextItem] = []
        self._log_measure_item: QGraphicsTextItem | None = None
        self._log_group = QGraphicsItemGroup(log_bg)
        self._system_group = QGraphicsItemGroup(self)
        self._processes_group = QGraphicsItemGroup(self)
        self._log_group.setZValue(3)
        self._system_group.setZValue(3)
        self._processes_group.setZValue(3)

        _log_pad_top = 4.0
        _log_pad_bottom = 14.0
        _log_text_width = log_w - 8.0
        self._log_pad_top = _log_pad_top
        self._log_pad_bottom = _log_pad_bottom
        self._log_text_width = _log_text_width
        line_h = (log_h - _log_pad_top - _log_pad_bottom) / self.LOG_VISIBLE_LINES
        for i in range(self.LOG_VISIBLE_LINES):
            item = QGraphicsTextItem("", log_bg)
            item.setFont(_MONO_FONT)
            item.setDefaultTextColor(_LOG_DIM)
            item.setTextWidth(_log_text_width)
            item.setPos(log_x + 4, log_y + _log_pad_top + i * line_h)
            item.setZValue(3)
            self._log_text_items.append(item)
            self._log_group.addToGroup(item)

        # Offscreen measurer for wrapped log entry heights.
        self._log_measure_item = QGraphicsTextItem("", log_bg)
        self._log_measure_item.setFont(_MONO_FONT)
        self._log_measure_item.setTextWidth(_log_text_width)
        self._log_measure_item.setVisible(False)
        self._log_group.addToGroup(self._log_measure_item)

        class _LogWheelProxy(QGraphicsRectItem):
            def __init__(proxy_self, panel: HypervisorPanelItem, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                proxy_self._panel = panel
                proxy_self.setPen(QPen(Qt.PenStyle.NoPen))
                proxy_self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                proxy_self.setZValue(4)
                proxy_self.setToolTip("Scroll the mouse wheel to view older log lines")

            def wheelEvent(proxy_self, event) -> None:  # type: ignore[override]
                if proxy_self._panel._active_tab != "LOG":
                    event.ignore()
                    return
                # QGraphicsSceneWheelEvent uses delta(), while some wheel events
                # expose angleDelta().y(); support both for compatibility.
                delta = 0
                if hasattr(event, "delta"):
                    delta = int(event.delta())
                elif hasattr(event, "angleDelta"):
                    angle_delta = event.angleDelta()
                    delta = int(angle_delta.y()) if angle_delta is not None else 0
                if delta == 0:
                    event.ignore()
                    return
                steps = max(-8, min(8, delta // 120))
                # Wheel up (positive delta) → see older entries.
                proxy_self._panel._log_skip_tail = max(
                    0, proxy_self._panel._log_skip_tail + steps
                )
                proxy_self._panel._render_wrapped_log(proxy_self._panel.hv_log.lines)
                event.accept()

        self._log_wheel_proxy = _LogWheelProxy(self, log_x, log_y, log_w, log_h, self)

        # System monitor content
        sys_title = QGraphicsSimpleTextItem("SYSTEM MONITOR", self)
        sys_title.setFont(_LCD_FONT)
        sys_title.setBrush(QBrush(QColor("#8ea58e")))
        sys_title.setPos(log_x + 4, log_y + 2)
        self._system_group.addToGroup(sys_title)

        hw_x = log_x + 6
        hw_y = log_y + 18
        right_x = log_x + (log_w * 0.52)

        # ── Left column: CPU model / cores / util + RAM ──────────

        cpu_model_lbl = QGraphicsSimpleTextItem("CPU MODEL", self)
        cpu_model_lbl.setFont(_LCD_FONT)
        cpu_model_lbl.setBrush(QBrush(QColor("#808890")))
        cpu_model_lbl.setPos(hw_x, hw_y)
        self._system_group.addToGroup(cpu_model_lbl)

        self._cpu_name_value = QGraphicsTextItem("--", self)
        self._cpu_name_value.setFont(_LCD_FONT)
        self._cpu_name_value.setDefaultTextColor(QColor("#b6d9b6"))
        self._cpu_name_value.setPos(hw_x + 58, hw_y - 2)
        self._system_group.addToGroup(self._cpu_name_value)

        cpu_cores_lbl = QGraphicsSimpleTextItem("CORES", self)
        cpu_cores_lbl.setFont(_LCD_FONT)
        cpu_cores_lbl.setBrush(QBrush(QColor("#808890")))
        cpu_cores_lbl.setPos(hw_x, hw_y + 12)
        self._system_group.addToGroup(cpu_cores_lbl)

        self._cpu_cores_value = QGraphicsTextItem("--", self)
        self._cpu_cores_value.setFont(_LCD_FONT)
        self._cpu_cores_value.setDefaultTextColor(QColor("#b6d9b6"))
        self._cpu_cores_value.setPos(hw_x + 58, hw_y + 10)
        self._system_group.addToGroup(self._cpu_cores_value)

        cpu_util_lbl = QGraphicsSimpleTextItem("CPU UTIL", self)
        cpu_util_lbl.setFont(_LCD_FONT)
        cpu_util_lbl.setBrush(QBrush(QColor("#808890")))
        cpu_util_lbl.setPos(hw_x, hw_y + 24)
        self._system_group.addToGroup(cpu_util_lbl)

        self._cpu_value = QGraphicsTextItem("--%", self)
        self._cpu_value.setFont(_LCD_FONT)
        self._cpu_value.setDefaultTextColor(QColor("#30b848"))
        self._cpu_value.setPos(hw_x + 58, hw_y + 22)
        self._system_group.addToGroup(self._cpu_value)

        self._cpu_leds = self._build_led_bar(
            hw_x + 90, hw_y + 26, segments=14, parent=self, seg_w=7.0, seg_h=7.0
        )
        for led in self._cpu_leds:
            self._system_group.addToGroup(led)

        ram_lbl = QGraphicsSimpleTextItem("RAM", self)
        ram_lbl.setFont(_LCD_FONT)
        ram_lbl.setBrush(QBrush(QColor("#808890")))
        ram_lbl.setPos(hw_x, hw_y + 38)
        self._system_group.addToGroup(ram_lbl)

        self._ram_value = QGraphicsTextItem("--/--G", self)
        self._ram_value.setFont(_LCD_FONT)
        self._ram_value.setDefaultTextColor(QColor("#30b848"))
        self._ram_value.setPos(hw_x + 58, hw_y + 36)
        self._system_group.addToGroup(self._ram_value)

        self._ram_leds = self._build_led_bar(
            hw_x + 112, hw_y + 40, segments=14, parent=self, seg_w=7.0, seg_h=7.0
        )
        for led in self._ram_leds:
            self._system_group.addToGroup(led)

        # ── Right column: GPU / Backend / OS / Disks ─────────────

        gpu_lbl = QGraphicsSimpleTextItem("GPU", self)
        gpu_lbl.setFont(_LCD_FONT)
        gpu_lbl.setBrush(QBrush(QColor("#808890")))
        gpu_lbl.setPos(right_x, hw_y)
        self._system_group.addToGroup(gpu_lbl)

        self._gpu_value = QGraphicsTextItem("UNKNOWN", self)
        self._gpu_value.setFont(_LCD_FONT)
        self._gpu_value.setDefaultTextColor(QColor("#a0a6ad"))
        self._gpu_value.setPos(right_x + 30, hw_y - 2)
        self._system_group.addToGroup(self._gpu_value)

        backend_lbl = QGraphicsSimpleTextItem("DETECTED", self)
        backend_lbl.setFont(_LCD_FONT)
        backend_lbl.setBrush(QBrush(QColor("#808890")))
        backend_lbl.setPos(right_x, hw_y + 14)
        self._system_group.addToGroup(backend_lbl)

        self._backend_value = QGraphicsTextItem("--", self)
        self._backend_value.setFont(_LCD_FONT)
        self._backend_value.setDefaultTextColor(QColor("#a0a6ad"))
        self._backend_value.setPos(right_x + 56, hw_y + 12)
        self._system_group.addToGroup(self._backend_value)

        self._backend_cuda_led = QGraphicsEllipseItem(right_x + 104, hw_y + 15, 7, 7, self)
        self._backend_cuda_led.setPen(QPen(QColor("#1a1a1a"), 0.7))
        self._backend_cuda_led.setBrush(QBrush(QColor("#1a1d1a")))
        self._system_group.addToGroup(self._backend_cuda_led)

        cuda_lbl = QGraphicsSimpleTextItem("CUDA", self)
        cuda_lbl.setFont(QFont("Consolas", 6))
        cuda_lbl.setBrush(QBrush(QColor("#808890")))
        cuda_lbl.setPos(right_x + 114, hw_y + 14)
        self._system_group.addToGroup(cuda_lbl)

        self._backend_vulkan_led = QGraphicsEllipseItem(right_x + 136, hw_y + 15, 7, 7, self)
        self._backend_vulkan_led.setPen(QPen(QColor("#1a1a1a"), 0.7))
        self._backend_vulkan_led.setBrush(QBrush(QColor("#1a1d1a")))
        self._system_group.addToGroup(self._backend_vulkan_led)

        vk_lbl = QGraphicsSimpleTextItem("VK", self)
        vk_lbl.setFont(QFont("Consolas", 6))
        vk_lbl.setBrush(QBrush(QColor("#808890")))
        vk_lbl.setPos(right_x + 146, hw_y + 14)
        self._system_group.addToGroup(vk_lbl)

        self._backend_rocm_led = QGraphicsEllipseItem(right_x + 162, hw_y + 15, 7, 7, self)
        self._backend_rocm_led.setPen(QPen(QColor("#1a1a1a"), 0.7))
        self._backend_rocm_led.setBrush(QBrush(QColor("#1a1d1a")))
        self._system_group.addToGroup(self._backend_rocm_led)

        rocm_lbl = QGraphicsSimpleTextItem("ROCm", self)
        rocm_lbl.setFont(QFont("Consolas", 6))
        rocm_lbl.setBrush(QBrush(QColor("#808890")))
        rocm_lbl.setPos(right_x + 172, hw_y + 14)
        self._system_group.addToGroup(rocm_lbl)

        os_lbl = QGraphicsSimpleTextItem("OS", self)
        os_lbl.setFont(_LCD_FONT)
        os_lbl.setBrush(QBrush(QColor("#808890")))
        os_lbl.setPos(right_x, hw_y + 28)
        self._system_group.addToGroup(os_lbl)

        self._os_value = QGraphicsTextItem("--", self)
        self._os_value.setFont(_LCD_FONT)
        self._os_value.setDefaultTextColor(QColor("#a0a6ad"))
        self._os_value.setPos(right_x + 20, hw_y + 26)
        self._system_group.addToGroup(self._os_value)

        disk_lbl = QGraphicsSimpleTextItem("DISKS", self)
        disk_lbl.setFont(_LCD_FONT)
        disk_lbl.setBrush(QBrush(QColor("#808890")))
        disk_lbl.setPos(right_x, hw_y + 42)
        self._system_group.addToGroup(disk_lbl)

        self._disk_lines: list[QGraphicsTextItem] = []
        for i in range(3):
            d = QGraphicsTextItem("--", self)
            d.setFont(QFont("Consolas", 7))
            d.setDefaultTextColor(QColor("#8ea58e"))
            d.setPos(right_x + 38, hw_y + 40 + i * 11)
            self._system_group.addToGroup(d)
            self._disk_lines.append(d)

        # ── Backend install status (below disks) ──────────────────
        bstat_y = hw_y + 78
        bstat_lbl = QGraphicsSimpleTextItem("INSTALLED", self)
        bstat_lbl.setFont(_LCD_FONT)
        bstat_lbl.setBrush(QBrush(QColor("#808890")))
        bstat_lbl.setPos(right_x, bstat_y)
        self._system_group.addToGroup(bstat_lbl)

        _bstat_font = QFont("Consolas", 6)
        self._bstat_items: dict[str, tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        bstat_backends = [("CUDA", "cuda"), ("VK", "vulkan"), ("ROCm", "rocm"), ("CPU", "cpu")]
        bx = right_x + 56
        for label, key in bstat_backends:
            led = QGraphicsEllipseItem(bx, bstat_y + 2, 7, 7, self)
            led.setPen(QPen(QColor("#1a1a1a"), 0.7))
            led.setBrush(QBrush(QColor("#1a1d1a")))
            led.setZValue(3)
            self._system_group.addToGroup(led)

            txt = QGraphicsSimpleTextItem(label, self)
            txt.setFont(_bstat_font)
            txt.setBrush(QBrush(QColor("#808890")))
            txt.setPos(bx + 10, bstat_y + 1)
            txt.setZValue(3)
            self._system_group.addToGroup(txt)

            self._bstat_items[key] = (led, txt)
            bx += 38 + len(label) * 2

        active_lbl = QGraphicsSimpleTextItem("RUNTIME", self)
        active_lbl.setFont(_LCD_FONT)
        active_lbl.setBrush(QBrush(QColor("#808890")))
        active_lbl.setPos(right_x, bstat_y + 14)
        self._system_group.addToGroup(active_lbl)

        self._active_backend_value = QGraphicsTextItem("--", self)
        self._active_backend_value.setFont(_LCD_FONT)
        self._active_backend_value.setDefaultTextColor(QColor("#b6d9b6"))
        self._active_backend_value.setPos(right_x + 56, bstat_y + 12)
        self._system_group.addToGroup(self._active_backend_value)

        status_lbl = QGraphicsSimpleTextItem("STATUS", self)
        status_lbl.setFont(_LCD_FONT)
        status_lbl.setBrush(QBrush(QColor("#808890")))
        status_lbl.setPos(right_x, bstat_y + 28)
        self._system_group.addToGroup(status_lbl)

        self._backend_status_value = QGraphicsTextItem("--", self)
        self._backend_status_value.setFont(QFont("Consolas", 7))
        self._backend_status_value.setDefaultTextColor(QColor("#a0a6ad"))
        self._backend_status_value.setPos(right_x + 56, bstat_y + 26)
        self._system_group.addToGroup(self._backend_status_value)

        self._cached_installed_backends: dict[str, bool] | None = None
        self._cached_backend_diagnostics: dict[str, object] | None = None
        self._installed_probe_started = False

        self._system_group.setVisible(False)

        # ── Processes tab: registered processes (left) + port bindings (right) ──
        proc_left_x = log_x + 4
        proc_right_x = log_x + log_w * 0.48
        proc_y = log_y + 2

        reg_title = QGraphicsSimpleTextItem("REGISTERED", self)
        reg_title.setFont(_LCD_FONT)
        reg_title.setBrush(QBrush(QColor("#8ea58e")))
        reg_title.setPos(proc_left_x, proc_y)
        self._processes_group.addToGroup(reg_title)

        hv_ref = self.hypervisor
        hv_log_ref = self.hv_log

        class _KillProxy(QGraphicsRectItem):
            def __init__(proxy_self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                proxy_self.setAcceptHoverEvents(True)
                proxy_self.setCursor(Qt.CursorShape.PointingHandCursor)
                proxy_self._pid: int = 0

            def mousePressEvent(proxy_self, event) -> None:  # type: ignore[override]
                if event.button() == Qt.MouseButton.LeftButton and proxy_self._pid:
                    hv_ref.kill_registered_process(proxy_self._pid)
                    hv_log_ref.info(f"Killed process {proxy_self._pid}")
                event.accept()

        self._registered_process_rows: list[tuple[QGraphicsTextItem, QGraphicsRectItem, _KillProxy]] = []
        for i in range(8):
            row_y = proc_y + 14 + i * 14
            txt = QGraphicsTextItem("", self)
            txt.setFont(_MONO_FONT)
            txt.setDefaultTextColor(_LOG_DIM)
            txt.setPos(proc_left_x + 18, row_y - 2)
            self._processes_group.addToGroup(txt)

            kill_btn = QGraphicsRectItem(0, 0, 12, 10, self)
            kill_btn.setPos(proc_left_x, row_y - 1)
            kill_btn.setBrush(QBrush(QColor("#2a1515")))
            kill_btn.setPen(QPen(QColor("#6a2020"), 0.8))
            kill_btn.setZValue(4)
            self._processes_group.addToGroup(kill_btn)
            kill_x = QGraphicsSimpleTextItem("×", kill_btn)
            kill_x.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            kill_x.setBrush(QBrush(QColor("#e04040")))
            kill_x.setPos(2, -1)
            kill_x.setZValue(5)

            proxy = _KillProxy(0, 0, 12, 10, self)
            proxy.setPos(proc_left_x, row_y - 1)
            proxy.setPen(QPen(Qt.PenStyle.NoPen))
            proxy.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            proxy.setZValue(6)
            proxy.setToolTip("Kill process")
            self._processes_group.addToGroup(proxy)
            self._registered_process_rows.append((txt, kill_btn, proxy))

        ports_title = QGraphicsSimpleTextItem("127.0.0.1 PORTS", self)
        ports_title.setFont(_LCD_FONT)
        ports_title.setBrush(QBrush(QColor("#8ea58e")))
        ports_title.setPos(proc_right_x, proc_y)
        self._processes_group.addToGroup(ports_title)

        self._port_binding_rows: list[QGraphicsTextItem] = []
        for i in range(10):
            row_y = proc_y + 14 + i * 12
            txt = QGraphicsTextItem("", self)
            txt.setFont(QFont("Consolas", 7))
            txt.setDefaultTextColor(_LOG_DIM)
            txt.setPos(proc_right_x, row_y - 2)
            self._processes_group.addToGroup(txt)
            self._port_binding_rows.append(txt)

        self._processes_group.setVisible(False)

    def _make_tab_button(self, x: float, y: float, label: str, is_active: bool, on_click) -> QGraphicsRectItem:
        btn_w, btn_h = 72.0, 18.0
        btn = QGraphicsRectItem(0, 0, btn_w, btn_h, self)
        btn.setPos(x, y)
        btn.setBrush(QBrush(QColor("#11151b") if is_active else QColor("#0c0f14")))
        btn.setPen(QPen(QColor("#2a3a2a") if is_active else QColor("#2a2d32"), 1.0))
        btn.setZValue(3)
        txt = QGraphicsSimpleTextItem(label, btn)
        txt.setFont(QFont("Consolas", 7))
        txt.setBrush(QBrush(QColor("#30b848") if is_active else QColor("#7a818a")))
        trect = txt.boundingRect()
        txt.setPos((btn_w - trect.width()) / 2, (btn_h - trect.height()) / 2 - 1)
        txt.setZValue(4)

        class _ClickProxy(QGraphicsRectItem):
            def mousePressEvent(self, event) -> None:  # type: ignore[override]
                on_click()
                event.accept()

        proxy = _ClickProxy(0, 0, btn_w, btn_h, self)
        proxy.setPos(x, y)
        proxy.setPen(QPen(Qt.PenStyle.NoPen))
        proxy.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        proxy.setZValue(5)
        proxy.setToolTip(f"Switch to {label} view")

        btn._label_item = txt  # type: ignore[attr-defined]
        btn._proxy_item = proxy  # type: ignore[attr-defined]
        btn._active = is_active  # type: ignore[attr-defined]
        btn._label = label  # type: ignore[attr-defined]
        return btn

    def set_tab(self, tab: str) -> None:
        tab = tab.upper()
        if tab not in ("LOG", "SYSTEM", "PROCESSES"):
            return
        self._active_tab = tab
        self._log_group.setVisible(tab == "LOG")
        self._system_group.setVisible(tab == "SYSTEM")
        self._processes_group.setVisible(tab == "PROCESSES")
        self._log_wheel_proxy.setVisible(tab == "LOG")
        self._set_tab_active(self._tab_log, tab == "LOG")
        self._set_tab_active(self._tab_sys, tab == "SYSTEM")
        self._set_tab_active(self._tab_proc, tab == "PROCESSES")

    def _set_tab_active(self, tab_item: QGraphicsRectItem, active: bool) -> None:
        tab_item.setBrush(QBrush(QColor("#11151b") if active else QColor("#0c0f14")))
        tab_item.setPen(QPen(QColor("#2a3a2a") if active else QColor("#2a2d32"), 1.0))
        lbl = getattr(tab_item, "_label_item", None)
        if isinstance(lbl, QGraphicsSimpleTextItem):
            lbl.setBrush(QBrush(QColor("#30b848") if active else QColor("#7a818a")))

    @staticmethod
    def _build_led_bar(
        x: float,
        y: float,
        segments: int,
        parent: QGraphicsItem,
        *,
        seg_w: float = 6.0,
        seg_h: float = 6.0,
        gap: float = 2.0,
    ) -> list[QGraphicsRectItem]:
        leds: list[QGraphicsRectItem] = []
        for i in range(segments):
            led = QGraphicsRectItem(x + i * (seg_w + gap), y, seg_w, seg_h, parent)
            led.setBrush(QBrush(QColor("#1a1d1a")))
            led.setPen(QPen(QColor("#0e120e"), 0.6))
            led.setZValue(3)
            leds.append(led)
        return leds

    def set_hardware_snapshot(self, snapshot: HardwareSnapshot) -> None:
        self._hardware = snapshot

    def refresh(self) -> None:
        # Drain system_log (report_state from modules) into hv_log for display
        while self.hypervisor.system_log:
            payload = self.hypervisor.system_log.popleft()
            msg = payload.data.get("log_message")
            if msg:
                src = payload.source_module or ""
                full = f"{src}: {msg}" if src else msg
                sev = payload.data.get("severity", "INFO")
                if sev == "WARN":
                    self.hv_log.warn(full)
                elif sev == "ERROR":
                    self.hv_log.error(full)
                else:
                    self.hv_log.info(full)

        modules = self.hypervisor.active_modules
        wires = self.hypervisor.active_wires
        workers = getattr(self.hypervisor, "_workers", {})

        self._stats_items["MODULES"].setPlainText(f"{len(modules):>2}")
        self._stats_items["WIRES"].setPlainText(f"{len(wires):>2}")
        self._stats_items["WORKERS"].setPlainText(f"{len(workers):>2}")

        # READY indicator
        if not self.hypervisor.system_powered:
            self._ready_value.setPlainText("OFF")
            self._ready_value.setDefaultTextColor(QColor("#6f6f6f"))
        else:
            enabled = [m for m in modules.values() if m.enabled]
            ready_count = sum(1 for m in enabled if m.status == ModuleStatus.READY)
            if len(enabled) > 0 and ready_count == len(enabled):
                self._ready_value.setPlainText("READY")
                self._ready_value.setDefaultTextColor(QColor("#39d353"))
            elif len(enabled) > 0:
                self._ready_value.setPlainText(f"{ready_count}/{len(enabled)}")
                self._ready_value.setDefaultTextColor(QColor("#d68c1a"))
            else:
                self._ready_value.setPlainText("--")
                self._ready_value.setDefaultTextColor(QColor("#6f6f6f"))

        power_busy = self._power_transition_task is not None and not self._power_transition_task.done()
        init_busy = self._init_task is not None and not self._init_task.done()

        # Keep power icon synced to true rack state while transitions run.
        self._power_on = self.hypervisor.system_powered or power_busy
        self._apply_power_visuals()

        # INIT button dimming
        if self.hypervisor.system_powered and not power_busy and not init_busy:
            self._init_btn_bg.setOpacity(1.0)
            self._init_label.setOpacity(1.0)
        else:
            self._init_btn_bg.setOpacity(0.35)
            self._init_label.setOpacity(0.35)

        if self._hardware is not None and self._active_tab == "SYSTEM":
            self._render_hardware(self._hardware)

        if self._active_tab == "PROCESSES":
            self._render_processes()

        sec_status = self.hypervisor.get_security_status()
        if sec_status.vault_locked:
            self._sec_led.setBrush(QBrush(QColor("#c03030")))
        elif all(sec_status.auth_enabled.values()) and sec_status.wire_signing_active:
            self._sec_led.setBrush(QBrush(QColor("#39d353")))
        elif any(sec_status.auth_enabled.values()) or sec_status.wire_signing_active:
            self._sec_led.setBrush(QBrush(QColor("#d68c1a")))
        else:
            self._sec_led.setBrush(QBrush(QColor("#c03030")))

        for mid, led in self._module_leds.items():
            mod = modules.get(mid)
            if mod:
                led.setBrush(QBrush(_status_color(mod.status)))

        lines = self.hv_log.lines
        self._render_wrapped_log(lines)

    def _clamp_log_skip(self, num_lines: int) -> None:
        max_skip = max(0, num_lines - 1)
        if self._log_skip_tail > max_skip:
            self._log_skip_tail = max_skip

    def _render_wrapped_log(self, lines: list[tuple[str, str, QColor]]) -> None:
        if not self._log_text_items:
            return

        measure = self._log_measure_item
        if measure is None:
            return

        n = len(lines)
        self._clamp_log_skip(n)
        visible_end = n - self._log_skip_tail
        if visible_end <= 0:
            for item in self._log_text_items:
                item.setPlainText("")
            return

        # Keep latest entries visible while allowing wrapped rows to occupy
        # more than one baseline without overlapping the row below.
        take_start = max(0, visible_end - self.LOG_WRAP_LOOKBACK)
        candidates = lines[take_start:visible_end]
        available_h = max(0.0, self._log_h - self._log_pad_top - self._log_pad_bottom)
        min_h = 10.0
        packed: list[tuple[str, QColor, float]] = []
        used_h = 0.0

        for ts, msg, color in reversed(candidates):
            text = f"[{ts}] {msg}"
            measure.setPlainText(text)
            row_h = max(min_h, measure.boundingRect().height())
            extra_gap = self.LOG_ENTRY_GAP if packed else 0.0
            if used_h + row_h + extra_gap > available_h:
                break
            packed.append((text, color, row_h))
            used_h += row_h + extra_gap
            if len(packed) >= len(self._log_text_items):
                break

        packed.reverse()

        y = self._log_y + self._log_pad_top
        for i, item in enumerate(self._log_text_items):
            if i < len(packed):
                text, color, row_h = packed[i]
                item.setPlainText(text)
                item.setDefaultTextColor(color)
                item.setPos(self._log_x + 4, y)
                y += row_h + self.LOG_ENTRY_GAP
            else:
                item.setPlainText("")

    def _render_hardware(self, hw: HardwareSnapshot) -> None:
        cpu_pct = max(0.0, min(100.0, hw.cpu_percent))
        self._cpu_value.setPlainText(f"{cpu_pct:>3.0f}%")
        self._cpu_name_value.setPlainText((hw.cpu_name or "CPU")[:24])
        self._cpu_name_value.setToolTip(hw.cpu_name or "CPU")
        self._cpu_cores_value.setPlainText(f"{hw.cpu_physical_cores}P / {hw.cpu_logical_cores}L")

        ram_total_gb = hw.ram_total_mb / 1024.0
        ram_used_gb  = hw.ram_used_mb  / 1024.0
        ram_avail_gb = hw.ram_available_mb / 1024.0
        self._ram_value.setPlainText(f"{ram_used_gb:.1f}/{ram_total_gb:.0f}G")
        self._ram_value.setToolTip(f"{ram_used_gb:.1f} used / {ram_total_gb:.1f}G total  ({ram_avail_gb:.1f}G free)")

        def paint_bar(leds: list[QGraphicsRectItem], pct: float) -> None:
            filled = int(round((pct / 100.0) * len(leds)))
            for i, led in enumerate(leds):
                if i < filled:
                    c = _LOG_GREEN if pct < 60 else (_LOG_AMBER if pct < 85 else _LOG_RED)
                    led.setBrush(QBrush(c))
                else:
                    led.setBrush(QBrush(QColor("#1a1d1a")))

        paint_bar(self._cpu_leds, cpu_pct)
        ram_pct = (hw.ram_used_mb / hw.ram_total_mb) * 100.0 if hw.ram_total_mb > 0 else 0.0
        paint_bar(self._ram_leds, ram_pct)

        gpu_name = hw.gpu_name.strip() or "UNKNOWN"
        gpu_label = f"{hw.gpu_vendor.value.upper()}: {gpu_name}" if gpu_name != "UNKNOWN" else hw.gpu_vendor.value.upper()
        if hw.vram_total_mb > 0:
            vram_gb = hw.vram_total_mb / 1024.0
            vram_used_gb = hw.vram_used_mb / 1024.0
            gpu_label += f"  [{vram_used_gb:.1f}/{vram_gb:.1f}G]"
        self._gpu_value.setPlainText(gpu_label[:42])
        self._gpu_value.setToolTip(gpu_name)

        backend = hw.ai_backend_hint.value.upper()
        self._backend_value.setPlainText(backend)
        _backend_colors = {
            AiBackend.VULKAN: QColor("#b6cbe8"),
            AiBackend.CPU: QColor("#a0a6ad"),
        }
        self._backend_value.setDefaultTextColor(_backend_colors.get(hw.ai_backend_hint, QColor("#a0a6ad")))

        # CUDA LED repurposed: off (no CUDA backend in current build)
        self._backend_cuda_led.setBrush(QBrush(QColor("#1a1d1a")))
        self._backend_vulkan_led.setBrush(
            QBrush(QColor("#4e8fe8") if hw.ai_backend_hint == AiBackend.VULKAN else QColor("#1a1d1a"))
        )
        # ROCm LED repurposed: off (no ROCm backend in current build)
        self._backend_rocm_led.setBrush(QBrush(QColor("#1a1d1a")))

        self._os_value.setPlainText(hw.os_info[:38])

        for i, line_item in enumerate(self._disk_lines):
            if i < len(hw.disks):
                line_item.setPlainText(hw.disks[i][:38])
                line_item.setToolTip(hw.disks[i])
            else:
                line_item.setPlainText("")

        # Backend install status (cached — probed in background to avoid UI stalls)
        if self._cached_installed_backends is None and not self._installed_probe_started:
            self._installed_probe_started = True
            threading.Thread(
                target=self._probe_installed_backends,
                daemon=True,
            ).start()

        _installed_colors = {
            "cuda":   QColor("#39d353"),
            "vulkan": QColor("#4e8fe8"),
            "rocm":   QColor("#e87020"),
            "cpu":    QColor("#39d353"),
        }
        for key, (led, _txt) in self._bstat_items.items():
            is_installed = bool(self._cached_installed_backends and self._cached_installed_backends.get(key, False))
            if is_installed:
                led.setBrush(QBrush(_installed_colors.get(key, QColor("#39d353"))))
            else:
                led.setBrush(QBrush(QColor("#1a1d1a")))

        active = hw.ai_backend_hint.value
        runtime_backend = "idle"
        runtime_reason = "No basic brain loaded"
        for module in self.hypervisor.active_modules.values():
            if module.MODULE_NAME != "basic_brain":
                continue
            runtime_backend = str(module.params.get("_runtime_backend", runtime_backend)).strip().lower() or "idle"
            runtime_reason = str(module.params.get("_runtime_backend_reason", runtime_reason)).strip() or runtime_reason
            break

        if runtime_backend == "idle":
            self._active_backend_value.setPlainText("IDLE")
            self._active_backend_value.setDefaultTextColor(QColor("#a0a6ad"))
            self._backend_status_value.setPlainText(runtime_reason[:42])
            self._backend_status_value.setDefaultTextColor(QColor("#a0a6ad"))
        elif runtime_backend == "error":
            self._active_backend_value.setPlainText("ERROR")
            self._active_backend_value.setDefaultTextColor(QColor("#e04040"))
            self._backend_status_value.setPlainText(runtime_reason[:42])
            self._backend_status_value.setDefaultTextColor(QColor("#e04040"))
        elif self._cached_installed_backends is None:
            self._active_backend_value.setPlainText(f"{runtime_backend.upper()} (probing...)")
            self._active_backend_value.setDefaultTextColor(QColor("#a0a6ad"))
            self._backend_status_value.setPlainText("Checking installed packages...")
            self._backend_status_value.setDefaultTextColor(QColor("#a0a6ad"))
        elif runtime_backend in _installed_colors and self._cached_installed_backends.get(runtime_backend, False):
            self._active_backend_value.setPlainText(runtime_backend.upper())
            self._active_backend_value.setDefaultTextColor(
                _installed_colors.get(runtime_backend, QColor("#b6d9b6"))
            )
            self._backend_status_value.setPlainText(runtime_reason[:42] or "Backend ready")
            self._backend_status_value.setDefaultTextColor(QColor("#8ea58e"))
        else:
            mismatch = get_probe_service().explain_mismatch(hw.ai_backend_hint)
            self._active_backend_value.setPlainText(runtime_backend.upper())
            self._active_backend_value.setDefaultTextColor(QColor("#d68c1a"))
            status_text = mismatch or runtime_reason or f"{active.upper()} not installed"
            self._backend_status_value.setPlainText(status_text[:42])
            self._backend_status_value.setDefaultTextColor(QColor("#d68c1a"))

    def is_power_transition_busy(self) -> bool:
        task = self._power_transition_task
        return task is not None and not task.done()

    def power_transition_message(self) -> str:
        if self._power_transition_phase == "off":
            return "Powering rack off..."
        if self._power_transition_phase == "on":
            return "Powering rack on..."
        return "Applying rack power state..."

    def _probe_installed_backends(self) -> None:
        """Run backend package checks in a subprocess off the UI thread.

        Uses a clean child process so ``ensure_ggml_backends()`` can load
        Vulkan DLLs without hitting the ``llama_cpp already imported``
        guard that exists in the UI process.
        """
        try:
            self._cached_backend_diagnostics = get_probe_service().backend_diagnostics()
            self._cached_installed_backends = dict(
                self._cached_backend_diagnostics.get("installed_backends", {})
            )
        except Exception:
            self._cached_installed_backends = {}
            self._cached_backend_diagnostics = {}

    def _render_processes(self) -> None:
        """Update the Processes tab: registered processes and port bindings."""
        registered = self.hypervisor.get_registered_processes()
        for i, (txt_item, kill_btn, proxy) in enumerate(self._registered_process_rows):
            if i < len(registered):
                pid, info = registered[i]
                label = info.get("label", "")
                internal_port = info.get("internal_port")
                if internal_port is not None:
                    label = f"{label} (int :{internal_port})"
                txt_item.setPlainText(f"PID {pid}  {label}")
                txt_item.setDefaultTextColor(_LOG_GREEN)
                kill_btn.setVisible(True)
                proxy.setVisible(True)
                proxy._pid = pid
            else:
                txt_item.setPlainText("")
                kill_btn.setVisible(False)
                proxy.setVisible(False)
                proxy._pid = 0

        bindings = self.hypervisor.get_local_port_bindings()
        for i, row_item in enumerate(self._port_binding_rows):
            if i < len(bindings):
                port, pid, name = bindings[i]
                row_item.setPlainText(f":{port}  PID {pid}  {name[:20]}")
                row_item.setDefaultTextColor(_LOG_CYAN)
            else:
                row_item.setPlainText("")
