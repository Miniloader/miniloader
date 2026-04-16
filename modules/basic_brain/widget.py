"""
basic_brain/widget.py — Basic Brain Card Controls
===================================================
Builds and manages the expanded UI controls on the Basic Brain
module card: model selection, GPU/CTX/TEMP parameter rows, spinning
fans, and the load/eject lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.hypervisor_panel import HypervisorLog
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QLinearGradient, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
)

from core.base_module import ModuleStatus
from core.settings_store import SettingsStore
from ui.rack_items import CardButtonItem, _LCD_FONT, _LOG_CYAN, _LOG_DIM, _LOG_GREEN

_QUANT_RE = re.compile(r"((?:I)?Q[1-8]_[A-Z0-9_]+|F16|F32)", re.IGNORECASE)
_PARAM_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)[Bb]")

_LOG_FONT = QFont("Consolas", 7)
_LOG_FONT.setStyleHint(QFont.StyleHint.Monospace)

_LOG_AMBER = QColor("#d4a020")
_SIMPLE_QUALITY_PRESETS: tuple[tuple[str, str, dict[str, float | int]], ...] = (
    (
        "speed",
        "SPEED",
        {"ctx_length": 2048, "n_batch": 512, "top_k": 30, "top_p": 0.90, "repeat_penalty": 1.05},
    ),
    (
        "balanced",
        "BALANCED",
        {"ctx_length": 4096, "n_batch": 256, "top_k": 40, "top_p": 0.95, "repeat_penalty": 1.10},
    ),
    (
        "quality",
        "QUALITY",
        {"ctx_length": 8192, "n_batch": 128, "top_k": 80, "top_p": 0.98, "repeat_penalty": 1.15},
    ),
)


# ── Inference log data model ─────────────────────────────────────────────────

@dataclass
class InferenceEntry:
    timestamp: str
    prompt_preview: str
    response_preview: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float


@dataclass
class ModelEntry:
    model_path: Path
    mmproj_path: Path | None
    display_name: str
    vision: bool


class BrainInferenceLog:
    """Ring buffer of timestamped inference events for the brain status panel."""

    MAX_ENTRIES = 100
    VISIBLE_LINES = 6  # displayed rows (3 request/response pairs)

    def __init__(self) -> None:
        self._entries: deque[InferenceEntry] = deque(maxlen=self.MAX_ENTRIES)
        self._boot_time = time.monotonic()

    def _ts(self) -> str:
        elapsed = time.monotonic() - self._boot_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log(
        self,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_s: float,
        total_s: float,
    ) -> None:
        self._entries.append(InferenceEntry(
            timestamp=self._ts(),
            prompt_preview=prompt[:80].replace("\n", " "),
            response_preview=response[:80].replace("\n", " "),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft_s,
            total_s=total_s,
        ))

    def render_lines(self) -> list[tuple[str, QColor]]:
        """Return the last VISIBLE_LINES display rows as (text, color) pairs."""
        lines: list[tuple[str, QColor]] = []
        for entry in self._entries:
            total_tok = entry.prompt_tokens + entry.completion_tokens
            tps = (
                entry.completion_tokens / entry.total_s
                if entry.total_s > 0 else 0.0
            )
            req = f"[{entry.timestamp}] \u25b6 {entry.prompt_preview}"
            metrics = (
                f"  {total_tok}tok"
                f"  TTFT {entry.ttft_s:.2f}s"
                f"  {tps:.1f}t/s"
            )
            rsp = f"           \u25c0 {entry.response_preview[:46]}"
            lines.append((req[:120], _LOG_CYAN))
            lines.append((rsp + metrics, _LOG_GREEN))
        return lines[-self.VISIBLE_LINES :]


class BasicBrainCardBuilder:
    """Builds and manages the Basic Brain module's expanded card controls.

    Owns the parameter text items, model-state indicators, model-info
    readouts, and busy-lock.  The parent RackWindow delegates
    brain-specific card building and refresh calls to this object.
    """

    CONTROLS_HEIGHT: float = 276.0
    SIMPLE_CONTROLS_HEIGHT: float = 228.0
    ADVANCED_CONTROLS_HEIGHT: float = 250.0
    _VISIBLE_MODEL_ROWS = 5
    _UI_MODE_SETTING_KEY = "basic_brain.ui_mode"

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._param_items: dict[str, dict[str, QGraphicsSimpleTextItem]] = {}
        self._model_indicators: dict[str, dict[str, QGraphicsItem]] = {}
        self._model_info_items: dict[str, dict[str, QGraphicsItem]] = {}
        self._model_list_items: dict[str, list[dict[str, object]]] = {}
        self._model_entries: dict[str, list[ModelEntry]] = {}
        self._folder_items: dict[str, QGraphicsSimpleTextItem] = {}
        self._scroll_offsets: dict[str, int] = {}
        self._brain_logs: dict[str, BrainInferenceLog] = {}
        self._log_displays: dict[str, list[QGraphicsTextItem]] = {}
        self._mode_toggles: dict[str, dict[str, CardButtonItem]] = {}
        self._mode_groups: dict[str, dict[str, list[QGraphicsItem]]] = {}
        self._quality_labels: dict[str, QGraphicsSimpleTextItem] = {}
        self._gpu_suggested: set[str] = set()
        self._busy: set[str] = set()
        self._ui_mode: str = "simple"
        self._ui_mode_load_started = False

    # ── Lifecycle helpers ─────────────────────────────────────────

    def clear(self) -> None:
        """Reset internal state (called when the rack layout is rebuilt)."""
        self._param_items.clear()
        self._model_indicators.clear()
        self._model_info_items.clear()
        self._model_list_items.clear()
        self._model_entries.clear()
        self._folder_items.clear()
        self._log_displays.clear()
        self._mode_toggles.clear()
        self._mode_groups.clear()
        self._quality_labels.clear()
        self._gpu_suggested.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._param_items.keys())

    def refresh_all(self) -> None:
        """Refresh parameter readouts for every tracked brain module."""
        for module_id in list(self._param_items):
            self._auto_suggest_gpu_layers(module_id)
            self._refresh_param_text(module_id)
            self._refresh_model_list(module_id)
            self._refresh_inference_log(module_id)

    def _settings_store(self) -> SettingsStore | None:
        vault = getattr(self._controller.hypervisor, "vault", None)
        if vault is None:
            return None
        return SettingsStore(vault)

    def _ensure_ui_mode_loaded(self) -> None:
        if self._ui_mode_load_started:
            return
        self._ui_mode_load_started = True
        asyncio.create_task(self._load_ui_mode_setting())

    async def _load_ui_mode_setting(self) -> None:
        store = self._settings_store()
        if store is None:
            return
        previous_mode = self._ui_mode
        raw = await store.get(self._UI_MODE_SETTING_KEY, machine_id=SettingsStore.get_machine_id())
        mode = str(raw or "").strip().lower()
        if mode not in {"simple", "advanced"}:
            return
        self._ui_mode = mode
        for module_id in list(self._mode_groups):
            self._apply_mode_visibility(module_id)
        if self._param_items and mode != previous_mode:
            self._controller._rebuild_layout()

    async def _persist_ui_mode_setting(self) -> None:
        store = self._settings_store()
        if store is None:
            return
        await store.set(
            self._UI_MODE_SETTING_KEY,
            self._ui_mode,
            machine_id=SettingsStore.get_machine_id(),
        )

    def _set_ui_mode(self, mode: str) -> None:
        mode = mode.strip().lower()
        if mode not in {"simple", "advanced"} or mode == self._ui_mode:
            return
        self._ui_mode = mode
        for module_id in list(self._mode_groups):
            self._apply_mode_visibility(module_id)
        asyncio.create_task(self._persist_ui_mode_setting())
        self._controller._rebuild_layout()

    def get_controls_height(self, module_id: str | None = None) -> float:
        self._ensure_ui_mode_loaded()
        if self._ui_mode == "simple":
            return self.SIMPLE_CONTROLS_HEIGHT
        return self.ADVANCED_CONTROLS_HEIGHT

    def _apply_mode_visibility(self, module_id: str) -> None:
        groups = self._mode_groups.get(module_id)
        if groups is None:
            return
        simple_mode = self._ui_mode == "simple"
        simple_items = groups.get("simple", [])
        advanced_items = groups.get("advanced", [])
        # Items in both groups are shared (visible in all modes); only mode-exclusive
        # items should be toggled.
        advanced_id_set = {id(i) for i in advanced_items}
        simple_id_set = {id(i) for i in simple_items}
        for item in simple_items:
            item.setVisible(simple_mode or id(item) in advanced_id_set)
        for item in advanced_items:
            item.setVisible((not simple_mode) or id(item) in simple_id_set)

        toggle = self._mode_toggles.get(module_id, {})
        simple_btn = toggle.get("simple")
        adv_btn = toggle.get("advanced")
        if isinstance(simple_btn, CardButtonItem):
            self._style_mode_button(simple_btn, active=simple_mode)
        if isinstance(adv_btn, CardButtonItem):
            self._style_mode_button(adv_btn, active=not simple_mode)

    @staticmethod
    def _style_mode_button(btn: CardButtonItem, *, active: bool) -> None:
        if active:
            btn.setBrush(QBrush(QColor("#d9dee3")))
            btn.setPen(QPen(QColor("#aeb7c2"), 0.8))
            color = QColor("#0b0d10")
        else:
            btn.setBrush(QBrush(QColor("#0d0e10")))
            btn.setPen(QPen(QColor("#7d858f"), 0.8))
            color = QColor("#d6dbe0")
        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setBrush(QBrush(color))

    def _cycle_quality_preset(self, module_id: str, direction: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        keys = [name for name, _, _ in _SIMPLE_QUALITY_PRESETS]
        labels = {name: label for name, label, _ in _SIMPLE_QUALITY_PRESETS}
        current = str(module.params.get("_simple_quality_preset", "balanced")).strip().lower()
        idx = keys.index(current) if current in keys else keys.index("balanced")
        idx = (idx + direction) % len(keys)
        key = keys[idx]
        module.params["_simple_quality_preset"] = key
        updates = dict(_SIMPLE_QUALITY_PRESETS[idx][2])
        module.params.update(updates)
        label_item = self._quality_labels.get(module_id)
        if isinstance(label_item, QGraphicsSimpleTextItem):
            label_item.setText(labels[key])
            parent = label_item.parentItem()
            if isinstance(parent, QGraphicsRectItem):
                rect = parent.rect()
                text_rect = label_item.boundingRect()
                label_item.setPos(
                    (rect.width() - text_rect.width()) / 2,
                    (rect.height() - text_rect.height()) / 2 - 1,
                )
        self._refresh_param_text(module_id)
        self._controller.hv_log.info(
            f"{module_id}: quality preset {labels[key]} applied (apply with LOAD)"
        )

    # ── Inference log public API ──────────────────────────────────

    def log_inference(
        self,
        module_id: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_s: float,
        total_s: float,
    ) -> None:
        """Record a completed inference event and refresh the display."""
        brain_log = self._brain_logs.get(module_id)
        if brain_log is None:
            return
        brain_log.log(prompt, response, prompt_tokens, completion_tokens, ttft_s, total_s)
        self._refresh_inference_log(module_id)

    # ── Card construction ─────────────────────────────────────────

    def build_controls(
        self,
        card: ModuleCardItem,
        module: BaseModule,
        width: float,
        top_y: float,
        P: float = 0.0,
    ) -> list:
        """Build brain-specific controls on *card* and return fan items."""
        module_id = module.module_id
        btn_w   = 16.0
        btn_gap = 4.0
        gpu_x   = 18 + P
        right_edge = width - 18 - P
        _btn_font = QFont("Consolas", 7)
        _btn_font.setStyleHint(QFont.StyleHint.Monospace)
        self._ensure_ui_mode_loaded()
        simple_mode = self._ui_mode == "simple"

        simple_items: list[QGraphicsItem] = []
        advanced_items: list[QGraphicsItem] = []

        def _track(item: QGraphicsItem, *, simple: bool = False, advanced: bool = False) -> QGraphicsItem:
            if simple:
                simple_items.append(item)
            if advanced:
                advanced_items.append(item)
            return item

        mode_btn_y = top_y - 1
        simple_btn = CardButtonItem(
            right_edge - 84, mode_btn_y, 40, 14, "SIMPLE",
            lambda: self._set_ui_mode("simple"),
            card,
            label_font=_btn_font,
        )
        adv_btn = CardButtonItem(
            right_edge - 40, mode_btn_y, 40, 14, "ADV",
            lambda: self._set_ui_mode("advanced"),
            card,
            label_font=_btn_font,
        )
        self._mode_toggles[module_id] = {"simple": simple_btn, "advanced": adv_btn}

        # ── Header-row status LEDs (hypervisor style) ────────────
        _led_label_font = QFont("Consolas", 6)
        init_btn_x = width - 22 - P - 78
        led_spacing = 36.0
        header_led_y = 10 + P
        header_lbl_y = 22 + P

        led_anchor_x = init_btn_x - 26
        model_led_x = led_anchor_x - 2 * led_spacing
        tools_led_x = led_anchor_x - led_spacing
        vision_led_x = led_anchor_x

        model_state_led = QGraphicsEllipseItem(model_led_x, header_led_y, 8, 8, card)
        model_state_led.setPen(QPen(QColor("#1a1a1a"), 0.6))
        model_state_led.setBrush(QBrush(QColor("#3a3f45")))
        model_state_led.setZValue(3)

        model_state_text = QGraphicsSimpleTextItem("MDL", card)
        model_state_text.setFont(_led_label_font)
        model_state_text.setBrush(QBrush(QColor("#606870")))
        model_state_text.setPos(model_led_x - 2, header_lbl_y)
        model_state_text.setZValue(3)

        mi_tools_led = QGraphicsEllipseItem(tools_led_x, header_led_y, 8, 8, card)
        mi_tools_led.setPen(QPen(QColor("#1a1a1a"), 0.6))
        mi_tools_led.setBrush(QBrush(QColor("#3a3f45")))
        mi_tools_led.setZValue(3)

        tools_lbl = QGraphicsSimpleTextItem("TOOL", card)
        tools_lbl.setFont(_led_label_font)
        tools_lbl.setBrush(QBrush(QColor("#606870")))
        tools_lbl.setPos(tools_led_x - 4, header_lbl_y)
        tools_lbl.setZValue(3)

        mi_vision_led = QGraphicsEllipseItem(vision_led_x, header_led_y, 8, 8, card)
        mi_vision_led.setPen(QPen(QColor("#1a1a1a"), 0.6))
        mi_vision_led.setBrush(QBrush(QColor("#3a3f45")))
        mi_vision_led.setZValue(3)

        vision_lbl = QGraphicsSimpleTextItem("VIS", card)
        vision_lbl.setFont(_led_label_font)
        vision_lbl.setBrush(QBrush(QColor("#606870")))
        vision_lbl.setPos(vision_led_x - 2, header_lbl_y)
        vision_lbl.setZValue(3)

        # ── Three-column zone ────────────────────────────────────
        zone_y = top_y + (30.0 if simple_mode else 18.0)

        # --- LEFT: Two-column parameter grid (LED readout style) ─
        grid_y = zone_y
        row_h  = 16.0
        lbl_w  = 38.0
        scr_w  = 50.0
        scr_h  = 13.0
        col2_x = gpu_x + 130

        _screen_font = QFont("Consolas", 8, QFont.Weight.Bold)
        _screen_font.setStyleHint(QFont.StyleHint.Monospace)

        def _led_row(cx: float, cy: float, label: str, key: str) -> QGraphicsSimpleTextItem:
            lbl = QGraphicsSimpleTextItem(label, card)
            lbl.setBrush(QBrush(QColor("#9ea4ad")))
            lbl.setPos(cx, cy - 1)
            _track(lbl, advanced=True)

            sx = cx + lbl_w
            bg = QGraphicsRectItem(sx, cy - 1, scr_w, scr_h, card)
            g = QLinearGradient(sx, cy - 1, sx, cy - 1 + scr_h)
            g.setColorAt(0.0, QColor("#0a100a"))
            g.setColorAt(1.0, QColor("#070c07"))
            bg.setBrush(QBrush(g))
            bg.setPen(QPen(QColor("#2a3a2a"), 0.8))
            bg.setZValue(2)
            _track(bg, advanced=True)

            val = QGraphicsSimpleTextItem("", card)
            val.setFont(_screen_font)
            val.setBrush(QBrush(QColor("#30e848")))
            val.setPos(sx + 3, cy - 2)
            val.setZValue(3)
            _track(val, advanced=True)

            bx = cx + lbl_w + scr_w + 4
            _track(CardButtonItem(
                bx, cy - 2, btn_w, btn_w, "-",
                lambda k=key: self._adjust_param(module_id, k, -1), card,
            ), advanced=True)
            _track(CardButtonItem(
                bx + btn_w + btn_gap, cy - 2, btn_w, btn_w, "+",
                lambda k=key: self._adjust_param(module_id, k, +1), card,
            ), advanced=True)
            return val

        gpu_val  = _led_row(gpu_x,  grid_y,              "GPU",   "gpu_layers")
        ctx_val  = _led_row(gpu_x,  grid_y + row_h,      "CTX",   "ctx_length")
        bat_val  = _led_row(gpu_x,  grid_y + 2 * row_h,  "BATCH", "n_batch")
        thr_val  = _led_row(gpu_x,  grid_y + 3 * row_h,  "THRD",  "cpu_threads")
        seed_val = _led_row(gpu_x,  grid_y + 4 * row_h,  "SEED",  "seed")
        vkdev_val = _led_row(gpu_x, grid_y + 5 * row_h,  "VK#",   "vulkan_device")

        temp_val = _led_row(col2_x, grid_y,              "TEMP",  "temperature")
        topp_val = _led_row(col2_x, grid_y + row_h,      "TOP P", "top_p")
        topk_val = _led_row(col2_x, grid_y + 2 * row_h,  "TOP K", "top_k")
        rpen_val = _led_row(col2_x, grid_y + 3 * row_h,  "R PEN", "repeat_penalty")
        fa_val   = _led_row(col2_x, grid_y + 4 * row_h,  "FA",    "flash_attn")
        kv_val   = _led_row(col2_x, grid_y + 5 * row_h,  "KV",    "cache_type_k")

        # --- Model info LCD (compact vertical stack) ─────────────
        param_end_x = col2_x + lbl_w + scr_w + 4 + btn_w + btn_gap + btn_w
        if simple_mode:
            info_x = gpu_x
            info_w = 220.0
        else:
            info_x = param_end_x + 6.0
            info_w = 220.0
        info_y = zone_y
        info_row_h = 14.0
        info_pad = 4.0
        panel_h = 90.0
        info_h = panel_h

        info_bezel = QGraphicsRectItem(
            info_x - 2, info_y - 2, info_w + 4, info_h + 4, card,
        )
        info_bezel.setBrush(QBrush(QColor("#161a16")))
        info_bezel.setPen(QPen(QColor("#2a3a2a"), 0.8))
        info_bezel.setZValue(2)
        _track(info_bezel, simple=True, advanced=True)

        grad_info = QLinearGradient(info_x, info_y, info_x, info_y + info_h)
        grad_info.setColorAt(0.0, QColor("#0a100a"))
        grad_info.setColorAt(1.0, QColor("#070c07"))
        info_bg = QGraphicsRectItem(info_x, info_y, info_w, info_h, card)
        info_bg.setBrush(QBrush(grad_info))
        info_bg.setPen(QPen(Qt.PenStyle.NoPen))
        info_bg.setZValue(2)
        _track(info_bg, simple=True, advanced=True)

        _ifont = QFont("Consolas", 7)
        _ifont.setStyleHint(QFont.StyleHint.Monospace)
        _lbl_c = QColor("#607060")
        _val_c = QColor("#8ea58e")

        def _info_label(x: float, y: float, text: str) -> None:
            it = QGraphicsSimpleTextItem(text, card)
            it.setFont(_ifont)
            it.setBrush(QBrush(_lbl_c))
            it.setPos(x, y)
            it.setZValue(3)
            _track(it, simple=True, advanced=True)

        def _info_value(x: float, y: float, default: str = "\u2014") -> QGraphicsSimpleTextItem:
            it = QGraphicsSimpleTextItem(default, card)
            it.setFont(_ifont)
            it.setBrush(QBrush(_val_c))
            it.setPos(x, y)
            it.setZValue(3)
            _track(it, simple=True, advanced=True)
            return it

        ix = info_x + 4
        iy0 = info_y + info_pad
        iv_off = 48.0

        _info_label(ix, iy0, "NAME")
        mi_name = _info_value(ix + iv_off, iy0)

        _info_label(ix, iy0 + info_row_h, "QUANT")
        mi_quant = _info_value(ix + iv_off, iy0 + info_row_h)

        _info_label(ix, iy0 + 2 * info_row_h, "SIZE")
        mi_size = _info_value(ix + iv_off, iy0 + 2 * info_row_h)

        _info_label(ix, iy0 + 3 * info_row_h, "N CTX")
        mi_nctx = _info_value(ix + iv_off, iy0 + 3 * info_row_h)

        _info_label(ix, iy0 + 4 * info_row_h, "MMPROJ")
        mi_mmproj = _info_value(ix + iv_off, iy0 + 4 * info_row_h)

        # --- Model selector panel ────────────────────────────────
        sel_x = info_x + info_w + 10.0
        sel_w = right_edge - sel_x

        list_y = zone_y
        list_h = panel_h

        list_panel = QGraphicsRectItem(sel_x, list_y, sel_w, list_h, card)
        pgrad = QLinearGradient(sel_x, list_y, sel_x, list_y + list_h)
        pgrad.setColorAt(0.0, QColor("#0e0f10"))
        pgrad.setColorAt(1.0, QColor("#060707"))
        list_panel.setBrush(QBrush(pgrad))
        list_panel.setPen(QPen(QColor("#4f565f"), 1.0))
        list_panel.setZValue(2)
        _track(list_panel, simple=True, advanced=True)

        list_title = QGraphicsSimpleTextItem("DETECTED MODELS", card)
        list_title.setFont(_LCD_FONT)
        list_title.setBrush(QBrush(QColor("#c8d0d8")))
        list_title.setPos(sel_x + 8, list_y + 2)
        list_title.setZValue(3)
        _track(list_title, simple=True, advanced=True)

        folder_val = QGraphicsSimpleTextItem("", card)
        folder_val.setFont(_LCD_FONT)
        folder_val.setBrush(QBrush(QColor("#9aa3ae")))
        folder_val.setPos(sel_x + 112, list_y + 2)
        folder_val.setZValue(3)
        _track(folder_val, simple=True, advanced=True)

        _track(CardButtonItem(
            sel_x + sel_w - 22, list_y + 2, 18, 14, "^",
            lambda: self._scroll_model_list(module_id, -1),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)
        _track(CardButtonItem(
            sel_x + sel_w - 22, list_y + list_h - 16, 18, 14, "v",
            lambda: self._scroll_model_list(module_id, +1),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)

        row_items: list[dict[str, object]] = []
        row_top = list_y + 18
        model_row_h = 14.0
        for i in range(self._VISIBLE_MODEL_ROWS):
            ry = row_top + i * model_row_h
            row_bg = QGraphicsRectItem(sel_x + 4, ry, sel_w - 30, 12, card)
            row_bg.setBrush(QBrush(QColor("#121315")))
            row_bg.setPen(QPen(QColor("#2f3338"), 0.6))
            row_bg.setZValue(2)
            _track(row_bg, simple=True, advanced=True)

            sel_btn = CardButtonItem(
                sel_x + 6, ry, 14, 12, " ",
                lambda ridx=i: self._select_visible_row(module_id, ridx),
                card,
                label_font=_btn_font,
            )
            sel_btn.setBrush(QBrush(QColor("#0d0e10")))
            sel_btn.setPen(QPen(QColor("#7d858f"), 0.8))
            _track(sel_btn, simple=True, advanced=True)

            file_txt = QGraphicsSimpleTextItem("", card)
            file_txt.setFont(_LCD_FONT)
            file_txt.setBrush(QBrush(QColor("#d6dbe0")))
            file_txt.setPos(sel_x + 24, ry - 1)
            file_txt.setZValue(3)
            _track(file_txt, simple=True, advanced=True)

            row_items.append({"sel_btn": sel_btn, "file_txt": file_txt})

        action_btn_y = mode_btn_y
        action_total_w = 44 + 4 + 90 + 4 + 48 + 4 + 16
        simple_adv_left = right_edge - 84
        action_x = max(sel_x, simple_adv_left - action_total_w - 8)

        _track(CardButtonItem(
            action_x, action_btn_y, 44, 14, "CONFIG",
            lambda: self._open_backend_config(module_id),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)
        _track(CardButtonItem(
            action_x + 48, action_btn_y, 90, 14, "VULKAN DIAG",
            lambda: self._run_vulkan_diagnostics(module_id),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)
        browse_x = action_x + 48 + 94
        _track(CardButtonItem(
            browse_x, action_btn_y, 48, 14, "BROWSE",
            lambda: self._browse_folder(module_id),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)
        _track(CardButtonItem(
            browse_x + 52, action_btn_y, 16, 14, "+",
            lambda: self._add_folder(module_id),
            card,
            label_font=_btn_font,
        ), simple=True, advanced=True)

        _track(CardButtonItem(
            gpu_x, mode_btn_y, 16, 14, "<",
            lambda: self._cycle_quality_preset(module_id, -1),
            card,
            label_font=_btn_font,
        ), simple=True)
        quality_bg = QGraphicsRectItem(0, 0, 68, 14, card)
        quality_bg.setPos(gpu_x + 20, mode_btn_y)
        quality_bg.setBrush(QBrush(QColor("#1e242e")))
        quality_bg.setPen(QPen(QColor("#2d3846"), 0.9))
        quality_bg.setZValue(2)
        _track(quality_bg, simple=True)
        quality_val = QGraphicsSimpleTextItem("BALANCED", quality_bg)
        quality_val.setFont(_btn_font)
        quality_val.setBrush(QBrush(QColor("#d3d8df")))
        rect = quality_bg.rect()
        text_rect = quality_val.boundingRect()
        quality_val.setPos((rect.width() - text_rect.width()) / 2, (rect.height() - text_rect.height()) / 2 - 1)
        _track(CardButtonItem(
            gpu_x + 92, mode_btn_y, 16, 14, ">",
            lambda: self._cycle_quality_preset(module_id, +1),
            card,
            label_font=_btn_font,
        ), simple=True)
        self._quality_labels[module_id] = quality_val

        # ── Inference log panel ──────────────────────────────────
        if simple_mode:
            tallest_bottom = max(info_y + info_h, list_y + list_h)
        else:
            tallest_bottom = max(
                grid_y + 5 * row_h,
                info_y + info_h,
                list_y + list_h,
            )
        full_w = right_edge - gpu_x

        _LOG_VISIBLE = BrainInferenceLog.VISIBLE_LINES
        _log_line_h  = 12.0
        _log_title_h = 13.0
        _log_pad_top = 3.0
        _log_pad_bot = 4.0
        _log_inner_h = _log_pad_top + _log_title_h + _LOG_VISIBLE * _log_line_h + _log_pad_bot

        log_panel_y = tallest_bottom + (4.0 if simple_mode else 6.0)

        log_bezel = QGraphicsRectItem(
            gpu_x - 2, log_panel_y - 2, full_w + 4, _log_inner_h + 4, card,
        )
        log_bezel.setBrush(QBrush(QColor("#161a16")))
        log_bezel.setPen(QPen(QColor("#2a3a2a"), 0.8))
        log_bezel.setZValue(2)

        grad_log = QLinearGradient(gpu_x, log_panel_y, gpu_x, log_panel_y + _log_inner_h)
        grad_log.setColorAt(0.0, QColor("#0a100a"))
        grad_log.setColorAt(1.0, QColor("#070c07"))
        log_bg = QGraphicsRectItem(gpu_x, log_panel_y, full_w, _log_inner_h, card)
        log_bg.setBrush(QBrush(grad_log))
        log_bg.setPen(QPen(Qt.PenStyle.NoPen))
        log_bg.setZValue(2)

        log_title_item = QGraphicsSimpleTextItem("INFERENCE LOG", card)
        log_title_item.setFont(_LOG_FONT)
        log_title_item.setBrush(QBrush(QColor("#607060")))
        log_title_item.setPos(gpu_x + 4, log_panel_y + _log_pad_top)
        log_title_item.setZValue(3)

        sep_log = QGraphicsRectItem(
            gpu_x + 4, log_panel_y + _log_pad_top + _log_title_h - 2,
            full_w - 8, 1, card,
        )
        sep_log.setBrush(QBrush(QColor("#1e2a1e")))
        sep_log.setPen(QPen(Qt.PenStyle.NoPen))
        sep_log.setZValue(3)

        log_rows: list[QGraphicsTextItem] = []
        rows_y0 = log_panel_y + _log_pad_top + _log_title_h
        for i in range(_LOG_VISIBLE):
            row_item = QGraphicsTextItem("", card)
            row_item.setFont(_LOG_FONT)
            row_item.setDefaultTextColor(_LOG_DIM)
            row_item.setPos(gpu_x + 4, rows_y0 + i * _log_line_h)
            row_item.setZValue(3)
            log_rows.append(row_item)

        # ── Store references ─────────────────────────────────────
        self._param_items[module_id] = {
            "gpu_layers":     gpu_val,
            "ctx_length":     ctx_val,
            "temperature":    temp_val,
            "cpu_threads":    thr_val,
            "top_p":          topp_val,
            "top_k":          topk_val,
            "repeat_penalty": rpen_val,
            "n_batch":        bat_val,
            "seed":           seed_val,
            "vulkan_device":  vkdev_val,
            "flash_attn":     fa_val,
            "cache_type_k":   kv_val,
        }
        self._model_indicators[module_id] = {
            "led": model_state_led,
            "text": model_state_text,
        }
        self._model_info_items[module_id] = {
            "name":      mi_name,
            "quant":     mi_quant,
            "param_size": mi_size,
            "native_ctx": mi_nctx,
            "mmproj":    mi_mmproj,
            "tools_led": mi_tools_led,
            "tools_label": tools_lbl,
            "vision_led": mi_vision_led,
            "vision_label": vision_lbl,
        }
        self._model_list_items[module_id] = row_items
        self._folder_items[module_id] = folder_val
        self._log_displays[module_id] = log_rows
        self._mode_groups[module_id] = {
            "simple": simple_items,
            "advanced": advanced_items,
        }
        self._scroll_offsets.setdefault(module_id, 0)
        if module_id not in self._brain_logs:
            self._brain_logs[module_id] = BrainInferenceLog()

        self._refresh_param_text(module_id)
        self._refresh_model_indicator(module_id)
        self._refresh_model_info(module_id)
        self._refresh_model_list(module_id)
        self._refresh_inference_log(module_id)
        preset_key = str(module.params.get("_simple_quality_preset", "balanced")).strip().lower()
        for key, label, _ in _SIMPLE_QUALITY_PRESETS:
            if key == preset_key:
                quality_val.setText(label)
                break
        self._apply_mode_visibility(module_id)

        return []

    # ── Refresh helpers ───────────────────────────────────────────

    def _refresh_param_text(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        text_items = self._param_items.get(module_id)
        if module is None or text_items is None:
            return
        params = module.params

        gpu_layers = int(params.get("gpu_layers", -1))
        text_items["gpu_layers"].setText("ALL" if gpu_layers < 0 else str(gpu_layers))
        text_items["ctx_length"].setText(str(int(params.get("ctx_length", 4096))))
        text_items["n_batch"].setText(str(int(params.get("n_batch", 256))))
        text_items["cpu_threads"].setText(str(int(params.get("cpu_threads", 4))))
        seed = int(params.get("seed", -1))
        text_items["seed"].setText("RND" if seed < 0 else str(seed))
        if "vulkan_device" in text_items:
            text_items["vulkan_device"].setText(str(int(params.get("vulkan_device", 0))))

        text_items["temperature"].setText(f"{float(params.get('temperature', 0.7)):.2f}")
        text_items["top_p"].setText(f"{float(params.get('top_p', 0.95)):.2f}")
        text_items["top_k"].setText(str(int(params.get("top_k", 40))))
        text_items["repeat_penalty"].setText(f"{float(params.get('repeat_penalty', 1.1)):.2f}")
        if "flash_attn" in text_items:
            fa_raw = params.get("flash_attn", "auto")
            if isinstance(fa_raw, bool):
                fa_label = "ON" if fa_raw else "OFF"
            else:
                fa_label = str(fa_raw).upper()
            text_items["flash_attn"].setText(fa_label)
        if "cache_type_k" in text_items:
            cache_k = str(params.get("cache_type_k", "f16")).upper()
            cache_v = str(params.get("cache_type_v", "f16")).upper()
            cache_label = cache_k if cache_k == cache_v else f"{cache_k}/{cache_v}"
            text_items["cache_type_k"].setText(cache_label[:8])

        self._refresh_model_indicator(module_id)
        self._refresh_model_info(module_id)

    def _refresh_model_indicator(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        indicator = self._model_indicators.get(module_id)
        if module is None or indicator is None:
            return
        led = indicator["led"]
        text = indicator["text"]
        if not isinstance(led, QGraphicsEllipseItem) or not isinstance(text, QGraphicsSimpleTextItem):
            return
        model_path = str(module.params.get("model_path", "")).strip()
        dim = QColor("#3a3f45")
        if module.status == ModuleStatus.READY:
            led_color = QColor("#39d353")
        elif module.status == ModuleStatus.LOADING:
            led_color = QColor("#d6b745")
        elif module.status == ModuleStatus.ERROR:
            led_color = QColor("#e24c4c")
        elif module.status == ModuleStatus.RUNNING and model_path:
            led_color = QColor("#d68c1a")
        else:
            led_color = dim
        led.setBrush(QBrush(led_color))
        text.setBrush(QBrush(led_color if led_color != dim else QColor("#606870")))

    def _refresh_inference_log(self, module_id: str) -> None:
        rows = self._log_displays.get(module_id)
        brain_log = self._brain_logs.get(module_id)
        if rows is None or brain_log is None:
            return
        lines = brain_log.render_lines()
        for i, row_item in enumerate(rows):
            if i < len(lines):
                text, color = lines[i]
                row_item.setPlainText(text)
                row_item.setDefaultTextColor(color)
            else:
                row_item.setPlainText("")
                row_item.setDefaultTextColor(_LOG_DIM)

    def _refresh_model_info(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        info = self._model_info_items.get(module_id)
        if module is None or info is None:
            return
        model_path = str(module.params.get("model_path", "")).strip()
        if model_path:
            meta = self._parse_gguf_filename(model_path)
        else:
            meta = {"name": "\u2014", "quant": "\u2014", "param_size": "\u2014"}

        name_item = info.get("name")
        if isinstance(name_item, QGraphicsSimpleTextItem):
            name_item.setText(meta["name"][:28])
            name_item.setToolTip(meta["name"])
        quant_item = info.get("quant")
        if isinstance(quant_item, QGraphicsSimpleTextItem):
            quant_item.setText(meta["quant"])
        size_item = info.get("param_size")
        if isinstance(size_item, QGraphicsSimpleTextItem):
            size_item.setText(meta["param_size"])
        ctx_item = info.get("native_ctx")
        if isinstance(ctx_item, QGraphicsSimpleTextItem):
            native = int(module.params.get("_native_ctx", 0))
            trained = int(module.params.get("_trained_ctx", 0))
            if native > 0 and trained > 0:
                ctx_item.setText(f"{native}/{trained}")
                ctx_item.setToolTip(f"active context / model supported context: {native} / {trained}")
            elif native > 0:
                ctx_item.setText(str(native))
                ctx_item.setToolTip("active context window")
            elif trained > 0:
                ctx_item.setText(f"\u2014/{trained}")
                ctx_item.setToolTip(f"model supported context: {trained}")
            else:
                ctx_item.setText("\u2014")
                ctx_item.setToolTip("")
        mmproj_item = info.get("mmproj")
        if isinstance(mmproj_item, QGraphicsSimpleTextItem):
            mmproj_path = str(module.params.get("mmproj_path", "")).strip()
            active_mmproj = str(module.params.get("_active_mmproj_path", "")).strip()
            path = active_mmproj or mmproj_path
            if path:
                mmproj_item.setText(Path(path).name[:28])
                mmproj_item.setToolTip(path)
            else:
                mmproj_item.setText("\u2014")
                mmproj_item.setToolTip("")

        tools_led = info.get("tools_led")
        if isinstance(tools_led, QGraphicsEllipseItem):
            has_tools = bool(module.params.get("_tool_use", False))
            tools_led.setBrush(
                QBrush(QColor("#39d353") if has_tools else QColor("#3a3f45"))
            )
            tools_label = info.get("tools_label")
            if isinstance(tools_label, QGraphicsSimpleTextItem):
                tools_label.setBrush(
                    QBrush(QColor("#39d353") if has_tools else QColor("#606870"))
                )
        vision_led = info.get("vision_led")
        if isinstance(vision_led, QGraphicsEllipseItem):
            has_vision = bool(module.params.get("_vision_handler_name", ""))
            vision_led.setBrush(
                QBrush(QColor("#39d353") if has_vision else QColor("#3a3f45"))
            )
            vision_label = info.get("vision_label")
            if isinstance(vision_label, QGraphicsSimpleTextItem):
                vision_label.setBrush(
                    QBrush(QColor("#39d353") if has_vision else QColor("#606870"))
                )

    @staticmethod
    def _parse_gguf_filename(filepath: str) -> dict[str, str]:
        stem = Path(filepath).stem
        quant_m = _QUANT_RE.search(stem)
        quant = quant_m.group(1).upper() if quant_m else "\u2014"
        param_m = _PARAM_SIZE_RE.search(stem)
        param_size = f"{param_m.group(1)}B" if param_m else "\u2014"
        clean = _QUANT_RE.sub("", stem)
        clean = _PARAM_SIZE_RE.sub("", clean)
        clean = re.sub(r"[-_.]+$|^[-_.]+", "", clean).strip("-_. ")
        return {"name": clean or stem, "quant": quant, "param_size": param_size}

    # ── Model selector helpers ─────────────────────────────────

    def _resolve_folder(self, module_id: str) -> Path:
        folders = self._resolve_folders(module_id)
        return folders[0] if folders else Path(".")

    def _resolve_folders(self, module_id: str) -> list[Path]:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return [Path(".")]

        roots_raw = module.params.get("model_roots", [])
        roots: list[str] = []
        if isinstance(roots_raw, str):
            txt = roots_raw.strip()
            if txt:
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, list):
                        roots = [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    roots = []
        elif isinstance(roots_raw, list):
            roots = [str(v).strip() for v in roots_raw if str(v).strip()]

        if not roots:
            roots = [str(module.params.get("model_root", ".")).strip() or "."]

        out: list[Path] = []
        seen: set[str] = set()
        for raw in roots:
            p = Path(raw)
            if not p.is_absolute():
                p = Path.cwd() / p
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out or [Path(".")]

    @staticmethod
    def _pair_models(folders: list[Path]) -> list[ModelEntry]:
        all_files: list[Path] = []
        seen: set[str] = set()

        for folder in folders:
            if not folder.exists() or not folder.is_dir():
                continue
            try:
                ggufs = folder.rglob("*.gguf")
            except Exception:
                continue

            for entry in ggufs:
                if not entry.is_file():
                    continue
                try:
                    key = str(entry.resolve()).lower()
                except Exception:
                    key = str(entry).lower()
                if key not in seen:
                    all_files.append(entry)
                    seen.add(key)

        mmproj_files = [p for p in all_files if "mmproj" in p.name.lower()]
        main_models = [p for p in all_files if "mmproj" not in p.name.lower()]

        def _mmproj_rank(model_file: Path, candidate: Path) -> tuple[int, int, str]:
            name = candidate.name.lower()
            if "f16" in name:
                pref = 0
            elif "bf16" in name:
                pref = 1
            else:
                pref = 2

            if candidate.parent == model_file.parent:
                near = 0
            elif candidate.parent == model_file.parent.parent:
                near = 1
            elif candidate.parent.parent == model_file.parent.parent:
                near = 2
            else:
                near = 3
            return (near, pref, name)

        def _is_same_project(model_file: Path, candidate: Path) -> bool:
            model_parent = model_file.parent
            cand_parent = candidate.parent
            if cand_parent == model_parent:
                return True
            if cand_parent == model_parent.parent:
                return True
            if cand_parent.parent == model_parent.parent:
                return True
            return False

        def _family_tokens(path: Path) -> set[str]:
            stem = path.stem.lower().replace("mmproj", " ")
            tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
            stop = {
                "gguf",
                "instruct",
                "instruction",
                "chat",
                "it",
                "model",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q8",
                "f16",
                "f32",
                "bf16",
                "fp16",
                "fp32",
                "k",
                "m",
                "s",
                "xs",
                "xxs",
                "small",
                "base",
                "large",
            }
            return {t for t in tokens if t not in stop and not t.isdigit()}

        def _same_family(model_file: Path, candidate: Path) -> bool:
            model_tokens = _family_tokens(model_file)
            mmproj_tokens = _family_tokens(candidate)
            return bool(model_tokens & mmproj_tokens)

        entries: list[ModelEntry] = []
        for model_path in sorted(main_models, key=lambda p: p.name.lower()):
            mmproj: Path | None = None
            if mmproj_files:
                candidates = [
                    p
                    for p in mmproj_files
                    if _is_same_project(model_path, p)
                    and (p.parent == model_path.parent or _same_family(model_path, p))
                ]
                if candidates:
                    ranked = sorted(candidates, key=lambda p: _mmproj_rank(model_path, p))
                    mmproj = ranked[0]
            entries.append(
                ModelEntry(
                    model_path=model_path,
                    mmproj_path=mmproj,
                    display_name=model_path.stem,
                    vision=mmproj is not None,
                )
            )

        return entries

    @staticmethod
    def _tail_path(path: Path, max_len: int = 30) -> str:
        s = str(path)
        if len(s) <= max_len:
            return s
        return "..." + s[-(max_len - 3):]

    def _refresh_model_list(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._model_list_items.get(module_id)
        if module is None or items is None:
            return
        folders = self._resolve_folders(module_id)
        entries = self._pair_models(folders)
        self._model_entries[module_id] = entries
        selected = str(module.params.get("model_path", "")).strip()

        folder_item = self._folder_items.get(module_id)
        if isinstance(folder_item, QGraphicsSimpleTextItem):
            if len(folders) > 1:
                folder_item.setText(f"({len(folders)} FOLDERS)")
                folder_item.setToolTip("\n".join(str(p) for p in folders))
            else:
                folder_item.setText(f"({self._tail_path(folders[0], max_len=24)})")
                folder_item.setToolTip(str(folders[0]))

        offset = self._scroll_offsets.get(module_id, 0)
        max_offset = max(0, len(entries) - self._VISIBLE_MODEL_ROWS)
        offset = max(0, min(offset, max_offset))
        self._scroll_offsets[module_id] = offset

        visible = entries[offset : offset + self._VISIBLE_MODEL_ROWS]
        for ridx, row in enumerate(items):
            if not isinstance(row, dict):
                continue
            entry = visible[ridx] if ridx < len(visible) else None
            sel_btn = row.get("sel_btn")
            file_txt = row.get("file_txt")

            if isinstance(file_txt, QGraphicsSimpleTextItem):
                if entry is not None:
                    prefix = "◉ " if entry.vision else ""
                    file_txt.setText((prefix + entry.display_name)[:60])
                    if entry.vision and entry.mmproj_path is not None:
                        file_txt.setToolTip(
                            f"model: {entry.model_path}\nmmproj: {entry.mmproj_path}"
                        )
                        file_txt.setBrush(QBrush(QColor("#a5b6d9")))
                    else:
                        file_txt.setToolTip(str(entry.model_path))
                        file_txt.setBrush(QBrush(QColor("#d6dbe0")))
                else:
                    file_txt.setText("")
                    file_txt.setToolTip("")
                    file_txt.setBrush(QBrush(QColor("#4d5258")))

            if isinstance(sel_btn, CardButtonItem):
                checked = entry is not None and str(entry.model_path) == selected
                self._style_select_button(sel_btn, checked, enabled=(entry is not None))

    def _style_select_button(
        self, btn: CardButtonItem, checked: bool, enabled: bool,
    ) -> None:
        if not enabled:
            btn.setBrush(QBrush(QColor("#0c0d0e")))
            btn.setPen(QPen(QColor("#3a3e44"), 0.8))
            text, color = "", QColor("#4a4f55")
        elif checked:
            btn.setBrush(QBrush(QColor("#d9dee3")))
            btn.setPen(QPen(QColor("#aeb7c2"), 0.8))
            text, color = "o", QColor("#0b0d10")
        else:
            btn.setBrush(QBrush(QColor("#0d0e10")))
            btn.setPen(QPen(QColor("#7d858f"), 0.8))
            text, color = " ", QColor("#d6dbe0")
        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setText(text)
                child.setBrush(QBrush(color))

    def _browse_folder(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        base = self._resolve_folder(module_id)
        chosen = QFileDialog.getExistingDirectory(
            self._controller,
            "Select GGUF model folder",
            str(base if base.exists() else base.parent),
        )
        if chosen:
            module.params["model_root"] = chosen
            module.params["model_roots"] = [chosen]
            module.params["model_path"] = ""
            self._scroll_offsets[module_id] = 0
            self._refresh_model_list(module_id)
            self._refresh_param_text(module_id)

    def _add_folder(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        folders = self._resolve_folders(module_id)
        base = folders[0] if folders else Path(".")
        chosen = QFileDialog.getExistingDirectory(
            self._controller,
            "Add GGUF model folder",
            str(base if base.exists() else base.parent),
        )
        if not chosen:
            return
        new_path = str(Path(chosen).resolve())
        roots = [str(p.resolve()) for p in folders]
        if new_path not in roots:
            roots.append(new_path)
        module.params["model_roots"] = roots
        module.params["model_root"] = roots[0] if roots else new_path
        self._scroll_offsets[module_id] = 0
        self._refresh_model_list(module_id)
        self._refresh_param_text(module_id)

    def _scroll_model_list(self, module_id: str, delta: int) -> None:
        offset = self._scroll_offsets.get(module_id, 0)
        self._scroll_offsets[module_id] = max(0, offset + delta)
        self._refresh_model_list(module_id)

    def _select_visible_row(self, module_id: str, visible_row_index: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        entries = self._model_entries.get(module_id, [])
        if not entries:
            entries = self._pair_models(self._resolve_folders(module_id))
            self._model_entries[module_id] = entries
        offset = self._scroll_offsets.get(module_id, 0)
        idx = offset + visible_row_index
        if idx < 0 or idx >= len(entries):
            return
        entry = entries[idx]
        module.params["model_path"] = str(entry.model_path)
        module.params["mmproj_path"] = str(entry.mmproj_path) if entry.mmproj_path else ""
        self._refresh_model_list(module_id)
        self._refresh_param_text(module_id)

    def _auto_suggest_gpu_layers(self, module_id: str) -> None:
        """One-shot: set gpu_layers=-1 (all layers to GPU) when Vulkan is detected."""
        if module_id in self._gpu_suggested:
            return
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        if int(module.params.get("gpu_layers", 0)) != 0:
            self._gpu_suggested.add(module_id)
            return
        hv_panel = getattr(self._controller, "hv_panel", None)
        if hv_panel is None:
            return
        hw = getattr(hv_panel, "_hardware", None)
        if hw is None:
            return
        self._gpu_suggested.add(module_id)
        from core.hardware_probe import AiBackend
        if hw.ai_backend_hint == AiBackend.VULKAN:
            module.params["gpu_layers"] = -1
            if int(module.params.get("n_batch", 512)) == 512:
                module.params["n_batch"] = 256
            self._controller.hv_log.info(
                f"{module_id}: auto-set gpu_layers=-1 (ALL), n_batch=256 (Vulkan detected)"
            )

    # ── Parameter adjustment ──────────────────────────────────────

    def _adjust_param(self, module_id: str, key: str, direction: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return

        if key == "gpu_layers":
            cur = int(module.params.get("gpu_layers", -1))
            module.params[key] = max(-1, cur + direction)
        elif key == "ctx_length":
            ctx_options = [2048, 4096, 8192, 16384, 32768]
            cur = int(module.params.get("ctx_length", 4096))
            idx = ctx_options.index(cur) if cur in ctx_options else 1
            idx = max(0, min(len(ctx_options) - 1, idx + direction))
            module.params[key] = ctx_options[idx]
        elif key == "n_batch":
            batch_options = [128, 256, 512, 1024, 2048]
            cur = int(module.params.get("n_batch", 512))
            idx = batch_options.index(cur) if cur in batch_options else 2
            idx = max(0, min(len(batch_options) - 1, idx + direction))
            module.params[key] = batch_options[idx]
        elif key == "cpu_threads":
            cur = int(module.params.get("cpu_threads", 4))
            module.params[key] = max(1, cur + direction)
        elif key == "seed":
            cur = int(module.params.get("seed", -1))
            module.params[key] = max(-1, cur + direction)
        elif key == "temperature":
            cur = float(module.params.get("temperature", 0.7))
            module.params[key] = max(0.0, min(2.0, round(cur + 0.1 * direction, 2)))
        elif key == "top_p":
            cur = float(module.params.get("top_p", 0.95))
            module.params[key] = max(0.0, min(1.0, round(cur + 0.05 * direction, 2)))
        elif key == "top_k":
            cur = int(module.params.get("top_k", 40))
            module.params[key] = max(0, cur + 5 * direction)
        elif key == "repeat_penalty":
            cur = float(module.params.get("repeat_penalty", 1.1))
            module.params[key] = max(0.0, min(2.0, round(cur + 0.05 * direction, 2)))
        elif key == "flash_attn":
            _fa_cycle = ["auto", True, False]
            cur = module.params.get("flash_attn", "auto")
            try:
                idx = _fa_cycle.index(cur)
            except ValueError:
                idx = 0
            idx = (idx + direction) % len(_fa_cycle)
            module.params[key] = _fa_cycle[idx]
        elif key == "cache_type_k":
            _cache_cycle = ["f16", "q8_0", "q4_1", "q4_0", "f32"]
            cur = str(module.params.get("cache_type_k", "q8_0")).lower()
            idx = _cache_cycle.index(cur) if cur in _cache_cycle else 1
            idx = (idx + direction) % len(_cache_cycle)
            selected = _cache_cycle[idx]
            module.params["cache_type_k"] = selected
            module.params["cache_type_v"] = selected
        elif key == "vulkan_device":
            cur = int(module.params.get("vulkan_device", 0))
            module.params[key] = max(0, cur + direction)

        self._refresh_param_text(module_id)
        if key == "cache_type_k":
            self._controller.hv_log.info(
                f"{module_id}: set kv_cache={module.params['cache_type_k']}/{module.params['cache_type_v']} "
                f"(apply with LOAD)"
            )
        else:
            self._controller.hv_log.info(
                f"{module_id}: set {key}={module.params[key]} (apply with LOAD)"
            )

    # ── Vulkan diagnostics ─────────────────────────────────────────

    def _run_vulkan_diagnostics(self, module_id: str) -> None:
        """Run Vulkan driver diagnostics and write results to the hypervisor log."""
        import ctypes.util
        import pathlib
        import sys

        log = self._controller.hv_log

        from core.hardware_probe import (
            AiBackend,
            check_driver_compatibility,
        )
        from core.probe_service import get_probe_service

        hw = get_probe_service().hardware()
        log.info(f"{module_id}: ── Vulkan Diagnostics ──────────────────")
        log.info(f"{module_id}: GPU    : {hw.gpu_name or 'Unknown'} ({hw.gpu_vendor.value})")
        log.info(f"{module_id}: Driver : {hw.gpu_driver_version or 'Unknown'}")
        log.info(f"{module_id}: OS     : {hw.os_info}")

        # Driver version check against known Vulkan 1.2 minimums
        if hw.gpu_driver_version:
            ok, warning = check_driver_compatibility(
                hw.gpu_vendor, AiBackend.VULKAN, hw.gpu_driver_version
            )
            if ok:
                log.info(f"{module_id}: Driver : meets Vulkan 1.2 minimum")
            else:
                log.warn(f"{module_id}: Driver : {warning}")
        else:
            log.warn(f"{module_id}: Driver : version not detected — cannot verify")

        # vulkan-1.dll presence (Windows only)
        if sys.platform == "win32":
            vk_dll = ctypes.util.find_library("vulkan-1")
            if vk_dll:
                log.info(f"{module_id}: vulkan-1.dll : FOUND ({vk_dll})")
            else:
                log.error(
                    f"{module_id}: vulkan-1.dll : NOT FOUND — "
                    "install or update your GPU drivers"
                )

        # ggml bin/ directory contents
        try:
            from core.backend_downloader import get_vendor_dir
            import llama_cpp._ggml as _ggml_mod
            bin_dir = Path(_ggml_mod.__file__).resolve().parent / "bin"
            vendor_dir = get_vendor_dir()
            if bin_dir.is_dir():
                dlls = sorted(
                    f.name for f in bin_dir.iterdir()
                    if f.suffix in (".dll", ".so")
                )
                log.info(
                    f"{module_id}: bin/ ({len(dlls)} DLLs) : "
                    + ", ".join(dlls[:8])
                    + (" …" if len(dlls) > 8 else "")
                )
                has_vulkan_dll = any("vulkan" in n.lower() for n in dlls)
                if has_vulkan_dll:
                    log.info(f"{module_id}: ggml-vulkan   : present in bin/")
                else:
                    log.warn(
                        f"{module_id}: ggml-vulkan   : missing from bin/ "
                        "(vendor/ is the runtime source of truth)"
                    )
            else:
                log.info(
                    f"{module_id}: bin/ dir not found at {bin_dir} "
                    "(expected in some frozen/vendor-first builds)"
                )

            if vendor_dir.is_dir():
                vendor_dlls = sorted(
                    f.name for f in vendor_dir.iterdir()
                    if f.suffix in (".dll", ".so")
                )
                log.info(
                    f"{module_id}: vendor/ ({len(vendor_dlls)} DLLs) : "
                    + ", ".join(vendor_dlls[:8])
                    + (" …" if len(vendor_dlls) > 8 else "")
                )
                has_vulkan_vendor = any("vulkan" in n.lower() for n in vendor_dlls)
                if has_vulkan_vendor:
                    log.info(f"{module_id}: ggml-vulkan   : present in vendor/")
                else:
                    log.warn(f"{module_id}: ggml-vulkan   : missing from vendor/")
            else:
                log.warn(f"{module_id}: vendor/ dir not found at {vendor_dir}")
        except Exception as exc:
            log.error(f"{module_id}: llama_cpp._ggml import failed: {exc}")

        # GPU offload support reported by the loaded library
        try:
            import llama_cpp  # type: ignore
            supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            if callable(supports_gpu):
                gpu_ok = bool(supports_gpu())
                if gpu_ok:
                    log.info(f"{module_id}: GPU offload   : YES (backends loaded)")
                else:
                    log.warn(
                        f"{module_id}: GPU offload   : NO — "
                        "Vulkan backend may not have loaded; check driver and bin/"
                    )
            else:
                log.warn(
                    f"{module_id}: GPU offload   : llama_supports_gpu_offload "
                    "not available in this build"
                )
        except Exception as exc:
            log.error(f"{module_id}: llama_cpp not importable: {exc}")

        log.info(f"{module_id}: ─────────────────────────────────────────")
        self._controller.statusBar().showMessage(
            "Vulkan diagnostics complete — see log panel", 4000
        )

    # ── Backend config ─────────────────────────────────────────────

    def _open_backend_config(self, module_id: str) -> None:
        callback = getattr(self._controller, '_on_run_backend_config', None)
        if not callable(callback):
            return
        selected = callback()
        if not selected:
            return
        persist_callback = getattr(
            self._controller, "_on_persist_backend_selection", None
        )
        if callable(persist_callback):
            selected = persist_callback(selected) or selected
        else:
            self._controller.hypervisor.set_stored_backend(selected)
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is not None:
            module.params["_ai_backend"] = selected
        self._controller.hv_log.info(
            f"{module_id}: backend set to {selected.upper()} — restart module to apply"
        )
        self._controller.statusBar().showMessage(
            f"Backend set to {selected.upper()} — restart module to apply", 4000
        )

