"""Rack chassis window — module cards, wiring, and module tray drag-and-drop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from core.hypervisor import Hypervisor
    from core.port_system import Wire

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.base_module import ModuleStatus
from core.port_system import ConnectionMode, PortDirection
from core.probe_service import get_probe_service
from ui.fan_item import SpinningFanItem
from ui.hypervisor_panel import HypervisorLog, HypervisorPanelItem
from ui.mcp_slot_config_dialog import run_mcp_slot_config_dialog
from ui.module_tray import MODULE_TRAY_MIME, ModuleTray
from ui.port_tooltip import PortTooltipWidget
from ui.label_aliases import module_ui_label, port_faceplate_label, port_ui_label
from ui.rack_card_layout import CARD_BUILDERS, compute_card_height
from ui.rack_items import (
    CardButtonItem,
    DragGripItem,
    ModuleCardItem,
    PendingWire,
    PortJackItem,
    PortLabelItem,
    WirePathItem,
    _draw_screw,
    _port_color,
    _status_color,
    _wire_color,
)
from ui.rack_scene import RackScene
from ui.wire_renderer import compute_bezier_points


class RackWindow(QMainWindow):
    RACK_RAIL_W = 18.0
    RACK_COLS = 4
    CARD_W = 260.0
    CARD_GAP_X = 0.0
    BRACKET_H = 18.0
    GAP_PREVIEW_DELAY_MS = 80
    GAP_PREVIEW_ANIM_MS = 180
    AUTOSCROLL_ZONE_PX = 60       # px from viewport top/bottom that activates scroll
    AUTOSCROLL_MAX_SPEED_PX = 14  # pixels scrolled per tick at the extreme edge
    AUTOSCROLL_INTERVAL_MS = 16   # ~60 fps
    ZOOM_MIN = 0.2
    ZOOM_MAX = 4.0
    ZOOM_STEP = 1.15
    # Left/right module tray: prior implicit width ~300px; default opening width 60% narrower.
    _MODULE_TRAY_REF_SIDE_WIDTH = 300
    MODULE_TRAY_SIDE_DEFAULT_WIDTH = max(
        120,
        int(round(_MODULE_TRAY_REF_SIDE_WIDTH * (1.0 - 0.60))),
    )
    MODULE_TITLE_FONT_SIZE = 11  # +2pt from prior 9pt prototype
    PORT_LABEL_FONT_SIZE = 10    # reduced by 1pt
    PORT_LABEL_FONT_WEIGHT = QFont.Weight.Medium
    MODULE_TITLE_COLOR = QColor("#b8cfe8")
    PORT_LABEL_BASE_COLOR = QColor("#546878")

    def __init__(
        self,
        hypervisor: Hypervisor,
        module_plugins: list[dict[str, Any]] | None = None,
        module_registry: dict[str, type] | None = None,
    ) -> None:
        super().__init__()
        self.hypervisor = hypervisor
        self.module_registry: dict[str, type] = module_registry or {}
        self.hv_log = HypervisorLog()
        self.setWindowTitle("Miniloader \u2014 Rack")

        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.setMinimumSize(int(geo.width() * 0.25), int(geo.height() * 0.25))
            self.resize(min(1400, geo.width()), min(900, geo.height()))
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2,
            )
        else:
            self.setMinimumSize(640, 400)
            self.resize(1400, 900)

        self.scene = RackScene(self)
        self.scene.setBackgroundBrush(QColor("#121316"))
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.view.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setAcceptDrops(True)
        self.view.viewport().setAcceptDrops(True)
        self.view.viewport().installEventFilter(self)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setCentralWidget(self.view)
        self._zoom_level = 1.0
        self._build_zoom_overlay()
        self._build_transition_overlay()
        self._manual_transition_message: str | None = None

        self.port_items: dict[str, PortJackItem] = {}
        self._port_label_items: dict[str, PortLabelItem] = {}
        self._port_label_base_colors: dict[str, QColor] = {}
        self._port_label_hover_colors: dict[str, QColor] = {}
        self.module_leds: dict[str, QGraphicsEllipseItem] = {}
        self.wire_items: dict[str, WirePathItem] = {}
        self.pending_wire: PendingWire | None = None
        self.pending_wire_item: QGraphicsPathItem | None = None
        self._power_buttons: dict[str, CardButtonItem] = {}
        self._init_buttons: dict[str, CardButtonItem] = {}

        self._builders: dict[str, Any] = {
            name: cls(self) for name, cls in CARD_BUILDERS.items()
        }

        self._fan_items: list[SpinningFanItem] = []
        self._module_order: list[str] = list(hypervisor.active_modules.keys())
        self._card_rects: dict[str, QRectF] = {}
        self._card_items: dict[str, ModuleCardItem] = {}
        self._dragging_mid: str | None = None
        self._preview_insert_idx: int | None = None
        self._pending_preview_idx: int | None = None
        self._card_anim_start: dict[str, QPointF] = {}
        self._card_anim_target: dict[str, QPointF] = {}
        self._gap_delay_timer = QTimer(self)
        self._gap_delay_timer.setSingleShot(True)
        self._gap_delay_timer.setInterval(self.GAP_PREVIEW_DELAY_MS)
        self._gap_delay_timer.timeout.connect(self._commit_gap_preview)
        self._gap_anim = QVariantAnimation(self)
        self._gap_anim.setDuration(self.GAP_PREVIEW_ANIM_MS)
        self._gap_anim.setStartValue(0.0)
        self._gap_anim.setEndValue(1.0)
        self._gap_anim.valueChanged.connect(self._on_gap_anim_tick)
        self._autoscroll_timer = QTimer(self)
        self._autoscroll_timer.setInterval(self.AUTOSCROLL_INTERVAL_MS)
        self._autoscroll_timer.timeout.connect(self._on_autoscroll_tick)
        self._autoscroll_dy: int = 0
        self._tray_drag_name: str | None = None
        self._tray_drag_span: int = 1
        self._tray_drag_height: float | None = None
        self._module_register_fns: dict[str, Callable[[Any], None]] = {}
        for plugin in module_plugins or []:
            name = plugin.get("name")
            register_fn = plugin.get("register_fn")
            if isinstance(name, str) and callable(register_fn):
                self._module_register_fns[name] = register_fn
        self._hidden_placeable_modules: set[str] = {"mcp_bus"}

        self._module_tray = ModuleTray(
            module_plugins or [],
            self.module_registry,
            hidden_module_names=self._hidden_placeable_modules,
            parent=self,
        )
        self._module_tray_dock = QDockWidget("Modules", self)
        self._module_tray_dock.setWidget(self._module_tray)
        self._module_tray_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._module_tray_dock)
        screen_geo = screen.availableGeometry() if screen is not None else None
        if screen_geo is not None:
            self._module_tray_bottom_max_height = max(180, int(screen_geo.height() * 0.25))
        else:
            self._module_tray_bottom_max_height = 240
        self._apply_module_tray_dock_constraints(Qt.DockWidgetArea.LeftDockWidgetArea)
        self._module_tray_dock.dockLocationChanged.connect(self._apply_module_tray_dock_constraints)
        self._module_tray_dock.visibilityChanged.connect(self._sync_module_tray_button_text)

        self._build_hypervisor_panel()
        self._emit_boot_log()
        self._build_module_cards()
        self._render_existing_wires()
        self._start_status_refresh()
        self._start_hardware_poll()
        self._start_fan_timer()

        self._tooltip_jack: PortJackItem | None = None
        self._tooltip_widget = PortTooltipWidget(self.view.viewport())
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.timeout.connect(self._fire_port_tooltip)

        _status_btn_style = (
            "QPushButton {"
            "  background: #1b2433; color: #b7c8e8; border: 1px solid #324661;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #233146; }"
        )
        _toggle_checked_style = (
            "QPushButton {"
            "  background: #1b2433; color: #b7c8e8; border: 1px solid #324661;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #233146; }"
            "QPushButton:checked {"
            "  background: #1a3a2a; color: #6fcf97; border-color: #2e7d52;"
            "}"
        )

        self._auto_wire_toggle = QPushButton("Auto-Wire")
        self._auto_wire_toggle.setCheckable(True)
        self._auto_wire_toggle.setChecked(False)
        self._auto_wire_toggle.setToolTip(
            "When active, newly placed modules are automatically wired"
        )
        self._auto_wire_toggle.setStyleSheet(_toggle_checked_style)
        self.statusBar().addPermanentWidget(self._auto_wire_toggle)

        self._wire_all_btn = QPushButton("Wire All")
        self._wire_all_btn.setToolTip(
            "Attempt to auto-wire every unconnected port on the rack"
        )
        self._wire_all_btn.setStyleSheet(_status_btn_style)
        self._wire_all_btn.clicked.connect(
            lambda: asyncio.create_task(self._auto_wire_all())
        )
        self.statusBar().addPermanentWidget(self._wire_all_btn)

        self._delete_all_wires_btn = QPushButton("Delete All Wires")
        self._delete_all_wires_btn.setToolTip(
            "Remove every wire from the rack (powers rack off first if running)"
        )
        self._delete_all_wires_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1b2433; color: #cf6679; border: 1px solid #4a2030;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #2a1a22; }"
        )
        self._delete_all_wires_btn.clicked.connect(
            lambda: asyncio.create_task(self._delete_all_wires())
        )
        self.statusBar().addPermanentWidget(self._delete_all_wires_btn)

        self._module_tray_toggle_btn = QPushButton("Hide Modules Tray")
        self._module_tray_toggle_btn.setStyleSheet(_status_btn_style)
        self._module_tray_toggle_btn.clicked.connect(self._toggle_module_tray_visibility)
        self.statusBar().addPermanentWidget(self._module_tray_toggle_btn)
        self._sync_module_tray_button_text(self._module_tray_dock.isVisible())
        self.statusBar().showMessage("Drag from an OUT/CHANNEL port to an IN/CHANNEL port")

    def add_runtime_module(
        self,
        module_name: str,
        register_fn: Callable[[Any], None],
        module_cls: type | None = None,
    ) -> None:
        """Register a newly installed module for immediate tray deployment."""
        name = str(module_name or "").strip()
        if not name or not callable(register_fn):
            return
        self._module_register_fns[name] = register_fn
        if module_cls is not None:
            self.module_registry[name] = module_cls
        self._module_tray.add_module(name, register_fn, module_cls)
        self.statusBar().showMessage(f"Installed module available: {name}", 4000)

    def _build_zoom_overlay(self) -> None:
        self._zoom_overlay = QWidget(self)
        self._zoom_overlay.setStyleSheet(
            "QWidget { background: rgba(12, 16, 24, 220); border: 1px solid #324661; border-radius: 8px; }"
            "QPushButton {"
            "  background: #1b2433; color: #b7c8e8; border: 1px solid #324661;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px; min-width: 26px;"
            "}"
            "QPushButton:hover { background: #233146; }"
            "QPushButton:disabled { color: #5b6a80; border-color: #2a3038; background: #141a24; }"
            "QLabel { color: #9eb4d8; border: none; padding: 0 4px; }"
        )

        layout = QHBoxLayout(self._zoom_overlay)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(6)

        self._zoom_out_btn = QPushButton("-", self._zoom_overlay)
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QPushButton("+", self._zoom_overlay)
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        layout.addWidget(self._zoom_in_btn)

        self._zoom_fit_btn = QPushButton("Fit", self._zoom_overlay)
        self._zoom_fit_btn.setToolTip("Fit rack to view")
        self._zoom_fit_btn.clicked.connect(self._zoom_fit)
        layout.addWidget(self._zoom_fit_btn)

        self._zoom_label = QLabel("100%", self._zoom_overlay)
        self._zoom_label.setToolTip("Current zoom")
        layout.addWidget(self._zoom_label)

        self._zoom_overlay.adjustSize()
        self._position_zoom_overlay()
        self._refresh_zoom_ui()

    def _build_transition_overlay(self) -> None:
        self._transition_overlay = QWidget(self)
        self._transition_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._transition_overlay.setStyleSheet(
            "QWidget { background: rgba(8, 10, 14, 205); }"
            "QLabel { color: #9ab0cc; font-size: 12px; font-weight: 600; background: transparent; }"
            "QProgressBar {"
            "  border: 1px solid #2f415a;"
            "  border-radius: 7px;"
            "  background: #121a25;"
            "  height: 12px;"
            "}"
            "QProgressBar::chunk {"
            "  background: #4f86c6;"
            "  width: 20px;"
            "  margin: 1px;"
            "}"
        )
        self._transition_overlay.hide()

        layout = QVBoxLayout(self._transition_overlay)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self._transition_label = QLabel("Applying rack power state...")
        self._transition_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._transition_label)

        self._transition_spinner = QProgressBar(self._transition_overlay)
        self._transition_spinner.setFixedWidth(220)
        self._transition_spinner.setTextVisible(False)
        self._transition_spinner.setRange(0, 0)
        layout.addWidget(self._transition_spinner)

    def _position_zoom_overlay(self) -> None:
        margin_x = 14
        right_safe_inset = 120
        margin_top = 14
        rect = self.rect()
        self._zoom_overlay.adjustSize()
        x = max(0, rect.width() - self._zoom_overlay.width() - margin_x - right_safe_inset)
        y = max(0, margin_top)
        self._zoom_overlay.move(x, y)
        self._zoom_overlay.raise_()

    def _sync_transition_overlay(self) -> None:
        if not hasattr(self, "hv_panel"):
            return
        manual_message = str(self._manual_transition_message or "").strip()
        manual_busy = bool(manual_message)
        busy_fn = getattr(self.hv_panel, "is_power_transition_busy", None)
        message_fn = getattr(self.hv_panel, "power_transition_message", None)
        power_busy = bool(busy_fn()) if callable(busy_fn) else False
        busy = manual_busy or power_busy
        if busy:
            if manual_busy:
                text = manual_message
            else:
                text = (
                    str(message_fn()).strip()
                    if callable(message_fn)
                    else "Applying rack power state..."
                )
            self._transition_label.setText(text or "Applying rack power state...")
            self._transition_overlay.setGeometry(0, 0, self.width(), self.height())
            self._transition_overlay.raise_()
            self._transition_overlay.show()
        else:
            self._transition_overlay.hide()

    def show_transition_overlay(self, message: str) -> None:
        """Show fullscreen transition overlay with a custom status message."""
        self._manual_transition_message = message.strip() or "Applying rack state..."
        self._sync_transition_overlay()

    def hide_transition_overlay(self) -> None:
        """Hide any custom transition overlay message."""
        self._manual_transition_message = None
        self._sync_transition_overlay()

    def _refresh_zoom_ui(self) -> None:
        self._zoom_level = self.view.transform().m11()
        pct = int(round(self._zoom_level * 100.0))
        self._zoom_label.setText(f"{pct}%")
        self._zoom_out_btn.setEnabled(self._zoom_level > self.ZOOM_MIN + 1e-6)
        self._zoom_in_btn.setEnabled(self._zoom_level < self.ZOOM_MAX - 1e-6)

    def _apply_zoom(
        self,
        factor: float,
        anchor: QGraphicsView.ViewportAnchor | None = None,
    ) -> None:
        if factor <= 0:
            return
        current = self.view.transform().m11()
        target = max(self.ZOOM_MIN, min(self.ZOOM_MAX, current * factor))
        if abs(target - current) < 1e-8:
            self._refresh_zoom_ui()
            return

        applied = target / current
        old_anchor = self.view.transformationAnchor()
        if anchor is not None:
            self.view.setTransformationAnchor(anchor)
        self.view.scale(applied, applied)
        if anchor is not None:
            self.view.setTransformationAnchor(old_anchor)
        self._refresh_zoom_ui()

    def _zoom_in(self) -> None:
        self._apply_zoom(self.ZOOM_STEP, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def _zoom_out(self) -> None:
        self._apply_zoom(1.0 / self.ZOOM_STEP, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def _zoom_fit(self) -> None:
        target_rect = self.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        if target_rect.isEmpty():
            return
        self.view.fitInView(target_rect, Qt.AspectRatioMode.KeepAspectRatio)
        current = self.view.transform().m11()
        if current < self.ZOOM_MIN:
            self._apply_zoom(self.ZOOM_MIN / current, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)
            return
        if current > self.ZOOM_MAX:
            self._apply_zoom(self.ZOOM_MAX / current, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)
            return
        self._refresh_zoom_ui()

    def _build_hypervisor_panel(self) -> None:
        rack_width = self.RACK_RAIL_W * 2 + self.RACK_COLS * self.CARD_W
        self.hv_panel = HypervisorPanelItem(rack_width, self.hypervisor, self.hv_log)
        self.hv_panel.setPos(0, 0)
        self.scene.addItem(self.hv_panel)

    def _toggle_module_tray_visibility(self) -> None:
        if self._module_tray_dock.isVisible():
            self._module_tray_dock.hide()
            return
        self._module_tray_dock.show()
        self._module_tray_dock.raise_()

    def _sync_module_tray_button_text(self, visible: bool) -> None:
        self._module_tray_toggle_btn.setText(
            "Hide Modules Tray" if visible else "Show Modules Tray"
        )

    def _apply_module_tray_dock_constraints(self, area: Qt.DockWidgetArea) -> None:
        if area == Qt.DockWidgetArea.BottomDockWidgetArea:
            self._module_tray_dock.setMaximumHeight(self._module_tray_bottom_max_height)
            self.resizeDocks(
                [self._module_tray_dock],
                [self._module_tray_bottom_max_height],
                Qt.Orientation.Vertical,
            )
            return
        # Side-docked: lift the bottom-only height cap and set a narrow default width.
        self._module_tray_dock.setMaximumHeight(16777215)
        if area in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ):
            self.resizeDocks(
                [self._module_tray_dock],
                [self.MODULE_TRAY_SIDE_DEFAULT_WIDTH],
                Qt.Orientation.Horizontal,
            )

    def _emit_boot_log(self) -> None:
        import threading

        def _probe_and_log():
            try:
                hw = get_probe_service().hardware()
                # QTimer.singleShot(0, ...) bridges the background thread to the main thread.
                QTimer.singleShot(0, lambda: self._apply_boot_hw(hw))
            except Exception as exc:
                QTimer.singleShot(
                    0, lambda e=exc: self.hv_log.warn(f"System probe unavailable: {e}")
                )

        threading.Thread(target=_probe_and_log, daemon=True).start()
        self.hv_log.info("Hypervisor initialized \u2014 hardware probe running in background")

    def _apply_boot_hw(self, hw) -> None:
        """Main-thread callback after the background hardware probe completes."""
        self.hv_panel.set_hardware_snapshot(hw)
        self.hv_log.info(
            f"System: CPU/RAM OK; GPU='{hw.gpu_name or 'UNKNOWN'}'; backend={hw.ai_backend_hint.value}"
        )
        if hw.vram_total_mb > 0:
            self.hv_log.info(
                f"VRAM: {hw.vram_used_mb:.0f}/{hw.vram_total_mb:.0f} MB"
            )
        try:
            for w in get_probe_service().verify_backend(hw.ai_backend_hint):
                self.hv_log.warn(w)
        except Exception:
            pass
        for mod in self.hypervisor.active_modules.values():
            self.hv_log.info(f"Registered module: {mod.MODULE_NAME} [{mod.module_id}]")
            if mod.PROCESS_ISOLATION:
                self.hv_log.worker(f"  \u2514\u2500 process isolation requested")
        _arrow = " \u2192 "
        self.hv_log.info(f"Boot order: {_arrow.join(self.hypervisor.topological_sort())}")
        for mod in self.hypervisor.active_modules.values():
            color_word = {
                ModuleStatus.RUNNING: "RUNNING",
                ModuleStatus.ERROR: "ERROR",
                ModuleStatus.LOADING: "LOADING",
            }.get(mod.status, mod.status.value.upper())
            if mod.status == ModuleStatus.ERROR:
                self.hv_log.error(f"{mod.MODULE_NAME}: {color_word}")
            elif mod.status == ModuleStatus.RUNNING:
                self.hv_log.info(f"{mod.MODULE_NAME}: {color_word}")
            else:
                self.hv_log.warn(f"{mod.MODULE_NAME}: {color_word}")
        workers = getattr(self.hypervisor, "_workers", {})
        for wid, handle in workers.items():
            pid = handle.process.pid if handle.process.is_alive() else "DEAD"
            self.hv_log.worker(f"Worker [{wid}] pid={pid}")
        self.hv_log.info(f"System ready \u2014 {len(self.hypervisor.active_modules)} modules loaded")

    def _start_fan_timer(self) -> None:
        timer = QTimer(self)
        timer.timeout.connect(self._advance_fans)
        timer.start(16)
        self._fan_timer = timer

    def _advance_fans(self) -> None:
        for fan in self._fan_items:
            fan.advance()

    def _start_hardware_poll(self) -> None:
        import threading
        self._hw_snapshot_pending = None
        self._hw_poll_lock = threading.Lock()

        timer = QTimer(self)
        timer.timeout.connect(self._poll_hardware_apply)
        timer.start(1500)
        self._hw_timer = timer
        threading.Thread(target=self._poll_hardware_bg, daemon=True).start()

    def _poll_hardware_bg(self) -> None:
        """Background thread: periodically probe hardware and stash the snapshot."""
        import time as _time
        while True:
            try:
                hw = get_probe_service().hardware_live()
                with self._hw_poll_lock:
                    self._hw_snapshot_pending = hw
            except Exception:
                pass
            _time.sleep(1.5)

    def _poll_hardware_apply(self) -> None:
        """Main thread: pick up the latest snapshot and push it to the UI."""
        with self._hw_poll_lock:
            hw = self._hw_snapshot_pending
        if hw is not None:
            self.hv_panel.set_hardware_snapshot(hw)

    def _rack_x_origin(self) -> float:
        return self.RACK_RAIL_W

    def _rack_total_w(self) -> float:
        return self.RACK_RAIL_W * 2 + self.RACK_COLS * self.CARD_W

    def _get_controls_height_for_module(
        self,
        module: BaseModule | None = None,
        module_name: str | None = None,
    ) -> float | None:
        name = module.MODULE_NAME if module is not None else module_name
        if not name:
            return None
        builder = self._builders.get(name)
        if builder is None:
            return None
        getter = getattr(builder, "get_controls_height", None)
        if callable(getter):
            try:
                module_id = module.module_id if module is not None else None
                return float(getter(module_id))
            except Exception:
                pass
        raw = getattr(builder, "CONTROLS_HEIGHT", None)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _rail_gradient(self, rail_x: float) -> QLinearGradient:
        g = QLinearGradient(rail_x, 0, rail_x + self.RACK_RAIL_W, 0)
        g.setColorAt(0.0,  QColor("#0d0e10"))
        g.setColorAt(0.18, QColor("#1a1b1f"))
        g.setColorAt(0.82, QColor("#131417"))
        g.setColorAt(1.0,  QColor("#0d0e10"))
        return g

    def _build_rack_chassis(self, total_h: float) -> None:
        w = self._rack_total_w()
        hv_h = HypervisorPanelItem.PANEL_HEIGHT

        for rail_x in (0.0, w - self.RACK_RAIL_W):
            rail = QGraphicsRectItem(rail_x, 0, self.RACK_RAIL_W, total_h)
            rail.setBrush(QBrush(self._rail_gradient(rail_x)))
            rail.setPen(QPen(QColor("#28292e"), 1.0))
            rail.setZValue(-1)
            self.scene.addItem(rail)

            cx = rail_x + self.RACK_RAIL_W / 2
            _draw_screw(self.scene, cx, hv_h + 9)
            _draw_screw(self.scene, cx, total_h - 9)

            y = hv_h + 30.0
            while y < total_h - 20:
                hole = QGraphicsEllipseItem(-4, -4, 8, 8)
                hole.setPos(cx, y)
                hole.setBrush(QBrush(QColor("#07080a")))
                hole.setPen(QPen(QColor("#232428"), 0.8))
                hole.setZValue(10)
                self.scene.addItem(hole)
                y += 26.0

    def _build_row_bracket(self, y: float, width: float) -> None:
        bracket = QGraphicsRectItem(self.RACK_RAIL_W, y, width, self.BRACKET_H)
        g = QLinearGradient(0, y, 0, y + self.BRACKET_H)
        g.setColorAt(0.0,  QColor("#0d0e10"))
        g.setColorAt(0.18, QColor("#1a1b1f"))
        g.setColorAt(0.82, QColor("#131417"))
        g.setColorAt(1.0,  QColor("#0d0e10"))
        bracket.setBrush(QBrush(g))
        bracket.setPen(QPen(QColor("#28292e"), 1.0))
        bracket.setZValue(-1)
        self.scene.addItem(bracket)

        screw_y = y + self.BRACKET_H / 2
        step = width / 6.0
        for i in range(1, 6):
            _draw_screw(self.scene, self.RACK_RAIL_W + i * step, screw_y)

    def _pack_modules(self) -> list[tuple[str, int, int, int]]:
        placements, _row_heights, _row_y, _total_h = self._compute_layout_for_order(
            self._module_order
        )
        return [
            (mid, row, col, span)
            for mid, row, col, span in placements
            if mid is not None
        ]

    def _compute_layout_for_order(
        self,
        order: list[str],
        gap_index: int | None = None,
        gap_span: int = 1,
        gap_height: float | None = None,
    ) -> tuple[list[tuple[str | None, int, int, int]], dict[int, float], dict[int, float], float]:
        active_order = [mid for mid in order if mid in self.hypervisor.active_modules]
        if gap_index is not None:
            gap_index = max(0, min(gap_index, len(active_order)))
            gap_span = max(1, min(gap_span, self.RACK_COLS))

        placements: list[tuple[str | None, int, int, int]] = []
        row = 0
        col = 0

        def _place(entry_mid: str | None, entry_span: int) -> None:
            nonlocal row, col
            if col + entry_span > self.RACK_COLS:
                row += 1
                col = 0
            placements.append((entry_mid, row, col, entry_span))
            col += entry_span

        for idx in range(len(active_order) + 1):
            if gap_index is not None and idx == gap_index:
                _place(None, gap_span)
            if idx == len(active_order):
                break
            mid = active_order[idx]
            mod = self.hypervisor.active_modules.get(mid)
            if mod is None:
                continue
            span = min(getattr(mod, "UI_COL_SPAN", 1), self.RACK_COLS)
            _place(mid, span)

        row_heights: dict[int, float] = {}
        for mid, packed_row, _col, _span in placements:
            if mid is None:
                if gap_height is not None:
                    row_heights[packed_row] = max(row_heights.get(packed_row, 0.0), gap_height)
                continue
            mod = self.hypervisor.active_modules.get(mid)
            if mod is None:
                continue
            port_rows = max(len(mod.inputs), len(mod.outputs), 1)
            h = compute_card_height(
                mod.MODULE_NAME,
                port_rows,
                controls_height=self._get_controls_height_for_module(module=mod),
            )
            row_heights[packed_row] = max(row_heights.get(packed_row, 0.0), h)

        num_rows = max(row_heights.keys(), default=-1) + 1
        row_y: dict[int, float] = {}
        cursor_y = HypervisorPanelItem.PANEL_HEIGHT
        for packed_row in range(num_rows):
            if packed_row > 0:
                cursor_y += self.BRACKET_H
            row_y[packed_row] = cursor_y
            cursor_y += row_heights[packed_row]
        return placements, row_heights, row_y, cursor_y

    def _compute_positions_for_order(
        self,
        order: list[str],
        gap_index: int | None = None,
        gap_span: int = 1,
        gap_height: float | None = None,
    ) -> dict[str, tuple[float, float]]:
        placements, _row_heights, row_y, _total_h = self._compute_layout_for_order(
            order, gap_index=gap_index, gap_span=gap_span, gap_height=gap_height
        )
        x0 = self._rack_x_origin()
        unit_w = self.CARD_W
        positions: dict[str, tuple[float, float]] = {}
        for mid, packed_row, col, _span in placements:
            if mid is None:
                continue
            positions[mid] = (x0 + col * unit_w, row_y[packed_row])
        return positions

    def _build_module_cards(self) -> None:
        placements, row_heights, row_y, total_h = self._compute_layout_for_order(self._module_order)
        cols = self.RACK_COLS
        unit_w = self.CARD_W
        x0 = self._rack_x_origin()

        for r in sorted(row_y.keys()):
            if r > 0:
                self._build_row_bracket(row_y[r] - self.BRACKET_H, cols * unit_w)

        self._build_rack_chassis(total_h)

        self.hv_panel.setRect(0, 0, self._rack_total_w(), HypervisorPanelItem.PANEL_HEIGHT)
        self.hv_panel.setPos(0, 0)

        for mid, row, col, span in placements:
            if mid is None:
                continue
            mod = self.hypervisor.active_modules[mid]
            card_w = span * unit_w
            x = x0 + col * unit_w
            y = row_y[row]
            h = row_heights[row]
            self._add_module_card(mod, x, y, card_w, h)
            self._card_rects[mid] = QRectF(x, y, card_w, h)

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40))

    def _on_card_dropped(self, module_id: str, drop_pos: QPointF) -> None:
        if module_id not in self._module_order:
            return

        old_order = list(self._module_order)
        self._module_order.remove(module_id)
        if self._preview_insert_idx is not None:
            insert_idx = max(0, min(self._preview_insert_idx, len(self._module_order)))
        else:
            insert_idx = self._insert_index_for_drop_pos(drop_pos, order=self._module_order)

        self._module_order.insert(insert_idx, module_id)
        self._preview_insert_idx = None

        if self._module_order != old_order:
            self._rebuild_layout()

    def _on_card_drag_started(self, module_id: str) -> None:
        self._dragging_mid = module_id
        self._pending_preview_idx = None
        self._preview_insert_idx = None
        self._gap_delay_timer.stop()
        self._gap_anim.stop()
        self._card_anim_start.clear()
        self._card_anim_target.clear()

        active_order = [
            mid for mid in self._module_order if mid in self.hypervisor.active_modules
        ]
        if module_id not in active_order:
            return
        original_idx = active_order.index(module_id)
        preview_order = [mid for mid in active_order if mid != module_id]

        drag_mod = self.hypervisor.active_modules.get(module_id)
        gap_span = min(getattr(drag_mod, "UI_COL_SPAN", 1), self.RACK_COLS) if drag_mod else 1
        gap_height = None
        if drag_mod is not None:
            port_rows = max(len(drag_mod.inputs), len(drag_mod.outputs), 1)
            gap_height = compute_card_height(
                drag_mod.MODULE_NAME,
                port_rows,
                controls_height=self._get_controls_height_for_module(module=drag_mod),
            )

        self._preview_insert_idx = original_idx
        self._pending_preview_idx = original_idx
        target_positions = self._compute_positions_for_order(
            preview_order,
            gap_index=original_idx,
            gap_span=gap_span,
            gap_height=gap_height,
        )
        self._sync_card_rects(target_positions)
        self._start_card_position_animation(target_positions, exclude_mid=module_id)

    def _sync_card_rects(self, positions: dict[str, tuple[float, float]]) -> None:
        for mid, (x, y) in positions.items():
            old = self._card_rects.get(mid)
            if old is not None:
                self._card_rects[mid] = QRectF(x, y, old.width(), old.height())

    def _on_card_drag_move(self, module_id: str, drop_pos: QPointF) -> None:
        if self._dragging_mid != module_id or module_id not in self._module_order:
            return
        preview_order = [mid for mid in self._module_order if mid != module_id]
        insert_idx = self._insert_index_for_drop_pos(drop_pos, order=preview_order)
        if insert_idx == self._pending_preview_idx and insert_idx == self._preview_insert_idx:
            return
        self._pending_preview_idx = insert_idx
        self._gap_delay_timer.start()

    def _start_card_position_animation(
        self,
        target_positions: dict[str, tuple[float, float]],
        *,
        exclude_mid: str | None = None,
    ) -> None:
        self._card_anim_start.clear()
        self._card_anim_target.clear()
        for mid, (x, y) in target_positions.items():
            if mid == exclude_mid:
                continue
            card = self._card_items.get(mid)
            if card is None:
                continue
            start = card.pos()
            target = QPointF(x, y)
            if (target - start).manhattanLength() < 0.25:
                continue
            self._card_anim_start[mid] = QPointF(start)
            self._card_anim_target[mid] = target

        if not self._card_anim_target:
            return
        self._gap_anim.stop()
        self._gap_anim.setCurrentTime(0)
        self._gap_anim.start()

    def _commit_gap_preview(self) -> None:
        if self._pending_preview_idx is None:
            return
        if self._dragging_mid is None and self._tray_drag_name is None:
            return
        if self._pending_preview_idx == self._preview_insert_idx:
            return

        if self._dragging_mid is not None:
            dragging_mid = self._dragging_mid
            drag_mod = self.hypervisor.active_modules.get(dragging_mid)
            gap_span = min(getattr(drag_mod, "UI_COL_SPAN", 1), self.RACK_COLS) if drag_mod else 1
            gap_height = None
            if drag_mod is not None:
                port_rows = max(len(drag_mod.inputs), len(drag_mod.outputs), 1)
                gap_height = compute_card_height(
                    drag_mod.MODULE_NAME,
                    port_rows,
                    controls_height=self._get_controls_height_for_module(module=drag_mod),
                )
            preview_order = [mid for mid in self._module_order if mid != dragging_mid]
        else:
            dragging_mid = None
            gap_span = self._tray_drag_span
            gap_height = self._tray_drag_height
            preview_order = list(self._module_order)

        self._preview_insert_idx = self._pending_preview_idx
        target_positions = self._compute_positions_for_order(
            preview_order,
            gap_index=self._preview_insert_idx,
            gap_span=gap_span,
            gap_height=gap_height,
        )
        self._sync_card_rects(target_positions)
        self._start_card_position_animation(target_positions, exclude_mid=dragging_mid)

    def _on_gap_anim_tick(self, value: object) -> None:
        try:
            t = float(value)
        except (TypeError, ValueError):
            return
        for mid, target in self._card_anim_target.items():
            if mid == self._dragging_mid:
                continue
            card = self._card_items.get(mid)
            start = self._card_anim_start.get(mid)
            if card is None or start is None:
                continue
            card.setPos(
                start.x() + (target.x() - start.x()) * t,
                start.y() + (target.y() - start.y()) * t,
            )

    def _on_card_drag_ended(self, module_id: str, moved: bool = False) -> None:
        if self._dragging_mid != module_id:
            return

        self._gap_delay_timer.stop()
        self._stop_autoscroll()
        if not moved:
            target_positions = self._compute_positions_for_order(self._module_order)
            self._sync_card_rects(target_positions)
            self._start_card_position_animation(target_positions, exclude_mid=module_id)
            self._preview_insert_idx = None
        self._pending_preview_idx = None
        self._dragging_mid = None

    def _rebuild_layout(self) -> None:
        self.cancel_port_tooltip()
        hw_snapshot = getattr(self.hv_panel, "_hardware", None)
        self._gap_delay_timer.stop()
        self._gap_anim.stop()
        self._dragging_mid = None
        self._tray_drag_name = None
        self._pending_preview_idx = None
        self._preview_insert_idx = None
        self._card_anim_start.clear()
        self._card_anim_target.clear()

        self.scene.clear()
        self.port_items.clear()
        self._port_label_items.clear()
        self._port_label_base_colors.clear()
        self._port_label_hover_colors.clear()
        self.module_leds.clear()
        self.wire_items.clear()
        self._power_buttons.clear()
        self._init_buttons.clear()
        self._card_items.clear()
        for builder in self._builders.values():
            builder.clear()
        self._card_rects.clear()
        self._fan_items.clear()

        self._build_hypervisor_panel()
        if hw_snapshot is not None:
            self.hv_panel.set_hardware_snapshot(hw_snapshot)
        self._build_module_cards()
        self._render_existing_wires()

        self.hv_log.info("Rack layout reordered")

    def _build_wiring_submenu(self, parent_menu: QMenu) -> None:
        """Populate a 'Wiring' submenu with the rack-level wiring actions."""
        wiring_menu = parent_menu.addMenu("Wiring")

        auto_wire_toggle_action = wiring_menu.addAction(
            "Auto-Wire: ON" if self._auto_wire_toggle.isChecked() else "Auto-Wire: OFF"
        )
        auto_wire_toggle_action.setCheckable(True)
        auto_wire_toggle_action.setChecked(self._auto_wire_toggle.isChecked())

        wiring_menu.addAction("Wire All")
        wiring_menu.addSeparator()
        wiring_menu.addAction("Delete All Wires")

    def _template_tray_visible(self) -> bool | None:
        """Return template tray visibility when hosted in AppShell, else None."""
        host_window = self.window()
        templates_dock = getattr(host_window, "_templates_dock", None)
        if isinstance(templates_dock, QDockWidget):
            return templates_dock.isVisible()
        return None

    def _toggle_template_tray_visibility(self) -> None:
        """Toggle template tray visibility when available."""
        host_window = self.window()
        toggle_fn = getattr(host_window, "_toggle_templates_dock", None)
        if callable(toggle_fn):
            toggle_fn()
            return
        templates_dock = getattr(host_window, "_templates_dock", None)
        if isinstance(templates_dock, QDockWidget):
            if templates_dock.isVisible():
                templates_dock.hide()
            else:
                templates_dock.show()
                templates_dock.raise_()

    def _build_view_submenu(self, parent_menu: QMenu) -> None:
        """Populate a 'View' submenu with tray visibility toggles."""
        view_menu = parent_menu.addMenu("View")
        module_label = (
            "Hide Module Tray" if self._module_tray_dock.isVisible() else "Show Module Tray"
        )
        view_menu.addAction(module_label)

        template_visible = self._template_tray_visible()
        template_label = "Hide Template Tray" if template_visible else "Show Template Tray"
        template_action = view_menu.addAction(template_label)
        if template_visible is None:
            template_action.setEnabled(False)

    def _handle_wiring_submenu_action(self, action: Any) -> None:
        """Dispatch a wiring submenu action selected from any context menu."""
        from PySide6.QtGui import QAction
        if not isinstance(action, QAction):
            return

        text = action.text()
        if text in ("Auto-Wire: ON", "Auto-Wire: OFF"):
            new_state = not self._auto_wire_toggle.isChecked()
            self._auto_wire_toggle.setChecked(new_state)
        elif text == "Wire All":
            asyncio.create_task(self._auto_wire_all())
        elif text == "Delete All Wires":
            asyncio.create_task(self._delete_all_wires())

    def _handle_view_submenu_action(self, action: Any) -> None:
        """Dispatch a view submenu action selected from any context menu."""
        from PySide6.QtGui import QAction
        if not isinstance(action, QAction):
            return

        text = action.text()
        if text in ("Show Module Tray", "Hide Module Tray"):
            self._toggle_module_tray_visibility()
        elif text in ("Show Template Tray", "Hide Template Tray"):
            self._toggle_template_tray_visibility()

    def on_empty_right_clicked(self, screen_pos: QPoint | QPointF) -> None:
        """Show New-module and Wiring submenus on empty rack space."""
        menu_pos = screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos

        menu = QMenu(self)
        new_menu = menu.addMenu("New Module")
        self._build_wiring_submenu(menu)
        self._build_view_submenu(menu)
        self._build_presets_submenu(menu)

        new_actions: dict[Any, str] = {}
        for module_name in sorted(self._module_register_fns.keys()):
            if not self._is_module_placeable(module_name):
                continue
            action = new_menu.addAction(module_ui_label(module_name))
            new_actions[action] = module_name

        selected = menu.exec(menu_pos)
        if selected in new_actions:
            asyncio.create_task(self._add_new_module(new_actions[selected]))
            return
        self._handle_wiring_submenu_action(selected)
        self._handle_view_submenu_action(selected)
        self._handle_presets_submenu_action(selected)

    def on_card_right_clicked(
        self,
        card: ModuleCardItem,
        screen_pos: QPoint | QPointF,
        local_pos: QPointF | None = None,
    ) -> None:
        """Context menu for a module card (clone / new / delete)."""
        menu_pos = screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos
        module_id = card.module.module_id

        if card.module.MODULE_NAME == "mcp_bus" and local_pos is not None:
            slot_index = self._builders["mcp_bus"].slot_index_at(module_id, local_pos)
            if slot_index is not None:
                self._show_mcp_bus_slot_context_menu(module_id, slot_index, menu_pos)
                return

        menu = QMenu(self)
        clone_action = menu.addAction("Clone")
        new_menu = menu.addMenu("New Module")
        self._build_wiring_submenu(menu)
        self._build_view_submenu(menu)
        self._build_presets_submenu(menu)
        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        new_actions: dict[Any, str] = {}
        for module_name in sorted(self._module_register_fns.keys()):
            if not self._is_module_placeable(module_name):
                continue
            action = new_menu.addAction(module_ui_label(module_name))
            new_actions[action] = module_name

        selected = menu.exec(menu_pos)
        if selected is None:
            return
        if selected == clone_action:
            asyncio.create_task(self._clone_module(module_id))
            return
        if selected == delete_action:
            asyncio.create_task(self._remove_module(module_id))
            return
        if selected in new_actions:
            asyncio.create_task(self._add_new_module(new_actions[selected], insert_after_id=module_id))
            return
        self._handle_wiring_submenu_action(selected)
        self._handle_view_submenu_action(selected)
        self._handle_presets_submenu_action(selected)

    def _show_mcp_bus_slot_context_menu(
        self,
        bus_module_id: str,
        slot_index: int,
        menu_pos: QPoint,
    ) -> None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return

        get_ref = getattr(bus_module, "get_slot_ref", None)
        slot_ref = get_ref(slot_index) if callable(get_ref) else None
        module_id = ""
        if isinstance(slot_ref, dict):
            module_id = str(slot_ref.get("module_id", "")).strip()
        target = self.hypervisor.active_modules.get(module_id) if module_id else None

        menu = QMenu(self)
        clone_action = menu.addAction("Clone")
        new_menu = menu.addMenu("New")
        menu.addSeparator()

        power_action = menu.addAction("Power On" if target is not None and not target.enabled else "Power Off")
        init_action = menu.addAction("Initialize")
        cfg_action = menu.addAction("Configure")
        remove_action = menu.addAction("Remove Tool")

        is_populated = target is not None
        power_action.setEnabled(is_populated)
        init_action.setEnabled(is_populated and target.enabled if target is not None else False)
        cfg_action.setEnabled(is_populated)
        remove_action.setEnabled(slot_ref is not None)

        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        new_actions: dict[Any, str] = {}
        for module_name in sorted(self._module_register_fns.keys()):
            if not self._is_module_placeable(module_name):
                continue
            action = new_menu.addAction(module_ui_label(module_name))
            new_actions[action] = module_name

        selected = menu.exec(menu_pos)
        if selected is None:
            return
        if selected == clone_action:
            asyncio.create_task(self._clone_module(bus_module_id))
            return
        if selected == delete_action:
            asyncio.create_task(self._remove_module(bus_module_id))
            return
        if selected in new_actions:
            asyncio.create_task(self._add_new_module(new_actions[selected], insert_after_id=bus_module_id))
            return
        if selected == power_action:
            self._mcp_bus_slot_power_toggle(bus_module_id, slot_index)
            return
        if selected == init_action:
            self._mcp_bus_slot_init(bus_module_id, slot_index)
            return
        if selected == cfg_action:
            self._mcp_bus_slot_config(bus_module_id, slot_index)
            return
        if selected == remove_action:
            asyncio.create_task(self._mcp_bus_slot_remove(bus_module_id, slot_index))

    def _log_task_exception(self, task: asyncio.Task, label: str) -> None:
        try:
            task.result()
        except Exception as exc:
            self.hv_log.error(f"{label} failed: {exc}")
            self.statusBar().showMessage(f"{label} failed: {exc}", 3500)

    def _is_module_placeable(self, module_name: str) -> bool:
        return module_name.strip().lower() not in self._hidden_placeable_modules

    async def _clone_module(self, module_id: str) -> None:
        module = self.hypervisor.active_modules.get(module_id)
        if module is None:
            return

        clone = type(module)()
        clone.params = dict(module.params)
        self.hypervisor.register_module(clone)
        if self.hypervisor.system_powered:
            try:
                await self.hypervisor.initialize_module(clone.module_id)
            except Exception as exc:
                self.hv_log.error(f"Clone failed for {module_id}: {exc}")
                await self.hypervisor.unregister_module(clone.module_id)
                return
        else:
            clone.enabled = False
            clone.status = ModuleStatus.STOPPED
            # Preserve "new modules start ON" intent after next rack power-on.
            power_memory = getattr(self.hypervisor, "_module_power_memory", None)
            if isinstance(power_memory, dict):
                power_memory[clone.module_id] = True

        insert_idx = self._module_order.index(module_id) + 1 if module_id in self._module_order else len(self._module_order)
        self._module_order.insert(insert_idx, clone.module_id)
        self._rebuild_layout()
        self.hv_log.info(f"Cloned module: {module_id} -> {clone.module_id}")
        self.statusBar().showMessage("Module cloned", 1800)

    async def _add_new_module(
        self,
        module_name: str,
        insert_after_id: str | None = None,
        insert_index: int | None = None,
    ) -> None:
        register_fn = self._module_register_fns.get(module_name)
        if register_fn is None:
            self.hv_log.error(f"No register function for module '{module_name}'")
            return

        before_ids = set(self.hypervisor.active_modules.keys())
        register_fn(self.hypervisor)
        created_ids = [mid for mid in self.hypervisor.active_modules.keys() if mid not in before_ids]
        if not created_ids:
            self.hv_log.error(f"Failed to add module '{module_name}'")
            return
        new_module_id = created_ids[-1]

        if self.hypervisor.system_powered:
            try:
                await self.hypervisor.initialize_module(new_module_id)
            except Exception as exc:
                self.hv_log.error(f"Initialize failed for {new_module_id}: {exc}")
                await self.hypervisor.unregister_module(new_module_id)
                return
        else:
            module = self.hypervisor.active_modules.get(new_module_id)
            if module is not None:
                module.enabled = False
                module.status = ModuleStatus.STOPPED
            # Preserve "new modules start ON" intent after next rack power-on.
            power_memory = getattr(self.hypervisor, "_module_power_memory", None)
            if isinstance(power_memory, dict):
                power_memory[new_module_id] = True

        if insert_index is not None:
            idx = max(0, min(insert_index, len(self._module_order)))
            self._module_order.insert(idx, new_module_id)
        elif insert_after_id is not None and insert_after_id in self._module_order:
            idx = self._module_order.index(insert_after_id) + 1
            self._module_order.insert(idx, new_module_id)
        else:
            self._module_order.append(new_module_id)
        self._rebuild_layout()
        self.hv_log.info(f"Added module: {new_module_id}")

        if self._auto_wire_toggle.isChecked():
            from core.auto_wire import auto_wire_module
            proposals = auto_wire_module(self.hypervisor, new_module_id)
            wired = 0
            for src_id, tgt_id in proposals:
                try:
                    self.hypervisor.connect_ports(src_id, tgt_id)
                    wired += 1
                except Exception as exc:
                    self.hv_log.warn(f"Auto-wire skipped: {exc}")
            if wired:
                self._rebuild_layout()
                self.statusBar().showMessage(
                    f"Added '{module_name}' — auto-wired {wired} connection(s)", 2500
                )
            else:
                self.statusBar().showMessage(f"Added module '{module_name}'", 1800)
        else:
            self.statusBar().showMessage(f"Added module '{module_name}'", 1800)

    async def _remove_module(self, module_id: str) -> None:
        try:
            await self.hypervisor.unregister_module(module_id)
        except Exception as exc:
            self.hv_log.error(f"Delete failed for {module_id}: {exc}")
            self.statusBar().showMessage(f"Delete failed: {exc}", 3500)
            return

        if module_id in self._module_order:
            self._module_order.remove(module_id)
        self.clear_pending_wire()
        self._rebuild_layout()
        self.hv_log.warn(f"Removed module: {module_id}")
        self.statusBar().showMessage("Module removed", 1800)

    async def _auto_wire_all(self) -> None:
        from core.auto_wire import auto_wire_all
        proposals = auto_wire_all(self.hypervisor)
        wired = 0
        for src_id, tgt_id in proposals:
            try:
                self.hypervisor.connect_ports(src_id, tgt_id)
                wired += 1
            except Exception as exc:
                self.hv_log.warn(f"Auto-wire skipped: {exc}")
        if wired:
            self._rebuild_layout()
            self.hv_log.info(f"Auto-wired {wired} connection(s)")
            self.statusBar().showMessage(f"Auto-wired {wired} connection(s)", 3000)
        else:
            self.statusBar().showMessage("No new connections found", 2500)

    async def _delete_all_wires(self) -> None:
        if not self.hypervisor.active_wires:
            self.statusBar().showMessage("No wires to remove", 2000)
            return

        if self.hypervisor.system_powered:
            self.show_transition_overlay("Powering rack off before clearing wires...")
            try:
                await self.hypervisor.graceful_stop_all()
            except Exception as exc:
                self.hv_log.error(f"Power-off before wire clear failed: {exc}")
            finally:
                self._sync_transition_overlay()

        wire_ids = [w.id for w in list(self.hypervisor.active_wires)]
        removed = 0
        for wire_id in wire_ids:
            try:
                self.hypervisor.disconnect_wire(wire_id)
                removed += 1
            except Exception as exc:
                self.hv_log.warn(f"Could not remove wire {wire_id}: {exc}")

        self._rebuild_layout()
        self.hv_log.warn(f"Deleted all wires ({removed} removed)")
        self.statusBar().showMessage(f"Removed {removed} wire(s)", 2500)

    def _build_presets_submenu(self, parent_menu: QMenu) -> None:
        from core.presets import ALL_PRESETS
        presets_menu = parent_menu.addMenu("Presets")
        for preset in ALL_PRESETS:
            action = presets_menu.addAction(preset.name)
            action.setToolTip(preset.description)

    def _handle_presets_submenu_action(self, action: Any) -> None:
        from PySide6.QtGui import QAction
        from core.presets import ALL_PRESETS
        if not isinstance(action, QAction):
            return
        for preset in ALL_PRESETS:
            if action.text() == preset.name:
                asyncio.create_task(self._apply_preset(preset))
                return

    async def _apply_preset(self, preset: Any) -> None:
        """Register all preset modules and apply their pre-defined wiring."""
        from core.presets import Preset
        if not isinstance(preset, Preset):
            return

        missing = [
            m for m in preset.modules
            if m not in self._module_register_fns
        ]
        if missing:
            self.hv_log.error(
                f"Preset '{preset.name}' requires missing modules: {', '.join(missing)}"
            )
            self.statusBar().showMessage(
                f"Preset unavailable — missing: {', '.join(missing)}", 4000
            )
            return

        self.hv_log.info(f"Applying preset: {preset.name}")
        self.statusBar().showMessage(f"Loading preset '{preset.name}'…")

        module_ids: list[str] = []
        for module_name in preset.modules:
            register_fn = self._module_register_fns[module_name]
            before_ids = set(self.hypervisor.active_modules.keys())
            register_fn(self.hypervisor)
            created = [
                mid for mid in self.hypervisor.active_modules.keys()
                if mid not in before_ids
            ]
            if not created:
                self.hv_log.error(
                    f"Preset '{preset.name}': failed to register '{module_name}'"
                )
                module_ids.append("")
                continue
            new_id = created[-1]
            module_ids.append(new_id)

            if self.hypervisor.system_powered:
                try:
                    await self.hypervisor.initialize_module(new_id)
                except Exception as exc:
                    self.hv_log.error(
                        f"Preset '{preset.name}': init failed for '{module_name}': {exc}"
                    )
                    await self.hypervisor.unregister_module(new_id)
                    module_ids[-1] = ""
                    continue
            else:
                mod = self.hypervisor.active_modules.get(new_id)
                if mod is not None:
                    mod.enabled = False
                    mod.status = ModuleStatus.STOPPED
                power_memory = getattr(self.hypervisor, "_module_power_memory", None)
                if isinstance(power_memory, dict):
                    power_memory[new_id] = True

            self._module_order.append(new_id)

        # Wire using port names; skip any slot that failed to register
        wired = 0
        for wire_spec in preset.wires:
            src_id = module_ids[wire_spec.src_idx] if wire_spec.src_idx < len(module_ids) else ""
            tgt_id = module_ids[wire_spec.tgt_idx] if wire_spec.tgt_idx < len(module_ids) else ""
            if not src_id or not tgt_id:
                continue
            src_port = self.hypervisor.port_registry.get_by_name(src_id, wire_spec.src_port)
            tgt_port = self.hypervisor.port_registry.get_by_name(tgt_id, wire_spec.tgt_port)
            if src_port is None or tgt_port is None:
                self.hv_log.warn(
                    f"Preset '{preset.name}': port not found — "
                    f"{wire_spec.src_port} / {wire_spec.tgt_port}"
                )
                continue
            try:
                self.hypervisor.connect_ports(src_port.id, tgt_port.id)
                wired += 1
            except Exception as exc:
                self.hv_log.warn(f"Preset '{preset.name}': wire skipped — {exc}")

        self._rebuild_layout()
        registered = sum(1 for mid in module_ids if mid)
        self.hv_log.info(
            f"Preset '{preset.name}' applied — {registered} module(s), {wired} wire(s)"
        )
        self.statusBar().showMessage(
            f"Preset '{preset.name}': {registered} modules, {wired} connections", 3500
        )

    def _resolve_mcp_bus_slot_module(
        self, bus_module_id: str, slot_index: int
    ) -> BaseModule | None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return None
        get_ref = getattr(bus_module, "get_slot_ref", None)
        slot_ref = get_ref(slot_index) if callable(get_ref) else None
        if not isinstance(slot_ref, dict):
            return None
        module_id = str(slot_ref.get("module_id", "")).strip()
        if not module_id:
            return None
        return self.hypervisor.active_modules.get(module_id)

    def _mcp_bus_slot_module_ids(self, bus_module_id: str) -> list[str]:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return []
        get_refs = getattr(bus_module, "get_slot_refs", None)
        refs = get_refs() if callable(get_refs) else []
        if not isinstance(refs, list):
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            module_id = str(ref.get("module_id", "")).strip()
            if not module_id or module_id in seen:
                continue
            if module_id in self.hypervisor.active_modules:
                ids.append(module_id)
                seen.add(module_id)
        return ids

    def _mcp_bus_slot_power_toggle(self, bus_module_id: str, slot_index: int) -> None:
        target = self._resolve_mcp_bus_slot_module(bus_module_id, slot_index)
        if target is None:
            return
        self._toggle_module_power(target.module_id)

    def _mcp_bus_slot_init(self, bus_module_id: str, slot_index: int) -> None:
        target = self._resolve_mcp_bus_slot_module(bus_module_id, slot_index)
        if target is None:
            return
        self._trigger_module_init(target.module_id)

    def _mcp_bus_slot_config(self, bus_module_id: str, slot_index: int) -> None:
        target = self._resolve_mcp_bus_slot_module(bus_module_id, slot_index)
        if target is None:
            return
        run_mcp_slot_config_dialog(
            self, target, slot_index,
            on_apply_init=lambda: asyncio.create_task(self._init_module(target.module_id)),
        )

    async def _mcp_bus_slot_remove(self, bus_module_id: str, slot_index: int) -> None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return
        get_ref = getattr(bus_module, "get_slot_ref", None)
        slot_ref = get_ref(slot_index) if callable(get_ref) else None
        if not isinstance(slot_ref, dict):
            return

        module_id = str(slot_ref.get("module_id", "")).strip()
        if not module_id:
            return

        target = self.hypervisor.active_modules.get(module_id)
        clear_ref = getattr(bus_module, "clear_slot_ref", None)
        if callable(clear_ref):
            clear_ref(slot_index)

        if target is None:
            self.statusBar().showMessage(f"Cleared slot {slot_index + 1}", 1800)
            return

        reply = QMessageBox.question(
            self,
            "Remove MCP Tool",
            "Remove the cartridge module from the rack?\n\n"
            "Yes = remove module entirely\n"
            "No = keep module as standalone",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            set_ref = getattr(bus_module, "set_slot_ref", None)
            if callable(set_ref):
                set_ref(
                    slot_index=slot_index,
                    module_id=module_id,
                    cartridge_type=str(slot_ref.get("cartridge_type", "")),
                    label=str(slot_ref.get("label", "")),
                )
            return

        if reply == QMessageBox.StandardButton.No:
            if module_id not in self._module_order:
                self._module_order.append(module_id)
                self._rebuild_layout()
            self.statusBar().showMessage(
                f"Slot {slot_index + 1} cleared; module kept as standalone", 2200
            )
            return

        await self._remove_module(module_id)

    def _add_module_card(
        self, module: BaseModule, x: float, y: float, width: float, height: float | None = None
    ) -> None:
        P = 13.0
        port_rows = max(len(module.inputs), len(module.outputs), 1)
        builder = self._builders.get(module.MODULE_NAME)
        controls_h = self._get_controls_height_for_module(module=module)
        if height is None:
            height = compute_card_height(
                module.MODULE_NAME,
                port_rows,
                P,
                controls_height=controls_h,
            )

        card = ModuleCardItem(module, width, height, self.scene, self)
        card.setPos(x, y)
        self.scene.addItem(card)
        self._card_items[module.module_id] = card

        grip = DragGripItem(card)
        grip_x = P - 4.0
        grip_y = (ModuleCardItem.TITLE_BAR_HEIGHT - grip.boundingRect().height()) / 2.0
        grip.setPos(grip_x, grip_y)
        grip.setZValue(2.5)

        title = QGraphicsSimpleTextItem(module_ui_label(module.MODULE_NAME).upper(), card)
        _title_font = QFont("Consolas", self.MODULE_TITLE_FONT_SIZE, QFont.Weight.Bold)
        title.setFont(_title_font)
        title.setBrush(QBrush(self.MODULE_TITLE_COLOR))
        title.setPos(18 + P, 8 + P)
        title.setToolTip(f"Serial: {module.module_id}")

        led = QGraphicsEllipseItem(width - 22 - P, 10 + P, 10, 10, card)
        led.setPen(QPen(Qt.GlobalColor.black, 0.5))
        led.setBrush(QBrush(_status_color(module.status)))
        self.module_leds[module.module_id] = led

        pwr_btn = CardButtonItem(
            width - 22 - P - 40, 8 + P, 34, 14, "PWR",
            lambda mid=module.module_id: self._toggle_module_power(mid), card,
        )
        self._power_buttons[module.module_id] = pwr_btn
        init_btn = CardButtonItem(
            width - 22 - P - 78, 8 + P, 34, 14, "INIT",
            lambda mid=module.module_id: self._trigger_module_init(mid), card,
        )
        self._init_buttons[module.module_id] = init_btn

        in_ports = list(module.inputs.values())
        out_ports = list(module.outputs.values())
        y0 = 56.0 + P
        step = 22.0

        sep = QGraphicsRectItem(P, y0 - 13.0, width - 2 * P, 1, card)
        sep.setBrush(QBrush(QColor("#353b44")))
        sep.setPen(QPen(Qt.PenStyle.NoPen))

        if builder is not None:
            result = builder.build_controls(card, module, width, y0, P)
            if isinstance(result, list):
                self._fan_items.extend(result)
            if controls_h is None:
                controls_h = float(getattr(builder, "CONTROLS_HEIGHT", 0.0))
            ports_y = y0 + controls_h + 14.0
            if module.MODULE_NAME != "gap_filler":
                port_sep = QGraphicsRectItem(P, ports_y - 14.0, width - 2 * P, 1, card)
                port_sep.setBrush(QBrush(QColor("#353b44")))
                port_sep.setPen(QPen(Qt.PenStyle.NoPen))
        else:
            ports_y = y0

        for i, port in enumerate(in_ports):
            py = ports_y + i * step
            jack = PortJackItem(port, self, 10.0 + P, py)
            jack.setParentItem(card)
            self.port_items[port.id] = jack

            _port_text = port_faceplate_label(module.MODULE_NAME, port.name)
            label = PortLabelItem(_port_text, jack, self, card)
            label_font = QFont("Consolas", self.PORT_LABEL_FONT_SIZE)
            label_font.setWeight(self.PORT_LABEL_FONT_WEIGHT)
            label.setFont(label_font)
            self._register_port_label_item(port.id, label, self.PORT_LABEL_BASE_COLOR)
            label.setPos(22.0 + P, py - 8.0)

        for i, port in enumerate(out_ports):
            py = ports_y + i * step
            jack = PortJackItem(port, self, width - 10.0 - P, py)
            jack.setParentItem(card)
            self.port_items[port.id] = jack

            _port_text = port_faceplate_label(module.MODULE_NAME, port.name)
            label = PortLabelItem(_port_text, jack, self, card)
            label_font = QFont("Consolas", self.PORT_LABEL_FONT_SIZE)
            label_font.setWeight(self.PORT_LABEL_FONT_WEIGHT)
            label.setFont(label_font)
            self._register_port_label_item(port.id, label, self.PORT_LABEL_BASE_COLOR)
            text_w = label.boundingRect().width()
            label.setPos(width - 22.0 - P - text_w, py - 8.0)

    def _toggle_module_power(self, module_id: str) -> None:
        module = self.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        if module.MODULE_NAME == "mcp_bus":
            if module.enabled:
                asyncio.create_task(self._power_off_mcp_bus(module_id))
            else:
                asyncio.create_task(self._power_on_mcp_bus(module_id))
            return
        if module.enabled:
            asyncio.create_task(self._power_off_module(module_id))
        else:
            asyncio.create_task(self._power_on_module(module_id))

    def _trigger_module_init(self, module_id: str) -> None:
        module = self.hypervisor.active_modules.get(module_id)
        if module is None or not module.enabled:
            return
        if module.MODULE_NAME == "mcp_bus":
            asyncio.create_task(self._init_mcp_bus(module_id))
            return
        if module.MODULE_NAME == "basic_brain":
            asyncio.create_task(self._full_stack_restart_and_init(module_id))
            return
        asyncio.create_task(self._init_module(module_id))

    async def _power_off_mcp_bus(self, bus_module_id: str) -> None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return
        slot_ids = self._mcp_bus_slot_module_ids(bus_module_id)
        # Remember per-slot power state so bus power-on restores only prior ON cartridges.
        bus_module.params["slot_power_memory"] = {
            module_id: bool(self.hypervisor.active_modules[module_id].enabled)
            for module_id in slot_ids
            if module_id in self.hypervisor.active_modules
        }

        for module_id in slot_ids:
            module = self.hypervisor.active_modules.get(module_id)
            if module is not None and module.enabled:
                await self._power_off_module(module_id)

        await self._power_off_module(bus_module_id)

    async def _power_on_mcp_bus(self, bus_module_id: str) -> None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None:
            return
        await self._power_on_module(bus_module_id)

        slot_ids = self._mcp_bus_slot_module_ids(bus_module_id)
        raw_memory = bus_module.params.get("slot_power_memory", {})
        slot_power_memory = raw_memory if isinstance(raw_memory, dict) else {}
        for module_id in slot_ids:
            should_power_on = bool(slot_power_memory.get(module_id, False))
            module = self.hypervisor.active_modules.get(module_id)
            if module is None:
                continue
            if should_power_on and not module.enabled:
                await self._power_on_module(module_id)

    async def _init_mcp_bus(self, bus_module_id: str) -> None:
        bus_module = self.hypervisor.active_modules.get(bus_module_id)
        if bus_module is None or not bus_module.enabled:
            return

        await self._init_module(bus_module_id)
        slot_ids = self._mcp_bus_slot_module_ids(bus_module_id)
        attempted = 0
        ready = 0
        skipped_off = 0
        for module_id in slot_ids:
            module = self.hypervisor.active_modules.get(module_id)
            if module is None or not module.enabled:
                skipped_off += 1
                continue
            attempted += 1
            await self._init_module(module_id)
            module_after = self.hypervisor.active_modules.get(module_id)
            if module_after is not None and module_after.status == ModuleStatus.READY:
                ready += 1

        summary = (
            f"{bus_module_id}: bus init complete \u2014 "
            f"{attempted}/{len(slot_ids)} cartridges initialized, "
            f"{ready} READY, {skipped_off} skipped (power off)"
        )
        self.hv_log.info(summary)
        self.statusBar().showMessage(summary, 3500)

    async def _power_off_module(self, module_id: str) -> None:
        self.hv_log.warn(f"{module_id}: powering OFF")
        self.statusBar().showMessage(f"Powering off {module_id}...", 2000)
        try:
            await self.hypervisor.stop_module(module_id)
            self.hv_log.error(f"{module_id}: OFF")
        except Exception as exc:
            self.hv_log.error(f"{module_id}: power-off failed: {exc}")

    async def _power_on_module(self, module_id: str) -> None:
        self.hv_log.info(f"{module_id}: powering ON")
        self.statusBar().showMessage(f"Powering on {module_id}...", 2000)
        try:
            await self.hypervisor.start_module(module_id)
            self.hv_log.info(f"{module_id}: ON")
        except Exception as exc:
            self.hv_log.error(f"{module_id}: power-on failed: {exc}")

    async def _init_module(self, module_id: str) -> None:
        self.hv_log.info(f"{module_id}: INIT triggered")
        self.statusBar().showMessage(f"Initializing {module_id}...", 2000)
        try:
            await self.hypervisor.init_module(module_id)
            module = self.hypervisor.active_modules.get(module_id)
            if module is not None and module.status == ModuleStatus.READY:
                self.hv_log.info(f"{module_id}: READY")
            else:
                self.hv_log.warn(f"{module_id}: init complete (not ready)")
        except Exception as exc:
            self.hv_log.error(f"{module_id}: init failed: {exc}")

    async def _full_stack_restart_and_init(self, module_id: str) -> None:
        self.hv_log.info(f"{module_id}: INIT triggered (full stack restart)")
        self.statusBar().showMessage(
            f"{module_id}: restarting stack OFF -> ON...", 3000
        )
        try:
            await self.hypervisor.graceful_stop_all()
            await self.hypervisor.boot_all()
            await self.hypervisor.init_module(module_id)
            module = self.hypervisor.active_modules.get(module_id)
            if module is not None and module.status == ModuleStatus.READY:
                self.hv_log.info(f"{module_id}: full stack restart + INIT complete (READY)")
            else:
                self.hv_log.warn(
                    f"{module_id}: full stack restart complete, init finished (not ready)"
                )
        except Exception as exc:
            self.hv_log.error(f"{module_id}: full stack restart + init failed: {exc}")
            self.statusBar().showMessage(f"{module_id}: restart/init failed: {exc}", 5000)

    def _start_status_refresh(self) -> None:
        timer = QTimer(self)
        timer.timeout.connect(self._refresh)
        timer.start(500)
        self._status_timer = timer

    def _refresh(self) -> None:
        self._sync_transition_overlay()
        for module_id, led in self.module_leds.items():
            module = self.hypervisor.active_modules.get(module_id)
            if module is None:
                continue
            led.setBrush(QBrush(_status_color(module.status)))

            btn = self._power_buttons.get(module_id)
            if btn is not None:
                for child in btn.childItems():
                    if isinstance(child, QGraphicsSimpleTextItem):
                        if module.enabled:
                            child.setBrush(QBrush(QColor("#39d353")))
                        else:
                            child.setBrush(QBrush(QColor("#e24c4c")))
                if module.enabled:
                    btn.setBrush(QBrush(QColor("#1a2e1a")))
                    btn.setPen(QPen(QColor("#39d353"), 1.0))
                else:
                    btn.setBrush(QBrush(QColor("#2e1a1a")))
                    btn.setPen(QPen(QColor("#e24c4c"), 1.0))

            init_btn = self._init_buttons.get(module_id)
            if init_btn is not None:
                init_btn.setEnabled(module.enabled)
                init_btn.setOpacity(1.0 if module.enabled else 0.35)
                for child in init_btn.childItems():
                    if isinstance(child, QGraphicsSimpleTextItem):
                        child.setBrush(
                            QBrush(QColor("#6090c0") if module.enabled else QColor("#3a4552"))
                        )
                if module.enabled:
                    init_btn.setBrush(QBrush(QColor("#0c1218")))
                    init_btn.setPen(QPen(QColor("#2a4060"), 1.0))
                else:
                    init_btn.setBrush(QBrush(QColor("#101216")))
                    init_btn.setPen(QPen(QColor("#2a3038"), 1.0))

        for wire_id, path_item in self.wire_items.items():
            wire = next((w for w in self.hypervisor.active_wires if w.id == wire_id), None)
            if wire is None:
                continue
            src_mod = self.hypervisor.active_modules.get(wire.source_port.owner_module_id)
            tgt_mod = self.hypervisor.active_modules.get(wire.target_port.owner_module_id)
            both_on = (src_mod is not None and src_mod.enabled) and (tgt_mod is not None and tgt_mod.enabled)
            path_item.setOpacity(1.0 if both_on else 0.25)

        brain_builder = self._builders.get("basic_brain")
        if brain_builder is not None:
            while self.hypervisor._inference_log_queue:
                entry = self.hypervisor._inference_log_queue.popleft()
                (mid, prompt, response, p_tok, c_tok, ttft, total) = entry
                brain_builder.log_inference(
                    mid, prompt, response, p_tok, c_tok, ttft, total
                )

        for builder in self._builders.values():
            builder.refresh_all()
        self.hv_panel.refresh()

    def _port_center(self, port_id: str) -> QPointF | None:
        item = self.port_items.get(port_id)
        if item is None:
            return None
        return item.mapToScene(item.boundingRect().center())

    def _set_path(self, path_item: QGraphicsPathItem, p0: QPointF, p1: QPointF) -> None:
        points = compute_bezier_points(p0.x(), p0.y(), p1.x(), p1.y(), segments=28)
        painter_path = QPainterPath(QPointF(points[0][0], points[0][1]))
        for bx, by in points[1:]:
            painter_path.lineTo(bx, by)
        path_item.setPath(painter_path)

    def _register_port_label_item(
        self,
        port_id: str,
        label_item: PortLabelItem,
        base_color: QColor,
    ) -> None:
        base = QColor(base_color)
        self._port_label_items[port_id] = label_item
        self._port_label_base_colors[port_id] = base
        self._port_label_hover_colors[port_id] = base.lighter(165)
        label_item.setBrush(QBrush(base))

    def set_port_label_hover(self, port_id: str, hovered: bool) -> None:
        label_item = self._port_label_items.get(port_id)
        if label_item is None:
            return
        base = self._port_label_base_colors.get(port_id, self.PORT_LABEL_BASE_COLOR)
        hover = self._port_label_hover_colors.get(port_id, QColor(base).lighter(165))
        label_item.setBrush(QBrush(hover if hovered else base))

    def on_port_hover_enter(self, jack: PortJackItem) -> None:
        self.set_port_label_hover(jack.port.id, True)
        self.schedule_port_tooltip(jack)

    def on_port_hover_leave(self, jack: PortJackItem) -> None:
        self.set_port_label_hover(jack.port.id, False)
        self.cancel_port_tooltip()

    def schedule_port_tooltip(self, jack: PortJackItem) -> None:
        """Begin the hover-delay timer for showing a port tooltip."""
        if self.pending_wire is not None:
            return
        self._tooltip_jack = jack
        self._tooltip_timer.start(PortJackItem._HOVER_DELAY_MS)

    def cancel_port_tooltip(self) -> None:
        """Hide any pending or visible port tooltip."""
        self._tooltip_timer.stop()
        self._tooltip_jack = None
        self._tooltip_widget.setVisible(False)

    def _fire_port_tooltip(self) -> None:
        if self._tooltip_jack is None or self.pending_wire is not None:
            return

        jack = self._tooltip_jack
        self._tooltip_widget.set_port(jack.port)

        scene_center = jack.mapToScene(jack.boundingRect().center())
        vp: QPoint = self.view.mapFromScene(scene_center)

        tw = self._tooltip_widget.width()
        th = self._tooltip_widget.height()
        vw = self.view.viewport().width()
        vh = self.view.viewport().height()

        tip_x = vp.x() + 16
        tip_y = vp.y() - th // 2

        if tip_x + tw > vw - 4:
            tip_x = vp.x() - tw - 16

        tip_y = max(4, min(tip_y, vh - th - 4))

        self._tooltip_widget.move(tip_x, tip_y)
        self._tooltip_widget.setVisible(True)
        self._tooltip_widget.raise_()

    def _render_existing_wires(self) -> None:
        for wire in self.hypervisor.active_wires:
            self._render_wire(wire)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._position_zoom_overlay()
        self._sync_transition_overlay()

    def _render_wire(self, wire: Wire) -> None:
        source_pt = self._port_center(wire.source_port.id)
        target_pt = self._port_center(wire.target_port.id)
        if source_pt is None or target_pt is None:
            return

        key = wire.id
        if key in self.wire_items:
            path_item = self.wire_items[key]
        else:
            color = _wire_color(wire.source_port, wire.target_port)
            path_item = WirePathItem(wire.source_port.id, wire.target_port.id, color,
                                     controller=self)
            self.scene.addItem(path_item)
            self.wire_items[key] = path_item
        self._set_path(path_item, source_pt, target_pt)

    def on_port_clicked(self, jack: PortJackItem) -> None:
        """Handle a port click — start or complete a wire connection."""
        port = jack.port

        if self.pending_wire is None:
            if port.direction != PortDirection.OUT and port.connection_mode != ConnectionMode.CHANNEL:
                self.statusBar().showMessage("Select an OUT/CHANNEL source port first", 2500)
                return
            self.pending_wire = PendingWire(source_port_id=port.id)
            self.pending_wire_item = QGraphicsPathItem()
            self.pending_wire_item.setPen(QPen(_port_color(port), 1.8, Qt.PenStyle.DashLine))
            self.pending_wire_item.setZValue(2)
            self.scene.addItem(self.pending_wire_item)
            source_pt = self._port_center(port.id)
            if source_pt is not None:
                self._set_path(self.pending_wire_item, source_pt, source_pt)
            self.statusBar().showMessage(
                f"Connecting from {port.owner_module_id}:{port_ui_label(port.name)}"
            )
            return

        source_id = self.pending_wire.source_port_id
        target_id = port.id
        if source_id == target_id:
            self.clear_pending_wire()
            self.statusBar().showMessage("Cancelled wire creation", 1500)
            return

        try:
            wire = self.hypervisor.connect_ports(source_id, target_id)
            self._render_wire(wire)
            self.hv_log.info(
                f"Wire: {wire.source_port.owner_module_id}:{port_ui_label(wire.source_port.name)}"
                f" \u2192 {wire.target_port.owner_module_id}:{port_ui_label(wire.target_port.name)}"
            )
            self.statusBar().showMessage("Wire connected", 1500)
        except Exception as exc:
            self.hv_log.error(f"Connection rejected: {exc}")
            self.statusBar().showMessage(f"Connection rejected: {exc}", 4000)
        finally:
            self.clear_pending_wire()

    def update_pending_wire(self, cursor_pos: QPointF) -> None:
        """Redraw the in-progress wire rubber band to follow the cursor."""
        if self.pending_wire is None or self.pending_wire_item is None:
            return
        source_pt = self._port_center(self.pending_wire.source_port_id)
        if source_pt is None:
            return
        self._set_path(self.pending_wire_item, source_pt, cursor_pos)

    def on_wire_right_clicked(self, path_item: WirePathItem) -> None:
        """Remove an existing wire on right-click."""
        wire_id = next(
            (wid for wid, item in self.wire_items.items() if item is path_item), None
        )
        if wire_id is None:
            return
        try:
            wire = next((w for w in self.hypervisor.active_wires if w.id == wire_id), None)
            if wire is not None:
                src_name = f"{wire.source_port.owner_module_id}:{port_ui_label(wire.source_port.name)}"
                tgt_name = f"{wire.target_port.owner_module_id}:{port_ui_label(wire.target_port.name)}"
                self.hypervisor.disconnect_wire(wire_id)
                self.scene.removeItem(path_item)
                del self.wire_items[wire_id]
                self.hv_log.warn(f"Wire removed: {src_name} \u2192 {tgt_name}")
                self.statusBar().showMessage("Wire removed", 1500)
        except Exception as exc:
            self.hv_log.error(f"Failed to remove wire: {exc}")

    def clear_pending_wire(self) -> None:
        """Cancel and remove any in-progress wire."""
        self.pending_wire = None
        if self.pending_wire_item is not None:
            self.scene.removeItem(self.pending_wire_item)
            self.pending_wire_item = None

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is self.view.viewport() and isinstance(event, QEvent):
            if event.type() == QEvent.Type.DragEnter:
                self.dragEnterEvent(event)
                return True
            if event.type() == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return True
            if event.type() == QEvent.Type.Drop:
                self.dropEvent(event)
                return True
            if event.type() == QEvent.Type.DragLeave:
                self._on_tray_drag_leave()
                return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: object) -> None:
        if hasattr(event, "mimeData") and event.mimeData().hasFormat(MODULE_TRAY_MIME):
            event.acceptProposedAction()
            module_name = bytes(event.mimeData().data(MODULE_TRAY_MIME)).decode("utf-8").strip()
            module_cls = self.module_registry.get(module_name)
            self._tray_drag_name = module_name
            self._tray_drag_span = min(
                int(getattr(module_cls, "UI_COL_SPAN", 1)) if module_cls else 1,
                self.RACK_COLS,
            )
            self._tray_drag_height = compute_card_height(
                module_name,
                1,
                controls_height=self._get_controls_height_for_module(module_name=module_name),
            )
            self._pending_preview_idx = None
            self._preview_insert_idx = None
            return
        if hasattr(event, "ignore"):
            event.ignore()

    def dragMoveEvent(self, event: object) -> None:
        if hasattr(event, "mimeData") and event.mimeData().hasFormat(MODULE_TRAY_MIME):
            event.acceptProposedAction()
            if self._tray_drag_name is not None:
                drop_pos = self._extract_drop_scene_pos(event)
                insert_idx = self._insert_index_for_drop_pos(drop_pos)
                if insert_idx != self._pending_preview_idx or insert_idx != self._preview_insert_idx:
                    self._pending_preview_idx = insert_idx
                    self._gap_delay_timer.start()
                if hasattr(event, "position"):
                    vp_y = int(event.position().y())
                elif hasattr(event, "pos"):
                    vp_y = int(event.pos().y())
                else:
                    vp_y = 0
                self._update_autoscroll(vp_y)
            return
        if hasattr(event, "ignore"):
            event.ignore()

    def dropEvent(self, event: object) -> None:
        if not hasattr(event, "mimeData") or not event.mimeData().hasFormat(MODULE_TRAY_MIME):
            if hasattr(event, "ignore"):
                event.ignore()
            return

        module_name = bytes(event.mimeData().data(MODULE_TRAY_MIME)).decode("utf-8").strip()
        if module_name in self._module_register_fns and self._is_module_placeable(module_name):
            if self._preview_insert_idx is not None:
                insert_index = self._preview_insert_idx
            else:
                drop_pos = self._extract_drop_scene_pos(event)
                insert_index = self._insert_index_for_drop_pos(drop_pos)
            self._end_tray_drag()
            asyncio.create_task(self._add_new_module(module_name, insert_index=insert_index))
            self.statusBar().showMessage(f"Dropped module '{module_name}'", 1400)
            event.acceptProposedAction()
            return

        self._end_tray_drag()
        self.hv_log.error(f"Unknown module dropped: {module_name}")
        self.statusBar().showMessage(f"Unknown module: '{module_name}'", 3000)
        if hasattr(event, "ignore"):
            event.ignore()

    def _end_tray_drag(self) -> None:
        self._gap_delay_timer.stop()
        self._stop_autoscroll()
        self._tray_drag_name = None
        self._pending_preview_idx = None
        self._preview_insert_idx = None

    def _on_tray_drag_leave(self) -> None:
        if self._tray_drag_name is None:
            return
        self._gap_delay_timer.stop()
        self._gap_anim.stop()
        self._stop_autoscroll()
        target_positions = self._compute_positions_for_order(self._module_order)
        self._sync_card_rects(target_positions)
        self._start_card_position_animation(target_positions)
        self._tray_drag_name = None
        self._pending_preview_idx = None
        self._preview_insert_idx = None

    def _extract_drop_scene_pos(self, event: object) -> QPointF:
        if hasattr(event, "position"):
            view_pos = event.position().toPoint()
        elif hasattr(event, "pos"):
            view_pos = event.pos()
        else:
            view_pos = QPoint(0, 0)
        return self.view.mapToScene(view_pos)

    def _update_autoscroll(self, viewport_y: int) -> None:
        """Start/update/stop the autoscroll timer based on proximity to viewport edges."""
        vh = self.view.viewport().height()
        zone = self.AUTOSCROLL_ZONE_PX
        if viewport_y < zone:
            ratio = max(0.0, 1.0 - viewport_y / zone)
            dy = -max(1, int(self.AUTOSCROLL_MAX_SPEED_PX * ratio))
        elif viewport_y > vh - zone:
            ratio = max(0.0, 1.0 - (vh - viewport_y) / zone)
            dy = max(1, int(self.AUTOSCROLL_MAX_SPEED_PX * ratio))
        else:
            dy = 0
        self._autoscroll_dy = dy
        if dy != 0:
            if not self._autoscroll_timer.isActive():
                self._autoscroll_timer.start()
        else:
            self._autoscroll_timer.stop()

    def _on_autoscroll_tick(self) -> None:
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.value() + self._autoscroll_dy)

    def _stop_autoscroll(self) -> None:
        self._autoscroll_timer.stop()
        self._autoscroll_dy = 0

    def _insert_index_for_drop_pos(
        self, drop_pos: QPointF, order: list[str] | None = None
    ) -> int:
        eval_order = order if order is not None else self._module_order
        if not eval_order:
            return 0

        leftmost_bonus = 0.0
        if self._dragging_mid is not None:
            drag_mod = self.hypervisor.active_modules.get(self._dragging_mid)
            if drag_mod is not None:
                drag_span = max(1, min(getattr(drag_mod, "UI_COL_SPAN", 1), self.RACK_COLS))
                # Reorder drags use the card center as the probe point; for wider
                # cards this otherwise requires pulling too far off the rack edge
                # to trigger "insert before first slot".
                leftmost_bonus = (drag_span - 1) * self.CARD_W * 0.5

        insert_idx = len(eval_order)
        prev_cy: float | None = None
        prev_h: float | None = None
        for i, mid in enumerate(eval_order):
            rect = self._card_rects.get(mid)
            if rect is None:
                continue
            cy = rect.center().y()
            cx = rect.center().x()

            # Row-transition check: when the loop crosses into a new visual row,
            # see if the drop position was already in the previous row's y-band.
            # If so, the user is targeting a trailing empty slot in that row —
            # insert before the current module (end of previous row).
            # Use a generous 0.75× height band so the slot is easy to hit.
            if prev_cy is not None and prev_h is not None:
                if abs(cy - prev_cy) > prev_h * 0.75:  # new visual row
                    if abs(drop_pos.y() - prev_cy) < prev_h * 0.75:
                        insert_idx = i
                        break

            same_row = abs(drop_pos.y() - cy) < rect.height() * 0.5
            # For the leftmost slot use the full right edge as threshold so the
            # user only needs to hover over (not past) the leftmost module to
            # trigger its rightward slide.  All other slots keep the centre-point
            # threshold so insert boundaries fall naturally between cards.
            threshold_x = rect.right() + leftmost_bonus if i == 0 else cx
            if drop_pos.y() < cy - rect.height() * 0.3 or (same_row and drop_pos.x() < threshold_x):
                insert_idx = i
                break

            prev_cy = cy
            prev_h = rect.height()
        return insert_idx
