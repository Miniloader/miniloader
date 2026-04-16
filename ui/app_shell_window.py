"""Application shell — nav sidebar with stacked views for Rack, Portal, and Settings."""

from __future__ import annotations

import asyncio
import shlex
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.hypervisor import Hypervisor
    from core.vault import VaultManager

from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.download_manager import DownloadManager
from core.module_installer import ModuleInstaller
from core.settings_store import SettingsStore
from ui.portal_page import PortalPage
from ui.pages import _AccountPage, DownloadsPage, _HomePage, _SettingsPage
from ui.rag_setup_dialog import (
    RAG_MODEL_SETTING_KEY,
    RAG_VECTOR_KEY_SETTING,
    RagSetupDialog,
)
from ui.rack_window import RackWindow
from ui.shell import AppShellLifecycleMixin, AppShellLocalAiMixin, AppShellTemplateMixin

_WINDOW_REF: "AppShellWindow | None" = None


class _WindowControlsBar(QWidget):
    """Floating minimize + resize-shrink + close controls for frameless shell window."""

    def __init__(self, parent: QMainWindow) -> None:
        super().__init__(parent)
        self._win = parent
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "_WindowControlsBar {"
            "  background: rgba(13, 15, 20, 220);"
            "  border: 1px solid rgba(255,255,255,18);"
            "  border-radius: 6px;"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(2)

        _ss = (
            "QPushButton {"
            "  background: transparent; color: #5a6878; border: none;"
            "  border-radius: 4px; font-size: 13px; padding: 0 3px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,14); color: #c0d0e0; }"
            "QPushButton:pressed { background: rgba(255,255,255,26); }"
        )
        _close_ss = (
            "QPushButton {"
            "  background: transparent; color: #6a3838; border: none;"
            "  border-radius: 4px; font-size: 13px; padding: 0 3px;"
            "}"
            "QPushButton:hover { background: rgba(180,40,40,40); color: #e87878; }"
            "QPushButton:pressed { background: rgba(180,40,40,70); }"
        )

        self._min_btn = QPushButton("\u2500")
        self._min_btn.setFixedSize(26, 20)
        self._min_btn.setToolTip("Minimize")
        self._min_btn.setStyleSheet(_ss)
        self._min_btn.clicked.connect(parent.showMinimized)

        self._toggle_btn = QPushButton("\u2921")
        self._toggle_btn.setFixedSize(26, 20)
        self._toggle_btn.setToolTip("Shrink window")
        self._toggle_btn.setStyleSheet(_ss)
        self._toggle_btn.clicked.connect(self._toggle_size)

        self._close_btn = QPushButton("\u00d7")
        self._close_btn.setFixedSize(26, 20)
        self._close_btn.setToolTip("Exit (graceful shutdown)")
        self._close_btn.setStyleSheet(_close_ss)

        for btn in (self._min_btn, self._toggle_btn, self._close_btn):
            layout.addWidget(btn)

        self.adjustSize()
        self.raise_()

    def _is_filling_screen(self) -> bool:
        screen = self._win.screen() or QApplication.primaryScreen()
        if screen is None:
            return False
        return self._win.geometry() == screen.availableGeometry()

    def _toggle_size(self) -> None:
        screen = self._win.screen() or QApplication.primaryScreen()
        if screen is None:
            return

        avail = screen.availableGeometry()

        if self._is_filling_screen():
            new_w = int(avail.width() * 0.75)
            new_h = int(avail.height() * 0.75)
            self._win.setGeometry(QRect(
                avail.x() + (avail.width() - new_w) // 2,
                avail.y() + (avail.height() - new_h) // 2,
                new_w,
                new_h,
            ))
        else:
            self._win.setGeometry(avail)

        self.sync_state()

    def sync_state(self) -> None:
        if self._is_filling_screen():
            self._toggle_btn.setText("\u2921")
            self._toggle_btn.setToolTip("Shrink window")
        else:
            self._toggle_btn.setText("\u2922")
            self._toggle_btn.setToolTip("Maximize window")


class _DraggableNav(QWidget):
    """Nav sidebar that doubles as the window drag handle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_start_global: QPoint | None = None
        self._win_start_pos: QPoint | None = None
        self._saved_geometry: QRect | None = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_global = event.globalPosition().toPoint()
            self._win_start_pos = self.window().pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_start_global is not None
            and self._win_start_pos is not None
        ):
            delta = event.globalPosition().toPoint() - self._drag_start_global
            self.window().move(self._win_start_pos + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start_global = None
        self._win_start_pos = None
        super().mouseReleaseEvent(event)

    def _is_filling_screen(self) -> bool:
        win = self.window()
        screen = win.screen() or QApplication.primaryScreen()
        if screen is None:
            return False
        return win.geometry() == screen.availableGeometry()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            screen = win.screen() or QApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()
            if self._is_filling_screen():
                if self._saved_geometry is not None:
                    win.setGeometry(self._saved_geometry)
                else:
                    w = int(avail.width() * 0.75)
                    h = int(avail.height() * 0.75)
                    win.setGeometry(QRect(
                        avail.x() + (avail.width() - w) // 2,
                        avail.y() + (avail.height() - h) // 2,
                        w, h,
                    ))
            else:
                self._saved_geometry = win.geometry()
                win.setGeometry(avail)
        super().mouseDoubleClickEvent(event)


class AppShellWindow(
    AppShellTemplateMixin,
    AppShellLocalAiMixin,
    AppShellLifecycleMixin,
    QMainWindow,
):
    """Application shell hosting Rack, Portal, and Settings views."""

    def __init__(
        self,
        hypervisor: Hypervisor,
        module_plugins: list[dict[str, Any]] | None = None,
        module_registry: dict[str, type] | None = None,
        vault: VaultManager | None = None,
        startup_errors: list[dict[str, str]] | None = None,
        initial_preset: str | None = None,
    ) -> None:
        super().__init__()
        self.hypervisor = hypervisor
        self._primary_hypervisor = hypervisor
        self._background_hypervisor_tasks: dict[int, asyncio.Task] = {}
        self._module_registry: dict[str, type] = module_registry or {}
        self._module_plugins: list[dict[str, Any]] = list(module_plugins or [])
        self._vault = vault
        self._startup_errors = list(startup_errors or [])
        self._initial_preset: str | None = initial_preset
        self._current_template_name: str | None = None
        self._shutdown_started = False
        self._nav_buttons: list[QPushButton] = []
        self._nav_leds: list[QLabel] = []
        self._nav_rows: list[QWidget] = []
        self._nav_page_indices: list[int] = []
        self._console_output: QPlainTextEdit | None = None
        self._console_input: QLineEdit | None = None

        self.setWindowTitle("Miniloader")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        _screen = QApplication.primaryScreen()
        if _screen is not None:
            _avail = _screen.availableGeometry()
            self.setMinimumSize(int(_avail.width() * 0.3), int(_avail.height() * 0.3))
        else:
            self.setMinimumSize(800, 500)

        root = QWidget(self)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        nav = _DraggableNav(root)
        nav.setFixedWidth(190)
        nav.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "  stop:0 #0c0e12, stop:0.96 #0e1016, stop:1.0 #161c24);"
        )
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(12, 12, 12, 12)
        nav_layout.setSpacing(4)

        # ── Logo — fills nav content width ───────────────────
        # Nav is 190px wide; margins are 12px each side → 166px content width.
        _NAV_INNER_W = 166
        logo_label = QLabel()
        logo_label.setStyleSheet("background: transparent;")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        _logo_path = Path(__file__).parent / "assets" / "miniloader-logo.png"
        _logo_pix = QPixmap(str(_logo_path))
        if not _logo_pix.isNull():
            logo_label.setPixmap(
                _logo_pix.scaled(
                    _NAV_INNER_W, 32,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            logo_label.setText("MINILOADER")
            logo_label.setStyleSheet(
                "color: #d8d8d8; font-size: 13px; font-weight: 700;"
                " letter-spacing: 0.5px; background: transparent;"
            )
        nav_layout.addWidget(logo_label)
        nav_layout.addSpacing(2)

        user_display_name = self._vault.username if self._vault is not None else "Guest"
        self._user_badge = QPushButton(f">_ {user_display_name}")
        self._user_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._user_badge.setStyleSheet(
            "QPushButton {"
            "  color: #28c840; font-size: 11px; background: transparent;"
            "  border: none; padding: 1px 2px; text-align: left;"
            "  font-family: 'Courier New', Courier, monospace;"
            "  text-shadow: 0 0 6px #28c840;"
            "}"
            "QPushButton:hover { color: #5af07a; }"
        )
        self._user_badge.setToolTip("Open account page")
        nav_layout.addWidget(self._user_badge)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #181e28;")
        nav_layout.addWidget(sep)
        nav_layout.addSpacing(6)

        self._download_manager: DownloadManager | None = None
        if self._vault is not None:
            self._download_manager = DownloadManager(vault=self._vault)
        builtin_names = set(self._module_registry.keys())
        builtin_names.update(
            str(p.get("name") or "").strip()
            for p in self._module_plugins
            if isinstance(p, dict)
        )
        community_modules_root = (
            self._vault.ensure_user_data_dir() / "community_modules"
            if self._vault is not None
            else (Path.home() / ".miniloader" / "community_modules")
        )
        self._module_installer = ModuleInstaller(
            community_modules_root,
            builtin_module_names={n for n in builtin_names if n},
        )

        self._stack = QStackedWidget(root)
        self._stack.setStyleSheet("background-color: #121316;")

        self._home_page = _HomePage()
        self._rack_page = RackWindow(
            hypervisor,
            module_plugins=self._module_plugins,
            module_registry=self._module_registry,
        )
        self._rack_page._on_run_backend_config = lambda: self._run_local_ai_setup(service="llm")
        self._rack_page._on_persist_backend_selection = self._persist_backend_selection
        self._rack_page._on_run_rag_config = self._run_rag_config
        self._store_page = PortalPage(
            vault=self._vault,
            download_manager=self._download_manager,
            module_installer=self._module_installer,
            on_module_hotload=self._on_portal_module_hotload,
        )
        self._settings_page = _SettingsPage(
            self._vault,
            startup_errors=self._startup_errors,
            on_run_ai_setup=self._run_local_ai_setup,
            get_local_ai_status=self._get_local_ai_status,
            on_open_test_suite=self._run_test_benchmark_wizard,
            get_template_preferences=self._get_template_preferences,
            on_set_autosave_on_close=self._set_autosave_on_close_preference,
            on_apply_agent_preset=self._apply_agent_preset,
            get_active_agent_preset_name=self._get_active_agent_preset_name,
            get_agent_tool_names=self._get_agent_tool_names,
        )
        self._downloads_page = DownloadsPage(
            download_manager=self._download_manager,
        )
        self._account_page = _AccountPage(
            vault=self._vault,
            on_username_changed=self._update_user_badge,
            on_logout=self._request_logout,
        )

        self._stack.addWidget(self._home_page)
        self._stack.addWidget(self._rack_page)
        self._stack.addWidget(self._store_page)
        self._stack.addWidget(self._settings_page)
        self._stack.addWidget(self._downloads_page)
        self._stack.addWidget(self._account_page)

        for label, idx in [("Home", 0), ("Rack", 1), ("Portal", 2), ("Downloads", 4), ("Settings", 3)]:
            nav_layout.addWidget(self._make_nav_button(label, idx))

        nav_layout.addStretch(1)

        self._build_templates_dock()
        self._build_command_console_dock()
        asyncio.create_task(self._startup_checks())
        self._user_badge.clicked.connect(lambda: self._set_page(5))

        sep2 = QWidget()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: #181e28;")
        nav_layout.addWidget(sep2)

        exit_btn = QPushButton("\u23fb  Exit")
        exit_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #5a3838;"
            "  border: none; border-radius: 4px;"
            "  padding: 7px 10px; text-align: left; font-size: 12px;"
            "}"
            "QPushButton:hover { background: #1a1214; color: #c87070; }"
        )
        exit_btn.clicked.connect(self._request_exit)
        nav_layout.addWidget(exit_btn)

        root_layout.addWidget(nav)
        root_layout.addWidget(self._stack, 1)
        if self._initial_preset is not None:
            self._set_page(1)
            self.statusBar().showMessage("Press the power button to start the rack")
        else:
            self._set_page(0)
            self.statusBar().showMessage("Press the power button to start the rack")

        self._controls = _WindowControlsBar(self)
        self._controls._close_btn.clicked.connect(self._request_exit)
        self._controls.adjustSize()
        self._controls.raise_()
        self._reposition_controls()

        self._resize_grip = QSizeGrip(self)
        self._resize_grip.setStyleSheet("background: transparent;")
        self._resize_grip.resize(16, 16)
        self._reposition_resize_grip()
        self._console_shortcut = QShortcut(QKeySequence("Ctrl+`"), self)
        self._console_shortcut.activated.connect(self._toggle_command_console)
        self._console_shortcut_alt = QShortcut(QKeySequence("F12"), self)
        self._console_shortcut_alt.activated.connect(self._toggle_command_console)

        self._exit_overlay = self._build_exit_overlay()

    def _on_portal_module_hotload(self, payload: dict[str, Any]) -> None:
        plugin = payload.get("plugin")
        module_cls = payload.get("module_cls")
        if not isinstance(plugin, dict):
            return
        name = str(plugin.get("name") or "").strip()
        register_fn = plugin.get("register_fn")
        if not name or not callable(register_fn):
            return
        if module_cls is not None:
            self._module_registry[name] = module_cls
        self._rack_page.add_runtime_module(name, register_fn, module_cls)


    _LED_OFF_SS = (
        "background: #3f4652; border: 1px solid #2a3038; border-radius: 3px;"
    )
    _LED_ON_SS = (
        "background: #30e848; border: 1px solid #18a830; border-radius: 3px;"
    )

    _NAV_ROW_OFF_SS = (
        "#NavRow {"
        "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        "    stop:0 #0e1014, stop:1 #080a0e);"
        "  border: 1px solid #1a1e24;"
        "  border-radius: 5px;"
        "}"
    )
    _NAV_ROW_ON_SS = (
        "#NavRow {"
        "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
        "    stop:0 #10181c, stop:1 #0a1014);"
        "  border: 1px solid #2a3a2a;"
        "  border-radius: 5px;"
        "}"
    )

    def _make_nav_button(self, label: str, page_index: int) -> QWidget:
        row = QWidget()
        row.setObjectName("NavRow")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(self._NAV_ROW_OFF_SS)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(10, 6, 8, 6)
        rl.setSpacing(8)

        led = QLabel()
        led.setFixedSize(7, 7)
        led.setStyleSheet(self._LED_OFF_SS)
        rl.addWidget(led, 0, Qt.AlignmentFlag.AlignVCenter)

        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #4a5568;"
            "  border: none; padding: 4px 2px;"
            "  text-align: left; font-size: 12px;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-weight: 600; letter-spacing: 0.5px;"
            "}"
            "QPushButton:checked { color: #90c8a0; }"
            "QPushButton:hover:!checked { color: #6a8090; }"
        )
        btn.clicked.connect(lambda: self._set_page(page_index))
        rl.addWidget(btn, 1)

        self._nav_buttons.append(btn)
        self._nav_leds.append(led)
        self._nav_rows.append(row)
        self._nav_page_indices.append(page_index)
        return row

    def _set_page(self, page_index: int) -> None:
        self._stack.setCurrentIndex(page_index)
        for btn, led, row, idx in zip(
            self._nav_buttons, self._nav_leds, self._nav_rows, self._nav_page_indices
        ):
            active = idx == page_index
            btn.setChecked(active)
            led.setStyleSheet(self._LED_ON_SS if active else self._LED_OFF_SS)
            row.setStyleSheet(self._NAV_ROW_ON_SS if active else self._NAV_ROW_OFF_SS)

    # ── Templates dock (lives inside RackWindow) ─────────────

    def _build_templates_dock(self) -> None:
        """Create a QDockWidget for templates and add it to the rack page."""
        dock_ss = (
            "QDockWidget {"
            "  color: #8a9ab0; font-size: 11px; font-weight: 600;"
            "}"
            "QDockWidget::title {"
            "  background: #141820; padding: 5px 8px;"
            "  border-bottom: 1px solid #232a38;"
            "}"
        )
        self._templates_dock = QDockWidget("Templates", self._rack_page)
        self._templates_dock.setStyleSheet(dock_ss)
        self._templates_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._templates_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        container = QWidget()
        container.setObjectName("TemplatesDockBody")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            "#TemplatesDockBody { background: #141820; }"
        )
        cl = QVBoxLayout(container)
        cl.setContentsMargins(8, 6, 8, 8)
        cl.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current rack as template")
        save_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1a2030; color: #8eb8e8; border: 1px solid #2a4060;"
            "  border-radius: 4px; padding: 4px 8px; font-size: 10px;"
            "}"
            "QPushButton:hover { background: #1e2840; }"
        )
        save_btn.clicked.connect(self._save_current_template)
        toolbar.addWidget(save_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset rack to factory defaults")
        reset_btn.setStyleSheet(
            "QPushButton {"
            "  background: #201a18; color: #d4a06a; border: 1px solid #504030;"
            "  border-radius: 4px; padding: 4px 8px; font-size: 10px;"
            "}"
            "QPushButton:hover { background: #2a2018; }"
        )
        reset_btn.clicked.connect(self._request_reset_to_defaults)
        toolbar.addWidget(reset_btn)

        refresh_btn = QPushButton("\u21BB")
        refresh_btn.setToolTip("Refresh template list")
        refresh_btn.setFixedSize(26, 24)
        refresh_btn.setStyleSheet(
            "QPushButton {"
            "  background: transparent; color: #6a7a90; border: 1px solid #2a3444;"
            "  border-radius: 4px; font-size: 13px; padding: 0;"
            "}"
            "QPushButton:hover { color: #a0b8d8; background: #1a2030; }"
        )
        refresh_btn.clicked.connect(self._populate_template_list)
        toolbar.addWidget(refresh_btn)

        cl.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._template_list_widget = QWidget()
        self._template_list_layout = QVBoxLayout(self._template_list_widget)
        self._template_list_layout.setContentsMargins(0, 2, 0, 2)
        self._template_list_layout.setSpacing(2)
        scroll.setWidget(self._template_list_widget)
        cl.addWidget(scroll, 1)

        self._templates_dock.setWidget(container)
        self._rack_page.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._templates_dock,
        )
        module_tray_dock = getattr(self._rack_page, "_module_tray_dock", None)
        if isinstance(module_tray_dock, QDockWidget):
            self._rack_page.tabifyDockWidget(module_tray_dock, self._templates_dock)
            module_tray_dock.raise_()
        self._templates_dock.setMinimumWidth(170)

        self._templates_dock_toggle = QPushButton("Hide Templates")
        self._templates_dock_toggle.setStyleSheet(
            "QPushButton {"
            "  background: #1b2433; color: #b7c8e8; border: 1px solid #324661;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #233146; }"
        )
        self._templates_dock_toggle.clicked.connect(self._toggle_templates_dock)
        self._rack_page.statusBar().addPermanentWidget(self._templates_dock_toggle)

        self._templates_dock.visibilityChanged.connect(self._sync_templates_dock_btn)

    def _toggle_templates_dock(self) -> None:
        if self._templates_dock.isVisible():
            self._templates_dock.hide()
        else:
            self._templates_dock.show()
            self._templates_dock.raise_()

    def _sync_templates_dock_btn(self, visible: bool) -> None:
        self._templates_dock_toggle.setText(
            "Hide Templates" if visible else "Show Templates"
        )

    def _build_command_console_dock(self) -> None:
        dock_ss = (
            "QDockWidget {"
            "  color: #8a9ab0; font-size: 11px; font-weight: 600;"
            "}"
            "QDockWidget::title {"
            "  background: #141820; padding: 5px 8px;"
            "  border-bottom: 1px solid #232a38;"
            "}"
        )
        self._console_dock = QDockWidget("Console", self)
        self._console_dock.setStyleSheet(dock_ss)
        self._console_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        body = QWidget()
        body.setObjectName("CommandConsoleBody")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body.setStyleSheet(
            "#CommandConsoleBody { background: #10141c; }"
            "QPlainTextEdit {"
            "  background: #0a0e14; color: #88c4ff; border: 1px solid #24364a;"
            "  border-radius: 4px; padding: 6px;"
            "  font-family: Consolas, 'Courier New', monospace; font-size: 11px;"
            "}"
            "QLineEdit {"
            "  background: #0d141f; color: #d8e8ff; border: 1px solid #2d4664;"
            "  border-radius: 4px; padding: 5px 8px;"
            "  font-family: Consolas, 'Courier New', monospace; font-size: 11px;"
            "}"
            "QLineEdit:focus { border-color: #4b79aa; }"
        )
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)

        self._console_output = QPlainTextEdit(body)
        self._console_output.setReadOnly(True)
        self._console_output.document().setMaximumBlockCount(1000)
        body_layout.addWidget(self._console_output, 1)

        self._console_input = QLineEdit(body)
        self._console_input.setPlaceholderText("Type command and press Enter (help)")
        self._console_input.returnPressed.connect(self._submit_console_command)
        body_layout.addWidget(self._console_input)

        self._console_dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
        self._console_dock.hide()
        self._console_dock.visibilityChanged.connect(self._on_command_console_visibility_changed)

        self._console_toggle_btn = QPushButton("Show Console")
        self._console_toggle_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1b2433; color: #b7c8e8; border: 1px solid #324661;"
            "  border-radius: 5px; padding: 2px 8px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #233146; }"
        )
        self._console_toggle_btn.clicked.connect(self._toggle_command_console)
        self.statusBar().addPermanentWidget(self._console_toggle_btn)
        self._sync_command_console_btn(self._console_dock.isVisible())
        self._append_console_line("Console ready. Opening the console shows command list.")

    def _toggle_command_console(self) -> None:
        if not hasattr(self, "_console_dock"):
            return
        if self._console_dock.isVisible():
            self._console_dock.hide()
            return
        self._console_dock.show()
        self._console_dock.raise_()
        if self._console_input is not None:
            self._console_input.setFocus()

    def _sync_command_console_btn(self, visible: bool) -> None:
        self._console_toggle_btn.setText("Hide Console" if visible else "Show Console")

    def _on_command_console_visibility_changed(self, visible: bool) -> None:
        self._sync_command_console_btn(visible)
        if visible:
            self._append_console_line("Available commands:")
            self._append_console_help_lines()

    def _append_console_line(self, text: str) -> None:
        if self._console_output is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self._console_output.appendPlainText(f"[{ts}] {text}")
        sb = self._console_output.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _submit_console_command(self) -> None:
        if self._console_input is None:
            return
        command_line = self._console_input.text().strip()
        if not command_line:
            return
        self._console_input.clear()
        self._append_console_line(f"> {command_line}")
        task = asyncio.create_task(self._run_console_command(command_line))
        task.add_done_callback(self._on_console_command_finished)

    def _on_console_command_finished(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            self._append_console_line(f"error: {exc}")

    async def _run_console_command(self, command_line: str) -> None:
        try:
            tokens = shlex.split(command_line)
        except ValueError as exc:
            self._append_console_line(f"parse error: {exc}")
            return
        if not tokens:
            return

        cmd = tokens[0].lower()
        args = tokens[1:]
        handlers = {
            "help": self._console_cmd_help,
            "clear": self._console_cmd_clear,
            "status": self._console_cmd_status,
            "modules": self._console_cmd_modules,
            "module": self._console_cmd_module,
            "power": self._console_cmd_power,
            "init": self._console_cmd_init,
            "restart": self._console_cmd_restart,
            "logs": self._console_cmd_logs,
        }
        handler = handlers.get(cmd)
        if handler is None:
            self._append_console_line(f"unknown command: {cmd}. Try 'help'.")
            return
        await handler(args)

    async def _console_cmd_help(self, _args: list[str]) -> None:
        self._append_console_help_lines()

    def _append_console_help_lines(self) -> None:
        self._append_console_line("help                              Show this command list")
        self._append_console_line("clear                             Clear console output")
        self._append_console_line("status                            Show rack power and module counts")
        self._append_console_line("modules                           List all module ids")
        self._append_console_line("module <module_id>                Show module details")
        self._append_console_line("power <on|off|cycle>              Rack power controls")
        self._append_console_line("power <on|off|cycle> <module_id>  Module power controls")
        self._append_console_line("init <module_id>                  Run module init/check_ready")
        self._append_console_line("restart <module_id>               Restart one module")
        self._append_console_line("logs [count]                      Show recent system log entries")

    async def _console_cmd_clear(self, _args: list[str]) -> None:
        if self._console_output is not None:
            self._console_output.clear()

    async def _console_cmd_status(self, _args: list[str]) -> None:
        modules = list(self.hypervisor.active_modules.values())
        total = len(modules)
        enabled = sum(1 for m in modules if m.enabled)
        ready = sum(1 for m in modules if str(m.status.value) == "ready")
        running = sum(1 for m in modules if str(m.status.value) == "running")
        errors = sum(1 for m in modules if str(m.status.value) == "error")
        self._append_console_line(
            f"rack_powered={self.hypervisor.system_powered} total={total} "
            f"enabled={enabled} running={running} ready={ready} error={errors}"
        )

    async def _console_cmd_modules(self, _args: list[str]) -> None:
        if not self.hypervisor.active_modules:
            self._append_console_line("no modules loaded")
            return
        for module_id in sorted(self.hypervisor.active_modules.keys()):
            module = self.hypervisor.active_modules[module_id]
            isolation = "isolated" if module.PROCESS_ISOLATION else "inproc"
            power_state = "on" if module.enabled else "off"
            self._append_console_line(
                f"{module_id} ({module.MODULE_NAME}) power={power_state} "
                f"status={module.status.value} mode={isolation}"
            )

    async def _console_cmd_module(self, args: list[str]) -> None:
        if len(args) != 1:
            self._append_console_line("usage: module <module_id>")
            return
        module_id = args[0]
        module = self.hypervisor.active_modules.get(module_id)
        if module is None:
            self._append_console_line(f"module not found: {module_id}")
            return
        input_names = ", ".join(sorted(module.inputs.keys())) or "-"
        output_names = ", ".join(sorted(module.outputs.keys())) or "-"
        self._append_console_line(
            f"{module_id}: name={module.MODULE_NAME} power={'on' if module.enabled else 'off'} "
            f"status={module.status.value} process_isolation={module.PROCESS_ISOLATION}"
        )
        self._append_console_line(f"  in: {input_names}")
        self._append_console_line(f"  out: {output_names}")

    async def _console_cmd_power(self, args: list[str]) -> None:
        if not args:
            self._append_console_line("usage: power <on|off|cycle> [module_id]")
            return
        action = args[0].lower()
        if action not in {"on", "off", "cycle"}:
            self._append_console_line("usage: power <on|off|cycle> [module_id]")
            return

        if len(args) == 1:
            if action == "on":
                await self.hypervisor.boot_all()
                self._append_console_line("rack powered on")
            elif action == "off":
                await self.hypervisor.graceful_stop_all()
                self._append_console_line("rack powered off")
            else:
                await self.hypervisor.graceful_stop_all()
                await self.hypervisor.boot_all()
                self._append_console_line("rack power cycled")
            return

        module_id = args[1]
        module = self.hypervisor.active_modules.get(module_id)
        if module is None:
            self._append_console_line(f"module not found: {module_id}")
            return
        if action == "on":
            await self.hypervisor.start_module(module_id)
            self._append_console_line(f"module powered on: {module_id}")
        elif action == "off":
            await self.hypervisor.stop_module(module_id)
            self._append_console_line(f"module powered off: {module_id}")
        else:
            await self.hypervisor.stop_module(module_id)
            await self.hypervisor.start_module(module_id)
            self._append_console_line(f"module power cycled: {module_id}")

    async def _console_cmd_init(self, args: list[str]) -> None:
        if len(args) != 1:
            self._append_console_line("usage: init <module_id>")
            return
        module_id = args[0]
        if module_id not in self.hypervisor.active_modules:
            self._append_console_line(f"module not found: {module_id}")
            return
        await self.hypervisor.init_module(module_id)
        self._append_console_line(f"init requested: {module_id}")

    async def _console_cmd_restart(self, args: list[str]) -> None:
        if len(args) != 1:
            self._append_console_line("usage: restart <module_id>")
            return
        module_id = args[0]
        if module_id not in self.hypervisor.active_modules:
            self._append_console_line(f"module not found: {module_id}")
            return
        await self.hypervisor.restart_module(module_id)
        self._append_console_line(f"restart requested: {module_id}")

    async def _console_cmd_logs(self, args: list[str]) -> None:
        count = 10
        if args:
            try:
                count = max(1, min(100, int(args[0])))
            except ValueError:
                self._append_console_line("usage: logs [count]")
                return
        entries = list(self.hypervisor.system_log)[-count:]
        if not entries:
            self._append_console_line("system log is empty")
            return
        for payload in entries:
            signal = getattr(payload.signal_type, "value", str(payload.signal_type))
            source = str(payload.source_module)
            message = str(payload.data.get("log_message", "")).strip()
            if message:
                self._append_console_line(f"{source} {signal}: {message}")
            else:
                self._append_console_line(f"{source} {signal}")

    async def _startup_checks(self) -> None:
        """Run one-time startup tasks after the UI is built."""
        await self._load_template_exit_preferences()
        if hasattr(self, "_settings_page"):
            self._settings_page.set_autosave_on_close_enabled(
                self._get_template_preferences().get("autosave_on_close", False)
            )
        await self.seed_default_templates()
        self._populate_template_list()
        await self._restore_last_open_template_on_startup()
        await self._check_new_machine()
        await self._apply_initial_preset()

    async def _apply_initial_preset(self) -> None:
        """Apply the onboarding-chosen preset to the rack (first boot only)."""
        preset_name = self._initial_preset
        if preset_name is None:
            return
        self._initial_preset = None

        if preset_name == "blank_rack":
            self.statusBar().showMessage(
                "Empty rack — drag modules from the tray to get started", 5000
            )
            return

        from core.presets import ALL_PRESETS

        for preset in ALL_PRESETS:
            if preset.name == preset_name:
                await self._rack_page._apply_preset(preset)
                return

        self.statusBar().showMessage(
            f"Preset '{preset_name}' not found — showing empty rack", 4000
        )

    def _agent_modules(self) -> list[Any]:
        modules: list[Any] = []
        for module in self.hypervisor.active_modules.values():
            if getattr(module, "MODULE_NAME", "") == "agent_engine":
                modules.append(module)
        return modules

    def _apply_agent_preset(self, preset: dict[str, Any]) -> None:
        name = str(preset.get("name", "")).strip() or "unnamed"
        for module in self._agent_modules():
            loader = getattr(module, "load_preset", None)
            if callable(loader):
                loader(preset)
        self.statusBar().showMessage(f"Applied agent preset '{name}'", 3500)

    def _get_active_agent_preset_name(self) -> str:
        for module in self._agent_modules():
            name = str(module.params.get("active_preset", "")).strip()
            if name:
                return name
        return ""

    def _get_agent_tool_names(self) -> list[str]:
        names: set[str] = set()
        for module in self._agent_modules():
            tools_registry = module.params.get("tools_registry", {})
            if isinstance(tools_registry, dict):
                for entries in tools_registry.values():
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        tool_name = str(entry.get("name", "")).strip()
                        if tool_name:
                            names.add(tool_name)
        return sorted(names)

    def _run_rag_config(self, module_id: str | None = None) -> None:
        if self._vault is None:
            QMessageBox.warning(
                self,
                "Vault Required",
                "RAG setup requires an active account vault.",
            )
            return
        dialog = RagSetupDialog(
            vault=self._vault,
            download_manager=self._download_manager,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        asyncio.create_task(self._apply_rag_model_path(module_id))

    def _run_test_benchmark_wizard(self) -> None:
        from ui.test_benchmark_wizard import TestBenchmarkWizard

        status = self._get_local_ai_status()
        configured_backend = str(status.get("preferred_backend", "cpu")).strip().lower()
        dlg = TestBenchmarkWizard(parent=self, configured_backend=configured_backend)
        dlg.exec()

    async def _apply_rag_model_path(self, module_id: str | None) -> None:
        if self._vault is None:
            return
        store = SettingsStore(self._vault)
        model_path = str(await store.get(RAG_MODEL_SETTING_KEY) or "").strip()
        vector_key = str(await store.get(RAG_VECTOR_KEY_SETTING) or "").strip()
        if not model_path:
            return
        machine = self._get_machine_settings_entry()
        machine["rag_setup_completed"] = True
        machine["rag_install_result"] = "configured_via_wizard"
        await self._save_machine_settings_async(machine)

        abs_store_path = str(
            (self._vault.get_user_data_dir() / "rag_store").resolve()
        )

        if module_id:
            module = self.hypervisor.active_modules.get(module_id)
            if module is not None:
                module.params["embedding_model_path"] = model_path
                if vector_key:
                    module.params["vector_store_encryption_key"] = vector_key
                raw_store = str(module.params.get("vector_store_path", "")).strip()
                if not raw_store or not Path(raw_store).is_absolute():
                    module.params["vector_store_path"] = abs_store_path
                module.params["_rag_status"] = "model_configured"
                module.params["_rag_status_detail"] = model_path
        if hasattr(self, "_settings_page"):
            self._settings_page._refresh_rag_status()
        self.statusBar().showMessage(
            "RAG setup saved — restart/power-cycle the RAG module to apply.",
            5000,
        )

    def _update_user_badge(self, username: str) -> None:
        self._user_badge.setText(f">_ {username}")
        if hasattr(self, "_account_page") and hasattr(self._account_page, "_username_input"):
            self._account_page._username_input.setText(username)

    def _reposition_controls(self) -> None:
        if not hasattr(self, "_controls"):
            return
        margin = 10
        controls_w = self._controls.width()
        x = max(margin, self.width() - controls_w - margin)
        self._controls.move(x, margin)
        self._controls.sync_state()
        self._controls.raise_()

    def _reposition_resize_grip(self) -> None:
        if not hasattr(self, "_resize_grip"):
            return
        self._resize_grip.move(
            max(0, self.width() - self._resize_grip.width()),
            max(0, self.height() - self._resize_grip.height()),
        )
        self._resize_grip.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_controls()
        self._reposition_resize_grip()
        if hasattr(self, "_exit_overlay") and self._exit_overlay.isVisible():
            self._exit_overlay.setGeometry(0, 0, self.width(), self.height())

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._reposition_controls()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reposition_controls()
        self._reposition_resize_grip()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if isinstance(event, QEvent) and event.type() == QEvent.Type.WindowStateChange:
            self._reposition_controls()
            self._reposition_resize_grip()

    # ── Exit overlay ──────────────────────────────────────────

    def _build_exit_overlay(self) -> QWidget:
        overlay = QWidget(self)
        overlay.setObjectName("ExitOverlay")
        overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        overlay.setStyleSheet(
            "#ExitOverlay {"
            "  background: rgba(10, 12, 16, 235);"
            "}"
        )
        inner = QVBoxLayout(overlay)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Exiting\u2026")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "color: #6a7a90; font-size: 22px; font-weight: 600;"
            " letter-spacing: 1px; background: transparent;"
        )
        inner.addWidget(label)

        hint = QLabel("Shutting down rack and modules")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            "color: #3a4858; font-size: 12px; background: transparent;"
        )
        inner.addWidget(hint)

        overlay.hide()
        return overlay

    def show_exit_overlay(self) -> None:
        self._exit_overlay.setGeometry(0, 0, self.width(), self.height())
        self._exit_overlay.raise_()
        self._exit_overlay.show()


def launch_ui(
    hypervisor: Hypervisor,
    module_plugins: list[dict[str, Any]] | None = None,
    module_registry: dict[str, type] | None = None,
    vault: VaultManager | None = None,
    startup_errors: list[dict[str, str]] | None = None,
    initial_preset: str | None = None,
) -> None:
    app = QApplication.instance()
    if app is None:
        raise ImportError("Qt application is not initialized")

    global _WINDOW_REF
    _WINDOW_REF = AppShellWindow(
        hypervisor,
        module_plugins=module_plugins,
        module_registry=module_registry,
        vault=vault,
        startup_errors=startup_errors,
        initial_preset=initial_preset,
    )
    screen = QApplication.primaryScreen()
    if screen is not None:
        _WINDOW_REF.setGeometry(screen.availableGeometry())
    _WINDOW_REF.show()
