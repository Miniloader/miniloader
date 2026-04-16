from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMimeData, QPoint, Qt, QTimer
from PySide6.QtGui import QDrag, QFontMetrics, QMouseEvent, QResizeEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ui.label_aliases import module_ui_label

MODULE_TRAY_MIME = "application/x-module-name"
MODULE_IDENTITY: dict[str, tuple[str, str]] = {
    "basic_brain": ("#d93232", "\u2699"),
    "cloud_brain": ("#4e8fe8", "\u2601"),
    "database": ("#2d93a2", "\u2395"),
    "discord_terminal": ("#7289da", "\u2689"),
    "file_access": ("#d4932c", "\u2387"),
    "gap_filler": ("#6f6f6f", "\u2502"),
    "gpt_server": ("#8a57e8", "\u25B6"),
    "gpt_terminal": ("#5a66c8", "\u2328"),
    "mcp_bus": ("#e08b3a", "\u2B29"),
    "ngrok_tunnel": ("#5a66c8", "\u21C5"),
    "pg_cartridge": ("#3ea874", "\u2697"),
    "rag_engine": ("#80c65f", "\u2B22"),
}
_DEFAULT_IDENTITY: tuple[str, str] = ("#7b8594", "\u25A3")


def _module_identity(display_name: str, payload_name: str) -> tuple[str, str]:
    return MODULE_IDENTITY.get(display_name) or MODULE_IDENTITY.get(payload_name) or _DEFAULT_IDENTITY


class ModuleTrayItem(QFrame):
    def __init__(
        self,
        payload_name: str,
        display_name: str,
        module_description: str,
        module_version: str,
        ui_col_span: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.payload_name = payload_name
        self._module_description = module_description.strip()
        self._module_version = module_version.strip() or "0.1.0"
        self._ui_col_span = max(1, int(ui_col_span))
        self._drag_start_pos: QPoint | None = None
        self._display_name = display_name.upper()
        accent_color, icon_glyph = _module_identity(
            display_name.strip().lower(),
            payload_name.strip().lower(),
        )
        self._accent_color = accent_color

        self.setObjectName("moduleTrayItem")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(56)
        self.setStyleSheet(
            "QFrame#moduleTrayItem {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "                           stop:0 #252830, stop:1 #1c1e24);"
            "  border: 1px solid #5c636e;"
            "  border-radius: 6px;"
            "}"
            f"QFrame#moduleTrayAccent {{ background: {accent_color}; border: 0; }}"
            "QFrame#moduleTrayItem:hover {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            "                           stop:0 #2b3039, stop:1 #21252c);"
            f"  border: 1px solid {accent_color};"
            "}"
            f"QLabel#moduleTrayIcon {{ color: {accent_color}; font-size: 16px; font-weight: 700; }}"
            "QLabel#moduleTrayName { color: #d8d8d8; font-size: 11px; font-weight: 700; }"
            "QLabel#moduleTrayDescription { color: #9aa3ae; font-size: 10px; }"
            "QLabel#moduleTrayGrab {"
            "  color: #7e8896; font-size: 11px; font-weight: 700;"
            "  letter-spacing: 1px; padding: 0 2px;"
            "}"
        )

        self._shell = QHBoxLayout(self)
        self._shell.setContentsMargins(0, 0, 0, 0)
        self._shell.setSpacing(0)

        accent = QFrame(self)
        accent.setObjectName("moduleTrayAccent")
        accent.setFixedWidth(4)
        self._shell.addWidget(accent, 0)

        self._body = QWidget(self)
        self._shell.addWidget(self._body, 1)

        self._layout = QHBoxLayout(self._body)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(8)

        self._icon_label = QLabel(icon_glyph)
        self._icon_label.setObjectName("moduleTrayIcon")
        self._layout.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self._text_col = QVBoxLayout()
        self._text_col.setContentsMargins(0, 0, 0, 0)
        self._text_col.setSpacing(2)

        self._name_label = QLabel(self._display_name)
        self._name_label.setObjectName("moduleTrayName")
        self._text_col.addWidget(self._name_label, 0, Qt.AlignmentFlag.AlignLeft)

        self._description_label = QLabel("")
        self._description_label.setObjectName("moduleTrayDescription")
        self._text_col.addWidget(self._description_label, 0, Qt.AlignmentFlag.AlignLeft)

        self._layout.addLayout(self._text_col, 1)

        self._grab_handle = QLabel("|||")
        self._grab_handle.setObjectName("moduleTrayGrab")
        self._grab_handle.setFixedWidth(18)
        self._layout.addWidget(self._grab_handle, 0, Qt.AlignmentFlag.AlignRight)

        self._update_tooltip()
        self._apply_responsive_state(self.width())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_responsive_state(event.size().width())

    def _elided_description(self, max_width: int) -> str:
        description = self._module_description or "No module description available yet."
        if max_width <= 0:
            return description
        metrics = QFontMetrics(self._description_label.font())
        return metrics.elidedText(description, Qt.TextElideMode.ElideRight, max_width)

    def _apply_responsive_state(self, width: int) -> None:
        compact = width < 290
        very_compact = width < 230

        self._icon_label.setVisible(not very_compact)
        self._description_label.setVisible(not compact)

        if very_compact:
            self._layout.setContentsMargins(8, 6, 8, 6)
            self._layout.setSpacing(6)
        else:
            self._layout.setContentsMargins(10, 8, 10, 8)
            self._layout.setSpacing(8)

        description_width = max(80, width - 150 if not compact else width - 90)
        self._description_label.setText(self._elided_description(description_width))

    def _update_tooltip(self) -> None:
        description = self._module_description or "No module description available yet."
        self.setToolTip(
            f"{self._display_name}\n"
            f"ID: {self.payload_name}\n"
            f"v{self._module_version}\n"
            f"{description}\n\n"
            "Drag into the rack to add this module."
        )

    def enterEvent(self, event: Any) -> None:
        self._update_tooltip()
        super().enterEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_pos is None
            or not (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            super().mouseMoveEvent(event)
            return

        if (
            event.pos() - self._drag_start_pos
        ).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        mime = QMimeData()
        mime.setData(MODULE_TRAY_MIME, self.payload_name.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.exec(Qt.DropAction.CopyAction)

        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_start_pos = None
        super().mouseMoveEvent(event)


class ModuleTray(QWidget):
    def __init__(
        self,
        plugins: list[dict[str, Any]],
        module_registry: dict[str, type],
        hidden_module_names: set[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plugins = plugins
        self._module_registry = module_registry
        self._hidden_module_names = {
            name.strip().lower() for name in (hidden_module_names or set()) if name
        }
        self._max_grid_columns = 5
        self._min_item_width = 180
        self._items: list[ModuleTrayItem] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        title = QLabel("MODULES")
        title.setStyleSheet(
            "color: #5c636e; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;"
        )
        root.addWidget(title)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        root.addWidget(self._scroll, 1)

        content = QWidget()
        self._items_layout = QGridLayout(content)
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(6)
        self._scroll.setWidget(content)

        self._populate()

    def add_module(
        self,
        name: str,
        register_fn: Any,
        module_cls: type | None,
    ) -> None:
        """Add or refresh a module in the tray at runtime."""
        module_name = str(name or "").strip()
        if not module_name:
            return
        plugin_found = False
        for plugin in self._plugins:
            if str(plugin.get("name", "")).strip() == module_name:
                plugin["register_fn"] = register_fn
                plugin_found = True
                break
        if not plugin_found:
            self._plugins.append({"name": module_name, "register_fn": register_fn})
        if module_cls is not None:
            self._module_registry[module_name] = module_cls
        self._populate()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._reflow_grid)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._reflow_grid()

    def _current_column_count(self) -> int:
        viewport_w = self._scroll.viewport().width()
        spacing = max(0, self._items_layout.horizontalSpacing())
        if viewport_w <= 0:
            return 1
        slot_w = self._min_item_width + spacing
        cols = max(1, (viewport_w + spacing) // max(1, slot_w))
        return max(1, min(self._max_grid_columns, int(cols)))

    def _reflow_grid(self) -> None:
        while self._items_layout.count():
            self._items_layout.takeAt(0)

        columns = self._current_column_count()
        for idx, item in enumerate(self._items):
            row = idx // columns
            col = idx % columns
            self._items_layout.addWidget(item, row, col)

        for row in range(len(self._items) + 2):
            self._items_layout.setRowStretch(row, 0)
        final_row = max((len(self._items) + columns - 1) // columns, 0)
        self._items_layout.setRowStretch(final_row, 1)

        for col in range(self._max_grid_columns):
            self._items_layout.setColumnStretch(col, 1 if col < columns else 0)

    def _populate(self) -> None:
        while self._items_layout.count():
            item = self._items_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._items.clear()

        plugin_names = []
        for plugin in self._plugins:
            name = plugin.get("name")
            if isinstance(name, str):
                plugin_names.append(name)

        names = sorted(set(plugin_names) | set(self._module_registry.keys()))
        for name in names:
            if name.strip().lower() in self._hidden_module_names:
                continue
            module_cls = self._module_registry.get(name)
            description = ""
            version = "0.1.0"
            span = 1
            payload_name = name
            if module_cls is not None:
                description = str(getattr(module_cls, "MODULE_DESCRIPTION", ""))
                version = str(getattr(module_cls, "MODULE_VERSION", "0.1.0"))
                span = int(getattr(module_cls, "UI_COL_SPAN", 1))
                class_module = str(getattr(module_cls, "__module__", ""))
                parts = class_module.split(".")
                if len(parts) >= 3 and parts[0] == "modules":
                    payload_name = parts[1]

            item = ModuleTrayItem(
                payload_name=payload_name,
                display_name=module_ui_label(name),
                module_description=description,
                module_version=version,
                ui_col_span=span,
            )
            self._items.append(item)

        self._reflow_grid()
