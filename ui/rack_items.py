"""
rack_items.py — Rack UI Building Blocks
=========================================
Shared constants, helper functions, and reusable QGraphicsItem
subclasses used throughout the rack canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from core.port_system import Port
    from ui.main_window import RackWindow

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
)

from core.base_module import ModuleStatus
from core.port_system import SignalType

# Shared color maps.
# Port-name overrides take precedence over signal-type colors.
# This keeps specific ports visually consistent across signal sets.
PORT_NAME_COLORS: dict[str, QColor] = {
    "LOCAL_IP_OUT":  QColor("#43b4e5"),  # inference bus — sky blue
    "LOCAL_IP_IN":   QColor("#43b4e5"),
    "BRAIN_OUT":     QColor("#d93232"),  # brain — red
    "BRAIN_IN":      QColor("#d93232"),
    "MCP_OUT":       QColor("#e08b3a"),  # MCP bus — orange
    "MCP_IN":        QColor("#e08b3a"),
    "TOOLS_OUT":     QColor("#e08b3a"),
    "MCP_UPSTREAM":  QColor("#e08b3a"),
    "MCP_DOWNSTREAM": QColor("#e08b3a"),
    "WEB_OUT":       QColor("#5a66c8"),  # web/tunnel — indigo
    "WEB_IN":        QColor("#5a66c8"),
    "DB_IN_OUT":     QColor("#2d93a2"),  # database — teal
    "FILES_OUT":     QColor("#d4932c"),  # file docs — amber
    "FILES_IN":      QColor("#d4932c"),
    "CONTEXT_OUT":   QColor("#80c65f"),  # context/RAG — green
    "CONTEXT_IN":    QColor("#80c65f"),
}

JACK_COLORS: dict[SignalType, QColor] = {
    SignalType.DOCS_PAYLOAD: QColor("#d4932c"),
    SignalType.QUERY_PAYLOAD: QColor("#d1c748"),
    SignalType.CONTEXT_PAYLOAD: QColor("#80c65f"),
    SignalType.BRAIN_STREAM_PAYLOAD: QColor("#43b4e5"),
    SignalType.CHAT_REQUEST: QColor("#4e8fe8"),
    SignalType.SERVER_CONFIG_PAYLOAD: QColor("#8a57e8"),
    SignalType.TOOL_SCHEMA_PAYLOAD: QColor("#e08b3a"),
    SignalType.TOOL_EXECUTION_PAYLOAD: QColor("#d46535"),
    SignalType.DB_QUERY_PAYLOAD: QColor("#2d93a2"),
    SignalType.DB_TRANSACTION_PAYLOAD: QColor("#206f7b"),
    SignalType.DB_RESPONSE_PAYLOAD: QColor("#3ea874"),
    SignalType.ROUTING_CONFIG_PAYLOAD: QColor("#5a66c8"),
    SignalType.TUNNEL_STATUS_PAYLOAD: QColor("#838383"),
    SignalType.MODEL_LOAD_PAYLOAD: QColor("#f3a95f"),
    SignalType.SYSTEM_STATE_PAYLOAD: QColor("#66c6ff"),
    SignalType.PORT_ERROR_PAYLOAD: QColor("#c94848"),
}

_LOG_GREEN = QColor("#30c040")
_LOG_AMBER = QColor("#d4a020")
_LOG_RED = QColor("#e04040")
_LOG_CYAN = QColor("#40c8e0")
_LOG_DIM = QColor("#607060")

# ── Shared fonts ─────────────────────────────────────────────────

_MONO_FONT = QFont("Consolas", 9)
_MONO_FONT.setStyleHint(QFont.StyleHint.Monospace)

_LCD_FONT = QFont("Consolas", 8)
_LCD_FONT.setStyleHint(QFont.StyleHint.Monospace)

_TITLE_FONT = QFont("Consolas", 13, QFont.Weight.Bold)
_TITLE_FONT.setStyleHint(QFont.StyleHint.Monospace)

# ── Screw appearance ─────────────────────────────────────────────

_SCREW_COLOR = QColor("#3a3d42")
_SCREW_SLOT = QColor("#22242a")


# ── Helper functions ─────────────────────────────────────────────

def _status_color(status: ModuleStatus) -> QColor:
    if status == ModuleStatus.READY:
        return QColor("#39d353")
    if status == ModuleStatus.RUNNING:
        return QColor("#d68c1a")
    if status == ModuleStatus.LOADING:
        return QColor("#d6b745")
    if status == ModuleStatus.ERROR:
        return QColor("#e24c4c")
    if status == ModuleStatus.STOPPED:
        return QColor("#6f6f6f")
    return QColor("#909090")


def _port_color(port: Port) -> QColor:
    if port.name in PORT_NAME_COLORS:
        return PORT_NAME_COLORS[port.name]
    signal = sorted(port.accepted_signals, key=lambda s: s.value)[0]
    return JACK_COLORS.get(signal, QColor("#8c8c8c"))


def _wire_color(source: Port, target: Port) -> QColor:
    if source.name in PORT_NAME_COLORS:
        return PORT_NAME_COLORS[source.name]
    if target.name in PORT_NAME_COLORS:
        return PORT_NAME_COLORS[target.name]
    shared = source.accepted_signals & target.accepted_signals
    if not shared:
        return QColor("#9b9b9b")
    signal = sorted(shared, key=lambda s: s.value)[0]
    return JACK_COLORS.get(signal, QColor("#9b9b9b"))


def _draw_screw(
    scene: QGraphicsScene | None,
    cx: float,
    cy: float,
    parent: QGraphicsItem | None = None,
) -> None:
    r = 5.0
    outer = QGraphicsEllipseItem(-r, -r, r * 2, r * 2, parent)
    outer.setPos(cx, cy)
    outer.setBrush(QBrush(_SCREW_COLOR))
    outer.setPen(QPen(QColor("#2a2d32"), 0.8))
    outer.setZValue(10)
    if parent is None:
        scene.addItem(outer)  # type: ignore[union-attr]

    slot_w, slot_h = r * 1.4, 1.2
    slot = QGraphicsRectItem(-slot_w / 2, -slot_h / 2, slot_w, slot_h, outer)
    slot.setBrush(QBrush(_SCREW_SLOT))
    slot.setPen(QPen(Qt.PenStyle.NoPen))
    slot.setZValue(11)


# ── Graphics items ───────────────────────────────────────────────

@dataclass
class PendingWire:
    source_port_id: str


class PortJackItem(QGraphicsEllipseItem):
    """Clickable graphics item for a module port."""

    _HOVER_DELAY_MS = 1100

    def __init__(self, port: Port, controller: RackWindow, x: float, y: float) -> None:
        r = 6.0
        super().__init__(-r, -r, r * 2, r * 2)
        self.port = port
        self.controller = controller
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#1c1c1c")))
        self.setPen(QPen(_port_color(port), 2))
        self.setZValue(5)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.controller.on_port_clicked(self)
        event.accept()

    def hoverEnterEvent(self, event) -> None:  # type: ignore[override]
        self.controller.on_port_hover_enter(self)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.controller.on_port_hover_leave(self)
        super().hoverLeaveEvent(event)


class PortLabelItem(QGraphicsSimpleTextItem):
    """Clickable port legend with a tight text-only hit shape."""

    def __init__(
        self,
        text: str,
        jack: PortJackItem,
        controller: RackWindow,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._jack = jack
        self._controller = controller
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def shape(self) -> QPainterPath:  # type: ignore[override]
        # Keep the click target tight to visible glyphs so label hitboxes do not bleed.
        fm = QFontMetricsF(self.font())
        tight = fm.tightBoundingRect(self.text())
        if tight.isNull():
            tight = self.boundingRect()
        path = QPainterPath()
        path.addRect(tight.adjusted(-0.35, -0.35, 0.35, 0.35))
        return path

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._controller.on_port_clicked(self._jack)
        event.accept()

    def hoverEnterEvent(self, event) -> None:  # type: ignore[override]
        self._controller.on_port_hover_enter(self._jack)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._controller.on_port_hover_leave(self._jack)
        super().hoverLeaveEvent(event)


class ModuleCardItem(QGraphicsRectItem):
    """Wireframe rack card for a module, with corner screws.  Draggable for reordering."""

    TITLE_BAR_HEIGHT = 56.0

    def __init__(self, module: BaseModule, width: float, height: float,
                 scene: QGraphicsScene, controller: RackWindow) -> None:
        super().__init__(0, 0, width, height)
        self.module = module
        self._controller = controller
        self._drag_origin: QPointF | None = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

        grad = QLinearGradient(0, 0, 0, height)
        grad.setColorAt(0.0, QColor("#252830"))
        grad.setColorAt(1.0, QColor("#1c1e24"))
        self.setBrush(QBrush(grad))
        self.setPen(QPen(QColor("#5c636e"), 1.5))

        for cx, cy in [(8, 8), (width - 8, 8), (8, height - 8), (width - 8, height - 8)]:
            _draw_screw(scene, cx, cy, parent=self)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self._controller.on_card_right_clicked(self, event.screenPos(), event.pos())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if getattr(self._controller, "pending_wire", None) is not None:
                event.ignore()
                return
            if event.pos().y() > self.TITLE_BAR_HEIGHT:
                event.ignore()
                return
            self._drag_origin = self.pos()
            self._controller._on_card_drag_started(self.module.module_id)
            self.setOpacity(0.5)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.setZValue(50)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        super().mouseMoveEvent(event)
        if self._drag_origin is not None:
            self._controller._on_card_drag_move(
                self.module.module_id,
                self.sceneBoundingRect().center(),
            )
            vp_y = int(self._controller.view.mapFromScene(event.scenePos()).y())
            self._controller._update_autoscroll(vp_y)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            drop_pos = self.pos()
            self.setPos(self._drag_origin)
            self.setOpacity(1.0)
            self._set_hover_cursor(float(event.pos().y()))
            self.setZValue(0)
            moved = (drop_pos - self._drag_origin).manhattanLength() > 20
            self._drag_origin = None
            self._controller._on_card_drag_ended(self.module.module_id, moved)
            super().mouseReleaseEvent(event)
            if moved:
                controller = self._controller
                mid = self.module.module_id
                QTimer.singleShot(0, lambda: controller._on_card_dropped(mid, drop_pos))
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_origin is None:
            self._set_hover_cursor(float(event.pos().y()))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_origin is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def _set_hover_cursor(self, local_y: float) -> None:
        if local_y <= self.TITLE_BAR_HEIGHT:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)


class DragGripItem(QGraphicsItem):
    """Subtle dot grip to indicate draggable title bar area."""

    DOT_SIZE = 3.0
    DOT_RADIUS = 0.9
    COLS = 2
    ROWS = 3
    COL_GAP = 3.0
    ROW_GAP = 4.0
    SHADOW_OFFSET = 0.5

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        width = self.COLS * self.DOT_SIZE + (self.COLS - 1) * self.COL_GAP
        height = self.ROWS * self.DOT_SIZE + (self.ROWS - 1) * self.ROW_GAP
        self._rect = QRectF(0.0, 0.0, width, height)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return self._rect.adjusted(-0.5, -0.5, 0.5, 0.5)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        shadow_brush = QBrush(QColor("#1a1c20"))
        dot_brush = QBrush(QColor("#3a3f4a"))
        for row in range(self.ROWS):
            y = row * (self.DOT_SIZE + self.ROW_GAP)
            for col in range(self.COLS):
                x = col * (self.DOT_SIZE + self.COL_GAP)
                painter.setBrush(shadow_brush)
                painter.drawRoundedRect(
                    QRectF(
                        x + self.SHADOW_OFFSET,
                        y + self.SHADOW_OFFSET,
                        self.DOT_SIZE,
                        self.DOT_SIZE,
                    ),
                    self.DOT_RADIUS,
                    self.DOT_RADIUS,
                )
                painter.setBrush(dot_brush)
                painter.drawRoundedRect(
                    QRectF(
                        x,
                        y,
                        self.DOT_SIZE,
                        self.DOT_SIZE,
                    ),
                    self.DOT_RADIUS,
                    self.DOT_RADIUS,
                )

class WirePathItem(QGraphicsPathItem):
    """Rendered bezier wire between two port jacks. Right-click removes the wire."""

    _HIT_WIDTH = 12.0

    def __init__(self, source_port_id: str, target_port_id: str, color: QColor,
                 controller: RackWindow | None = None) -> None:
        super().__init__()
        self.source_port_id = source_port_id
        self.target_port_id = target_port_id
        self._controller = controller
        self.setPen(QPen(color, 2.2))
        self.setZValue(1)
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton
        )

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(self._HIT_WIDTH)
        return stroker.createStroke(self.path())

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton and self._controller is not None:
            self._controller.on_wire_right_clicked(self)
            event.accept()
            return
        super().mousePressEvent(event)


class CardButtonItem(QGraphicsRectItem):
    """Simple clickable button for module cards."""

    def __init__(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        on_click,
        parent: QGraphicsItem,
        *,
        label_font: QFont | None = None,
    ) -> None:
        super().__init__(0, 0, w, h, parent)
        self.setPos(x, y)
        self._on_click = on_click
        self.setBrush(QBrush(QColor("#2a2e36")))
        self.setPen(QPen(QColor("#626a77"), 1.0))
        self.setZValue(3)
        txt = QGraphicsSimpleTextItem(label, self)
        if label_font is not None:
            txt.setFont(label_font)
        txt.setBrush(QBrush(QColor("#d3d8df")))
        rect = txt.boundingRect()
        txt.setPos((w - rect.width()) / 2, (h - rect.height()) / 2 - 1)
        txt.setZValue(4)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._on_click()
        event.accept()
