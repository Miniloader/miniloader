"""
port_tooltip.py — Port Tooltip Overlay
========================================
Custom styled tooltip overlay shown after hovering over a port jack
for a short delay.  Parented to the QGraphicsView viewport so it
stays in view coordinates and isn't clipped by the scene.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.port_system import Port

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from core.port_system import ConnectionMode, PortDirection, SignalType
from ui.label_aliases import port_ui_label
from ui.rack_items import JACK_COLORS, _port_color

_SIGNAL_READABLE: dict[SignalType, str] = {
    SignalType.DOCS_PAYLOAD:           "Document text chunks",
    SignalType.CONTEXT_PAYLOAD:        "RAG context for LLM",
    SignalType.QUERY_PAYLOAD:          "User query string",
    SignalType.TOOL_SCHEMA_PAYLOAD:    "MCP tool schema",
    SignalType.TOOL_EXECUTION_PAYLOAD: "MCP tool result",
    SignalType.CHAT_REQUEST:           "Chat message / request",
    SignalType.BRAIN_STREAM_PAYLOAD:   "Streamed LLM tokens",
    SignalType.DB_QUERY_PAYLOAD:       "Database SELECT query",
    SignalType.DB_RESPONSE_PAYLOAD:    "Database query result",
    SignalType.DB_TRANSACTION_PAYLOAD: "Database write op",
    SignalType.ROUTING_CONFIG_PAYLOAD: "Local server address",
    SignalType.TUNNEL_STATUS_PAYLOAD:  "Public tunnel URL",
    SignalType.SERVER_CONFIG_PAYLOAD:  "Server address & caps",
    SignalType.MODEL_LOAD_PAYLOAD:     "Model load command",
    SignalType.SYSTEM_STATE_PAYLOAD:   "Health telemetry",
    SignalType.PORT_ERROR_PAYLOAD:     "Port error report",
}


class PortTooltipWidget(QWidget):
    """
    Custom styled tooltip overlay shown after hovering over a port jack
    for a short delay.  Parented to the QGraphicsView viewport so it
    stays in view coordinates and isn't clipped by the scene.
    """

    _W = 256
    _PAD = 10
    _LINE_H = 14
    _WRAP = 33

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setVisible(False)
        self._port: Port | None = None
        self._desc_lines: list[str] = []
        self._sig_list: list[tuple[QColor, str, str]] = []

    def set_port(self, port: Port) -> None:
        self._port = port
        desc = port.description if hasattr(port, "description") else ""
        self._desc_lines = textwrap.wrap(desc, width=self._WRAP) if desc else []
        self._sig_list = [
            (
                JACK_COLORS.get(s, QColor("#8c8c8c")),
                s.value,
                _SIGNAL_READABLE.get(s, ""),
            )
            for s in sorted(port.accepted_signals, key=lambda s: s.value)
        ]
        self._recompute_size()
        self.update()

    def _recompute_size(self) -> None:
        LH = self._LINE_H
        h = self._PAD
        h += 17
        h += LH
        h += 8
        if self._desc_lines:
            h += len(self._desc_lines) * LH
            h += 8
        h += LH
        for _, val, readable in self._sig_list:
            h += LH
            if readable:
                h += LH - 3
        h += self._PAD
        self.setFixedSize(self._W, h)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self._port is None:
            return

        port = self._port
        W, PAD, LH = self._W, self._PAD, self._LINE_H
        H = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#4a5264"), 1.5))
        painter.setBrush(QBrush(QColor("#1a1d26")))
        painter.drawRoundedRect(1, 1, W - 2, H - 2, 6, 6)

        dot_color = _port_color(port)
        y = PAD

        # Header: colored dot + port name + direction badge
        painter.setPen(QPen(dot_color.darker(130), 1.0))
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(PAD, y + 4, 8, 8)

        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#e8eaed")))
        painter.drawText(PAD + 13, y + 13, port_ui_label(port.name))

        is_channel = port.connection_mode == ConnectionMode.CHANNEL
        is_out = port.direction == PortDirection.OUT
        if is_channel:
            badge_bg  = QColor("#1e2240")
            badge_fg  = QColor("#7a8ae0")
            badge_txt = "CHANNEL"
        elif is_out:
            badge_bg  = QColor("#1a3828")
            badge_fg  = QColor("#55c880")
            badge_txt = "OUT"
        else:
            badge_bg  = QColor("#302c14")
            badge_fg  = QColor("#c8b840")
            badge_txt = "IN"
        painter.setFont(QFont("Consolas", 7))
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(badge_txt) + 8
        bx = W - PAD - bw
        painter.setPen(QPen(badge_fg.darker(140), 0.8))
        painter.setBrush(QBrush(badge_bg))
        painter.drawRoundedRect(bx, y + 2, bw, 12, 3, 3)
        painter.setPen(QPen(badge_fg))
        painter.drawText(bx + 4, y + 11, badge_txt)

        y += 17

        # Mode / max-connections
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#5a6070")))
        mode_str = f"{port.connection_mode.value}  ·  max {port.max_connections}"
        painter.drawText(PAD + 2, y + 10, mode_str)
        y += LH

        # Separator
        y += 3
        painter.setPen(QPen(QColor("#2c3040")))
        painter.drawLine(PAD, y, W - PAD, y)
        y += 5

        # Description
        if self._desc_lines:
            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QPen(QColor("#a0aab8")))
            for line in self._desc_lines:
                painter.drawText(PAD + 2, y + 11, line)
                y += LH
            y += 3
            painter.setPen(QPen(QColor("#2c3040")))
            painter.drawLine(PAD, y, W - PAD, y)
            y += 5

        # Signals section
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#484e60")))
        painter.drawText(PAD + 2, y + 10, "SIGNALS")
        y += LH

        for sig_color, val, readable in self._sig_list:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(sig_color))
            painter.drawRoundedRect(PAD + 2, y + 3, 7, 7, 1, 1)

            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QPen(QColor("#8a96a8")))
            painter.drawText(PAD + 14, y + 11, val)
            y += LH

            if readable:
                painter.setFont(QFont("Consolas", 7))
                painter.setPen(QPen(QColor("#48505e")))
                painter.drawText(PAD + 14, y + 9, readable)
                y += LH - 3
