"""
api_tooltip.py — API Reference Overlay
========================================
Styled overlay panel shown when the user clicks the API button on
a GPT Server card.  Displays the OpenAI-compatible base URL,
Python SDK snippet, and endpoint list.  Click anywhere on the panel
(or click the API button again) to dismiss it.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


# ── Code colour palette (VSCode Dark+ inspired) ─────────────────

_COL_KEYWORD  = QColor("#569cd6")   # blue   — from / import / class
_COL_BUILTIN  = QColor("#4ec9b0")   # teal   — OpenAI / client names
_COL_STRING   = QColor("#ce9178")   # orange — string literals
_COL_PARAM    = QColor("#9cdcfe")   # light blue — parameter names
_COL_COMMENT  = QColor("#6a9955")   # green  — # comment
_COL_PLAIN    = QColor("#d4d4d4")   # white-ish — default text
_COL_PUNCT    = QColor("#d4d4d4")


def _code_lines(port: int) -> list[tuple[str, QColor]]:
    """Return (text, color) pairs for the SDK snippet."""
    p = port
    kw  = _COL_KEYWORD
    bi  = _COL_BUILTIN
    st  = _COL_STRING
    pm  = _COL_PARAM
    pl  = _COL_PLAIN
    co  = _COL_COMMENT
    return [
        (f"from openai import OpenAI",           kw),
        (f"",                                     pl),
        (f"client = OpenAI(",                     pl),
        (f'  base_url="{f"http://127.0.0.1:{p}/v1"}",', pm),
        (f'  api_key="not-needed"',               pm),
        (f")",                                    pl),
        (f"",                                     pl),
        (f"resp = client.chat.completions.create(", pl),
        (f'  model="local",',                    pm),
        (f'  messages=[',                         pl),
        (f'    {{"role": "user",',                st),
        (f'     "content": "Hello!"}}]',          st),
        (f")",                                    pl),
    ]


_ENDPOINTS = [
    ("POST", "/v1/chat/completions", QColor("#e8b830")),
    ("GET ",  "/v1/models",           QColor("#43b4e5")),
    ("GET ",  "/v1/health",           QColor("#43b4e5")),
]


class ApiTooltipWidget(QWidget):
    """Floating API reference panel.

    Parented to the QGraphicsView viewport so it stays in view
    coordinates.  Clicking anywhere on the widget dismisses it.
    """

    _W   = 284
    _PAD = 10
    _LH  = 14   # normal line height
    _CLH = 13   # code line height

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setVisible(False)
        self._port = 5000
        self._recompute_size()

    # ── Public API ────────────────────────────────────────────────

    def show_at(self, port: int, viewport_pos: QPoint) -> None:
        """Toggle the panel at *viewport_pos*.  Hides if already visible."""
        if self.isVisible():
            self.setVisible(False)
            return

        self._port = port
        self._recompute_size()
        self.update()

        vw = self.parentWidget().width()
        vh = self.parentWidget().height()
        x = viewport_pos.x() + 14
        y = viewport_pos.y() - self.height() // 3
        x = max(4, min(x, vw - self._W - 4))
        y = max(4, min(y, vh - self.height() - 4))
        self.move(x, y)
        self.setVisible(True)
        self.raise_()

    # ── Layout ────────────────────────────────────────────────────

    def _recompute_size(self) -> None:
        lines  = _code_lines(self._port)
        LH, CLH, PAD = self._LH, self._CLH, self._PAD
        h = PAD           # top pad
        h += 16           # header title
        h += 11           # dismiss hint
        h += 10           # separator
        h += LH           # BASE URL label
        h += LH + 2       # URL value
        h += 10           # separator
        h += LH           # PYTHON SDK label
        h += len(lines) * CLH + 8    # code block (with inner pad)
        h += 10           # separator
        h += LH           # ENDPOINTS label
        h += len(_ENDPOINTS) * LH
        h += PAD          # bottom pad
        self.setFixedSize(self._W, h)

    # ── Interaction ───────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.setVisible(False)
        event.accept()

    # ── Painting ─────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # type: ignore[override]
        lines = _code_lines(self._port)
        W, PAD, LH, CLH = self._W, self._PAD, self._LH, self._CLH
        H = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # ── Background ───────────────────────────────────────────
        painter.setPen(QPen(QColor("#4a5264"), 1.5))
        painter.setBrush(QBrush(QColor("#1a1d26")))
        painter.drawRoundedRect(1, 1, W - 2, H - 2, 6, 6)

        y = PAD

        # ── Header ───────────────────────────────────────────────
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#43b4e5")))
        painter.drawText(PAD, y + 12, "OpenAI-Compatible API")
        y += 16

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#363e50")))
        painter.drawText(PAD + 2, y + 9, "click anywhere to dismiss")
        y += 11

        self._sep(painter, y, W, PAD)
        y += 10

        # ── Base URL ─────────────────────────────────────────────
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#485060")))
        painter.drawText(PAD + 2, y + 9, "BASE URL")
        y += LH

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#e8b830")))
        painter.drawText(PAD + 2, y + 11, f"http://127.0.0.1:{self._port}/v1")
        y += LH + 2

        self._sep(painter, y, W, PAD)
        y += 10

        # ── Python SDK code block ─────────────────────────────────
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#485060")))
        painter.drawText(PAD + 2, y + 9, "PYTHON  SDK  ( pip install openai )")
        y += LH

        code_h = len(lines) * CLH + 8
        painter.setPen(QPen(QColor("#111418"), 1.0))
        painter.setBrush(QBrush(QColor("#12141c")))
        painter.drawRoundedRect(PAD, y, W - 2 * PAD, code_h, 3, 3)

        code_font = QFont("Consolas", 7)
        y_code = y + 4
        for line_text, line_color in lines:
            painter.setFont(code_font)
            painter.setPen(QPen(line_color))
            painter.drawText(PAD + 6, y_code + CLH - 3, line_text)
            y_code += CLH

        y += code_h + 6

        self._sep(painter, y, W, PAD)
        y += 10

        # ── Endpoints ────────────────────────────────────────────
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QPen(QColor("#485060")))
        painter.drawText(PAD + 2, y + 9, "ENDPOINTS")
        y += LH

        method_font = QFont("Consolas", 7, QFont.Weight.Bold)
        path_font   = QFont("Consolas", 7)

        for method, path, color in _ENDPOINTS:
            # colored dot
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(PAD + 2, y + 3, 7, 7, 1, 1)

            painter.setFont(method_font)
            painter.setPen(QPen(color))
            painter.drawText(PAD + 13, y + 11, method)

            painter.setFont(path_font)
            painter.setPen(QPen(QColor("#a0aab8")))
            painter.drawText(PAD + 13 + 34, y + 11, path)
            y += LH

    @staticmethod
    def _sep(painter: QPainter, y: int, W: int, PAD: int) -> None:
        painter.setPen(QPen(QColor("#2c3040")))
        painter.drawLine(PAD, y, W - PAD, y)
