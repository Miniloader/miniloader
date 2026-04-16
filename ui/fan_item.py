"""
fan_item.py — Spinning CPU-fan animation for the rack UI.

Drop a SpinningFanItem anywhere in the scene or as a child item,
position it with .setPos(cx, cy), then call .advance() from any
existing QTimer tick to animate it.

Draws blades behind a protective wire grill — the grill stays fixed
while the impeller rotates underneath.
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem


class SpinningFanItem(QGraphicsItem):
    """
    Decorative CPU-style fan with a wire finger-guard drawn over
    the spinning impeller.

    Call advance() once per timer tick to step the rotation.
    """

    _RADIUS = 23.0
    _HUB_R  =  6.5
    _BLADES =  7
    _STEP   = 2

    def __init__(self, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self._angle: float = 0.0

    def boundingRect(self) -> QRectF:
        r = self._RADIUS + 4.0
        return QRectF(-r, -r, r * 2, r * 2)

    def advance(self) -> None:
        self._angle = (self._angle + self._STEP) % 360
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        R   = self._RADIUS
        HUB = self._HUB_R
        N   = self._BLADES

        # ── Outer housing (square frame with rounded corners) ─────
        frame = R + 3.0
        painter.setPen(QPen(QColor("#2a2d32"), 1.8))
        painter.setBrush(QBrush(QColor("#101215")))
        painter.drawRoundedRect(QRectF(-frame, -frame, frame * 2, frame * 2), 4.0, 4.0)

        # ── Circular bore (dark well the blades spin in) ──────────
        painter.setPen(QPen(QColor("#1a1d22"), 1.0))
        painter.setBrush(QBrush(QColor("#080b0d")))
        painter.drawEllipse(QRectF(-R - 1, -R - 1, (R + 1) * 2, (R + 1) * 2))

        # ── Blades (rotated) ─────────────────────────────────────
        painter.save()
        painter.rotate(self._angle)

        blade_pen = QPen(QColor("#1a5035"), 0.5)
        blade_brush = QBrush(QColor("#1e3a2c"))

        for i in range(N):
            base_a = math.radians(i * 360.0 / N)

            # Root attachment on the hub
            r0x = math.cos(base_a) * HUB * 0.8
            r0y = math.sin(base_a) * HUB * 0.8

            # Wide leading-edge attachment (offset clockwise from root)
            le_root_a = base_a + math.radians(18)
            le_rx = math.cos(le_root_a) * HUB * 0.7
            le_ry = math.sin(le_root_a) * HUB * 0.7

            # Blade tip — swept forward
            tip_a = base_a + math.radians(32)
            tip_r = R - 1.5
            tip_x = math.cos(tip_a) * tip_r
            tip_y = math.sin(tip_a) * tip_r

            # Second tip point (gives the blade width at the rim)
            tip2_a = base_a + math.radians(12)
            tip2_x = math.cos(tip2_a) * (tip_r - 0.5)
            tip2_y = math.sin(tip2_a) * (tip_r - 0.5)

            # Control points for the curved leading and trailing edges
            mid_a = base_a + math.radians(24)
            le_cx = math.cos(mid_a) * R * 0.68
            le_cy = math.sin(mid_a) * R * 0.68

            tr_a = base_a + math.radians(4)
            tr_cx = math.cos(tr_a) * R * 0.55
            tr_cy = math.sin(tr_a) * R * 0.55

            path = QPainterPath()
            path.moveTo(le_rx, le_ry)
            path.quadTo(le_cx, le_cy, tip_x, tip_y)
            path.lineTo(tip2_x, tip2_y)
            path.quadTo(tr_cx, tr_cy, r0x, r0y)
            path.closeSubpath()

            painter.setPen(blade_pen)
            painter.setBrush(blade_brush)
            painter.drawPath(path)

        painter.restore()

        # ── Hub cap (drawn above blades, below grill) ─────────────
        painter.setPen(QPen(QColor("#1a1d22"), 1.0))
        painter.setBrush(QBrush(QColor("#3c3f46")))
        painter.drawEllipse(QRectF(-HUB, -HUB, HUB * 2, HUB * 2))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#14171c")))
        painter.drawEllipse(QRectF(-1.8, -1.8, 3.6, 3.6))

        # ── Wire grill (static, drawn on top of everything) ──────
        grill_pen = QPen(QColor("#3a3e45"), 1.4)
        grill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(grill_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Concentric rings
        for ring_r in (R * 0.45, R * 0.72, R * 0.95):
            painter.drawEllipse(QRectF(-ring_r, -ring_r, ring_r * 2, ring_r * 2))

        # Radial spokes
        spoke_count = 8
        for i in range(spoke_count):
            a = math.radians(i * 360.0 / spoke_count + 22.5)
            inner_r = HUB + 1.5
            outer_r = R * 0.95
            painter.drawLine(
                QPointF(math.cos(a) * inner_r, math.sin(a) * inner_r),
                QPointF(math.cos(a) * outer_r, math.sin(a) * outer_r),
            )

        # ── Corner screws ────────────────────────────────────────
        screw_r = 2.8
        slot_w, slot_h = screw_r * 1.3, 1.0
        inset = frame - 4.0
        for sx, sy in ((-inset, -inset), (inset, -inset),
                       (-inset,  inset), (inset,  inset)):
            painter.setPen(QPen(QColor("#2a2d32"), 0.8))
            painter.setBrush(QBrush(QColor("#3a3d42")))
            painter.drawEllipse(QRectF(sx - screw_r, sy - screw_r,
                                       screw_r * 2, screw_r * 2))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#22242a")))
            painter.drawRect(QRectF(sx - slot_w / 2, sy - slot_h / 2,
                                    slot_w, slot_h))
