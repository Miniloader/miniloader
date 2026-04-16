"""
wire_renderer.py — Bezier Curve Math for Rack Cables
======================================================
Draws the visual patch cables between module port jacks.

The renderer is purely cosmetic — it reads the Hypervisor's
active_wires list and maps source/target port screen coordinates
to smooth cubic Bezier curves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.port_system import Wire


def compute_bezier_points(
    x0: float, y0: float,
    x1: float, y1: float,
    segments: int = 32,
) -> list[tuple[float, float]]:
    """
    Given two jack coordinates (source and target), return a list
    of (x, y) points along a cubic Bezier curve suitable for
    rendering a drooping patch cable.
    """
    # Control-point offsets: cable droops downward
    droop = abs(y1 - y0) * 0.5 + 60
    cx0, cy0 = x0, y0 + droop
    cx1, cy1 = x1, y1 + droop

    points: list[tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        inv = 1 - t
        x = (inv ** 3) * x0 + 3 * (inv ** 2) * t * cx0 + 3 * inv * (t ** 2) * cx1 + (t ** 3) * x1
        y = (inv ** 3) * y0 + 3 * (inv ** 2) * t * cy0 + 3 * inv * (t ** 2) * cy1 + (t ** 3) * y1
        points.append((x, y))
    return points
