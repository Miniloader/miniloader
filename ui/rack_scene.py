"""
rack_scene.py — Rack Graphics Scene
=====================================
QGraphicsScene subclass that intercepts mouse events to drive
pending-wire updates and cancellation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import RackWindow

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsScene


class RackScene(QGraphicsScene):
    def __init__(self, controller: RackWindow) -> None:
        super().__init__()
        self.controller = controller

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        self.controller.update_pending_wire(event.scenePos())
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform()) if self.views() else None
            if item is None:
                self.controller.clear_pending_wire()
        elif event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.scenePos(), self.views()[0].transform()) if self.views() else None
            if item is None:
                self.controller.on_empty_right_clicked(event.screenPos())
                event.accept()
                return
        super().mousePressEvent(event)
