"""Downloads page — native Qt view for the managed download queue."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPropertyAnimation, Property, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from core.download_manager import DownloadManager


_STATUS_LABELS: dict[str, str] = {
    "queued": "QUEUED",
    "downloading": "DOWNLOADING",
    "complete": "COMPLETE",
    "paused": "PAUSED",
    "cancelled": "CANCELLED",
    "failed": "FAILED",
}

_STATUS_COLORS: dict[str, str] = {
    "queued": "#607060",
    "downloading": "#30e848",
    "complete": "#30e848",
    "paused": "#d4a020",
    "cancelled": "#5c636e",
    "failed": "#e24c4c",
}

_LED_COLORS: dict[str, str] = {
    "queued": "#d4a020",
    "downloading": "#d4a020",
    "complete": "#30e848",
    "paused": "#d4a020",
    "cancelled": "#3a3f45",
    "failed": "#e24c4c",
}


def _fmt_speed(bps: float) -> str:
    if bps <= 0:
        return ""
    if bps >= 1_073_741_824:
        return f"{bps / 1_073_741_824:.1f} GB/s"
    if bps >= 1_048_576:
        return f"{bps / 1_048_576:.1f} MB/s"
    if bps >= 1024:
        return f"{bps / 1024:.0f} KB/s"
    return f"{bps:.0f} B/s"


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return ""
    total = int(seconds)
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


class _AnimatedProgressBar(QProgressBar):
    """Progress bar with a subtle animated pulse on the chunk when actively downloading."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pulse_opacity = 1.0
        self._pulse_anim = QPropertyAnimation(self, b"pulseOpacity")
        self._pulse_anim.setDuration(1800)
        self._pulse_anim.setLoopCount(-1)

    def _get_pulse_opacity(self) -> float:
        return self._pulse_opacity

    def _set_pulse_opacity(self, v: float) -> None:
        self._pulse_opacity = v
        self._apply_chunk_style()

    pulseOpacity = Property(float, _get_pulse_opacity, _set_pulse_opacity)

    def start_pulse(self, base_color: str) -> None:
        self._base_color = base_color
        self._pulse_anim.setStartValue(1.0)
        self._pulse_anim.setEndValue(0.45)
        self._pulse_anim.start()

    def stop_pulse(self, base_color: str) -> None:
        self._base_color = base_color
        self._pulse_anim.stop()
        self._pulse_opacity = 1.0
        self._apply_chunk_style()

    def _apply_chunk_style(self) -> None:
        c = self._base_color if hasattr(self, "_base_color") else "#30e848"
        o = self._pulse_opacity
        self.setStyleSheet(
            "QProgressBar {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #0a100a, stop:1 #070c07);"
            "  border: 1px solid #1e281e; border-radius: 3px;"
            "}"
            f"QProgressBar::chunk {{"
            f"  background: rgba({_hex_to_rgb(c)},{o:.2f});"
            f"  border-radius: 2px;"
            f"}}"
        )


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


class _DownloadCard(QWidget):
    """Single download item rendered as an LCD-style rack card."""

    def __init__(self, row: dict[str, Any], manager: DownloadManager) -> None:
        super().__init__()
        self._id: str = str(row.get("id") or "")
        self._manager = manager
        self._local_path: str = str(row.get("local_path") or "")

        self.setObjectName("DlCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "#DlCard {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #0e120e, stop:1 #080c08);"
            "  border: 1px solid #1e281e;"
            "  border-radius: 6px;"
            "}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(5)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self._led = QLabel()
        self._led.setFixedSize(8, 8)
        self._led.setStyleSheet(
            "background: #3a3f45; border: 1px solid #1a1a1a; border-radius: 4px;"
        )
        top_row.addWidget(self._led, 0, Qt.AlignmentFlag.AlignVCenter)

        self._filename_label = QLabel(str(row.get("filename") or "unknown"))
        self._filename_label.setStyleSheet(
            "color: #8ea58e; font-size: 12px; font-weight: 600;"
            " font-family: 'Consolas', 'Courier New', monospace;"
            " background: transparent;"
        )
        top_row.addWidget(self._filename_label, 1)

        self._status_label = QLabel()
        self._status_label.setStyleSheet(
            "font-size: 10px; font-family: 'Consolas', monospace;"
            " font-weight: 600; background: transparent;"
        )
        top_row.addWidget(self._status_label)

        outer.addLayout(top_row)

        repo_label = QLabel(str(row.get("repo_id") or ""))
        repo_label.setStyleSheet(
            "color: #405040; font-size: 10px;"
            " font-family: 'Consolas', monospace; background: transparent;"
        )
        outer.addWidget(repo_label)

        self._progress_bar = _AnimatedProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet(
            "QProgressBar {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #0a100a, stop:1 #070c07);"
            "  border: 1px solid #1e281e; border-radius: 3px;"
            "}"
            "QProgressBar::chunk {"
            "  background: #30e848; border-radius: 2px;"
            "}"
        )
        outer.addWidget(self._progress_bar)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        size_text = str(row.get("size") or "")
        self._size_label = QLabel(size_text)
        self._size_label.setStyleSheet(
            "color: #506050; font-size: 10px;"
            " font-family: 'Consolas', monospace; background: transparent;"
        )
        info_row.addWidget(self._size_label)

        self._speed_label = QLabel("")
        self._speed_label.setStyleSheet(
            "color: #30e848; font-size: 10px;"
            " font-family: 'Consolas', monospace; background: transparent;"
        )
        info_row.addWidget(self._speed_label)

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(
            "color: #607060; font-size: 10px;"
            " font-family: 'Consolas', monospace; background: transparent;"
        )
        info_row.addWidget(self._eta_label)

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            "color: #e24c4c; font-size: 10px;"
            " font-family: 'Consolas', monospace; background: transparent;"
        )
        info_row.addWidget(self._error_label, 1)
        outer.addLayout(info_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)
        btn_row.addStretch(1)

        self._pause_btn = self._action_button("PAUSE")
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._resume_btn = self._action_button("RESUME")
        self._resume_btn.clicked.connect(self._on_resume)
        btn_row.addWidget(self._resume_btn)

        self._cancel_btn = self._action_button("CANCEL", danger=True)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._reveal_btn = self._action_button("OPEN FOLDER")
        self._reveal_btn.clicked.connect(self._on_reveal)
        btn_row.addWidget(self._reveal_btn)

        self._clear_btn = self._action_button("CLEAR")
        self._clear_btn.clicked.connect(self._on_dismiss)
        btn_row.addWidget(self._clear_btn)

        self._delete_btn = self._action_button("DELETE", danger=True)
        self._delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._delete_btn)

        self._retry_btn = self._action_button("RETRY")
        self._retry_btn.clicked.connect(self._on_resume)
        btn_row.addWidget(self._retry_btn)

        outer.addLayout(btn_row)

        self.apply_state(row)

    @staticmethod
    def _action_button(label: str, danger: bool = False) -> QPushButton:
        if danger:
            color, bg, border, hover = "#c87070", "#1e0e0e", "#4a2020", "#2a1414"
        else:
            color, bg, border, hover = "#8ea58e", "#0c140c", "#2a3a2a", "#142014"
        btn = QPushButton(label)
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: {bg}; color: {color}; border: 1px solid {border};"
            f"  border-radius: 4px; padding: 3px 8px; font-size: 10px;"
            f"  font-family: 'Consolas', monospace; font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{ background: {hover}; }}"
        )
        return btn

    def apply_state(self, row: dict[str, Any]) -> None:
        status = str(row.get("status") or "queued")
        progress = float(row.get("progress") or 0)
        error = str(row.get("error") or "")
        self._local_path = str(row.get("local_path") or "")

        self._status_label.setText(_STATUS_LABELS.get(status, status.upper()))
        status_c = _STATUS_COLORS.get(status, "#607060")
        self._status_label.setStyleSheet(
            f"color: {status_c}; font-size: 10px;"
            f" font-family: 'Consolas', monospace; font-weight: 600;"
            f" background: transparent;"
        )

        led_c = _LED_COLORS.get(status, "#3a3f45")
        self._led.setStyleSheet(
            f"background: {led_c}; border: 1px solid #1a1a1a; border-radius: 4px;"
        )

        self._progress_bar.setValue(int(progress))
        self._progress_bar.setVisible(status in ("queued", "downloading", "paused"))

        chunk_colors = {
            "complete": "#30e848",
            "failed": "#e24c4c",
            "paused": "#d4a020",
        }
        chunk_c = chunk_colors.get(status, "#30e848")
        if status == "downloading":
            self._progress_bar.start_pulse(chunk_c)
        else:
            self._progress_bar.stop_pulse(chunk_c)

        speed_bps = float(row.get("speedBps") or row.get("speed") or 0)
        eta_seconds = float(row.get("etaSeconds") or 0)
        self._speed_label.setText(_fmt_speed(speed_bps) if status == "downloading" else "")
        self._eta_label.setText(f"ETA {_fmt_eta(eta_seconds)}" if status == "downloading" and eta_seconds > 0 else "")
        self._error_label.setText(error if status == "failed" else "")

        is_active = status in ("queued", "downloading")
        is_paused = status == "paused"
        is_broken = status in ("cancelled", "failed")

        self._pause_btn.setVisible(is_active)
        self._resume_btn.setVisible(is_paused)
        self._cancel_btn.setVisible(is_active or is_paused)
        self._reveal_btn.setVisible(status == "complete" and bool(self._local_path))
        self._clear_btn.setVisible(status == "complete")
        self._delete_btn.setVisible(is_broken)
        self._retry_btn.setVisible(status == "failed")

    def _on_pause(self) -> None:
        self._manager.pause(self._id)

    def _on_resume(self) -> None:
        self._manager.resume(self._id)

    def _on_cancel(self) -> None:
        self._manager.cancel(self._id)

    def _on_reveal(self) -> None:
        p = Path(self._local_path)
        if not p.exists():
            return
        if sys.platform == "win32":
            os.system(f'explorer /select,"{p}"')
        elif sys.platform == "darwin":
            os.system(f'open -R "{p}"')
        else:
            os.system(f'xdg-open "{p.parent}"')

    def _on_dismiss(self) -> None:
        self._manager.dismiss(self._id)

    def _on_delete(self) -> None:
        self._manager.delete(self._id)


class DownloadsPage(QWidget):
    """Full-page download queue view with live refresh."""

    POLL_MS = 300

    def __init__(self, download_manager: DownloadManager | None = None) -> None:
        super().__init__()
        self._manager = download_manager
        self._cards: dict[str, _DownloadCard] = {}
        self.setStyleSheet("background: #121316;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(8)

        self._empty_label = QLabel("No downloads in queue")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(
            "color: #3a4858; font-size: 13px;"
            " font-family: 'Consolas', monospace;"
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        outer.addWidget(self._empty_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch(1)
        scroll.setWidget(self._list_widget)
        outer.addWidget(scroll, 1)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_MS)
        self._poll_timer.timeout.connect(self._refresh)

    def set_download_manager(self, manager: DownloadManager | None) -> None:
        self._manager = manager
        self._refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh()
        self._poll_timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._poll_timer.stop()

    def _refresh(self) -> None:
        if self._manager is None:
            self._empty_label.setVisible(True)
            return

        rows = self._manager.get_all()
        row_ids = {str(r.get("id") or "") for r in rows}
        self._empty_label.setVisible(len(rows) == 0)

        stale = [did for did in self._cards if did not in row_ids]
        for did in stale:
            card = self._cards.pop(did)
            self._list_layout.removeWidget(card)
            card.deleteLater()

        for row in rows:
            did = str(row.get("id") or "")
            if did in self._cards:
                self._cards[did].apply_state(row)
            else:
                card = _DownloadCard(row, self._manager)
                self._cards[did] = card
                insert_idx = max(0, self._list_layout.count() - 1)
                self._list_layout.insertWidget(insert_idx, card)
