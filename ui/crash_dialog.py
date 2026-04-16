"""
crash_dialog.py - Consent flow for sending local crash reports.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any

import httpx
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

DEFAULT_CRASH_ENDPOINT = ""
DEFAULT_MAILTO_TO = "dev@miniloader.ai"
_SETTINGS_ORG = "miniloader"
_SETTINGS_APP = "miniloader"
_SETTINGS_KEY = "telemetry/auto_send_crash_reports"


class CrashReportDialog(QDialog):
    """Prompt the user to optionally send a previously captured crash report."""

    def __init__(self, crash_data: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._crash_data = crash_data
        self.setWindowTitle("Miniloader closed unexpectedly")
        self.resize(760, 560)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(10)

        title = QLabel("Miniloader closed unexpectedly last time.")
        title.setStyleSheet("font-size: 17px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(
            "Would you like to send the crash report to help fix the issue?"
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9ea8b8;")
        root.addWidget(subtitle)

        tb_label = QLabel("Crash traceback")
        tb_label.setStyleSheet("font-weight: 600;")
        root.addWidget(tb_label)

        self._traceback_view = QTextEdit(self)
        self._traceback_view.setReadOnly(True)
        self._traceback_view.setFont(QFont("Consolas", 10))
        self._traceback_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._traceback_view.setPlainText(str(crash_data.get("traceback", "")))
        root.addWidget(self._traceback_view, 1)

        notes_label = QLabel("Optional notes")
        notes_label.setStyleSheet("font-weight: 600;")
        root.addWidget(notes_label)

        self._notes_input = QTextEdit(self)
        self._notes_input.setPlaceholderText("What were you doing when this happened?")
        self._notes_input.setFixedHeight(96)
        root.addWidget(self._notes_input)

        self._auto_send_checkbox = QCheckBox(
            "Always send crash reports automatically"
        )
        self._auto_send_checkbox.setToolTip(
            "Preference is saved for future releases that support auto-send."
        )
        self._load_preferences()
        root.addWidget(self._auto_send_checkbox)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(self.reject)
        buttons.addWidget(dismiss_btn)

        send_btn = QPushButton("Send Report")
        send_btn.setDefault(True)
        send_btn.clicked.connect(self._on_send_clicked)
        buttons.addWidget(send_btn)

        root.addLayout(buttons)

    def _load_preferences(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        value = settings.value(_SETTINGS_KEY, False)
        self._auto_send_checkbox.setChecked(str(value).lower() in {"true", "1", "yes"})

    def _save_preferences(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue(_SETTINGS_KEY, self._auto_send_checkbox.isChecked())

    def _on_send_clicked(self) -> None:
        self._save_preferences()
        notes = self._notes_input.toPlainText().strip()
        ok, detail = send_report(self._crash_data, notes)
        if ok:
            QMessageBox.information(self, "Crash Report Sent", "Thanks. The crash report was sent.")
            self.accept()
            return
        answer = QMessageBox.question(
            self,
            "Send Failed",
            f"Automatic send failed.\n\n{detail}\n\nWould you like to draft an email instead?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            fallback_ok = open_mailto_draft(self._crash_data, notes)
            if fallback_ok:
                self.accept()
            else:
                QMessageBox.warning(
                    self,
                    "Email Draft Failed",
                    "Could not open your default mail client for a draft email.",
                )


def send_report(crash_data: dict[str, Any], user_notes: str) -> tuple[bool, str]:
    endpoint = os.environ.get("MINILOADER_CRASH_ENDPOINT", "").strip() or DEFAULT_CRASH_ENDPOINT
    if not endpoint:
        return False, "No crash endpoint is configured."

    payload = dict(crash_data)
    payload["user_notes"] = user_notes
    payload["source"] = "desktop"

    try:
        response = httpx.post(endpoint, json=payload, timeout=8.0)
        response.raise_for_status()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def open_mailto_draft(crash_data: dict[str, Any], user_notes: str) -> bool:
    subject = urllib.parse.quote("Miniloader crash report")
    compact_payload = dict(crash_data)
    compact_payload["user_notes"] = user_notes
    crash_json = json.dumps(compact_payload, indent=2, ensure_ascii=True)
    body = urllib.parse.quote(
        "Crash report from Miniloader\n\n"
        "Please keep the JSON below when sending.\n\n"
        f"{crash_json}"
    )
    url = QUrl(f"mailto:{DEFAULT_MAILTO_TO}?subject={subject}&body={body}")
    return QDesktopServices.openUrl(url)
