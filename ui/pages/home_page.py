from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui._webengine_scrollbar import inject_dark_scrollbar_css

HOME_URL = "https://miniloader.ai/app"

try:
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineSettings = None  # type: ignore[assignment]
    QWebEngineView = None  # type: ignore[assignment]


class _PlaceholderPage(QWidget):
    """Simple placeholder page for non-rack app sections."""

    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #d8d8d8; font-size: 22px; font-weight: 600;")
        layout.addWidget(title_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("color: #9ea4ad; font-size: 13px;")
        layout.addWidget(subtitle_label)
        layout.addStretch(1)


class _HomeFallback(QWidget):
    """Fallback UI when Qt WebEngine is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Home page unavailable in embedded mode.")
        title.setStyleSheet("color: #d8d8d8; font-size: 16px; font-weight: 700;")
        layout.addWidget(title)

        hint = QLabel(
            "Install PySide6 WebEngine support to view the home page in-app, "
            "or open it in your browser."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa3b2; font-size: 12px;")
        layout.addWidget(hint)

        open_btn = QPushButton("Open in Browser")
        open_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1d2738; color: #b7d8ff; border: 1px solid #324661;"
            "  border-radius: 6px; padding: 8px 12px; text-align: left;"
            "}"
            "QPushButton:hover { background: #253149; }"
        )
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(HOME_URL)))
        layout.addWidget(open_btn)
        layout.addStretch(1)


class _HomePage(QWidget):
    """Landing page — loads miniloader.ai/app in an embedded web view."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if QWebEngineView is None:
            layout.addWidget(_HomeFallback())
            return

        view = QWebEngineView(self)
        view.page().setBackgroundColor(QColor("#121316"))
        if QWebEngineSettings is not None:
            settings = view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

        inject_dark_scrollbar_css(view)

        view.load(QUrl(HOME_URL))
        layout.addWidget(view)
