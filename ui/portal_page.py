"""Portal page widget with QWebChannel bridge wiring."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import unquote, urlparse

from PySide6.QtCore import QPropertyAnimation, QRect, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from core.download_manager import DownloadManager
from core.entitlement_store import EntitlementStore
from core.module_installer import ModuleInstaller
from core.portal_bridge import PortalBridge
from ui._webengine_scrollbar import inject_dark_scrollbar_css

if TYPE_CHECKING:
    from core.vault import VaultManager

PORTAL_URL = "https://portal.miniloader.ai/"
log = logging.getLogger(__name__)

try:
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEngineScript, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - optional dependency
    QWebChannel = None  # type: ignore[assignment]
    QWebEngineScript = None  # type: ignore[assignment]
    QWebEngineSettings = None  # type: ignore[assignment]
    QWebEngineView = None  # type: ignore[assignment]


def _portal_url(vault: "VaultManager | None") -> str:
    if vault is None:
        return PORTAL_URL
    try:
        return f"{PORTAL_URL}?lid={vault.user_id}"
    except Exception:
        return PORTAL_URL


def _load_qwebchannel_source() -> str | None:
    js_path = Path(__file__).resolve().parent.parent / "resources" / "portal" / "qwebchannel.js"
    if not js_path.exists():
        log.warning(
            "Portal bridge script is missing at %s; embedded portal will run in browser mode.",
            js_path,
        )
        return None
    try:
        return js_path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning(
            "Failed to load portal bridge script from %s (%s); embedded portal will run in browser mode.",
            js_path,
            exc,
        )
        return None


def _make_qwebchannel_script() -> "QWebEngineScript | None":
    if QWebEngineScript is None:
        return None
    source = _load_qwebchannel_source()
    if not source:
        return None

    script = QWebEngineScript()
    script.setSourceCode(source)
    script.setName("qwebchannel_api")
    script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    script.setRunsOnSubFrames(True)
    return script


class _DownloadToast(QLabel):
    """Animated toast that slides in and fades out when a download is enqueued."""

    _SHOW_MS = 3500
    _FADE_MS = 600

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(22, 34, 52, 230);"
            "color: #8ec8ff;"
            "border: 1px solid #2a5080;"
            "border-radius: 8px;"
            "padding: 10px 16px;"
            "font-size: 12px;"
            "font-weight: 600;"
        )
        self.setFixedHeight(42)
        self.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.hide()

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)

        self._slide_anim = QPropertyAnimation(self, b"geometry")
        self._slide_anim.setDuration(320)

        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_anim.setDuration(self._FADE_MS)
        self._fade_anim.finished.connect(self.hide)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._start_fade)

    def show_download(self, filename: str) -> None:
        self._dismiss_timer.stop()
        self._fade_anim.stop()
        self._slide_anim.stop()

        display = filename if len(filename) <= 48 else filename[:45] + "\u2026"
        self.setText(f"\u2B07  Download started: {display}")
        self.adjustSize()

        pw = self.parent().width() if self.parent() else 400
        w = min(pw - 32, self.sizeHint().width() + 24)
        h = self.height()
        x = pw - w - 16
        end_y = 12

        self.setGeometry(x, -h, w, h)
        self._opacity.setOpacity(1.0)
        self.show()
        self.raise_()

        self._slide_anim.setStartValue(QRect(x, -h, w, h))
        self._slide_anim.setEndValue(QRect(x, end_y, w, h))
        self._slide_anim.start()

        self._dismiss_timer.start(self._SHOW_MS)

    def _start_fade(self) -> None:
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()


class _PortalFallback(QWidget):
    """Fallback UI when Qt WebEngine is unavailable."""

    def __init__(self, vault: "VaultManager | None" = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Portal unavailable in embedded mode.")
        title.setStyleSheet("color: #d8d8d8; font-size: 16px; font-weight: 700;")
        layout.addWidget(title)

        hint = QLabel("Install PySide6 WebEngine support to open the portal in-app, or launch it externally.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa3b2; font-size: 12px;")
        layout.addWidget(hint)

        open_btn = QPushButton("Open Portal in Browser")
        open_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1d2738; color: #b7d8ff; border: 1px solid #324661;"
            "  border-radius: 6px; padding: 8px 12px; text-align: left;"
            "}"
            "QPushButton:hover { background: #253149; }"
        )
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_portal_url(vault))))
        layout.addWidget(open_btn)
        layout.addStretch(1)


class PortalPage(QWidget):
    """Embedded portal tab with QWebChannel bridge registration."""

    def __init__(
        self,
        vault: "VaultManager | None" = None,
        download_manager: "DownloadManager | None" = None,
        module_installer: ModuleInstaller | None = None,
        on_module_hotload: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._vault = vault
        self._bridge: PortalBridge | None = None
        self._channel: QWebChannel | None = None
        self._web_view: QWebEngineView | None = None
        self._download_manager = download_manager
        self._module_installer = module_installer
        self._on_module_hotload = on_module_hotload

        if QWebEngineView is None:
            layout.addWidget(_PortalFallback(vault=vault))
            return

        self._view_stack = QStackedWidget(self)

        loading_page = QWidget()
        loading_page.setStyleSheet("background: #121316;")
        ll = QVBoxLayout(loading_page)
        ll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_label = QLabel("Loading Portal\u2026")
        loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_label.setStyleSheet(
            "color: #4a5868; font-size: 14px; font-weight: 500;"
            " background: transparent;"
        )
        ll.addWidget(loading_label)
        self._view_stack.addWidget(loading_page)     # index 0

        view = QWebEngineView(self)
        self._web_view = view
        view.page().setBackgroundColor(QColor("#121316"))
        if QWebEngineSettings is not None:
            settings = view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)

        self._toast = _DownloadToast(self)

        if QWebChannel is not None:
            entitlement_store = None
            if self._vault is not None:
                entitlement_store = EntitlementStore(
                    self._vault.ensure_user_data_dir() / "entitlements.json"
                )
            self._bridge = PortalBridge(
                vault=self._vault,
                download_manager=self._download_manager,
                module_installer=self._module_installer,
                entitlement_store=entitlement_store,
                parent=self,
            )
            self._bridge.downloadStarted.connect(self._toast.show_download)
            self._bridge.moduleHotloadRequested.connect(self._handle_module_hotload)
            self._channel = QWebChannel(self)
            self._channel.registerObject("bridge", self._bridge)
            view.page().setWebChannel(self._channel)

        script = _make_qwebchannel_script()
        if script is not None:
            scripts = view.page().profile().scripts()
            for existing in scripts.find("qwebchannel_api"):
                scripts.remove(existing)
            scripts.insert(script)

        inject_dark_scrollbar_css(view)

        view.page().profile().downloadRequested.connect(self._on_download_requested)
        view.load(QUrl(_portal_url(self._vault)))
        self._view_stack.addWidget(view)              # index 1
        self._view_stack.setCurrentIndex(0)

        layout.addWidget(self._view_stack)

        view.loadFinished.connect(self._on_portal_loaded)

    def _on_portal_loaded(self, ok: bool) -> None:
        if hasattr(self, "_view_stack"):
            self._view_stack.setCurrentIndex(1)

    def _handle_module_hotload(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if self._on_module_hotload is not None:
            self._on_module_hotload(payload)

    @staticmethod
    def _parse_hf_url(url: str) -> tuple[str, str] | None:
        """
        Parse HF resolve URL and return ``(repo_id, filename)``.

        Expected shape: /{owner}/{repo}/resolve/{revision}/{filename}
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "huggingface.co" not in host:
            return None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 5:
            return None
        if parts[2] != "resolve":
            return None
        owner, repo = parts[0], parts[1]
        filename = "/".join(parts[4:])
        if not owner or not repo or not filename:
            return None
        return f"{owner}/{repo}", unquote(filename)

    @staticmethod
    def _parse_signed_url(url: str) -> tuple[str, str] | None:
        """
        Extract a filename from a signed S3 / CloudFront URL.

        Returns ``(item_id_or_host, filename)`` or ``None``.
        """
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        is_aws = "amazonaws.com" in host or "cloudfront.net" in host
        if not is_aws:
            return None
        path_parts = [p for p in parsed.path.split("/") if p]
        if not path_parts:
            return None
        filename = unquote(path_parts[-1])
        item_id = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        return item_id, filename

    def _on_download_requested(self, download) -> None:
        """
        Safety-net interception for direct file links opened in the webview.

        Handles both HF resolve URLs and signed AWS URLs.
        Managed handoff only occurs when the vault is available (logged in).
        """
        url = download.url().toString()

        if self._vault is None or self._download_manager is None:
            download.accept()
            return

        hf = self._parse_hf_url(url)
        if hf is not None:
            repo_id, filename = hf
            download.cancel()
            self._download_manager.enqueue(
                download_id=str(uuid.uuid4()),
                repo_id=repo_id,
                filename=filename,
                variant=filename,
                size="",
                url=url,
            )
            if hasattr(self, "_toast"):
                self._toast.show_download(filename)
            return

        signed = self._parse_signed_url(url)
        if signed is not None:
            item_id, filename = signed
            download.cancel()
            self._download_manager.enqueue(
                download_id=str(uuid.uuid4()),
                repo_id=item_id,
                filename=filename,
                variant=filename,
                size="",
                url=url,
            )
            if hasattr(self, "_toast"):
                self._toast.show_download(filename)
            return

        download.accept()

