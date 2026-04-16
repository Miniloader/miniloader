"""Account page — user identity, personas, security, storage, and session."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from core.vault import VaultManager

_CARD_SS = (
    "background: #141820; border: 1px solid #1e2638; border-radius: 8px;"
)
_KV_LABEL_SS = "color: #4e6070; font-size: 11px;"
_KV_VALUE_SS = "color: #b8cce0; font-size: 12px; font-weight: 500;"
_BTN_SS = (
    "QPushButton {"
    "  background: #182030; color: #7aaddc; border: 1px solid #2a4060;"
    "  border-radius: 5px; padding: 6px 16px; font-size: 12px;"
    "}"
    "QPushButton:hover { background: #1e2c48; border-color: #3a5880; }"
    "QPushButton:pressed { background: #162038; }"
)
_DANGER_BTN_SS = (
    "QPushButton {"
    "  background: #1c1318; color: #c08888; border: 1px solid #4a2828;"
    "  border-radius: 5px; padding: 6px 16px; font-size: 12px;"
    "}"
    "QPushButton:hover { background: #261a1a; color: #dc9898; border-color: #6a3030; }"
    "QPushButton:pressed { background: #1a1010; }"
)


def _fmt_bytes(value: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024 and idx < len(suffixes) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {suffixes[idx]}"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


class _AvatarWidget(QWidget):
    """Circle showing the user's initial letter."""

    _SIZE = 52

    def __init__(self, initial: str) -> None:
        super().__init__()
        self._initial = (initial[0].upper() if initial else "?")
        self.setFixedSize(QSize(self._SIZE, self._SIZE))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addEllipse(0, 0, self._SIZE, self._SIZE)
        painter.setClipPath(path)

        painter.fillRect(0, 0, self._SIZE, self._SIZE, QColor("#1a2740"))

        painter.setPen(QColor("#4a7ab0"))
        painter.drawEllipse(1, 1, self._SIZE - 2, self._SIZE - 2)

        font = QFont("Segoe UI", 20, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor("#8ec4f0"))
        painter.drawText(
            0, 0, self._SIZE, self._SIZE,
            Qt.AlignmentFlag.AlignCenter,
            self._initial,
        )


def _section_card(title: str) -> tuple[QWidget, QVBoxLayout]:
    """Return a styled card widget and its inner layout."""
    card = QWidget()
    card.setStyleSheet(_CARD_SS)

    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    accent = QWidget()
    accent.setFixedWidth(3)
    accent.setStyleSheet(
        "background: #2a5080; border-top-left-radius: 8px;"
        " border-bottom-left-radius: 8px;"
    )

    header_row = QHBoxLayout()
    header_row.setContentsMargins(14, 12, 14, 10)
    header_row.setSpacing(10)
    header_row.addWidget(accent)

    header_lbl = QLabel(title)
    header_lbl.setStyleSheet(
        "color: #7090b0; font-size: 10px; font-weight: 700;"
        " letter-spacing: 0.8px; background: transparent; border: none; padding: 0;"
    )
    header_row.addWidget(header_lbl)
    header_row.addStretch(1)
    outer.addLayout(header_row)

    sep = QWidget()
    sep.setFixedHeight(1)
    sep.setStyleSheet("background: #1e2638;")
    outer.addWidget(sep)

    body = QWidget()
    body.setStyleSheet("background: transparent;")
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(16, 12, 16, 14)
    body_layout.setSpacing(10)
    outer.addWidget(body)

    return card, body_layout


class _AccountPage(QWidget):
    """User account overview: identity, local data, and session controls."""

    _KV_LABEL_WIDTH = 130

    def __init__(
        self,
        vault: VaultManager | None,
        on_username_changed: Callable[[str], None] | None = None,
        on_logout: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._vault = vault
        self._on_username_changed = on_username_changed
        self._on_logout = on_logout

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        outer.addWidget(scroll)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)
        scroll.setWidget(body)

        # ── Page header ───────────────────────────────────────
        page_header = QHBoxLayout()
        page_header.setSpacing(16)

        if vault is not None:
            avatar = _AvatarWidget(vault.username)
            page_header.addWidget(avatar, 0, Qt.AlignmentFlag.AlignVCenter)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Account")
        title.setStyleSheet("color: #d4dce8; font-size: 20px; font-weight: 700;")
        title_col.addWidget(title)
        if vault is not None:
            sub = QLabel(vault.username)
            sub.setStyleSheet("color: #506070; font-size: 12px;")
            title_col.addWidget(sub)
        page_header.addLayout(title_col)
        page_header.addStretch(1)

        layout.addLayout(page_header)

        if vault is None:
            msg = QLabel("Sign in to view account details.")
            msg.setStyleSheet("color: #506070; font-size: 13px;")
            layout.addWidget(msg)
            layout.addStretch(1)
            return

        self._build_identity_section(layout)
        self._build_security_section(layout)
        self._build_storage_section(layout)
        self._build_session_section(layout)
        layout.addStretch(1)

    # ── Identity ─────────────────────────────────────────────

    def _build_identity_section(self, parent: QVBoxLayout) -> None:
        assert self._vault is not None

        card, cl = _section_card("IDENTITY")

        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel("Username")
        lbl.setFixedWidth(self._KV_LABEL_WIDTH)
        lbl.setStyleSheet(_KV_LABEL_SS + " border: none; padding: 0;")

        self._username_input = QLineEdit(self._vault.username)
        self._username_input.setStyleSheet(
            "background: #0d1118; color: #b8cce0; border: 1px solid #1e2a3a;"
            " border-radius: 5px; padding: 5px 9px; font-size: 12px;"
            " selection-background-color: #2a4870;"
        )
        self._username_input.setPlaceholderText("Enter username…")

        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(64)
        save_btn.setStyleSheet(_BTN_SS)
        save_btn.clicked.connect(self._save_username)

        row.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._username_input, 1)
        row.addWidget(save_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        cl.addLayout(row)

        self._username_status = QLabel("")
        self._username_status.setStyleSheet(
            "color: #4a8a64; font-size: 10px; border: none; padding: 0 0 0 "
            + str(self._KV_LABEL_WIDTH + 10) + "px;"
        )
        cl.addWidget(self._username_status)

        uid_row = QHBoxLayout()
        uid_lbl = QLabel("User ID")
        uid_lbl.setFixedWidth(self._KV_LABEL_WIDTH)
        uid_lbl.setStyleSheet(_KV_LABEL_SS + " border: none; padding: 0;")
        uid_val = QLabel(self._vault.user_id)
        uid_val.setStyleSheet(
            _KV_VALUE_SS + " font-family: 'Consolas', monospace;"
            " font-size: 11px; border: none; padding: 0;"
        )
        uid_val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        uid_row.addWidget(uid_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        uid_row.addWidget(uid_val, 1)
        cl.addLayout(uid_row)

        parent.addWidget(card)

    def _save_username(self) -> None:
        if self._vault is None:
            return
        name = self._username_input.text().strip()
        if not name:
            self._username_status.setStyleSheet(
                "color: #c07070; font-size: 10px; border: none; padding: 0 0 0 "
                + str(self._KV_LABEL_WIDTH + 10) + "px;"
            )
            self._username_status.setText("Username cannot be empty.")
            return
        self._vault.username = name
        self._vault.save_profile(self._vault.profile)
        if self._on_username_changed is not None:
            self._on_username_changed(name)
        self._username_status.setStyleSheet(
            "color: #4a8a64; font-size: 10px; border: none; padding: 0 0 0 "
            + str(self._KV_LABEL_WIDTH + 10) + "px;"
        )
        self._username_status.setText("Saved.")

    # ── Security ──────────────────────────────────────────────

    def _build_security_section(self, parent: QVBoxLayout) -> None:
        assert self._vault is not None

        card, cl = _section_card("SECURITY")

        token = self._vault.get_or_create_secret(
            self._vault.DEFAULT_GPT_SERVER_SECRET_KEY
        )

        self._add_kv(cl, "Bearer Token", "")
        token_row = QHBoxLayout()
        token_row.setSpacing(8)
        token_input = QLineEdit(token)
        token_input.setReadOnly(True)
        token_input.setStyleSheet(
            "background: #0d1118; color: #8eb8e8; border: 1px solid #1e2a3a;"
            " border-radius: 5px; padding: 5px 9px; font-size: 11px;"
            " font-family: 'Consolas', monospace;"
        )
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(56)
        copy_btn.setStyleSheet(_BTN_SS)
        copy_btn.clicked.connect(lambda: (
            __import__("PySide6.QtWidgets", fromlist=["QApplication"])
            .QApplication.clipboard().setText(token)
        ))
        token_row.addWidget(token_input, 1)
        token_row.addWidget(copy_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        cl.addLayout(token_row)

        try:
            from core.cert_manager import CertManager
            cm = CertManager()
            expiry = cm.get_cert_expiry()
            expiry_text = expiry.strftime("%Y-%m-%d") if expiry else "No certs generated"
        except Exception:
            expiry_text = "Unavailable"
        self._add_kv(cl, "TLS Cert Expiry", expiry_text)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #1a2030;")
        cl.addWidget(sep)

        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(10)
        rotate_lbl = QLabel("IPC Signing Key")
        rotate_lbl.setFixedWidth(self._KV_LABEL_WIDTH)
        rotate_lbl.setStyleSheet(_KV_LABEL_SS + " border: none; padding: 0;")
        rotate_btn = QPushButton("Rotate Key")
        rotate_btn.setStyleSheet(_BTN_SS)

        def _rotate_key() -> None:
            self._vault.rotate_module_comm_key()
            rotate_btn.setText("Rotated")
            rotate_btn.setEnabled(False)

        rotate_btn.clicked.connect(_rotate_key)
        rotate_row.addWidget(rotate_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        rotate_row.addWidget(rotate_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        rotate_row.addStretch(1)
        cl.addLayout(rotate_row)

        parent.addWidget(card)

    # ── Local storage ────────────────────────────────────────

    def _build_storage_section(self, parent: QVBoxLayout) -> None:
        assert self._vault is not None

        card, cl = _section_card("LOCAL STORAGE")

        data_dir = self._vault.get_user_data_dir()
        self._add_kv(cl, "Data directory", str(data_dir))

        vault_path = self._vault.vault_path
        vault_size = vault_path.stat().st_size if vault_path.exists() else 0
        self._add_kv(cl, "Vault file", f"{vault_path.name}  ({_fmt_bytes(vault_size)})")

        if data_dir.exists():
            data_total = _dir_size(data_dir)
            self._add_kv(cl, "User data size", _fmt_bytes(data_total))

            db_path = data_dir / "miniloader_data.db"
            if db_path.exists():
                self._add_kv(
                    cl, "Database",
                    f"miniloader_data.db  ({_fmt_bytes(db_path.stat().st_size)})",
                )

            chroma_dir = data_dir / "chroma"
            if chroma_dir.exists():
                chroma_size = _dir_size(chroma_dir)
                self._add_kv(cl, "Vector index", _fmt_bytes(chroma_size))

        hf_cache = Path.home() / ".cache" / "huggingface"
        if hf_cache.exists():
            hf_size = _dir_size(hf_cache)
            self._add_kv(cl, "HuggingFace cache", _fmt_bytes(hf_size))

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #1a2030;")
        cl.addWidget(sep)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open Data Folder")
        open_btn.setStyleSheet(_BTN_SS)
        open_btn.clicked.connect(lambda: os.startfile(str(data_dir)))  # type: ignore[attr-defined]
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        parent.addWidget(card)

    def _add_kv(self, layout: QVBoxLayout, key: str, value: str) -> None:
        row = QHBoxLayout()
        row.setSpacing(10)
        k = QLabel(key)
        k.setFixedWidth(self._KV_LABEL_WIDTH)
        k.setStyleSheet(_KV_LABEL_SS + " border: none; padding: 0;")
        v = QLabel(value)
        v.setStyleSheet(_KV_VALUE_SS + " border: none; padding: 0;")
        v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.setWordWrap(True)
        row.addWidget(k, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(v, 1)
        layout.addLayout(row)

    # ── Session ──────────────────────────────────────────────

    def _build_session_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("SESSION")

        desc = QLabel("Signing out will clear your local session. Your downloaded files will not be affected.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #4e6070; font-size: 11px; border: none; padding: 0;")
        cl.addWidget(desc)

        logout_btn = QPushButton("Log Out")
        logout_btn.setStyleSheet(_DANGER_BTN_SS)
        if self._on_logout is not None:
            logout_btn.clicked.connect(self._on_logout)
        else:
            logout_btn.setEnabled(False)
        cl.addWidget(logout_btn, 0, Qt.AlignmentFlag.AlignLeft)

        parent.addWidget(card)
