"""
login_dialog.py — Manual login, recovery import, and vault gate.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.vault import VaultError, VaultLoginError, VaultManager, VaultValidationError
from ui.onboarding_dialog import OnboardingDialog


class LoginDialog(QDialog):
    """Password login + mnemonic import workflow for local vaults."""

    def __init__(self, vault_path: Path | None = None) -> None:
        super().__init__()
        self._vault_path = Path(vault_path or VaultManager.VAULT_PATH)
        self.vault_manager: VaultManager | None = None

        self.setWindowTitle("Miniloader Sign In")
        self.resize(560, 420)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(10)

        title = QLabel("Unlock your local vault")
        title.setStyleSheet("color: #d8d8d8; font-size: 18px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel("Sign in with your password or import an existing account using your 12 words.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9ea8b8; font-size: 12px;")
        root.addWidget(subtitle)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_signin_tab(), "Sign In")
        tabs.addTab(self._build_recover_tab(), "Import Existing Account")
        root.addWidget(tabs, 1)

    def _build_signin_tab(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        form = QFormLayout()

        self._password_input = QLineEdit()
        self._password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_input.setPlaceholderText("Password")
        self._save_key_checkbox = QCheckBox("Enable Auto-Login on this PC")
        self._save_key_checkbox.setChecked(True)
        form.addRow("Password", self._password_input)
        layout.addLayout(form)
        layout.addWidget(self._save_key_checkbox)

        self._signin_error = QLabel("")
        self._signin_error.setStyleSheet("color: #ff9f9f;")
        layout.addWidget(self._signin_error)
        layout.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)
        signin_btn = QPushButton("Sign In")
        signin_btn.clicked.connect(self._handle_signin)
        row.addWidget(signin_btn)
        layout.addLayout(row)
        return panel

    def _build_recover_tab(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)

        form = QFormLayout()
        self._recover_username = QLineEdit()
        self._recover_username.setPlaceholderText("Username")
        self._recover_mnemonic = QTextEdit()
        self._recover_mnemonic.setPlaceholderText("Enter your 12-word recovery phrase")
        self._recover_mnemonic.setFixedHeight(92)
        self._recover_password = QLineEdit()
        self._recover_password.setEchoMode(QLineEdit.EchoMode.Password)
        self._recover_password.setPlaceholderText("New local password")
        self._recover_password_confirm = QLineEdit()
        self._recover_password_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self._recover_password_confirm.setPlaceholderText("Confirm new local password")
        self._recover_keychain_checkbox = QCheckBox("Enable Auto-Login on this PC")
        self._recover_keychain_checkbox.setChecked(True)

        form.addRow("Username", self._recover_username)
        form.addRow("Recovery phrase", self._recover_mnemonic)
        form.addRow("New password", self._recover_password)
        form.addRow("Confirm", self._recover_password_confirm)
        layout.addLayout(form)
        layout.addWidget(self._recover_keychain_checkbox)

        self._recover_error = QLabel("")
        self._recover_error.setStyleSheet("color: #ff9f9f;")
        layout.addWidget(self._recover_error)
        layout.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)
        import_btn = QPushButton("Import Account")
        import_btn.clicked.connect(self._handle_import)
        row.addWidget(import_btn)
        layout.addLayout(row)
        return panel

    def _handle_signin(self) -> None:
        try:
            self.vault_manager = VaultManager.from_password(
                vault_path=self._vault_path,
                password=self._password_input.text(),
            )
            self.vault_manager.ensure_user_data_dir()
            if self._save_key_checkbox.isChecked():
                self.vault_manager.store_key_in_keyring()
            self.accept()
        except (VaultLoginError, VaultValidationError) as exc:
            self._signin_error.setText(str(exc))
        except VaultError as exc:
            QMessageBox.critical(self, "Sign In Failed", str(exc))

    def _handle_import(self) -> None:
        password = self._recover_password.text()
        confirm = self._recover_password_confirm.text()
        if password != confirm:
            self._recover_error.setText("Passwords do not match.")
            return
        if self._vault_path.exists():
            answer = QMessageBox.question(
                self,
                "Replace Existing Vault",
                "A local vault already exists on this PC. Importing will replace it. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.vault_manager = VaultManager.from_mnemonic(
                mnemonic=self._recover_mnemonic.toPlainText(),
                vault_path=self._vault_path,
                password=password,
                username=self._recover_username.text().strip(),
                save_to_keyring=self._recover_keychain_checkbox.isChecked(),
            )
            self.vault_manager.ensure_user_data_dir()
            self.accept()
        except VaultValidationError as exc:
            self._recover_error.setText(str(exc))
        except VaultError as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))


def run_vault_gate(
    vault_path: Path | None = None,
) -> tuple[VaultManager | None, str | None]:
    """Show onboarding or login and return (vault, initial_preset).

    ``initial_preset`` is non-None only when the user came through the
    onboarding questionnaire and chose (or was assigned) a rack preset.
    """
    selected_path = Path(vault_path or VaultManager.VAULT_PATH)
    if selected_path.exists():
        auto_vault = VaultManager.from_keyring(selected_path)
        if auto_vault is not None:
            auto_vault.ensure_user_data_dir()
            return auto_vault, None
        login = LoginDialog(vault_path=selected_path)
        if login.exec() == QDialog.DialogCode.Accepted:
            return login.vault_manager, None
        return None, None

    onboarding = OnboardingDialog(vault_path=selected_path)
    if onboarding.exec() == QDialog.DialogCode.Accepted:
        return onboarding.vault_manager, onboarding.chosen_preset
    return None, None
