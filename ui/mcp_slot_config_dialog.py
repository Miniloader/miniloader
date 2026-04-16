"""MCP bus slot configuration dialog extracted from RackWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.base_module import BaseModule

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from modules.mcp_host.logic import PRESET_CATALOG
except Exception:
    PRESET_CATALOG = {}


def run_mcp_slot_config_dialog(
    parent: QWidget,
    target: BaseModule,
    slot_index: int,
    on_apply_init: Callable[[], None] | None = None,
) -> None:
    """Show the MCP slot configuration dialog and apply changes on accept."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"MCP Slot {slot_index + 1} Config")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)
    form = QFormLayout()

    spawn_edit = QLineEdit(str(target.params.get("spawn_command", "")))
    provider_edit = QLineEdit(str(target.params.get("provider_prefix", "")))
    type_combo = QComboBox()
    type_combo.addItems(["cloud", "local"])
    type_combo.setCurrentText(str(target.params.get("server_type", "cloud")))
    timeout_spin = QSpinBox()
    timeout_spin.setRange(1, 600)
    timeout_spin.setValue(int(target.params.get("call_timeout_s", 20)))
    restart_chk = QCheckBox("Restart on crash")
    restart_chk.setChecked(bool(target.params.get("restart_on_crash", True)))
    env_edit = QTextEdit()
    env_edit.setPlainText(str(target.params.get("env_vars", "")))
    env_edit.setPlaceholderText("KEY=VALUE (one per line)")
    env_edit.setMinimumHeight(120)

    provider_prefix = str(target.params.get("provider_prefix", "")).strip().lower()
    required_env: list[str] = []
    for preset in PRESET_CATALOG.values():
        if str(preset.get("provider_prefix", "")).strip().lower() == provider_prefix:
            raw_required = preset.get("required_env", [])
            if isinstance(raw_required, list):
                required_env = [str(k).strip() for k in raw_required if str(k).strip()]
            break
    required_env_lbl = QLabel(", ".join(required_env) if required_env else "none")
    required_env_lbl.setStyleSheet("color: #8f949c; font-size: 11px;")
    required_env_lbl.setWordWrap(True)

    form.addRow("Spawn command", spawn_edit)
    form.addRow("Provider prefix", provider_edit)
    form.addRow("Server type", type_combo)
    form.addRow("Call timeout (s)", timeout_spin)
    form.addRow("", restart_chk)
    form.addRow("Required env vars", required_env_lbl)
    form.addRow("Env vars", env_edit)

    status_lbl = QLabel(
        f"Status: {target.params.get('process_status', 'stopped')}   "
        f"Tools: {target.params.get('tools_count', 0)}   "
        f"Uptime: {target.params.get('uptime_s', 0)}s"
    )
    status_lbl.setStyleSheet("color: #8f949c; font-size: 11px;")
    err_text = str(target.params.get("last_error", "")).strip()
    if err_text:
        err_lbl = QLabel(f"Last error: {err_text}")
        err_lbl.setWordWrap(True)
        err_lbl.setStyleSheet("color: #d47f7f; font-size: 11px;")

    layout.addLayout(form)
    layout.addWidget(status_lbl)
    if err_text:
        layout.addWidget(err_lbl)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel
    )
    apply_init_btn = buttons.addButton(
        "Apply & Init", QDialogButtonBox.ButtonRole.ActionRole
    )
    layout.addWidget(buttons)

    do_init = {"value": False}

    def _on_apply_and_init() -> None:
        do_init["value"] = True
        dlg.accept()

    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    apply_init_btn.clicked.connect(_on_apply_and_init)

    if dlg.exec() != int(QDialog.DialogCode.Accepted):
        return

    target.params["spawn_command"] = spawn_edit.text().strip()
    target.params["provider_prefix"] = provider_edit.text().strip().lower() or "tool"
    target.params["server_type"] = type_combo.currentText().strip().lower()
    target.params["call_timeout_s"] = int(timeout_spin.value())
    target.params["restart_on_crash"] = bool(restart_chk.isChecked())
    target.params["env_vars"] = env_edit.toPlainText()

    if do_init["value"] and target.enabled and on_apply_init is not None:
        on_apply_init()
