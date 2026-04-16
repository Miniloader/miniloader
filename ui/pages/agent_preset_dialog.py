from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class AgentPresetDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        preset: dict[str, Any] | None = None,
        available_tools: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Agent Preset Editor")
        self.resize(700, 680)

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        form = QFormLayout(body)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        initial = dict(preset or {})
        initial_sampling = initial.get("sampling", {})
        if not isinstance(initial_sampling, dict):
            initial_sampling = {}
        initial_tool_cfg = initial.get("tool_config", {})
        if not isinstance(initial_tool_cfg, dict):
            initial_tool_cfg = {}

        self._name_input = QLineEdit(str(initial.get("name", "")).strip())
        self._description_input = QLineEdit(str(initial.get("description", "")).strip())
        self._system_prompt = QPlainTextEdit(str(initial.get("system_prompt", "")).strip())
        self._system_prompt.setPlaceholderText(
            "System prompt (supports {{date}} and {{user_name}} placeholders)"
        )
        self._system_prompt.setMinimumHeight(130)

        form.addRow("Name", self._name_input)
        form.addRow("Description", self._description_input)
        form.addRow("System Prompt", self._system_prompt)

        self._model_override = QLineEdit(str(initial.get("model_override", "") or ""))
        self._model_override.setPlaceholderText("Optional model id override")
        form.addRow("Model Override", self._model_override)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.0, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setValue(float(initial_sampling.get("temperature", 0.7)))
        form.addRow("Temperature", self._temperature)

        self._top_p = QDoubleSpinBox()
        self._top_p.setRange(0.0, 1.0)
        self._top_p.setSingleStep(0.01)
        self._top_p.setValue(float(initial_sampling.get("top_p", 1.0)))
        form.addRow("top_p", self._top_p)

        self._top_k = QSpinBox()
        self._top_k.setRange(0, 2000)
        self._top_k.setValue(int(initial_sampling.get("top_k", 0) or 0))
        form.addRow("top_k", self._top_k)

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(0, 32768)
        self._max_tokens.setValue(int(initial_sampling.get("max_tokens", 0) or 0))
        form.addRow("max_tokens", self._max_tokens)

        self._repeat_penalty = QDoubleSpinBox()
        self._repeat_penalty.setRange(0.0, 3.0)
        self._repeat_penalty.setSingleStep(0.05)
        self._repeat_penalty.setValue(float(initial_sampling.get("repeat_penalty", 0.0) or 0.0))
        form.addRow("repeat_penalty", self._repeat_penalty)

        self._max_rounds = QSpinBox()
        self._max_rounds.setRange(1, 10)
        self._max_rounds.setValue(int(initial_tool_cfg.get("max_rounds", 3) or 3))
        form.addRow("Max Tool Rounds", self._max_rounds)

        self._timeout_secs = QSpinBox()
        self._timeout_secs.setRange(5, 120)
        self._timeout_secs.setValue(int(initial_tool_cfg.get("timeout_secs", 30) or 30))
        form.addRow("Tool Timeout (s)", self._timeout_secs)

        tools_row = QWidget()
        tools_layout = QVBoxLayout(tools_row)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(4)
        tools_header = QLabel("Disable tools in this preset")
        tools_header.setStyleSheet("color: #8ea58e;")
        tools_layout.addWidget(tools_header)
        self._tool_checkboxes: dict[str, QCheckBox] = {}
        disabled = {
            str(tool).strip().lower()
            for tool in list(initial_tool_cfg.get("disabled_tools", []) or [])
            if str(tool).strip()
        }
        for tool_name in sorted(set(available_tools or [])):
            cb = QCheckBox(tool_name)
            cb.setChecked(tool_name.lower() in disabled)
            tools_layout.addWidget(cb)
            self._tool_checkboxes[tool_name] = cb
        if not self._tool_checkboxes:
            empty = QLabel("No discovered tools yet.")
            empty.setStyleSheet("color: #506050;")
            tools_layout.addWidget(empty)
        form.addRow("Tool Filters", tools_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_preset_payload(self) -> dict[str, Any]:
        sampling: dict[str, Any] = {"temperature": float(self._temperature.value())}
        if self._top_p.value() > 0:
            sampling["top_p"] = float(self._top_p.value())
        if self._top_k.value() > 0:
            sampling["top_k"] = int(self._top_k.value())
        if self._max_tokens.value() > 0:
            sampling["max_tokens"] = int(self._max_tokens.value())
        if self._repeat_penalty.value() > 0:
            sampling["repeat_penalty"] = float(self._repeat_penalty.value())

        disabled_tools = [
            name
            for name, cb in self._tool_checkboxes.items()
            if cb.isChecked()
        ]
        return {
            "name": self._name_input.text().strip(),
            "description": self._description_input.text().strip(),
            "system_prompt": self._system_prompt.toPlainText().strip(),
            "sampling": sampling,
            "tool_config": {
                "max_rounds": int(self._max_rounds.value()),
                "timeout_secs": int(self._timeout_secs.value()),
                "disabled_tools": disabled_tools,
            },
            "model_override": self._model_override.text().strip() or None,
        }
