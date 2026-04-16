"""Settings page — machine-level backend, model cache, and diagnostics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from huggingface_hub import scan_cache_dir
except ImportError:
    scan_cache_dir = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from core.vault import VaultManager

from core.agent_preset_store import AgentPresetStore
from ui.pages.agent_preset_dialog import AgentPresetDialog

_MONO = "'Consolas', 'Courier New', monospace"

_CARD_SS = (
    "background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "  stop:0 #0e120e, stop:1 #080c08);"
    " border: 1px solid #1e281e; border-radius: 6px;"
)

_BTN_SS = (
    "QPushButton {"
    f"  background: #0c140c; color: #8ea58e; border: 1px solid #2a3a2a;"
    f"  border-radius: 4px; padding: 5px 14px; font-size: 11px;"
    f"  font-family: {_MONO}; font-weight: 600;"
    "}"
    "QPushButton:hover { background: #142014; }"
    "QPushButton:pressed { background: #0a100a; }"
)


def _scan_cached_models() -> list[dict[str, Any]]:
    """Return Hugging Face model entries present in the local cache."""
    if scan_cache_dir is None:
        return []
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for repo in sorted(
        getattr(cache_info, "repos", []),
        key=lambda r: str(getattr(r, "repo_id", "")),
    ):
        repo_type = str(getattr(repo, "repo_type", "") or "")
        if repo_type not in {"", "model"}:
            continue
        model_id = str(getattr(repo, "repo_id", "") or "").strip()
        if not model_id:
            continue
        size_bytes = int(getattr(repo, "size_on_disk", 0) or 0)
        revisions = list(getattr(repo, "revisions", []) or [])
        results.append(
            {
                "model_id": model_id,
                "size_bytes": size_bytes,
                "revisions": len(revisions),
            }
        )
    return results


def _fmt_bytes(value: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024 and idx < len(suffixes) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {suffixes[idx]}"


def _section_card(title: str) -> tuple[QWidget, QVBoxLayout]:
    """Return a rack-styled LCD panel card and its inner body layout."""
    card = QWidget()
    card.setObjectName("SectionCard")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(f"#SectionCard {{ {_CARD_SS} }}")

    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    header_row = QHBoxLayout()
    header_row.setContentsMargins(14, 10, 14, 8)
    header_row.setSpacing(8)

    led = QLabel()
    led.setFixedSize(7, 7)
    led.setStyleSheet(
        "background: #d4a020; border: 1px solid #8a6a10; border-radius: 3px;"
    )
    header_row.addWidget(led, 0, Qt.AlignmentFlag.AlignVCenter)

    header_lbl = QLabel(title)
    header_lbl.setStyleSheet(
        f"color: #607060; font-size: 10px; font-weight: 700;"
        f" letter-spacing: 1px; font-family: {_MONO};"
        f" background: transparent; border: none; padding: 0;"
    )
    header_row.addWidget(header_lbl)
    header_row.addStretch(1)
    outer.addLayout(header_row)

    sep = QWidget()
    sep.setFixedHeight(1)
    sep.setStyleSheet("background: #1e281e;")
    outer.addWidget(sep)

    body = QWidget()
    body.setStyleSheet("background: transparent;")
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(14, 10, 14, 12)
    body_layout.setSpacing(8)
    outer.addWidget(body)

    return card, body_layout


class _SettingsPage(QWidget):
    """Machine-level settings: backend, model cache, and startup diagnostics."""

    def __init__(
        self,
        vault: VaultManager | None,
        startup_errors: list[dict[str, str]] | None = None,
        on_run_ai_setup: Callable[[str | None, str | None], None] | None = None,
        get_local_ai_status: Callable[[], dict[str, Any]] | None = None,
        on_open_test_suite: Callable[[], None] | None = None,
        get_template_preferences: Callable[[], dict[str, bool]] | None = None,
        on_set_autosave_on_close: Callable[[bool], None] | None = None,
        on_apply_agent_preset: Callable[[dict[str, Any]], None] | None = None,
        get_active_agent_preset_name: Callable[[], str] | None = None,
        get_agent_tool_names: Callable[[], list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._vault = vault
        self._startup_errors = list(startup_errors or [])
        self._on_run_ai_setup = on_run_ai_setup
        self._get_local_ai_status = get_local_ai_status
        self._on_open_test_suite = on_open_test_suite
        self._get_template_preferences = get_template_preferences
        self._on_set_autosave_on_close = on_set_autosave_on_close
        self._on_apply_agent_preset = on_apply_agent_preset
        self._get_active_agent_preset_name = get_active_agent_preset_name
        self._get_agent_tool_names = get_agent_tool_names
        self._agent_preset_rows_layout: QVBoxLayout | None = None
        self._agent_preset_status_label: QLabel | None = None
        self._agent_presets: list[dict[str, Any]] = []
        self.setStyleSheet("background: #121316;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        outer.addWidget(scroll)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        scroll.setWidget(body)

        self._build_backend_section(layout)
        self._build_rag_section(layout)
        self._build_cache_section(layout)
        self._build_template_preferences_section(layout)
        self._build_agent_presets_section(layout)
        self._build_test_section(layout)
        self._build_errors_section(layout)
        layout.addStretch(1)

        self._refresh_local_ai_status()
        self._refresh_rag_status()
        asyncio.create_task(self._seed_and_refresh_agent_presets())

    # ── LLM Backend ──────────────────────────────────────────

    def _build_backend_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("LLM INFERENCE")

        self._llm_status_label = QLabel("Loading LLM status\u2026")
        self._llm_status_label.setWordWrap(True)
        self._llm_status_label.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"  stop:0 #0a100a, stop:1 #070c07);"
            f" color: #8ea58e; border: 1px solid #1e281e;"
            f" border-radius: 4px; padding: 10px 12px; font-size: 11px;"
            f" font-family: {_MONO};"
        )
        cl.addWidget(self._llm_status_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        change_btn = QPushButton("CHANGE BACKEND")
        change_btn.setStyleSheet(_BTN_SS)
        change_btn.clicked.connect(lambda: self._trigger_ai_setup(None, "llm"))
        btn_row.addWidget(change_btn)

        cpu_btn = QPushButton("SWITCH TO CPU")
        cpu_btn.setStyleSheet(_BTN_SS)
        cpu_btn.clicked.connect(lambda: self._trigger_ai_setup("cpu", "llm"))
        btn_row.addWidget(cpu_btn)

        btn_row.addStretch(1)
        cl.addLayout(btn_row)
        parent.addWidget(card)

    def _trigger_ai_setup(
        self,
        preferred_backend: str | None = None,
        service: str | None = None,
    ) -> None:
        if self._on_run_ai_setup is not None:
            self._on_run_ai_setup(preferred_backend, service)
        self._refresh_local_ai_status()

    def _refresh_local_ai_status(self) -> None:
        if self._get_local_ai_status is None:
            self._llm_status_label.setText("Local AI status unavailable.")
            return
        status = self._get_local_ai_status()

        gpu_name = str(status.get("gpu_name", "")).strip()
        gpu_driver = str(status.get("gpu_driver_version", "")).strip()
        selected = str(status.get("selected_backend", "not configured")).strip().upper()
        setup_done = bool(status.get("setup_completed", False))

        lines = [f"GPU      {gpu_name or 'Unknown'}"]
        if gpu_driver:
            lines.append(f"DRIVER   {gpu_driver}")
        lines.append(f"BACKEND  {selected}")
        lines.append(f"SETUP    {'Complete' if setup_done else 'Not configured'}")
        self._llm_status_label.setText("\n".join(lines))

    # ── RAG Setup ─────────────────────────────────────────────

    def _build_rag_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("RAG EMBEDDINGS")

        self._rag_status_label = QLabel("Loading RAG setup status…")
        self._rag_status_label.setWordWrap(True)
        self._rag_status_label.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"  stop:0 #0a100a, stop:1 #070c07);"
            f" color: #8ea58e; border: 1px solid #1e281e;"
            f" border-radius: 4px; padding: 10px 12px; font-size: 11px;"
            f" font-family: {_MONO};"
        )
        cl.addWidget(self._rag_status_label)

        parent.addWidget(card)

    def _refresh_rag_status(self) -> None:
        if not hasattr(self, "_rag_status_label"):
            return
        status = self._get_local_ai_status() if self._get_local_ai_status is not None else {}
        model_path = str(status.get("rag_model_path", "")).strip()
        rag_status = str(status.get("rag_status", "not_checked")).strip() or "not_checked"
        rag_detail = str(status.get("rag_status_detail", "")).strip() or "-"
        setup_done = bool(status.get("rag_setup_completed", False))
        model_present = bool(model_path)
        lines = [
            f"MODEL PATH             {model_path if model_present else 'NOT CONFIGURED'}",
            f"MODEL STATUS           {'CONFIGURED' if model_present else 'MISSING'}",
            f"RAG STATUS             {rag_status}",
            f"DETAIL                 {rag_detail}",
            f"SETUP                  {'Complete' if setup_done else 'Not configured'}",
        ]
        self._rag_status_label.setText("\n".join(lines))

    # ── AI Model Cache ───────────────────────────────────────

    def _build_cache_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("AI MODEL CACHE")

        models = _scan_cached_models()
        if not models:
            empty = QLabel("No models found in the HuggingFace cache.")
            empty.setStyleSheet(
                f"color: #405040; font-size: 11px; font-family: {_MONO};"
                f" border: none; padding: 0;"
            )
            cl.addWidget(empty)
            parent.addWidget(card)
            return

        total_bytes = sum(m["size_bytes"] for m in models)
        summary = QLabel(
            f"{len(models)} model{'s' if len(models) != 1 else ''} cached"
            f"  \u00b7  {_fmt_bytes(total_bytes)} total"
        )
        summary.setStyleSheet(
            f"color: #8ea58e; font-size: 11px; font-weight: 600;"
            f" font-family: {_MONO}; border: none; padding: 0;"
        )
        cl.addWidget(summary)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #1e281e;")
        cl.addWidget(sep)

        for info in models:
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(10)

            name_lbl = QLabel(info["model_id"])
            name_lbl.setStyleSheet(
                f"color: #8ea58e; font-size: 11px; font-family: {_MONO};"
                f" border: none; padding: 0;"
            )
            name_lbl.setWordWrap(True)
            row.addWidget(name_lbl, 1)

            rev_count = info["revisions"]
            meta_lbl = QLabel(f"{rev_count} rev{'s' if rev_count != 1 else ''}")
            meta_lbl.setStyleSheet(
                f"color: #506050; font-size: 10px; font-family: {_MONO};"
                f" border: none; padding: 0;"
            )
            row.addWidget(meta_lbl)

            size_lbl = QLabel(_fmt_bytes(info["size_bytes"]))
            size_lbl.setStyleSheet(
                f"color: #30e848; font-size: 11px; font-weight: 600;"
                f" font-family: {_MONO}; border: none; padding: 0;"
            )
            size_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(size_lbl)

            cl.addLayout(row)

        parent.addWidget(card)

    # ── Test + Benchmark Suite ───────────────────────────────

    def _build_test_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("TEST & BENCHMARK SUITE")

        summary = QLabel(
            "Run compatibility tests and benchmarks with live logs,\n"
            "then export JSON/CSV results."
        )
        summary.setWordWrap(True)
        summary.setStyleSheet(
            f"color: #8ea58e; font-size: 11px; font-family: {_MONO};"
            f" border: none; padding: 0;"
        )
        cl.addWidget(summary)

        row = QHBoxLayout()
        open_btn = QPushButton("OPEN TEST SUITE")
        open_btn.setStyleSheet(_BTN_SS)
        open_btn.clicked.connect(self._open_test_suite)
        row.addWidget(open_btn)
        row.addStretch(1)
        cl.addLayout(row)
        parent.addWidget(card)

    def _open_test_suite(self) -> None:
        if self._on_open_test_suite is not None:
            self._on_open_test_suite()

    # ── Template Preferences ───────────────────────────────────

    def _build_template_preferences_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("TEMPLATES")

        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel("Autosave template on close")
        label.setStyleSheet(
            f"color: #8ea58e; font-size: 11px; font-family: {_MONO};"
            f" border: none; padding: 0;"
        )
        row.addWidget(label)
        row.addStretch(1)

        self._autosave_on_close_checkbox = QCheckBox("Enable")
        self._autosave_on_close_checkbox.setStyleSheet(
            f"QCheckBox {{ color: #8ea58e; font-size: 11px; font-family: {_MONO}; }}"
        )
        self._autosave_on_close_checkbox.toggled.connect(self._on_autosave_on_close_toggled)
        row.addWidget(self._autosave_on_close_checkbox)
        cl.addLayout(row)

        hint = QLabel(
            "When enabled, closing the app saves the current template automatically\n"
            "and exits without showing the save/discard prompt."
        )
        hint.setStyleSheet(
            f"color: #506050; font-size: 10px; font-family: {_MONO};"
            f" border: none; padding: 0;"
        )
        hint.setWordWrap(True)
        cl.addWidget(hint)

        enabled = False
        if self._get_template_preferences is not None:
            prefs = self._get_template_preferences()
            enabled = bool(prefs.get("autosave_on_close", False))
        self._autosave_on_close_checkbox.setChecked(enabled)

        parent.addWidget(card)

    def _on_autosave_on_close_toggled(self, checked: bool) -> None:
        if self._on_set_autosave_on_close is not None:
            self._on_set_autosave_on_close(bool(checked))

    def set_autosave_on_close_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_autosave_on_close_checkbox"):
            return
        checkbox = self._autosave_on_close_checkbox
        previous = checkbox.blockSignals(True)
        checkbox.setChecked(bool(enabled))
        checkbox.blockSignals(previous)

    # ── Agent Presets ───────────────────────────────────────────

    def _agent_preset_store(self) -> AgentPresetStore | None:
        if self._vault is None:
            return None
        return AgentPresetStore(self._vault)

    async def _seed_and_refresh_agent_presets(self) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("Sign in to manage agent presets.")
            return
        try:
            await store.seed_defaults()
        except Exception as exc:
            self._set_agent_preset_status(f"Preset seed failed: {exc}")
        await self._refresh_agent_presets_async()

    def _set_agent_preset_status(self, text: str) -> None:
        if self._agent_preset_status_label is not None:
            self._agent_preset_status_label.setText(text)

    def _build_agent_presets_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("AGENT PRESETS")

        self._agent_preset_status_label = QLabel("Loading presets…")
        self._agent_preset_status_label.setWordWrap(True)
        self._agent_preset_status_label.setStyleSheet(
            f"color: #8ea58e; font-size: 11px; font-family: {_MONO}; border: none; padding: 0;"
        )
        cl.addWidget(self._agent_preset_status_label)

        rows_container = QWidget()
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 4, 0, 0)
        rows_layout.setSpacing(6)
        cl.addWidget(rows_container)
        self._agent_preset_rows_layout = rows_layout

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        new_btn = QPushButton("NEW PRESET")
        new_btn.setStyleSheet(_BTN_SS)
        new_btn.clicked.connect(self._new_agent_preset)
        action_row.addWidget(new_btn)

        import_btn = QPushButton("IMPORT JSON")
        import_btn.setStyleSheet(_BTN_SS)
        import_btn.clicked.connect(self._import_agent_preset)
        action_row.addWidget(import_btn)

        export_btn = QPushButton("EXPORT ALL")
        export_btn.setStyleSheet(_BTN_SS)
        export_btn.clicked.connect(self._export_all_agent_presets)
        action_row.addWidget(export_btn)

        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setStyleSheet(_BTN_SS)
        refresh_btn.clicked.connect(lambda: asyncio.create_task(self._refresh_agent_presets_async()))
        action_row.addWidget(refresh_btn)
        action_row.addStretch(1)
        cl.addLayout(action_row)
        parent.addWidget(card)

    def _new_agent_preset(self) -> None:
        dialog = AgentPresetDialog(
            parent=self,
            preset=None,
            available_tools=self._get_agent_tool_names() if self._get_agent_tool_names else [],
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dialog.get_preset_payload()
        if not payload.get("name"):
            QMessageBox.warning(self, "Preset Name Required", "Please provide a preset name.")
            return
        asyncio.create_task(self._save_agent_preset_async(payload, overwrite=False))

    def _edit_agent_preset(self, preset_name: str) -> None:
        target = next((row for row in self._agent_presets if row.get("name") == preset_name), None)
        if not isinstance(target, dict):
            return
        dialog = AgentPresetDialog(
            parent=self,
            preset=target,
            available_tools=self._get_agent_tool_names() if self._get_agent_tool_names else [],
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dialog.get_preset_payload()
        if not payload.get("name"):
            QMessageBox.warning(self, "Preset Name Required", "Please provide a preset name.")
            return
        overwrite = preset_name == payload.get("name")
        asyncio.create_task(self._save_agent_preset_async(payload, overwrite=overwrite))

    async def _save_agent_preset_async(self, payload: dict[str, Any], *, overwrite: bool) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("No vault available for preset save.")
            return
        name = str(payload.get("name", "")).strip()
        if not name:
            return
        existing = await store.get_preset(name)
        if existing is not None and not overwrite:
            reply = QMessageBox.question(
                self,
                "Overwrite preset?",
                f"Preset '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        is_builtin = bool(payload.get("is_builtin", False))
        await store.save_preset(name, payload, is_builtin=is_builtin)
        self._set_agent_preset_status(f"Saved preset '{name}'.")
        await self._refresh_agent_presets_async()

    def _delete_agent_preset(self, preset_name: str, is_builtin: bool) -> None:
        if is_builtin:
            QMessageBox.information(self, "Built-in Preset", "Built-in presets cannot be deleted.")
            return
        reply = QMessageBox.question(
            self,
            "Delete preset?",
            f"Delete preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        asyncio.create_task(self._delete_agent_preset_async(preset_name))

    async def _delete_agent_preset_async(self, preset_name: str) -> None:
        store = self._agent_preset_store()
        if store is None:
            return
        await store.delete_preset(preset_name)
        self._set_agent_preset_status(f"Deleted preset '{preset_name}'.")
        await self._refresh_agent_presets_async()

    def _load_agent_preset(self, preset_name: str) -> None:
        asyncio.create_task(self._load_agent_preset_async(preset_name))

    async def _load_agent_preset_async(self, preset_name: str) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("No vault available for preset load.")
            return
        preset = await store.get_preset(preset_name)
        if preset is None:
            self._set_agent_preset_status(f"Preset '{preset_name}' was not found.")
            return
        if self._on_apply_agent_preset is not None:
            self._on_apply_agent_preset(preset)
        self._set_agent_preset_status(f"Loaded preset '{preset_name}'.")
        await self._refresh_agent_presets_async()

    async def _refresh_agent_presets_async(self) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("Sign in to manage agent presets.")
            self._agent_presets = []
            self._rebuild_agent_preset_rows([])
            return
        rows = await store.list_presets()
        presets: list[dict[str, Any]] = []
        for row in rows:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            payload = await store.get_preset(name)
            if isinstance(payload, dict):
                presets.append(payload)
        self._agent_presets = presets
        self._rebuild_agent_preset_rows(presets)
        active_name = self._get_active_agent_preset_name() if self._get_active_agent_preset_name else ""
        active_text = active_name or "default"
        self._set_agent_preset_status(
            f"{len(presets)} preset(s) available. Active preset: {active_text}."
        )

    def _rebuild_agent_preset_rows(self, presets: list[dict[str, Any]]) -> None:
        layout = self._agent_preset_rows_layout
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not presets:
            empty = QLabel("No presets found.")
            empty.setStyleSheet(f"color: #506050; font-size: 10px; font-family: {_MONO};")
            layout.addWidget(empty)
            return
        active_name = self._get_active_agent_preset_name() if self._get_active_agent_preset_name else ""
        active_name = str(active_name).strip()
        for preset in presets:
            name = str(preset.get("name", "")).strip()
            if not name:
                continue
            description = str(preset.get("description", "")).strip()
            sampling = preset.get("sampling", {})
            if not isinstance(sampling, dict):
                sampling = {}
            tool_cfg = preset.get("tool_config", {})
            if not isinstance(tool_cfg, dict):
                tool_cfg = {}
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(8)

            summary = QLabel(
                self._preset_summary_text(
                    name=name,
                    description=description,
                    sampling=sampling,
                    max_rounds=tool_cfg.get("max_rounds"),
                    active=(name == active_name),
                )
            )
            summary.setStyleSheet(f"color: #8ea58e; font-size: 10px; font-family: {_MONO};")
            summary.setWordWrap(True)
            row_layout.addWidget(summary, 1)

            load_btn = QPushButton("LOAD")
            load_btn.setStyleSheet(_BTN_SS)
            load_btn.clicked.connect(lambda _checked=False, n=name: self._load_agent_preset(n))
            row_layout.addWidget(load_btn)

            edit_btn = QPushButton("EDIT")
            edit_btn.setStyleSheet(_BTN_SS)
            edit_btn.clicked.connect(lambda _checked=False, n=name: self._edit_agent_preset(n))
            row_layout.addWidget(edit_btn)

            is_builtin = bool(preset.get("is_builtin", False))
            delete_btn = QPushButton("DELETE")
            delete_btn.setStyleSheet(_BTN_SS)
            delete_btn.setEnabled(not is_builtin)
            delete_btn.clicked.connect(
                lambda _checked=False, n=name, b=is_builtin: self._delete_agent_preset(n, b)
            )
            row_layout.addWidget(delete_btn)
            layout.addWidget(row)

    @staticmethod
    def _preset_summary_text(
        *,
        name: str,
        description: str,
        sampling: dict[str, Any],
        max_rounds: Any,
        active: bool,
    ) -> str:
        temp = sampling.get("temperature", 0.7)
        top_p = sampling.get("top_p", "")
        rounds = int(max_rounds or 3)
        marker = " (active)" if active else ""
        details = f"temp:{temp}"
        if top_p != "":
            details += f" top_p:{top_p}"
        details += f" rounds:{rounds}"
        if description:
            return f"{name}{marker} - {details} - {description}"
        return f"{name}{marker} - {details}"

    def _import_agent_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Agent Preset JSON",
            "",
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path:
            return
        asyncio.create_task(self._import_agent_preset_async(path))

    async def _import_agent_preset_async(self, file_path: str) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("No vault available for import.")
            return
        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_agent_preset_status(f"Import failed: {exc}")
            return
        presets = payload if isinstance(payload, list) else [payload]
        imported = 0
        for preset in presets:
            if not isinstance(preset, dict):
                continue
            name = str(preset.get("name", "")).strip()
            if not name:
                continue
            await store.save_preset(name, preset, is_builtin=False)
            imported += 1
        self._set_agent_preset_status(f"Imported {imported} preset(s).")
        await self._refresh_agent_presets_async()

    def _export_all_agent_presets(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Agent Presets",
            "agent_presets.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        asyncio.create_task(self._export_all_agent_presets_async(path))

    async def _export_all_agent_presets_async(self, file_path: str) -> None:
        store = self._agent_preset_store()
        if store is None:
            self._set_agent_preset_status("No vault available for export.")
            return
        rows = await store.list_presets()
        exported: list[dict[str, Any]] = []
        for row in rows:
            payload = await store.get_preset(str(row.get("name", "")))
            if isinstance(payload, dict):
                exported.append(payload)
        Path(file_path).write_text(
            json.dumps(exported, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        self._set_agent_preset_status(f"Exported {len(exported)} preset(s) to {file_path}.")

    # ── Startup Errors ───────────────────────────────────────

    def _build_errors_section(self, parent: QVBoxLayout) -> None:
        card, cl = _section_card("STARTUP DIAGNOSTICS")

        if not self._startup_errors:
            ok = QLabel("No startup errors detected.")
            ok.setStyleSheet(
                f"color: #30e848; font-size: 11px; font-family: {_MONO};"
                f" border: none; padding: 0;"
            )
            cl.addWidget(ok)
            parent.addWidget(card)
            return

        warning = QLabel(
            f"{len(self._startup_errors)} issue(s) detected during startup;"
            " affected modules were skipped."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            f"color: #d4a020; font-size: 11px; font-family: {_MONO};"
            f" border: none; padding: 0;"
        )
        cl.addWidget(warning)

        details = QTextEdit()
        details.setReadOnly(True)
        details.setMinimumHeight(100)
        details.setMaximumHeight(200)
        details.setStyleSheet(
            "QTextEdit {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #0a100a, stop:1 #070c07);"
            f"  color: #8ea58e; border: 1px solid #1e281e;"
            f"  border-radius: 4px; padding: 6px 8px; font-size: 10px;"
            f"  font-family: {_MONO};"
            "}"
        )
        lines: list[str] = []
        for entry in self._startup_errors:
            module = str(entry.get("module", "unknown"))
            phase = str(entry.get("phase", "startup"))
            error = str(entry.get("error", "unknown error"))
            lines.append(f"[{phase}] {module}: {error}")
        details.setPlainText("\n".join(lines))
        cl.addWidget(details)

        parent.addWidget(card)
