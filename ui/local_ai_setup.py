"""
local_ai_setup.py — Backend selection wizard.
===============================================
Detects GPU hardware, recommends Vulkan or CPU, and lets the user
confirm which backend to use. The selected backend is treated as
authoritative: if verification fails, setup shows an error and does
not silently switch to another backend.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QPushButton,
)

from core.backend_downloader import (
    get_backend_lib_path,
    get_vendor_dir,
    is_backend_ready,
)
from core.hardware_probe import (
    AiBackend,
    HardwareSnapshot,
)
from core.probe_service import get_probe_service

log = logging.getLogger(__name__)

_PY_RUNTIME = f"{sys.version_info.major}.{sys.version_info.minor}"


def register_backend_lib(backend: str) -> bool:
    """Point ``LLAMA_CPP_LIB_PATH`` at the directory containing the
    bundled backend DLL.  Must be called before the first
    ``import llama_cpp`` so the correct native library is loaded.

    ``llama_cpp`` reads ``LLAMA_CPP_LIB_PATH`` at import time and
    searches that directory for ``llama.dll`` (Windows) or
    ``libllama.so`` (Linux).

    Returns True if the backend library was found and registered.
    """
    lib_name = "llama.dll" if sys.platform == "win32" else "libllama.so"
    lib_path = get_backend_lib_path(backend)
    if lib_path is None:
        vendor_candidate = get_vendor_dir() / lib_name
        if vendor_candidate.is_file():
            lib_path = vendor_candidate
    if lib_path is None and getattr(sys, "frozen", False):
        from core.backend_downloader import _frozen_llama_bin_dir

        frozen_bin = _frozen_llama_bin_dir()
        if frozen_bin is not None:
            frozen_candidate = frozen_bin / lib_name
            if frozen_candidate.is_file():
                lib_path = frozen_candidate
    if lib_path is None or not lib_path.is_file():
        return False

    lib_dir = str(lib_path.parent)

    os.environ["LLAMA_CPP_LIB_PATH"] = lib_dir

    if sys.platform == "win32":
        cur = os.environ.get("PATH", "")
        if lib_dir not in cur.split(os.pathsep):
            os.environ["PATH"] = f"{lib_dir}{os.pathsep}{cur}"
        try:
            os.add_dll_directory(lib_dir)
        except OSError:
            pass
    else:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in ld_path.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_dir}{os.pathsep}{ld_path}" if ld_path else lib_dir
            )

    log.info("Registered %s backend: %s", backend, lib_path)
    return True


def _verify_backend(backend: str) -> dict[str, Any]:
    """Verify that llama_cpp works with *backend* in a clean subprocess.

    Running in a subprocess avoids the UI-process problem where
    ``llama_cpp`` has already been imported without ggml backend DLLs,
    which makes ``llama_supports_gpu_offload()`` permanently return
    ``False``.
    """
    register_backend_lib(backend)
    return get_probe_service().backend_diagnostics(backend=backend, max_age_s=0.0)


class LocalAiSetupDialog(QDialog):
    """Backend selection dialog.  Detects hardware, recommends Vulkan
    or CPU, and lets the user confirm.  No installs or downloads."""

    _probe_finished = Signal(object)

    def __init__(
        self,
        parent=None,
        preferred_backend: str | None = None,
        service_filter: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._hw: HardwareSnapshot | None = None
        self._preferred_backend = (preferred_backend or "").strip().lower()
        self._service_filter = (service_filter or "").strip().lower() or None
        self._target_backend = AiBackend.CPU
        self.installed_ok = False
        self.result_info: dict[str, object] = {}

        self.setWindowTitle("LLM Backend Setup")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.resize(560, 420)

        self._probe_finished.connect(self._on_probe_finished)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        title = QLabel("LLM Backend Setup")
        title.setStyleSheet("color: #d8d8d8; font-size: 19px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(
            "Select the compute backend for local AI inference. "
            "Vulkan is recommended for all GPUs (NVIDIA, AMD, Intel)."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9ea8b8; font-size: 12px;")
        root.addWidget(subtitle)

        # Hardware summary
        self._hw_label = QLabel("Detecting hardware…")
        self._hw_label.setWordWrap(True)
        self._hw_label.setStyleSheet(
            "background: #161a22; border: 1px solid #30384a; color: #cdd8e8;"
            " padding: 10px; border-radius: 6px; font-size: 12px;"
        )
        root.addWidget(self._hw_label)

        # Backend choice
        choice_label = QLabel("Choose backend:")
        choice_label.setStyleSheet("color: #b8c4d8; font-size: 13px; font-weight: 600;")
        root.addWidget(choice_label)

        self._radio_vulkan = QRadioButton("Vulkan — GPU-accelerated (recommended for most GPUs)")
        self._radio_cpu = QRadioButton("CPU — no GPU required (slower, always works)")

        for rb in (self._radio_vulkan, self._radio_cpu):
            rb.setStyleSheet(
                "QRadioButton { color: #c8d4e8; font-size: 12px; spacing: 8px; }"
                "QRadioButton::indicator { width: 14px; height: 14px; }"
            )
            root.addWidget(rb)

        self._radio_group = QButtonGroup(self)
        self._radio_group.addButton(self._radio_vulkan, 0)
        self._radio_group.addButton(self._radio_cpu, 1)

        # Status line (populated after probe)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "color: #8ea58e; font-size: 11px; padding: 4px 0;"
        )
        root.addWidget(self._status_label)

        root.addStretch(1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._skip_btn = QPushButton("Skip for Now")
        self._skip_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1c1f26; color: #9ea8b8; border: 1px solid #3a3e48;"
            "  border-radius: 6px; padding: 8px 18px; font-size: 12px;"
            "}"
            "QPushButton:hover { background: #24272f; }"
        )
        self._skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._skip_btn)

        self._confirm_btn = QPushButton("Continue")
        self._confirm_btn.setStyleSheet(
            "QPushButton {"
            "  background: #1a2e1a; color: #b6d9b6; border: 1px solid #39d353;"
            "  border-radius: 6px; padding: 8px 22px; font-size: 12px;"
            "}"
            "QPushButton:hover { background: #1e3a1e; }"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(self._confirm_btn)

        root.addLayout(btn_row)

        self._detect_hardware()

    # ── Hardware detection ─────────────────────────────────────────

    def _detect_hardware(self) -> None:
        try:
            self._hw = get_probe_service().hardware()
        except Exception as exc:
            self._hw_label.setText(f"Hardware detection failed: {exc}")
            return

        hw = self._hw
        target = hw.ai_backend_hint
        try:
            if self._preferred_backend:
                target = AiBackend(self._preferred_backend)
        except ValueError:
            target = hw.ai_backend_hint

        if target not in (AiBackend.VULKAN, AiBackend.CPU):
            target = AiBackend.VULKAN

        self._target_backend = target

        lines = [
            f"GPU:       {hw.gpu_name or 'Not detected'}",
            f"Vendor:    {hw.gpu_vendor.value.upper()}",
        ]
        if hw.gpu_driver_version:
            lines.append(f"Driver:    {hw.gpu_driver_version}")
        if hw.vram_total_mb > 0:
            lines.append(f"VRAM:      {hw.vram_total_mb:.0f} MB")
        lines.append(f"OS:        {hw.os_info}")
        lines.append(f"Runtime:   Python {_PY_RUNTIME}")
        self._hw_label.setText("\n".join(lines))

        if target == AiBackend.VULKAN:
            self._radio_vulkan.setChecked(True)
        else:
            self._radio_cpu.setChecked(True)

        vulkan_ok = is_backend_ready("vulkan")
        cpu_ok = is_backend_ready("cpu")

        if not vulkan_ok:
            self._radio_vulkan.setText(
                "Vulkan — GPU-accelerated (currently not ready)"
            )

        if not cpu_ok:
            self._radio_cpu.setText(
                "CPU — no GPU required (currently not ready)"
            )

        self._status_label.setText("Verifying backend…")
        import threading
        threading.Thread(
            target=self._probe_backend,
            args=(target.value,),
            daemon=True,
        ).start()

    def _probe_backend(self, backend: str) -> None:
        try:
            diag = _verify_backend(backend)
        except Exception as exc:
            diag = {"error": str(exc)}
        self._probe_finished.emit(diag)

    def _on_probe_finished(self, diag_obj: object) -> None:
        diag = diag_obj if isinstance(diag_obj, dict) else {}
        llama = diag.get("llama_cpp", {})
        if not isinstance(llama, dict):
            llama = {}

        active = self._target_backend.value
        import_ok = llama.get("import_ok", False)
        gpu_offload = llama.get("gpu_offload", False)

        if import_ok and (active == "cpu" or gpu_offload):
            self._status_label.setText(
                f"✓ {active.upper()} backend verified and ready."
            )
            self._status_label.setStyleSheet(
                "color: #6adf6a; font-size: 11px; padding: 4px 0;"
            )
        elif import_ok and active == "vulkan" and not gpu_offload:
            self._status_label.setText(
                f"⚠ llama_cpp loaded but GPU offload not detected. "
                f"Vulkan may still work — try continuing."
            )
            self._status_label.setStyleSheet(
                "color: #e8d880; font-size: 11px; padding: 4px 0;"
            )
        else:
            err = llama.get("error", "unknown error")
            self._status_label.setText(
                f"⚠ Could not verify backend: {err}"
            )
            self._status_label.setStyleSheet(
                "color: #e8b880; font-size: 11px; padding: 4px 0;"
            )

    # ── Confirm ────────────────────────────────────────────────────

    def _on_confirm(self) -> None:
        if self._radio_vulkan.isChecked():
            selected = "vulkan"
        else:
            selected = "cpu"

        # Validate the backend the user explicitly selected. Do not
        # silently switch to another backend on failure.
        try:
            diag = _verify_backend(selected)
        except Exception as exc:
            diag = {"llama_cpp": {"import_ok": False, "error": str(exc)}}

        llama = diag.get("llama_cpp", {}) if isinstance(diag, dict) else {}
        if not isinstance(llama, dict):
            llama = {}
        import_ok = bool(llama.get("import_ok", False))
        gpu_offload = bool(llama.get("gpu_offload", False))
        error_text = str(llama.get("error", "")).strip()

        verify_ok = import_ok and (selected == "cpu" or gpu_offload)
        if not verify_ok:
            detail = error_text or (
                "GPU offload not detected for Vulkan"
                if selected == "vulkan"
                else "llama_cpp import/CPU backend check failed"
            )
            self._status_label.setText(
                f"✗ {selected.upper()} verification failed: {detail}"
            )
            self._status_label.setStyleSheet(
                "color: #ff9b9b; font-size: 11px; padding: 4px 0;"
            )
            rec = "CPU" if selected == "vulkan" else "Vulkan"
            QMessageBox.warning(
                self,
                "Backend verification failed",
                (
                    f"The selected backend ({selected.upper()}) failed verification.\n\n"
                    f"{detail}\n\n"
                    f"No fallback was applied. If needed, switch to {rec} and retry."
                ),
            )
            return

        register_backend_lib(selected)

        detected = self._hw.ai_backend_hint.value if self._hw else "cpu"
        self.result_info = {
            "detected_backend": detected,
            "target_backend": selected,
            "selected_backend": selected,
            "setup_completed": True,
            "llm_setup_completed": True,
        }
        self.installed_ok = True
        self.accept()
