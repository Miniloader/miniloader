from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.hardware_probe import AiBackend
from core.probe_service import get_probe_service
from core.settings_store import SettingsStore

log = logging.getLogger(__name__)


class AppShellLocalAiMixin:

    def _settings_store(self) -> SettingsStore | None:
        if self._vault is None:
            return None
        return SettingsStore(self._vault)

    @staticmethod
    def _get_machine_id() -> str:
        return SettingsStore.get_machine_id()

    def _get_machine_settings_entry(self) -> dict[str, Any]:
        """Synchronous wrapper that returns cached machine settings.

        The async ``_load_machine_settings`` must be called at least once
        before this returns useful data; during the first call it returns
        the cache (which starts empty).
        """
        return dict(getattr(self, "_cached_machine_settings", {}))

    async def _load_machine_settings(self) -> dict[str, Any]:
        store = self._settings_store()
        if store is None:
            return {}
        mid = self._get_machine_id()
        entry = await store.get_machine_settings(mid)
        self._cached_machine_settings = entry
        return entry

    def _save_machine_settings_entry(self, entry: dict[str, Any]) -> None:
        """Update the in-memory cache immediately, then schedule a DB write."""
        self._cached_machine_settings = dict(entry)
        asyncio.create_task(self._save_machine_settings_async(entry))

    async def _save_machine_settings_async(self, entry: dict[str, Any]) -> None:
        store = self._settings_store()
        if store is None:
            return
        mid = self._get_machine_id()
        entry = dict(entry)
        entry["updated_at"] = int(time.time())
        await store.save_machine_settings(mid, entry)
        self._cached_machine_settings = entry

    def _persist_backend_selection(
        self,
        selected_backend: str | None,
        *,
        extra_entry: dict[str, Any] | None = None,
    ) -> str | None:
        """Persist explicit user backend selection and update hypervisor cache."""
        selected = str(selected_backend or "").strip().lower()
        if selected not in {"vulkan", "cpu"}:
            return None

        self.hypervisor.set_stored_backend(selected)
        if self._vault is not None:
            machine = self._get_machine_settings_entry()
            if extra_entry:
                machine.update(extra_entry)
            machine.update(
                {
                    "target_backend": selected,
                    "selected_backend": selected,
                    "setup_completed": True,
                    "llm_setup_completed": True,
                }
            )
            self._save_machine_settings_entry(machine)
        return selected

    def _get_local_ai_status(self) -> dict[str, Any]:
        machine = self._get_machine_settings_entry()
        try:
            hw = get_probe_service().hardware()
            detected_backend = hw.ai_backend_hint.value
            gpu_name = hw.gpu_name or "Unknown GPU"
            gpu_driver_version = hw.gpu_driver_version or ""
        except Exception:
            detected_backend = "unknown"
            gpu_name = "Unknown GPU"
            gpu_driver_version = ""

        selected_backend = str(
            machine.get("selected_backend", machine.get("target_backend", ""))
        ).strip().lower()
        preferred_backend = selected_backend or detected_backend
        backend_mismatch_reason = ""
        try:
            backend_mismatch_reason = get_probe_service().explain_mismatch(AiBackend(preferred_backend))
        except ValueError:
            # Legacy/invalid stored value. Keep the user-selected value intact
            # and avoid silently remapping to another backend.
            preferred_backend = detected_backend
            backend_mismatch_reason = (
                f"Stored backend '{selected_backend}' is invalid. "
                "Open LLM Backend Setup and choose Vulkan or CPU explicitly."
            )

        runtime_backend = "idle"
        runtime_reason = "No basic brain loaded"
        requested_gpu_layers = 0
        rag_status = "not_checked"
        rag_status_detail = ""
        rag_deps_installed = False
        rag_model_path = ""

        for module in self.hypervisor.active_modules.values():
            if module.MODULE_NAME == "basic_brain":
                runtime_backend = str(module.params.get("_runtime_backend", runtime_backend))
                runtime_reason = str(module.params.get("_runtime_backend_reason", runtime_reason))
                requested_gpu_layers = int(module.params.get("_requested_gpu_layers", 0) or 0)
            elif module.MODULE_NAME == "rag_engine":
                rag_status = str(module.params.get("_rag_status", rag_status))
                rag_status_detail = str(module.params.get("_rag_status_detail", rag_status_detail))
                rag_deps_installed = bool(module.params.get("_rag_deps_installed", False))
                rag_model_path = str(module.params.get("embedding_model_path", "")).strip()

        return {
            "gpu_name": gpu_name,
            "gpu_driver_version": gpu_driver_version,
            "detected_backend": detected_backend,
            "selected_backend": selected_backend,
            "preferred_backend": preferred_backend,
            "verified_backends": machine.get("verified_backends", {}),
            "backend_mismatch_reason": backend_mismatch_reason,
            "backend_build_tag": machine.get("backend_build_tag", ""),
            "last_install_result": machine.get("last_install_result", "never_run"),
            "last_failure_reason": machine.get("last_failure_reason", ""),
            "last_install_log": machine.get("last_install_log", ""),
            "runtime_backend": runtime_backend,
            "runtime_reason": runtime_reason,
            "requested_gpu_layers": requested_gpu_layers,
            "setup_completed": bool(machine.get("setup_completed", False)),
            "llm_install_result": machine.get("llm_install_result", machine.get("last_install_result", "never_run")),
            "rag_install_result": machine.get("rag_install_result", "never_run"),
            "llm_setup_completed": machine.get("llm_setup_completed"),
            "rag_setup_completed": machine.get("rag_setup_completed"),
            "rag_status": rag_status,
            "rag_status_detail": rag_status_detail,
            "rag_deps_installed": rag_deps_installed,
            "rag_model_path": rag_model_path,
        }

    async def _check_new_machine(self) -> None:
        """Create initial machine entry without auto-selecting backend."""
        store = self._settings_store()
        if store is None:
            return
        mid = self._get_machine_id()
        exists = await store.machine_exists(mid)
        if exists:
            await self._load_machine_settings()
            return

        try:
            hw = get_probe_service().hardware()
            detected = hw.ai_backend_hint.value
        except Exception:
            detected = "unknown"

        entry = {
            "detected_backend": detected,
            "recommended_backend": detected if detected in {"vulkan", "cpu"} else "vulkan",
            "target_backend": "",
            "selected_backend": "",
            "setup_completed": False,
            "llm_setup_completed": False,
        }
        self._save_machine_settings_entry(entry)
        self.hypervisor.set_stored_backend(None)
        log.info(
            "New machine profile created with detected backend hint: %s. "
            "User backend selection is required.",
            detected,
        )

    def _run_local_ai_setup(
        self,
        preferred_backend: str | None = None,
        service: str | None = None,
    ) -> str | None:
        """Open backend selection dialog. Returns the selected backend
        string (e.g. ``"vulkan"``, ``"cpu"``) or ``None`` if cancelled."""
        from ui.local_ai_setup import LocalAiSetupDialog
        dialog = LocalAiSetupDialog(
            parent=self,
            preferred_backend=preferred_backend,
            service_filter=service,
        )
        dialog.exec()
        selected = None
        if dialog.result_info:
            selected = self._persist_backend_selection(
                dialog.result_info.get("selected_backend"),
                extra_entry=dialog.result_info,
            )
        self._settings_page._refresh_local_ai_status()
        return selected
