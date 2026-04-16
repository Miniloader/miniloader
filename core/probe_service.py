from __future__ import annotations

import threading
import time
from typing import Any

from core.hardware_probe import (
    AiBackend,
    HardwareSnapshot,
    explain_backend_mismatch,
    format_vulkan_unavailable_message,
    get_backend_diagnostics_subprocess,
    get_hardware_snapshot,
)


class ProbeService:
    """Thread-safe probe facade with lightweight TTL caching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hardware_snapshot: HardwareSnapshot | None = None
        self._hardware_ts = 0.0
        self._backend_diag: dict[str, Any] | None = None
        self._backend_diag_ts = 0.0
        self._backend_key = ""

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def invalidate(self) -> None:
        with self._lock:
            self._hardware_snapshot = None
            self._hardware_ts = 0.0
            self._backend_diag = None
            self._backend_diag_ts = 0.0
            self._backend_key = ""

    def hardware(self, max_age_s: float = 300.0) -> HardwareSnapshot:
        now = self._now()
        with self._lock:
            cached = self._hardware_snapshot
            cached_ts = self._hardware_ts
        if cached is not None and max_age_s > 0 and (now - cached_ts) <= max_age_s:
            return cached
        return self.hardware_live()

    def hardware_live(self) -> HardwareSnapshot:
        snapshot = get_hardware_snapshot()
        with self._lock:
            self._hardware_snapshot = snapshot
            self._hardware_ts = self._now()
        return snapshot

    def backend_diagnostics(
        self,
        *,
        backend: str | None = None,
        max_age_s: float = 600.0,
    ) -> dict[str, Any]:
        key = str(backend or "").strip().lower()
        now = self._now()
        with self._lock:
            cached = self._backend_diag
            cached_ts = self._backend_diag_ts
            cached_key = self._backend_key
        if (
            cached is not None
            and key == cached_key
            and max_age_s > 0
            and (now - cached_ts) <= max_age_s
        ):
            return dict(cached)

        diag = get_backend_diagnostics_subprocess(backend if key else None)
        with self._lock:
            self._backend_diag = dict(diag)
            self._backend_diag_ts = self._now()
            self._backend_key = key
        return dict(diag)

    def explain_mismatch(
        self,
        backend: AiBackend,
        *,
        max_age_s: float = 600.0,
    ) -> str:
        diag = self.backend_diagnostics(max_age_s=max_age_s)
        return explain_backend_mismatch(backend, diag)

    def verify_backend(
        self,
        backend: AiBackend,
        *,
        max_age_s: float = 600.0,
    ) -> list[str]:
        reason = self.explain_mismatch(backend, max_age_s=max_age_s)
        if not reason:
            return []
        if backend == AiBackend.VULKAN:
            return [format_vulkan_unavailable_message(reason)]
        return [
            f"CPU backend is unavailable: {reason}. "
            f"Use Settings > Force CPU Setup to repair it."
        ]

    def classify_hw_tier(self, snapshot: HardwareSnapshot | None = None) -> str:
        hw = snapshot if snapshot is not None else self.hardware()
        vram_mb = float(hw.vram_total_mb)
        ram_mb = float(hw.ram_total_mb)
        has_dgpu = bool(hw.gpu_name.strip()) and hw.gpu_name.lower() not in {"unknown", "none"}
        if has_dgpu and vram_mb >= 8192.0:
            return "high"
        if vram_mb >= 4096.0 or ram_mb >= 16384.0:
            return "medium"
        return "low"

    def backend_ready_status(
        self,
        backend: str | None = None,
        *,
        max_age_s: float = 600.0,
    ) -> tuple[str, bool]:
        diag = self.backend_diagnostics(backend=backend, max_age_s=max_age_s)
        llm = diag.get("llm_backends", {})
        if not isinstance(llm, dict):
            llm = {}
        vulkan_ok = bool(llm.get("vulkan", False))
        cpu_ok = bool(llm.get("cpu", False))
        if vulkan_ok:
            return ("Vulkan ready", True)
        if cpu_ok:
            return ("CPU ready", True)
        return ("No backend verified", False)


_probe_service: ProbeService | None = None
_probe_service_lock = threading.Lock()


def get_probe_service() -> ProbeService:
    global _probe_service  # noqa: PLW0603
    if _probe_service is None:
        with _probe_service_lock:
            if _probe_service is None:
                _probe_service = ProbeService()
    return _probe_service
