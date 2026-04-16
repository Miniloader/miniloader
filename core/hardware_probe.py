from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import ctypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

log = logging.getLogger(__name__)

# ── ggml backend loading (call-once guard) ──────────────────────────
_backends_loaded_mode: str | None = None
_backends_loaded_mode_lock = threading.Lock()


def ensure_ggml_backends(mode: str = "all") -> None:
    """Load ggml backend plugins exactly once per process.

    Handles three things in order:
    1. Registers DLL search directories (bin/, lib/, _MEIPASS) so
       transitive dependencies between ggml DLLs resolve correctly,
       especially in frozen (PyInstaller) builds.
    2. Loads backend plugins from ``llama_cpp/bin`` **individually**
       so that a failure in one backend (e.g. CPU when the native lib
       is already partially initialised) does not prevent others
       (Vulkan, RPC) from loading.

    ``mode="all"`` loads every backend DLL found in the bin directory.
    ``mode="vulkan"`` loads only Vulkan backend DLLs.
    ``mode="cpu"`` loads only CPU backend DLLs.

    Guarded so that duplicate calls are free no-ops — calling the bulk
    loader more than once confuses the llama.cpp scheduler and causes
    a ~30x slowdown.
    """
    global _backends_loaded_mode  # noqa: PLW0603
    raw_mode = str(mode).strip().lower()
    if raw_mode == "cpu":
        chosen_mode = "cpu"
    elif raw_mode == "vulkan":
        chosen_mode = "vulkan"
    else:
        chosen_mode = "all"
    with _backends_loaded_mode_lock:
        if _backends_loaded_mode is not None:
            if _backends_loaded_mode != chosen_mode:
                log.info(
                    "ggml backends already loaded in %s mode; skipping %s request",
                    _backends_loaded_mode,
                    chosen_mode,
                )
            return
        _backends_loaded_mode = chosen_mode
    try:
        if "llama_cpp" in sys.modules:
            # llama_cpp has already initialized native state in this process
            # (common in the UI process after diagnostics). Re-loading ggml
            # backend DLLs is unsafe on Windows and may access-violate.
            _backends_loaded_mode = chosen_mode
            log.info(
                "llama_cpp already imported; skipping explicit ggml backend load (mode=%s)",
                chosen_mode,
            )
            return
        import llama_cpp._ggml as _ggml_mod  # type: ignore
        from pathlib import Path as _P
        from core.backend_downloader import get_vendor_dir

        _pkg = _P(_ggml_mod.__file__).resolve().parent
        _bin = _pkg / "bin"
        _lib = _pkg / "lib"
        _vendor = get_vendor_dir()

        _search_dirs: list[Path] = []
        for _candidate in (_vendor, _bin, _lib):
            if _candidate.is_dir() and str(_candidate) not in {str(d) for d in _search_dirs}:
                _search_dirs.append(_candidate)
        if getattr(sys, "frozen", False):
            _meipass = getattr(sys, "_MEIPASS", None)
            if _meipass:
                _search_dirs.append(_P(_meipass))
            _exe_parent = _P(sys.executable).parent
            if str(_exe_parent) not in {str(d) for d in _search_dirs}:
                _search_dirs.append(_exe_parent)

        for _d in _search_dirs:
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(_d))
                except OSError:
                    pass
            os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

        _valid_ext = {".dll", ".so", ".dylib"}
        _load_roots: list[Path] = []
        if _vendor.is_dir():
            _vendor_has_backends = any(
                f.is_file() and f.suffix in _valid_ext and f.name.lower().startswith("ggml-")
                for f in _vendor.iterdir()
            )
            if _vendor_has_backends:
                _load_roots.append(_vendor)
        if _bin.is_dir():
            _load_roots.append(_bin)

        if _load_roots:
            _load_one = _ggml_mod.libggml.ggml_backend_load
            _load_one.argtypes = [ctypes.c_char_p]
            _load_one.restype = ctypes.c_void_p

            loaded = 0
            skipped: list[str] = []
            seen_names: set[str] = set()

            for _root in _load_roots:
                for dll in sorted(_root.iterdir()):
                    if dll.suffix not in _valid_ext:
                        continue
                    name = dll.name.lower()
                    if not name.startswith("ggml-"):
                        continue
                    if name in seen_names:
                        continue
                    if chosen_mode == "cpu":
                        if not name.startswith("ggml-cpu"):
                            continue
                    elif chosen_mode == "vulkan":
                        if not name.startswith("ggml-vulkan"):
                            continue
                    try:
                        _load_one(str(dll).encode())
                        loaded += 1
                        seen_names.add(name)
                    except Exception as exc_inner:
                        skipped.append(f"{dll.name} ({exc_inner})")

            if skipped:
                log.debug("Skipped backend DLLs: %s", "; ".join(skipped))
            if loaded == 0:
                _roots_text = ", ".join(str(p) for p in _load_roots)
                raise RuntimeError(
                    f"no ggml backends loaded from {_roots_text} (mode={chosen_mode})"
                    + (f" (skipped: {'; '.join(skipped)})" if skipped else "")
                )
            log.info(
                "ggml backends loaded from %s (%d DLLs%s)",
                ", ".join(str(p) for p in _load_roots),
                loaded,
                f", {len(skipped)} skipped" if skipped else "",
            )
        _backends_loaded_mode = chosen_mode
    except Exception as exc:
        _backends_loaded_mode = chosen_mode
        log.warning("Failed to load ggml backends: %s", exc)


# PCI vendor IDs used for sysfs detection
_PCI_VENDOR_AMD = "0x1002"
_PCI_VENDOR_NVIDIA = "0x10de"
_PCI_VENDOR_INTEL = "0x8086"



class GpuVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    OTHER = "other"
    UNKNOWN = "unknown"


class AiBackend(str, Enum):
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass(frozen=True)
class HardwareSnapshot:
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_percent: float
    ram_total_mb: float
    ram_used_mb: float
    ram_available_mb: float
    gpu_vendor: GpuVendor
    gpu_name: str
    gpu_pci_device_id: str
    gpu_driver_version: str
    vram_total_mb: float
    vram_used_mb: float
    ai_backend_hint: AiBackend
    disks: list[str]
    os_info: str


def _classify_gpu_vendor(name: str) -> GpuVendor:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n or "gtx" in n:
        return GpuVendor.NVIDIA
    if "amd" in n or "radeon" in n or "strix" in n:
        return GpuVendor.AMD
    if "intel" in n or "arc" in n or "uhd" in n or "iris" in n:
        return GpuVendor.INTEL
    if name.strip():
        return GpuVendor.OTHER
    return GpuVendor.UNKNOWN


def _classify_gpu_vendor_by_pci(vendor_id: str) -> GpuVendor:
    vid = vendor_id.strip().lower()
    if vid == _PCI_VENDOR_NVIDIA:
        return GpuVendor.NVIDIA
    if vid == _PCI_VENDOR_AMD:
        return GpuVendor.AMD
    if vid == _PCI_VENDOR_INTEL:
        return GpuVendor.INTEL
    if vid:
        return GpuVendor.OTHER
    return GpuVendor.UNKNOWN


# ── Linux GPU detection ─────────────────────────────────────────────

def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return ""


def _read_gpu_linux_sysfs() -> tuple[str, str, str, float, float]:
    """
    Walk /sys/class/drm/card*/device to find the first discrete or
    integrated GPU.  Returns (gpu_name, vendor_id, device_id, vram_total_mb,
    vram_used_mb).  Falls back to lspci if sysfs doesn't yield a name.
    """
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return ("", "", "", 0.0, 0.0)

    for card_dir in sorted(drm.iterdir()):
        if not card_dir.name.startswith("card") or "-" in card_dir.name:
            continue
        dev = card_dir / "device"
        vendor_id = _read_sysfs(dev / "vendor")
        device_id = _read_sysfs(dev / "device")
        if not vendor_id:
            continue

        gpu_name = _read_sysfs(dev / "product_name")
        vram_total, vram_used = _read_amd_vram_sysfs(dev)

        if not gpu_name:
            gpu_name = _gpu_name_from_lspci(vendor_id, device_id)

        if gpu_name or vendor_id:
            return (gpu_name, vendor_id, device_id, vram_total, vram_used)

    return ("", "", "", 0.0, 0.0)


def _read_amd_vram_sysfs(dev: Path) -> tuple[float, float]:
    """Read VRAM from amdgpu sysfs nodes (available when amdgpu driver is loaded)."""
    total_path = dev / "mem_info_vram_total"
    used_path = dev / "mem_info_vram_used"
    total_str = _read_sysfs(total_path)
    used_str = _read_sysfs(used_path)
    try:
        total_mb = int(total_str) / (1024 * 1024) if total_str else 0.0
        used_mb = int(used_str) / (1024 * 1024) if used_str else 0.0
        return (total_mb, used_mb)
    except ValueError:
        return (0.0, 0.0)


def _gpu_name_from_lspci(vendor_id: str, device_id: str) -> str:
    """Use lspci to resolve a human-readable GPU name."""
    lspci = shutil.which("lspci")
    if not lspci:
        return ""
    try:
        result = subprocess.run(
            [lspci, "-nn"],
            capture_output=True, text=True, timeout=5,
        )
        needle_vid = vendor_id.replace("0x", "").lower()
        needle_did = device_id.replace("0x", "").lower()
        needle = f"{needle_vid}:{needle_did}"
        for line in result.stdout.splitlines():
            if needle in line.lower():
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    return parts[1].split(" [")[0].strip()
    except Exception:
        pass
    return ""


def _read_amd_vram_pyamdgpuinfo() -> tuple[float, float]:
    """Optional: use pyamdgpuinfo for VRAM if installed."""
    try:
        import pyamdgpuinfo  # type: ignore
        if pyamdgpuinfo.detect_gpus() > 0:
            gpu = pyamdgpuinfo.get_gpu(0)
            total = gpu.memory_info["vram_size"] / (1024 * 1024)
            used = total - (gpu.query_vram_usage() / (1024 * 1024))
            return (total, max(0.0, used))
    except Exception:
        pass
    return (0.0, 0.0)


# ── NVIDIA VRAM detection (cross-platform) ──────────────────────────

def _read_nvidia_vram_pynvml() -> tuple[float, float]:
    """Read NVIDIA VRAM via pynvml / nvidia-ml-py (works on both Windows and Linux)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = mem.total / (1024 * 1024)
        used = mem.used / (1024 * 1024)
        pynvml.nvmlShutdown()
        return (total, used)
    except Exception:
        pass
    return (0.0, 0.0)


def _read_nvidia_vram_smi() -> tuple[float, float]:
    """Fallback: parse nvidia-smi output for VRAM."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return (0.0, 0.0)
    try:
        result = subprocess.run(
            [smi, "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = result.stdout.strip().splitlines()[0]
        total_str, used_str = line.split(",")
        return (float(total_str.strip()), float(used_str.strip()))
    except Exception:
        pass
    return (0.0, 0.0)


def _read_nvidia_vram() -> tuple[float, float]:
    """Best-effort NVIDIA VRAM: pynvml first, nvidia-smi fallback."""
    total, used = _read_nvidia_vram_pynvml()
    if total > 0:
        return (total, used)
    return _read_nvidia_vram_smi()


# ── Windows GPU detection ───────────────────────────────────────────

_GPU_VENDOR_PRIORITY = {
    GpuVendor.NVIDIA: 0,
    GpuVendor.AMD:    1,
    GpuVendor.OTHER:  2,
    GpuVendor.INTEL:  3,
    GpuVendor.UNKNOWN: 4,
}


@dataclass
class _WmiGpuInfo:
    name: str
    driver_version: str
    vram_mb: float = 0.0


def _read_gpu_info_windows_wmi() -> list[_WmiGpuInfo]:
    """Return all GPUs with name, driver version, and VRAM from WMI.

    ``AdapterRAM`` is a uint32 in WMI, so it overflows at 4 GB.  For
    cards with > 4 GB we fall back to ``qwMemorySize`` from the
    DirectX diagnostic namespace, or dxdiag parsing.
    """
    try:
        import win32com.client  # type: ignore
    except Exception:
        return []
    try:
        locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = locator.ConnectServer(".", "root\\cimv2")
        controllers = service.ExecQuery(
            "SELECT Name, DriverVersion, AdapterRAM FROM Win32_VideoController"
        )
        results: list[_WmiGpuInfo] = []
        for c in controllers:
            name = getattr(c, "Name", None)
            driver = getattr(c, "DriverVersion", None)
            adapter_ram = getattr(c, "AdapterRAM", None)
            vram_mb = 0.0
            if adapter_ram is not None:
                try:
                    vram_mb = int(adapter_ram) / (1024 * 1024)
                except (ValueError, TypeError):
                    pass
            if isinstance(name, str) and name.strip():
                results.append(_WmiGpuInfo(
                    name=name.strip(),
                    driver_version=(driver or "").strip(),
                    vram_mb=vram_mb,
                ))
        return results
    except Exception:
        return []


def _read_vram_windows_registry() -> float:
    """Read dedicated VRAM from the Windows registry (64-bit, no overflow).

    GPU drivers store ``HardwareInformation.qwMemorySize`` (REG_QWORD)
    under the display adapter class key.  This works for AMD, NVIDIA,
    and Intel — unlike WMI ``AdapterRAM`` which is a uint32 and caps
    at ~4 GB.  Returns the largest value found in MB, or 0.0.
    """
    try:
        import winreg
    except ImportError:
        return 0.0

    display_class = (
        r"SYSTEM\ControlSet001\Control\Class"
        r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )
    best_mb = 0.0
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, display_class) as cls_key:
            idx = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(cls_key, idx)
                except OSError:
                    break
                idx += 1
                try:
                    with winreg.OpenKey(cls_key, sub_name) as sub_key:
                        val, _ = winreg.QueryValueEx(
                            sub_key, "HardwareInformation.qwMemorySize"
                        )
                        mb = int(val) / (1024 * 1024)
                        if mb > best_mb:
                            best_mb = mb
                except (OSError, ValueError, TypeError):
                    continue
    except OSError:
        pass

    if best_mb > 0:
        log.debug("Registry qwMemorySize: %.0f MB", best_mb)
    return best_mb


def _pick_best_gpu(gpus: list[_WmiGpuInfo]) -> _WmiGpuInfo:
    """Pick the best GPU from a list, preferring discrete (NVIDIA > AMD)
    over integrated (Intel)."""
    if not gpus:
        return _WmiGpuInfo(name="", driver_version="")
    best = gpus[0]
    best_priority = _GPU_VENDOR_PRIORITY.get(_classify_gpu_vendor(best.name), 4)
    for gpu in gpus[1:]:
        priority = _GPU_VENDOR_PRIORITY.get(_classify_gpu_vendor(gpu.name), 4)
        if priority < best_priority:
            best = gpu
            best_priority = priority
    return best


def _format_nvidia_driver_version(wmi_version: str) -> str:
    """Convert WMI DriverVersion (e.g. '32.0.15.6094') to the public
    NVIDIA driver number (e.g. '560.94').  The last 5 digits of the
    WMI string encode the public version: XXXXX -> XXX.XX."""
    parts = wmi_version.split(".")
    if len(parts) < 4:
        return wmi_version
    try:
        raw = parts[-2] + parts[-1]
        raw = raw.lstrip("0") or "0"
        if len(raw) >= 3:
            major = raw[:-2]
            minor = raw[-2:]
            return f"{major}.{minor}"
    except Exception:
        pass
    return wmi_version


def _read_driver_version_linux() -> str:
    """Best-effort GPU driver version on Linux via modinfo or nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            ver = result.stdout.strip().splitlines()[0].strip()
            if ver:
                return ver
        except Exception:
            pass
    for mod in ("amdgpu", "i915"):
        try:
            result = subprocess.run(
                ["modinfo", "-F", "version", mod],
                capture_output=True, text=True, timeout=5,
            )
            ver = result.stdout.strip()
            if ver:
                return ver
        except Exception:
            pass
    return ""


# ── Backend selection ───────────────────────────────────────────────

def _select_backend(vendor: GpuVendor) -> AiBackend:
    """Vulkan for any recognised GPU, CPU otherwise."""
    if vendor in (GpuVendor.NVIDIA, GpuVendor.AMD, GpuVendor.INTEL, GpuVendor.OTHER):
        return AiBackend.VULKAN
    return AiBackend.CPU


# ── Main entry point ────────────────────────────────────────────────

# Static fields are expensive (WMI, nvidia-smi, disk partitions) and
# don't change between polls.  We probe them once and cache forever.
_static_cache: dict[str, object] = {}
_static_cache_lock = threading.Lock()


def _probe_static() -> dict[str, object]:
    """One-time probe for GPU, vendor, backend, disks, OS.
    Cached so subsequent get_hardware_snapshot() calls are cheap."""
    with _static_cache_lock:
        if _static_cache:
            return _static_cache

    gpu_name = ""
    gpu_driver_version = ""
    pci_device_id = ""
    vram_total = 0.0
    vendor = GpuVendor.UNKNOWN

    system = platform.system()
    if system == "Linux":
        gpu_name, vendor_id, pci_device_id, vram_total, _ = _read_gpu_linux_sysfs()
        vendor = _classify_gpu_vendor(gpu_name) if gpu_name else _classify_gpu_vendor_by_pci(vendor_id)
        if vram_total == 0.0 and vendor == GpuVendor.AMD:
            vram_total, _ = _read_amd_vram_pyamdgpuinfo()
        if vram_total == 0.0 and vendor == GpuVendor.NVIDIA:
            vram_total, _ = _read_nvidia_vram()
        gpu_driver_version = _read_driver_version_linux()
    elif system == "Windows":
        all_gpus = _read_gpu_info_windows_wmi()
        best = _pick_best_gpu(all_gpus)
        gpu_name = best.name
        gpu_driver_version = best.driver_version
        vendor = _classify_gpu_vendor(gpu_name)
        if vendor == GpuVendor.NVIDIA and gpu_driver_version:
            gpu_driver_version = _format_nvidia_driver_version(gpu_driver_version)
        if vendor == GpuVendor.NVIDIA:
            vram_total, _ = _read_nvidia_vram()
        # WMI AdapterRAM (uint32, caps at ~4 GB) as first fallback
        if vram_total == 0.0 and best.vram_mb > 0:
            vram_total = best.vram_mb
            log.info("VRAM from WMI AdapterRAM: %.0f MB", vram_total)
        # Registry qwMemorySize gives the real 64-bit value for any vendor
        if vram_total <= 4096.0:
            dxgi_mb = _read_vram_windows_registry()
            if dxgi_mb > vram_total:
                log.info(
                    "VRAM from registry: %.0f MB (WMI reported %.0f MB)",
                    dxgi_mb, vram_total,
                )
                vram_total = dxgi_mb

    backend = _select_backend(vendor)

    disks: list[str] = []
    seen_mounts: set[str] = set()
    try:
        for part in psutil.disk_partitions(all=False):
            mount = (part.mountpoint or "").strip()
            if not mount or mount in seen_mounts or "cdrom" in (part.opts or "").lower():
                continue
            seen_mounts.add(mount)
            try:
                usage = psutil.disk_usage(mount)
            except Exception:
                continue
            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            label = mount.replace("\\", "")[:3]
            disks.append(f"{label:>3} {free_gb:>5.0f}G free / {total_gb:>5.0f}G")
            if len(disks) >= 3:
                break
    except Exception:
        pass

    if not disks:
        disks = ["No disk telemetry"]

    sys_name = platform.system()
    sys_ver  = platform.version()
    build    = sys_ver.split(".")[-1] if "." in sys_ver else ""
    if sys_name == "Windows":
        build_int = int(build) if build.isdigit() else 0
        win_ver = "11" if build_int >= 22000 else "10"
        os_info = f"Windows {win_ver} build {build}" if build else f"Windows {win_ver}"
    else:
        sys_rel = platform.release()
        os_info = f"{sys_name} {sys_rel} build {build}" if build else f"{sys_name} {sys_rel}"

    with _static_cache_lock:
        if _static_cache:
            return _static_cache
        _static_cache.update({
            "cpu_name":      platform.processor() or "CPU",
            "cpu_phys":      int(psutil.cpu_count(logical=False) or 0),
            "cpu_logical":   int(psutil.cpu_count(logical=True) or 0),
            "gpu_name":      gpu_name,
            "gpu_vendor":    vendor,
            "gpu_pci_device_id": pci_device_id,
            "gpu_driver_version": gpu_driver_version,
            "vram_total_mb": vram_total,
            "ai_backend":    backend,
            "disks":         disks,
            "os_info":       os_info,
        })
        return _static_cache


def _read_vram_used_fast(vendor: GpuVendor) -> float:
    """Cheap VRAM-used poll.  pynvml is fast after first init."""
    if vendor == GpuVendor.NVIDIA:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return mem.used / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def get_hardware_snapshot() -> HardwareSnapshot:
    """Build a snapshot.  Static fields are cached; only CPU %, RAM,
    and VRAM-used are polled each call (all sub-millisecond)."""
    s = _probe_static()

    vm = psutil.virtual_memory()
    cpu_pct = float(psutil.cpu_percent(interval=None))
    vram_used = _read_vram_used_fast(s["gpu_vendor"])  # type: ignore[arg-type]

    return HardwareSnapshot(
        cpu_name=s["cpu_name"],  # type: ignore[arg-type]
        cpu_physical_cores=s["cpu_phys"],  # type: ignore[arg-type]
        cpu_logical_cores=s["cpu_logical"],  # type: ignore[arg-type]
        cpu_percent=cpu_pct,
        ram_total_mb=vm.total / (1024 * 1024),
        ram_used_mb=vm.used / (1024 * 1024),
        ram_available_mb=vm.available / (1024 * 1024),
        gpu_vendor=s["gpu_vendor"],  # type: ignore[arg-type]
        gpu_name=s["gpu_name"],  # type: ignore[arg-type]
        gpu_pci_device_id=s["gpu_pci_device_id"],  # type: ignore[arg-type]
        gpu_driver_version=s["gpu_driver_version"],  # type: ignore[arg-type]
        vram_total_mb=s["vram_total_mb"],  # type: ignore[arg-type]
        vram_used_mb=vram_used,
        ai_backend_hint=s["ai_backend"],  # type: ignore[arg-type]
        disks=s["disks"],  # type: ignore[arg-type]
        os_info=s["os_info"],  # type: ignore[arg-type]
    )


# ── Backend verification / diagnostics ──────────────────────────────

def get_backend_diagnostics() -> dict[str, Any]:
    """Return detailed backend capability diagnostics for UI and setup.

    The result is intentionally JSON-serialisable so it can be stored in
    app settings or shown directly in the setup wizard.

    Keys returned:
      torch, llama_cpp       – individual library probes
      rag                    – RAG stack readiness (chromadb + llama-cpp)
      llm_backends           – LLM inference capabilities (llama-cpp only)
      rag_backends           – RAG embedding capabilities (llama-cpp)
      installed_backends     – kept for backward compat (= llm_backends)
    """
    torch_info: dict[str, Any] = {
        "import_ok": False,
        "version": "",
        "error": "",
    }
    llama_info: dict[str, Any] = {
        "import_ok": False,
        "version": "",
        "gpu_offload": False,
        "error": "",
        "distribution": "",
        "expected_fork": "jamepeng",
        "fork_matches": None,
    }
    rag_info: dict[str, Any] = {
        "ready": False,
        "sentence_transformers_ok": False,
        "sentence_transformers_version": "",
        "chromadb_ok": False,
        "chromadb_version": "",
        "error": "",
    }

    try:
        import torch

        torch_info["import_ok"] = True
        torch_info["version"] = str(getattr(torch, "__version__", ""))
    except Exception as exc:
        torch_info["error"] = str(exc)

    try:
        import llama_cpp  # type: ignore

        llama_info["import_ok"] = True
        llama_info["version"] = str(getattr(llama_cpp, "__version__", ""))

        try:
            from core.llama_runtime import (
                get_llama_package_metadata,
                is_jamepeng_distribution,
            )
            _meta = get_llama_package_metadata()
            llama_info["distribution"] = _meta.get("version", "")
            llama_info["fork_matches"] = is_jamepeng_distribution(_meta)
        except Exception:
            pass

        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(supports_gpu):
            llama_info["gpu_offload"] = bool(supports_gpu())
    except Exception as exc:
        llama_info["error"] = str(exc)

    # RAG diagnostics (legacy probes + active runtime requirements)
    try:
        import sentence_transformers  # type: ignore
        rag_info["sentence_transformers_ok"] = True
        rag_info["sentence_transformers_version"] = str(
            getattr(sentence_transformers, "__version__", "")
        )
    except Exception as exc:
        rag_info["error"] = str(exc)

    try:
        import chromadb  # type: ignore
        rag_info["chromadb_ok"] = True
        rag_info["chromadb_version"] = str(getattr(chromadb, "__version__", ""))
    except Exception as exc:
        if rag_info["error"]:
            rag_info["error"] += f"; chromadb: {exc}"
        else:
            rag_info["error"] = str(exc)

    # rag_engine uses llama.cpp GGUF embeddings + chromadb. Keep the
    # sentence-transformers/torch probes for informational diagnostics only.
    rag_info["ready"] = bool(
        rag_info["chromadb_ok"] and llama_info["import_ok"]
    )

    llama_gpu = bool(llama_info["gpu_offload"])
    llm_backends = {
        "vulkan": llama_gpu,
        "cpu": bool(llama_info["import_ok"]),
    }

    rag_backends = {
        "gpu": bool(rag_info["ready"] and llama_gpu),
        "cpu": bool(rag_info["ready"]),
    }

    return {
        "torch": torch_info,
        "llama_cpp": llama_info,
        "rag": rag_info,
        "llm_backends": llm_backends,
        "rag_backends": rag_backends,
        "installed_backends": llm_backends,
    }


def get_backend_diagnostics_subprocess(backend: str | None = None) -> dict[str, Any]:
    """Run :func:`get_backend_diagnostics` in a **clean child process**.

    The UI process imports ``llama_cpp`` early for lightweight package
    checks, which permanently prevents ``ggml-vulkan.dll`` from being
    loaded later (the ``ensure_ggml_backends`` guard skips loading once
    ``llama_cpp`` is in ``sys.modules``).  That makes
    ``llama_supports_gpu_offload()`` return ``False`` for the rest of
    the process lifetime even when Vulkan is fully functional.

    This helper spawns a short-lived subprocess that:

    1. Sets ``LLAMA_CPP_LIB_PATH`` for the requested *backend* (if given).
    2. Calls ``ensure_ggml_backends()`` so all backend DLLs are loaded.
    3. Runs ``get_backend_diagnostics()`` and returns the JSON result.

    The result is identical in schema to :func:`get_backend_diagnostics`.
    """
    import json
    import subprocess
    import sys

    if getattr(sys, "frozen", False):
        ensure_ggml_backends()
        diag = get_backend_diagnostics()
        llama_diag = diag.get("llama_cpp", {})
        if llama_diag.get("import_ok") and not llama_diag.get("gpu_offload"):
            # The UI process may have imported llama_cpp before backends
            # were loaded, making llama_supports_gpu_offload() return
            # False even though vendor/ has ggml-vulkan.dll.  Fall back
            # to a file-presence check so the wizard doesn't reject a
            # working Vulkan install.
            from core.backend_downloader import get_vendor_dir
            vk_dll = get_vendor_dir() / (
                "ggml-vulkan.dll" if sys.platform == "win32" else "ggml-vulkan.so"
            )
            if vk_dll.is_file():
                log.info(
                    "Frozen diagnostics: gpu_offload=False but %s exists; "
                    "promoting to gpu_offload=True", vk_dll.name,
                )
                llama_diag["gpu_offload"] = True
                diag["llm_backends"]["vulkan"] = True
        return diag

    project_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)

    pypath = env.get("PYTHONPATH", "")
    if project_root not in pypath.split(os.pathsep):
        env["PYTHONPATH"] = project_root + os.pathsep + pypath

    if backend:
        from core.backend_downloader import get_backend_lib_path

        lib_path = get_backend_lib_path(backend)
        if lib_path and lib_path.is_file():
            lib_dir = str(lib_path.parent)
            env["LLAMA_CPP_LIB_PATH"] = lib_dir
            cur_path = env.get("PATH", "")
            if lib_dir not in cur_path.split(os.pathsep):
                env["PATH"] = lib_dir + os.pathsep + cur_path

    script = (
        "import json, sys; "
        "from core.hardware_probe import ensure_ggml_backends, get_backend_diagnostics; "
        "ensure_ggml_backends(); "
        "json.dump(get_backend_diagnostics(), sys.stdout)"
    )

    _empty_llama: dict[str, Any] = {
        "import_ok": False,
        "gpu_offload": False,
        "version": "",
        "error": "",
        "distribution": "",
        "expected_fork": "jamepeng",
        "fork_matches": None,
    }

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            cwd=project_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
        _empty_llama["error"] = (
            result.stderr.strip() or "subprocess exited with non-zero status"
        )
    except Exception as exc:
        _empty_llama["error"] = str(exc)

    return {
        "torch": {"import_ok": False, "version": "", "error": ""},
        "llama_cpp": _empty_llama,
        "rag": {"ready": False, "error": ""},
        "llm_backends": {"vulkan": False, "cpu": False},
        "rag_backends": {"gpu": False, "cpu": False},
        "installed_backends": {"vulkan": False, "cpu": False},
    }


def explain_backend_mismatch(backend: AiBackend, diagnostics: dict[str, Any] | None = None) -> str:
    """Explain why a recommended LLM backend is not currently usable."""
    diags = diagnostics or get_backend_diagnostics()
    llama_info = diags.get("llama_cpp", {})

    if not llama_info.get("import_ok"):
        return (
            "llama-cpp-python is not installed — "
            "run  python scripts/setup_environment.py  to install the JamePeng wheel"
        )

    if backend == AiBackend.VULKAN:
        if not llama_info.get("gpu_offload"):
            fork_ok = llama_info.get("fork_matches")
            if fork_ok is False:
                return (
                    "llama.cpp build lacks GPU offload support (wrong fork — "
                    "run  python scripts/setup_environment.py  to install JamePeng)"
                )
            return "llama.cpp build lacks GPU offload support"

    return ""


def get_vulkan_repair_steps(for_frozen: bool | None = None) -> list[str]:
    """Return ordered Vulkan remediation steps for UI/log messages."""
    frozen = bool(getattr(sys, "frozen", False)) if for_frozen is None else bool(for_frozen)
    common_steps = [
        "Update your GPU driver to the latest stable version.",
        "Verify your GPU and driver support Vulkan 1.2+.",
    ]
    if frozen:
        return [
            *common_steps,
            (
                "Open Settings > CHANGE BACKEND, switch to CPU and back to Vulkan, "
                "then retry. If it still fails, reinstall Miniloader."
            ),
        ]
    return [
        *common_steps,
        "Run  python scripts/setup_environment.py  to reinstall wheel + Vulkan DLLs.",
        "Restart Miniloader and retry model load.",
    ]


def format_vulkan_unavailable_message(
    reason: str,
    *,
    for_frozen: bool | None = None,
    prefix: str = "Vulkan backend is unavailable",
) -> str:
    """Build a consistent user-facing Vulkan failure message."""
    lines = [f"{prefix}: {reason}", "", "To fix this:"]
    for idx, step in enumerate(get_vulkan_repair_steps(for_frozen=for_frozen), start=1):
        lines.append(f"  {idx}. {step}")
    return "\n".join(lines)


def verify_backend(backend: AiBackend) -> list[str]:
    """Return human-readable warnings for the detected backend."""
    diagnostics = get_backend_diagnostics()
    reason = explain_backend_mismatch(backend, diagnostics)
    if not reason:
        return []

    if backend == AiBackend.VULKAN:
        return [format_vulkan_unavailable_message(reason)]
    return [
        f"CPU backend is unavailable: {reason}. "
        f"Use Settings > Force CPU Setup to repair it."
    ]


def get_installed_backends() -> dict[str, bool]:
    """Return the installed LLM backend capability booleans."""
    return dict(get_backend_diagnostics().get("llm_backends", {}))


# ── Driver version requirements ─────────────────────────────────────

# Minimum driver versions that support the APIs each backend needs.
# Format: {(vendor, backend): (min_version_tuple, human_label, reason)}
_MIN_DRIVER_REQUIREMENTS: dict[
    tuple[GpuVendor, AiBackend],
    tuple[tuple[int, ...], str, str],
] = {
    (GpuVendor.NVIDIA, AiBackend.VULKAN): (
        (451, 48), "451.48",
        "Vulkan 1.2 (required by llama.cpp) first supported in NVIDIA 451.48",
    ),
    (GpuVendor.AMD, AiBackend.VULKAN): (
        (20, 1, 1), "20.1.1",
        "Vulkan 1.2 support requires AMD Adrenalin 20.1.1+",
    ),
    (GpuVendor.INTEL, AiBackend.VULKAN): (
        (27, 20, 100, 8280), "27.20.100.8280",
        "Vulkan 1.2 support requires Intel driver 27.20.100.8280+",
    ),
}


def _parse_driver_version(version_str: str) -> tuple[int, ...]:
    """Parse a dotted driver version string into a comparable tuple."""
    parts: list[int] = []
    for seg in version_str.replace("-", ".").split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def check_driver_compatibility(
    vendor: GpuVendor,
    backend: AiBackend,
    driver_version: str,
) -> tuple[bool, str]:
    """Check if the current driver meets the minimum for the backend.

    Returns (ok, warning_message).  ok=True means the driver is fine
    or no requirement is known.  warning_message is empty when ok.
    """
    if not driver_version:
        return (True, "")
    key = (vendor, backend)
    req = _MIN_DRIVER_REQUIREMENTS.get(key)
    if req is None:
        return (True, "")
    min_tuple, min_label, reason = req
    current = _parse_driver_version(driver_version)
    if current >= min_tuple:
        return (True, "")
    return (
        False,
        f"Driver {driver_version} may not fully support {backend.value.upper()}. "
        f"Recommended minimum: {min_label}. ({reason}) "
        f"You can still proceed — the backend may work, but you might experience issues.",
    )


# ── Vulkan environment setup ────────────────────────────────────────

def _setup_shader_cache() -> str:
    """Ensure a persistent Vulkan shader cache directory exists.

    Without this, llama.cpp recompiles GLSL compute shaders on every
    launch, causing multi-second freezes.
    """
    cache_dir = Path.home() / ".vulkan_cache" / "llama_cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


# Conservative threshold: when VRAM is unknown and the model file
# is larger than this, assume host-memory spill is safer than OOM.
_UNKNOWN_VRAM_HOST_MEM_THRESHOLD_GB = 3.0


def apply_vulkan_env_vars(
    model_size_gb: float = 0.0,
    *,
    vulkan_device_index: int = 0,
    host_memory_override: str = "auto",
) -> None:
    """Inject Vulkan environment variables into the process.

    **Must** be called before the first ``import llama_cpp`` so the
    C-extension picks up the correct device, memory strategy, and
    shader cache path.

    Parameters
    ----------
    vulkan_device_index:
        Vulkan physical-device ordinal (``GGML_VULKAN_DEVICE``).
    host_memory_override:
        ``"auto"`` applies heuristics, ``"0"``/``"1"`` forces the value.
    """
    hw = get_hardware_snapshot()

    os.environ.setdefault("GGML_VULKAN_DEVICE", str(vulkan_device_index))

    # Host-memory spill decision.
    # Set GGML_VK_PREFER_HOST_MEMORY only when value should be "1".
    # Leaving it unset is faster than explicitly setting "0" on Vulkan.
    vram_gb = hw.vram_total_mb / 1024.0
    if host_memory_override == "1":
        os.environ["GGML_VK_PREFER_HOST_MEMORY"] = "1"
    elif host_memory_override == "0":
        os.environ.pop("GGML_VK_PREFER_HOST_MEMORY", None)
    elif model_size_gb > 0 and vram_gb > 0 and model_size_gb > (vram_gb - 1.0):
        os.environ["GGML_VK_PREFER_HOST_MEMORY"] = "1"
        log.info(
            "Model (~%.1f GB) exceeds VRAM (~%.1f GB) — enabling host memory spill",
            model_size_gb, vram_gb,
        )
    elif (
        vram_gb <= 0
        and model_size_gb > _UNKNOWN_VRAM_HOST_MEM_THRESHOLD_GB
    ):
        os.environ.setdefault("GGML_VK_PREFER_HOST_MEMORY", "1")
        log.warning(
            "VRAM UNKNOWN and model (~%.1f GB) exceeds %.0f GB threshold "
            "— enabling host memory spill (GGML_VK_PREFER_HOST_MEMORY=1). "
            "THIS SEVERELY IMPACTS PERFORMANCE. If your GPU has enough VRAM, "
            "set vulkan_host_memory='0' in basic_brain params to disable.",
            model_size_gb, _UNKNOWN_VRAM_HOST_MEM_THRESHOLD_GB,
        )

    os.environ.setdefault("GGML_VULKAN_SHADER_CACHE", _setup_shader_cache())

    # RADV (Mesa AMD Vulkan driver) hint: avoid GTT memory spills that
    # cause severe throughput drops on RDNA2/RDNA3 cards.
    if hw.gpu_vendor and "amd" in hw.gpu_vendor.lower():
        existing = os.environ.get("RADV_PERFTEST", "")
        if "nogttspill" not in existing:
            new_val = f"{existing},nogttspill" if existing else "nogttspill"
            os.environ["RADV_PERFTEST"] = new_val
            log.info("Set RADV_PERFTEST=%s (AMD GPU detected)", new_val)

    log.info(
        "Vulkan env: DEVICE=%s HOST_MEM=%s SHADER_CACHE=%s RADV_PERFTEST=%s",
        os.environ.get("GGML_VULKAN_DEVICE"),
        os.environ.get("GGML_VK_PREFER_HOST_MEMORY"),
        os.environ.get("GGML_VULKAN_SHADER_CACHE"),
        os.environ.get("RADV_PERFTEST", "(not set)"),
    )
