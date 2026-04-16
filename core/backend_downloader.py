"""
backend_downloader.py — Backend path resolution.
==================================================
Resolves the llama.cpp shared library at runtime.  The library
may come from a bundled ``backends/`` directory or from the
JamePeng pre-built wheel installed via pip.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_BACKENDS_DIR_NAME = "backends"

_LIB_NAMES: dict[str, str] = {
    "Windows": "llama.dll",
    "Linux":   "libllama.so",
}


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return getattr(sys, "frozen", False)


def get_app_dir() -> Path:
    """Return the application root directory.

    - PyInstaller onefile: ``sys._MEIPASS`` (temp extraction dir)
    - PyInstaller onedir:  ``Path(sys.executable).parent``
    - Running from source: repository root (parent of ``core/``)
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass))
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def get_bundled_backends_dir() -> Path:
    """Return the path to backends bundled inside the app distribution."""
    return get_app_dir() / _BACKENDS_DIR_NAME


def get_vendor_dir() -> Path:
    """Return the external dependency folder location."""
    if is_frozen():
        return Path(sys.executable).parent / "vendor"
    return Path(__file__).resolve().parent.parent / "vendor"


def ensure_vendor_dll_search_path() -> None:
    """Add vendor/ to native library lookup for all runtimes."""
    vendor_dir = get_vendor_dir()
    if not vendor_dir.is_dir():
        return

    vendor_str = str(vendor_dir)
    cur_path = os.environ.get("PATH", "")
    if vendor_str not in cur_path.split(os.pathsep):
        os.environ["PATH"] = vendor_str + os.pathsep + cur_path

    if os.name == "nt":
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                add_dll_directory(vendor_str)
            except OSError:
                pass

    lib_name = _LIB_NAMES.get(platform.system(), "")
    if lib_name and (vendor_dir / lib_name).is_file():
        os.environ["LLAMA_CPP_LIB_PATH"] = vendor_str


def _find_lib_in_dir(root: Path, lib_name: str) -> Path | None:
    """Search a backend directory tree for the shared library."""
    if not root.is_dir():
        return None
    candidate = root / lib_name
    if candidate.is_file():
        return candidate
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        candidate = child / lib_name
        if candidate.is_file():
            return candidate
        for sub in child.iterdir():
            if sub.is_dir():
                candidate = sub / lib_name
                if candidate.is_file():
                    return candidate
    return None


def get_backend_lib_path(backend: str) -> Path | None:
    """Return the path to the backend's shared library, or None.

    Prefers ``vendor/``, then legacy ``{app_dir}/backends/{backend}/``.
    """
    lib_name = _LIB_NAMES.get(platform.system(), "")
    if not lib_name:
        return None
    vendor_candidate = get_vendor_dir() / lib_name
    if vendor_candidate.is_file():
        return vendor_candidate
    return _find_lib_in_dir(get_bundled_backends_dir() / backend, lib_name)


def _frozen_llama_bin_dir() -> Path | None:
    """Return the llama_cpp/bin/ dir bundled inside the frozen app, or None."""
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    bin_dir = Path(str(meipass)) / "llama_cpp" / "bin"
    return bin_dir if bin_dir.is_dir() else None


def _wheel_provides_vulkan() -> bool:
    """Check if llama_cpp is importable with Vulkan backend DLLs present.

    Uses per-backend loading so a Vulkan init failure doesn't crash the
    process — it just means GPU offload won't be available.
    """
    try:
        import ctypes
        import llama_cpp._ggml as _ggml_mod
        bin_dir = Path(_ggml_mod.__file__).resolve().parent / "bin"
        if bin_dir.is_dir():
            _load_one = _ggml_mod.libggml.ggml_backend_load
            _load_one.argtypes = [ctypes.c_char_p]
            _load_one.restype = ctypes.c_void_p
            for dll in sorted(bin_dir.iterdir()):
                if dll.suffix in (".dll", ".so") and dll.name.startswith("ggml-"):
                    try:
                        _load_one(str(dll).encode())
                    except OSError:
                        pass
        import llama_cpp  # type: ignore
        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(supports_gpu):
            return bool(supports_gpu())
        return True
    except Exception:
        return False


def is_backend_ready(backend: str) -> bool:
    """True if the backend is available, either bundled or via the wheel."""
    if get_backend_lib_path(backend) is not None:
        return True

    vendor_dir = get_vendor_dir()
    if vendor_dir.is_dir():
        if backend == "vulkan":
            if (
                (vendor_dir / "ggml-vulkan.dll").exists()
                or (vendor_dir / "ggml-vulkan.so").exists()
                or (vendor_dir / "ggml-vulkan.dylib").exists()
            ):
                return True
        if backend == "cpu":
            if (
                (vendor_dir / "ggml-cpu.dll").exists()
                or any(vendor_dir.glob("ggml-cpu*.dll"))
                or any(vendor_dir.glob("ggml-cpu*.so"))
                or any(vendor_dir.glob("ggml-cpu*.dylib"))
            ):
                return True

    # In a frozen (PyInstaller) app, DLLs are bundled in _MEIPASS/llama_cpp/bin/.
    # Check for the specific backend DLL directly rather than trying to import
    # and run llama_cpp (which would fail because backends aren't loaded yet).
    if is_frozen():
        bin_dir = _frozen_llama_bin_dir()
        if bin_dir is not None:
            if backend == "vulkan":
                return (bin_dir / "ggml-vulkan.dll").exists() or (
                    bin_dir / "ggml-vulkan.so"
                ).exists()
            if backend == "cpu":
                return (bin_dir / "ggml-cpu.dll").exists() or any(
                    bin_dir.glob("ggml-cpu*.dll")
                ) or any(bin_dir.glob("ggml-cpu*.so"))
        return False

    if backend == "vulkan":
        return _wheel_provides_vulkan()
    return False
