#!/usr/bin/env python3
"""
setup_environment.py — Developer environment setup
====================================================
Installs base dependencies and the JamePeng fork of llama-cpp-python
(pre-built Vulkan wheel — no C compiler or Vulkan SDK needed).

Usage
-----
    python scripts/setup_environment.py              # install everything
    python scripts/setup_environment.py --dry-run    # show what would happen
    python scripts/setup_environment.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.hardware_probe import (
    AiBackend,
    ensure_ggml_backends,
    get_hardware_snapshot,
    verify_backend,
)
from core.llama_runtime import (
    JAMEPENG_EXTRA_DEPS as _JAMEPENG_EXTRA_DEPS,
    JAMEPENG_PINNED_VERSION as _JAMEPENG_VERSION,
    JAMEPENG_RELEASES_BASE_URL as _JAMEPENG_BASE_URL,
    JAMEPENG_TAG_DATE as _JAMEPENG_TAG_DATE,
    JAMEPENG_TAG_PREFIX as _JAMEPENG_TAG_PREFIX,
    LLAMACPP_RELEASE_TAG as _LLAMACPP_RELEASE_TAG,
    LLAMACPP_RELEASE_URL as _LLAMACPP_RELEASE_URL,
)

_LIVEKIT_RELEASE_API = "https://api.github.com/repos/livekit/livekit/releases/latest"

def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _run(cmd: list[str], env: dict[str, str] | None = None, dry: bool = False) -> bool:
    full_env = {**os.environ, **(env or {})}
    display = " ".join(cmd)
    if env:
        display = " ".join(f"{k}={v}" for k, v in env.items()) + " " + display
    if dry:
        print(f"  [dry-run] {display}")
        return True
    print(f"  $ {display}")
    result = subprocess.run(cmd, env=full_env)
    if result.returncode != 0:
        print(f"  [ERROR] command exited with code {result.returncode}")
        return False
    return True


def _pip(*args: str, env: dict[str, str] | None = None, dry: bool = False) -> bool:
    return _run([sys.executable, "-m", "pip", *args], env=env, dry=dry)


def _jamepeng_wheel_url() -> str:
    """Build the download URL for the JamePeng llama-cpp-python wheel
    matching the current platform and Python version."""
    import platform as _plat

    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    system = _plat.system()

    if system == "Windows":
        plat_tag = "win_amd64"
        tag_os = "win"
    elif system == "Linux":
        plat_tag = "linux_x86_64"
        tag_os = "linux"
    else:
        raise RuntimeError(f"No pre-built JamePeng wheel for {system}")

    tag = f"v{_JAMEPENG_VERSION}-{_JAMEPENG_TAG_PREFIX}-{tag_os}-{_JAMEPENG_TAG_DATE}"
    filename = (
        f"llama_cpp_python-{_JAMEPENG_VERSION}"
        f"+{_JAMEPENG_TAG_PREFIX.lower().replace('-', '.')}"
        f"-{pyver}-{pyver}-{plat_tag}.whl"
    )
    return f"{_JAMEPENG_BASE_URL}/{tag}/{filename}"


def _vulkan_dll_zip_url() -> str:
    """URL for the pre-built Vulkan binary release matching the pinned
    llama.cpp version.  Windows-only for now."""
    import platform as _plat
    system = _plat.system()
    if system == "Windows":
        name = f"llama-{_LLAMACPP_RELEASE_TAG}-bin-win-vulkan-x64.zip"
    elif system == "Linux":
        name = f"llama-{_LLAMACPP_RELEASE_TAG}-bin-ubuntu-x64-vulkan.zip"
    else:
        raise RuntimeError(f"No pre-built Vulkan DLLs for {system}")
    return f"{_LLAMACPP_RELEASE_URL}/{_LLAMACPP_RELEASE_TAG}/{name}"


def _swap_vulkan_dlls(dry: bool = False) -> None:
    """Populate vendor/ and refresh package core DLLs for Vulkan runtime."""
    import importlib.util
    import platform as _plat
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    if _plat.system() not in ("Windows", "Linux"):
        print("  [skip] Vulkan DLL swap only supported on Windows/Linux")
        return

    # Use find_spec instead of import_module to locate the package without
    # triggering native DLL loads — the CUDA-linked DLLs from the JamePeng
    # wheel will fail to load on systems without CUDA runtime.
    _spec = importlib.util.find_spec("llama_cpp")
    if _spec is None or not _spec.origin:
        print("  [warn] llama_cpp not installed — cannot swap DLLs")
        return
    pkg_dir = Path(_spec.origin).resolve().parent

    pkg_bin = pkg_dir / "bin"
    pkg_lib = pkg_dir / "lib"
    vendor_dir = REPO_ROOT / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)

    url = _vulkan_dll_zip_url()
    print(f"  Vulkan DLLs: {url}")
    if dry:
        print("  [dry-run] Would download and swap DLLs")
        return

    tmp_zip = Path(tempfile.mkdtemp()) / "vulkan.zip"
    extract_dir = tmp_zip.parent / "vulkan_extract"

    print("  Downloading Vulkan DLLs …")
    urllib.request.urlretrieve(url, str(tmp_zip))

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(str(tmp_zip)) as zf:
        zf.extractall(str(extract_dir))

    ext = ".dll" if _plat.system() == "Windows" else ".so"
    # Copy all shared libraries from the Vulkan release into vendor/.
    vulkan_files = [f for f in extract_dir.iterdir() if f.suffix == ext]

    for f in vulkan_files:
        shutil.copy2(str(f), str(vendor_dir / f.name))
    print(f"  Copied {len(vulkan_files)} files -> vendor/")

    # Keep llama_cpp package core runtime DLLs aligned with the Vulkan zip.
    # `_ggml.py` loads ggml from llama_cpp/lib|bin directly and does not
    # consult LLAMA_CPP_LIB_PATH.
    core_runtime_names = {
        f"ggml{ext}",
        f"ggml-base{ext}",
        f"llama{ext}",
        f"mtmd{ext}",
    }
    core_runtime_files = [
        f
        for f in vulkan_files
        if f.name in core_runtime_names or f.name.startswith("libomp")
    ]
    for dst_dir in (pkg_bin, pkg_lib):
        if not dst_dir.exists():
            continue
        for f in core_runtime_files:
            shutil.copy2(str(f), str(dst_dir / f.name))
    print(
        "  Synced core Vulkan runtime files "
        f"({len(core_runtime_files)} names) -> llama_cpp/bin and lib"
    )

    # Keep wheel-provided ggml-cpu.dll for compatibility when the Vulkan zip
    # only includes CPU micro-arch variants (ggml-cpu-*.dll).
    plain_cpu_name = f"ggml-cpu{ext}"
    plain_cpu_dst = vendor_dir / plain_cpu_name
    if not plain_cpu_dst.exists():
        for src_dir in (pkg_bin, pkg_lib):
            src = src_dir / plain_cpu_name
            if src.is_file():
                shutil.copy2(str(src), str(plain_cpu_dst))
                print(f"  Preserved {plain_cpu_name} from {src_dir.name}/ -> vendor/")
                break

    # Clean up temp files
    shutil.rmtree(str(tmp_zip.parent), ignore_errors=True)
    print("  Vulkan DLL swap complete.")


def _download_livekit_server_binary(dry: bool = False) -> None:
    """Download the latest LiveKit server binary into vendor/.

    Keeping the binary in vendor/ ensures that both the dev environment and the
    frozen build (which copies vendor/ alongside the exe) resolve the same path.
    The livekit_voice module searches vendor/ first via _resolve_livekit_binary().
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    elif machine in {"aarch64", "arm64"}:
        arch = "arm64"
    elif machine in {"armv7l", "armv7"}:
        arch = "armv7"
    else:
        print(f"  [skip] unsupported CPU arch for LiveKit binary: {machine}")
        return

    if system == "windows":
        platform_name = "windows"
        archive_ext = "zip"
        binary_name = "livekit-server.exe"
    elif system == "linux":
        platform_name = "linux"
        archive_ext = "tar.gz"
        binary_name = "livekit-server"
    else:
        print(f"  [skip] unsupported OS for LiveKit binary auto-download: {system}")
        return

    target_dir = REPO_ROOT / "vendor"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_binary = target_dir / binary_name

    if target_binary.exists():
        print(f"  [ok] LiveKit binary already present: {target_binary}")
        return

    print(f"  LiveKit release API: {_LIVEKIT_RELEASE_API}")
    if dry:
        print("  [dry-run] Would fetch latest release and extract livekit-server")
        return

    with urllib.request.urlopen(_LIVEKIT_RELEASE_API, timeout=20) as resp:
        release = json.loads(resp.read().decode("utf-8"))
    assets = release.get("assets", []) if isinstance(release, dict) else []
    tag_name = str(release.get("tag_name", "latest")) if isinstance(release, dict) else "latest"
    expected = f"livekit_{tag_name.lstrip('v')}_{platform_name}_{arch}.{archive_ext}"

    asset_url = ""
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("name", "")).strip() == expected:
            asset_url = str(asset.get("browser_download_url", "")).strip()
            break
    if not asset_url:
        print(f"  [warn] no matching LiveKit asset found for {platform_name}/{arch}")
        return

    print(f"  Downloading {expected}")
    tmp_root = Path(tempfile.mkdtemp(prefix="miniloader-livekit-"))
    archive_path = tmp_root / expected
    urllib.request.urlretrieve(asset_url, str(archive_path))

    if archive_ext == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_root)
    else:
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(tmp_root)

    extracted = next((p for p in tmp_root.rglob(binary_name) if p.is_file()), None)
    if extracted is None:
        print("  [warn] could not locate livekit-server binary in downloaded archive")
        shutil.rmtree(tmp_root, ignore_errors=True)
        return

    shutil.copy2(str(extracted), str(target_binary))
    if os.name != "nt":
        os.chmod(target_binary, 0o755)
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"  Installed LiveKit binary -> {target_binary} (vendor/)")


def install_base_requirements(dry: bool = False) -> None:
    """Install requirements.txt, the JamePeng wheel, its extra deps,
    then swap in the Vulkan DLLs from the matching llama.cpp release."""
    req_file = REPO_ROOT / "requirements.txt"
    if not req_file.exists():
        print("  [warn] requirements.txt not found — skipping")
        return
    _banner("Installing base dependencies")
    _pip("install", "-r", str(req_file), dry=dry)

    _banner("Installing JamePeng llama-cpp-python")
    wheel_url = _jamepeng_wheel_url()
    print(f"  Wheel: {wheel_url}")
    _pip("install", wheel_url, "--force-reinstall", "--no-deps", dry=dry)

    _banner("Installing llama-cpp-python extra deps")
    _pip("install", *_JAMEPENG_EXTRA_DEPS, dry=dry)

    _banner("Swapping in Vulkan DLLs (replacing CUDA)")
    _swap_vulkan_dlls(dry=dry)

    _banner("Installing LiveKit server binary")
    _download_livekit_server_binary(dry=dry)


def _check_import(module: str, version_attr: str = "__version__") -> tuple[bool, str]:
    script = (
        f"import {module}; "
        f"print(getattr({module}, '{version_attr}', 'unknown'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return (False, "")
    return (True, result.stdout.strip())


def verify_installation(backend: AiBackend) -> bool:
    _banner("Verifying installation")
    ok = True

    found, ver = _check_import("llama_cpp")
    if found:
        print(f"  llama-cpp-python {ver}")
    else:
        print("  llama-cpp-python — NOT INSTALLED")
        ok = False

    from core.backend_downloader import is_backend_ready
    for b in ("vulkan", "cpu"):
        status = "ready" if is_backend_ready(b) else "NOT FOUND"
        print(f"  Backend DLLs ({b}): {status}")

    ensure_ggml_backends()
    warnings = verify_backend(backend)
    for w in warnings:
        print(f"  [WARN] {w}")
        ok = False

    print("\n  All checks passed." if ok else "\n  Some checks failed — see warnings above.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Miniloader developer environment setup."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    _banner("Miniloader Environment Setup")

    print("\n  Detecting hardware...")
    hw = get_hardware_snapshot()
    print(f"  OS:      {hw.os_info}")
    print(f"  GPU:     {hw.gpu_name or 'Not detected'}")
    print(f"  Vendor:  {hw.gpu_vendor.value}")
    print(f"  Backend: {hw.ai_backend_hint.value}")

    backend = hw.ai_backend_hint

    if args.verify_only:
        success = verify_installation(backend)
        sys.exit(0 if success else 1)

    if not args.skip_base:
        install_base_requirements(dry=args.dry_run)

    if not args.dry_run:
        success = verify_installation(backend)
        sys.exit(0 if success else 1)
    else:
        print("\n  [dry-run] Skipping verification.")


if __name__ == "__main__":
    main()
