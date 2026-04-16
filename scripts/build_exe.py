#!/usr/bin/env python3
"""
build_exe.py - Build Miniloader Windows executable (onedir, console).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = REPO_ROOT / "miniloader.spec"
DIST_DIR = REPO_ROOT / "dist" / "miniloader"
INTERNAL_DIR = DIST_DIR / "_internal"
VENDOR_DIR = DIST_DIR / "vendor"
REPO_VENDOR_DIR = REPO_ROOT / "vendor"
BUILD_DIR = REPO_ROOT / "build"

# Binary names that live directly in vendor/ (not inside a subdirectory).
_LIVEKIT_BIN_WINDOWS = "livekit-server.exe"
_LIVEKIT_BIN_LINUX = "livekit-server"
_PLAYWRIGHT_SUBDIR = "playwright"

sys.path.insert(0, str(REPO_ROOT))


def _banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    display = " ".join(cmd)
    print(f"  $ {display}")
    result = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {display}")


def _require_python_311() -> None:
    if (sys.version_info.major, sys.version_info.minor) != (3, 11):
        raise RuntimeError(
            f"Python 3.11 is required for Windows builds, got "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )


def _require_venv() -> None:
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        raise RuntimeError("No active virtual environment detected. Activate .venv first.")


def _ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    _banner("Installing PyInstaller")
    _run([sys.executable, "-m", "pip", "install", "pyinstaller"])


def _require_jamepeng_and_vulkan() -> None:
    from core.llama_runtime import get_llama_package_metadata, is_jamepeng_distribution

    _meta = get_llama_package_metadata()
    if not _meta.get("found"):
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: python scripts/setup_environment.py"
        )
    fork = is_jamepeng_distribution(_meta)
    if fork is False:
        raise RuntimeError(
            f"Installed llama-cpp-python ({_meta.get('version', '?')}) is not "
            "the JamePeng fork. Run: python scripts/setup_environment.py"
        )

    required = [
        REPO_VENDOR_DIR / "ggml-vulkan.dll",
        REPO_VENDOR_DIR / "llama.dll",
    ]
    has_cpu = (
        (REPO_VENDOR_DIR / "ggml-cpu.dll").exists()
        or any(REPO_VENDOR_DIR.glob("ggml-cpu*.dll"))
    )
    missing = [str(p) for p in required if not p.exists()]
    if not has_cpu:
        missing.append(str(REPO_VENDOR_DIR / "ggml-cpu.dll or ggml-cpu-*.dll"))
    if missing:
        raise RuntimeError(
            "Missing required vendor DLLs:\n  - " + "\n  - ".join(missing) + "\n"
            "Run: python scripts/setup_environment.py"
        )


def _validate_dist_output(*, demo: bool = False) -> None:
    import platform as _plat

    required = [
        DIST_DIR / "miniloader.exe",
        INTERNAL_DIR / "modules" / "gpt_terminal" / "app" / "dist" / "index.html",
        INTERNAL_DIR / "templates" / "blank_rack.json",
        INTERNAL_DIR / "resources" / "portal" / "qwebchannel.js",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if not demo:
        required.extend(
            [
                VENDOR_DIR / "ggml-vulkan.dll",
                VENDOR_DIR / "llama.dll",
            ]
        )
        missing.extend(str(p) for p in required if not p.exists() and str(p) not in missing)
        has_cpu_dist = (
            (VENDOR_DIR / "ggml-cpu.dll").exists()
            or any(VENDOR_DIR.glob("ggml-cpu*.dll"))
        )
        if not has_cpu_dist:
            missing.append(str(VENDOR_DIR / "ggml-cpu.dll or ggml-cpu-*.dll"))

    # Build-regression sentinels: fail early if critical packaged dependencies disappear.
    chroma_migrations_dir = INTERNAL_DIR / "chromadb" / "migrations"
    if not chroma_migrations_dir.is_dir() or not any(chroma_migrations_dir.rglob("*.sql")):
        missing.append(f"{chroma_migrations_dir} (expected at least one migration .sql)")

    onnx_markers = ("onnxruntime.dll", "onnxruntime_providers_shared.dll")
    onnx_found = any(INTERNAL_DIR.rglob(name) for name in onnx_markers)
    if not onnx_found:
        missing.append(
            f"{INTERNAL_DIR} (**/{onnx_markers[0]} or **/{onnx_markers[1]} expected)"
        )

    # livekit_voice module: report if binary is absent (warning, not hard failure -
    # the module supports on-demand download via its SETUP flow).
    lk_bin = _LIVEKIT_BIN_WINDOWS if _plat.system() == "Windows" else _LIVEKIT_BIN_LINUX
    if not (VENDOR_DIR / lk_bin).exists():
        print(
            f"  [warn] {lk_bin} not in dist/vendor - livekit_voice will require "
            "setup on first launch. Run: python scripts/setup_environment.py"
        )

    # web_browser module: report if Playwright Chromium is absent (warning only).
    playwright_dir = VENDOR_DIR / _PLAYWRIGHT_SUBDIR
    if not playwright_dir.exists() or not any(playwright_dir.rglob("chrome.exe")):
        print(
            "  [warn] vendor/playwright Chromium not found in dist - web_browser will "
            "require setup on first launch. Run: web_browser SETUP or "
            "PLAYWRIGHT_BROWSERS_PATH=vendor/playwright playwright install chromium"
        )

    # Confirm vr_terminal is NOT in the bundle.
    vr_path = INTERNAL_DIR / "modules" / "vr_terminal"
    if vr_path.exists():
        missing.append(f"{vr_path} (vr_terminal should NOT be bundled - check spec excludes)")

    if missing:
        raise RuntimeError(
            "Build succeeded but required output files are missing:\n  - "
            + "\n  - ".join(missing)
        )


def _sync_runtime_vendor_dlls() -> None:
    """Mirror bundled vendor DLLs to dist/vendor for frozen runtime lookup."""
    internal_vendor = INTERNAL_DIR / "vendor"
    if not internal_vendor.is_dir():
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for dll in sorted(internal_vendor.glob("*.dll")):
        shutil.copy2(dll, VENDOR_DIR / dll.name)


def _sync_vendor_playwright() -> None:
    """Copy the Playwright Chromium tree from repo/vendor/playwright -> dist/vendor/playwright.

    Playwright's browser binaries are large (200+ MB) and are NOT bundled through
    PyInstaller's _internal archive.  Instead they are synced directly alongside the
    executable so that ``PLAYWRIGHT_BROWSERS_PATH=<dist>/vendor/playwright`` resolves
    correctly at runtime (web_browser module default).
    """
    src = REPO_VENDOR_DIR / _PLAYWRIGHT_SUBDIR
    if not src.exists():
        print(f"  [warn] vendor/playwright not found at {src} - web_browser will require setup on first use")
        return
    dst = VENDOR_DIR / _PLAYWRIGHT_SUBDIR
    _banner(f"Syncing Playwright Chromium tree -> dist/vendor/{_PLAYWRIGHT_SUBDIR}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(str(src), str(dst))
    print(f"  Copied vendor/playwright -> {dst}")


def _copy_livekit_binary() -> None:
    """Copy the LiveKit server binary from repo/vendor -> dist/vendor.

    The livekit_voice module resolves the server binary from vendor/ first (see
    livekit_voice/logic.py _resolve_livekit_binary).  This step ensures the binary
    lands next to the executable at build time.
    """
    import platform as _plat
    bin_name = _LIVEKIT_BIN_WINDOWS if _plat.system() == "Windows" else _LIVEKIT_BIN_LINUX
    src = REPO_VENDOR_DIR / bin_name
    if not src.exists():
        print(f"  [warn] {bin_name} not found in repo vendor/ - livekit_voice will require setup on first use")
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    dst = VENDOR_DIR / bin_name
    shutil.copy2(str(src), str(dst))
    print(f"  Copied {bin_name} -> {dst}")


def _ensure_vendor_dir() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    readme = VENDOR_DIR / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Miniloader vendor dependencies",
                "==============================",
                "",
                "All backend items for Miniloader live here so both dev and frozen",
                "builds use the same resolved paths.",
                "",
                "LLM backend DLLs (Windows):",
                "  ggml.dll, ggml-base.dll, ggml-cpu*.dll, ggml-vulkan.dll,",
                "  llama.dll, mtmd.dll, libomp140.x86_64.dll",
                "",
                "LiveKit RTC server (livekit_voice module):",
                "  livekit-server.exe   (Windows)",
                "  livekit-server       (Linux)",
                "  Populated by: python scripts/setup_environment.py",
                "",
                "Playwright Chromium (web_browser module):",
                "  playwright/          (subtree managed by 'playwright install chromium')",
                "  Populated by: web_browser module SETUP button, or manually via",
                "    PLAYWRIGHT_BROWSERS_PATH=vendor/playwright playwright install chromium",
                "",
                "RAG embedding model (rag_engine module):",
                "  models/<model>.gguf  - place GGUF embedding models here for auto-discovery",
                "  Recommended: all-MiniLM-L6-v2-Q4_K_M.gguf (~21 MB)",
                "",
                "Other:",
                "  ngrok.exe            (ngrok_tunnel module)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _smoke_test_launch(timeout_s: float = 8.0) -> None:
    exe = DIST_DIR / "miniloader.exe"
    if not exe.exists():
        raise RuntimeError(f"Cannot smoke-test: missing {exe}")

    _banner("Smoke test launch")
    print(f"  Launching {exe} for ~{timeout_s:.0f}s...")
    proc = subprocess.Popen(
        [str(exe)],
        cwd=str(DIST_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )
    time.sleep(timeout_s)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    code = proc.returncode
    if code not in (None, 0, 1, -15):
        raise RuntimeError(f"Smoke test process exited unexpectedly with code {code}")
    print("  Smoke test completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Miniloader Windows .exe (onedir).")
    parser.add_argument("--clean", action="store_true", help="Delete build/ and dist/ before building.")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip post-build launch test.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build demo bundle without requiring llama_cpp DLLs during build.",
    )
    args = parser.parse_args()

    _banner("Miniloader Windows Build")
    _require_python_311()
    _require_venv()
    if not SPEC_FILE.exists():
        raise RuntimeError(f"Missing PyInstaller spec file: {SPEC_FILE}")

    if not args.demo:
        _require_jamepeng_and_vulkan()
    else:
        print("  Demo mode: skipping JamePeng llama_cpp and Vulkan DLL preflight checks.")
    _ensure_pyinstaller()

    if args.clean:
        _banner("Cleaning previous artifacts")
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "dist", ignore_errors=True)

    _banner("Running PyInstaller")
    pyinstaller_env = {**os.environ}
    if args.demo:
        pyinstaller_env["MINILOADER_DEMO_BUILD"] = "1"
    _run(
        [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"],
        cwd=REPO_ROOT,
        env=pyinstaller_env,
    )

    _banner("Creating vendor directory")
    _ensure_vendor_dir()
    _sync_runtime_vendor_dlls()
    _copy_livekit_binary()
    _sync_vendor_playwright()

    _banner("Validating output")
    _validate_dist_output(demo=args.demo)
    print(f"  Build output: {DIST_DIR}")

    if not args.skip_smoke_test:
        _smoke_test_launch()

    _banner("Build complete")


if __name__ == "__main__":
    main()
