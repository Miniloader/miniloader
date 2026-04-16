#!/usr/bin/env python3
"""
create_venv.py — Create a standardised Python 3.11 virtual environment.
========================================================================
Cross-platform helper that locates Python 3.11 on the system, creates a
.venv in the repo root, upgrades pip, and prints activation instructions.

Usage
-----
    python scripts/create_venv.py          # create .venv
    python scripts/create_venv.py --force  # recreate from scratch
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"
TARGET_MINOR = 11


def _find_python311() -> str | None:
    """Return a command string that launches Python 3.11, or None."""
    system = platform.system()

    if system == "Windows":
        candidates = [
            ["py", f"-3.{TARGET_MINOR}"],
            [f"python3.{TARGET_MINOR}"],
            ["python"],
        ]
    else:
        candidates = [
            [f"python3.{TARGET_MINOR}"],
            ["python3"],
            ["python"],
        ]

    for cmd in candidates:
        try:
            result = subprocess.run(
                [*cmd, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            version_str = result.stdout.strip() + result.stderr.strip()
            if f"3.{TARGET_MINOR}" in version_str:
                return cmd if len(cmd) > 1 else cmd[0]
        except Exception:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Python 3.11 venv for Miniloader.")
    parser.add_argument("--force", action="store_true", help="Delete existing .venv and recreate.")
    args = parser.parse_args()

    if VENV_DIR.exists() and not args.force:
        print(f"  .venv already exists at {VENV_DIR}")
        print("  Use --force to recreate it.")
        sys.exit(0)

    print("  Looking for Python 3.11...")
    py_cmd = _find_python311()
    if py_cmd is None:
        print(f"\n  ERROR: Python 3.{TARGET_MINOR} not found on this system.")
        print("  Install it from https://www.python.org/downloads/ and try again.")
        sys.exit(1)

    cmd_display = py_cmd if isinstance(py_cmd, str) else " ".join(py_cmd)
    print(f"  Found: {cmd_display}")

    if VENV_DIR.exists():
        print(f"  Removing existing .venv...")
        shutil.rmtree(VENV_DIR)

    print(f"  Creating .venv at {VENV_DIR}...")
    base = [py_cmd] if isinstance(py_cmd, str) else list(py_cmd)
    subprocess.check_call([*base, "-m", "venv", str(VENV_DIR)])

    pip_exe = (
        str(VENV_DIR / "Scripts" / "pip.exe")
        if platform.system() == "Windows"
        else str(VENV_DIR / "bin" / "pip")
    )
    print("  Upgrading pip...")
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)

    print("\n  Done. Activate the venv with:\n")
    if platform.system() == "Windows":
        print(f"    .venv\\Scripts\\activate        (cmd)")
        print(f"    .venv\\Scripts\\Activate.ps1    (PowerShell)")
    else:
        print(f"    source .venv/bin/activate")

    print(f"\n  Then install dependencies:")
    print(f"    pip install -r requirements.txt")
    print(f"    python scripts/setup_environment.py")


if __name__ == "__main__":
    main()
