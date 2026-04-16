from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    runtime_dir = root / "bundled_mcp" / "node_runtime"
    pkg_json = runtime_dir / "package.json"
    if not pkg_json.exists():
        print(f"[bootstrap] missing package.json: {pkg_json}")
        return 1

    npm_bin = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm_bin:
        print("[bootstrap] npm is not installed or not on PATH.")
        print("[bootstrap] install Node.js, then re-run this bootstrap script.")
        return 1

    if "--yes" not in sys.argv and "-y" not in sys.argv:
        print("[bootstrap] This script will download npm packages from the internet.")
        print(f"[bootstrap] Target: {runtime_dir}")
        try:
            answer = input("[bootstrap] Continue? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("[bootstrap] Aborted.")
            return 1

    print(f"[bootstrap] installing bundled MCP packages in {runtime_dir}")
    result = subprocess.run(
        [npm_bin, "install", "--no-audit", "--no-fund"],
        cwd=str(runtime_dir),
        check=False,
    )
    if result.returncode != 0:
        print("[bootstrap] npm install failed.")
        return result.returncode

    print("[bootstrap] done.")
    print("[bootstrap] cartridges using bundle://... now prefer local bundled providers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
