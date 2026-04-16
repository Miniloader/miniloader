"""
crash_reporter.py - Local crash sentinel for desktop-style reporting.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CRASH_LOG_PATH = Path.home() / ".miniloader" / "crash.log"
_HOOK_INSTALLED = False
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)


def _app_version() -> str:
    value = os.environ.get("MINILOADER_VERSION", "").strip()
    if value:
        return value
    return "unknown"


def _build_crash_payload(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        "exception_type": getattr(exc_type, "__name__", str(exc_type)),
        "exception_message": str(exc_value),
        "python_version": sys.version,
        "os_info": platform.platform(),
        "app_version": _app_version(),
        "frozen": bool(getattr(sys, "frozen", False)),
    }


def _write_crash_log(payload: dict[str, Any]) -> None:
    CRASH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CRASH_LOG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _handle_uncaught_exception(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)
        return
    try:
        payload = _build_crash_payload(exc_type, exc_value, exc_tb)
        _write_crash_log(payload)
    except Exception:
        pass
    _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_tb)


def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
    try:
        if not issubclass(args.exc_type, KeyboardInterrupt):
            payload = _build_crash_payload(args.exc_type, args.exc_value, args.exc_traceback)
            _write_crash_log(payload)
    except Exception:
        pass
    finally:
        if _ORIGINAL_THREADING_EXCEPTHOOK is not None:
            _ORIGINAL_THREADING_EXCEPTHOOK(args)


def install_crash_hooks() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    sys.excepthook = _handle_uncaught_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = _handle_thread_exception
    _HOOK_INSTALLED = True


def check_for_crash_log() -> dict[str, Any] | None:
    if not CRASH_LOG_PATH.exists():
        return None
    try:
        data = json.loads(CRASH_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "timestamp": "",
            "traceback": CRASH_LOG_PATH.read_text(encoding="utf-8", errors="replace"),
            "exception_type": "Unknown",
            "exception_message": "Unreadable crash log format",
            "python_version": "",
            "os_info": "",
            "app_version": "",
            "frozen": bool(getattr(sys, "frozen", False)),
        }
    if not isinstance(data, dict):
        return None
    return data


def clear_crash_log() -> None:
    try:
        CRASH_LOG_PATH.unlink(missing_ok=True)
    except Exception:
        pass
