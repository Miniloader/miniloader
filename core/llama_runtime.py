"""
llama_runtime.py — JamePeng llama-cpp-python fork metadata.
=============================================================
Single source of truth for version pins, URLs, and runtime
identification of the installed llama_cpp distribution.

Both ``scripts/setup_environment.py`` (install-time) and
``core/hardware_probe.py`` / ``modules/basic_brain/logic.py``
(runtime) import from here so pins never drift.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# ── Pin constants (shared with scripts/setup_environment.py) ─────────

JAMEPENG_REPO = "JamePeng/llama-cpp-python"
JAMEPENG_RELEASES_BASE_URL = (
    "https://github.com/JamePeng/llama-cpp-python/releases/download"
)
JAMEPENG_PINNED_VERSION = "0.3.35"
JAMEPENG_TAG_DATE = "20260406"
JAMEPENG_TAG_PREFIX = "cu130-Basic"

LLAMACPP_RELEASE_TAG = "b8640"
LLAMACPP_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download"
)

JAMEPENG_EXTRA_DEPS = ["diskcache", "Pillow"]

# Pattern for the local-version segment JamePeng wheels inject
# (e.g. "0.3.33+cu130.basic").
_JAMEPENG_LOCAL_RE = re.compile(r"\+cu\d+\.\w+", re.IGNORECASE)


# ── Runtime metadata helpers ─────────────────────────────────────────

def get_llama_package_metadata() -> dict[str, Any]:
    """Return installed ``llama_cpp_python`` distribution metadata.

    Keys: ``version``, ``summary``, ``home_page``, ``project_urls``,
    ``found`` (bool).  Missing fields default to empty strings.
    """
    result: dict[str, Any] = {
        "found": False,
        "version": "",
        "summary": "",
        "home_page": "",
        "project_urls": [],
    }
    try:
        from importlib.metadata import metadata as _pkg_metadata

        meta = _pkg_metadata("llama_cpp_python")
        result["found"] = True
        result["version"] = meta.get("Version", "")
        result["summary"] = meta.get("Summary", "")
        result["home_page"] = meta.get("Home-page", "")
        urls = meta.get_all("Project-URL") or []
        result["project_urls"] = [str(u) for u in urls]
    except Exception:
        pass
    return result


def is_jamepeng_distribution(meta: dict[str, Any] | None = None) -> bool | None:
    """Heuristic: is the installed llama_cpp_python the JamePeng fork?

    Returns ``True`` (positive match), ``False`` (definitely not), or
    ``None`` (cannot determine — metadata missing or ambiguous).
    """
    if meta is None:
        meta = get_llama_package_metadata()
    if not meta.get("found"):
        return None

    version = str(meta.get("version", ""))
    if _JAMEPENG_LOCAL_RE.search(version):
        return True

    all_urls = " ".join(
        [str(meta.get("home_page", ""))] + list(meta.get("project_urls", []))
    ).lower()
    if "jamepeng" in all_urls:
        return True
    if "abetlen" in all_urls:
        return False

    return None
