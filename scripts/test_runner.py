"""Process-isolated test/benchmark runner for a single model.

This module is the subprocess target for the test & benchmark wizard.
It is deliberately kept **minimal** — no PyQt, no heavy UI imports — so
that ``multiprocessing.Process`` on Windows can import it cleanly without
pulling in the entire application.

The pattern matches ``core/worker.py`` + ``basic_brain/logic.py``:
  1. Fresh subprocess, no prior ``llama_cpp`` state.
  2. ``ensure_ggml_backends()`` with **no mode filter** — loads ALL
     backend DLLs (RPC, Vulkan, CPU), exactly like ``basic_brain``.
  3. ``n_gpu_layers`` enforces the configured backend (``-1`` for
     Vulkan, ``0`` for CPU).  No silent fallback.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def run_functionality(config: dict[str, Any], q: Any) -> None:
    """Subprocess target: load model → raw + chat + image tests → push result."""
    _bootstrap(config)

    from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends

    backend = str(config.get("backend", "cpu")).strip().lower()
    model_path = config["model_path"]
    mmproj_path = config.get("mmproj_path", "")
    max_tokens = int(config.get("max_tokens", 100))

    try:
        size_gb = Path(model_path).stat().st_size / (1024**3) if Path(model_path).exists() else 0.0

        if backend == "vulkan":
            apply_vulkan_env_vars(
                model_size_gb=size_gb,
                vulkan_device_index=0,
                host_memory_override="auto",
            )

        ensure_ggml_backends()

        from llama_cpp import Llama
        from scripts.model_test_suite import (
            configure_vision_chat_handler,
            run_chat_completion,
            run_image_test,
            run_raw_completion,
        )

        init_kw = _build_llama_kwargs(config, model_path, mmproj_path, backend)

        t0 = time.perf_counter()
        llm = Llama(**init_kw)
        if mmproj_path and Path(mmproj_path).exists():
            configure_vision_chat_handler(
                llm=llm,
                model_path=model_path,
                mmproj_path=mmproj_path,
            )
        load_s = time.perf_counter() - t0

        raw_ok, raw_ttft, raw_total_s, raw_tokens, raw_tps, _ = run_raw_completion(llm, max_tokens)
        chat_ok, chat_ttft, chat_total_s, chat_tokens, chat_tps, _ = run_chat_completion(llm, max_tokens)

        image_ok = False
        image_skipped = True
        test_image = config.get("test_image", "")
        if config.get("multimodal") and test_image and Path(test_image).exists():
            image_ok, _ = run_image_test(llm, Path(test_image), max_tokens)
            image_skipped = False

        del llm
        gc.collect()

        q.put({
            "ok": True, "load_s": load_s, "size_gb": size_gb,
            "raw_ok": raw_ok, "raw_ttft": raw_ttft, "raw_total_s": raw_total_s,
            "raw_tokens": raw_tokens, "raw_tps": raw_tps,
            "chat_ok": chat_ok, "chat_ttft": chat_ttft, "chat_total_s": chat_total_s,
            "chat_tokens": chat_tokens, "chat_tps": chat_tps,
            "image_ok": image_ok, "image_skipped": image_skipped,
            "error": "",
        })
    except Exception as exc:
        q.put({"ok": False, "error": str(exc), "size_gb": 0.0})


def run_tool_use(config: dict[str, Any], q: Any) -> None:
    """Subprocess target: load model -> run tool-use template check -> push result."""
    _bootstrap(config)

    from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends

    backend = str(config.get("backend", "cpu")).strip().lower()
    model_path = config["model_path"]
    mmproj_path = config.get("mmproj_path", "")
    max_tokens = int(config.get("max_tokens", 100))

    try:
        size_gb = Path(model_path).stat().st_size / (1024**3) if Path(model_path).exists() else 0.0

        if backend == "vulkan":
            apply_vulkan_env_vars(
                model_size_gb=size_gb,
                vulkan_device_index=0,
                host_memory_override="auto",
            )

        ensure_ggml_backends()

        from llama_cpp import Llama
        from scripts.model_test_suite import configure_vision_chat_handler, run_tool_use_test

        init_kw = _build_llama_kwargs(config, model_path, mmproj_path, backend)

        t0 = time.perf_counter()
        llm = Llama(**init_kw)
        if mmproj_path and Path(mmproj_path).exists():
            configure_vision_chat_handler(
                llm=llm,
                model_path=model_path,
                mmproj_path=mmproj_path,
            )
        load_s = time.perf_counter() - t0

        tool_use_ok, detail = run_tool_use_test(llm, max_tokens)

        del llm
        gc.collect()

        q.put({
            "ok": True,
            "load_s": load_s,
            "size_gb": size_gb,
            "tool_use_ok": bool(tool_use_ok),
            "detail": str(detail),
            "error": "",
        })
    except Exception as exc:
        q.put({
            "ok": False,
            "error": str(exc),
            "size_gb": 0.0,
            "tool_use_ok": False,
            "detail": "",
        })


def run_benchmark(config: dict[str, Any], q: Any) -> None:
    """Subprocess target: load model → warmup → N benchmark passes → push result."""
    _bootstrap(config)

    from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends

    backend = str(config.get("backend", "cpu")).strip().lower()
    model_path = config["model_path"]
    mmproj_path = config.get("mmproj_path", "")
    max_tokens = int(config.get("max_tokens", 100))
    passes = int(config.get("passes", 5))

    try:
        size_gb = Path(model_path).stat().st_size / (1024**3) if Path(model_path).exists() else 0.0

        if backend == "vulkan":
            apply_vulkan_env_vars(
                model_size_gb=size_gb,
                vulkan_device_index=0,
                host_memory_override="auto",
            )

        ensure_ggml_backends()

        from llama_cpp import Llama
        from scripts.llm_tuner import (
            _run_benchmark_passes,
            _run_chat_completion_once,
            _run_completion_once,
        )
        from scripts.model_test_suite import configure_vision_chat_handler

        init_kw = _build_llama_kwargs(config, model_path, mmproj_path, backend)

        t0 = time.perf_counter()
        llm = Llama(**init_kw)
        if mmproj_path and Path(mmproj_path).exists():
            configure_vision_chat_handler(
                llm=llm,
                model_path=model_path,
                mmproj_path=mmproj_path,
            )
        load_s = time.perf_counter() - t0

        warmup_tokens, _, _ = _run_completion_once(
            llm=llm, prompt="Warmup.", max_tokens=min(64, max_tokens),
        )
        if warmup_tokens == 0:
            _run_chat_completion_once(
                llm=llm, prompt="Warmup.", max_tokens=min(64, max_tokens),
            )
        metrics = _run_benchmark_passes(llm, passes, max_tokens)

        del llm
        gc.collect()

        metrics_ser = [
            {"pass_index": m.pass_index, "tokens": m.tokens, "wall_s": m.wall_s,
             "tps": m.tps, "stale_retry": m.stale_retry, "error": m.error}
            for m in metrics
        ]
        q.put({"ok": True, "load_s": load_s, "size_gb": size_gb,
               "metrics": metrics_ser, "error": ""})
    except Exception as exc:
        q.put({"ok": False, "error": str(exc), "size_gb": 0.0, "metrics": []})


# ── internal helpers ────────────────────────────────────────────────


def _bootstrap(config: dict[str, Any]) -> None:
    """Ensure project root is on sys.path (same as standalone scripts)."""
    root = config.get("project_root", _PROJECT_ROOT)
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _build_llama_kwargs(
    config: dict[str, Any],
    model_path: str,
    mmproj_path: str,
    backend: str,
) -> dict[str, Any]:
    """Build Llama() init kwargs matching basic_brain's pattern."""
    gpu_layers = int(config.get("gpu_layers", -1))
    if backend == "cpu":
        gpu_layers = 0
    elif backend == "vulkan" and gpu_layers == 0:
        gpu_layers = -1

    kw: dict[str, Any] = {
        "model_path": model_path,
        "n_gpu_layers": gpu_layers,
        "n_ctx": int(config.get("ctx_length", 2048)),
        "n_batch": int(config.get("n_batch", 512)),
        "verbose": False,
    }
    return kw
