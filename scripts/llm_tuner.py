"""Local LLM tuner for GGUF models on the current hardware.

This script probes hardware, resolves models from the shared test-suite
manifest, benchmarks each model with multiple passes, applies stale-cache
guards, removes outliers, and emits a JSON report.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends, get_hardware_snapshot
from scripts.model_test_suite import (
    configure_vision_chat_handler,
    download_model,
    select_models,
)


@dataclass
class PassMetric:
    pass_index: int
    tokens: int
    wall_s: float
    tps: float
    stale_retry: bool = False
    error: str = ""


@dataclass
class ModelRunResult:
    model_id: str
    model_name: str
    family: str
    tier: int
    quant: str
    local_path: str
    size_gb: float
    load_s: float
    status: str
    error: str
    size_failure: bool
    passes_ok: int
    requested_passes: int
    avg_tps: float
    stddev_tps: float
    outlier_trimmed: bool
    tokens_mode: int
    benchmark_passes: list[PassMetric] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local LLM tuner")
    parser.add_argument("--tier", choices=["1", "2", "all"], default="all")
    parser.add_argument("--family", nargs="*", default=[], help="Filter manifest families")
    parser.add_argument("--passes", type=int, default=5, help="Benchmark passes (excluding warmup)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--model-dir", type=str, default=str(Path.home() / "Downloads" / "model_test_cache"))
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--ctx-length", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--max-models", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output-json", type=str, default="scripts/llm_tuner_results.json")
    parser.add_argument("--include-user-models", nargs="*", default=[], help="Inject local GGUF paths")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--backend",
        choices=["auto", "vulkan", "cpu"],
        default="auto",
        help="Execution backend for tuning. 'auto' uses hardware probe hint.",
    )
    return parser.parse_args()


def _is_size_failure(error_text: str, size_gb: float, vram_total_gb: float) -> bool:
    lower = error_text.lower()
    has_memory_error = any(x in lower for x in ("out of memory", "alloc", "memory"))
    exceeds_vram = vram_total_gb > 0.0 and size_gb > vram_total_gb
    return has_memory_error or exceeds_vram


def _resolve_backend_choice(requested: str, backend_hint: str) -> str:
    selected = (requested or "auto").strip().lower()
    if selected in {"vulkan", "cpu"}:
        return selected
    hint = (backend_hint or "").strip().lower()
    return "vulkan" if hint == "vulkan" else "cpu"


def _clear_kv_cache(llm: Any) -> None:
    llm.reset()
    try:
        import llama_cpp  # imported lazily to avoid side effects at startup

        llama_cpp.llama_memory_clear(
            llama_cpp.llama_get_memory(llm.ctx),
            False,
        )
    except Exception:
        pass


def _run_completion_once(llm: Any, prompt: str, max_tokens: int) -> tuple[int, float, float]:
    start = time.perf_counter()
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.95,
        min_p=0.15,
        stream=False,
    )
    wall_s = time.perf_counter() - start
    usage = out.get("usage", {}) if isinstance(out, dict) else {}
    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens <= 0:
        text = str(out.get("choices", [{}])[0].get("text", "")) if isinstance(out, dict) else ""
        completion_tokens = max(len(text.split()), 0)
    tps = completion_tokens / wall_s if wall_s > 0 else 0.0
    return completion_tokens, wall_s, tps


def _run_chat_completion_once(llm: Any, prompt: str, max_tokens: int) -> tuple[int, float, float]:
    """Chat-completion path for models that produce 0 tokens on raw completion
    (e.g. LFM2 which requires ChatML framing to generate anything)."""
    messages = [{"role": "user", "content": prompt}]
    start = time.perf_counter()
    token_count = 0
    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.95,
        min_p=0.15,
        stream=True,
    ):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            token_count += 1
    wall_s = time.perf_counter() - start
    tps = token_count / wall_s if wall_s > 0 else 0.0
    return token_count, wall_s, tps


def _mode_token_count(values: list[int]) -> int:
    if not values:
        return 0
    counter = Counter(values)
    most_common = counter.most_common()
    # Stable tie-breaker: choose the lower token count to avoid overestimating.
    top_freq = most_common[0][1]
    top_values = sorted([v for v, freq in most_common if freq == top_freq])
    return top_values[0]


def _run_benchmark_passes(llm: Any, pass_count: int, max_tokens: int) -> list[PassMetric]:
    prompt = "Write three short bullet points about reliable local LLM benchmarking."
    metrics: list[PassMetric] = []
    use_chat = False

    for idx in range(1, pass_count + 1):
        stale_retry = False
        error = ""
        tokens = 0
        wall_s = 0.0
        tps = 0.0

        run_fn = _run_chat_completion_once if use_chat else _run_completion_once

        try:
            _clear_kv_cache(llm)
            tokens, wall_s, tps = run_fn(llm=llm, prompt=prompt, max_tokens=max_tokens)

            # Chat-only models (e.g. LFM2) emit EOS immediately on raw
            # completion.  When the first pass yields 0 tokens, switch to
            # chat completion and redo this pass so every model gets a fair
            # benchmark.
            if idx == 1 and tokens == 0 and not use_chat:
                use_chat = True
                _clear_kv_cache(llm)
                tokens, wall_s, tps = _run_chat_completion_once(
                    llm=llm, prompt=prompt, max_tokens=max_tokens,
                )

            prior = [m.tokens for m in metrics if m.error == ""]
            if prior:
                mode_tokens = _mode_token_count(prior)
                if mode_tokens > 0:
                    delta = abs(tokens - mode_tokens) / float(mode_tokens)
                    if delta > 0.10:
                        stale_retry = True
                        _clear_kv_cache(llm)
                        tokens, wall_s, tps = run_fn(
                            llm=llm, prompt=prompt, max_tokens=max_tokens,
                        )
        except Exception as exc:
            error = str(exc)

        metrics.append(
            PassMetric(
                pass_index=idx,
                tokens=tokens,
                wall_s=wall_s,
                tps=tps,
                stale_retry=stale_retry,
                error=error,
            )
        )

    return metrics


def _compute_tps_stats(metrics: list[PassMetric]) -> tuple[float, float, bool]:
    good_tps = [m.tps for m in metrics if m.error == "" and m.tps > 0.0]
    if not good_tps:
        return 0.0, 0.0, False

    trimmed = list(good_tps)
    outlier_trimmed = False
    if len(trimmed) > 3:
        min_v = min(trimmed)
        max_v = max(trimmed)
        trimmed.remove(min_v)
        trimmed.remove(max_v)
        outlier_trimmed = True

    avg = statistics.mean(trimmed) if trimmed else 0.0
    stddev = statistics.pstdev(trimmed) if len(trimmed) > 1 else 0.0
    return avg, stddev, outlier_trimmed


def _build_selection_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        tier=args.tier,
        family=args.family,
        include_user_models=args.include_user_models,
        max_models=args.max_models,
    )


def _resolve_and_sort_models(args: argparse.Namespace, model_dir: Path) -> tuple[list[dict[str, Any]], list[ModelRunResult]]:
    selected = select_models(_build_selection_args(args))
    # Use params as a lightweight prefetch order; real run order is by on-disk size.
    selected.sort(key=lambda m: float(m.get("params_b", 0.0)))

    resolved: list[dict[str, Any]] = []
    early_failures: list[ModelRunResult] = []

    for model in selected:
        try:
            local_path, _, mmproj_path = download_model(
                model=model,
                model_dir=model_dir,
                skip_download=bool(args.skip_download),
                hf_token=str(args.hf_token or ""),
            )
            if not local_path.exists():
                raise FileNotFoundError(f"Model file missing locally: {local_path}")
            size_gb = local_path.stat().st_size / (1024**3)
            resolved.append(
                {
                    "model": model,
                    "local_path": local_path,
                    "mmproj_path": mmproj_path,
                    "size_gb": size_gb,
                }
            )
        except Exception as exc:
            early_failures.append(
                ModelRunResult(
                    model_id=str(model["id"]),
                    model_name=str(model["name"]),
                    family=str(model["family"]),
                    tier=int(model["tier"]),
                    quant=str(model["quant"]),
                    local_path="",
                    size_gb=0.0,
                    load_s=0.0,
                    status="resolve_failed",
                    error=str(exc),
                    size_failure=False,
                    passes_ok=0,
                    requested_passes=args.passes,
                    avg_tps=0.0,
                    stddev_tps=0.0,
                    outlier_trimmed=False,
                    tokens_mode=0,
                    benchmark_passes=[],
                )
            )

    resolved.sort(key=lambda x: x["size_gb"])
    return resolved, early_failures


def _print_row(name: str, size_gb: float, passes_ok: int, passes_requested: int, avg_tps: float, stddev: float, status: str) -> None:
    print(
        f"{name[:28]:<28} {size_gb:>6.2f} GB   "
        f"{passes_ok:>2}/{passes_requested:<2}   "
        f"{avg_tps:>8.1f}   {stddev:>7.1f}   {status}"
    )


def _print_ranked_tps(results: list[ModelRunResult], top_n: int = 10) -> None:
    ranked = sorted(
        [r for r in results if r.avg_tps > 0.0 and r.status in {"ok", "partial"}],
        key=lambda r: r.avg_tps,
        reverse=True,
    )
    if not ranked:
        return

    print("\nTop Models by Avg TPS:")
    print(f"{'Rank':<4} {'Model':<28} {'Avg TPS':>8} {'StdDev':>8} {'Size':>8}")
    print("-" * 64)
    for idx, item in enumerate(ranked[:top_n], start=1):
        print(
            f"{idx:<4} {item.model_name[:28]:<28} "
            f"{item.avg_tps:>8.1f} {item.stddev_tps:>8.1f} {item.size_gb:>6.2f}GB"
        )


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    hw = get_hardware_snapshot()
    vram_total_gb = float(hw.vram_total_mb) / 1024.0 if float(hw.vram_total_mb) > 0 else 0.0

    print("== Local LLM Tuner ==")
    configured_backend = _resolve_backend_choice(str(args.backend), hw.ai_backend_hint.value)
    print(
        f"GPU: {hw.gpu_name} ({hw.gpu_vendor}), VRAM: {vram_total_gb:.2f} GB, "
        f"Backend hint: {hw.ai_backend_hint}"
    )
    print(f"Configured backend: {configured_backend}")
    print("Resolving model files...")

    resolved_models, early_failures = _resolve_and_sort_models(args=args, model_dir=model_dir)
    results: list[ModelRunResult] = list(early_failures)
    size_fail_count = 0

    print("\nModel                         Size      Passes    Avg TPS    StdDev   Status")
    print("-" * 82)

    for item in resolved_models:
        model = item["model"]
        local_path: Path = item["local_path"]
        mmproj_path = str(item["mmproj_path"] or "")
        size_gb = float(item["size_gb"])

        if configured_backend == "vulkan":
            apply_vulkan_env_vars(model_size_gb=size_gb, vulkan_device_index=0, host_memory_override="auto")

        ensure_ggml_backends()

        llm = None
        load_s = 0.0
        status = "ok"
        error = ""
        metrics: list[PassMetric] = []

        try:
            from llama_cpp import Llama

            init_kwargs: dict[str, Any] = {
                "model_path": str(local_path),
                "n_gpu_layers": (0 if configured_backend == "cpu" else (-1 if int(args.gpu_layers) == 0 else int(args.gpu_layers))),
                "n_ctx": int(args.ctx_length),
                "n_batch": int(args.n_batch),
                "verbose": False,
            }

            t0 = time.perf_counter()
            llm = Llama(**init_kwargs)
            if mmproj_path and Path(mmproj_path).exists():
                handler_name = configure_vision_chat_handler(
                    llm=llm,
                    model_path=str(local_path),
                    mmproj_path=mmproj_path,
                    model_hint=str(model.get("family", "")),
                )
                if handler_name:
                    print(f"  vision handler: {handler_name} ({Path(mmproj_path).name})")
                else:
                    print("  warning: mmproj present but vision handler was not attached")
            load_s = time.perf_counter() - t0

            # Warmup pass to prime caches/shaders (discarded on purpose).
            _clear_kv_cache(llm)
            warmup_tokens, _, _ = _run_completion_once(
                llm=llm,
                prompt="Warmup: respond with one short sentence.",
                max_tokens=min(64, int(args.max_tokens)),
            )
            if warmup_tokens == 0:
                _clear_kv_cache(llm)
                _run_chat_completion_once(
                    llm=llm,
                    prompt="Warmup: respond with one short sentence.",
                    max_tokens=min(64, int(args.max_tokens)),
                )

            metrics = _run_benchmark_passes(
                llm=llm,
                pass_count=int(args.passes),
                max_tokens=int(args.max_tokens),
            )

            if not any(m.error == "" for m in metrics):
                status = "inference_failed"
                error = "; ".join(m.error for m in metrics if m.error) or "No successful benchmark pass."
            elif any(m.error for m in metrics):
                status = "partial"
                error = "; ".join(m.error for m in metrics if m.error)
        except Exception as exc:
            status = "load_failed"
            error = str(exc)
        finally:
            try:
                if llm is not None:
                    del llm
            finally:
                gc.collect()

        avg_tps, stddev_tps, outlier_trimmed = _compute_tps_stats(metrics)
        token_mode = _mode_token_count([m.tokens for m in metrics if m.error == ""])
        size_failure = status == "load_failed" and _is_size_failure(error, size_gb, vram_total_gb)
        if size_failure:
            size_fail_count += 1

        passes_ok = len([m for m in metrics if m.error == ""])
        result = ModelRunResult(
            model_id=str(model["id"]),
            model_name=str(model["name"]),
            family=str(model["family"]),
            tier=int(model["tier"]),
            quant=str(model["quant"]),
            local_path=str(local_path),
            size_gb=size_gb,
            load_s=load_s,
            status=status,
            error=error,
            size_failure=size_failure,
            passes_ok=passes_ok,
            requested_passes=int(args.passes),
            avg_tps=avg_tps,
            stddev_tps=stddev_tps,
            outlier_trimmed=outlier_trimmed,
            tokens_mode=token_mode,
            benchmark_passes=metrics,
        )
        results.append(result)
        _print_row(
            name=result.model_name,
            size_gb=result.size_gb,
            passes_ok=result.passes_ok,
            passes_requested=result.requested_passes,
            avg_tps=result.avg_tps,
            stddev=result.stddev_tps,
            status="OOM" if result.size_failure else result.status.upper(),
        )

        if size_fail_count >= 3:
            print("\nVRAM ceiling reached after 3 size-related load failures. Stopping early.")
            break

    ok_count = len([r for r in results if r.status == "ok"])
    partial_count = len([r for r in results if r.status == "partial"])
    failed_count = len(results) - ok_count - partial_count
    report = {
        "timestamp": time.time(),
        "args": vars(args),
        "hardware": asdict(hw),
        "summary": {
            "total": len(results),
            "ok": ok_count,
            "partial": partial_count,
            "failed": failed_count,
            "size_fail_count": size_fail_count,
        },
        "results": [asdict(r) for r in results],
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(
        f"total={report['summary']['total']} ok={ok_count} "
        f"partial={partial_count} failed={failed_count} size_fails={size_fail_count}"
    )
    _print_ranked_tps(results)
    print(f"JSON report: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
