"""End-to-end inference performance gauntlet.

Benchmarks three layers:
  Layer 0: direct llama_cpp (raw engine ceiling)
  Layer 1: gpt_server /v1/chat/completions (production inference path)
  Layer 2: gpt_terminal /api/chat (user-facing path)

Primary goal: validate target average TPS on Layer 1 for selected model(s).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends, get_hardware_snapshot
from scripts.llm_tuner import _clear_kv_cache
from scripts.model_test_suite import download_model
from scripts.model_test_suite import select_models


@dataclass
class PassMetric:
    pass_index: int
    tokens: int
    wall_s: float
    tps: float
    ttft_s: float = 0.0
    stale_retry: bool = False
    error: str = ""


@dataclass
class LayerResult:
    layer_id: int
    name: str
    status: str
    load_s: float
    passes: list[PassMetric] = field(default_factory=list)
    avg_tps: float = 0.0
    stddev_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0
    trimmed: bool = False
    notes: str = ""


@dataclass
class ModelResult:
    model_id: str
    model_name: str
    quant: str
    local_path: str
    size_gb: float
    layer_results: list[LayerResult]
    meets_target: bool
    target_tps: float


_DEFAULT_PROMPT = "Write a detailed explanation of how neural networks learn through backpropagation."
_LAYER_NAME = {0: "raw_llama_cpp", 1: "gpt_server_http", 2: "gpt_terminal_http"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference performance gauntlet")
    parser.add_argument("--focus", type=str, default="gemma3_27b_q4km", help="Model id or 'all'")
    parser.add_argument("--layers", type=str, default="0,1,2", help="Comma-separated subset: 0,1,2")
    parser.add_argument("--passes", type=int, default=5, help="Benchmark passes per layer")
    parser.add_argument("--max-tokens", type=int, default=200, help="Requested max output tokens")
    parser.add_argument("--warmup-tokens", type=int, default=64, help="Warmup tokens")
    parser.add_argument("--target-tps", type=float, default=30.0, help="Layer 1 target average TPS")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ctx-length", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--backend", choices=["auto", "vulkan", "cpu"], default="auto")
    parser.add_argument("--model-dir", type=str, default=str(Path.home() / "Downloads" / "model_test_cache"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--auth-token", type=str, default=os.environ.get("MINILOADER_API_TOKEN", ""))
    parser.add_argument("--server-base-url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--terminal-base-url", type=str, default="http://127.0.0.1:3000")
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--output-json", type=str, default="scripts/gauntlet_results.json")
    return parser.parse_args()


def _resolve_layers(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        val = int(p)
        if val not in {0, 1, 2}:
            raise ValueError(f"Invalid layer '{val}'. Allowed: 0,1,2")
        out.append(val)
    if not out:
        raise ValueError("No layers selected")
    return sorted(set(out))


def _resolve_backend_choice(requested: str, backend_hint: str) -> str:
    selected = (requested or "auto").strip().lower()
    if selected in {"vulkan", "cpu"}:
        return selected
    hint = (backend_hint or "").strip().lower()
    return "vulkan" if hint == "vulkan" else "cpu"


def _compute_stats(metrics: list[PassMetric]) -> tuple[float, float, float, float, bool]:
    good = [m.tps for m in metrics if not m.error and m.tps > 0.0]
    if not good:
        return 0.0, 0.0, 0.0, 0.0, False
    trimmed = list(good)
    outlier_trimmed = False
    if len(trimmed) > 3:
        trimmed.remove(min(trimmed))
        trimmed.remove(max(trimmed))
        outlier_trimmed = True
    avg = statistics.mean(trimmed) if trimmed else 0.0
    stddev = statistics.pstdev(trimmed) if len(trimmed) > 1 else 0.0
    return avg, stddev, min(good), max(good), outlier_trimmed


def _auth_headers(auth_token: str) -> dict[str, str]:
    token = str(auth_token or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _llm_precondition_hint(err: str) -> str:
    low = err.lower()
    if any(
        key in low
        for key in (
            "backend is not configured",
            "unconfigured",
            "open backend config",
            "no model selected",
            "no model loaded",
            "llm link unplugged",
            "llm backend unreachable",
        )
    ):
        return (
            "LLM backend is not ready. Configure backend selection explicitly "
            "(Vulkan or CPU), ensure basic_brain has a loaded model, and verify "
            "gpt_server/gpt_terminal wiring before rerunning."
        )
    return ""


async def _http_health_ok(url: str, headers: dict[str, str], timeout_s: float) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=timeout_s)
            if resp.status_code == 200:
                return True, ""
            return False, f"{url} returned HTTP {resp.status_code}: {resp.text[:250]}"
    except Exception as exc:
        return False, f"{url} health check failed: {exc}"


async def _run_http_stream_pass(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
    pass_index: int,
) -> PassMetric:
    start = time.perf_counter()
    ttft: float | None = None
    token_count = 0
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=timeout_s,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return PassMetric(
                        pass_index=pass_index,
                        tokens=0,
                        wall_s=time.perf_counter() - start,
                        tps=0.0,
                        ttft_s=0.0,
                        error=f"HTTP {resp.status_code}: {body.decode(errors='replace')[:400]}",
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_line = line[5:].strip()
                    if not data_line:
                        continue
                    if data_line == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        return PassMetric(
                            pass_index=pass_index,
                            tokens=token_count,
                            wall_s=time.perf_counter() - start,
                            tps=0.0,
                            ttft_s=ttft or 0.0,
                            error=str(obj.get("error")),
                        )
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta", {})
                    token = str((delta or {}).get("content", ""))
                    if token:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        token_count += 1
    except Exception as exc:
        return PassMetric(
            pass_index=pass_index,
            tokens=token_count,
            wall_s=time.perf_counter() - start,
            tps=0.0,
            ttft_s=ttft or 0.0,
            error=str(exc),
        )

    wall = time.perf_counter() - start
    tps = token_count / wall if wall > 0 else 0.0
    return PassMetric(
        pass_index=pass_index,
        tokens=token_count,
        wall_s=wall,
        tps=tps,
        ttft_s=ttft or 0.0,
    )


def _build_model_selection_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        tier="all",
        family=[],
        include_user_models=[],
        max_models=0,
    )


def _select_target_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = select_models(_build_model_selection_args(args))
    focus = str(args.focus).strip()
    if focus.lower() == "all":
        return models
    return [m for m in models if str(m.get("id")) == focus]


def _resolve_model_local(
    *,
    args: argparse.Namespace,
    model: dict[str, Any],
    model_dir: Path,
) -> tuple[Path, float]:
    local_path, _, _ = download_model(
        model=model,
        model_dir=model_dir,
        skip_download=bool(args.skip_download),
        hf_token=str(args.hf_token or ""),
    )
    if not local_path.exists():
        raise FileNotFoundError(f"Model file missing locally: {local_path}")
    size_gb = local_path.stat().st_size / (1024**3)
    return local_path, size_gb


def _run_layer0(
    *,
    args: argparse.Namespace,
    backend: str,
    model_path: Path,
    size_gb: float,
) -> LayerResult:
    layer = LayerResult(layer_id=0, name=_LAYER_NAME[0], status="ok", load_s=0.0)
    if backend == "vulkan":
        apply_vulkan_env_vars(
            model_size_gb=size_gb,
            vulkan_device_index=0,
            host_memory_override="auto",
        )
    ensure_ggml_backends()

    from llama_cpp import Llama

    gpu_layers = int(args.gpu_layers)
    if backend == "cpu":
        gpu_layers = 0
    elif gpu_layers == 0:
        gpu_layers = -1

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=gpu_layers,
        n_ctx=int(args.ctx_length),
        n_batch=int(args.n_batch),
        n_threads=os.cpu_count() or 8,
        n_threads_batch=os.cpu_count() or 8,
        type_k=8,
        type_v=8,
        seed=42,
        verbose=False,
    )
    layer.load_s = time.perf_counter() - t0

    try:
        # Warmup pass is intentionally discarded.
        _clear_kv_cache(llm)
        llm.create_completion(
            prompt="Warmup: respond with one short sentence.",
            max_tokens=min(int(args.warmup_tokens), int(args.max_tokens)),
            temperature=float(args.temperature),
            stream=False,
        )

        for i in range(1, int(args.passes) + 1):
            stale_retry = False
            _clear_kv_cache(llm)
            pass_start = time.perf_counter()
            out = llm.create_completion(
                prompt=str(args.prompt),
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=0.95,
                stream=False,
            )
            wall = time.perf_counter() - pass_start
            usage = out.get("usage", {}) if isinstance(out, dict) else {}
            tokens = int(usage.get("completion_tokens", 0))
            if tokens <= 0:
                text = str(out.get("choices", [{}])[0].get("text", "")) if isinstance(out, dict) else ""
                tokens = max(len(text.split()), 0)
            prior_good = [m.tokens for m in layer.passes if not m.error]
            if prior_good:
                mode_tokens = statistics.mode(prior_good)
                if mode_tokens > 0 and abs(tokens - mode_tokens) / float(mode_tokens) > 0.10:
                    stale_retry = True
                    _clear_kv_cache(llm)
                    pass_start = time.perf_counter()
                    out = llm.create_completion(
                        prompt=str(args.prompt),
                        max_tokens=int(args.max_tokens),
                        temperature=float(args.temperature),
                        top_p=0.95,
                        stream=False,
                    )
                    wall = time.perf_counter() - pass_start
                    usage = out.get("usage", {}) if isinstance(out, dict) else {}
                    tokens = int(usage.get("completion_tokens", 0))
                    if tokens <= 0:
                        text = str(out.get("choices", [{}])[0].get("text", "")) if isinstance(out, dict) else ""
                        tokens = max(len(text.split()), 0)

            layer.passes.append(
                PassMetric(
                    pass_index=i,
                    tokens=tokens,
                    wall_s=wall,
                    tps=(tokens / wall if wall > 0 else 0.0),
                    ttft_s=0.0,
                    stale_retry=stale_retry,
                )
            )
    except Exception as exc:
        layer.status = "failed"
        layer.notes = str(exc)
    finally:
        try:
            llm.close()
        except Exception:
            pass
        gc.collect()

    layer.avg_tps, layer.stddev_tps, layer.min_tps, layer.max_tps, layer.trimmed = _compute_stats(layer.passes)
    if layer.status == "ok" and layer.avg_tps <= 0.0:
        layer.status = "failed"
        if not layer.notes:
            layer.notes = "No successful benchmark passes."
    return layer


async def _run_layer1(args: argparse.Namespace) -> LayerResult:
    layer = LayerResult(layer_id=1, name=_LAYER_NAME[1], status="ok", load_s=0.0)
    base = str(args.server_base_url).rstrip("/")
    headers = _auth_headers(str(args.auth_token or ""))
    ok, health_err = await _http_health_ok(f"{base}/v1/health", headers, timeout_s=5.0)
    if not ok:
        layer.status = "skipped"
        layer.notes = health_err
        return layer

    for i in range(1, int(args.passes) + 1):
        payload = {
            "request_id": f"gauntlet_l1_{uuid.uuid4().hex[:12]}",
            "thread_id": f"gauntlet_l1_{i}_{uuid.uuid4().hex[:8]}",
            "messages": [{"role": "user", "content": str(args.prompt)}],
            "stream": True,
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
        }
        metric = await _run_http_stream_pass(
            url=f"{base}/v1/chat/completions",
            payload=payload,
            headers=headers,
            timeout_s=float(args.request_timeout_s),
            pass_index=i,
        )
        layer.passes.append(metric)
        if metric.error:
            hint = _llm_precondition_hint(metric.error)
            if hint:
                layer.status = "failed_precondition"
                layer.notes = hint
                break

    layer.avg_tps, layer.stddev_tps, layer.min_tps, layer.max_tps, layer.trimmed = _compute_stats(layer.passes)
    if layer.status == "ok" and layer.avg_tps <= 0.0:
        layer.status = "failed"
        layer.notes = "No successful HTTP streaming passes."
    return layer


async def _run_layer2(args: argparse.Namespace) -> LayerResult:
    layer = LayerResult(layer_id=2, name=_LAYER_NAME[2], status="ok", load_s=0.0)
    base = str(args.terminal_base_url).rstrip("/")
    headers = _auth_headers(str(args.auth_token or ""))
    ok, health_err = await _http_health_ok(f"{base}/api/health", headers, timeout_s=5.0)
    if not ok:
        layer.status = "skipped"
        layer.notes = health_err
        return layer

    for i in range(1, int(args.passes) + 1):
        payload = {
            "request_id": f"gauntlet_l2_{uuid.uuid4().hex[:12]}",
            "thread_id": f"gauntlet_l2_{i}_{uuid.uuid4().hex[:8]}",
            "messages": [{"role": "user", "content": str(args.prompt)}],
            "rag_enabled": False,
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
        }
        metric = await _run_http_stream_pass(
            url=f"{base}/api/chat",
            payload=payload,
            headers=headers,
            timeout_s=float(args.request_timeout_s),
            pass_index=i,
        )
        layer.passes.append(metric)
        if metric.error:
            hint = _llm_precondition_hint(metric.error)
            if hint:
                layer.status = "failed_precondition"
                layer.notes = hint
                break

    layer.avg_tps, layer.stddev_tps, layer.min_tps, layer.max_tps, layer.trimmed = _compute_stats(layer.passes)
    if layer.status == "ok" and layer.avg_tps <= 0.0:
        layer.status = "failed"
        layer.notes = "No successful terminal streaming passes."
    return layer


def _layer_by_id(results: list[LayerResult], layer_id: int) -> LayerResult | None:
    for item in results:
        if item.layer_id == layer_id:
            return item
    return None


def _print_layer(layer: LayerResult) -> None:
    print(
        f"  L{layer.layer_id} {layer.name:<17} status={layer.status:<18} "
        f"avg={layer.avg_tps:>7.2f} t/s  std={layer.stddev_tps:>6.2f}  "
        f"min/max={layer.min_tps:>6.2f}/{layer.max_tps:>6.2f}"
    )
    if layer.notes:
        print(f"    note: {layer.notes}")


async def _run_model(args: argparse.Namespace, model: dict[str, Any], layers: list[int], backend: str) -> ModelResult:
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    local_path, size_gb = _resolve_model_local(args=args, model=model, model_dir=model_dir)
    print(f"\n=== {model['id']} ({model['name']}) ===")
    print(f"model_path={local_path}")
    print(f"size_gb={size_gb:.2f} backend={backend}")

    layer_results: list[LayerResult] = []

    for layer_id in layers:
        if layer_id == 0:
            layer_results.append(_run_layer0(args=args, backend=backend, model_path=local_path, size_gb=size_gb))
        elif layer_id == 1:
            layer_results.append(await _run_layer1(args))
        elif layer_id == 2:
            layer_results.append(await _run_layer2(args))

    for layer in layer_results:
        _print_layer(layer)

    l0 = _layer_by_id(layer_results, 0)
    l1 = _layer_by_id(layer_results, 1)
    l2 = _layer_by_id(layer_results, 2)

    if l0 is not None and l1 is not None and l0.avg_tps > 0 and l1.avg_tps > 0:
        ipc_overhead = max(0.0, (l0.avg_tps - l1.avg_tps) / l0.avg_tps * 100.0)
        print(f"  overhead layer0->1: {ipc_overhead:.1f}%")
    if l1 is not None and l2 is not None and l1.avg_tps > 0 and l2.avg_tps > 0:
        terminal_overhead = max(0.0, (l1.avg_tps - l2.avg_tps) / l1.avg_tps * 100.0)
        print(f"  overhead layer1->2: {terminal_overhead:.1f}%")

    meets_target = bool(l1 and l1.status == "ok" and l1.avg_tps >= float(args.target_tps))
    print(
        f"  target (layer1 >= {float(args.target_tps):.2f} t/s): "
        f"{'PASS' if meets_target else 'FAIL'}"
    )

    return ModelResult(
        model_id=str(model["id"]),
        model_name=str(model["name"]),
        quant=str(model.get("quant", "unknown")),
        local_path=str(local_path),
        size_gb=size_gb,
        layer_results=layer_results,
        meets_target=meets_target,
        target_tps=float(args.target_tps),
    )


async def _main_async(args: argparse.Namespace) -> int:
    layers = _resolve_layers(str(args.layers))
    selected_models = _select_target_models(args)
    if not selected_models:
        print(f"No model matched --focus {args.focus!r}")
        return 1

    hw = get_hardware_snapshot()
    backend = _resolve_backend_choice(str(args.backend), hw.ai_backend_hint.value)

    print("=== Inference Performance Gauntlet ===")
    print(f"layers={layers} passes={args.passes} max_tokens={args.max_tokens}")
    print(f"target_tps={float(args.target_tps):.2f} focus={args.focus}")
    print(f"backend={backend} (hint={hw.ai_backend_hint.value})")

    results: list[ModelResult] = []
    for model in selected_models:
        results.append(await _run_model(args, model, layers, backend))

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "args": vars(args),
        "layers": layers,
        "hardware": {
            "gpu_name": hw.gpu_name,
            "gpu_vendor": hw.gpu_vendor.value,
            "vram_total_mb": hw.vram_total_mb,
            "backend_hint": hw.ai_backend_hint.value,
        },
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON report: {out_path}")

    total = len(results)
    passes = sum(1 for r in results if r.meets_target)
    fails = total - passes
    print(f"target summary: total={total} pass={passes} fail={fails}")
    return 0 if fails == 0 else 2


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
