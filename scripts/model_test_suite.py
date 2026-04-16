"""Standalone GGUF model compatibility and performance test suite.

This script validates model loading/inference against the project's local
llama.cpp stack (JamePeng fork + ggml backends), records metrics, and emits a
JSON report.

Example:
    python scripts/model_test_suite.py --tier 1
    python scripts/model_test_suite.py --tier all --family qwen gemma
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files

from core.hardware_probe import apply_vulkan_env_vars, ensure_ggml_backends, get_hardware_snapshot
from core.llama_runtime import JAMEPENG_PINNED_VERSION, LLAMACPP_RELEASE_TAG


MODEL_MANIFEST: list[dict[str, Any]] = [
    {
        "id": "gemma3_1b_q8",
        "family": "gemma",
        "tier": 1,
        "name": "Gemma 3 1B IT",
        "params_b": 1.0,
        "repo_id": "lmstudio-community/gemma-3-1b-it-GGUF",
        "filename_pattern": "*Q8_0.gguf",
        "quant": "Q8_0",
        "multimodal": True,
    },
    {
        "id": "gemma3_4b_q4km",
        "family": "gemma",
        "tier": 1,
        "name": "Gemma 3 4B IT",
        "params_b": 4.0,
        "repo_id": "lmstudio-community/gemma-3-4b-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "qwen3_4b_q4km",
        "family": "qwen",
        "tier": 1,
        "name": "Qwen3 4B",
        "params_b": 4.0,
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": False,
    },
    {
        "id": "qwen25_7b_q4km",
        "family": "qwen",
        "tier": 1,
        "name": "Qwen2.5 7B Instruct",
        "params_b": 7.0,
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": False,
    },
    {
        "id": "mistral7b_q4km",
        "family": "mistral",
        "tier": 1,
        "name": "Mistral 7B Instruct v0.3",
        "params_b": 7.0,
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": False,
    },
    {
        "id": "qwen25vl_7b_q4km",
        "family": "qwen",
        "tier": 1,
        "name": "Qwen2.5-VL 7B Instruct",
        "params_b": 9.0,
        "repo_id": "lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "gemma3_12b_q4km",
        "family": "gemma",
        "tier": 2,
        "name": "Gemma 3 12B IT",
        "params_b": 12.0,
        "repo_id": "lmstudio-community/gemma-3-12b-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "gemma3_27b_q4km",
        "family": "gemma",
        "tier": 2,
        "name": "Gemma 3 27B IT",
        "params_b": 27.0,
        "repo_id": "lmstudio-community/gemma-3-27b-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "gemma4_e4b_q4km",
        "family": "gemma4",
        "tier": 2,
        "name": "Gemma 4 E4B IT",
        "params_b": 8.0,
        "repo_id": "lmstudio-community/gemma-4-E4B-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "gemma4_26b_a4b_q4km",
        "family": "gemma4",
        "tier": 2,
        "name": "Gemma 4 26B A4B IT",
        "params_b": 26.0,
        "repo_id": "lmstudio-community/gemma-4-26B-A4B-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "gemma4_31b_q4km",
        "family": "gemma4",
        "tier": 2,
        "name": "Gemma 4 31B IT",
        "params_b": 31.0,
        "repo_id": "lmstudio-community/gemma-4-31B-it-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": True,
    },
    {
        "id": "lfm25_1b_q4km",
        "family": "lfm",
        "tier": 2,
        "name": "LFM2.5 1.2B Instruct",
        "params_b": 1.2,
        "repo_id": "lmstudio-community/LFM2.5-1.2B-Instruct-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": False,
    },
    {
        "id": "lfm2_2b_q4km",
        "family": "lfm",
        "tier": 2,
        "name": "LFM2 2.6B Exp",
        "params_b": 2.6,
        "repo_id": "unsloth/LFM2-2.6B-Exp-GGUF",
        "filename_pattern": "*Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "multimodal": False,
    },
]


@dataclass
class TestResult:
    model_id: str
    model_name: str
    family: str
    tier: int
    quant: str
    local_path: str
    size_gb: float
    passed: bool
    load_ok: bool
    chat_ok: bool
    raw_ok: bool
    image_ok: bool
    image_test_skipped: bool
    failure_type: str
    failure_hint: str
    error: str
    load_s: float
    raw_ttft_s: float
    raw_total_s: float
    raw_tokens: int
    raw_tps: float
    chat_ttft_s: float
    chat_total_s: float
    chat_tokens: int
    chat_tps: float
    timestamp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GGUF model compatibility test suite")
    parser.add_argument("--tier", choices=["1", "2", "all"], default="1")
    parser.add_argument("--family", nargs="*", default=[], help="Filter by family names")
    parser.add_argument("--model-dir", type=str, default=str(Path.home() / "Downloads" / "model_test_cache"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--include-user-models", nargs="*", default=[], help="Local GGUF model paths")
    parser.add_argument("--ctx-length", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-models", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--output-json", type=str, default="scripts/model_test_results.json")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--test-image", type=str, default="", help="Path to local image for multimodal tests")
    parser.add_argument(
        "--backend",
        choices=["auto", "vulkan", "cpu"],
        default="auto",
        help="Execution backend for tests. 'auto' uses hardware probe hint.",
    )
    return parser.parse_args()


def classify_error(exc: Exception | str) -> tuple[str, str]:
    text = str(exc).strip()
    lower = text.lower()
    if "unknown model architecture" in lower:
        return ("unknown_architecture", "Update llama.cpp/JamePeng build via: python scripts/setup_environment.py")
    if "failed to load model from file" in lower or "failed to load model" in lower:
        return ("load_failure", "Check GGUF validity and model file integrity.")
    if "out of memory" in lower or "alloc" in lower or "memory" in lower:
        return ("oom", "Reduce --ctx-length or --gpu-layers, or use smaller quant.")
    if "vulkan" in lower or "ggml_backend" in lower:
        return ("vulkan_fail", "Check Vulkan drivers and rerun setup/repair.")
    if "system role not supported" in lower or "chat template" in lower:
        return ("chat_template_error", "Use raw completion path or merge system into first user message.")
    return ("other", "Inspect traceback and llama.cpp logs for root cause.")


def parse_release_num(tag: str) -> int:
    match = re.search(r"b(\d+)", tag)
    return int(match.group(1)) if match else -1


def check_release_staleness() -> dict[str, Any]:
    info = {
        "local_tag": LLAMACPP_RELEASE_TAG,
        "local_num": parse_release_num(LLAMACPP_RELEASE_TAG),
        "latest_tag": "",
        "latest_num": -1,
        "delta": None,
        "warning": "",
        "error": "",
    }
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
            resp.raise_for_status()
            payload = resp.json()
            latest_tag = str(payload.get("tag_name", ""))
            latest_num = parse_release_num(latest_tag)
            info["latest_tag"] = latest_tag
            info["latest_num"] = latest_num
            if info["local_num"] > 0 and latest_num > 0:
                delta = latest_num - info["local_num"]
                info["delta"] = delta
                if delta >= 10:
                    info["warning"] = (
                        f"Local llama.cpp tag {LLAMACPP_RELEASE_TAG} is {delta} releases behind latest "
                        f"{latest_tag}. New model architectures may fail."
                    )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def select_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    family_filter = {f.lower() for f in args.family}
    for item in MODEL_MANIFEST:
        if args.tier != "all" and str(item["tier"]) != args.tier:
            continue
        if family_filter and str(item["family"]).lower() not in family_filter:
            continue
        selected.append(dict(item))
    if args.max_models > 0:
        selected = selected[: args.max_models]
    for idx, path_str in enumerate(args.include_user_models):
        path = Path(path_str).expanduser().resolve()
        selected.append(
            {
                "id": f"user_model_{idx + 1}",
                "family": "user",
                "tier": 1 if args.tier != "2" else 2,
                "name": path.stem,
                "params_b": 0.0,
                "repo_id": "",
                "filename_pattern": path.name,
                "quant": "unknown",
                "multimodal": False,
                "user_path": str(path),
            }
        )
    return selected


def _casefold_match(filename: str, pattern: str) -> bool:
    return fnmatch.fnmatch(filename.casefold(), pattern.casefold())


def resolve_repo_file(repo_id: str, pattern: str, hf_token: str) -> str:
    files = list_repo_files(repo_id=repo_id, token=hf_token or None)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    matches = [f for f in ggufs if _casefold_match(Path(f).name, pattern)]
    if not matches:
        raise FileNotFoundError(f"No file matched pattern '{pattern}' in repo '{repo_id}'")
    # Prefer smaller path depth, then lexical (stable).
    matches.sort(key=lambda x: (x.count("/"), x.lower()))
    return matches[0]


def resolve_repo_files(repo_id: str, pattern: str, hf_token: str) -> list[str]:
    files = list_repo_files(repo_id=repo_id, token=hf_token or None)
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    matches = [f for f in ggufs if _casefold_match(Path(f).name, pattern)]
    if not matches:
        raise FileNotFoundError(f"No file matched pattern '{pattern}' in repo '{repo_id}'")
    matches.sort(key=lambda x: (x.count("/"), x.lower()))
    first = Path(matches[0]).name
    shard_match = re.match(r"^(.*)-\d{5}-of-\d{5}\.gguf$", first, flags=re.IGNORECASE)
    if not shard_match:
        return [matches[0]]
    prefix = shard_match.group(1)
    shard_re = re.compile(rf"^{re.escape(prefix)}-\d{{5}}-of-\d{{5}}\.gguf$", flags=re.IGNORECASE)
    shard_files = [f for f in ggufs if shard_re.match(Path(f).name)]
    shard_files.sort(key=lambda x: Path(x).name.lower())
    return shard_files if shard_files else [matches[0]]


def resolve_mmproj_file(repo_id: str, hf_token: str) -> str | None:
    files = [f for f in list_repo_files(repo_id=repo_id, token=hf_token or None) if f.lower().endswith(".gguf")]
    mmproj = [f for f in files if "mmproj" in Path(f).name.lower()]
    if not mmproj:
        return None
    for preferred in ("f16", "bf16", "f32"):
        for f in mmproj:
            if preferred in Path(f).name.lower():
                return f
    mmproj.sort(key=lambda x: Path(x).name.lower())
    return mmproj[0]


def download_model(
    model: dict[str, Any], model_dir: Path, skip_download: bool, hf_token: str
) -> tuple[Path, str, str]:
    if model.get("user_path"):
        p = Path(str(model["user_path"]))
        if not p.exists():
            raise FileNotFoundError(f"User model not found: {p}")
        return (p, p.name, "")

    repo_id = str(model["repo_id"])
    pattern = str(model["filename_pattern"])
    selected_filenames = resolve_repo_files(repo_id=repo_id, pattern=pattern, hf_token=hf_token)
    selected_filename = selected_filenames[0]
    mmproj_filename = ""
    if bool(model.get("multimodal")):
        mmproj_filename = resolve_mmproj_file(repo_id=repo_id, hf_token=hf_token) or ""
    if skip_download:
        local_guess = model_dir / repo_id.replace("/", "__") / selected_filename
        return (local_guess, selected_filename, "")

    local_paths = []
    for filename in selected_filenames:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(model_dir),
            token=hf_token or None,
            local_files_only=False,
            resume_download=True,
        )
        local_paths.append(Path(local_path))

    mmproj_path = ""
    if mmproj_filename:
        mm_local = hf_hub_download(
            repo_id=repo_id,
            filename=mmproj_filename,
            cache_dir=str(model_dir),
            token=hf_token or None,
            local_files_only=False,
            resume_download=True,
        )
        mmproj_path = str(Path(mm_local))

    return (local_paths[0], selected_filename, mmproj_path)


def run_raw_completion(llm: Any, max_tokens: int) -> tuple[bool, float, float, int, float, str]:
    prompt = "The quick brown fox"
    start = time.perf_counter()
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stream=False,
        )
        total_s = time.perf_counter() - start
        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        completion_tokens = int(usage.get("completion_tokens", 0))
        if completion_tokens <= 0:
            text = str(out.get("choices", [{}])[0].get("text", "")) if isinstance(out, dict) else ""
            completion_tokens = max(len(text.split()), 0)
        tps = completion_tokens / total_s if total_s > 0 else 0.0
        sample = ""
        if isinstance(out, dict):
            sample = str(out.get("choices", [{}])[0].get("text", ""))[:120]
        return (True, 0.0, total_s, completion_tokens, tps, sample)
    except Exception as exc:
        return (False, 0.0, time.perf_counter() - start, 0, 0.0, str(exc))


def run_chat_completion(llm: Any, max_tokens: int) -> tuple[bool, float, float, int, float, str]:
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    start = time.perf_counter()
    ttft: float | None = None
    token_count = 0
    sample_parts: list[str] = []
    try:
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            tok = str(delta.get("content", ""))
            if tok:
                if ttft is None:
                    ttft = time.perf_counter() - start
                token_count += 1
                if len("".join(sample_parts)) < 120:
                    sample_parts.append(tok)
        total_s = time.perf_counter() - start
        tps = token_count / total_s if total_s > 0 else 0.0
        return (token_count > 0, ttft or 0.0, total_s, token_count, tps, "".join(sample_parts)[:120])
    except Exception as exc:
        return (False, ttft or 0.0, time.perf_counter() - start, token_count, 0.0, str(exc))


def run_tool_use_test(llm: Any, max_tokens: int) -> tuple[bool, str]:
    messages = [
        {
            "role": "user",
            "content": "Use the available tool and provide a short challenge phrase.",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "confirm_tool_use_template",
                "description": "Confirm tool-call behavior by returning a challenge phrase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "challenge": {
                            "type": "string",
                            "description": "Short challenge phrase proving tool-call output.",
                        },
                    },
                    "required": ["challenge"],
                },
            },
        }
    ]
    try:
        out = llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=False,
        )
    except Exception as exc:
        return (False, str(exc))

    message = out.get("choices", [{}])[0].get("message", {}) if isinstance(out, dict) else {}
    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    if not tool_calls:
        content = str(message.get("content", "")).strip() if isinstance(message, dict) else ""
        if content:
            return (False, f"No tool call emitted. Content: {content[:120]}")
        return (False, "No tool call emitted.")

    first_call = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
    function_obj = first_call.get("function", {}) if isinstance(first_call, dict) else {}
    function_name = str(function_obj.get("name", "")).strip()
    if function_name != "confirm_tool_use_template":
        return (False, f"Unexpected tool name: {function_name or 'missing'}")

    raw_arguments = function_obj.get("arguments", {})
    arguments: dict[str, Any] = {}
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            if isinstance(parsed, dict):
                arguments = parsed
        except Exception:
            return (False, f"Tool call arguments were not valid JSON: {raw_arguments[:120]}")

    challenge = str(arguments.get("challenge", "")).strip()
    if not challenge:
        return (False, "Tool call missing required 'challenge' argument.")
    return (True, challenge[:120])


def run_image_test(llm: Any, test_image: Path, max_tokens: int) -> tuple[bool, str]:
    if not test_image.exists():
        return (False, f"Image test skipped: file not found ({test_image})")
    try:
        image_uri = test_image.resolve().as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, stream=False)
        text = str(out.get("choices", [{}])[0].get("message", {}).get("content", ""))
        if text.strip():
            return (True, text[:120])
        return (False, "Image response empty")
    except Exception as exc:
        return (False, str(exc))


def _resolve_backend_choice(requested: str, backend_hint: str) -> str:
    selected = (requested or "auto").strip().lower()
    if selected in {"vulkan", "cpu"}:
        return selected
    hint = (backend_hint or "").strip().lower()
    return "vulkan" if hint == "vulkan" else "cpu"


def _resolve_vision_handler(
    metadata: dict[str, Any],
    model_path: str,
    mmproj_path: str,
    model_hint: str = "",
) -> Any | None:
    """Build a model-appropriate MTMD chat handler for multimodal GGUF."""
    if not mmproj_path or not Path(mmproj_path).exists():
        return None
    try:
        from llama_cpp.llama_chat_format import (
            Gemma3ChatHandler,
            Gemma4ChatHandler,
            LFM2VLChatHandler,
            Llava15ChatHandler,
            MTMDChatHandler,
            MoondreamChatHandler,
            Qwen25VLChatHandler,
            Qwen3VLChatHandler,
        )
    except Exception:
        return None

    hints = [
        str(model_hint or ""),
        str(Path(model_path).name),
        str(metadata.get("general.architecture", "")),
        str(metadata.get("general.name", "")),
        str(metadata.get("general.basename", "")),
        str(metadata.get("tokenizer.chat_template", "")),
    ]
    tag = " ".join(hints).lower()

    handler_cls: Any = MTMDChatHandler
    if "gemma4" in tag or "gemma-4" in tag or "gemma 4" in tag:
        handler_cls = Gemma4ChatHandler
    elif "gemma" in tag:
        handler_cls = Gemma3ChatHandler
    elif "moondream" in tag:
        handler_cls = MoondreamChatHandler
    elif "llava" in tag:
        handler_cls = Llava15ChatHandler
    elif (
        "qwen3-vl" in tag
        or "qwen3 vl" in tag
        or ("qwen3" in tag and "vl" in tag)
    ):
        handler_cls = Qwen3VLChatHandler
    elif (
        "qwen2.5-vl" in tag
        or "qwen25vl" in tag
        or ("qwen2" in tag and "vl" in tag)
    ):
        handler_cls = Qwen25VLChatHandler
    elif (
        "lfm2-vl" in tag
        or "lfm2 vl" in tag
        or ("lfm2" in tag and "vl" in tag)
        or "lfm2vl" in tag
    ):
        handler_cls = LFM2VLChatHandler

    try:
        return handler_cls(clip_model_path=mmproj_path, verbose=False)
    except Exception:
        return None


def configure_vision_chat_handler(
    llm: Any,
    model_path: str,
    mmproj_path: str,
    model_hint: str = "",
) -> str:
    """Attach a vision chat handler to an existing Llama instance."""
    metadata = dict(getattr(llm, "metadata", {}) or {})
    handler = _resolve_vision_handler(
        metadata=metadata,
        model_path=model_path,
        mmproj_path=mmproj_path,
        model_hint=model_hint,
    )
    if handler is None:
        return ""
    llm.chat_handler = handler
    return type(handler).__name__


def run_one_model(
    args: argparse.Namespace,
    model: dict[str, Any],
    model_dir: Path,
    backend: str,
) -> TestResult:
    started = time.time()
    load_ok = raw_ok = chat_ok = image_ok = False
    image_skipped = False
    failure_type = ""
    failure_hint = ""
    error_text = ""
    load_s = raw_ttft = raw_total_s = chat_ttft = chat_total_s = 0.0
    raw_tokens = chat_tokens = 0
    raw_tps = chat_tps = 0.0
    size_gb = 0.0
    local_path_str = ""

    llm = None
    try:
        local_path, picked_filename, mmproj_path = download_model(
            model=model,
            model_dir=model_dir,
            skip_download=bool(args.skip_download),
            hf_token=str(args.hf_token or ""),
        )
        local_path_str = str(local_path)
        print(f"\n[{model['id']}] file={picked_filename}")

        if not local_path.exists():
            raise FileNotFoundError(f"Model file missing locally: {local_path}")
        size_gb = local_path.stat().st_size / (1024 ** 3)

        if backend == "vulkan":
            apply_vulkan_env_vars(
                model_size_gb=size_gb,
                vulkan_device_index=0,
                host_memory_override="auto",
            )

        ensure_ggml_backends()

        from llama_cpp import Llama

        effective_gpu_layers = int(args.gpu_layers)
        if backend == "cpu":
            effective_gpu_layers = 0
        elif effective_gpu_layers == 0:
            # Vulkan tests must actually offload to the configured backend.
            effective_gpu_layers = -1

        t0 = time.perf_counter()
        init_kwargs: dict[str, Any] = {
            "model_path": str(local_path),
            "n_gpu_layers": effective_gpu_layers,
            "n_ctx": int(args.ctx_length),
            "n_batch": int(args.n_batch),
            "n_threads": os.cpu_count() or 8,
            "n_threads_batch": os.cpu_count() or 8,
            "type_k": 8,
            "type_v": 8,
            "seed": 42,
            "verbose": False,
        }
        llm = Llama(
            **init_kwargs
        )
        if mmproj_path:
            handler_name = configure_vision_chat_handler(
                llm=llm,
                model_path=str(local_path),
                mmproj_path=mmproj_path,
                model_hint=str(model.get("family", "")),
            )
            if handler_name:
                print(f"  vision_handler={handler_name} mmproj={Path(mmproj_path).name}")
            else:
                print("  warning: mmproj present but vision handler was not attached")
        load_s = time.perf_counter() - t0
        load_ok = True

        raw_ok, raw_ttft, raw_total_s, raw_tokens, raw_tps, raw_msg = run_raw_completion(llm, args.max_tokens)
        if not raw_ok and not error_text:
            error_text = f"raw: {raw_msg}"

        chat_ok, chat_ttft, chat_total_s, chat_tokens, chat_tps, chat_msg = run_chat_completion(llm, args.max_tokens)
        if not chat_ok and not error_text:
            error_text = f"chat: {chat_msg}"

        if bool(model.get("multimodal")):
            if args.test_image:
                image_ok, image_msg = run_image_test(llm, Path(args.test_image), args.max_tokens)
                if not image_ok and not error_text:
                    error_text = f"image: {image_msg}"
            else:
                image_skipped = True
        else:
            image_skipped = True

    except Exception as exc:
        error_text = str(exc)
        failure_type, failure_hint = classify_error(exc)
    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        gc.collect()

    passed = load_ok and raw_ok and chat_ok and (image_ok or image_skipped)
    if not passed and not failure_type:
        failure_type, failure_hint = classify_error(error_text)

    return TestResult(
        model_id=str(model["id"]),
        model_name=str(model["name"]),
        family=str(model["family"]),
        tier=int(model["tier"]),
        quant=str(model["quant"]),
        local_path=local_path_str,
        size_gb=size_gb,
        passed=passed,
        load_ok=load_ok,
        chat_ok=chat_ok,
        raw_ok=raw_ok,
        image_ok=image_ok,
        image_test_skipped=image_skipped,
        failure_type=failure_type,
        failure_hint=failure_hint,
        error=error_text,
        load_s=load_s,
        raw_ttft_s=raw_ttft,
        raw_total_s=raw_total_s,
        raw_tokens=raw_tokens,
        raw_tps=raw_tps,
        chat_ttft_s=chat_ttft,
        chat_total_s=chat_total_s,
        chat_tokens=chat_tokens,
        chat_tps=chat_tps,
        timestamp=started,
    )


def print_summary(results: list[TestResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    tps_values = [r.chat_tps for r in results if r.chat_tps > 0]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
    min_tps = min(tps_values) if tps_values else 0.0
    max_tps = max(tps_values) if tps_values else 0.0

    print("\n=== Model Compatibility Summary ===")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    print(f"Chat TPS (min/avg/max): {min_tps:.2f} / {avg_tps:.2f} / {max_tps:.2f}")
    print("\nPer-model:")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"- {status} {r.model_id:<18} "
            f"load={r.load_s:>6.2f}s chat_tps={r.chat_tps:>6.2f} "
            f"raw_tps={r.raw_tps:>6.2f} error={r.failure_type or '-'}"
        )
        if not r.passed and r.failure_hint:
            print(f"  hint: {r.failure_hint}")
            if r.error:
                print(f"  err: {r.error[:300]}")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    hw = get_hardware_snapshot()
    print("=== Environment ===")
    print(f"OS: {hw.os_info}")
    print(f"GPU: {hw.gpu_name or 'Unknown'} ({hw.gpu_vendor.value}) VRAM={hw.vram_total_mb/1024:.1f} GB")
    print(f"Backend hint: {hw.ai_backend_hint.value}")
    configured_backend = _resolve_backend_choice(str(args.backend), hw.ai_backend_hint.value)
    print(f"Configured backend: {configured_backend}")
    print(f"llama.cpp pin: {LLAMACPP_RELEASE_TAG}  JamePeng pin: {JAMEPENG_PINNED_VERSION}")

    release_info = check_release_staleness()
    if release_info.get("latest_tag"):
        print(
            f"Latest llama.cpp release: {release_info['latest_tag']} "
            f"(delta={release_info.get('delta')})"
        )
    if release_info.get("warning"):
        print(f"WARNING: {release_info['warning']}")
    if release_info.get("error"):
        print(f"Release check error: {release_info['error']}")

    selected_models = select_models(args)
    if not selected_models:
        print("No models selected. Adjust --tier/--family filters.")
        return 1

    print(f"\nSelected models: {len(selected_models)}")
    for m in selected_models:
        src = f"{m['repo_id']}:{m['filename_pattern']}" if m.get("repo_id") else str(m.get("user_path"))
        print(f"- {m['id']}: {src}")

    results: list[TestResult] = []
    for model in selected_models:
        print(f"\n=== Testing {model['id']} ({model['name']}) ===")
        try:
            result = run_one_model(args, model, model_dir, configured_backend)
        except Exception as exc:
            failure_type, failure_hint = classify_error(exc)
            result = TestResult(
                model_id=str(model["id"]),
                model_name=str(model["name"]),
                family=str(model["family"]),
                tier=int(model["tier"]),
                quant=str(model["quant"]),
                local_path="",
                size_gb=0.0,
                passed=False,
                load_ok=False,
                chat_ok=False,
                raw_ok=False,
                image_ok=False,
                image_test_skipped=True,
                failure_type=failure_type,
                failure_hint=failure_hint,
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                load_s=0.0,
                raw_ttft_s=0.0,
                raw_total_s=0.0,
                raw_tokens=0,
                raw_tps=0.0,
                chat_ttft_s=0.0,
                chat_total_s=0.0,
                chat_tokens=0,
                chat_tps=0.0,
                timestamp=time.time(),
            )
        results.append(result)
        print(f"Result: {'PASS' if result.passed else 'FAIL'}")

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "args": vars(args),
        "release_check": release_info,
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nWrote report: {out_path}")

    return 0 if all(r.passed for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
