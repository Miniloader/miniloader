"""
basic_brain/logic.py — Vulkan LLM Inference Engine
====================================================
Loads .gguf model files via the JamePeng llama-cpp-python fork
(Vulkan backend) and runs stateless streaming inference.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

from core.base_module import BaseModule, ModuleStatus
from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    SignalType,
)

log = logging.getLogger(__name__)

_CACHE_TYPE_MAP: dict[str, int] = {
    "f32": 0,
    "f16": 1,
    "q4_0": 2,
    "q4_1": 3,
    "q8_0": 8,
}


class BasicBrainModule(BaseModule):
    MODULE_NAME = "basic_brain"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Local LLM inference engine powered by llama.cpp"
    PROCESS_ISOLATION = True
    UI_COL_SPAN = 4

    def get_default_params(self) -> dict[str, Any]:
        import os
        default_model_root = str(Path.home() / ".miniloader")
        physical_cores = os.cpu_count() or 4
        return {
            "model_path": "",
            "mmproj_path": "",
            "model_root": default_model_root,
            "model_roots": [],
            "gpu_layers": 0,
            "ctx_length": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.0,
            "repeat_penalty": 1.10,
            "n_batch": 512,
            "n_ubatch": 512,
            "cpu_threads": physical_cores,
            "flash_attn": "auto",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "seed": -1,
            "vulkan_device": 0,
            "vulkan_host_memory": "auto",
            "_vision_handler_name": "",
            "_active_mmproj_path": "",
        }

    # ── Ports ───────────────────────────────────────────────────

    def define_ports(self) -> None:
        self.add_output(
            "BRAIN_OUT",
            accepted_signals={
                SignalType.CHAT_REQUEST,
                SignalType.BRAIN_STREAM_PAYLOAD,
            },
            connection_mode=ConnectionMode.CHANNEL,
            description=(
                "Bidirectional AI channel (AI_OUT label). Receives a "
                "fully assembled prompt (CHAT_REQUEST) and "
                "streams generated tokens back "
                "(BRAIN_STREAM_PAYLOAD)."
            ),
        )

    async def _resolve_backend(self) -> str | None:
        """Resolve and validate backend selection without silent fallback."""
        valid_backends = {"cpu", "vulkan"}
        selected = str(self.params.get("_ai_backend", "")).strip().lower()
        detected = str(self.params.get("_ai_backend_detected", "")).strip().lower()

        if detected not in valid_backends:
            try:
                from core.hardware_probe import get_hardware_snapshot

                detected = get_hardware_snapshot().ai_backend_hint.value
            except Exception:
                detected = "unknown"
        self.params["_ai_backend_detected"] = detected

        if selected and selected not in valid_backends:
            msg = (
                f"Invalid stored backend '{selected}'. "
                "Open backend config and choose Vulkan or CPU."
            )
            self.params["_runtime_backend"] = "unconfigured"
            self.params["_runtime_backend_reason"] = msg
            await self.report_state(severity="WARN", message=msg)
            return None

        if not selected:
            recommendation = detected if detected in valid_backends else "vulkan"
            self.params["_backend_recommendation"] = recommendation
            msg = (
                "Backend is not configured for this account. "
                f"Recommended: {recommendation.upper()} based on detected hardware. "
                "Open backend config and choose Vulkan or CPU explicitly."
            )
            self.params["_runtime_backend"] = "unconfigured"
            self.params["_runtime_backend_reason"] = msg
            await self.report_state(severity="WARN", message=msg)
            return None

        self.params["_ai_backend"] = selected
        if detected in valid_backends and selected != detected:
            log.warning(
                "Backend mismatch: user selected %s while detection suggests %s. "
                "Honoring user selection.",
                selected,
                detected,
            )
        return selected

    # ── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        self._llm: Any = None
        self._model_loaded: bool = False
        self._loaded_model_path: str = ""
        self._inference_lock = asyncio.Lock()
        self._last_thread_id: str | None = None
        self.params["_runtime_backend"] = "idle"
        self.params["_runtime_backend_reason"] = "No model loaded"
        self.params["_gpu_offload_active"] = False
        self.params["_requested_gpu_layers"] = int(self.params.get("gpu_layers", 0))
        self.params["_model_load_error"] = ""
        self.params["_vision_handler_name"] = ""
        self.params["_active_mmproj_path"] = ""
        self.params["_trained_ctx"] = 0

        resolved_backend = await self._resolve_backend()
        if not resolved_backend:
            self.status = ModuleStatus.RUNNING
            return

        self._apply_backend_defaults()

        model_path = str(self.params.get("model_path", "")).strip()
        if not model_path or not Path(model_path).exists():
            await self.report_state(
                severity="WARN",
                message="No model selected — browse and LOAD to begin",
            )
            self.status = ModuleStatus.RUNNING
            return

        await self._load_model(model_path)
        self.status = ModuleStatus.RUNNING

    def _apply_backend_defaults(self) -> None:
        """One-shot Vulkan defaults with NVIDIA-specific tuning."""
        backend = self.params.get("_ai_backend", "cpu")
        if backend != "vulkan":
            return

        is_nvidia = False
        try:
            from core.hardware_probe import GpuVendor, get_hardware_snapshot
            is_nvidia = get_hardware_snapshot().gpu_vendor == GpuVendor.NVIDIA
        except Exception:
            pass

        if int(self.params.get("gpu_layers", 0)) == 0:
            self.params["gpu_layers"] = -1
            log.info("auto-set gpu_layers=-1 (vulkan — all layers to GPU)")

        if int(self.params.get("n_batch", 512)) < 512:
            self.params["n_batch"] = 512
            log.info("auto-set n_batch=512 (vulkan detected)")

        if is_nvidia and int(self.params.get("ctx_length", 4096)) == 4096:
            self.params["ctx_length"] = 2048
            log.info("auto-set ctx_length=2048 (NVIDIA Vulkan perf preset)")

        # Resolve "auto" flash_attn: OFF for NVIDIA (perf), ON for AMD/Intel.
        fa = self.params.get("flash_attn", "auto")
        if fa == "auto":
            resolved = not is_nvidia
            self.params["flash_attn"] = resolved
            log.info(
                "auto-resolved flash_attn=%s (NVIDIA=%s)",
                resolved, is_nvidia,
            )

        if str(self.params.get("cache_type_k", "f16")) == "f16":
            self.params["cache_type_k"] = "q8_0"
            log.info("auto-set cache_type_k=q8_0 (vulkan detected)")

        if str(self.params.get("cache_type_v", "f16")) == "f16":
            self.params["cache_type_v"] = "q8_0"
            log.info("auto-set cache_type_v=q8_0 (vulkan detected)")

    @staticmethod
    def _load_backends() -> None:
        """Load ggml backend plugins exactly once (Vulkan, CPU, RPC).

        Delegates to the centralised ``ensure_ggml_backends`` which
        handles DLL search-path setup and the bulk backend loader.
        Must be called before the first ``Llama()`` instantiation.
        """
        from core.hardware_probe import ensure_ggml_backends

        ensure_ggml_backends()

    @staticmethod
    def _enumerate_ggml_devices() -> list[str]:
        """Query the ggml backend registry for all visible compute devices.

        Returns a list of human-readable device descriptions (e.g.
        ``["Vulkan0 (AMD Radeon RX 7900 XTX)", "CPU"]``).  Falls back
        to the ggml_backend_reg API if the device API is unavailable.
        """
        import ctypes

        try:
            import llama_cpp._ggml as _ggml_mod
        except Exception:
            return []

        libggml = _ggml_mod.libggml
        names: list[str] = []

        # Strategy 1: ggml_backend_dev_* (newer builds)
        try:
            dev_count_fn = libggml.ggml_backend_dev_count
            dev_count_fn.restype = ctypes.c_size_t
            dev_get_fn = libggml.ggml_backend_dev_get
            dev_get_fn.argtypes = [ctypes.c_size_t]
            dev_get_fn.restype = ctypes.c_void_p
            dev_name_fn = libggml.ggml_backend_dev_name
            dev_name_fn.argtypes = [ctypes.c_void_p]
            dev_name_fn.restype = ctypes.c_char_p
            dev_desc_fn = libggml.ggml_backend_dev_description
            dev_desc_fn.argtypes = [ctypes.c_void_p]
            dev_desc_fn.restype = ctypes.c_char_p

            count = dev_count_fn()
            for i in range(count):
                dev = dev_get_fn(i)
                if not dev:
                    continue
                raw_name = dev_name_fn(dev)
                raw_desc = dev_desc_fn(dev)
                name = raw_name.decode("utf-8", errors="replace") if raw_name else "?"
                desc = raw_desc.decode("utf-8", errors="replace") if raw_desc else ""
                label = f"{name} ({desc})" if desc else name
                names.append(label)
                log.info("ggml device[%d]: %s", i, label)
        except (AttributeError, OSError):
            pass
        except Exception as exc:
            log.debug("ggml device enumeration failed: %s", exc)

        if names:
            return names

        # Strategy 2: ggml_backend_reg_* (available in JamePeng builds)
        try:
            reg_count_fn = libggml.ggml_backend_reg_count
            reg_count_fn.restype = ctypes.c_size_t
            reg_get_fn = libggml.ggml_backend_reg_get
            reg_get_fn.argtypes = [ctypes.c_size_t]
            reg_get_fn.restype = ctypes.c_void_p
            reg_name_fn = libggml.ggml_backend_reg_name
            reg_name_fn.argtypes = [ctypes.c_void_p]
            reg_name_fn.restype = ctypes.c_char_p

            count = reg_count_fn()
            for i in range(count):
                reg = reg_get_fn(i)
                if not reg:
                    continue
                raw_name = reg_name_fn(reg)
                name = raw_name.decode("utf-8", errors="replace") if raw_name else "?"
                names.append(name)
                log.info("ggml backend reg[%d]: %s", i, name)
        except (AttributeError, OSError):
            log.debug("ggml backend registry API not available in this build")
        except Exception as exc:
            log.debug("ggml backend registry enumeration failed: %s", exc)

        return names

    async def _load_model(self, model_path: str) -> None:
        await self.report_state(
            severity="INFO",
            message=f"Loading model: {Path(model_path).name}",
        )
        t0 = time.perf_counter()
        try:
            backend = self.params.get("_ai_backend", "cpu")

            if backend == "vulkan":
                model_size_gb = 0.0
                try:
                    model_size_gb = Path(model_path).stat().st_size / (1024 ** 3)
                except Exception:
                    pass
                from core.hardware_probe import (
                    apply_vulkan_env_vars,
                    format_vulkan_unavailable_message,
                )
                apply_vulkan_env_vars(
                    model_size_gb,
                    vulkan_device_index=int(self.params.get("vulkan_device", 0)),
                    host_memory_override=str(
                        self.params.get("vulkan_host_memory", "auto")
                    ),
                )

            self._load_backends()

            import llama_cpp  # type: ignore
            from llama_cpp import Llama

            seed_val = int(self.params.get("seed", -1))
            if seed_val < 0:
                seed_val = 0xFFFFFFFF

            cpu_threads = int(self.params.get("cpu_threads", 0)) or None

            n_batch = int(self.params.get("n_batch", 512))
            n_ubatch = min(int(self.params.get("n_ubatch", 512)), n_batch)

            init_kwargs: dict[str, Any] = {
                "model_path": model_path,
                "n_gpu_layers": int(self.params.get("gpu_layers", -1)),
                "n_ctx": int(self.params.get("ctx_length", 4096)),
                "n_batch": n_batch,
                "n_ubatch": n_ubatch,
                "n_threads": cpu_threads,
                "n_threads_batch": cpu_threads,
                "seed": seed_val,
                "verbose": False,
            }

            fa = self.params.get("flash_attn", "auto")
            if fa is True or (fa == "auto" and backend == "vulkan"):
                init_kwargs["flash_attn"] = True

            cache_k = str(self.params.get("cache_type_k", "f16"))
            cache_v = str(self.params.get("cache_type_v", "f16"))
            if cache_k in _CACHE_TYPE_MAP and cache_k != "f16":
                init_kwargs["type_k"] = _CACHE_TYPE_MAP[cache_k]
            if cache_v in _CACHE_TYPE_MAP and cache_v != "f16":
                init_kwargs["type_v"] = _CACHE_TYPE_MAP[cache_v]

            mmproj = str(self.params.get("mmproj_path", "")).strip()
            resolved_mmproj = ""
            if mmproj and Path(mmproj).exists():
                resolved_mmproj = mmproj
                self.params["_auto_mmproj_path"] = ""
            else:
                auto_mmproj = self._find_mmproj_for_model(model_path)
                if auto_mmproj:
                    resolved_mmproj = auto_mmproj
                    self.params["_auto_mmproj_path"] = auto_mmproj
                    log.info("Auto-detected mmproj for model: %s", auto_mmproj)
                else:
                    self.params["_auto_mmproj_path"] = ""
            self.params["_active_mmproj_path"] = resolved_mmproj
            self.params["_vision_handler_name"] = ""

            log.info(
                "Llama init kwargs: %s",
                {k: v for k, v in init_kwargs.items() if k != "model_path"},
            )

            try:
                self._llm = Llama(**init_kwargs)
            except TypeError:
                for opt_key in ("flash_attn", "type_k", "type_v", "n_ubatch"):
                    init_kwargs.pop(opt_key, None)
                log.warning(
                    "Llama() rejected perf kwargs (flash_attn/type_k/type_v/n_ubatch) "
                    "— retrying without them. Update llama-cpp-python for "
                    "full performance."
                )
                self._llm = Llama(**init_kwargs)

            if resolved_mmproj:
                metadata = dict(getattr(self._llm, "metadata", {}) or {})
                handler = self._resolve_vision_handler(
                    metadata=metadata,
                    model_path=model_path,
                    mmproj_path=resolved_mmproj,
                )
                if handler is None:
                    log.warning(
                        "mmproj configured but no vision chat handler could be created: %s",
                        resolved_mmproj,
                    )
                else:
                    self._patch_mtmd_template_globals(handler)
                    self._llm.chat_handler = handler
                    handler_name = type(handler).__name__
                    self.params["_vision_handler_name"] = handler_name
                    log.info(
                        "Vision handler active: %s (mmproj: %s)",
                        handler_name,
                        resolved_mmproj,
                    )
                    # Gemma 4 vision uses non-causal attention for image
                    # tokens — n_ubatch must be >= the image token budget
                    # (max 1120).  Bump if needed so images don't crash.
                    if handler_name == "Gemma4ChatHandler":
                        min_ubatch = 1120
                        cur_ubatch = int(
                            init_kwargs.get("n_ubatch", 512)
                        )
                        if cur_ubatch < min_ubatch:
                            self.params["n_ubatch"] = min_ubatch
                            log.info(
                                "auto-bumped n_ubatch %d->%d "
                                "(Gemma 4 vision non-causal attention)",
                                cur_ubatch, min_ubatch,
                            )
            self._extract_model_metadata()

            # ── Device enumeration diagnostic ─────────────────────
            device_names = self._enumerate_ggml_devices()
            self.params["_ggml_devices"] = device_names
            if device_names:
                log.info("ggml devices visible: %s", device_names)
            else:
                log.debug(
                    "ggml device enumeration API not available in this "
                    "build — Vulkan device info comes from plugin load log"
                )

            requested_gpu_layers = int(self.params.get("gpu_layers", -1))

            effective_n_gpu = 0
            try:
                effective_n_gpu = int(self._llm.model_params.n_gpu_layers)
            except Exception:
                pass
            self.params["_effective_n_gpu_layers"] = effective_n_gpu

            supports_gpu_offload = False
            try:
                supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", None)
                if callable(supports_gpu):
                    supports_gpu_offload = bool(supports_gpu())
            except Exception:
                pass

            gpu_was_requested = backend == "vulkan" and requested_gpu_layers != 0

            gpu_working = (
                gpu_was_requested
                and effective_n_gpu > 0
                and supports_gpu_offload
            )

            # Cross-check: if we think Vulkan is working but no Vulkan
            # device appeared in the device list, flag it.
            has_vulkan_device = any(
                "vulkan" in d.lower() or "radeon" in d.lower() or "amd" in d.lower()
                for d in device_names
            )
            if gpu_working and not has_vulkan_device and device_names:
                log.warning(
                    "GPU offload checks passed but no Vulkan/AMD device in "
                    "ggml device list: %s — inference may actually be on CPU!",
                    device_names,
                )

            if gpu_working:
                runtime_backend = "vulkan"
            elif gpu_was_requested:
                runtime_backend = "cpu_fallback"
            else:
                runtime_backend = "cpu"

            self.params["_requested_gpu_layers"] = requested_gpu_layers
            self.params["_gpu_offload_active"] = gpu_working
            self.params["_runtime_backend"] = runtime_backend

            layers_label = "ALL" if effective_n_gpu >= 0x7FFFFFFF else str(effective_n_gpu)
            if gpu_working:
                reason = f"Vulkan GPU offload active ({layers_label} layers)"
                if has_vulkan_device:
                    vk_devs = [d for d in device_names if "vulkan" in d.lower() or "radeon" in d.lower() or "amd" in d.lower()]
                    reason += f" — device(s): {', '.join(vk_devs)}"
            elif not gpu_was_requested:
                reason = "CPU mode"
            else:
                parts = []
                if not supports_gpu_offload:
                    parts.append("build lacks GPU offload")
                if effective_n_gpu == 0:
                    parts.append("effective gpu_layers=0")
                detail = "; ".join(parts) if parts else "unknown cause"
                reason = f"Vulkan GPU offload unavailable ({detail})"
            self.params["_runtime_backend_reason"] = reason

            # Vulkan was selected but GPU offload failed — refuse to run
            # on CPU silently.  The user must fix the Vulkan setup or
            # explicitly switch to the CPU backend via the config button.
            if runtime_backend == "cpu_fallback":
                try:
                    self._llm.close()
                except Exception:
                    pass
                self._llm = None
                gc.collect()
                self._model_loaded = False
                self._loaded_model_path = ""

                fail_msg = format_vulkan_unavailable_message(
                    reason,
                    prefix="Vulkan GPU offload failed — model NOT loaded",
                )
                fail_msg += (
                    "\n\nIf you want to run on CPU instead, change the AI "
                    "backend to CPU in Settings."
                )
                self.params["_model_load_error"] = fail_msg
                self.params["_runtime_backend"] = "error"
                log.error("VULKAN FAILED: %s", fail_msg)
                await self.report_state(severity="ERROR", message=fail_msg)
                raise RuntimeError(fail_msg)

            self._model_loaded = True
            self._loaded_model_path = model_path

            # VRAM pressure check — warn when model + KV overhead likely
            # spills to system RAM, causing severe TPS degradation.
            try:
                from core.hardware_probe import get_hardware_snapshot
                hw = get_hardware_snapshot()
                vram_gb = float(hw.vram_total_mb) / 1024.0
                model_size_gb = 0.0
                try:
                    model_size_gb = Path(model_path).stat().st_size / (1024 ** 3)
                except Exception:
                    pass
                if (
                    vram_gb > 0
                    and model_size_gb > 0
                    and backend == "vulkan"
                    and model_size_gb > vram_gb * 0.85
                ):
                    headroom_pct = (1.0 - model_size_gb / vram_gb) * 100
                    log.warning(
                        "VRAM pressure: model %.1f GB vs %.1f GB VRAM "
                        "(%.0f%% headroom) — KV cache + context overhead "
                        "will likely spill to system RAM, severely reducing "
                        "inference speed. Consider a smaller quant or model.",
                        model_size_gb, vram_gb, headroom_pct,
                    )
            except Exception:
                pass

            try:
                from core.llama_runtime import (
                    get_llama_package_metadata,
                    is_jamepeng_distribution,
                )
                _meta = get_llama_package_metadata()
                _fork = is_jamepeng_distribution(_meta)
                self.params["_llama_package_fork"] = (
                    "jamepeng" if _fork is True
                    else "other" if _fork is False
                    else "unknown"
                )
                if _fork is False:
                    log.warning(
                        "Installed llama_cpp_python (%s) does not appear to be "
                        "the JamePeng fork — Vulkan support may be degraded. "
                        "Run  python scripts/setup_environment.py  to fix.",
                        _meta.get("version", "?"),
                    )
            except Exception:
                self.params["_llama_package_fork"] = "unknown"

            total_s = time.perf_counter() - t0
            fa_active = "flash_attn" in init_kwargs
            cache_k_label = str(self.params.get("cache_type_k", "f16"))
            cache_v_label = str(self.params.get("cache_type_v", "f16"))
            n_batch_actual = int(self.params.get("n_batch", 512))
            n_ubatch_actual = int(self.params.get("n_ubatch", 512))
            model_family = str(self.params.get("_model_family", "unknown"))
            devices_short = ", ".join(device_names[:3]) if device_names else "none detected"
            load_summary = (
                f"Model loaded: {Path(model_path).name} ({total_s:.2f}s) "
                f"[family={model_family} "
                f"runtime={runtime_backend.upper()} "
                f"gpu_layers={requested_gpu_layers} "
                f"effective={layers_label} "
                f"gpu_offload={'yes' if supports_gpu_offload else 'no'} "
                f"flash_attn={'yes' if fa_active else 'no'} "
                f"kv_cache={cache_k_label}/{cache_v_label} "
                f"n_batch={n_batch_actual} n_ubatch={n_ubatch_actual} "
                f"devices=({devices_short})]"
            )
            await self.report_state(severity="INFO", message=load_summary)
            log.info(load_summary)
        except Exception as exc:
            self._model_loaded = False
            self._loaded_model_path = ""
            self.params["_vision_handler_name"] = ""
            self.params["_active_mmproj_path"] = ""
            self.params["_trained_ctx"] = 0
            detail = self._build_load_error_detail(model_path, exc)
            self.params["_model_load_error"] = detail
            self.params["_runtime_backend"] = "error"
            self.params["_runtime_backend_reason"] = detail
            log.error("basic_brain model load failed:\n%s", detail)
            await self.report_state(severity="ERROR", message=detail)
            raise RuntimeError(detail) from exc

    @staticmethod
    def _coerce_ctx_value(raw: Any) -> int:
        """Best-effort conversion for context-length metadata values."""
        if raw is None:
            return 0
        if isinstance(raw, bool):
            return 0
        if isinstance(raw, (int, float)):
            try:
                return int(raw)
            except Exception:
                return 0
        text = str(raw).strip()
        if not text:
            return 0
        try:
            return int(text)
        except Exception:
            return 0

    @classmethod
    def _trained_ctx_from_metadata(cls, metadata: dict[str, Any]) -> int:
        """Extract trained/max context size from GGUF metadata."""
        if not metadata:
            return 0
        for key in (
            "llama.context_length",
            "general.context_length",
            "n_ctx_train",
            "context_length",
        ):
            value = cls._coerce_ctx_value(metadata.get(key))
            if value > 0:
                return value
        return 0

    @staticmethod
    def _find_mmproj_for_model(model_path: str) -> str:
        """Best-effort mmproj auto-discovery for vision-language GGUF models.

        Searches near the model file and in parent snapshot/cache directories.
        Prefers F16, then BF16, then any mmproj*.gguf.
        """
        model_file = Path(model_path)
        if not model_file.exists():
            return ""

        search_dirs: list[Path] = []
        for candidate in (
            model_file.parent,
            model_file.parent.parent,
            model_file.parent.parent.parent,
        ):
            if candidate and candidate.is_dir() and candidate not in search_dirs:
                search_dirs.append(candidate)

        candidates: list[Path] = []
        for d in search_dirs:
            try:
                for p in d.glob("*mmproj*.gguf"):
                    if p.is_file():
                        candidates.append(p)
            except Exception:
                continue

        if not candidates:
            return ""

        def _rank(p: Path) -> tuple[int, int, str]:
            name = p.name.lower()
            if "f16" in name:
                pref = 0
            elif "bf16" in name:
                pref = 1
            else:
                pref = 2
            # Prefer same directory as model when possible.
            near = 0 if p.parent == model_file.parent else 1
            return (pref, near, name)

        candidates.sort(key=_rank)
        return str(candidates[0])

    @staticmethod
    def _patch_mtmd_template_globals(handler: Any) -> None:
        """Inject raise_exception/strftime_now into an MTMDChatHandler.

        llama-cpp-python's MTMDChatHandler._process_mtmd_prompt renders the
        Jinja2 chat template via ``**self.extra_template_arguments`` but never
        populates raise_exception or strftime_now — both of which many GGUF
        chat templates call.  The non-vision AutoChatFormatter does pass them.
        """
        import datetime

        extra = getattr(handler, "extra_template_arguments", None)
        if not isinstance(extra, dict):
            extra = {}
            handler.extra_template_arguments = extra

        def _raise_exception(message: str) -> None:
            raise ValueError(message)

        def _strftime_now(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
            return datetime.datetime.now().strftime(fmt)

        extra.setdefault("raise_exception", _raise_exception)
        extra.setdefault("strftime_now", _strftime_now)

    @staticmethod
    def _resolve_vision_handler(
        metadata: dict[str, Any],
        model_path: str,
        mmproj_path: str,
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

    def _build_load_error_detail(self, model_path: str, exc: Exception) -> str:
        backend = str(self.params.get("_ai_backend", "cpu")).strip().lower() or "cpu"
        gpu_layers = int(self.params.get("gpu_layers", -1) or -1)
        ctx_length = int(self.params.get("ctx_length", 4096) or 4096)
        n_batch = int(self.params.get("n_batch", 256) or 256)

        size_mb = 0.0
        try:
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        except Exception:
            pass

        hint = ""
        exc_text = str(exc).strip()
        lower = exc_text.lower()
        if "unknown model architecture" in lower:
            hint = (
                "Hint: the installed llama-cpp-python build does not recognise "
                "this model's architecture. Re-install the JamePeng wheel to "
                "get the latest model support."
            )
        elif backend == "vulkan":
            hint = (
                "Hint: Vulkan backend failures usually mean the Vulkan "
                "driver/runtime is missing or outdated. Update your GPU "
                "drivers and re-run Repair Vulkan Setup."
            )
        elif "failed to load model from file" in lower:
            hint = (
                "Hint: the GGUF file may be invalid/corrupt, or the "
                "context/offload settings are too aggressive."
            )
        elif any(token in lower for token in ["out of memory", "alloc"]):
            hint = "Hint: try reducing gpu_layers or ctx_length."

        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        detail_lines = [
            f"Model load failed: {Path(model_path).name}",
            f"Cause: {tb or repr(exc)}",
            f"Backend={backend.upper()} gpu_layers={gpu_layers} ctx={ctx_length} n_batch={n_batch}",
        ]
        if size_mb > 0:
            detail_lines.append(f"Model file: {Path(model_path)} (~{size_mb:.1f} MB)")
        if hint:
            detail_lines.append(hint)
        return " | ".join(detail_lines)

    @staticmethod
    def _detect_model_family(metadata: dict[str, Any], model_path: str) -> str:
        """Derive a short model-family tag from GGUF metadata + filename."""
        hints = " ".join([
            str(Path(model_path).name),
            str(metadata.get("general.architecture", "")),
            str(metadata.get("general.name", "")),
            str(metadata.get("general.basename", "")),
        ]).lower()
        if "gemma4" in hints or "gemma-4" in hints or "gemma 4" in hints:
            return "gemma4"
        if "gemma" in hints:
            return "gemma3"
        if "qwen3" in hints:
            return "qwen3"
        if "qwen" in hints:
            return "qwen"
        if "lfm2" in hints or "lfm 2" in hints or "lfm-2" in hints:
            return "lfm2"
        if "lfm" in hints:
            return "lfm"
        if "llama" in hints:
            return "llama"
        if "phi" in hints:
            return "phi"
        if "mistral" in hints:
            return "mistral"
        return "unknown"

    # Per-family sampling recommendations derived from benchmark data and
    # vendor documentation.  Keys that match get_default_params() values
    # are auto-applied only when the user hasn't changed the default.
    _FAMILY_SAMPLING: dict[str, dict[str, float]] = {
        "lfm": {
            "temperature": 0.3,
            "min_p": 0.15,
            "repeat_penalty": 1.05,
            "top_k": 50,
        },
        "lfm2": {
            "temperature": 0.3,
            "min_p": 0.15,
            "repeat_penalty": 1.05,
            "top_k": 50,
        },
    }

    def _apply_model_family_tuning(self, family: str) -> None:
        """Auto-tune sampling params when the current values still match
        module defaults.  Avoids overriding anything the user explicitly set."""
        overrides = self._FAMILY_SAMPLING.get(family)
        if not overrides:
            return
        defaults = self.get_default_params()
        applied: list[str] = []
        for key, recommended in overrides.items():
            current = self.params.get(key)
            default = defaults.get(key)
            if current is not None and default is not None:
                try:
                    if abs(float(current) - float(default)) < 1e-6:
                        self.params[key] = recommended
                        applied.append(f"{key}={recommended}")
                except (TypeError, ValueError):
                    pass
        if applied:
            log.info(
                "auto-tuned sampling for family=%s: %s",
                family, ", ".join(applied),
            )

    def _extract_model_metadata(self) -> None:
        """Read native context length and chat template info from loaded model."""
        if self._llm is None:
            return
        metadata = dict(getattr(self._llm, "metadata", {}) or {})
        try:
            native_ctx = self._llm.n_ctx()
            self.params["_native_ctx"] = native_ctx
        except Exception:
            self.params["_native_ctx"] = 0
        self.params["_trained_ctx"] = self._trained_ctx_from_metadata(metadata)

        family = self._detect_model_family(
            metadata, str(self.params.get("model_path", "")),
        )
        self.params["_model_family"] = family
        self.params["_is_gemma4"] = family == "gemma4"

        self._apply_model_family_tuning(family)

        try:
            import llama_cpp
            model = self._llm.model
            count = llama_cpp.llama_model_meta_count(model)
            has_template = False
            for i in range(count):
                buf = b"\x00" * 256
                llama_cpp.llama_model_meta_key_by_index(model, i, buf, len(buf))
                key = buf.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                if "chat_template" in key:
                    has_template = True
                    break
            self.params["_tool_use"] = has_template
        except Exception:
            self.params["_tool_use"] = False

    async def process(self, payload: Payload, source_port: Port) -> None:
        if payload.signal_type == SignalType.CHAT_REQUEST:
            request_id = payload.data.get("request_id", payload.id)
            log.info(
                "basic_brain: received CHAT_REQUEST request_id=%s on port=%s",
                request_id, source_port.name,
            )
            await self.report_state(
                severity="INFO",
                message=f"Request received: {request_id}",
            )
            async with self._inference_lock:
                await self._run_inference(payload)

    @staticmethod
    def _merge_system_messages(messages: list[dict]) -> list[dict]:
        """Fold system messages into the first user message for models
        whose chat template does not support the system role (e.g. Gemma)."""
        def _as_text(content: Any) -> str:
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                return "\n".join(parts).strip()
            return str(content or "").strip()

        system_parts: list[str] = []
        other: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                text = _as_text(m.get("content"))
                if text:
                    system_parts.append(text)
            else:
                other.append(m)

        if not system_parts:
            return other

        prefix = "\n\n".join(system_parts)
        for i, m in enumerate(other):
            if m.get("role") == "user":
                merged = dict(m)
                content = m.get("content", "")
                if isinstance(content, list):
                    merged["content"] = [{"type": "text", "text": prefix}, *content]
                else:
                    merged["content"] = f"{prefix}\n\n{content}"
                other[i] = merged
                return other

        other.insert(0, {"role": "user", "content": prefix})
        return other

    @staticmethod
    def _ensure_starts_with_user(messages: list[dict]) -> list[dict]:
        """Guarantee the first non-system message is a user turn.

        After system messages have been merged away, some chat templates
        (Gemma, Llama-Instruct) require the conversation to begin with a
        user turn.  If the sequence starts with an assistant message
        (common when Discord channel history begins with a bot reply),
        insert a minimal user stub before it.
        """
        if not messages:
            return messages
        first_conv = 0
        for i, m in enumerate(messages):
            if m.get("role") != "system":
                first_conv = i
                break
        else:
            return messages
        if messages[first_conv].get("role") != "assistant":
            return messages
        result = list(messages)
        result.insert(first_conv, {"role": "user", "content": "."})
        return result

    @staticmethod
    def _coalesce_consecutive_roles(messages: list[dict]) -> list[dict]:
        """Merge consecutive messages that share the same role so that the
        sequence strictly alternates user/assistant (required by some Jinja
        chat templates like Gemma and Llama-Instruct)."""
        if not messages:
            return messages

        def _join_content(a: Any, b: Any) -> Any:
            """Concatenate two content values (string or multimodal list)."""
            if isinstance(a, list) and isinstance(b, list):
                return a + b
            if isinstance(a, list):
                return a + [{"type": "text", "text": str(b or "")}]
            if isinstance(b, list):
                return [{"type": "text", "text": str(a or "")}] + b
            return f"{a}\n\n{b}"

        result: list[dict] = [dict(messages[0])]
        for msg in messages[1:]:
            if msg.get("role") == result[-1].get("role"):
                result[-1] = dict(result[-1])
                result[-1]["content"] = _join_content(
                    result[-1].get("content", ""),
                    msg.get("content", ""),
                )
            else:
                result.append(dict(msg))
        return result

    @staticmethod
    def _strip_old_images(messages: list[dict]) -> list[dict]:
        """Keep image inputs only in the latest user turn.

        Older user-turn images are replaced with a text placeholder so prior
        conversational context remains available without re-sending media.
        """
        last_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_idx = idx
                break

        if last_user_idx < 0:
            return messages

        stripped: list[dict] = []
        for idx, msg in enumerate(messages):
            if idx == last_user_idx or msg.get("role") != "user":
                stripped.append(dict(msg))
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                stripped.append(dict(msg))
                continue

            new_content: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    new_content.append({"type": "text", "text": "[image]"})
                else:
                    new_content.append(part)

            updated = dict(msg)
            updated["content"] = new_content
            stripped.append(updated)

        return stripped

    @staticmethod
    def _strip_all_images(messages: list[dict]) -> list[dict]:
        """Remove every image_url part from all messages, replacing with a
        placeholder.  Used when no vision handler (mmproj) is loaded so raw
        base64 data never reaches the tokenizer."""
        stripped: list[dict] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                stripped.append(dict(msg))
                continue
            new_content: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    new_content.append({"type": "text", "text": "[image omitted — no vision model loaded]"})
                else:
                    new_content.append(part)
            updated = dict(msg)
            updated["content"] = new_content
            stripped.append(updated)
        return stripped

    @staticmethod
    def _has_images(messages: list[dict]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
        return False

    @staticmethod
    def _flatten_tool_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert ``tool`` role messages and ``tool_calls`` assistant
        messages into plain user/assistant text so templates that only
        understand user/assistant/system can render them."""
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()

            if role == "assistant" and msg.get("tool_calls"):
                calls = msg["tool_calls"]
                content = str(msg.get("content", "") or "").strip()
                parts: list[str] = []
                if content:
                    parts.append(content)
                for c in (calls if isinstance(calls, list) else []):
                    fn = (c.get("function") or {}) if isinstance(c, dict) else {}
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "{}")
                    parts.append(f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
                out.append({"role": "assistant", "content": "\n".join(parts)})

            elif role == "tool":
                name = str(msg.get("name", "tool"))
                body = str(msg.get("content", "") or "")
                out.append({
                    "role": "user",
                    "content": (
                        f'[System: tool "{name}" returned the following data. '
                        "Summarize it for the user in natural language; do not call "
                        "tools again unless the user asks for something new.]\n"
                        f"{body}"
                    ),
                })

            else:
                out.append(msg)
        return out

    @staticmethod
    def _build_tool_prompt(tools: list[dict[str, Any]]) -> str:
        """Build a system-level tool description when the model's Jinja
        template does not natively render the ``tools`` variable."""
        lines = [
            "You have access to these tools (use them only when the user needs "
            "external data such as calendar, email, or other connected services):",
            "",
        ]
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))
            sig_parts: list[str] = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                marker = " (required)" if pname in required else ""
                sig_parts.append(f"{pname}: {ptype}{marker}")
            sig = ", ".join(sig_parts) if sig_parts else ""
            lines.append(f"- {name}({sig}): {desc}")
        lines.extend(
            [
                "",
                "Rules:",
                "- Do not call tools for greetings, small talk, or questions you can "
                "answer without external data.",
                "- Call a tool only when the user clearly needs information or an "
                "action from that service.",
                "- When you call a tool, respond with only:",
                "<tool_call>",
                '{"name": "tool_name", "arguments": {"param": "value"}}',
                "</tool_call>",
                "",
                "When a message shows tool output with returned data, read it and "
                "summarize the useful parts for the user in natural language.",
                "Do not call the same tool again with the same arguments after you "
                "already received a result for that call.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls_from_text(
        text: str, available_tools: list[dict[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], str]:
        """Parse tool calls from model text while preserving surrounding prose."""
        raw = str(text or "")
        if not raw.strip():
            return [], raw

        valid_names: set[str] = set()
        if isinstance(available_tools, list):
            for tool in available_tools:
                if not isinstance(tool, dict):
                    continue
                fn = tool.get("function")
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name", "")).strip().lower()
                if name:
                    valid_names.add(name)

        def _normalize_arguments(value: Any) -> str:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return "{}"
                try:
                    parsed = json.loads(value)
                    return json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    return value
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            if value is None:
                return "{}"
            return str(value)

        def _candidate_from_obj(obj: Any) -> tuple[str, str] | None:
            if not isinstance(obj, dict):
                return None
            name = ""
            arguments: Any = {}
            if "name" in obj and "arguments" in obj:
                name = str(obj.get("name", "")).strip().lower()
                arguments = obj.get("arguments")
            else:
                fn = obj.get("function")
                if isinstance(fn, dict):
                    name = str(fn.get("name", "")).strip().lower()
                    arguments = fn.get("arguments")
            if not name:
                return None
            if valid_names and name not in valid_names:
                return None
            return name, _normalize_arguments(arguments)

        parsed_calls: list[dict[str, Any]] = []
        spans_to_strip: list[tuple[int, int]] = []

        # Tier 1: Gemma 4 native tool-call tokens.
        # Format: <|tool_call>call:func_name{key:<|"|>val<|"|>}<tool_call|>
        # The opening tag may appear as <|tool_call|> or <|tool_call> depending
        # on GGUF revision / chat-template version.
        _GEMMA4_TC = re.compile(
            r"<\|tool_call\|?>\s*call:([a-zA-Z0-9_]+)\s*(\{.*?\})\s*(?:<tool_call\|>)?",
            flags=re.DOTALL,
        )
        for m in _GEMMA4_TC.finditer(raw):
            name = str(m.group(1) or "").strip().lower()
            # Gemma 4 uses <|"|> as escaped quote inside its non-JSON syntax;
            # normalise to standard JSON quotes before parsing.
            args_blob = (m.group(2) or "").replace('<|"|>', '"')
            if valid_names and name not in valid_names:
                continue
            try:
                args_obj = json.loads(args_blob)
            except Exception:
                continue
            parsed_calls.append(
                {
                    "id": f"call_{len(parsed_calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args_obj, ensure_ascii=False),
                    },
                }
            )
            spans_to_strip.append((m.start(), m.end()))

        # Tier 2: XML-wrapped <tool_call> blocks (closing tag optional —
        # </tool_call> may have been consumed as a stop token).
        for m in re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>)?",
            raw,
            flags=re.DOTALL,
        ):
            blob = str(m.group(1) or "").strip()
            try:
                parsed = json.loads(blob)
            except Exception:
                continue
            call = _candidate_from_obj(parsed)
            if call is None:
                continue
            name, args = call
            parsed_calls.append(
                {
                    "id": f"call_{len(parsed_calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            )
            spans_to_strip.append((m.start(), m.end()))

        # Tiers 3/4: Entire payload is a plain object/array.
        whole_text = raw.strip()
        if not parsed_calls and whole_text:
            try:
                whole = json.loads(whole_text)
            except Exception:
                whole = None
            if isinstance(whole, dict):
                call = _candidate_from_obj(whole)
                if call is not None:
                    name, args = call
                    parsed_calls.append(
                        {
                            "id": f"call_{len(parsed_calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    )
                    return parsed_calls, ""
            elif isinstance(whole, list):
                for item in whole:
                    call = _candidate_from_obj(item)
                    if call is None:
                        continue
                    name, args = call
                    parsed_calls.append(
                        {
                            "id": f"call_{len(parsed_calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    )
                if parsed_calls:
                    return parsed_calls, ""

        # Tier 5: Embedded JSON objects in mixed content.
        if not parsed_calls:
            depth = 0
            in_string = False
            escape = False
            start_idx = -1
            for idx, ch in enumerate(raw):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    if depth == 0:
                        start_idx = idx
                    depth += 1
                    continue
                if ch == "}":
                    if depth <= 0:
                        continue
                    depth -= 1
                    if depth == 0 and start_idx >= 0:
                        candidate_text = raw[start_idx : idx + 1]
                        try:
                            obj = json.loads(candidate_text)
                        except Exception:
                            start_idx = -1
                            continue
                        call = _candidate_from_obj(obj)
                        if call is not None:
                            name, args = call
                            parsed_calls.append(
                                {
                                    "id": f"call_{len(parsed_calls)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": args},
                                }
                            )
                            spans_to_strip.append((start_idx, idx + 1))
                        start_idx = -1

        if not parsed_calls:
            return [], raw

        if not spans_to_strip:
            return parsed_calls, raw.strip()

        spans_to_strip.sort(key=lambda it: it[0])
        merged: list[tuple[int, int]] = []
        for start, end in spans_to_strip:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        chunks: list[str] = []
        cursor = 0
        for start, end in merged:
            if start > cursor:
                chunks.append(raw[cursor:start])
            cursor = max(cursor, end)
        if cursor < len(raw):
            chunks.append(raw[cursor:])

        cleaned = "".join(chunks)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return parsed_calls, cleaned.strip()

    def _estimate_message_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for a list of chat messages.

        Uses the model's tokenizer for text content. Image parts are counted
        as a fixed embedding cost (~600 tokens) since the vision handler
        processes them through the mmproj rather than tokenizing raw base64.
        """
        IMAGE_EMBED_TOKENS = 600
        image_count = 0
        text_parts: list[str] = []
        for msg in messages:
            text_parts.append(msg.get("role", ""))
            content = msg.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            image_count += 1
                    elif isinstance(part, str):
                        text_parts.append(part)

        combined = " ".join(text_parts)
        try:
            tokens = self._llm.tokenize(combined.encode("utf-8", errors="replace"),
                                        add_bos=False)
            text_tokens = len(tokens)
        except Exception:
            text_tokens = len(combined) // 3

        return text_tokens + (image_count * IMAGE_EMBED_TOKENS)

    def _trim_messages_to_fit(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict] | None:
        """Drop older middle messages until the prompt fits *max_tokens*.

        Preserves the system prompt (index 0 if role == 'system') and the
        most recent user message.  Returns ``None`` if even the minimal
        set exceeds the limit (caller should return an error to the user).
        """
        if not messages:
            return messages

        est = self._estimate_message_tokens(messages)
        if est <= max_tokens:
            return messages

        keep_front: list[dict] = []
        start = 0
        if messages[0].get("role") == "system":
            keep_front.append(messages[0])
            start = 1

        keep_back = [messages[-1]] if len(messages) > start else []

        minimal = keep_front + keep_back
        if self._estimate_message_tokens(minimal) > max_tokens:
            return None

        middle = messages[start: -1 if keep_back else len(messages)]
        result = keep_front + middle + keep_back
        while self._estimate_message_tokens(result) > max_tokens and middle:
            middle.pop(0)
            result = keep_front + middle + keep_back

        return result

    async def _run_inference(self, payload: Payload) -> None:
        request_id = payload.data.get("request_id", payload.id)
        if self._llm is None:
            await self.report_state(
                severity="ERROR",
                message=f"Inference failed ({request_id}): No model loaded",
            )
            await self.outputs["BRAIN_OUT"].emit(
                Payload(
                    signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                    source_module=self.module_id,
                    data={
                        "request_id": request_id,
                        "error": "No model loaded",
                        "token": "",
                        "done": True,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "ttft_s": 0.0,
                        "total_s": 0.0,
                    },
                )
            )
            return

        messages = payload.data.get("messages", [])
        if not messages:
            prompt_text = payload.data.get("prompt", "")
            if prompt_text:
                messages = [{"role": "user", "content": prompt_text}]
            else:
                await self.outputs["BRAIN_OUT"].emit(
                    Payload(
                        signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                        source_module=self.module_id,
                        data={
                            "request_id": request_id,
                            "error": "No prompt/messages provided",
                            "token": "",
                            "done": True,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "ttft_s": 0.0,
                            "total_s": 0.0,
                        },
                    )
                )
                return

        thread_id = payload.data.get("thread_id")
        raw_tools = payload.data.get("tools")
        tools: list[dict[str, Any]] | None = None
        if isinstance(raw_tools, list):
            tools = [row for row in raw_tools if isinstance(row, dict)]
        tool_choice = payload.data.get("tool_choice")
        if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
            tool_choice = None

        has_images = self._has_images(messages)

        vision_active = getattr(self._llm, "chat_handler", None) is not None
        if has_images and not vision_active:
            log.warning(
                "basic_brain: images in request but no vision handler loaded "
                "(no mmproj) — stripping image data request_id=%s",
                request_id,
            )
            messages = self._strip_all_images(messages)
            has_images = False

        # Clear KV cache when switching to a different conversation thread, when
        # thread_id is None (we can't assume continuity), or when a turn contains
        # image inputs (avoid stale multimodal state and BOS/add_special mismatch).
        if thread_id is None or thread_id != self._last_thread_id or has_images:
            import llama_cpp

            self._llm.reset()
            llama_cpp.llama_memory_clear(
                llama_cpp.llama_get_memory(self._llm.ctx),
                True,
            )
            self._last_thread_id = thread_id
            if has_images:
                log.debug(
                    "basic_brain: cleared KV cache for image turn request_id=%s thread_id=%s",
                    request_id, thread_id,
                )
            elif thread_id is not None:
                log.debug(
                    "basic_brain: cleared KV cache for new thread request_id=%s thread_id=%s",
                    request_id, thread_id,
                )

        log.info(
            "basic_brain: starting inference request_id=%s messages=%d",
            request_id, len(messages),
        )
        await self.report_state(
            severity="INFO",
            message=f"Inference started: {request_id}",
        )

        t0 = time.perf_counter()
        ttft: float | None = None
        full_response: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        first_token_logged = False

        # Accumulate tokens and flush in batches to amortize IPC cost.
        # First token is always sent immediately for responsiveness.
        _TOKEN_FLUSH_INTERVAL = 0.05  # seconds between IPC flushes
        _pending_tokens: list[str] = []
        _last_flush = 0.0
        _tool_suppress_buffer: list[str] = []
        _suppressing_tool_tokens = False
        _tool_detection_tail = ""
        tool_call_parts: dict[int, dict[str, Any]] = {}

        def _merge_tool_call_delta(entry: dict[str, Any]) -> None:
            idx_raw = entry.get("index", 0)
            try:
                idx = int(idx_raw)
            except Exception:
                idx = 0
            current = tool_call_parts.setdefault(
                idx,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call_id = entry.get("id")
            if isinstance(call_id, str) and call_id:
                current["id"] = call_id
            call_type = entry.get("type")
            if isinstance(call_type, str) and call_type:
                current["type"] = call_type
            fn = entry.get("function")
            if isinstance(fn, dict):
                fn_cur = current.setdefault("function", {"name": "", "arguments": ""})
                name = fn.get("name")
                if isinstance(name, str) and name:
                    fn_cur["name"] = name
                args = fn.get("arguments")
                if isinstance(args, str) and args:
                    fn_cur["arguments"] = str(fn_cur.get("arguments", "")) + args

        async def _flush_tokens(force: bool = False) -> None:
            nonlocal _pending_tokens, _last_flush
            if not _pending_tokens:
                return
            now = time.perf_counter()
            if not force and (now - _last_flush) < _TOKEN_FLUSH_INTERVAL:
                return
            batch = "".join(_pending_tokens)
            _pending_tokens = []
            _last_flush = now
            await self.outputs["BRAIN_OUT"].emit(
                Payload(
                    signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                    source_module=self.module_id,
                    data={
                        "request_id": request_id,
                        "token": batch,
                        "done": False,
                    },
                )
            )

        def _is_tool_opener(candidate: str) -> bool:
            lowered = candidate.lower()
            if "<tool_call" in lowered or "<|tool_call" in lowered:
                return True
            if "<|tool_response>" in lowered:
                return True
            return bool(re.search(r"(?:^|\s)\{\s*\"name\"\s*:", lowered))

        try:
            sanitized_messages = self._strip_old_images(messages)
            safe_messages = self._coalesce_consecutive_roles(sanitized_messages)
            if len(safe_messages) != len(messages):
                log.info(
                    "Coalesced %d consecutive same-role messages before inference",
                    len(messages) - len(safe_messages),
                )

            ctx_limit = self._llm.n_ctx()
            reserve = max(128, int(ctx_limit * 0.1))
            prompt_budget = ctx_limit - reserve

            trimmed = self._trim_messages_to_fit(safe_messages, prompt_budget)
            if trimmed is None:
                err_msg = (
                    f"Prompt too large for context window ({ctx_limit} tokens). "
                    "Try a shorter message or remove attached images."
                )
                log.warning("basic_brain: %s request_id=%s", err_msg, request_id)
                await self.outputs["BRAIN_OUT"].emit(
                    Payload(
                        signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                        source_module=self.module_id,
                        data={
                            "request_id": request_id,
                            "error": err_msg,
                            "token": "",
                            "done": True,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "ttft_s": 0.0,
                            "total_s": 0.0,
                        },
                    )
                )
                return
            if len(trimmed) < len(safe_messages):
                log.info(
                    "basic_brain: trimmed %d messages to fit context (%d -> %d) request_id=%s",
                    len(safe_messages) - len(trimmed),
                    len(safe_messages), len(trimmed), request_id,
                )
            safe_messages = trimmed

            # Check if the GGUF's Jinja template natively renders tools.
            template_str = (
                getattr(self._llm, "metadata", {})
                .get("tokenizer.chat_template", "")
            )
            template_supports_tools = "tools" in template_str

            if tools and not template_supports_tools:
                tool_prompt = self._build_tool_prompt(tools)
                if safe_messages and safe_messages[0].get("role") == "system":
                    existing = str(safe_messages[0].get("content", ""))
                    safe_messages = [
                        {"role": "system", "content": tool_prompt + "\n\n" + existing},
                    ] + list(safe_messages[1:])
                else:
                    safe_messages = [
                        {"role": "system", "content": tool_prompt},
                    ] + list(safe_messages)
                safe_messages = self._flatten_tool_messages(safe_messages)
                safe_messages = self._coalesce_consecutive_roles(safe_messages)
                safe_messages = self._ensure_starts_with_user(safe_messages)

            min_p = float(self.params.get("min_p", 0.0))
            chat_kwargs: dict[str, Any] = {
                "messages": safe_messages,
                "temperature": float(self.params.get("temperature", 0.7)),
                "top_p": float(self.params.get("top_p", 0.95)),
                "top_k": int(self.params.get("top_k", 40)),
                "repeat_penalty": float(self.params.get("repeat_penalty", 1.1)),
                "stream": True,
            }
            if min_p > 0.0:
                chat_kwargs["min_p"] = min_p
            if tools:
                if template_supports_tools:
                    chat_kwargs["tools"] = tools
                chat_kwargs["stop"] = [
                    "</tool_call>",
                    "<tool_call|>",
                    "<end_of_turn>",
                    "<|tool_response>",
                ]
            if tool_choice is not None and template_supports_tools:
                chat_kwargs["tool_choice"] = tool_choice
            _saved_handler: Any = None
            handler = getattr(self._llm, "chat_handler", None)
            if tools and not has_images:
                try:
                    from llama_cpp.llama_chat_format import (
                        Gemma3ChatHandler,
                        Gemma4ChatHandler,
                    )

                    if isinstance(handler, (Gemma3ChatHandler, Gemma4ChatHandler)):
                        _saved_handler = handler
                        self._llm.chat_handler = None
                except Exception:
                    pass

            log.debug(
                "basic_brain: inference setup request_id=%s tools=%d "
                "template_supports_tools=%s handler_bypass=%s chat_handler=%s",
                request_id,
                len(tools) if tools else 0,
                template_supports_tools,
                _saved_handler is not None,
                type(getattr(self._llm, "chat_handler", None)).__name__,
            )
            try:
                stream = self._llm.create_chat_completion(**chat_kwargs)
            except ValueError as ve:
                ve_str = str(ve).lower()
                if "exceed context window" in ve_str or "requested tokens" in ve_str:
                    err_msg = (
                        f"Prompt too large for context window ({ctx_limit} tokens). "
                        "Try a shorter message or remove attached images."
                    )
                    log.warning("basic_brain: %s request_id=%s raw=%s",
                                err_msg, request_id, ve)
                    await self.outputs["BRAIN_OUT"].emit(
                        Payload(
                            signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                            source_module=self.module_id,
                            data={
                                "request_id": request_id,
                                "error": err_msg,
                                "token": "",
                                "done": True,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "ttft_s": 0.0,
                                "total_s": 0.0,
                            },
                        )
                    )
                    if _saved_handler is not None:
                        self._llm.chat_handler = _saved_handler
                    return
                elif (
                    "system role not supported" in ve_str
                    or "roles must alternate" in ve_str
                ):
                    merged = self._merge_system_messages(safe_messages)
                    merged = self._coalesce_consecutive_roles(merged)
                    merged = self._ensure_starts_with_user(merged)
                    log.info(
                        "Chat template rejected message sequence (%s) — "
                        "merged %d system message(s) and enforced alternation",
                        ve, sum(1 for m in safe_messages if m.get("role") == "system"),
                    )
                    chat_kwargs["messages"] = merged
                    stream = self._llm.create_chat_completion(**chat_kwargs)
                else:
                    raise

            t_inference_start = time.perf_counter()
            ipc_time_total = 0.0

            for chunk in stream:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                delta_tool_calls = delta.get("tool_calls")
                if isinstance(delta_tool_calls, list):
                    for row in delta_tool_calls:
                        if isinstance(row, dict):
                            _merge_tool_call_delta(row)
                token = delta.get("content", "")
                if token:
                    if ttft is None:
                        ttft = time.perf_counter() - t0

                    full_response.append(token)
                    completion_tokens += 1

                    visible_token = token
                    if tools:
                        probe = (_tool_detection_tail + token)
                        _tool_detection_tail = probe[-200:]
                        if _suppressing_tool_tokens or _is_tool_opener(probe):
                            _suppressing_tool_tokens = True
                            _tool_suppress_buffer.append(token)
                            visible_token = ""

                    if visible_token:
                        if not first_token_logged:
                            first_token_logged = True
                            log.info(
                                "basic_brain: first token request_id=%s ttft=%.3fs",
                                request_id, ttft,
                            )
                            t_ipc = time.perf_counter()
                            await self.outputs["BRAIN_OUT"].emit(
                                Payload(
                                    signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                                    source_module=self.module_id,
                                    data={
                                        "request_id": request_id,
                                        "token": visible_token,
                                        "done": False,
                                    },
                                )
                            )
                            ipc_time_total += time.perf_counter() - t_ipc
                            _last_flush = time.perf_counter()
                        else:
                            _pending_tokens.append(visible_token)
                            t_ipc = time.perf_counter()
                            await _flush_tokens()
                            ipc_time_total += time.perf_counter() - t_ipc

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

            t_ipc = time.perf_counter()
            await _flush_tokens(force=True)
            ipc_time_total += time.perf_counter() - t_ipc

            if _saved_handler is not None:
                self._llm.chat_handler = _saved_handler

            t_inference_end = time.perf_counter()
            inference_wall = t_inference_end - t_inference_start
            inference_pure = inference_wall - ipc_time_total

        except Exception as exc:
            if "_saved_handler" in locals() and _saved_handler is not None:
                self._llm.chat_handler = _saved_handler
            log.exception("Inference error")
            await self.report_state(
                severity="ERROR",
                message=f"Inference error ({request_id}): {exc}",
            )
            await self.outputs["BRAIN_OUT"].emit(
                Payload(
                    signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                    source_module=self.module_id,
                    data={
                        "request_id": request_id,
                        "error": str(exc),
                        "done": True,
                    },
                )
            )
            return

        total_s = time.perf_counter() - t0
        response_text = "".join(full_response)
        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_call_parts):
            row = tool_call_parts[idx]
            fn = row.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "")).strip()
            args = str(fn.get("arguments", ""))
            if not name:
                continue
            tool_calls.append({
                "id": str(row.get("id") or f"call_{idx}"),
                "type": str(row.get("type") or "function"),
                "function": {
                    "name": name,
                    "arguments": args,
                },
            })

        if tools and not tool_calls:
            parsed_calls, cleaned_text = self._parse_tool_calls_from_text(
                response_text, tools
            )
            if parsed_calls:
                tool_calls = parsed_calls
                response_text = cleaned_text

        if tools and _tool_suppress_buffer and not tool_calls:
            late_visible_text = "".join(_tool_suppress_buffer)
            if late_visible_text:
                await self.outputs["BRAIN_OUT"].emit(
                    Payload(
                        signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                        source_module=self.module_id,
                        data={
                            "request_id": request_id,
                            "token": late_visible_text,
                            "done": False,
                        },
                    )
                )

        await self.outputs["BRAIN_OUT"].emit(
            Payload(
                signal_type=SignalType.BRAIN_STREAM_PAYLOAD,
                source_module=self.module_id,
                data={
                    "request_id": request_id,
                    "token": "",
                    "done": True,
                    "full_response": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "ttft_s": ttft or 0.0,
                    "total_s": total_s,
                    "tool_calls": tool_calls,
                },
            )
        )

        raw_tps = completion_tokens / inference_pure if inference_pure > 0 else 0
        effective_tps = completion_tokens / total_s if total_s > 0 else 0
        ipc_pct = (ipc_time_total / inference_wall * 100) if inference_wall > 0 else 0

        await self.report_state(
            severity="INFO",
            message=(
                f"Inference complete: {prompt_tokens}+{completion_tokens} tokens, "
                f"{total_s:.2f}s"
            ),
        )
        log.info(
            "basic_brain: done request_id=%s tokens=%d total=%.3fs "
            "raw=%.1ft/s effective=%.1ft/s ipc=%.0fms(%.0f%%)",
            request_id, completion_tokens, total_s,
            raw_tps, effective_tps,
            ipc_time_total * 1000, ipc_pct,
        )

    async def check_ready(self) -> bool:
        model_path = self.params.get("model_path", "")
        if not model_path or not Path(model_path).exists():
            return False
        # In-worker: _model_loaded confirms Llama() succeeded.
        # Parent process relies on worker init setting ERROR on failure.
        # If we're still RUNNING here, consider the model load successful.
        if hasattr(self, "_model_loaded"):
            return self._model_loaded
        return True

    async def init(self) -> None:
        model_path = str(self.params.get("model_path", "")).strip()
        if not model_path:
            log.warning("basic_brain: no model_path set — browse for a .gguf file")
            return
        if not Path(model_path).exists():
            log.warning("basic_brain: model_path does not exist: %s", model_path)
            return

        if self._model_loaded and self._loaded_model_path == model_path and self._llm is not None:
            await self.report_state(
                severity="INFO",
                message=f"Model already loaded: {Path(model_path).name}",
            )
            return

        # INIT can be used as an explicit load/reload trigger when a model path is set.
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            gc.collect()
            self._model_loaded = False

        self.status = ModuleStatus.LOADING
        await self._load_model(model_path)
        self.status = ModuleStatus.RUNNING

    async def shutdown(self) -> None:
        if hasattr(self, "_llm") and self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            gc.collect()
            log.info("Model unloaded")

        self.status = ModuleStatus.STOPPED


def register(hypervisor: Any) -> None:
    """Plugin entry-point called by the module scanner."""
    module = BasicBrainModule()
    hypervisor.register_module(module)
