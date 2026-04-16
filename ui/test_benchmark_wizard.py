"""Settings wizard for LLM test and benchmark runs.

**Architecture** — mirrors basic_brain's process-isolation pattern:

* Subprocess targets live in ``scripts.test_runner`` — a minimal module
  with **zero PyQt / heavy UI imports**.  On Windows ``spawn``,
  ``multiprocessing.Process`` must import the target's module to unpickle
  it; keeping it lean avoids initialising Qt native libraries in a child
  process that has no ``QApplication`` (which causes the NULL-pointer
  access violation).

* Each subprocess calls ``ensure_ggml_backends()`` with **no mode
  filter** — loads ALL backend DLLs (RPC, Vulkan, CPU) exactly like
  ``basic_brain._load_backends()``.  Backend enforcement is done via
  ``n_gpu_layers`` (``-1`` for Vulkan, ``0`` for CPU).  No silent
  fallback.

* This file is **UI only** — it never imports ``llama_cpp``.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.probe_service import get_probe_service
from scripts.llm_tuner import (
    ModelRunResult,
    PassMetric,
    _compute_tps_stats,
    _is_size_failure,
    _mode_token_count,
)
from scripts.model_test_suite import (
    MODEL_MANIFEST,
    TestResult,
    download_model,
    select_models,
)
from scripts.test_runner import run_benchmark as _subprocess_benchmark
from scripts.test_runner import run_functionality as _subprocess_functionality
from scripts.test_runner import run_tool_use as _subprocess_tool_use

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

_CARD_SS = (
    "background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "  stop:0 #0e120e, stop:1 #080c08);"
    " border: 1px solid #1e281e; border-radius: 6px;"
)

_BTN_SS = (
    "QPushButton {"
    "  background: #0c140c; color: #8ea58e; border: 1px solid #2a3a2a;"
    "  border-radius: 4px; padding: 5px 14px; font-size: 11px;"
    "  font-family: 'Consolas', 'Courier New', monospace; font-weight: 600;"
    "}"
    "QPushButton:hover { background: #142014; }"
    "QPushButton:pressed { background: #0a100a; }"
)

_MODEL_TIMEOUT_S = 600


# ── Engine (QThread — orchestrates downloads + subprocess per model) ─


class TestBenchmarkEngine(QThread):
    """Worker thread that executes system/specific tests and benchmarks.

    Model downloads happen in-process (safe). Inference runs in a fresh
    ``multiprocessing.Process`` per model so the native llama.cpp library
    starts with clean state every time, avoiding access violations.
    """

    log_line = Signal(str)
    model_done = Signal(dict)
    finished = Signal(dict)

    def __init__(
        self,
        *,
        mode: str,
        run_type: str,
        user_model_paths: list[str],
        model_dir: Path,
        passes: int,
        max_tokens: int,
        gpu_layers: int,
        backend: str,
        ctx_length: int,
        n_batch: int,
        tool_use_check: bool = False,
        test_image: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._run_type = run_type
        self._user_model_paths = list(user_model_paths)[:5]
        self._model_dir = model_dir
        self._passes = passes
        self._max_tokens = max_tokens
        self._gpu_layers = gpu_layers
        self._backend = "vulkan" if str(backend).strip().lower() == "vulkan" else "cpu"
        self._ctx_length = ctx_length
        self._n_batch = n_batch
        self._tool_use_check = bool(tool_use_check)
        self._test_image = test_image
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    # ── shared subprocess config ────────────────────────────────

    def _base_config(self, model_path: str, mmproj_path: str, multimodal: bool) -> dict[str, Any]:
        return {
            "project_root": _PROJECT_ROOT,
            "backend": self._backend,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "gpu_layers": self._gpu_layers,
            "ctx_length": self._ctx_length,
            "n_batch": self._n_batch,
            "max_tokens": self._max_tokens,
            "passes": self._passes,
            "test_image": self._test_image,
            "multimodal": multimodal,
        }

    # ── main run loop ───────────────────────────────────────────

    def run(self) -> None:  # noqa: D401
        started = time.time()
        hw = asdict(get_probe_service().hardware())
        vram_total_gb = float(hw.get("vram_total_mb", 0.0) or 0.0) / 1024.0
        self._model_dir.mkdir(parents=True, exist_ok=True)

        models = self._build_model_list()
        results: list[dict[str, Any]] = []
        size_fail_count = 0
        stopped_early = False

        self.log_line.emit(
            f"Starting {self._run_type} run for {len(models)} model(s) [{self._mode}] "
            f"on backend={self._backend} tool_use_check={'on' if self._tool_use_check else 'off'}"
        )
        for idx, model in enumerate(models, start=1):
            if self._cancel:
                self.log_line.emit("Cancelled by user.")
                break

            model_name = str(model.get("name", model.get("id", "unknown")))
            self.log_line.emit(f"[{idx}/{len(models)}] {model_name}")

            if self._run_type == "functionality":
                row = self._run_functionality_model(model)
            else:
                row = self._run_benchmark_model(model, vram_total_gb=vram_total_gb)
                if bool(row.get("size_failure")):
                    size_fail_count += 1
                    if size_fail_count >= 3:
                        self.log_line.emit("VRAM ceiling reached after 3 size-related load failures.")
                        stopped_early = True
                        results.append(row)
                        self.model_done.emit(row)
                        break

            results.append(row)
            self.model_done.emit(row)

        ok_count = len([r for r in results if str(r.get("status")) == "ok"])
        partial_count = len([r for r in results if str(r.get("status")) == "partial"])
        failed_count = len(results) - ok_count - partial_count

        report = {
            "timestamp": time.time(),
            "duration_s": max(time.time() - started, 0.0),
            "mode": self._mode,
            "run_type": self._run_type,
            "backend": self._backend,
            "hardware": hw,
            "summary": {
                "total": len(results),
                "ok": ok_count,
                "partial": partial_count,
                "failed": failed_count,
                "size_fail_count": size_fail_count,
                "cancelled": bool(self._cancel),
                "stopped_early": stopped_early,
            },
            "results": results,
        }
        self.finished.emit(report)

    # ── model list ──────────────────────────────────────────────

    def _build_model_list(self) -> list[dict[str, Any]]:
        if self._mode == "specific":
            models: list[dict[str, Any]] = []
            for idx, p in enumerate(self._user_model_paths):
                path = Path(p).expanduser().resolve()
                models.append(
                    {
                        "id": f"user_model_{idx + 1}",
                        "family": "user",
                        "tier": 1,
                        "name": path.stem,
                        "params_b": 0.0,
                        "repo_id": "",
                        "filename_pattern": path.name,
                        "quant": "unknown",
                        "multimodal": False,
                        "user_path": str(path),
                    }
                )
            return models

        selection_args = argparse.Namespace(
            tier="all",
            family=[],
            include_user_models=[],
            max_models=0,
        )
        return select_models(selection_args)

    # ── per-model runners (download in-process, inference in subprocess) ─

    def _run_functionality_model(self, model: dict[str, Any]) -> dict[str, Any]:
        model_dir = Path(self._model_dir)
        error_text = ""
        local_path_str = ""
        size_gb = 0.0

        try:
            local_path, _, mmproj_path = download_model(
                model=model, model_dir=model_dir, skip_download=False, hf_token="",
            )
            local_path_str = str(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Model file missing locally: {local_path}")
            size_gb = local_path.stat().st_size / (1024**3)
        except Exception as exc:
            error_text = str(exc)
            return self._make_functionality_row(model, local_path_str, size_gb, error=error_text)

        cfg = self._base_config(local_path_str, mmproj_path, bool(model.get("multimodal")))
        self.log_line.emit(f"  Spawning inference subprocess ({self._backend})…")
        result = self._exec_subprocess(_subprocess_functionality, cfg)

        if not result.get("ok"):
            error_text = result.get("error", "subprocess failed")
            self.log_line.emit(f"  FAILED: {error_text[:200]}")
            row = self._make_functionality_row(model, local_path_str, size_gb, error=error_text)
            return self._attach_tool_use_result(row, cfg)

        load_ok = True
        raw_ok = bool(result.get("raw_ok"))
        chat_ok = bool(result.get("chat_ok"))
        image_ok = bool(result.get("image_ok"))
        image_skipped = bool(result.get("image_skipped", True))
        passed = load_ok and raw_ok and chat_ok and (image_ok or image_skipped)
        status = "ok" if passed else "failed"

        self.log_line.emit(
            f"  {status.upper()} load={result.get('load_s', 0):.2f}s "
            f"chat_tps={result.get('chat_tps', 0):.1f}"
        )

        row = asdict(TestResult(
            model_id=str(model["id"]),
            model_name=str(model["name"]),
            family=str(model["family"]),
            tier=int(model["tier"]),
            quant=str(model["quant"]),
            local_path=local_path_str,
            size_gb=result.get("size_gb", size_gb),
            passed=passed,
            load_ok=load_ok,
            chat_ok=chat_ok,
            raw_ok=raw_ok,
            image_ok=image_ok,
            image_test_skipped=image_skipped,
            failure_type="",
            failure_hint="",
            error="",
            load_s=float(result.get("load_s", 0)),
            raw_ttft_s=float(result.get("raw_ttft", 0)),
            raw_total_s=float(result.get("raw_total_s", 0)),
            raw_tokens=int(result.get("raw_tokens", 0)),
            raw_tps=float(result.get("raw_tps", 0)),
            chat_ttft_s=float(result.get("chat_ttft", 0)),
            chat_total_s=float(result.get("chat_total_s", 0)),
            chat_tokens=int(result.get("chat_tokens", 0)),
            chat_tps=float(result.get("chat_tps", 0)),
            timestamp=time.time(),
        ))
        row["status"] = status
        row["result_type"] = "functionality"
        return self._attach_tool_use_result(row, cfg)

    def _run_benchmark_model(self, model: dict[str, Any], vram_total_gb: float) -> dict[str, Any]:
        local_path_str = ""
        size_gb = 0.0
        error = ""

        try:
            local_path, _, mmproj_path = download_model(
                model=model, model_dir=self._model_dir, skip_download=False, hf_token="",
            )
            local_path_str = str(local_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Model file missing locally: {local_path}")
            size_gb = local_path.stat().st_size / (1024**3)
        except Exception as exc:
            error = str(exc)
            return self._make_benchmark_row(model, local_path_str, size_gb, error=error, vram_gb=vram_total_gb)

        cfg = self._base_config(local_path_str, mmproj_path, bool(model.get("multimodal")))
        self.log_line.emit(f"  Spawning benchmark subprocess ({self._backend})…")
        result = self._exec_subprocess(_subprocess_benchmark, cfg)

        if not result.get("ok"):
            error = result.get("error", "subprocess failed")
            self.log_line.emit(f"  FAILED: {error[:200]}")
            row = self._make_benchmark_row(
                model, local_path_str, result.get("size_gb", size_gb),
                error=error, vram_gb=vram_total_gb,
            )
            return self._attach_tool_use_result(row, cfg)

        raw_metrics = result.get("metrics", [])
        metrics = [
            PassMetric(
                pass_index=m["pass_index"], tokens=m["tokens"],
                wall_s=m["wall_s"], tps=m["tps"],
                stale_retry=m.get("stale_retry", False), error=m.get("error", ""),
            )
            for m in raw_metrics
        ]

        status = "ok"
        if not any(m.error == "" for m in metrics):
            status = "inference_failed"
            error = "; ".join(m.error for m in metrics if m.error) or "No successful pass."
        elif any(m.error for m in metrics):
            status = "partial"
            error = "; ".join(m.error for m in metrics if m.error)

        avg_tps, stddev_tps, outlier_trimmed = _compute_tps_stats(metrics)
        self.log_line.emit(
            f"  {status.upper()} load={result.get('load_s', 0):.2f}s avg_tps={avg_tps:.1f}"
        )

        token_mode = _mode_token_count([m.tokens for m in metrics if m.error == ""])
        size_failure = status == "load_failed" and _is_size_failure(error, size_gb, vram_total_gb)

        row = asdict(ModelRunResult(
            model_id=str(model["id"]),
            model_name=str(model["name"]),
            family=str(model["family"]),
            tier=int(model["tier"]),
            quant=str(model["quant"]),
            local_path=local_path_str,
            size_gb=result.get("size_gb", size_gb),
            load_s=float(result.get("load_s", 0)),
            status=status,
            error=error,
            size_failure=size_failure,
            passes_ok=len([m for m in metrics if m.error == ""]),
            requested_passes=int(self._passes),
            avg_tps=avg_tps,
            stddev_tps=stddev_tps,
            outlier_trimmed=outlier_trimmed,
            tokens_mode=token_mode,
            benchmark_passes=metrics,
        ))
        row["result_type"] = "benchmark"
        return self._attach_tool_use_result(row, cfg)

    def _attach_tool_use_result(self, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        if not self._tool_use_check:
            row["tool_use_ok"] = None
            row["tool_use_detail"] = "Skipped (checkbox disabled)."
            return row

        self.log_line.emit("  Running tool-use template probe…")
        probe_result = self._exec_subprocess(_subprocess_tool_use, config)
        if not probe_result.get("ok"):
            detail = str(probe_result.get("error", "tool-use probe failed"))
            row["tool_use_ok"] = False
            row["tool_use_detail"] = detail
            self.log_line.emit(f"  Tool-use probe FAILED: {detail[:180]}")
            return row

        tool_use_ok = bool(probe_result.get("tool_use_ok"))
        detail = str(probe_result.get("detail", "")).strip()
        row["tool_use_ok"] = tool_use_ok
        row["tool_use_detail"] = detail
        if tool_use_ok:
            self.log_line.emit(f"  Tool-use probe PASS: {detail[:120]}")
        else:
            self.log_line.emit(f"  Tool-use probe FAIL: {detail[:180]}")
        return row

    # ── subprocess execution helper ─────────────────────────────

    @staticmethod
    def _exec_subprocess(target: Any, config: dict[str, Any]) -> dict[str, Any]:
        q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=target, args=(config, q), daemon=True)
        proc.start()
        proc.join(timeout=_MODEL_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
            return {"ok": False, "error": f"Subprocess timed out after {_MODEL_TIMEOUT_S}s"}
        try:
            return q.get_nowait()
        except Exception:
            code = proc.exitcode
            return {"ok": False, "error": f"Subprocess exited with code {code} and produced no result"}

    # ── row factories for error / missing-data cases ────────────

    @staticmethod
    def _make_functionality_row(
        model: dict[str, Any], local_path: str, size_gb: float, *, error: str,
    ) -> dict[str, Any]:
        row = asdict(TestResult(
            model_id=str(model["id"]),
            model_name=str(model["name"]),
            family=str(model["family"]),
            tier=int(model["tier"]),
            quant=str(model["quant"]),
            local_path=local_path,
            size_gb=size_gb,
            passed=False, load_ok=False, chat_ok=False, raw_ok=False,
            image_ok=False, image_test_skipped=True,
            failure_type="", failure_hint="", error=error,
            load_s=0, raw_ttft_s=0, raw_total_s=0, raw_tokens=0, raw_tps=0,
            chat_ttft_s=0, chat_total_s=0, chat_tokens=0, chat_tps=0,
            timestamp=time.time(),
        ))
        row["status"] = "failed"
        row["result_type"] = "functionality"
        return row

    @staticmethod
    def _make_benchmark_row(
        model: dict[str, Any], local_path: str, size_gb: float,
        *, error: str, vram_gb: float,
    ) -> dict[str, Any]:
        size_failure = _is_size_failure(error, size_gb, vram_gb)
        row = asdict(ModelRunResult(
            model_id=str(model["id"]),
            model_name=str(model["name"]),
            family=str(model["family"]),
            tier=int(model["tier"]),
            quant=str(model["quant"]),
            local_path=local_path,
            size_gb=size_gb,
            load_s=0, status="load_failed", error=error,
            size_failure=size_failure, passes_ok=0,
            requested_passes=0, avg_tps=0, stddev_tps=0,
            outlier_trimmed=False, tokens_mode=0, benchmark_passes=[],
        ))
        row["result_type"] = "benchmark"
        return row


# ── Wizard dialog (UI only — no llama_cpp imports) ──────────────────


class TestBenchmarkWizard(QDialog):
    """4-step modal wizard to run tests and benchmarks."""

    def __init__(self, parent: QWidget | None = None, configured_backend: str = "vulkan") -> None:
        super().__init__(parent)
        self.setWindowTitle("Test & Benchmark Suite")
        self.setModal(True)
        self.resize(980, 720)
        self.setStyleSheet("background: #121316; color: #8ea58e;")

        self._engine: TestBenchmarkEngine | None = None
        self._results_report: dict[str, Any] | None = None
        self._specific_model_paths: list[str] = []
        self._configured_backend = (
            "vulkan" if str(configured_backend).strip().lower() == "vulkan" else "cpu"
        )
        self._completed_models = 0
        self._expected_models = len(MODEL_MANIFEST)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("TEST & BENCHMARK SUITE")
        title.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #607060; letter-spacing: 1px;"
            "font-family: 'Consolas', 'Courier New', monospace;"
        )
        root.addWidget(title)

        self._stack = QStackedWidget(self)
        root.addWidget(self._stack, 1)

        self._page_mode = self._build_mode_page()
        self._page_type = self._build_type_page()
        self._page_files = self._build_files_page()
        self._page_running = self._build_running_page()
        self._page_results = self._build_results_page()

        self._stack.addWidget(self._page_mode)
        self._stack.addWidget(self._page_type)
        self._stack.addWidget(self._page_files)
        self._stack.addWidget(self._page_running)
        self._stack.addWidget(self._page_results)
        self._stack.setCurrentWidget(self._page_mode)

    def _build_mode_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setSpacing(10)
        card = QWidget()
        card.setStyleSheet(_CARD_SS)
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(14, 14, 14, 14)
        label = QLabel("Step 1: Choose test scope")
        label.setStyleSheet("font-size: 12px; font-weight: 600; color: #8ea58e;")
        card_l.addWidget(label)
        self._radio_system = QRadioButton("Test System")
        self._radio_specific = QRadioButton("Test Specific Models")
        self._radio_system.setChecked(True)
        card_l.addWidget(self._radio_system)
        card_l.addWidget(self._radio_specific)
        card_l.addStretch(1)
        next_btn = QPushButton("NEXT")
        next_btn.setStyleSheet(_BTN_SS)
        next_btn.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_type))
        card_l.addWidget(next_btn, 0, Qt.AlignmentFlag.AlignRight)
        box.addWidget(card)
        return page

    def _build_type_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        card = QWidget()
        card.setStyleSheet(_CARD_SS)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.addWidget(QLabel("Step 2: Choose run type"))
        self._radio_functionality = QRadioButton("Test Functionality")
        self._radio_benchmark = QRadioButton("Benchmark")
        self._radio_functionality.setChecked(True)
        cl.addWidget(self._radio_functionality)
        cl.addWidget(self._radio_benchmark)
        self._tool_use_checkbox = QCheckBox("Confirm tool use template")
        self._tool_use_checkbox.setChecked(False)
        self._tool_use_checkbox.setStyleSheet(
            "QCheckBox { color: #8ea58e; font-size: 11px; "
            "font-family: 'Consolas', 'Courier New', monospace; }"
        )
        cl.addWidget(self._tool_use_checkbox)
        row = QHBoxLayout()
        back_btn = QPushButton("BACK")
        back_btn.setStyleSheet(_BTN_SS)
        back_btn.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_mode))
        next_btn = QPushButton("NEXT")
        next_btn.setStyleSheet(_BTN_SS)
        next_btn.clicked.connect(self._advance_after_type)
        row.addWidget(back_btn)
        row.addStretch(1)
        row.addWidget(next_btn)
        cl.addStretch(1)
        cl.addLayout(row)
        box.addWidget(card)
        return page

    def _build_files_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        card = QWidget()
        card.setStyleSheet(_CARD_SS)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.addWidget(QLabel("Step 3: Select up to 5 GGUF model files"))
        self._files_table = QTableWidget(0, 2)
        self._files_table.setHorizontalHeaderLabels(["Model file", "Size"])
        self._files_table.horizontalHeader().setStretchLastSection(False)
        self._files_table.horizontalHeader().setSectionResizeMode(0, self._files_table.horizontalHeader().ResizeMode.Stretch)
        self._files_table.horizontalHeader().setSectionResizeMode(1, self._files_table.horizontalHeader().ResizeMode.ResizeToContents)
        cl.addWidget(self._files_table, 1)
        btns = QHBoxLayout()
        add_btn = QPushButton("ADD MODEL")
        add_btn.setStyleSheet(_BTN_SS)
        add_btn.clicked.connect(self._add_specific_model)
        remove_btn = QPushButton("REMOVE SELECTED")
        remove_btn.setStyleSheet(_BTN_SS)
        remove_btn.clicked.connect(self._remove_selected_specific_model)
        btns.addWidget(add_btn)
        btns.addWidget(remove_btn)
        btns.addStretch(1)
        cl.addLayout(btns)
        row = QHBoxLayout()
        back_btn = QPushButton("BACK")
        back_btn.setStyleSheet(_BTN_SS)
        back_btn.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_type))
        start_btn = QPushButton("START RUN")
        start_btn.setStyleSheet(_BTN_SS)
        start_btn.clicked.connect(self._start_run)
        row.addWidget(back_btn)
        row.addStretch(1)
        row.addWidget(start_btn)
        cl.addLayout(row)
        box.addWidget(card)
        return page

    def _build_running_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        card = QWidget()
        card.setStyleSheet(_CARD_SS)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.addWidget(QLabel("Step 4: Running"))
        self._progress_label = QLabel("Preparing...")
        cl.addWidget(self._progress_label)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        cl.addWidget(self._progress)
        self._running_table = QTableWidget(0, 6)
        self._running_table.setHorizontalHeaderLabels(
            ["Model", "Status", "Size (GB)", "Avg TPS", "Tool Use", "Error"]
        )
        self._running_table.horizontalHeader().setSectionResizeMode(0, self._running_table.horizontalHeader().ResizeMode.Stretch)
        self._running_table.horizontalHeader().setSectionResizeMode(5, self._running_table.horizontalHeader().ResizeMode.Stretch)
        cl.addWidget(self._running_table, 1)
        cl.addWidget(QLabel("Live Log"))
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setStyleSheet(
            "QTextEdit { background: #0a100a; border: 1px solid #1e281e; "
            "color: #8ea58e; font-family: 'Consolas', 'Courier New', monospace; font-size: 10px; }"
        )
        cl.addWidget(self._log_box, 1)
        row = QHBoxLayout()
        self._cancel_btn = QPushButton("CANCEL")
        self._cancel_btn.setStyleSheet(_BTN_SS)
        self._cancel_btn.clicked.connect(self._cancel_run)
        row.addStretch(1)
        row.addWidget(self._cancel_btn)
        cl.addLayout(row)
        box.addWidget(card)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        card = QWidget()
        card.setStyleSheet(_CARD_SS)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.addWidget(QLabel("Results"))
        self._results_summary = QLabel("No run completed yet.")
        self._results_summary.setWordWrap(True)
        cl.addWidget(self._results_summary)
        self._results_table = QTableWidget(0, 6)
        self._results_table.setHorizontalHeaderLabels(
            ["Model", "Status", "Size (GB)", "Avg TPS", "Tool Use", "Error"]
        )
        self._results_table.horizontalHeader().setSectionResizeMode(0, self._results_table.horizontalHeader().ResizeMode.Stretch)
        self._results_table.horizontalHeader().setSectionResizeMode(5, self._results_table.horizontalHeader().ResizeMode.Stretch)
        cl.addWidget(self._results_table, 1)
        row = QHBoxLayout()
        export_json_btn = QPushButton("DOWNLOAD JSON")
        export_json_btn.setStyleSheet(_BTN_SS)
        export_json_btn.clicked.connect(self._save_json_report)
        export_csv_btn = QPushButton("DOWNLOAD CSV")
        export_csv_btn.setStyleSheet(_BTN_SS)
        export_csv_btn.clicked.connect(self._save_csv_report)
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(_BTN_SS)
        close_btn.clicked.connect(self.accept)
        row.addWidget(export_json_btn)
        row.addWidget(export_csv_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        cl.addLayout(row)
        box.addWidget(card)
        return page

    def _advance_after_type(self) -> None:
        if self._radio_specific.isChecked():
            self._stack.setCurrentWidget(self._page_files)
            return
        self._start_run()

    def _add_specific_model(self) -> None:
        if len(self._specific_model_paths) >= 5:
            QMessageBox.information(self, "Limit reached", "You can select up to 5 models.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Select GGUF model", str(Path.home()), "GGUF (*.gguf)")
        if not filename:
            return
        p = Path(filename).expanduser().resolve()
        if str(p) in self._specific_model_paths:
            return
        self._specific_model_paths.append(str(p))
        self._refresh_files_table()

    def _remove_selected_specific_model(self) -> None:
        selected = self._files_table.currentRow()
        if selected < 0 or selected >= len(self._specific_model_paths):
            return
        self._specific_model_paths.pop(selected)
        self._refresh_files_table()

    def _refresh_files_table(self) -> None:
        self._files_table.setRowCount(0)
        for idx, path_str in enumerate(self._specific_model_paths):
            p = Path(path_str)
            size_gb = p.stat().st_size / (1024**3) if p.exists() else 0.0
            self._files_table.insertRow(idx)
            self._files_table.setItem(idx, 0, QTableWidgetItem(path_str))
            self._files_table.setItem(idx, 1, QTableWidgetItem(f"{size_gb:.2f}"))

    def _start_run(self) -> None:
        mode = "specific" if self._radio_specific.isChecked() else "system"
        run_type = "benchmark" if self._radio_benchmark.isChecked() else "functionality"

        if mode == "specific" and not self._specific_model_paths:
            QMessageBox.warning(self, "Models required", "Select at least one GGUF model.")
            self._stack.setCurrentWidget(self._page_files)
            return

        self._running_table.setRowCount(0)
        self._log_box.clear()
        self._results_report = None
        self._completed_models = 0
        self._expected_models = len(self._specific_model_paths) if mode == "specific" else len(MODEL_MANIFEST)
        self._progress.setRange(0, max(1, self._expected_models))
        self._progress.setValue(0)
        self._progress_label.setText(f"0 / {self._expected_models} completed")
        self._cancel_btn.setEnabled(True)
        self._stack.setCurrentWidget(self._page_running)

        self._engine = TestBenchmarkEngine(
            mode=mode,
            run_type=run_type,
            user_model_paths=self._specific_model_paths,
            model_dir=Path.home() / "Downloads" / "model_test_cache",
            passes=5,
            max_tokens=100,
            gpu_layers=-1,
            backend=self._configured_backend,
            ctx_length=2048,
            n_batch=512,
            tool_use_check=self._tool_use_checkbox.isChecked(),
            test_image="",
            parent=self,
        )
        self._engine.log_line.connect(self._append_log)
        self._engine.model_done.connect(self._on_model_done)
        self._engine.finished.connect(self._on_finished)
        self._engine.start()

    def _append_log(self, text: str) -> None:
        self._log_box.append(text)

    def _on_model_done(self, row: dict[str, Any]) -> None:
        self._completed_models += 1
        self._progress.setValue(min(self._completed_models, self._progress.maximum()))
        self._progress_label.setText(f"{self._completed_models} / {self._expected_models} completed")

        table = self._running_table
        ridx = table.rowCount()
        table.insertRow(ridx)
        model_name = str(row.get("model_name", row.get("model_id", "-")))
        status = str(row.get("status", "unknown"))
        size_gb = float(row.get("size_gb", 0.0) or 0.0)
        avg_tps = float(row.get("avg_tps", row.get("chat_tps", 0.0)) or 0.0)
        tool_use_label = self._tool_use_label(row)
        error = str(row.get("error", ""))
        table.setItem(ridx, 0, QTableWidgetItem(model_name))
        table.setItem(ridx, 1, QTableWidgetItem(status.upper()))
        table.setItem(ridx, 2, QTableWidgetItem(f"{size_gb:.2f}"))
        table.setItem(ridx, 3, QTableWidgetItem(f"{avg_tps:.1f}"))
        table.setItem(ridx, 4, QTableWidgetItem(tool_use_label))
        table.setItem(ridx, 5, QTableWidgetItem(error[:200]))

    def _on_finished(self, report: dict[str, Any]) -> None:
        self._results_report = report
        self._cancel_btn.setEnabled(False)
        summary = report.get("summary", {})
        self._results_summary.setText(
            f"Total: {summary.get('total', 0)}   "
            f"OK: {summary.get('ok', 0)}   "
            f"Partial: {summary.get('partial', 0)}   "
            f"Failed: {summary.get('failed', 0)}   "
            f"SizeFails: {summary.get('size_fail_count', 0)}"
        )
        self._populate_results_table(report.get("results", []))
        self._stack.setCurrentWidget(self._page_results)

    def _populate_results_table(self, rows: list[dict[str, Any]]) -> None:
        self._results_table.setRowCount(0)
        for row in rows:
            ridx = self._results_table.rowCount()
            self._results_table.insertRow(ridx)
            model_name = str(row.get("model_name", row.get("model_id", "-")))
            status = str(row.get("status", "unknown")).upper()
            size_gb = float(row.get("size_gb", 0.0) or 0.0)
            avg_tps = float(row.get("avg_tps", row.get("chat_tps", 0.0)) or 0.0)
            tool_use_label = self._tool_use_label(row)
            error = str(row.get("error", ""))
            self._results_table.setItem(ridx, 0, QTableWidgetItem(model_name))
            self._results_table.setItem(ridx, 1, QTableWidgetItem(status))
            self._results_table.setItem(ridx, 2, QTableWidgetItem(f"{size_gb:.2f}"))
            self._results_table.setItem(ridx, 3, QTableWidgetItem(f"{avg_tps:.1f}"))
            self._results_table.setItem(ridx, 4, QTableWidgetItem(tool_use_label))
            self._results_table.setItem(ridx, 5, QTableWidgetItem(error[:220]))

    @staticmethod
    def _tool_use_label(row: dict[str, Any]) -> str:
        flag = row.get("tool_use_ok", None)
        if flag is True:
            return "PASS"
        if flag is False:
            return "FAIL"
        return "SKIPPED"

    def _cancel_run(self) -> None:
        if self._engine is None:
            return
        self._append_log("Cancellation requested...")
        self._engine.cancel()
        self._cancel_btn.setEnabled(False)

    def _save_json_report(self) -> None:
        if not self._results_report:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON report",
            str(Path.home() / "Downloads" / "test_benchmark_results.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        out = Path(path).expanduser().resolve()
        out.write_text(json.dumps(self._results_report, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Saved", f"Saved JSON report:\n{out}")

    def _save_csv_report(self) -> None:
        if not self._results_report:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV report",
            str(Path.home() / "Downloads" / "test_benchmark_results.csv"),
            "CSV (*.csv)",
        )
        if not path:
            return
        out = Path(path).expanduser().resolve()
        rows = list(self._results_report.get("results", []))
        with out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "model_id",
                    "model_name",
                    "family",
                    "quant",
                    "size_gb",
                    "status",
                    "avg_tps",
                    "stddev_tps",
                    "tool_use_ok",
                    "tool_use_detail",
                    "error",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.get("model_id", ""),
                        row.get("model_name", ""),
                        row.get("family", ""),
                        row.get("quant", ""),
                        f"{float(row.get('size_gb', 0.0) or 0.0):.3f}",
                        row.get("status", ""),
                        f"{float(row.get('avg_tps', row.get('chat_tps', 0.0)) or 0.0):.3f}",
                        f"{float(row.get('stddev_tps', 0.0) or 0.0):.3f}",
                        row.get("tool_use_ok", ""),
                        row.get("tool_use_detail", ""),
                        row.get("error", ""),
                    ]
                )
        QMessageBox.information(self, "Saved", f"Saved CSV report:\n{out}")
