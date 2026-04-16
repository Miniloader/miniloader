"""
download_manager.py — Managed download queue.

Supports two backend sources:
  1. Hugging Face URLs  — via ``huggingface_hub.hf_hub_download``
  2. Signed HTTP URLs   — generic streaming GET (S3 / CloudFront pre-signed)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from core.download_store import DownloadStore

if TYPE_CHECKING:
    from core.vault import VaultManager

log = logging.getLogger(__name__)

_CHUNK_SIZE = 256 * 1024  # 256 KiB per read for more responsive progress updates


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DownloadInterrupted(Exception):
    """Raised from progress hooks when a pause/cancel is requested."""


class _ProgressTqdm:
    """
    Lightweight tqdm-compatible shim for huggingface_hub progress callbacks.

    huggingface_hub instantiates this class as ``tqdm_class(...)`` and calls
    ``update`` as chunks arrive.
    """

    def __init__(
        self,
        *args: Any,
        total: int | float | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> None:
        self.total = int(total or 0)
        self.n = 0
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event
        self._last_n = 0
        self._last_t = time.monotonic()
        self._last_speed_bps = 0.0

    def update(self, n: int = 1) -> int:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise DownloadInterrupted("download interrupted")

        n_int = int(n or 0)
        self.n += n_int

        speed_bps = self._last_speed_bps
        now = time.monotonic()
        dt = now - self._last_t
        # Avoid resetting speed to 0 on ultra-fast back-to-back updates.
        if dt >= 0.12:
            dn = self.n - self._last_n
            if dn > 0:
                speed_bps = float(dn) / dt
                self._last_speed_bps = speed_bps
            self._last_n = self.n
            self._last_t = now

        if self._progress_callback is not None:
            self._progress_callback(self.n, self.total, speed_bps)
        return n_int

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __enter__(self) -> _ProgressTqdm:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class DownloadManager:
    """Queue-backed manager for managed downloads (HF + signed HTTP)."""

    def __init__(
        self,
        vault: VaultManager,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        on_failed: Callable[[dict[str, Any]], None] | None = None,
        max_concurrent: int = 2,
    ) -> None:
        self._vault = vault
        self._store = DownloadStore(vault)
        self._models_root = vault.ensure_user_data_dir() / "models"
        self._models_root.mkdir(parents=True, exist_ok=True)
        self._portal_root = vault.ensure_user_data_dir() / "portal"
        self._portal_root.mkdir(parents=True, exist_ok=True)

        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_failed = on_failed

        self._max_concurrent = max(1, int(max_concurrent))
        self._queue: deque[str] = deque()
        self._workers: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._desired_states: dict[str, str] = {}
        self._live_metrics: dict[str, dict[str, float | None]] = {}
        self._progress_db_state: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

        self._db_loop = asyncio.new_event_loop()
        self._db_loop_thread = threading.Thread(
            target=self._db_loop_worker,
            name="download-manager-db-loop",
            daemon=True,
        )
        self._db_loop_thread.start()

        self._run_db(self._store.ensure_table())
        self._run_db(self._store.reset_stale_downloading())

    def set_callbacks(
        self,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
        on_failed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if on_progress is not None:
            self._on_progress = on_progress
        if on_complete is not None:
            self._on_complete = on_complete
        if on_failed is not None:
            self._on_failed = on_failed

    def enqueue(
        self,
        *,
        download_id: str,
        repo_id: str,
        filename: str,
        variant: str,
        size: str,
        url: str,
        kind: str = "file",
    ) -> None:
        self._run_db(
            self._store.insert(
                download_id=download_id,
                repo_id=repo_id,
                filename=filename,
                variant=variant,
                size=size,
                url=url,
                status="queued",
                kind=str(kind or "file"),
                progress=0.0,
                local_path=None,
                error=None,
                created_at=now_iso(),
                completed_at=None,
            )
        )
        with self._lock:
            if download_id not in self._queue and download_id not in self._workers:
                self._queue.append(download_id)
        self._emit_progress({"id": download_id, "status": "queued", "progress": 0})
        self._drain_queue()

    def cancel(self, download_id: str) -> None:
        should_mark_cancelled = False
        with self._lock:
            self._desired_states[download_id] = "cancelled"
            if download_id in self._queue:
                self._queue = deque(x for x in self._queue if x != download_id)
                self._run_db(
                    self._store.update_status(
                        download_id,
                        status="cancelled",
                        error=None,
                        completed_at=now_iso(),
                    )
                )
                self._clear_live_metrics(download_id)
                self._emit_progress({"id": download_id, "status": "cancelled", "progress": 0})
                return
            evt = self._cancel_events.get(download_id)
            has_worker = download_id in self._workers
        if evt is not None:
            self._run_db(
                self._store.update_status(
                    download_id,
                    status="cancelled",
                    error=None,
                    completed_at=now_iso(),
                )
            )
            self._clear_live_metrics(download_id)
            self._emit_progress({"id": download_id, "status": "cancelled"})
            evt.set()
            return
        if not has_worker:
            row = self.get(download_id)
            if row is not None:
                status = str(row.get("status") or "")
                should_mark_cancelled = status not in ("complete", "cancelled", "failed")
        if should_mark_cancelled:
            self._run_db(
                self._store.update_status(
                    download_id,
                    status="cancelled",
                    error=None,
                    completed_at=now_iso(),
                )
            )
            self._clear_live_metrics(download_id)
            self._emit_progress({"id": download_id, "status": "cancelled", "progress": 0})

    def pause(self, download_id: str) -> None:
        with self._lock:
            self._desired_states[download_id] = "paused"
            evt = self._cancel_events.get(download_id)
        if evt is not None:
            self._run_db(self._store.update_status(download_id, status="paused", error=None))
            self._clear_live_metrics(download_id)
            self._emit_progress({"id": download_id, "status": "paused"})
            evt.set()

    def resume(self, download_id: str) -> None:
        row = self.get(download_id)
        if row is None:
            return
        with self._lock:
            self._desired_states.pop(download_id, None)
            if download_id not in self._workers and download_id not in self._queue:
                self._queue.append(download_id)
        self._run_db(self._store.update_status(download_id, status="queued", error=None))
        self._drain_queue()

    def delete(self, download_id: str) -> None:
        """Cancel, remove the partial/broken file from disk, and delete the DB record."""
        row = self.get(download_id)
        self.cancel(download_id)
        if row is not None:
            local_path = str(row.get("local_path") or "").strip()
            if local_path:
                try:
                    Path(local_path).unlink(missing_ok=True)
                except Exception:
                    log.debug("download_manager: failed to remove file %s", local_path, exc_info=True)
        self._run_db(self._store.delete(download_id))

    def dismiss(self, download_id: str) -> None:
        """Remove the DB record only — the downloaded file is kept on disk."""
        self._run_db(self._store.delete(download_id))

    def get(self, download_id: str) -> dict[str, Any] | None:
        row = self._run_db(self._store.get(download_id))
        return self._with_live_metrics(row)

    def get_all(self) -> list[dict[str, Any]]:
        rows = self._run_db(self._store.get_all())
        return [self._with_live_metrics(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            for evt in self._cancel_events.values():
                evt.set()
        self._db_loop.call_soon_threadsafe(self._db_loop.stop)

    def _db_loop_worker(self) -> None:
        asyncio.set_event_loop(self._db_loop)
        self._db_loop.run_forever()

    def _run_db(self, coro: Any, timeout: float = 30.0) -> Any:
        fut: Future = asyncio.run_coroutine_threadsafe(coro, self._db_loop)
        return fut.result(timeout=timeout)

    def _drain_queue(self) -> None:
        while True:
            with self._lock:
                if len(self._workers) >= self._max_concurrent or not self._queue:
                    return
                download_id = self._queue.popleft()
                if download_id in self._workers:
                    continue
                cancel_event = threading.Event()
                self._cancel_events[download_id] = cancel_event
                worker = threading.Thread(
                    target=self._download_worker,
                    args=(download_id, cancel_event),
                    name=f"download-worker-{download_id[:8]}",
                    daemon=True,
                )
                self._workers[download_id] = worker
                worker.start()

    @staticmethod
    def _is_hf_url(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host.endswith("huggingface.co")

    def _download_worker(self, download_id: str, cancel_event: threading.Event) -> None:
        try:
            row = self.get(download_id)
            if row is None:
                return

            url = str(row.get("url") or "").strip()

            self._run_db(self._store.update_progress(download_id, status="downloading", progress=0.0))
            self._emit_progress({"id": download_id, "status": "downloading", "progress": 0})

            if self._is_hf_url(url):
                local_path = self._download_from_hf(download_id, row, cancel_event)
            else:
                local_path = self._download_from_signed_http(download_id, row, cancel_event)

            self._run_db(
                self._store.update_progress(
                    download_id,
                    status="downloading",
                    progress=100.0,
                    error=None,
                )
            )
            self._run_db(
                self._store.update_status(
                    download_id,
                    status="complete",
                    error=None,
                    completed_at=now_iso(),
                    local_path=str(local_path),
                )
            )
            self._emit_complete(
                {
                    "id": download_id,
                    "localPath": str(local_path),
                    "kind": str(row.get("kind") or "file"),
                }
            )
        except DownloadInterrupted:
            desired = self._desired_states.get(download_id, "cancelled")
            self._run_db(
                self._store.update_status(
                    download_id,
                    status=desired,
                    error=None,
                    completed_at=now_iso(),
                )
            )
            self._emit_progress({"id": download_id, "status": desired})
        except Exception as exc:
            err = str(exc).strip() or exc.__class__.__name__
            self._run_db(
                self._store.update_status(
                    download_id,
                    status="failed",
                    error=err,
                    completed_at=now_iso(),
                )
            )
            self._emit_failed({"id": download_id, "error": err})
        finally:
            with self._lock:
                self._workers.pop(download_id, None)
                self._cancel_events.pop(download_id, None)
                self._desired_states.pop(download_id, None)
            self._clear_live_metrics(download_id)
            self._drain_queue()

    def _download_from_hf(
        self,
        download_id: str,
        row: dict[str, Any],
        cancel_event: threading.Event,
    ) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for Hugging Face downloads. "
                "Install with: pip install huggingface_hub"
            ) from exc

        repo_id = str(row.get("repo_id") or "").strip()
        filename = str(row.get("filename") or "").strip()

        tracker_class = partial(
            _ProgressTqdm,
            progress_callback=lambda done, total, speed: self._on_worker_progress(
                download_id, done=done, total=total, speed_bps=speed,
            ),
            cancel_event=cancel_event,
        )

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self._models_root),
            tqdm_class=tracker_class,
        )
        return str(local_path)

    def _download_from_signed_http(
        self,
        download_id: str,
        row: dict[str, Any],
        cancel_event: threading.Event,
    ) -> str:
        import requests

        url = str(row.get("url") or "")
        repo_id = str(row.get("repo_id") or "unknown").strip()
        filename = str(row.get("filename") or "download").strip()

        out_dir = self._portal_root / repo_id.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code == 403:
                raise RuntimeError("Signed URL expired or access denied (403)")
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", "0") or 0)
            written = 0
            last_n = 0
            last_t = time.monotonic()
            last_speed_bps = 0.0

            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    if cancel_event.is_set():
                        raise DownloadInterrupted("download interrupted")
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)

                    now = time.monotonic()
                    dt = now - last_t
                    speed_bps = last_speed_bps
                    if dt >= 0.12:
                        dn = written - last_n
                        if dn > 0:
                            speed_bps = float(dn) / dt
                            last_speed_bps = speed_bps
                        last_n = written
                        last_t = now

                    self._on_worker_progress(
                        download_id,
                        done=written,
                        total=total,
                        speed_bps=speed_bps,
                    )

        tmp_path.replace(out_path)
        return str(out_path)

    def _on_worker_progress(
        self,
        download_id: str,
        *,
        done: int,
        total: int,
        speed_bps: float,
    ) -> None:
        with self._lock:
            desired = self._desired_states.get(download_id)
        if desired in ("paused", "cancelled"):
            return

        now = time.monotonic()
        if speed_bps <= 0:
            with self._lock:
                live = self._live_metrics.get(download_id)
            if live is not None:
                speed_bps = float(live.get("speedBps") or 0.0)
        else:
            with self._lock:
                live = self._live_metrics.get(download_id)
            prev_speed = float((live or {}).get("speedBps") or 0.0)
            # Exponential smoothing to reduce jitter while staying responsive.
            speed_bps = (prev_speed * 0.65) + (float(speed_bps) * 0.35) if prev_speed > 0 else float(speed_bps)

        progress = 0.0
        if total > 0:
            progress = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
        eta_seconds: float | None = None
        if total > 0 and speed_bps > 0:
            remaining = max(0.0, float(total - done))
            eta_seconds = remaining / float(speed_bps)
        self._set_live_metrics(download_id, speed_bps=speed_bps, eta_seconds=eta_seconds)

        should_persist = False
        with self._lock:
            state = self._progress_db_state.get(download_id, {"last_t": 0.0, "last_p": -1.0})
            last_t = float(state.get("last_t", 0.0))
            last_p = float(state.get("last_p", -1.0))
            if (now - last_t) >= 0.30 or abs(progress - last_p) >= 0.35:
                should_persist = True
                self._progress_db_state[download_id] = {"last_t": now, "last_p": progress}
        if should_persist:
            self._run_db(
                self._store.update_progress(
                    download_id,
                    status="downloading",
                    progress=progress,
                    error=None,
                )
            )
        self._emit_progress(
            {
                "id": download_id,
                "status": "downloading",
                "progress": round(progress, 2),
                "speed": round(float(speed_bps), 2),
                "speedBps": round(float(speed_bps), 2),
                "etaSeconds": int(eta_seconds) if eta_seconds is not None else None,
            }
        )

    def _set_live_metrics(self, download_id: str, *, speed_bps: float, eta_seconds: float | None) -> None:
        with self._lock:
            self._live_metrics[download_id] = {
                "speedBps": round(float(speed_bps), 2),
                "etaSeconds": float(eta_seconds) if eta_seconds is not None else None,
            }

    def _clear_live_metrics(self, download_id: str) -> None:
        with self._lock:
            self._live_metrics.pop(download_id, None)
            self._progress_db_state.pop(download_id, None)

    def _with_live_metrics(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        download_id = str(row.get("id") or "")
        if not download_id:
            return row
        with self._lock:
            metrics = self._live_metrics.get(download_id)
        if not metrics:
            return row
        merged = dict(row)
        merged.update(metrics)
        return merged

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if self._on_progress is not None:
            self._on_progress(payload)

    def _emit_complete(self, payload: dict[str, Any]) -> None:
        if self._on_complete is not None:
            self._on_complete(payload)

    def _emit_failed(self, payload: dict[str, Any]) -> None:
        if self._on_failed is not None:
            self._on_failed(payload)
