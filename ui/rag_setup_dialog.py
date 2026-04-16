"""
rag_setup_dialog.py — RAG embedding model setup UI.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Callable

from cryptography.fernet import Fernet
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.download_manager import DownloadManager
from core.settings_store import SettingsStore
from core.vault import VaultManager

RAG_REPO_ID = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
RAG_FILENAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"
RAG_URL = (
    "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/"
    "all-MiniLM-L6-v2-Q4_K_M.gguf"
)
RAG_DOWNLOAD_ID = "rag-embedding-model-minilm-q4km-v2"
RAG_MODEL_SETTING_KEY = "rag_engine.embedding_model_path"
RAG_VECTOR_KEY_SETTING = "rag_engine.vector_store_encryption_key"


class RagSetupPanel(QWidget):
    """Reusable RAG setup content used by onboarding and CONFIG dialog."""

    def __init__(
        self,
        *,
        vault: VaultManager | None,
        download_manager: DownloadManager | None,
        continue_text: str,
        on_back: Callable[[], None] | None = None,
        on_skip: Callable[[], None] | None = None,
        on_done: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vault = vault
        self._download_manager = download_manager
        self._on_back = on_back
        self._on_skip = on_skip
        self._on_done = on_done
        self._persist_task: asyncio.Task | None = None
        self._completing = False
        self._complete_no_path_since: float | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(10)

        title = QLabel("RAG Document Search")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #d8dce4;")
        root.addWidget(title)

        explainer = QLabel(
            "RAG lets local AI search your documents and inject relevant context.\n"
            "To enable this, download a local GGUF embedding model and confirm\n"
            "that document vectors are stored in your encrypted local vault database."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #9ea8b8; font-size: 12px;")
        root.addWidget(explainer)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #b6d9b6; font-size: 12px;")
        root.addWidget(self._status_label)

        self._download_checkbox = QCheckBox(
            "Download embedding model (all-MiniLM-L6-v2-Q4_K_M.gguf, ~21MB) from HuggingFace"
        )
        self._encrypt_checkbox = QCheckBox(
            "Store document embeddings locally in an encrypted vector database"
        )
        self._download_checkbox.toggled.connect(self._refresh_actions)
        self._encrypt_checkbox.toggled.connect(self._refresh_actions)
        root.addWidget(self._download_checkbox)
        root.addWidget(self._encrypt_checkbox)

        self._progress_wrap = QWidget(self)
        progress_layout = QVBoxLayout(self._progress_wrap)
        progress_layout.setContentsMargins(0, 4, 0, 0)
        progress_layout.setSpacing(4)
        self._progress_label = QLabel("Waiting to start download…")
        self._progress_label.setStyleSheet("color: #8ea58e;")
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._progress_bar)
        self._progress_wrap.hide()
        root.addWidget(self._progress_wrap)

        skip_note = QLabel(
            "Skip note: RAG will stay disabled until setup is complete. "
            "Use CONFIG on the RAG module card anytime to resume."
        )
        skip_note.setWordWrap(True)
        skip_note.setStyleSheet("color: #7e8796; font-size: 11px;")
        root.addWidget(skip_note)

        buttons = QHBoxLayout()
        if self._on_back is not None:
            back_btn = QPushButton("Back")
            back_btn.clicked.connect(self._on_back)
            buttons.addWidget(back_btn)
        if self._on_skip is not None:
            skip_btn = QPushButton("Skip")
            skip_btn.clicked.connect(self._on_skip)
            buttons.addWidget(skip_btn)
        buttons.addStretch(1)
        self._continue_btn = QPushButton(continue_text)
        self._continue_btn.clicked.connect(self._on_continue)
        buttons.addWidget(self._continue_btn)
        root.addLayout(buttons)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(300)
        self._poll_timer.timeout.connect(self._poll_download)

        self._found_model_path: str = ""
        self._probe_existing_files()

    # ── Probe for already-present dependencies & model ────────

    @staticmethod
    def _check_chromadb() -> tuple[bool, str]:
        try:
            import chromadb
            return True, getattr(chromadb, "__version__", "installed")
        except ImportError:
            return False, ""

    def _probe_existing_files(self) -> None:
        """Check chromadb and whether the model already exists locally."""
        self._found_model_path = ""

        chroma_ok, chroma_ver = self._check_chromadb()

        if self._download_manager is not None:
            row = self._download_manager.get(RAG_DOWNLOAD_ID)
            if (
                row
                and str(row.get("status", "")) == "complete"
                and str(row.get("repo_id") or "").strip() == RAG_REPO_ID
                and str(row.get("filename") or "").strip() == RAG_FILENAME
            ):
                p = str(row.get("local_path") or "").strip()
                if p and Path(p).exists():
                    self._found_model_path = p

        if not self._found_model_path:
            self._found_model_path = self._try_resolve_hf_cache()

        lines: list[str] = []

        if chroma_ok:
            lines.append(f"chromadb {chroma_ver} — installed")
        else:
            lines.append(
                "chromadb — NOT INSTALLED (required). "
                "Run: pip install chromadb"
            )

        if self._found_model_path:
            lines.append(f"Embedding model found:\n{self._found_model_path}")
            self._download_checkbox.setChecked(True)
            self._download_checkbox.setEnabled(False)
            self._download_checkbox.setText(
                f"Embedding model already downloaded ({RAG_FILENAME})"
            )
        else:
            lines.append(
                "Embedding model not found locally. Check both boxes, "
                "then click the button below to download."
            )

        all_good = chroma_ok and bool(self._found_model_path)
        self._status_label.setText("\n".join(lines))
        self._status_label.setStyleSheet(
            f"color: {'#3fb950' if all_good else '#d29922'}; font-size: 12px;"
        )

        self._chromadb_ok = chroma_ok
        self._refresh_actions()

    def _try_resolve_hf_cache(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            return ""
        cache_dir = None
        if self._download_manager is not None:
            root = getattr(self._download_manager, "_models_root", None)
            if root:
                cache_dir = str(root)
        try:
            resolved = hf_hub_download(
                repo_id=RAG_REPO_ID,
                filename=RAG_FILENAME,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            return ""
        p = str(resolved or "").strip()
        return p if p and Path(p).exists() else ""

    # ── Action gating ────────────────────────────────────────

    def _refresh_actions(self) -> None:
        all_checked = self._download_checkbox.isChecked() and self._encrypt_checkbox.isChecked()
        chroma_ok = getattr(self, "_chromadb_ok", True)
        self._continue_btn.setEnabled(all_checked and chroma_ok and not self._completing)
        if self._found_model_path:
            self._continue_btn.setText(
                self._continue_btn.text().replace("Download & ", "")
                if "Download" in self._continue_btn.text()
                else self._continue_btn.text()
            )

    # ── Continue button handler ──────────────────────────────

    def _on_continue(self) -> None:
        if self._found_model_path and Path(self._found_model_path).exists():
            self._persist_model_path(self._found_model_path)
            return
        self._start_download()

    def _start_download(self) -> None:
        if self._download_manager is None:
            QMessageBox.warning(
                self,
                "Download Manager Unavailable",
                "RAG setup cannot download the embedding model right now.",
            )
            return

        self._progress_wrap.show()
        self._progress_label.setText("Queueing model download…")
        self._progress_bar.setValue(0)
        self._download_manager.enqueue(
            download_id=RAG_DOWNLOAD_ID,
            repo_id=RAG_REPO_ID,
            filename=RAG_FILENAME,
            variant="q4_k_m",
            size="~21MB",
            url=RAG_URL,
        )
        self._complete_no_path_since = None
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_download(self) -> None:
        if self._download_manager is None:
            self._poll_timer.stop()
            return
        row = self._download_manager.get(RAG_DOWNLOAD_ID)
        if row is None:
            return
        status = str(row.get("status") or "queued")
        progress = float(row.get("progress") or 0.0)
        self._progress_bar.setValue(max(0, min(100, int(progress))))
        self._progress_label.setText(f"Download status: {status} ({progress:.1f}%)")
        if status == "complete":
            self._poll_timer.stop()
            self._progress_label.setText("Download complete — verifying files…")
            self._probe_existing_files()
            if self._found_model_path:
                self._persist_model_path(self._found_model_path)
            else:
                now = time.monotonic()
                if self._complete_no_path_since is None:
                    self._complete_no_path_since = now
                if (now - self._complete_no_path_since) < 4.0:
                    self._poll_timer.start()
                    return
                self._complete_no_path_since = None
                QMessageBox.warning(
                    self,
                    "RAG Setup Error",
                    "Download completed but no local model path was found.",
                )
        elif status == "failed":
            self._poll_timer.stop()
            self._complete_no_path_since = None
            err = str(row.get("error") or "Unknown error")
            QMessageBox.warning(self, "RAG Download Failed", err)
        else:
            self._complete_no_path_since = None

    def _persist_model_path(self, model_path: str) -> None:
        if self._completing:
            return
        if self._vault is None:
            QMessageBox.warning(
                self,
                "RAG Setup Unavailable",
                "Create an account first so settings can be stored in your vault.",
            )
            return
        self._completing = True
        self._continue_btn.setEnabled(False)
        self._progress_wrap.show()
        self._progress_label.setText("Saving model path and RAG encryption key to encrypted settings…")

        async def _save() -> None:
            store = SettingsStore(self._vault)
            await store.set(RAG_MODEL_SETTING_KEY, model_path)
            key = str(await store.get(RAG_VECTOR_KEY_SETTING) or "").strip()
            if not key:
                key = Fernet.generate_key().decode("ascii")
                await store.set(RAG_VECTOR_KEY_SETTING, key)

        def _finish_on_ui(error: Exception | None = None) -> None:
            if error is not None:
                self._completing = False
                self._refresh_actions()
                QMessageBox.warning(
                    self,
                    "RAG Setup Failed",
                    f"Model downloaded, but saving settings failed: {error}",
                )
                return
            if self._on_done is not None:
                self._on_done(model_path)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            async def _save_and_finish() -> None:
                try:
                    await _save()
                except Exception as exc:
                    _finish_on_ui(exc)
                    return
                _finish_on_ui()

            self._persist_task = asyncio.create_task(_save_and_finish())
        else:
            def _run_in_thread() -> None:
                try:
                    asyncio.run(_save())
                except Exception as exc:
                    QTimer.singleShot(0, lambda: _finish_on_ui(exc))
                    return
                QTimer.singleShot(0, lambda: _finish_on_ui())

            threading.Thread(
                target=_run_in_thread, name="rag-setup-save", daemon=True,
            ).start()


class RagSetupDialog(QDialog):
    """Standalone dialog that hosts the reusable RAG setup panel."""

    def __init__(
        self,
        *,
        vault: VaultManager,
        download_manager: DownloadManager | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RAG Setup")
        self.setModal(True)
        self.setMinimumSize(700, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._panel = RagSetupPanel(
            vault=vault,
            download_manager=download_manager,
            continue_text="Download & Save",
            on_skip=self.reject,
            on_done=lambda _p: self.accept(),
            parent=self,
        )
        layout.addWidget(self._panel)
