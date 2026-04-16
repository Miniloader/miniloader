"""
file_access/logic.py — Context and File Tool Access
======================================================
Manages local files and serves two concerns:
- FILES: bulk document extraction for RAG indexing
- MCP: on-demand AI-invoked file tools
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from core.base_module import BaseModule, ModuleStatus
from core.port_system import ConnectionMode, Payload, Port, SignalType

log = logging.getLogger(__name__)


class FileAccessModule(BaseModule):
    MODULE_NAME = "file_access"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Local file reader and text extraction engine"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "active_files": [],
            "root_path": ".",
            "access_mode": "read_write",
            "file_access_map": {},
        }

    # ── Ports ───────────────────────────────────────────────────

    def define_ports(self) -> None:
        self.add_output(
            "FILES_OUT",
            accepted_signals={SignalType.DOCS_PAYLOAD},
            connection_mode=ConnectionMode.PUSH,
            description=(
                "Emits extracted text from configured local files for "
                "downstream knowledge-engine indexing."
            ),
        )
        self.add_output(
            "TOOLS_OUT",
            accepted_signals={SignalType.TOOL_SCHEMA_PAYLOAD, SignalType.TOOL_EXECUTION_PAYLOAD},
            connection_mode=ConnectionMode.CHANNEL,
            max_connections=8,
            description=(
                "Bidirectional tools channel. Emits file tool schemas/results and accepts "
                "on-demand file operations from TOOLS_IN consumers."
            ),
        )

    # ── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        self.status = ModuleStatus.LOADING
        self._normalize_active_files()
        self._pending_auto_ingest: asyncio.Task[Any] | None = None
        self.outputs["TOOLS_OUT"]._listeners.clear()
        self.outputs["TOOLS_OUT"].on_receive(self._on_tools_payload)
        await self._emit_tool_schema()
        self.status = ModuleStatus.RUNNING

    async def process(self, payload: Payload, source_port: Port) -> None:
        # TOOLS_OUT inbound payloads are handled by a port listener.
        pass

    async def check_ready(self) -> bool:
        root = self.params.get("root_path", "")
        return bool(root) and Path(root).is_dir()

    async def init(self) -> None:
        root = self.params.get("root_path", "")
        if not root or not Path(root).is_dir():
            log.warning("file_access: root_path '%s' does not exist — browse for a folder", root)

    async def shutdown(self) -> None:
        task = getattr(self, "_pending_auto_ingest", None)
        if task is not None and not task.done():
            task.cancel()
        self.status = ModuleStatus.STOPPED

    async def emit_active_files(self) -> dict[str, Any]:
        """Extract configured active files and emit DOCS payload for RAG ingestion."""
        docs = []
        errors = []
        for p in self._active_paths():
            try:
                text = self._extract_text(p)
                if not text.strip():
                    continue
                docs.append(
                    {
                        "path": str(p),
                        "name": p.name,
                        "mime_type": self._guess_mime(p),
                        "text": text,
                        "char_count": len(text),
                    }
                )
            except Exception as exc:
                errors.append({"path": str(p), "error": str(exc)})

        if docs:
            await self.outputs["FILES_OUT"].emit(
                Payload(
                    signal_type=SignalType.DOCS_PAYLOAD,
                    source_module=self.module_id,
                    data={
                        "documents": docs,
                        "count": len(docs),
                        "errors": errors,
                    },
                )
            )
            await self.report_state(
                severity="INFO",
                message=f"file_access: emitted {len(docs)} documents ({len(errors)} errors)",
            )
        elif errors:
            await self.report_state(
                severity="WARN",
                message=f"file_access: no documents emitted ({len(errors)} extraction errors)",
            )
        return {"count": len(docs), "errors": errors}

    def _is_rag_linked(self) -> bool:
        hv = self._hypervisor
        if hv is None or not bool(getattr(hv, "system_powered", False)):
            return False
        if self.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
            return False

        files_out = self.outputs.get("FILES_OUT")
        if files_out is None:
            return False

        for wire in files_out.connected_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == self.module_id and src.name == "FILES_OUT":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == self.module_id and tgt.name == "FILES_OUT":
                peer_id = src.owner_module_id
            else:
                continue
            peer = hv.active_modules.get(peer_id)
            if peer is None or not peer.enabled:
                continue
            if peer.MODULE_NAME != "rag_engine":
                continue
            if peer.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            return True
        return False

    async def auto_ingest_if_ready(self) -> None:
        if not self._is_rag_linked():
            return
        if not self.params.get("active_files"):
            await self.report_state(
                severity="INFO",
                message=(
                    "file_access: RAG is linked, but no active files are selected. "
                    "Select one or more files to enable auto-ingest."
                ),
            )
            return
        if self._pending_auto_ingest is not None and not self._pending_auto_ingest.done():
            self._pending_auto_ingest.cancel()
        self._pending_auto_ingest = asyncio.create_task(self.emit_active_files())

    async def refresh_wiring_state(self) -> None:
        if self._is_rag_linked():
            if self.params.get("active_files"):
                await self.auto_ingest_if_ready()
            return
        if self._pending_auto_ingest is not None and not self._pending_auto_ingest.done():
            self._pending_auto_ingest.cancel()

    async def _on_tools_payload(self, payload: Payload) -> None:
        if payload.signal_type == SignalType.TOOL_SCHEMA_PAYLOAD:
            await self._emit_tool_schema()
            return
        if payload.signal_type != SignalType.TOOL_EXECUTION_PAYLOAD:
            return
        action = str(payload.data.get("action", "")).strip().lower()
        if action == "execute":
            await self._handle_tool_execute(payload)

    async def _emit_tool_schema(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "file_list",
                    "description": "List files under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative directory path from root_path"},
                            "recursive": {"type": "boolean", "description": "Recursively list files"},
                            "include_hidden": {
                                "type": "boolean",
                                "description": "Include hidden files and directories",
                            },
                            "include_dirs": {
                                "type": "boolean",
                                "description": "Include directory entries in the response",
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 2000,
                                "description": "Maximum number of entries to return",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read_text",
                    "description": "Read text from a file under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path from root_path"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_create_text",
                    "description": "Create a new text file under file_access root_path. Fails if file exists.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path from root_path"},
                            "text": {"type": "string", "description": "File content to write"},
                            "parents": {
                                "type": "boolean",
                                "description": "Create parent directories when missing (default: true)",
                            },
                        },
                        "required": ["path", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write_text",
                    "description": "Write text to a file under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path from root_path"},
                            "text": {"type": "string", "description": "File content to write"},
                            "create": {
                                "type": "boolean",
                                "description": "Create the file when missing (default: true)",
                            },
                            "parents": {
                                "type": "boolean",
                                "description": "Create parent directories when missing (default: true)",
                            },
                        },
                        "required": ["path", "text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_mkdir",
                    "description": "Create a directory under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative directory path from root_path"},
                            "parents": {
                                "type": "boolean",
                                "description": "Create missing parent directories (default: true)",
                            },
                            "exist_ok": {
                                "type": "boolean",
                                "description": "Do not fail when the directory already exists (default: true)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_delete",
                    "description": "Delete a file or directory under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file or directory path from root_path"},
                            "recursive": {
                                "type": "boolean",
                                "description": "Recursively delete directories (default: false)",
                            },
                            "missing_ok": {
                                "type": "boolean",
                                "description": "Do not fail if the target does not exist (default: false)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_stat",
                    "description": "Get file or directory metadata under file_access root_path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file or directory path from root_path"}
                        },
                        "required": ["path"],
                    },
                },
            },
        ]
        await self.outputs["TOOLS_OUT"].emit(
            Payload(
                signal_type=SignalType.TOOL_SCHEMA_PAYLOAD,
                source_module=self.module_id,
                data={
                    "provider": "file",
                    "connection_status": "connected",
                    "tools": tools,
                },
            )
        )

    async def _handle_tool_execute(self, payload: Payload) -> None:
        tool_name = str(payload.data.get("tool_name", "")).strip().lower()
        tool_call_id = str(payload.data.get("tool_call_id", "")).strip()
        arguments = payload.data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        try:
            if tool_name == "file_list":
                rel = str(arguments.get("path", ".")).strip() or "."
                recursive = self._coerce_bool(arguments.get("recursive", False))
                include_hidden = self._coerce_bool(arguments.get("include_hidden", False))
                include_dirs = self._coerce_bool(arguments.get("include_dirs", False))
                max_results = self._safe_int(arguments.get("max_results", 200), default=200, minimum=1, maximum=2000)
                root = self._resolve_root()
                base = self._resolve_under_root(rel)
                if not base.exists():
                    raise FileNotFoundError(f"Directory does not exist: {rel}")
                if not base.is_dir():
                    raise NotADirectoryError(f"Path is not a directory: {rel}")
                iterator = base.rglob("*") if recursive else base.iterdir()
                entries: list[dict[str, Any]] = []
                for entry in iterator:
                    rel_entry = entry.relative_to(root).as_posix()
                    if not include_hidden and any(part.startswith(".") for part in Path(rel_entry).parts):
                        continue
                    is_dir = entry.is_dir()
                    if is_dir and not include_dirs:
                        continue
                    stat = entry.stat()
                    entries.append(
                        {
                            "path": rel_entry,
                            "kind": "directory" if is_dir else "file",
                            "size_bytes": int(stat.st_size),
                            "modified_ts": float(stat.st_mtime),
                        }
                    )
                    if len(entries) >= max_results:
                        break
                result = {
                    "root": str(root),
                    "base_path": "." if base == root else base.relative_to(root).as_posix(),
                    "count": len(entries),
                    "entries": entries,
                }
            elif tool_name == "file_read_text":
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                result = {
                    "path": str(target),
                    "text": self._extract_text(target),
                }
            elif tool_name == "file_create_text":
                self._require_writable()
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                if target.exists():
                    raise FileExistsError(f"Path already exists: {rel}")
                create_parents = self._coerce_bool(arguments.get("parents", True), default=True)
                if create_parents:
                    target.parent.mkdir(parents=True, exist_ok=True)
                elif not target.parent.exists():
                    raise FileNotFoundError(f"Parent directory does not exist: {target.parent}")
                text = str(arguments.get("text", ""))
                target.write_text(text, encoding="utf-8")
                result = {"path": str(target), "created": True, "bytes_written": len(text.encode("utf-8"))}
            elif tool_name == "file_write_text":
                self._require_writable()
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                create_if_missing = self._coerce_bool(arguments.get("create", True), default=True)
                create_parents = self._coerce_bool(arguments.get("parents", True), default=True)
                existed = target.exists()
                if existed and target.is_dir():
                    raise IsADirectoryError(f"Path is a directory: {rel}")
                if not existed and not create_if_missing:
                    raise FileNotFoundError(f"File does not exist: {rel}")
                text = str(arguments.get("text", ""))
                if create_parents:
                    target.parent.mkdir(parents=True, exist_ok=True)
                elif not target.parent.exists():
                    raise FileNotFoundError(f"Parent directory does not exist: {target.parent}")
                target.write_text(text, encoding="utf-8")
                result = {
                    "path": str(target),
                    "created": not existed,
                    "updated": existed,
                    "bytes_written": len(text.encode("utf-8")),
                }
            elif tool_name == "file_mkdir":
                self._require_writable()
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                parents = self._coerce_bool(arguments.get("parents", True), default=True)
                exist_ok = self._coerce_bool(arguments.get("exist_ok", True), default=True)
                if target.exists() and target.is_file():
                    raise FileExistsError(f"Path exists and is a file: {rel}")
                target.mkdir(parents=parents, exist_ok=exist_ok)
                result = {"path": str(target), "created": True}
            elif tool_name == "file_delete":
                self._require_writable()
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                root = self._resolve_root()
                if target == root:
                    raise PermissionError("Refusing to delete root_path")
                recursive = self._coerce_bool(arguments.get("recursive", False))
                missing_ok = self._coerce_bool(arguments.get("missing_ok", False))
                if not target.exists():
                    if missing_ok:
                        result = {"path": str(target), "deleted": False, "missing": True}
                    else:
                        raise FileNotFoundError(f"Path does not exist: {rel}")
                elif target.is_dir():
                    if recursive:
                        shutil.rmtree(target)
                    else:
                        target.rmdir()
                    result = {"path": str(target), "deleted": True, "kind": "directory"}
                else:
                    target.unlink()
                    result = {"path": str(target), "deleted": True, "kind": "file"}
            elif tool_name == "file_stat":
                rel = str(arguments.get("path", "")).strip()
                if not rel:
                    raise ValueError("Missing required argument: path")
                target = self._resolve_under_root(rel)
                if not target.exists():
                    raise FileNotFoundError(f"Path does not exist: {rel}")
                stat = target.stat()
                result = {
                    "path": str(target),
                    "name": target.name,
                    "kind": "directory" if target.is_dir() else "file",
                    "size_bytes": int(stat.st_size),
                    "created_ts": float(stat.st_ctime),
                    "modified_ts": float(stat.st_mtime),
                    "is_hidden": target.name.startswith("."),
                }
            else:
                log.warning("file_access: received unknown tool execute request: %s", tool_name)
                await self._emit_tool_result(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error={
                        "code": "TOOL_NOT_FOUND",
                        "message": f"Unknown tool: {tool_name}",
                    },
                )
                return

            await self._emit_tool_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=True,
                result=result,
                error=None,
            )
        except Exception as exc:
            await self._emit_tool_result(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                success=False,
                result=None,
                error={"code": "TOOL_EXECUTION_FAILED", "message": str(exc)},
            )

    async def _emit_tool_result(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        result: Any,
        error: dict[str, Any] | None,
    ) -> None:
        await self.outputs["TOOLS_OUT"].emit(
            Payload(
                signal_type=SignalType.TOOL_EXECUTION_PAYLOAD,
                source_module=self.module_id,
                data={
                    "action": "result",
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "success": success,
                    "result": result,
                    "error": error,
                },
            )
        )

    def _require_writable(self) -> None:
        if str(self.params.get("access_mode", "read_write")).strip().lower() == "read_only":
            raise PermissionError("file_access is in read_only mode")

    def _resolve_root(self) -> Path:
        raw = str(self.params.get("root_path", ".")).strip() or "."
        p = Path(raw)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    def _normalize_active_files(self) -> None:
        active = self.params.get("active_files", [])
        if not isinstance(active, list):
            self.params["active_files"] = []
            return
        normalized = []
        for item in active:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            normalized.append(s)
        self.params["active_files"] = normalized

    def _active_paths(self) -> list[Path]:
        root = self._resolve_root()
        files = self.params.get("active_files", [])
        if not isinstance(files, list):
            return []
        out = []
        for raw in files:
            if not isinstance(raw, str):
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = root / raw
            try:
                rp = p.resolve()
            except Exception:
                continue
            if rp.exists() and rp.is_file():
                out.append(rp)
        return out

    def _resolve_under_root(self, rel_path: str) -> Path:
        root = self._resolve_root()
        target = (root / rel_path).resolve()
        if root != target and root not in target.parents:
            raise PermissionError("Path escapes configured root_path")
        return target

    @staticmethod
    def _coerce_bool(raw: Any, *, default: bool = False) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _safe_int(raw: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw)
        except Exception:
            value = default
        return max(minimum, min(maximum, value))

    def _extract_text(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except Exception as exc:
                raise RuntimeError(f"PDF support unavailable: {exc}") from exc
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if ext == ".docx":
            try:
                from docx import Document
            except Exception as exc:
                raise RuntimeError(f"DOCX support unavailable: {exc}") from exc
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        raise ValueError(f"Unsupported file extension: {ext or '[none]'}")

    def _guess_mime(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return "application/pdf"
        if ext == ".docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if ext in {".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml"}:
            return "text/plain"
        return "application/octet-stream"


def register(hypervisor: Any) -> None:
    module = FileAccessModule()
    hypervisor.register_module(module)
