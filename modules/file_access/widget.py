"""
file_access/widget.py — File Access Card Controls
=================================================
Builds and manages the expanded UI controls on the File Access module card:
read/write mode switch, pointed folder display, and a monochrome LCD-style
scrollable file list with per-file access checkboxes.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import QFileDialog, QGraphicsRectItem, QGraphicsSimpleTextItem

from ui.rack_items import CardButtonItem, _LCD_FONT


class FileAccessCardBuilder:
    """Builds and manages File Access controls on the module card."""

    CONTROLS_HEIGHT: float = 182.0
    _VISIBLE_ROWS = 6

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._items: dict[str, dict[str, object]] = {}
        self._scroll_offsets: dict[str, int] = {}

    def clear(self) -> None:
        self._items.clear()
        self._scroll_offsets.clear()

    @property
    def tracked_module_ids(self) -> list[str]:
        return list(self._items.keys())

    def refresh_all(self) -> None:
        for module_id in list(self._items):
            self._refresh(module_id)

    def build_controls(
        self,
        card: ModuleCardItem,
        module: BaseModule,
        width: float,
        top_y: float,
        P: float = 0.0,
    ) -> None:
        module_id = module.module_id
        lx = 18 + P
        _sm = _LCD_FONT
        _btn_font = QFont("Consolas", 7)
        _btn_font.setStyleHint(QFont.StyleHint.Monospace)

        # Row 1: access mode toggle
        mode_btn = CardButtonItem(
            lx, top_y, 52, 16, "R/W",
            lambda: self._toggle_access_mode(module_id),
            card,
            label_font=_btn_font,
        )
        CardButtonItem(
            lx + 56, top_y, 52, 16, "SEND",
            lambda: self._send_active_files(module_id),
            card,
            label_font=_btn_font,
        )

        # Row 2: browse + pointed folder

        folder_val = QGraphicsSimpleTextItem("", card)
        folder_val.setFont(_sm)
        folder_val.setBrush(QBrush(QColor("#d8d8d8")))
        folder_val.setPos(lx + 52, top_y + 22)

        CardButtonItem(
            lx, top_y + 20, 48, 16, "BROWSE",
            lambda: self._browse_root(module_id),
            card,
            label_font=_btn_font,
        )

        # LCD list container
        panel_x = lx
        panel_y = top_y + 42
        panel_w = width - 2 * (18 + P)
        panel_h = 132.0

        panel = QGraphicsRectItem(panel_x, panel_y, panel_w, panel_h, card)
        grad = QLinearGradient(panel_x, panel_y, panel_x, panel_y + panel_h)
        grad.setColorAt(0.0, QColor("#0e0f10"))
        grad.setColorAt(1.0, QColor("#060707"))
        panel.setBrush(QBrush(grad))
        panel.setPen(QPen(QColor("#4f565f"), 1.0))
        panel.setZValue(2)

        title = QGraphicsSimpleTextItem("FILES", card)
        title.setFont(_sm)
        title.setBrush(QBrush(QColor("#c8d0d8")))
        title.setPos(panel_x + 8, panel_y + 4)
        title.setZValue(3)

        check_all_btn = CardButtonItem(
            panel_x + panel_w - 82, panel_y + 2, 58, 14, "CHECK ALL",
            lambda: self._check_all_in_folder(module_id),
            card,
            label_font=_btn_font,
        )
        check_all_btn.setToolTip("Select every file in this folder")

        # Scroll controls on right side
        up_btn = CardButtonItem(
            panel_x + panel_w - 22, panel_y + 2, 18, 14, "^",
            lambda: self._scroll(module_id, -1),
            card,
            label_font=_btn_font,
        )
        down_btn = CardButtonItem(
            panel_x + panel_w - 22, panel_y + panel_h - 16, 18, 14, "v",
            lambda: self._scroll(module_id, +1),
            card,
            label_font=_btn_font,
        )

        row_items: list[dict[str, object]] = []
        row_top = panel_y + 22
        row_h = 16.0
        for i in range(self._VISIBLE_ROWS):
            ry = row_top + i * row_h

            row_bg = QGraphicsRectItem(panel_x + 4, ry, panel_w - 30, 14, card)
            row_bg.setBrush(QBrush(QColor("#121315")))
            row_bg.setPen(QPen(QColor("#2f3338"), 0.6))
            row_bg.setZValue(2)

            cb_btn = CardButtonItem(
                panel_x + 6, ry + 1, 12, 12, " ",
                lambda ridx=i: self._toggle_visible_row(module_id, ridx),
                card,
                label_font=_btn_font,
            )
            cb_btn.setBrush(QBrush(QColor("#0d0e10")))
            cb_btn.setPen(QPen(QColor("#7d858f"), 0.8))

            file_txt = QGraphicsSimpleTextItem("", card)
            file_txt.setFont(_sm)
            file_txt.setBrush(QBrush(QColor("#d6dbe0")))
            file_txt.setPos(panel_x + 24, ry + 1)
            file_txt.setZValue(3)

            row_items.append({
                "cb_btn": cb_btn,
                "file_txt": file_txt,
            })

        self._items[module_id] = {
            "folder_val": folder_val,
            "mode_btn": mode_btn,
            "check_all_btn": check_all_btn,
            "up_btn": up_btn,
            "down_btn": down_btn,
            "rows": row_items,
        }
        self._scroll_offsets.setdefault(module_id, 0)
        self._refresh(module_id)

    def _resolve_root(self, module: BaseModule) -> Path:
        raw = str(module.params.get("root_path", ".")).strip() or "."
        p = Path(raw)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    @staticmethod
    def _tail_path(path: Path, max_len: int = 26) -> str:
        s = str(path)
        if len(s) <= max_len:
            return s
        return "..." + s[-(max_len - 3):]

    def _list_folder_files(self, root: Path) -> list[str]:
        if not root.exists() or not root.is_dir():
            return []
        files = [p.name for p in root.iterdir() if p.is_file()]
        files.sort(key=lambda n: n.lower())
        return files

    def _get_access_map(self, module: BaseModule) -> dict[str, bool]:
        raw = module.params.get("file_access_map", {})
        if isinstance(raw, dict):
            return {str(k): bool(v) for k, v in raw.items()}
        return {}

    def _refresh(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._items.get(module_id)
        if module is None or items is None:
            return

        mode = str(module.params.get("access_mode", "read_write"))
        if mode not in {"read_write", "read_only"}:
            mode = "read_write"
            module.params["access_mode"] = mode

        root = self._resolve_root(module)
        folder_val = items["folder_val"]
        if isinstance(folder_val, QGraphicsSimpleTextItem):
            folder_val.setText(self._tail_path(root))
            folder_val.setToolTip(str(root))

        mode_btn = items.get("mode_btn")
        if isinstance(mode_btn, CardButtonItem):
            self._style_mode_btn(mode_btn, mode)

        files = self._list_folder_files(root)
        check_all_btn = items.get("check_all_btn")
        if isinstance(check_all_btn, CardButtonItem):
            has_files = bool(files)
            check_all_btn.setOpacity(1.0 if has_files else 0.38)
            check_all_btn.setAcceptedMouseButtons(
                Qt.MouseButton.LeftButton if has_files else Qt.MouseButton.NoButton
            )
        access_map = self._get_access_map(module)

        offset = self._scroll_offsets.get(module_id, 0)
        max_offset = max(0, len(files) - self._VISIBLE_ROWS)
        offset = max(0, min(offset, max_offset))
        self._scroll_offsets[module_id] = offset

        rows = items["rows"]
        if isinstance(rows, list):
            visible = files[offset:offset + self._VISIBLE_ROWS]
            for ridx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                cb_btn = row.get("cb_btn")
                file_txt = row.get("file_txt")
                name = visible[ridx] if ridx < len(visible) else ""
                if isinstance(file_txt, QGraphicsSimpleTextItem):
                    file_txt.setText(name[:24] if name else "")
                    file_txt.setToolTip(name if name else "")
                    file_txt.setBrush(QBrush(QColor("#d6dbe0" if name else "#4d5258")))
                if isinstance(cb_btn, CardButtonItem):
                    checked = bool(access_map.get(name, False)) if name else False
                    self._style_checkbox(cb_btn, checked, enabled=bool(name))

        module.params["active_files"] = [name for name, allowed in access_map.items() if allowed]

    def _style_mode_btn(self, btn: CardButtonItem, mode: str) -> None:
        if mode == "read_write":
            btn.setBrush(QBrush(QColor("#1a2e1a")))
            btn.setPen(QPen(QColor("#39d353"), 1.0))
            label, color = "R/W", QColor("#b6d9b6")
        else:
            btn.setBrush(QBrush(QColor("#2e2a1a")))
            btn.setPen(QPen(QColor("#e8a838"), 1.0))
            label, color = "READ", QColor("#e8c878")
        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setText(label)
                child.setBrush(QBrush(color))

    def _style_checkbox(self, btn: CardButtonItem, checked: bool, enabled: bool) -> None:
        if not enabled:
            btn.setBrush(QBrush(QColor("#0c0d0e")))
            btn.setPen(QPen(QColor("#3a3e44"), 0.8))
            text = ""
            color = QColor("#4a4f55")
        elif checked:
            btn.setBrush(QBrush(QColor("#d9dee3")))
            btn.setPen(QPen(QColor("#aeb7c2"), 0.8))
            text = "x"
            color = QColor("#0b0d10")
        else:
            btn.setBrush(QBrush(QColor("#0d0e10")))
            btn.setPen(QPen(QColor("#7d858f"), 0.8))
            text = " "
            color = QColor("#d6dbe0")

        for child in btn.childItems():
            if isinstance(child, QGraphicsSimpleTextItem):
                child.setText(text)
                child.setBrush(QBrush(color))

    def _toggle_access_mode(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        cur = str(module.params.get("access_mode", "read_write"))
        module.params["access_mode"] = "read_only" if cur == "read_write" else "read_write"
        self._refresh(module_id)

    def _send_active_files(self, module_id: str) -> None:
        asyncio.create_task(self._send_active_files_async(module_id))

    async def _send_active_files_async(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        emitter = getattr(module, "emit_active_files", None)
        if not callable(emitter):
            self._controller.hv_log.error(f"{module_id}: file emit not supported by module logic")
            self._controller.statusBar().showMessage("File emit not available for this module", 2500)
            return
        try:
            result = await emitter()
            count = int(result.get("count", 0)) if isinstance(result, dict) else 0
            errors = result.get("errors", []) if isinstance(result, dict) else []
            self._controller.hv_log.info(
                f"{module_id}: emitted {count} active files ({len(errors)} errors)"
            )
            # FILES_OUT -> rag_engine runs across async/process boundaries.
            # Poll the peer briefly so parent-side rag params reflect the
            # worker's updated vector_count after ingestion completes.
            await self._sync_linked_rag_vector_counts(module_id)
            self._controller.statusBar().showMessage(
                f"Sent {count} files for indexing", 2500
            )
        except Exception as exc:
            self._controller.hv_log.error(f"{module_id}: failed to emit files: {exc}")
            self._controller.statusBar().showMessage(f"File emit failed: {exc}", 5000)

    def _linked_rag_module_ids(self, module_id: str) -> list[str]:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return []
        out = module.outputs.get("FILES_OUT")
        if out is None:
            return []
        hv = self._controller.hypervisor
        rag_ids: list[str] = []
        for wire in out.connected_wires:
            src = wire.source_port
            tgt = wire.target_port
            if src.owner_module_id == module_id and src.name == "FILES_OUT":
                peer_id = tgt.owner_module_id
            elif tgt.owner_module_id == module_id and tgt.name == "FILES_OUT":
                peer_id = src.owner_module_id
            else:
                continue
            peer = hv.active_modules.get(peer_id)
            if peer is None or peer.MODULE_NAME != "rag_engine":
                continue
            if peer_id not in rag_ids:
                rag_ids.append(peer_id)
        return rag_ids

    async def _sync_linked_rag_vector_counts(self, module_id: str) -> None:
        hv = self._controller.hypervisor
        rag_ids = self._linked_rag_module_ids(module_id)
        if not rag_ids:
            return

        attempts = 10
        for _ in range(attempts):
            for rag_id in rag_ids:
                try:
                    await hv.call_module_method(rag_id, "get_vector_count")
                except Exception as exc:
                    self._controller.hv_log.warning(
                        f"{module_id}: rag vector sync skipped for {rag_id}: {exc}"
                    )
            await asyncio.sleep(0.35)

    def _browse_root(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        base = self._resolve_root(module)
        chosen = QFileDialog.getExistingDirectory(
            self._controller,
            "Select file access root folder",
            str(base if base.exists() else base.parent),
        )
        if chosen:
            module.params["root_path"] = chosen
            self._scroll_offsets[module_id] = 0
            self._refresh(module_id)

    def _scroll(self, module_id: str, delta: int) -> None:
        offset = self._scroll_offsets.get(module_id, 0)
        self._scroll_offsets[module_id] = max(0, offset + delta)
        self._refresh(module_id)

    def _check_all_in_folder(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        root = self._resolve_root(module)
        files = self._list_folder_files(root)
        if not files:
            return
        access_map = self._get_access_map(module)
        for name in files:
            access_map[name] = True
        module.params["file_access_map"] = access_map
        self._refresh(module_id)
        auto_ingest = getattr(module, "auto_ingest_if_ready", None)
        if asyncio.iscoroutinefunction(auto_ingest):
            asyncio.create_task(auto_ingest())

    def _toggle_visible_row(self, module_id: str, visible_row_index: int) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        root = self._resolve_root(module)
        files = self._list_folder_files(root)
        offset = self._scroll_offsets.get(module_id, 0)
        idx = offset + visible_row_index
        if idx < 0 or idx >= len(files):
            return
        name = files[idx]
        access_map = self._get_access_map(module)
        access_map[name] = not bool(access_map.get(name, False))
        module.params["file_access_map"] = access_map
        self._refresh(module_id)
        auto_ingest = getattr(module, "auto_ingest_if_ready", None)
        if asyncio.iscoroutinefunction(auto_ingest):
            asyncio.create_task(auto_ingest())
