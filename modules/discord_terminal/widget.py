"""
discord_terminal/widget.py — Discord Terminal Card Controls
===========================================================
Builds and manages the expanded UI controls on the Discord Terminal module card:
connection LEDs and quick status readouts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base_module import BaseModule
    from ui.main_window import RackWindow
    from ui.rack_items import ModuleCardItem

from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
    QInputDialog,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from core.base_module import ModuleStatus
from ui.rack_items import CardButtonItem, _LCD_FONT

_CHANNEL_ID_RE = re.compile(r"\d{15,25}")
_CHANNEL_URL_RE = re.compile(r"discord(?:app)?\.com/channels/(?:@me|\d{15,25})/(\d{15,25})")


class DiscordTerminalCardBuilder:
    """Builds and manages Discord terminal card controls."""

    CONTROLS_HEIGHT: float = 114.0

    def __init__(self, controller: RackWindow) -> None:
        self._controller = controller
        self._items: dict[str, dict[str, object]] = {}

    def clear(self) -> None:
        self._items.clear()

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

        title_font = QFont("Consolas", 8, QFont.Weight.Bold)
        title_font.setStyleHint(QFont.StyleHint.Monospace)
        val_font = QFont("Consolas", 8)
        val_font.setStyleHint(QFont.StyleHint.Monospace)
        btn_font = QFont("Consolas", 7)
        btn_font.setStyleHint(QFont.StyleHint.Monospace)

        # Row 1: bot connection
        bot_led = QGraphicsEllipseItem(lx, top_y + 2, 8, 8, card)
        bot_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        bot_led.setBrush(QBrush(QColor("#3a3f45")))

        bot_lbl = QGraphicsSimpleTextItem("BOT OFFLINE", card)
        bot_lbl.setFont(_sm)
        bot_lbl.setBrush(QBrush(QColor("#e24c4c")))
        bot_lbl.setPos(lx + 12, top_y)

        # Row 2: token and channel count
        token_key = QGraphicsSimpleTextItem("TOKEN", card)
        token_key.setFont(title_font)
        token_key.setBrush(QBrush(QColor("#8ea58e")))
        token_key.setPos(lx, top_y + 20)

        token_val = QGraphicsSimpleTextItem("MISSING", card)
        token_val.setFont(val_font)
        token_val.setBrush(QBrush(QColor("#e24c4c")))
        token_val.setPos(lx + 48, top_y + 20)

        chan_key = QGraphicsSimpleTextItem("CHANNELS", card)
        chan_key.setFont(title_font)
        chan_key.setBrush(QBrush(QColor("#8ea58e")))
        chan_key.setPos(lx + 140, top_y + 20)

        chan_val = QGraphicsSimpleTextItem("ALL", card)
        chan_val.setFont(val_font)
        chan_val.setBrush(QBrush(QColor("#d8d8d8")))
        chan_val.setPos(lx + 208, top_y + 20)

        CardButtonItem(
            width - P - 116.0,
            top_y + 18,
            34.0,
            14.0,
            "TOKEN",
            lambda: self._set_token(module_id),
            card,
            label_font=btn_font,
        )
        CardButtonItem(
            width - P - 78.0,
            top_y + 18,
            34.0,
            14.0,
            "CHAN",
            lambda: self._set_channels(module_id),
            card,
            label_font=btn_font,
        )
        CardButtonItem(
            width - P - 40.0,
            top_y + 18,
            36.0,
            14.0,
            "PERSONA",
            lambda: self._set_prompt(module_id),
            card,
            label_font=btn_font,
        )

        # Mention-mode toggle: right-aligned on row 1
        mention_btn = CardButtonItem(
            width - P - 76.0,
            top_y,
            72.0,
            14.0,
            "@MENTION: ON",
            lambda: self._toggle_mention(module_id),
            card,
            label_font=btn_font,
        )

        # Row 3: LLM + DB + RAG status lights
        llm_led = QGraphicsEllipseItem(lx, top_y + 42, 8, 8, card)
        llm_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        llm_led.setBrush(QBrush(QColor("#3a3f45")))

        llm_lbl = QGraphicsSimpleTextItem("LLM DISCONNECTED", card)
        llm_lbl.setFont(_sm)
        llm_lbl.setBrush(QBrush(QColor("#5c636e")))
        llm_lbl.setPos(lx + 12, top_y + 40)

        db_led = QGraphicsEllipseItem(lx, top_y + 62, 8, 8, card)
        db_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        db_led.setBrush(QBrush(QColor("#3a3f45")))

        db_lbl = QGraphicsSimpleTextItem("DATABASE OFFLINE", card)
        db_lbl.setFont(_sm)
        db_lbl.setBrush(QBrush(QColor("#5c636e")))
        db_lbl.setPos(lx + 12, top_y + 60)

        rag_led = QGraphicsEllipseItem(lx + 175, top_y + 62, 8, 8, card)
        rag_led.setPen(QPen(QColor("#1a1a1a"), 0.8))
        rag_led.setBrush(QBrush(QColor("#3a3f45")))

        rag_lbl = QGraphicsSimpleTextItem("RAG OFFLINE", card)
        rag_lbl.setFont(_sm)
        rag_lbl.setBrush(QBrush(QColor("#5c636e")))
        rag_lbl.setPos(lx + 187, top_y + 60)

        prompt_preview = QGraphicsSimpleTextItem("", card)
        prompt_preview.setFont(_sm)
        prompt_preview.setBrush(QBrush(QColor("#6f7782")))
        prompt_preview.setPos(lx, top_y + 80)

        self._items[module_id] = {
            "bot_led": bot_led,
            "bot_lbl": bot_lbl,
            "mention_btn": mention_btn,
            "token_val": token_val,
            "chan_val": chan_val,
            "llm_led": llm_led,
            "llm_lbl": llm_lbl,
            "db_led": db_led,
            "db_lbl": db_lbl,
            "rag_led": rag_led,
            "rag_lbl": rag_lbl,
            "prompt_preview": prompt_preview,
        }
        self._refresh(module_id)

    def _refresh(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        items = self._items.get(module_id)
        if module is None or items is None:
            return

        bot_online = bool(module.params.get("bot_connected", False))
        bot_user = str(module.params.get("bot_user", "")).strip()
        token_set = self._has_bot_token(module)
        channel_ids = self._parse_channel_ids(str(module.params.get("channel_ids", "")))
        llm_status = str(module.params.get("llm_status", "disconnected")).strip().lower()

        db_online = self._connected_module(module_id, "DB_IN_OUT", "database")
        rag_online = self._connected_module(
            module_id,
            "CONTEXT_IN",
            "rag_engine",
            require_running=True,
        )

        self._set_led(items["bot_led"], "#39d353" if bot_online else "#e24c4c")
        self._set_label(
            items["bot_lbl"],
            f"BOT ONLINE ({bot_user})" if bot_online and bot_user else "BOT OFFLINE",
            "#39d353" if bot_online else "#e24c4c",
        )

        require_mention = bool(module.params.get("require_mention", True))
        mention_btn = items.get("mention_btn")
        if isinstance(mention_btn, CardButtonItem):
            label_text = "@MENTION: ON" if require_mention else "@MENTION: OFF"
            for child in mention_btn.childItems():
                if isinstance(child, QGraphicsSimpleTextItem):
                    child.setText(label_text)
                    child.setBrush(QBrush(QColor("#39d353" if require_mention else "#e2c14c")))

        self._set_label(
            items["token_val"],
            "SET" if token_set else "MISSING",
            "#39d353" if token_set else "#e24c4c",
        )
        self._set_label(
            items["chan_val"],
            str(len(channel_ids)) if channel_ids else "ALL",
            "#d8d8d8",
        )

        llm_online = llm_status == "ready"
        llm_warn = llm_status == "unreachable"
        llm_color = "#39d353" if llm_online else ("#e2c14c" if llm_warn else "#5c636e")
        llm_text = {
            "ready": "LLM READY",
            "linked": "LLM LINKED",
            "unreachable": "LLM UNREACHABLE",
        }.get(llm_status, "LLM DISCONNECTED")
        self._set_led(items["llm_led"], llm_color)
        self._set_label(items["llm_lbl"], llm_text, llm_color)

        self._set_led(items["db_led"], "#39d353" if db_online else "#3a3f45")
        self._set_label(
            items["db_lbl"],
            "DATABASE ONLINE" if db_online else "DATABASE OFFLINE",
            "#39d353" if db_online else "#5c636e",
        )
        self._set_led(items["rag_led"], "#39d353" if rag_online else "#3a3f45")
        self._set_label(
            items["rag_lbl"],
            "RAG ENABLED" if rag_online else "RAG OFFLINE",
            "#39d353" if rag_online else "#5c636e",
        )

        prompt = str(module.params.get("system_prompt", "")).strip()
        preview = prompt[:60].replace("\n", " ")
        if len(prompt) > 60:
            preview += "..."
        self._set_label(
            items["prompt_preview"],
            f"PERSONA: {preview}" if preview else "PERSONA: (default)",
            "#8ea58e" if preview else "#5c636e",
        )

    def _set_token(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = str(module.params.get("discord_bot_token", "")).strip()
        if current == "__vault__":
            current = ""
        text, ok = QInputDialog.getText(
            self._controller,
            "Discord Terminal - Bot Token",
            "Paste your Discord bot token:",
            QLineEdit.EchoMode.Password,
            current,
        )
        if ok:
            token = text.strip()
            module.params["discord_bot_token"] = token
            if module._vault is not None:
                try:
                    module._vault.set_secret("discord_terminal.discord_bot_token", token)
                except Exception:
                    pass
            self._controller.statusBar().showMessage("Discord bot token updated", 2000)
        self._refresh(module_id)

    def _set_channels(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = str(module.params.get("channel_ids", "")).strip()
        text, ok = QInputDialog.getText(
            self._controller,
            "Discord Terminal - Allowed Channels",
            "Comma-separated channel IDs (blank = allow all):",
            QLineEdit.EchoMode.Normal,
            current,
        )
        if ok:
            parsed = self._parse_channel_ids(text)
            module.params["channel_ids"] = ",".join(str(i) for i in parsed)
            if parsed:
                self._controller.statusBar().showMessage(
                    f"Discord allowlist updated ({len(parsed)} channel{'s' if len(parsed) != 1 else ''})",
                    2500,
                )
            else:
                self._controller.statusBar().showMessage(
                    "Discord allowlist cleared (ALL channels allowed)",
                    2500,
                )
        self._refresh(module_id)

    def _toggle_mention(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = bool(module.params.get("require_mention", True))
        module.params["require_mention"] = not current
        state = "ON (only responds to @mentions)" if not current else "OFF (responds to all messages)"
        self._controller.statusBar().showMessage(f"@mention mode: {state}", 2500)
        self._refresh(module_id)

    def _set_prompt(self, module_id: str) -> None:
        module = self._controller.hypervisor.active_modules.get(module_id)
        if module is None:
            return
        current = str(module.params.get("system_prompt", "")).strip()

        dlg = QDialog(self._controller)
        dlg.setWindowTitle("Discord Terminal - System Prompt / Persona")
        dlg.setMinimumWidth(460)
        layout = QVBoxLayout(dlg)

        hint = QLabel(
            "Set the bot's personality. This is injected as the system prompt "
            "at the start of every conversation turn."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9ea8b8; font-size: 11px;")
        layout.addWidget(hint)

        editor = QTextEdit()
        editor.setPlainText(current)
        editor.setMinimumHeight(140)
        editor.setStyleSheet(
            "QTextEdit {"
            "  background: #0a100a; border: 1px solid #2a3a2a; border-radius: 4px;"
            "  color: #b6d9b6; padding: 6px; font-family: 'Consolas', monospace;"
            "  font-size: 11px; selection-background-color: #2a4a70;"
            "}"
            "QTextEdit:focus { border-color: #39d353; }"
        )
        editor.setPlaceholderText("You are a helpful assistant.")
        layout.addWidget(editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_prompt = editor.toPlainText().strip()
            module.params["system_prompt"] = new_prompt
            self._controller.statusBar().showMessage("System prompt updated", 2000)
        self._refresh(module_id)

    def _connected_module(
        self,
        module_id: str,
        port_name: str,
        peer_module_name: str,
        *,
        require_running: bool = False,
    ) -> bool:
        for wire in self._controller.hypervisor.active_wires:
            src_mid = wire.source_port.owner_module_id
            tgt_mid = wire.target_port.owner_module_id
            src_name = wire.source_port.name
            tgt_name = wire.target_port.name

            if src_mid == module_id and src_name == port_name:
                peer = self._controller.hypervisor.active_modules.get(tgt_mid)
            elif tgt_mid == module_id and tgt_name == port_name:
                peer = self._controller.hypervisor.active_modules.get(src_mid)
            else:
                continue

            if peer is None or not peer.enabled or peer.MODULE_NAME != peer_module_name:
                continue
            if require_running and peer.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            return True
        return False

    @staticmethod
    def _set_led(item: object, color: str) -> None:
        if isinstance(item, QGraphicsEllipseItem):
            item.setBrush(QBrush(QColor(color)))

    @staticmethod
    def _set_label(item: object, text: str, color: str) -> None:
        if isinstance(item, QGraphicsSimpleTextItem):
            item.setText(text)
            item.setBrush(QBrush(QColor(color)))

    @staticmethod
    def _parse_channel_ids(raw: str) -> list[int]:
        out: list[int] = []
        for token in re.split(r"[\s,]+", str(raw or "").strip()):
            token = token.strip()
            if not token:
                continue

            m = _CHANNEL_URL_RE.search(token)
            if m:
                try:
                    out.append(int(m.group(1)))
                except ValueError:
                    pass
                continue

            if token.startswith("<#") and token.endswith(">"):
                token = token[2:-1]
            for match in _CHANNEL_ID_RE.findall(token):
                try:
                    out.append(int(match))
                except ValueError:
                    continue

        return list(dict.fromkeys(out))

    @staticmethod
    def _has_bot_token(module: BaseModule) -> bool:
        if module._vault is not None:
            try:
                vt = str(module._vault.get_secret("discord_terminal.discord_bot_token") or "").strip()
                if vt:
                    return True
            except Exception:
                pass
        token = str(module.params.get("discord_bot_token", "")).strip()
        return bool(token) and token != "__vault__"
