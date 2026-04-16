from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
)

log = logging.getLogger(__name__)


class AppShellTemplateMixin:
    """
    Template management using the user's encrypted SQLite database.

    All user templates (shipped defaults + user-created) live in the
    ``templates`` table of ``~/.miniloader/{user_id}/miniloader_data.db``.
    Shipped defaults are seeded once on first account load; the user
    can delete them like any other template.
    """

    _LAST_OPEN_TEMPLATE_KEY = "last_open_template"
    _AUTO_SAVE_ON_CLOSE_KEY = "autosave_template_on_close"
    _ASK_SAVE_ON_CLOSE_KEY = "ask_to_save_template_on_close"

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "on"}

    def _get_template_preferences(self) -> dict[str, bool]:
        return {
            "autosave_on_close": bool(getattr(self, "_autosave_on_close", False)),
            "ask_save_on_close": bool(getattr(self, "_ask_save_on_close", True)),
        }

    async def _load_template_exit_preferences(self) -> None:
        store = self._settings_store()
        if store is None:
            self._autosave_on_close = False
            self._ask_save_on_close = True
            return
        try:
            raw = await store.get_json(self._AUTO_SAVE_ON_CLOSE_KEY)
        except Exception as exc:
            log.warning("template_mixin: failed to read autosave-on-close: %s", exc)
            raw = None
        self._autosave_on_close = self._coerce_bool(raw)
        try:
            ask_raw = await store.get_json(self._ASK_SAVE_ON_CLOSE_KEY)
        except Exception as exc:
            log.warning("template_mixin: failed to read ask-save-on-close: %s", exc)
            ask_raw = None
        if ask_raw is None:
            self._ask_save_on_close = True
        else:
            self._ask_save_on_close = self._coerce_bool(ask_raw)

    def _set_autosave_on_close_preference(self, enabled: bool) -> None:
        self._autosave_on_close = bool(enabled)
        asyncio.create_task(self._save_autosave_on_close_preference_async(bool(enabled)))

    async def _save_autosave_on_close_preference_async(self, enabled: bool) -> None:
        store = self._settings_store()
        if store is None:
            return
        try:
            await store.set(self._AUTO_SAVE_ON_CLOSE_KEY, bool(enabled))
        except Exception as exc:
            log.warning("template_mixin: failed to persist autosave-on-close: %s", exc)

    async def _save_ask_save_on_close_preference_async(self, enabled: bool) -> None:
        store = self._settings_store()
        if store is None:
            return
        try:
            await store.set(self._ASK_SAVE_ON_CLOSE_KEY, bool(enabled))
        except Exception as exc:
            log.warning("template_mixin: failed to persist ask-save-on-close: %s", exc)

    async def _ask_exit_save_choice(
        self,
        prompt: str,
        autosave_on_close_enabled: bool,
        ask_save_on_close_enabled: bool,
    ) -> tuple[QMessageBox.StandardButton, bool, bool]:
        """
        Show a non-blocking save/discard/cancel dialog during async shutdown.

        Using QMessageBox.question() inside qasync tasks can re-enter the event
        loop and trigger "Cannot enter into task" RuntimeError. This helper
        uses open() + signals so shutdown stays single-loop and safe.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[QMessageBox.StandardButton, bool, bool]] = loop.create_future()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Save Rack Template?")
        box.setText(prompt)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)

        autosave_toggle = QCheckBox("Autosave on close every time")
        autosave_toggle.setChecked(bool(autosave_on_close_enabled))
        autosave_toggle.setStyleSheet("color: #9aa8bc; padding-top: 4px;")
        box.setCheckBox(autosave_toggle)

        dont_ask_toggle = QCheckBox("Don't ask again")
        dont_ask_toggle.setChecked(not bool(ask_save_on_close_enabled))
        dont_ask_toggle.setStyleSheet("color: #9aa8bc; padding-top: 2px;")
        grid = box.layout()
        if grid is not None:
            row = grid.rowCount()
            grid.addWidget(dont_ask_toggle, row, 0, 1, grid.columnCount())

        def _resolve(button: QMessageBox.StandardButton) -> None:
            if not fut.done():
                fut.set_result((
                    button,
                    autosave_toggle.isChecked(),
                    dont_ask_toggle.isChecked(),
                ))

        def _on_clicked(btn) -> None:
            _resolve(box.standardButton(btn))

        def _on_finished(_result: int) -> None:
            # Fallback for close/Esc paths where buttonClicked may not fire.
            _resolve(QMessageBox.StandardButton.Cancel)
            box.deleteLater()

        box.buttonClicked.connect(_on_clicked)
        box.finished.connect(_on_finished)
        box.open()
        return await fut

    async def _ask_exit_template_name(self, current_name: str) -> str:
        """Prompt for a template name without nested modal event loops."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()

        dialog = QInputDialog(self)
        dialog.setWindowTitle("Save Template")
        dialog.setLabelText("Template name:")
        dialog.setTextValue(current_name)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)

        def _resolve(value: str) -> None:
            if not fut.done():
                fut.set_result(value)

        dialog.textValueSelected.connect(lambda text: _resolve(str(text).strip()))
        dialog.rejected.connect(lambda: _resolve(""))
        dialog.finished.connect(lambda _result: dialog.deleteLater())
        dialog.open()
        return await fut

    async def _ask_yes_no(self, title: str, prompt: str) -> bool:
        """Return True when user confirms Yes in a non-blocking dialog."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(prompt)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        box.setWindowModality(Qt.WindowModality.ApplicationModal)

        def _resolve(value: bool) -> None:
            if not fut.done():
                fut.set_result(value)

        def _on_clicked(btn) -> None:
            _resolve(box.standardButton(btn) == QMessageBox.StandardButton.Yes)

        def _on_finished(_result: int) -> None:
            _resolve(False)
            box.deleteLater()

        box.buttonClicked.connect(_on_clicked)
        box.finished.connect(_on_finished)
        box.open()
        return await fut

    def _template_store(self):
        """Return a TemplateStore for the active vault, or None."""
        if self._vault is None:
            return None
        from core.template_store import TemplateStore
        return TemplateStore(self._vault)

    def _settings_store(self):
        if self._vault is None:
            return None
        from core.settings_store import SettingsStore
        return SettingsStore(self._vault)

    # ── List population ──────────────────────────────────────────

    def _populate_template_list(self) -> None:
        """Schedule an async refresh of the sidebar template list."""
        asyncio.create_task(self._populate_template_list_async())

    async def _populate_template_list_async(self) -> None:
        """Fetch template names from DB and rebuild the sidebar list."""
        while self._template_list_layout.count():
            item = self._template_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        store = self._template_store()
        if store is None:
            lbl = QLabel("Sign in to manage templates")
            lbl.setStyleSheet("color: #4a5060; font-size: 10px; padding: 4px 2px;")
            self._template_list_layout.addWidget(lbl)
            return

        try:
            rows = await store.list_templates()
        except Exception as exc:
            log.warning("template_mixin: list_templates failed: %s", exc)
            rows = []

        if not rows:
            empty_lbl = QLabel("No templates saved yet")
            empty_lbl.setStyleSheet("color: #4a5060; font-size: 10px; padding: 4px 2px;")
            empty_lbl.setWordWrap(True)
            self._template_list_layout.addWidget(empty_lbl)
            self._template_list_layout.addStretch(1)
            return

        for row in rows:
            name: str = row["name"]
            is_default: bool = row["is_default"]
            label = f"{name} ●" if is_default else name
            btn = QPushButton(label)
            btn.setToolTip("Shipped default" if is_default else "User template")
            btn.setStyleSheet(
                "QPushButton {"
                "  background: #161a22; color: #a8b4c8; border: 1px solid #252d3a;"
                "  border-radius: 4px; padding: 5px 8px; text-align: left; font-size: 11px;"
                "}"
                "QPushButton:hover { background: #1c2230; color: #c8d8f0; }"
            )
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.clicked.connect(lambda _checked, n=name: asyncio.create_task(
                self._load_template(n)
            ))
            btn.customContextMenuRequested.connect(
                lambda pos, n=name, b=btn: self._template_context_menu(n, b)
            )
            self._template_list_layout.addWidget(btn)

        self._template_list_layout.addStretch(1)

    # ── Context menu ─────────────────────────────────────────────

    def _template_context_menu(self, name: str, btn: QPushButton) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1c2230; color: #c8d8f0; border: 1px solid #2e3a4e; }"
            "QMenu::item { padding: 5px 18px 5px 12px; }"
            "QMenu::item:selected { background: #253350; }"
        )
        load_action = menu.addAction("Load")
        menu.addSeparator()
        clone_action = menu.addAction("Clone")
        new_action = menu.addAction("New from current rack")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        selected = menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        if selected is None:
            return
        if selected == load_action:
            asyncio.create_task(self._load_template(name))
        elif selected == clone_action:
            asyncio.create_task(self._clone_template(name))
        elif selected == new_action:
            name = self._prompt_template_name()
            if name:
                asyncio.create_task(self._save_current_template_async(name))
        elif selected == delete_action:
            asyncio.create_task(self._delete_template(name))

    # ── CRUD operations ──────────────────────────────────────────

    async def _load_template(
        self,
        name: str,
        *,
        remember_last_open: bool = True,
        in_place: bool = False,
    ) -> bool:
        """Load a template from the DB, optionally on the current hypervisor."""
        from core.rack_state import RackSnapshot

        store = self._template_store()
        if store is None:
            self.statusBar().showMessage("No vault — cannot load template", 3000)
            return False

        try:
            payload = await store.get_template(name)
        except Exception as exc:
            self.statusBar().showMessage(f"Template read failed: {exc}", 4000)
            return False

        if payload is None:
            self.statusBar().showMessage(f"Template '{name}' not found", 3000)
            if remember_last_open:
                await self._set_last_open_template_name(None)
            return False

        try:
            snapshot = RackSnapshot.model_validate(payload)
        except Exception as exc:
            self.statusBar().showMessage(f"Template parse failed: {exc}", 4000)
            self._rack_page.hv_log.error(f"Template parse failed: {exc}")
            return False

        self.statusBar().showMessage(f"Loading template '{name}'…")
        self._rack_page.hv_log.info(f"Loading template: {name}")

        current_name = (self._current_template_name or "").strip()
        switching = bool(current_name) and current_name != name
        transition_label = (
            f"Switching template to '{name}'..."
            if switching
            else f"Loading template '{name}'..."
        )
        self._show_template_transition_overlay(transition_label)
        try:
            if in_place:
                new_order = await self.hypervisor.load_rack(snapshot, self._module_registry)
            else:
                fresh_hypervisor = await self._create_fresh_hypervisor_with_defaults()
                new_order = await fresh_hypervisor.load_rack(snapshot, self._module_registry)
                await self._swap_active_hypervisor(fresh_hypervisor)
        except Exception as exc:
            self.statusBar().showMessage(f"Template load error: {exc}", 4000)
            self._rack_page.hv_log.error(f"Template load error: {exc}")
            return False
        finally:
            self._hide_template_transition_overlay()

        self._rack_page._module_order = new_order
        self._rack_page._rebuild_layout()
        self._rack_page.hv_log.info(f"Template '{name}' loaded — press power to start")
        self.statusBar().showMessage(f"Template '{name}' loaded — press ⏻ to start", 4000)
        self._current_template_name = name
        if remember_last_open:
            await self._set_last_open_template_name(name)
        return True

    def _save_current_template(self) -> None:
        """Synchronous entry point from the sidebar Save button."""
        name = self._prompt_template_name()
        if not name:
            return
        asyncio.create_task(self._save_current_template_async(name))

    def _prompt_template_name(self) -> str:
        """Prompt synchronously for a template name from the Qt UI thread."""
        name, ok = QInputDialog.getText(self, "Save Template", "Template name:")
        if not ok or not name.strip():
            return ""
        return name.strip()

    async def _save_current_template_async(self, name: str, *, confirm_overwrite: bool = True) -> bool:
        """Save the current rack as a template using a pre-collected name."""
        store = self._template_store()
        if store is None:
            self.statusBar().showMessage("No vault — cannot save template", 3000)
            return False

        existing = await store.get_template(name)
        if existing is not None and confirm_overwrite:
            should_overwrite = await self._ask_yes_no(
                "Overwrite?",
                f"'{name}' already exists. Overwrite?",
            )
            if not should_overwrite:
                return False

        try:
            snapshot = self.hypervisor.snapshot_rack(
                self._rack_page._module_order, template_name=name
            )
            await store.save_template(name, snapshot.model_dump())
            self._populate_template_list()
            self.statusBar().showMessage(f"Template '{name}' saved", 2500)
            self._rack_page.hv_log.info(f"Template saved: {name}")
            self._current_template_name = name
            await self._set_last_open_template_name(name)
            return True
        except Exception as exc:
            self.statusBar().showMessage(f"Save failed: {exc}", 4000)
            self._rack_page.hv_log.error(f"Template save failed: {exc}")
            return False

    async def _clone_template(self, name: str) -> None:
        """Duplicate a template under a new name."""
        store = self._template_store()
        if store is None:
            return

        new_name, ok = QInputDialog.getText(
            self, "Clone Template", "New template name:", text=f"{name}_copy"
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()

        payload = await store.get_template(name)
        if payload is None:
            self.statusBar().showMessage(f"Template '{name}' not found", 3000)
            return

        existing = await store.get_template(new_name)
        if existing is not None:
            reply = QMessageBox.question(
                self, "Overwrite?", f"'{new_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        payload["template_name"] = new_name
        await store.save_template(new_name, payload)
        self._populate_template_list()
        self.statusBar().showMessage(f"Template cloned as '{new_name}'", 2500)

    async def _delete_template(self, name: str) -> None:
        """Delete a template after confirmation."""
        reply = QMessageBox.question(
            self,
            "Delete Template",
            f"Delete '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        store = self._template_store()
        if store is None:
            return

        try:
            await store.delete_template(name)
            self._populate_template_list()
            self.statusBar().showMessage(f"Template '{name}' deleted", 2500)
        except Exception as exc:
            self.statusBar().showMessage(f"Delete failed: {exc}", 4000)

    async def seed_default_templates(self) -> None:
        """Seed shipped defaults into the user's DB (idempotent)."""
        store = self._template_store()
        if store is None:
            return
        try:
            inserted = await store.seed_defaults()
            if inserted:
                self._populate_template_list()
                log.info("template_mixin: seeded %d default template(s)", inserted)
        except Exception as exc:
            log.warning("template_mixin: seed_defaults failed: %s", exc)

    async def _set_last_open_template_name(self, name: str | None) -> None:
        store = self._settings_store()
        if store is None:
            return
        try:
            if name:
                await store.set(self._LAST_OPEN_TEMPLATE_KEY, name)
            else:
                await store.delete(self._LAST_OPEN_TEMPLATE_KEY)
        except Exception as exc:
            log.warning("template_mixin: failed to persist last-open template: %s", exc)

    async def _get_last_open_template_name(self) -> str | None:
        store = self._settings_store()
        if store is None:
            return None
        try:
            value = await store.get(self._LAST_OPEN_TEMPLATE_KEY)
        except Exception as exc:
            log.warning("template_mixin: failed to read last-open template: %s", exc)
            return None
        if value is None:
            return None
        name = str(value).strip()
        return name or None

    @staticmethod
    def _normalized_snapshot_payload(payload: dict[str, Any]) -> str:
        modules = sorted(
            payload.get("modules", []),
            key=lambda m: (
                str(m.get("module_id", "")),
                str(m.get("module_type", "")),
            ),
        )
        wires = sorted(
            payload.get("wires", []),
            key=lambda w: (
                str(w.get("src_module_id", "")),
                str(w.get("src_port_name", "")),
                str(w.get("dst_module_id", "")),
                str(w.get("dst_port_name", "")),
            ),
        )
        normalized = {
            "modules": modules,
            "wires": wires,
            "module_order": list(payload.get("module_order", [])),
        }
        return json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    async def _is_current_rack_saved_as_template(self) -> bool:
        """True when the current rack matches the currently selected template."""
        if not self._current_template_name:
            return False
        store = self._template_store()
        if store is None:
            return True
        saved = await store.get_template(self._current_template_name)
        if saved is None:
            return False
        current_snapshot = self.hypervisor.snapshot_rack(
            self._rack_page._module_order,
            template_name=self._current_template_name,
        )
        current_payload = current_snapshot.model_dump()
        return (
            self._normalized_snapshot_payload(current_payload)
            == self._normalized_snapshot_payload(saved)
        )

    async def _restore_last_open_template_on_startup(self) -> None:
        name = await self._get_last_open_template_name()
        if not name:
            return
        loaded = await self._load_template(name, remember_last_open=False, in_place=True)
        if loaded:
            self.statusBar().showMessage(f"Restored last template '{name}'", 3500)

    async def _prompt_save_before_exit_if_needed(self) -> bool:
        """
        Ask whether to save current rack as a template when it differs from saved state.

        Returns True if shutdown should continue, False if it should be cancelled.
        """
        store = self._template_store()
        if store is None:
            return True

        try:
            is_saved = await self._is_current_rack_saved_as_template()
        except Exception as exc:
            log.warning("template_mixin: save-check failed during shutdown: %s", exc)
            is_saved = False

        if is_saved:
            if self._current_template_name:
                await self._set_last_open_template_name(self._current_template_name)
            return True

        current_name = (self._current_template_name or "").strip()
        if bool(getattr(self, "_autosave_on_close", False)):
            autosave_name = current_name or "autosave"
            return await self._save_current_template_async(
                autosave_name,
                confirm_overwrite=False,
            )
        if not bool(getattr(self, "_ask_save_on_close", True)):
            if self._current_template_name:
                await self._set_last_open_template_name(self._current_template_name)
            return True

        if current_name:
            prompt = (
                "The current rack has unsaved changes.\n\n"
                f"Save updates to template '{current_name}' before exiting?"
            )
        else:
            prompt = (
                "The current rack is not saved as a template.\n\n"
                "Save it as a template before exiting?"
            )

        choice, autosave_on_close_checked, dont_ask_checked = await self._ask_exit_save_choice(
            prompt,
            bool(getattr(self, "_autosave_on_close", False)),
            bool(getattr(self, "_ask_save_on_close", True)),
        )
        if autosave_on_close_checked != bool(getattr(self, "_autosave_on_close", False)):
            await self._save_autosave_on_close_preference_async(autosave_on_close_checked)
            self._autosave_on_close = autosave_on_close_checked
            if hasattr(self, "_settings_page") and hasattr(self._settings_page, "set_autosave_on_close_enabled"):
                self._settings_page.set_autosave_on_close_enabled(autosave_on_close_checked)
        ask_save_on_close = not dont_ask_checked
        if ask_save_on_close != bool(getattr(self, "_ask_save_on_close", True)):
            await self._save_ask_save_on_close_preference_async(ask_save_on_close)
            self._ask_save_on_close = ask_save_on_close
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.No:
            if self._current_template_name:
                await self._set_last_open_template_name(self._current_template_name)
            return True

        target_name = current_name or await self._ask_exit_template_name(current_name)
        if not target_name:
            return False
        return await self._save_current_template_async(target_name)

    # ── Rack control ─────────────────────────────────────────────

    def _request_reset_to_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "This will shut down all modules and reset the rack to factory defaults.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        asyncio.create_task(self._reset_to_defaults())

    async def _reset_to_defaults(self) -> None:
        self.statusBar().showMessage("Resetting to defaults…")
        self._rack_page.hv_log.warn("Resetting rack to defaults")
        self._show_template_transition_overlay("Unloading template and restoring defaults...")
        try:
            await self.hypervisor.clear_rack()

            module_plugins = getattr(self._rack_page, "_module_register_fns", {})
            for name, register_fn in module_plugins.items():
                try:
                    register_fn(self.hypervisor)
                except Exception as exc:
                    self._rack_page.hv_log.error(f"Failed to register {name}: {exc}")

            self._rack_page._module_order = list(self.hypervisor.active_modules.keys())
            self._rack_page._rebuild_layout()
            self._rack_page.hv_log.info(
                f"Reset complete — {len(self.hypervisor.active_modules)} modules with default settings"
            )
            self.statusBar().showMessage("Rack reset to defaults — press power to start", 4000)
        finally:
            self._hide_template_transition_overlay()

    def _show_template_transition_overlay(self, message: str) -> None:
        rack_page = getattr(self, "_rack_page", None)
        if rack_page is None:
            return
        show_fn = getattr(rack_page, "show_transition_overlay", None)
        if callable(show_fn):
            show_fn(message)

    def _hide_template_transition_overlay(self) -> None:
        rack_page = getattr(self, "_rack_page", None)
        if rack_page is None:
            return
        hide_fn = getattr(rack_page, "hide_transition_overlay", None)
        if callable(hide_fn):
            hide_fn()

    # ── Hypervisor helpers ───────────────────────────────────────

    async def _create_fresh_hypervisor_with_defaults(self):
        """Create a vault-aware hypervisor for template swap loads."""
        from core.hypervisor import Hypervisor

        fresh = Hypervisor(vault=self._vault)
        await fresh.initialize()
        return fresh

    async def _swap_active_hypervisor(self, fresh_hypervisor) -> None:
        old_hypervisor = self.hypervisor

        fresh_task = asyncio.create_task(fresh_hypervisor.run_event_loop())
        self._background_hypervisor_tasks[id(fresh_hypervisor)] = fresh_task

        if old_hypervisor is self._primary_hypervisor:
            await old_hypervisor.clear_rack()
        else:
            await old_hypervisor.shutdown()
            old_task = self._background_hypervisor_tasks.pop(id(old_hypervisor), None)
            if old_task is not None:
                try:
                    await asyncio.wait_for(old_task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    old_task.cancel()
                except Exception:
                    pass

        self.hypervisor = fresh_hypervisor
        self._rack_page.hypervisor = fresh_hypervisor
