from __future__ import annotations

import asyncio
import os
import sys

from PySide6.QtGui import QCloseEvent


class AppShellLifecycleMixin:
    def _request_exit(self) -> None:
        if self._shutdown_started:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._shutdown_and_quit())
        except RuntimeError:
            self._shutdown_started = True
            self.hide()
            self._primary_hypervisor._running = False

    def _request_logout(self) -> None:
        if self._shutdown_started:
            return
        asyncio.create_task(self._logout_and_restart())

    async def _shutdown_and_quit(self) -> None:
        if self._shutdown_started:
            return
        if hasattr(self, "_prompt_save_before_exit_if_needed"):
            proceed = await self._prompt_save_before_exit_if_needed()
            if not proceed:
                return
        self._shutdown_started = True
        if hasattr(self, "show_exit_overlay"):
            self.show_exit_overlay()
        self.statusBar().showMessage("Shutting down modules…")
        try:
            dm = getattr(self, "_download_manager", None)
            if dm is not None:
                dm.close()
            await self.hypervisor.shutdown()

            # Keep the primary loop alive during cleanup.
            self._primary_hypervisor._running = True

            for hv_id, task in list(self._background_hypervisor_tasks.items()):
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
                except Exception:
                    pass
                self._background_hypervisor_tasks.pop(hv_id, None)

            if self._primary_hypervisor is not self.hypervisor:
                await self._primary_hypervisor.shutdown()
                self._primary_hypervisor._running = True
        except Exception as exc:
            self.statusBar().showMessage(f"Shutdown encountered errors: {exc}", 4000)

        self.hide()
        self._primary_hypervisor._running = False

    async def _logout_and_restart(self) -> None:
        if self._vault is not None:
            self._vault.clear_keyring_key()
            self._vault.clear_memory_key()
        await self._shutdown_and_quit()
        os.execv(sys.executable, [sys.executable, *sys.argv])

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._shutdown_started:
            event.accept()
            return
        event.ignore()
        try:
            self._request_exit()
        except RuntimeError:
            event.accept()

