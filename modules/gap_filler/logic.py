"""
gap_filler/logic.py — Decorative Gap-Filler Module
====================================================
A blank-panel module with no ports and no logic.  Exists purely as a
visual spacer in the rack UI, displaying the miniloader logo,
copyright info, and spinning exhaust fans.
"""

from __future__ import annotations

from typing import Any

from core.base_module import BaseModule, ModuleStatus
from core.port_system import Payload, Port


class GapFillerModule(BaseModule):
    MODULE_NAME = "gap_filler"
    MODULE_VERSION = "0.1.0"
    MODULE_DESCRIPTION = "Decorative rack spacer panel"
    PROCESS_ISOLATION = False
    UI_COL_SPAN = 1

    def get_default_params(self) -> dict[str, Any]:
        return {}

    def define_ports(self) -> None:
        pass

    async def initialize(self) -> None:
        self.status = ModuleStatus.RUNNING

    async def process(self, payload: Payload, source_port: Port) -> None:
        pass

    async def shutdown(self) -> None:
        self.status = ModuleStatus.STOPPED


def register(hypervisor: Any) -> None:
    """Plugin entry-point called by the module scanner."""
    module = GapFillerModule()
    hypervisor.register_module(module)
