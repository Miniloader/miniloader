"""
core — Miniloader framework internals.

Exports the four pillars:
  • Hypervisor   – DAG manager, resource monitor, topological executor
  • BaseModule   – Abstract base every plugin inherits from
  • Port / Wire  – The pub/sub wiring and payload routing system
  • Transport    – Pluggable delivery layer (local / cross-process)
"""

from core.hypervisor import Hypervisor
from core.base_module import BaseModule
from core.port_system import (
    Port,
    Wire,
    PortDirection,
    ConnectionMode,
    SignalType,
    PortRegistry,
)
from core.transport import (
    Transport,
    LocalTransport,
    ProcessTransport,
)

__all__ = [
    "Hypervisor",
    "BaseModule",
    "Port",
    "Wire",
    "PortDirection",
    "ConnectionMode",
    "SignalType",
    "PortRegistry",
    "Transport",
    "LocalTransport",
    "ProcessTransport",
]
