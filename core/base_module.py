"""
base_module.py — The Master Class Every Module Inherits From
=============================================================
Provides lifecycle hooks (initialize / process / shutdown),
automatic port registration, and health telemetry reporting.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    PortDirection,
    PortRegistry,
    SignalType,
)

if TYPE_CHECKING:
    from core.vault import VaultManager


class ModuleStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


class BaseModule(ABC):
    """
    Abstract base for every Miniloader plugin.

    Subclasses MUST implement:
        define_ports()  — declare input/output jacks
        initialize()    — one-time setup (load models, open files, etc.)
        process()       — handle an incoming payload
        shutdown()      — release resources

    Subclasses SHOULD override:
        get_default_params() — return a dict of default configuration values

    Subclasses MAY override:
        check_ready()   — return True when the module can fully operate
        init()          — one-time setup to reach the READY state
    """

    # ── Class-level metadata (override in subclasses) ───────────

    MODULE_NAME: str = "unnamed_module"
    MODULE_VERSION: str = "0.1.0"
    MODULE_DESCRIPTION: str = ""
    PROCESS_ISOLATION: bool = False
    UI_COL_SPAN: int = 1
    SENSITIVE_PARAMS: ClassVar[set[str]] = set()

    def __init__(self, module_id: Optional[str] = None) -> None:
        self.module_id: str = module_id or f"{self.MODULE_NAME}_{uuid.uuid4().hex[:6]}"
        self.status: ModuleStatus = ModuleStatus.IDLE
        self.enabled: bool = True
        self.params: dict[str, Any] = self.get_default_params()

        # Ports are populated by define_ports() during registration
        self.inputs: dict[str, Port] = {}
        self.outputs: dict[str, Port] = {}

        # Back-reference set by the Hypervisor after registration
        self._hypervisor: Any = None
        self._port_registry: Optional[PortRegistry] = None
        self._vault: VaultManager | None = None

    # ── Port helpers ────────────────────────────────────────────

    def add_input(
        self,
        name: str,
        accepted_signals: set[SignalType],
        connection_mode: ConnectionMode = ConnectionMode.PUSH,
        max_connections: int = 1,
        description: str = "",
    ) -> Port:
        port = Port(
            name=name,
            owner_module_id=self.module_id,
            direction=PortDirection.IN,
            accepted_signals=accepted_signals,
            connection_mode=connection_mode,
            max_connections=max_connections,
            description=description,
        )
        self.inputs[name] = port
        if self._port_registry:
            self._port_registry.register(port)
        return port

    def add_output(
        self,
        name: str,
        accepted_signals: set[SignalType],
        connection_mode: ConnectionMode = ConnectionMode.PUSH,
        max_connections: int = 8,
        description: str = "",
    ) -> Port:
        port = Port(
            name=name,
            owner_module_id=self.module_id,
            direction=PortDirection.OUT,
            accepted_signals=accepted_signals,
            connection_mode=connection_mode,
            max_connections=max_connections,
            description=description,
        )
        self.outputs[name] = port
        if self._port_registry:
            self._port_registry.register(port)
        return port

    # ── Vault access ─────────────────────────────────────────────

    def set_vault(self, vault: VaultManager | None) -> None:
        """Inject the VaultManager (called by the Hypervisor during registration)."""
        self._vault = vault

    @property
    def vault(self) -> VaultManager:
        """Access the vault.  Raises RuntimeError when running without one."""
        if self._vault is None:
            raise RuntimeError(
                f"{self.MODULE_NAME}: vault is not available "
                "(headless mode or vault not injected)"
            )
        return self._vault

    @property
    def has_vault(self) -> bool:
        """True if a VaultManager has been injected."""
        return self._vault is not None

    # ── Telemetry shortcut ──────────────────────────────────────

    async def report_state(
        self,
        severity: str = "INFO",
        message: str = "",
        ram_mb: float = 0,
        vram_mb: float = 0,
    ) -> None:
        """Emit a SYSTEM_STATE_PAYLOAD to the Hypervisor."""
        if self._hypervisor is None:
            return
        payload = Payload(
            signal_type=SignalType.SYSTEM_STATE_PAYLOAD,
            source_module=self.module_id,
            data={
                "severity": severity,
                "log_message": message,
                "telemetry": {
                    "module_ram_allocation_mb": ram_mb,
                    "module_vram_allocation_mb": vram_mb,
                    "is_healthy": self.status != ModuleStatus.ERROR,
                },
            },
        )
        await self._hypervisor.ingest_system_state(payload)

    # ── Lifecycle hooks (implement in subclasses) ───────────────

    @abstractmethod
    def define_ports(self) -> None:
        """Declare all input/output ports via add_input / add_output."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """One-time startup: load models, open sockets, etc."""
        ...

    @abstractmethod
    async def process(self, payload: Payload, source_port: Port) -> None:
        """Handle a single inbound payload on any of this module's ports."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release all resources (close files, kill subprocesses, etc.)."""
        ...

    # ── Readiness hooks (optional overrides) ────────────────────

    async def check_ready(self) -> bool:
        """Return True if this module has everything it needs to operate.

        Default returns True so modules with no external dependencies are
        automatically promoted to READY after boot.  Override to inspect
        files, params, wired dependencies, etc.
        """
        return True

    async def init(self) -> None:
        """One-time setup to bring the module into the READY state.

        Called by ``Hypervisor.init_all()`` (the INIT button) in
        topological order.  Use this to create files, tables, directories,
        or apply sensible defaults.  The base implementation is a no-op.
        """

    def get_default_params(self) -> dict[str, Any]:
        """Override to provide default configuration parameters."""
        return {}

    # ── Dunder ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<{self.MODULE_NAME} id={self.module_id} "
            f"status={self.status.value}>"
        )
