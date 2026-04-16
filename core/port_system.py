"""
port_system.py — The Pub/Sub Invisible Wiring Logic
=====================================================
Defines Ports, Wires, signal types, and the asynchronous
emit → route → receive data-flow engine.

Two connection modes:
  PUSH    — Unidirectional (OUT emits, IN receives).
  CHANNEL — Bidirectional (both sides can emit/receive payloads).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────

class PortDirection(str, Enum):
    IN = "IN"
    OUT = "OUT"


class ConnectionMode(str, Enum):
    """
    PUSH    — Standard one-way data flow (OUT → IN).
    CHANNEL — Bidirectional pipe; both endpoints can emit payloads.
    """
    PUSH = "PUSH"
    CHANNEL = "CHANNEL"


class SignalType(str, Enum):
    """Master catalogue of every payload type in the system."""
    DOCS_PAYLOAD = "DOCS_PAYLOAD"
    CONTEXT_PAYLOAD = "CONTEXT_PAYLOAD"
    QUERY_PAYLOAD = "QUERY_PAYLOAD"
    TOOL_SCHEMA_PAYLOAD = "TOOL_SCHEMA_PAYLOAD"
    TOOL_EXECUTION_PAYLOAD = "TOOL_EXECUTION_PAYLOAD"
    CHAT_REQUEST = "CHAT_REQUEST"
    BRAIN_STREAM_PAYLOAD = "BRAIN_STREAM_PAYLOAD"
    DB_QUERY_PAYLOAD = "DB_QUERY_PAYLOAD"
    DB_RESPONSE_PAYLOAD = "DB_RESPONSE_PAYLOAD"
    DB_TRANSACTION_PAYLOAD = "DB_TRANSACTION_PAYLOAD"
    ROUTING_CONFIG_PAYLOAD = "ROUTING_CONFIG_PAYLOAD"
    TUNNEL_STATUS_PAYLOAD = "TUNNEL_STATUS_PAYLOAD"
    SERVER_CONFIG_PAYLOAD = "SERVER_CONFIG_PAYLOAD"
    MODEL_LOAD_PAYLOAD = "MODEL_LOAD_PAYLOAD"
    SYSTEM_STATE_PAYLOAD = "SYSTEM_STATE_PAYLOAD"
    PORT_ERROR_PAYLOAD = "PORT_ERROR_PAYLOAD"
    AGENT_PAYLOAD = "AGENT_PAYLOAD"


# ── Payload Envelope ────────────────────────────────────────────

class Payload(BaseModel):
    """
    Universal envelope that wraps every piece of data flowing
    through the Miniloader rack.  Modules read `signal_type` to
    decide how to deserialise `data`.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType
    source_module: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data: dict[str, Any] = Field(default_factory=dict)


# ── Port ────────────────────────────────────────────────────────

class Port:
    """
    Represents a single jack on a module.

    Attributes
    ----------
    id              Globally unique identifier.
    name            Human-readable label (e.g. "BRAIN OUT").
    owner_module_id Module this port belongs to.
    direction       IN or OUT.
    accepted_signals Set of SignalType values this port will accept.
    connection_mode PUSH (one-way) or CHANNEL (bidirectional).
    max_connections How many wires can plug into this jack.
    connected_wires Active Wire objects attached to this port.
    _queue          asyncio.Queue for inbound payloads.
    _listeners      Callback list fired on each received payload.
    """

    def __init__(
        self,
        name: str,
        owner_module_id: str,
        direction: PortDirection,
        accepted_signals: set[SignalType],
        connection_mode: ConnectionMode = ConnectionMode.PUSH,
        max_connections: int = 1,
        description: str = "",
    ) -> None:
        self.id: str = str(uuid.uuid4())
        self.name = name
        self.owner_module_id = owner_module_id
        self.direction = direction
        self.accepted_signals = accepted_signals
        self.connection_mode = connection_mode
        self.max_connections = max_connections
        self.description = description

        self.connected_wires: list[Wire] = []
        self._queue: asyncio.Queue[Payload] = asyncio.Queue()
        self._listeners: list[Callable[[Payload], Any]] = []
        self._owner_module: Any = None  # back-ref set by Hypervisor

    # ── Emit (send a payload out of this port) ──────────────────

    async def emit(self, payload: Payload) -> None:
        """
        Push a payload to every wire connected to this port.
        For PUSH ports only the OUT side calls emit().
        For CHANNEL ports either side may call emit().

        Delivery is delegated to the Wire's transport so the same
        code works for in-process and cross-process connections.
        """
        for wire in self.connected_wires:
            await wire.send(payload, sender=self)

    async def emit_to(self, payload: Payload, *, target_module_id: str) -> bool:
        """
        Push a payload to a specific connected peer module.

        Returns True when a matching peer wire is found and the payload is sent.
        Returns False when no connected wire belongs to target_module_id.
        """
        for wire in self.connected_wires:
            peer = wire.source_port if wire.target_port is self else wire.target_port
            if peer.owner_module_id != target_module_id:
                continue
            await wire.send(payload, sender=self)
            return True
        return False

    # ── Receive (accept a payload into this port) ────────────────

    async def receive(self, payload: Payload) -> None:
        """
        Validate the incoming signal type, enqueue, and notify listeners.
        Silently discards the payload if the owning module is disabled.
        """
        if self._owner_module is not None and not self._owner_module.enabled:
            return
        if payload.signal_type not in self.accepted_signals:
            raise ValueError(
                f"Port '{self.name}' rejects signal {payload.signal_type}. "
                f"Accepted: {self.accepted_signals}"
            )
        await self._queue.put(payload)
        for listener in self._listeners:
            await listener(payload)

    def on_receive(self, callback: Callable[[Payload], Any]) -> None:
        """Register an async callback for every inbound payload."""
        self._listeners.append(callback)

    def __repr__(self) -> str:
        return (
            f"<Port {self.name} [{self.direction.value}] "
            f"mode={self.connection_mode.value} "
            f"owner={self.owner_module_id}>"
        )


# ── Wire ────────────────────────────────────────────────────────

class Wire:
    """
    An active connection between two ports.

    Attributes
    ----------
    id           Globally unique wire identifier.
    source_port  The OUT (or CHANNEL) port where data originates.
    target_port  The IN (or CHANNEL) port where data arrives.
    transport    Pluggable delivery backend (set by the Hypervisor).
                 ``None`` until the Hypervisor assigns one — at which
                 point it will be a LocalTransport or ProcessTransport.
    """

    def __init__(self, source_port: Port, target_port: Port) -> None:
        self.id: str = str(uuid.uuid4())
        self.source_port = source_port
        self.target_port = target_port
        self.transport: Any = None  # assigned by Hypervisor after creation

    async def send(self, payload: Payload, sender: Port) -> None:
        """Route a payload through whatever transport backs this wire."""
        if self.transport is not None:
            await self.transport.send(payload, sender)
        else:
            # Fallback: direct delivery (same as original behaviour)
            target = self.target_port if sender is self.source_port else self.source_port
            await target.receive(payload)

    def __repr__(self) -> str:
        return (
            f"<Wire {self.source_port.name} -> {self.target_port.name}>"
        )


# ── Port Registry ──────────────────────────────────────────────

class PortRegistry:
    """
    Global lookup table for every port in the rack.
    The Hypervisor owns one instance of this.
    """

    def __init__(self) -> None:
        self._ports: dict[str, Port] = {}

    def register(self, port: Port) -> None:
        self._ports[port.id] = port

    def unregister(self, port: Port) -> None:
        self._ports.pop(port.id, None)

    def get(self, port_id: str) -> Optional[Port]:
        return self._ports.get(port_id)

    def get_by_name(self, module_id: str, port_name: str) -> Optional[Port]:
        for p in self._ports.values():
            if p.owner_module_id == module_id and p.name == port_name:
                return p
        return None

    def all_ports(self) -> list[Port]:
        return list(self._ports.values())
