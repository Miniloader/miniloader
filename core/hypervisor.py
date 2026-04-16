"""
hypervisor.py — The DAG Manager / Master Chassis
==================================================
Responsibilities:
  1. Module registry & plugin lifecycle management
  2. Port connection validation (direction, signal types, connection mode)
  3. Topological sort for boot-order execution
  4. System-state telemetry aggregation
  5. Async event-loop orchestration
  6. Process isolation for modules that request it

NOTE: Resource limiting (RAM / VRAM caps) has been intentionally disabled.
The attributes and method stubs remain for API compatibility but are inert.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.connection
import os
import socket
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from core.base_module import BaseModule, ModuleStatus
from core.port_system import (
    ConnectionMode,
    Payload,
    Port,
    PortDirection,
    PortRegistry,
    SignalType,
    Wire,
)
from core.probe_service import get_probe_service
from core.rack_state import (
    ModuleSnapshot,
    RackSnapshot,
    WireSnapshot,
    hydrate_sensitive_params,
    strip_sensitive_params,
)
from core.transport import LocalTransport, ProcessTransport

log = logging.getLogger(__name__)


@dataclass
class SecurityStatus:
    """Snapshot of the security posture across all layers."""
    tls_enabled: dict[str, bool] = field(default_factory=dict)
    auth_enabled: dict[str, bool] = field(default_factory=dict)
    wire_signing_active: bool = False
    module_comm_key_age_days: int = -1
    vault_locked: bool = True


class HypervisorError(Exception):
    """Base exception for all Hypervisor-level failures."""


class ConnectionError(HypervisorError):
    """Raised when a wire cannot be created."""


class ResourceLimitError(HypervisorError):
    """Kept for API compatibility — never raised (resource limiting disabled)."""


class Hypervisor:
    """
    The master motherboard.  Owns the module registry, the port
    registry, the active wire list, and the global resource limits.
    Manages child processes for modules that set PROCESS_ISOLATION.
    """

    def __init__(
        self,
        vault: Any | None = None,
    ) -> None:
        # ── Registries ──────────────────────────────────────────
        self.port_registry = PortRegistry()
        self.active_modules: dict[str, BaseModule] = {}
        self.active_wires: list[Wire] = []

        # ── Vault (injected from main) ──────────────────────────
        self._vault = vault

        # ── Resource limits (disabled — kept as inert attributes) ─
        self.max_ram_mb: float = 0
        self.max_vram_mb: float = 0

        # ── Telemetry log ───────────────────────────────────────
        self.system_log: deque[Payload] = deque(maxlen=500)

        # ── Inference log (brain module cards) ───────────────────
        self._inference_log_queue: deque[tuple] = deque(maxlen=50)

        # ── Internal event loop flag ────────────────────────────
        self._running = False

        # Rack power state.
        # True after boot_all(); False after graceful_stop_all() or load_rack().
        # Used by the UI power button to restore state after layout rebuilds.
        self.system_powered: bool = False

        # Per-module power memory: stores each module's enabled state
        # before a hypervisor power-off so boot_all() can restore only
        # the modules that were previously on.
        self._module_power_memory: dict[str, bool] = {}

        # ── Process isolation bookkeeping ───────────────────────
        self._workers: dict[str, _WorkerHandle] = {}

        # ── Registered process registry (for UI display and kill) ─────────────
        # Maps pid (int) -> {"module_id": str, "port": int, "label": str}
        self._registered_processes: dict[int, dict[str, Any]] = {}

    @property
    def vault(self) -> Any:
        """The VaultManager instance, or None in headless mode."""
        return self._vault

    def set_stored_backend(self, backend: str | None) -> None:
        """Cache the user's selected backend for worker spawning."""
        self._stored_backend = backend

    def _get_stored_backend(self) -> str | None:
        """Return the cached user backend selection, or None."""
        return getattr(self, "_stored_backend", None)

    def get_security_status(self) -> SecurityStatus:
        """Build a snapshot of the current security posture."""
        tls: dict[str, bool] = {}
        auth: dict[str, bool] = {}
        http_modules = {"gpt_server", "gpt_terminal", "cloud_brain"}

        for module in self.active_modules.values():
            if module.MODULE_NAME in http_modules:
                tls[module.MODULE_NAME] = bool(module.params.get("tls_enabled", False))
                auth[module.MODULE_NAME] = self._vault is not None

        wire_signing = (
            self._vault is not None
            and bool(self._vault.get_module_comm_key())
        )

        key_age = -1

        return SecurityStatus(
            tls_enabled=tls,
            auth_enabled=auth,
            wire_signing_active=wire_signing,
            module_comm_key_age_days=key_age,
            vault_locked=self._vault is None,
        )

    # ── Process registry ───────────────────────────────────────

    def register_process(
        self,
        pid: int,
        module_id: str,
        module_type: str,
        port: int,
        label: str,
        *,
        internal_port: int | None = None,
    ) -> None:
        """Register a child process (e.g. Node) for UI display and optional kill."""
        self._registered_processes[pid] = {
            "module_id": module_id,
            "module_type": module_type,
            "port": port,
            "label": label,
            "internal_port": internal_port,
        }

    def unregister_process(self, pid: int) -> None:
        """Remove a process from the registry (e.g. after it exits)."""
        self._registered_processes.pop(pid, None)

    def get_registered_processes(self) -> list[tuple[int, dict[str, Any]]]:
        """Return list of (pid, info) for all registered processes."""
        return list(self._registered_processes.items())

    def kill_registered_process(self, pid: int) -> bool:
        """Kill a registered process by PID. Returns True if killed."""
        try:
            if pid not in self._registered_processes:
                return False
            if os.name == "nt":
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.kill(pid, 9)
            self._registered_processes.pop(pid, None)
            return True
        except Exception:
            self._registered_processes.pop(pid, None)
            return False

    def kill_registered_processes_for_module_type(self, module_type: str) -> list[int]:
        """Kill all registered processes for a given module type (e.g. gpt_terminal)."""
        pids_to_kill = [
            p for p, info in self._registered_processes.items()
            if info.get("module_type") == module_type
        ]
        for pid in pids_to_kill:
            try:
                if os.name == "nt":
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    os.kill(pid, 9)
            except Exception:
                pass
            self._registered_processes.pop(pid, None)
        return pids_to_kill

    def get_local_port_bindings(self) -> list[tuple[int, int, str]]:
        """
        Return list of (port, pid, process_name) for processes listening on 127.0.0.1.
        """
        result: list[tuple[int, int, str]] = []
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.status != psutil.CONN_LISTEN:
                    continue
                laddr = getattr(conn, "laddr", None)
                if not laddr:
                    continue
                addr = laddr[0] if isinstance(laddr, (list, tuple)) else getattr(laddr, "ip", None)
                port = laddr[1] if isinstance(laddr, (list, tuple)) else getattr(laddr, "port", 0)
                if addr != "127.0.0.1":
                    continue
                pid = getattr(conn, "pid", None) or 0
                name = ""
                if pid:
                    try:
                        p = psutil.Process(pid)
                        name = p.name() or ""
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                result.append((port, pid, name))
            result.sort(key=lambda x: x[0])
        except (psutil.AccessDenied, AttributeError):
            pass
        return result

    # ── Initialisation ──────────────────────────────────────────

    async def initialize(self) -> None:
        """Boot-time setup for the Hypervisor itself."""
        self._running = True

    # ── Module registration ─────────────────────────────────────

    def register_module(self, module: BaseModule) -> None:
        """
        Add a module to the rack.  Wires up the port registry
        and back-references so the module can emit telemetry.

        If the module declares PROCESS_ISOLATION = True the actual
        instance will later be hosted in a child process.  We still
        register its ports here so the Hypervisor can validate
        connections before any process is spawned.
        """
        module._hypervisor = self
        module._port_registry = self.port_registry
        module.set_vault(self._vault)
        module.define_ports()

        for port in list(module.inputs.values()) + list(module.outputs.values()):
            port._owner_module = module
            self.port_registry.register(port)

        self.active_modules[module.module_id] = module

    # ── Connection validation ───────────────────────────────────

    def connect_ports(self, source_id: str, target_id: str) -> Wire:
        """
        Validate and create a wire between two ports.

        Rules
        -----
        1. Directionality: source must be OUT (or CHANNEL), target must be IN (or CHANNEL).
        2. Signal compatibility: at least one accepted signal type must overlap.
        3. Connection limits: target must not exceed max_connections.
        4. No duplicate wires.

        The transport is assigned automatically:
        - LocalTransport when both modules live in the core process.
        - ProcessTransport when one end is in a worker process
          (transport is attached later during ``_spawn_worker``).
        """
        source = self.port_registry.get(source_id)
        target = self.port_registry.get(target_id)

        if source is None or target is None:
            raise ConnectionError("One or both port IDs are invalid.")

        valid_source = source.direction == PortDirection.OUT or source.connection_mode == ConnectionMode.CHANNEL
        valid_target = target.direction == PortDirection.IN or target.connection_mode == ConnectionMode.CHANNEL
        if not valid_source or not valid_target:
            raise ConnectionError(
                f"Invalid direction: {source.direction.value} -> {target.direction.value}. "
                "Must connect OUT->IN or use CHANNEL mode."
            )

        shared_signals = source.accepted_signals & target.accepted_signals
        if not shared_signals:
            raise ConnectionError(
                f"Signal type mismatch: {source.accepted_signals} vs {target.accepted_signals}."
            )

        if len(target.connected_wires) >= target.max_connections:
            raise ConnectionError(
                f"Port '{target.name}' already at max connections ({target.max_connections})."
            )

        # CHANNEL links are bidirectional over a single Wire object, so
        # A->B and B->A must be considered the same connection.
        req_pair = {source.id, target.id}
        for w in self.active_wires:
            existing_pair = {w.source_port.id, w.target_port.id}
            if existing_pair == req_pair:
                raise ConnectionError(
                    "Duplicate wire — this port pair is already connected."
                )

        wire = Wire(source, target)
        source.connected_wires.append(wire)
        target.connected_wires.append(wire)
        self.active_wires.append(wire)

        if source.connection_mode == ConnectionMode.CHANNEL or target.connection_mode == ConnectionMode.CHANNEL:
            source.connection_mode = ConnectionMode.CHANNEL
            target.connection_mode = ConnectionMode.CHANNEL

        src_module = source.owner_module_id
        tgt_module = target.owner_module_id
        src_isolated = src_module in self._workers
        tgt_isolated = tgt_module in self._workers

        if not src_isolated and not tgt_isolated:
            wire.transport = LocalTransport(source, target)

        self._set_affected_modules_standby(source, target)
        self._schedule_wiring_state_refresh(source, target)
        self._schedule_routing_config_replay(source, target)
        self._schedule_server_config_replay(source, target)
        return wire

    def _schedule_server_config_replay(self, source: Port, target: Port) -> None:
        """
        When a local-IP/server channel is (re)connected while powered on,
        ask provider modules to replay their server config.
        """
        if not self.system_powered:
            return

        local_port_names = {"LOCAL_IP_IN", "LOCAL_IP_OUT", "SERVER_IN", "SERVER_OUT"}
        if source.name not in local_port_names and target.name not in local_port_names:
            return

        module_ids = {source.owner_module_id, target.owner_module_id}
        replay_coros = []
        for module_id in module_ids:
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            replay = getattr(module, "broadcast_server_config", None)
            if callable(replay):
                replay_coros.append(replay())

        if not replay_coros:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for coro in replay_coros:
            loop.create_task(coro)

    def _schedule_routing_config_replay(self, source: Port, target: Port) -> None:
        """Replay routing config when WEB tunnel channels are connected."""
        if not self.system_powered:
            return
        web_port_names = {"WEB_IN", "WEB_OUT", "WEB"}
        if source.name not in web_port_names and target.name not in web_port_names:
            return

        module_ids = {source.owner_module_id, target.owner_module_id}
        replay_coros = []
        for module_id in module_ids:
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            replay = getattr(module, "broadcast_routing_config", None)
            if callable(replay):
                replay_coros.append(replay())

        if not replay_coros:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for coro in replay_coros:
            loop.create_task(coro)

    def disconnect_wire(self, wire_id: str) -> None:
        """Remove a wire by ID and clean up port references."""
        wire = next((w for w in self.active_wires if w.id == wire_id), None)
        if wire is None:
            raise ConnectionError(f"Wire {wire_id} not found.")

        source = wire.source_port
        target = wire.target_port
        if wire in wire.source_port.connected_wires:
            wire.source_port.connected_wires.remove(wire)
        if wire in wire.target_port.connected_wires:
            wire.target_port.connected_wires.remove(wire)
        if wire in self.active_wires:
            self.active_wires.remove(wire)
        self._set_affected_modules_standby(source, target)
        self._schedule_wiring_state_refresh(source, target)

    def _set_affected_modules_standby(self, source: Port, target: Port) -> None:
        """Demote READY modules to RUNNING (standby) when their wiring changes."""
        if not self.system_powered:
            return
        for module_id in (source.owner_module_id, target.owner_module_id):
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status == ModuleStatus.READY:
                module.status = ModuleStatus.RUNNING

    def _schedule_wiring_state_refresh(self, source: Port, target: Port) -> None:
        """Refresh wiring-derived module state for LOCAL_IP/SERVER channel changes."""
        if not self.system_powered:
            return

        local_port_names = {
            "LOCAL_IP_IN", "LOCAL_IP_OUT", "SERVER_IN", "SERVER_OUT",
            "WEB_IN", "WEB_OUT", "WEB",
            "FILES_OUT", "FILES_IN",
            "AGENT_IN", "AGENT_OUT",
            "API_IN", "LLM_IN", "TOOLS_IN",
        }
        if source.name not in local_port_names and target.name not in local_port_names:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        module_ids = {source.owner_module_id, target.owner_module_id}
        for module_id in module_ids:
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            refresh = getattr(module, "refresh_wiring_state", None)
            if asyncio.iscoroutinefunction(refresh):
                loop.create_task(refresh())

    # ── Topological boot order ──────────────────────────────────

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """
        Build an adjacency set: module A depends on module B if any
        of A's IN ports are wired to B's OUT ports.

        For CHANNEL wires the Wire's source_port/target_port assignment
        is arbitrary (whichever end the user clicked first), so we
        normalise using each port's declared PortDirection instead.
        Wires between two same-direction CHANNEL ports are skipped —
        they don't imply a meaningful boot-order dependency.
        """
        deps: dict[str, set[str]] = defaultdict(set)
        for mid in self.active_modules:
            deps[mid]  # ensure every module appears even if no deps

        for wire in self.active_wires:
            port_a = wire.source_port
            port_b = wire.target_port

            if port_a.direction == PortDirection.OUT and port_b.direction == PortDirection.IN:
                provider_module = port_a.owner_module_id
                consumer_module = port_b.owner_module_id
            elif port_b.direction == PortDirection.OUT and port_a.direction == PortDirection.IN:
                provider_module = port_b.owner_module_id
                consumer_module = port_a.owner_module_id
            else:
                # Both same direction (e.g. two CHANNEL/IN ports) — no
                # unambiguous boot-order dependency; skip to avoid cycles.
                continue

            if consumer_module != provider_module:
                deps[consumer_module].add(provider_module)

        return deps

    def topological_sort(self) -> list[str]:
        """
        Kahn's algorithm.  Returns module IDs in safe boot order.
        Raises HypervisorError on cycles.
        """
        deps = self._build_dependency_graph()
        in_degree: dict[str, int] = {m: 0 for m in deps}
        reverse: dict[str, list[str]] = defaultdict(list)

        for node, parents in deps.items():
            in_degree[node] = len(parents)
            for parent in parents:
                reverse[parent].append(node)

        queue = deque(m for m, d in in_degree.items() if d == 0)
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for child in reverse[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(deps):
            raise HypervisorError(
                "Cycle detected in the module graph — cannot determine boot order."
            )
        return order

    # ── Process isolation ───────────────────────────────────────

    async def _persist_worker_sensitive_params(self, module: BaseModule) -> None:
        """Write any worker-generated sensitive params back to SettingsStore.

        When an isolated module auto-generates values during initialize()
        (e.g. a Fernet key), those values flow back in the init response
        and are merged into the parent's params.  This method persists
        them to the encrypted settings DB so they survive restarts.
        """
        if self._vault is None:
            return
        sensitive = getattr(module.__class__, "SENSITIVE_PARAMS", set())
        if not sensitive:
            return
        from core.rack_state import VAULT_SENTINEL
        from core.settings_store import SettingsStore

        store = SettingsStore(self._vault)
        for key in sensitive:
            value = str(module.params.get(key, "")).strip()
            if not value or value == VAULT_SENTINEL:
                continue
            setting_key = f"{module.MODULE_NAME}.{key}"
            try:
                await store.set(setting_key, value)
            except Exception:
                log.warning(
                    "Failed to persist %s to settings store", setting_key,
                    exc_info=True,
                )

    async def _hydrate_settings_store_params(self, module: BaseModule) -> None:
        """Resolve missing/sentinel params from SettingsStore before spawn.

        Workers have no vault access, so any param that was stored via
        ``SettingsStore`` (RAG keys, model paths, etc.) must be resolved
        here and injected into ``module.params`` before the process
        starts.
        """
        if self._vault is None:
            return
        from core.rack_state import VAULT_SENTINEL
        from core.settings_store import SettingsStore

        sensitive = getattr(module.__class__, "SENSITIVE_PARAMS", set())
        keys_to_check = set(sensitive)
        keys_to_check.add("embedding_model_path")

        needs_store = any(
            module.params.get(k) in (VAULT_SENTINEL, "", None)
            for k in keys_to_check
            if k in module.params or k in sensitive
        )
        if not needs_store:
            return
        store = SettingsStore(self._vault)
        for key in keys_to_check:
            current = module.params.get(key)
            if current not in (VAULT_SENTINEL, "", None):
                continue
            setting_key = f"{module.MODULE_NAME}.{key}"
            try:
                value = await store.get(setting_key)
                if value:
                    module.params[key] = str(value)
            except Exception:
                pass

    async def _spawn_worker(self, module: BaseModule) -> None:
        """
        Launch a child process for an isolated module and set up
        the transport pipes for every port that has a cross-process wire.
        """
        from core.worker import _run_worker

        control_parent, control_child = multiprocessing.Pipe(duplex=True)
        log_parent, log_child = multiprocessing.Pipe(duplex=False)

        # Build per-port data pipes for ports with active wires
        data_parent: dict[str, multiprocessing.connection.Connection] = {}
        data_child: dict[str, multiprocessing.connection.Connection] = {}
        all_ports = {**module.inputs, **module.outputs}

        for port_name, port in all_ports.items():
            if port.connected_wires:
                p_conn, c_conn = multiprocessing.Pipe(duplex=True)
                data_parent[port_name] = p_conn
                data_child[port_name] = c_conn

        module_class_path = (
            f"{type(module).__module__}.{type(module).__qualname__}"
        )

        # Hydrate any remaining vault sentinels from SettingsStore before
        # passing params to the worker (which has no vault access).
        await self._hydrate_settings_store_params(module)

        # Inject hypervisor-level context so the worker uses the correct
        # AI backend without running its own hardware probe.
        module.params["_max_ram_mb"] = 0
        module.params["_max_vram_mb"] = 0
        if self._vault is not None:
            comm_key = self._vault.get_module_comm_key()
            if comm_key:
                module.params["_hmac_key_hex"] = comm_key.hex()
            module.params["_user_data_dir"] = str(
                self._vault.get_user_data_dir()
            )
        try:
            hw = get_probe_service().hardware()
            detected = hw.ai_backend_hint.value
        except Exception:
            detected = "cpu"

        # Pass both values explicitly; backend resolution/enforcement lives in
        # basic_brain and must never silently fallback from user selection.
        stored = self._get_stored_backend()
        module.params["_ai_backend"] = str(stored or "").strip().lower()
        module.params["_ai_backend_detected"] = detected

        proc = multiprocessing.Process(
            target=_run_worker,
            args=(
                module_class_path,
                module.module_id,
                module.params,
                control_child,
                data_child,
                log_child,
            ),
            daemon=True,
        )
        proc.start()

        loop = asyncio.get_running_loop()

        ready = await loop.run_in_executor(None, control_parent.recv)
        if ready.get("status") != "ready":
            raise HypervisorError(
                f"Worker for {module.module_id} failed to start: {ready}"
            )

        handle = _WorkerHandle(
            process=proc,
            control_conn=control_parent,
            data_conns=data_parent,
            log_conn=log_parent,
        )
        self._workers[module.module_id] = handle
        handle.log_task = asyncio.create_task(
            self._pump_worker_logs(module.module_id, log_parent)
        )

        hmac_key = (
            self._vault.get_module_comm_key()
            if self._vault is not None else None
        ) or None

        for port_name, parent_conn in data_parent.items():
            port = all_ports[port_name]
            for wire in port.connected_wires:
                local_port = (
                    wire.source_port
                    if wire.target_port is port
                    else wire.target_port
                )
                wire.transport = ProcessTransport(
                    conn=parent_conn,
                    local_port=local_port,
                    direction="bidirectional",
                    loop=loop,
                    hmac_key=hmac_key,
                )

        log.info("Worker spawned for %s (pid %d)", module.module_id, proc.pid)

    async def _pump_worker_logs(
        self,
        module_id: str,
        conn: multiprocessing.connection.Connection,
    ) -> None:
        """Forward worker log records into system telemetry."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    has_data = await loop.run_in_executor(None, conn.poll, 0.25)
                except (EOFError, OSError):
                    break
                if not has_data:
                    if module_id not in self._workers:
                        break
                    continue
                try:
                    event = await loop.run_in_executor(None, conn.recv)
                except (EOFError, OSError):
                    break
                if not isinstance(event, dict):
                    continue
                severity = str(event.get("level", "INFO")).upper()
                logger_name = str(event.get("logger", "")).strip()
                message = str(event.get("message", "")).strip()
                if not message:
                    continue
                msg = f"[worker:{logger_name}] {message}" if logger_name else f"[worker] {message}"
                payload = Payload(
                    signal_type=SignalType.SYSTEM_STATE_PAYLOAD,
                    source_module=module_id,
                    data={
                        "severity": severity,
                        "log_message": msg,
                        "telemetry": {
                            "module_ram_allocation_mb": 0,
                            "module_vram_allocation_mb": 0,
                            "is_healthy": True,
                        },
                    },
                )
                await self.ingest_system_state(payload)
        except asyncio.CancelledError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    async def _send_worker_command(
        self, module_id: str, cmd: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a command to a worker and wait for the response."""
        handle = self._workers.get(module_id)
        if handle is None:
            raise HypervisorError(f"No worker for {module_id}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handle.control_conn.send, cmd)
        response = await loop.run_in_executor(None, handle.control_conn.recv)
        return response

    async def _initialize_registered_module(self, module: BaseModule) -> None:
        """Initialize a module that is already registered with the hypervisor."""
        module_id = module.module_id
        if module.PROCESS_ISOLATION:
            is_brain = module.MODULE_NAME == "basic_brain"
            model_name = Path(str(module.params.get("model_path", "")).strip()).name or "[none]"
            if module_id not in self._workers:
                await self._spawn_worker(module)
            module.status = ModuleStatus.LOADING
            t0 = time.perf_counter()
            if is_brain and model_name != "[none]":
                self._log_brain_model_load_event(module_id, model_name, "loading...", 0.0)
            resp = await self._send_worker_command(
                module_id, {"cmd": "initialize"}
            )
            if isinstance(resp.get("params"), dict):
                module.params.update(resp["params"])
                await self._persist_worker_sensitive_params(module)
            if resp.get("status") == "initialized":
                module.status = ModuleStatus.RUNNING
                if is_brain and model_name != "[none]":
                    self._log_brain_model_load_event(
                        module_id, model_name, "loaded", time.perf_counter() - t0
                    )
            else:
                module.status = ModuleStatus.ERROR
                if is_brain and model_name != "[none]":
                    self._log_brain_model_load_event(
                        module_id,
                        model_name,
                        f"load failed: {resp.get('detail', 'unknown error')}",
                        time.perf_counter() - t0,
                    )
                log.error(
                    "Isolated module %s failed to initialize: %s",
                    module_id, resp.get("detail"),
                )
        else:
            self._check_resource_limits(module)
            module.status = ModuleStatus.LOADING
            try:
                await module.initialize()
                if module.status != ModuleStatus.ERROR:
                    module.status = ModuleStatus.RUNNING
            except Exception as exc:
                module.status = ModuleStatus.ERROR
                await module.report_state(
                    severity="ERROR",
                    message=f"Failed to initialise: {exc}",
                )

    async def initialize_module(self, module_id: str) -> None:
        """Initialize one module at runtime without rebooting the rack."""
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")
        await self._initialize_registered_module(module)

    # ── Boot modules in order ───────────────────────────────────

    async def boot_modules(self) -> None:
        """
        Initialise every *enabled* registered module in topological order.

        In-process modules are initialised directly.  Isolated modules
        are spawned into worker processes first, then sent an
        ``initialize`` command over the control pipe.

        Modules with ``enabled=False`` are skipped — they remain in
        STOPPED state until individually powered on by the user.
        """
        boot_order = self.topological_sort()
        for module_id in boot_order:
            module = self.active_modules[module_id]
            if not module.enabled:
                continue
            await self._initialize_registered_module(module)

    # ── Module enable / disable ─────────────────────────────────

    async def stop_module(self, module_id: str) -> None:
        """
        Power-off a module without removing it from the rack.
        Wires stay patched but payloads sent to the module are discarded.
        """
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")
        if not module.enabled:
            return
        if module.status == ModuleStatus.IDLE:
            module.enabled = False
            module.status = ModuleStatus.STOPPED
            log.info("Module %s powered off", module_id)
            return

        module.enabled = False

        if module.PROCESS_ISOLATION:
            await self._kill_worker(module_id)
        else:
            try:
                await module.shutdown()
            except Exception:
                log.exception("Shutdown error while stopping %s", module_id)

        module.status = ModuleStatus.STOPPED
        log.info("Module %s powered off", module_id)

    async def start_module(self, module_id: str) -> None:
        """
        Power-on a module that was previously stopped.
        Re-initializes it and re-attaches transports for existing wires.
        """
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")
        if module.enabled:
            return

        module.enabled = True

        # Resolve port conflicts before init so modules using local IP get correct ports
        self.resolve_port_conflicts()

        try:
            await self._initialize_registered_module(module)
        except Exception as exc:
            module.status = ModuleStatus.ERROR
            await module.report_state(
                severity="ERROR",
                message=f"Start failed: {exc}",
            )
            raise

        try:
            resp = await self.call_module_method(module_id, "check_ready")
            if resp.get("status") == "ok" and bool(resp.get("result")):
                module.status = ModuleStatus.READY
            elif module.status == ModuleStatus.READY:
                module.status = ModuleStatus.RUNNING
        except Exception:
            log.exception("check_ready failed for %s during start_module", module_id)

        if self.system_powered and module.status in (ModuleStatus.RUNNING, ModuleStatus.READY):
            refresh = getattr(module, "refresh_wiring_state", None)
            if asyncio.iscoroutinefunction(refresh):
                try:
                    await refresh()
                except Exception:
                    log.exception("refresh_wiring_state failed for %s during start_module", module_id)

        log.info("Module %s powered on", module_id)

    async def unregister_module(self, module_id: str) -> None:
        """
        Remove a module from the rack and clean up all attached resources.
        """
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")

        if module.enabled:
            await self.stop_module(module_id)
        elif module.PROCESS_ISOLATION:
            await self._kill_worker(module_id)

        ports = list(module.inputs.values()) + list(module.outputs.values())
        wire_ids = {
            wire.id
            for port in ports
            for wire in list(port.connected_wires)
        }
        for wire_id in wire_ids:
            if any(w.id == wire_id for w in self.active_wires):
                self.disconnect_wire(wire_id)

        for port in ports:
            self.port_registry.unregister(port)
            port.connected_wires.clear()

        del self.active_modules[module_id]
        log.info("Module %s unregistered", module_id)

    # ── Module restart ──────────────────────────────────────────

    async def emergency_stop(self) -> None:
        """
        Emergency halt — terminate every worker process immediately and
        disable all in-process modules without waiting for graceful shutdown.
        Sets _running = False so the event loop exits on its next iteration.
        """
        self._running = False
        log.warning("EMERGENCY STOP initiated")

        loop = asyncio.get_running_loop()

        # Kill workers immediately — no graceful "shutdown" command
        for module_id, handle in list(self._workers.items()):
            for conn in handle.data_conns.values():
                try:
                    conn.close()
                except OSError:
                    pass
            try:
                handle.control_conn.close()
            except OSError:
                pass
            if handle.process.is_alive():
                handle.process.terminate()
                await loop.run_in_executor(None, lambda h=handle: h.process.join(2))
                if handle.process.is_alive():
                    handle.process.kill()
        self._workers.clear()

        # Disable all in-process modules
        for module in self.active_modules.values():
            module.enabled = False
            module.status = ModuleStatus.STOPPED

        log.warning("EMERGENCY STOP complete — all modules halted")

    async def emergency_resume(self) -> None:
        """
        Reverse an emergency_stop: re-enable the event loop and
        re-initialize every module that was halted.
        """
        self._running = True
        log.info("Emergency resume — restarting all modules")
        for module in list(self.active_modules.values()):
            module.enabled = True
            await self._initialize_registered_module(module)
        log.info("Emergency resume complete")

    async def restart_module(self, module_id: str) -> None:
        """
        Restart a module.  For isolated modules this kills the
        worker process and spawns a fresh one.  For in-process
        modules this calls shutdown → initialize.
        """
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")

        if module.PROCESS_ISOLATION:
            await self._kill_worker(module_id)
            await self._spawn_worker(module)
            resp = await self._send_worker_command(
                module_id, {"cmd": "initialize"}
            )
            if isinstance(resp.get("params"), dict):
                module.params.update(resp["params"])
                await self._persist_worker_sensitive_params(module)
            if resp.get("status") == "initialized":
                module.status = ModuleStatus.RUNNING
            else:
                module.status = ModuleStatus.ERROR
        else:
            try:
                await module.shutdown()
            except Exception:
                log.exception("Shutdown error during restart of %s", module_id)
            module.status = ModuleStatus.LOADING
            try:
                await module.initialize()
                module.status = ModuleStatus.RUNNING
            except Exception as exc:
                module.status = ModuleStatus.ERROR
                await module.report_state(
                    severity="ERROR",
                    message=f"Restart failed: {exc}",
                )

    async def call_module_method(
        self, module_id: str, method: str, **kwargs: Any,
    ) -> dict[str, Any]:
        """Call an async method on a module, routing through the worker if isolated."""
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")
        if module.PROCESS_ISOLATION and module_id in self._workers:
            resp = await self._send_worker_command(module_id, {
                "cmd": "call_method",
                "method": method,
                "kwargs": kwargs,
            })
            if isinstance(resp.get("params"), dict):
                module.params.update(resp["params"])
            return resp
        fn = getattr(module, method, None)
        if fn is None or not callable(fn):
            return {"status": "error", "detail": f"No callable method '{method}'"}
        try:
            result = await fn(**kwargs)
            return {"status": "ok", "result": result}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    async def sync_module_params(self, module_id: str) -> None:
        """Push the parent's module.params to the worker process."""
        module = self.active_modules.get(module_id)
        if module is None:
            return
        if module.PROCESS_ISOLATION and module_id in self._workers:
            await self._send_worker_command(
                module_id,
                {"cmd": "update_params", "params": dict(module.params)},
            )

    async def _kill_worker(self, module_id: str) -> None:
        handle = self._workers.get(module_id)
        if handle is None:
            return
        if handle.log_task is not None:
            handle.log_task.cancel()
            try:
                await handle.log_task
            except asyncio.CancelledError:
                pass
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, handle.control_conn.send, {"cmd": "shutdown"}
            )
            await loop.run_in_executor(None, handle.control_conn.recv)
        except Exception:
            pass

        # Close transports on wires that connected to this worker
        module = self.active_modules[module_id]
        for port in list(module.inputs.values()) + list(module.outputs.values()):
            for wire in port.connected_wires:
                if wire.transport and isinstance(wire.transport, ProcessTransport):
                    await wire.transport.close()
                    wire.transport = None

        for conn in handle.data_conns.values():
            try:
                conn.close()
            except OSError:
                pass
        try:
            handle.control_conn.close()
        except OSError:
            pass
        if handle.log_conn is not None:
            try:
                handle.log_conn.close()
            except OSError:
                pass

        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.kill()
        self._workers.pop(module_id, None)

    # ── Resource guard ──────────────────────────────────────────

    def _check_resource_limits(self, module: BaseModule) -> None:
        """No-op — resource limiting is disabled. Stub kept for call-site compat."""
        return

    # ── Worker health monitoring ────────────────────────────────

    async def _check_worker_health(self) -> None:
        """Check that all worker processes are still alive."""
        for module_id, handle in list(self._workers.items()):
            if not handle.process.is_alive():
                module = self.active_modules.get(module_id)
                if module and not module.enabled:
                    continue
                if module:
                    module.status = ModuleStatus.ERROR
                log.error(
                    "Worker for %s (pid %d) died unexpectedly.",
                    module_id, handle.process.pid,
                )

    # ── Telemetry ingestion ─────────────────────────────────────

    async def ingest_system_state(self, payload: Payload) -> None:
        """Accept a SYSTEM_STATE_PAYLOAD from any module."""
        self.system_log.append(payload)

    def ingest_inference(
        self,
        module_id: str,
        prompt_preview: str,
        response_preview: str,
        prompt_tokens: int,
        completion_tokens: int,
        ttft_s: float,
        total_s: float,
    ) -> None:
        """Queue inference stats for the brain card's inference log panel."""
        self._inference_log_queue.append((
            module_id,
            prompt_preview,
            response_preview,
            prompt_tokens,
            completion_tokens,
            ttft_s,
            total_s,
        ))

    def _log_brain_model_load_event(
        self,
        module_id: str,
        model_name: str,
        message: str,
        elapsed_s: float = 0.0,
    ) -> None:
        """Queue a model-load lifecycle marker into the brain inference log."""
        self.ingest_inference(
            module_id=module_id,
            prompt_preview=f"MODEL LOAD {model_name}",
            response_preview=message,
            prompt_tokens=0,
            completion_tokens=0,
            ttft_s=max(0.0, elapsed_s),
            total_s=max(0.0, elapsed_s),
        )

    # ── Main event loop ─────────────────────────────────────────

    _PAYLOADS_PER_TICK = 50  # Limit per iteration to avoid starving Qt event loop

    async def run_event_loop(self) -> None:
        """
        Spin forever, processing queued payloads on every in-process
        port.  Isolated modules process their own queues inside their
        worker process.  Periodically checks worker health.
        """
        health_counter = 0
        while self._running:
            processed = 0
            # Snapshot module refs so runtime add/remove operations are safe.
            for module in list(self.active_modules.values()):
                if module.PROCESS_ISOLATION or not module.enabled:
                    continue
                for port in module.inputs.values():
                    while not port._queue.empty() and processed < self._PAYLOADS_PER_TICK:
                        payload = await port._queue.get()
                        try:
                            await module.process(payload, port)
                        except Exception as exc:
                            log.exception(
                                "Hypervisor: process() error module=%s port=%s signal=%s",
                                module.module_id,
                                port.name,
                                getattr(payload.signal_type, "value", str(payload.signal_type)),
                            )
                            await module.report_state(
                                severity="ERROR",
                                message=f"Process error on {port.name}: {exc}",
                            )
                        processed += 1

            health_counter += 1
            if health_counter >= 1000:
                await self._check_worker_health()
                health_counter = 0

            await asyncio.sleep(0.001)

    async def shutdown(self) -> None:
        """Gracefully stop all modules in reverse boot order."""
        # Disable modules first to stop dispatch during teardown.
        # Keep _running=True until the end so run_event_loop stays alive.
        # Otherwise this task can be cancelled before shutdown completes.
        for module in self.active_modules.values():
            module.enabled = False

        # Shutdown workers first
        for module_id in list(self._workers.keys()):
            try:
                await self._kill_worker(module_id)
                module = self.active_modules.get(module_id)
                if module:
                    module.status = ModuleStatus.STOPPED
            except Exception:
                log.exception("Error shutting down worker %s", module_id)

        # Then in-process modules in reverse topological order
        boot_order = self.topological_sort()
        for module_id in reversed(boot_order):
            module = self.active_modules[module_id]
            if module.PROCESS_ISOLATION:
                continue
            try:
                await module.shutdown()
                module.status = ModuleStatus.STOPPED
            except Exception as exc:
                module.status = ModuleStatus.ERROR
                await module.report_state(
                    severity="ERROR",
                    message=f"Shutdown error: {exc}",
                )

        # Final safety net: force-kill any subprocesses still registered.
        # This prevents orphan Node/ngrok/etc processes from surviving app exit.
        for pid in list(self._registered_processes.keys()):
            self.kill_registered_process(pid)

        # Signal the event loop to exit only after all teardown is complete.
        self._running = False


    # ── Template persistence ────────────────────────────────────

    def snapshot_rack(
        self,
        module_order: list[str],
        template_name: str = "default",
    ) -> RackSnapshot:
        """
        Capture the current rack as a serialisable ``RackSnapshot``.

        Wire endpoints are stored as ``(module_id, port_name)`` pairs
        so they survive across sessions where port UUIDs are regenerated.
        """
        modules: list[ModuleSnapshot] = []
        for module in self.active_modules.values():
            safe_params = strip_sensitive_params(module, module.params)
            modules.append(
                ModuleSnapshot(
                    module_id=module.module_id,
                    module_type=module.MODULE_NAME,
                    params=safe_params,
                    enabled=module.enabled,
                )
            )

        wires: list[WireSnapshot] = []
        for wire in self.active_wires:
            wires.append(
                WireSnapshot(
                    src_module_id=wire.source_port.owner_module_id,
                    src_port_name=wire.source_port.name,
                    dst_module_id=wire.target_port.owner_module_id,
                    dst_port_name=wire.target_port.name,
                )
            )

        return RackSnapshot(
            template_name=template_name,
            modules=modules,
            wires=wires,
            module_order=[mid for mid in module_order if mid in self.active_modules],
        )

    def save_template(
        self,
        path: Path | str,
        module_order: list[str],
        template_name: str | None = None,
    ) -> None:
        """Serialise the current rack to a JSON template file."""
        path = Path(path)
        name = template_name or path.stem
        snapshot = self.snapshot_rack(module_order, template_name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        log.info("Template saved: %s", path)

    async def clear_rack(self) -> None:
        """
        Gracefully tear down every module in reverse topological order,
        disconnect all wires, and wipe the registries.

        After this call the Hypervisor is empty and ready for load_rack().
        """
        if self.active_modules:
            boot_order = self.topological_sort()
            for module_id in reversed(boot_order):
                module = self.active_modules.get(module_id)
                if module is not None and module.enabled:
                    try:
                        await self.stop_module(module_id)
                    except Exception:
                        log.exception("clear_rack: error stopping %s", module_id)

        # Disconnect every wire
        for wire in list(self.active_wires):
            try:
                wire.source_port.connected_wires.discard(wire) if hasattr(
                    wire.source_port.connected_wires, "discard"
                ) else (
                    wire.source_port.connected_wires.remove(wire)
                    if wire in wire.source_port.connected_wires
                    else None
                )
                wire.target_port.connected_wires.discard(wire) if hasattr(
                    wire.target_port.connected_wires, "discard"
                ) else (
                    wire.target_port.connected_wires.remove(wire)
                    if wire in wire.target_port.connected_wires
                    else None
                )
            except Exception:
                pass
        self.active_wires.clear()

        # Unregister all ports
        for module in self.active_modules.values():
            for port in list(module.inputs.values()) + list(module.outputs.values()):
                self.port_registry.unregister(port)

        self.active_modules.clear()
        self._workers.clear()
        self._module_power_memory.clear()
        self.system_powered = False
        log.info("Rack cleared")

    async def load_rack(
        self,
        snapshot: RackSnapshot,
        module_registry: dict[str, type[BaseModule]],
    ) -> list[str]:
        """
        Reconstruct a rack from a ``RackSnapshot`` in the **OFF state**.

        Steps
        -----
        1. ``clear_rack()`` — graceful teardown of the current rack.
        2. Instantiate each module from ``module_registry``, overlay saved
           params on top of ``get_default_params()``, leave enabled=False.
        3. Connect wires using stable ``(module_id, port_name)`` references.
        4. Return ``snapshot.module_order`` for ``RackWindow`` to restore layout.

        The rack is NOT booted — call ``boot_all()`` (via the power button)
        to initialise all modules.
        """
        await self.clear_rack()

        for mod_snap in snapshot.modules:
            cls = module_registry.get(mod_snap.module_type)
            if cls is None:
                log.warning(
                    "load_rack: unknown module type '%s' — skipped", mod_snap.module_type
                )
                continue

            module = cls(module_id=mod_snap.module_id)
            hydrated = hydrate_sensitive_params(
                mod_snap.module_type, mod_snap.params, self._vault,
            )
            module.params.update(hydrated)
            module.enabled = False
            module.status = ModuleStatus.STOPPED
            self.register_module(module)

        # Reconnect wires using stable name-based port lookup
        for wire_snap in snapshot.wires:
            src_name = wire_snap.src_port_name
            dst_name = wire_snap.dst_port_name
            # Backward-compatible port rename map for saved templates.
            if src_name == "LLM_IN":
                src_name = "API_IN"
            if dst_name == "LLM_IN":
                dst_name = "API_IN"

            src_port = self.port_registry.get_by_name(
                wire_snap.src_module_id, src_name
            )
            dst_port = self.port_registry.get_by_name(
                wire_snap.dst_module_id, dst_name
            )
            if src_port is None or dst_port is None:
                log.warning(
                    "load_rack: could not resolve wire %s:%s -> %s:%s -- skipped",
                    wire_snap.src_module_id,
                    wire_snap.src_port_name,
                    wire_snap.dst_module_id,
                    wire_snap.dst_port_name,
                )
                continue
            try:
                self.connect_ports(src_port.id, dst_port.id)
            except Exception as exc:
                log.warning("load_rack: wire connection failed: %s", exc)

        self.system_powered = False
        log.info(
            "Template '%s' loaded — %d modules, %d wires (rack is OFF)",
            snapshot.template_name,
            len(self.active_modules),
            len(self.active_wires),
        )
        return list(snapshot.module_order)

    # ── Rack-level power ────────────────────────────────────────

    async def boot_all(self) -> None:
        """
        Power ON — restore each module's remembered power state, then
        initialise enabled modules in topological order, run ``init()``
        + ``check_ready()`` so services like gpt_server start and
        broadcast their config to downstream modules (e.g. gpt_terminal
        → browser), and auto-promote any that pass to READY.

        On first boot (no power memory yet) every module is enabled.
        On subsequent boots after a hypervisor power-off, only modules
        that were previously on are re-enabled.
        """
        self._running = True

        if self._module_power_memory:
            for mid, module in self.active_modules.items():
                module.enabled = self._module_power_memory.get(mid, False)
        else:
            for module in self.active_modules.values():
                module.enabled = True

        self.resolve_port_conflicts()

        try:
            loop = asyncio.get_running_loop()
            probe = get_probe_service()
            hw = await loop.run_in_executor(None, probe.hardware)
            backend_warnings = probe.verify_backend(hw.ai_backend_hint)
            for w in backend_warnings:
                log.warning("Backend check: %s", w)
        except Exception as exc:
            log.debug("Backend verification skipped: %s", exc)

        await self.boot_modules()
        self.system_powered = True

        boot_order = self.topological_sort()
        for module_id in boot_order:
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            await self._run_module_init(module)

        for module in list(self.active_modules.values()):
            if not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue
            refresh = getattr(module, "refresh_wiring_state", None)
            if asyncio.iscoroutinefunction(refresh):
                try:
                    await refresh()
                except Exception:
                    log.exception(
                        "refresh_wiring_state failed for %s during boot_all",
                        module.module_id,
                    )

        log.info("Rack powered ON — all modules booted and initialised")

    # ── Readiness & validation ──────────────────────────────────

    def _port_in_use(self, port: int) -> bool:
        """Quick check whether something is already listening on the port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def check_port_conflicts(self) -> list[tuple[str, str, int]]:
        """
        Scan every enabled module's params for keys ending in ``_port``
        with integer values and return any duplicates as
        ``(module_id_a, module_id_b, port_number)`` triples.
        """
        port_map: dict[int, str] = {}
        conflicts: list[tuple[str, str, int]] = []
        for module in self.active_modules.values():
            if not module.enabled and module.status == ModuleStatus.STOPPED:
                continue
            for key, value in module.params.items():
                if key.endswith("_port") and isinstance(value, int) and value > 0:
                    if value in port_map:
                        other_id = port_map[value]
                        conflicts.append((other_id, module.module_id, value))
                        log.warning(
                            "Port conflict: %s and %s both use port %d",
                            other_id, module.module_id, value,
                        )
                    else:
                        port_map[value] = module.module_id
        return conflicts

    def resolve_port_conflicts(self) -> list[tuple[str, str, int, int]]:
        """
        Auto-assign alternative ports when conflicts are detected.
        Returns list of (module_id, param_key, old_port, new_port) changes.
        """
        # Build (module_id, param_key, port) for all _port params
        assignments: list[tuple[str, str, int]] = []
        for module in self.active_modules.values():
            if not module.enabled and module.status == ModuleStatus.STOPPED:
                continue
            for key, value in module.params.items():
                if key.endswith("_port") and isinstance(value, int) and value > 0:
                    assignments.append((module.module_id, key, value))

        used_ports: dict[int, str] = {}
        changes: list[tuple[str, str, int, int]] = []

        for module_id, param_key, port in assignments:
            module = self.active_modules.get(module_id)
            if module is None:
                continue
            if port in used_ports and used_ports[port] != module_id:
                # Conflict: find next port not used by another module
                candidate = port
                while candidate in used_ports:
                    candidate += 1
                module.params[param_key] = candidate
                changes.append((module_id, param_key, port, candidate))
                log.info(
                    "Port conflict resolved: %s %s %d -> %d",
                    module_id, param_key, port, candidate,
                )
                port = candidate
            used_ports[port] = module_id

        return changes

    async def init_all(self) -> None:
        """
        Run ``init()`` and ``check_ready()`` on every enabled module
        in topological order.  Called by the hypervisor panel INIT button.
        """
        self.resolve_port_conflicts()
        boot_order = self.topological_sort()

        for module_id in boot_order:
            module = self.active_modules.get(module_id)
            if module is None or not module.enabled:
                continue
            if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
                continue

            await self._run_module_init(module)

        ready = sum(
            1 for m in self.active_modules.values()
            if m.status == ModuleStatus.READY
        )
        enabled = sum(1 for m in self.active_modules.values() if m.enabled)
        log.info("init_all complete — %d/%d modules READY", ready, enabled)

    async def _run_module_init(self, module: BaseModule) -> None:
        """Run init/check_ready for one module, including worker-backed modules."""
        module_id = module.module_id
        try:
            if module.PROCESS_ISOLATION:
                if module_id in self._workers:
                    # Keep worker params in sync with latest UI edits before init().
                    await self._send_worker_command(
                        module_id, {"cmd": "update_params", "params": dict(module.params)}
                    )
                if module_id not in self._workers:
                    await self._spawn_worker(module)
                is_brain = module.MODULE_NAME == "basic_brain"
                model_name = Path(str(module.params.get("model_path", "")).strip()).name or "[none]"
                t0 = time.perf_counter()
                if is_brain and model_name != "[none]":
                    self._log_brain_model_load_event(
                        module_id, model_name, "init: loading...", 0.0
                    )
                resp = await self._send_worker_command(module_id, {"cmd": "init"})
                if isinstance(resp.get("params"), dict):
                    module.params.update(resp["params"])
                if resp.get("status") not in {"inited", "initialized"}:
                    module.status = ModuleStatus.ERROR
                    if is_brain and model_name != "[none]":
                        self._log_brain_model_load_event(
                            module_id,
                            model_name,
                            f"init load failed: {resp.get('detail', 'unknown error')}",
                            time.perf_counter() - t0,
                        )
                    log.error(
                        "init() failed for isolated module %s: %s",
                        module_id, resp.get("detail"),
                    )
                    return
                if is_brain and model_name != "[none]":
                    self._log_brain_model_load_event(
                        module_id, model_name, "init: loaded", time.perf_counter() - t0
                    )
            else:
                await module.init()
        except Exception:
            log.exception("init() failed for %s", module_id)
            module.status = ModuleStatus.ERROR
            return

        if module.status == ModuleStatus.ERROR:
            return

        try:
            resp = await self.call_module_method(module_id, "check_ready")
            if resp.get("status") == "ok" and bool(resp.get("result")):
                module.status = ModuleStatus.READY
            else:
                if module.status == ModuleStatus.READY:
                    module.status = ModuleStatus.RUNNING
                log.info("%s: not ready yet", module_id)
        except Exception:
            log.exception("check_ready() failed for %s", module_id)

    async def init_module(self, module_id: str) -> None:
        """Run init/check_ready for one module when it is powered on."""
        module = self.active_modules.get(module_id)
        if module is None:
            raise HypervisorError(f"Unknown module: {module_id}")
        if not module.enabled:
            return
        if module.status not in (ModuleStatus.RUNNING, ModuleStatus.READY):
            return
        await self._run_module_init(module)

    async def graceful_stop_all(self) -> None:
        """
        Power OFF — remember each module's individual power state, then
        shut down every enabled module in reverse topological order.
        The event loop keeps running so the application stays alive and
        the user can load a new template or press power ON again.

        This is what the hypervisor power button calls when transitioning
        from ON → OFF.  Unlike ``emergency_stop``, this calls each module's
        ``shutdown()`` hook and sends the graceful shutdown command to worker
        processes before terminating them.

        Note: ``_running`` is NOT set to False here.  Only ``shutdown()``
        (called by the Exit button / app close) terminates the event loop.
        """
        log.info("Rack powering OFF — graceful shutdown initiated")

        # Remember which modules were on so boot_all() can restore them
        self._module_power_memory = {
            mid: module.enabled
            for mid, module in self.active_modules.items()
        }

        boot_order = self.topological_sort()
        for module_id in reversed(boot_order):
            module = self.active_modules.get(module_id)
            if module is not None and module.enabled:
                try:
                    await self.stop_module(module_id)
                except Exception:
                    log.exception("graceful_stop_all: error stopping %s", module_id)

        self.system_powered = False
        log.info("Rack powered OFF — all modules stopped")


class _WorkerHandle:
    """Bookkeeping for a single worker process."""

    __slots__ = ("process", "control_conn", "data_conns", "log_conn", "log_task")

    def __init__(
        self,
        process: multiprocessing.Process,
        control_conn: multiprocessing.connection.Connection,
        data_conns: dict[str, multiprocessing.connection.Connection],
        log_conn: multiprocessing.connection.Connection | None = None,
    ) -> None:
        self.process = process
        self.control_conn = control_conn
        self.data_conns = data_conns
        self.log_conn = log_conn
        self.log_task: asyncio.Task | None = None
