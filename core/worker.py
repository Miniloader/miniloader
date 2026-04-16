"""
worker.py — Isolated Module Process Host
==========================================
Entry-point for child processes spawned by the Hypervisor.  Each
worker hosts exactly one module and bridges its ports to the parent
process over multiprocessing pipes.

The parent sends commands as JSON dicts over the *control* pipe:
    {"cmd": "initialize"}
    {"cmd": "shutdown"}

Payload traffic flows over per-port *data* pipes managed by
ProcessTransport instances created from connection handles the
parent passes at spawn time.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import multiprocessing.connection
import sys
import traceback
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _run_worker(
    module_class_path: str,
    module_id: str,
    module_params: dict[str, Any],
    control_conn: multiprocessing.connection.Connection,
    data_conns: dict[str, multiprocessing.connection.Connection],
    log_conn: multiprocessing.connection.Connection | None = None,
) -> None:
    """
    Target function for ``multiprocessing.Process``.

    Parameters
    ----------
    module_class_path
        Dotted import path to the module class,
        e.g. ``"modules.basic_brain.logic.BasicBrainModule"``.
    module_id
        The module_id the parent assigned.
    module_params
        ``module.params`` dict (serialisable).
    control_conn
        Pipe for lifecycle commands from the parent.
    data_conns
        ``{port_name: Connection}`` — one pipe per port that crosses
        the process boundary.
    log_conn
        Optional one-way pipe used to stream worker log events back to
        the Hypervisor UI.
    """
    # Frozen GUI apps can have null stdio handles / stderr FILE*.
    # Native code writing to stderr can segfault in that state.
    # Ensure fds 0/1/2 are valid before any native code executes.
    if getattr(sys, "frozen", False):
        import os as _os
        _devnull = _os.open(_os.devnull, _os.O_RDWR)
        for _fd in (0, 1, 2):
            try:
                _os.fstat(_fd)
            except OSError:
                _os.dup2(_devnull, _fd)
        _os.close(_devnull)
        if sys.stdout is None:
            sys.stdout = open(_os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(_os.devnull, "w")

    # Child process inherits no logging config from the parent under
    # the 'spawn' start method — set up a basic handler so worker and
    # transport logs are visible in the terminal.
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s (worker %(process)d): %(message)s",
    )
    if getattr(sys, "frozen", False):
        for _h in logging.getLogger().handlers:
            if isinstance(_h, logging.StreamHandler) and hasattr(_h, "stream"):
                try:
                    _h.stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
                except Exception:
                    pass
    if log_conn is not None:
        class _PipeLogHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    log_conn.send(
                        {
                            "module_id": module_id,
                            "logger": record.name,
                            "level": record.levelname,
                            "message": msg,
                        }
                    )
                except Exception:
                    return

        _pipe_handler = _PipeLogHandler()
        _pipe_handler.setLevel(logging.INFO)
        _pipe_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        logging.getLogger().addHandler(_pipe_handler)

    # In frozen builds, write a debug log so errors are diagnosable.
    if getattr(sys, "frozen", False):
        _log_dir = Path.home() / ".miniloader" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(_log_dir / "worker_debug.log"), mode="a")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (worker %(process)d): %(message)s"
        ))
        logging.getLogger().addHandler(_fh)

    asyncio.run(_async_worker(
        module_class_path, module_id, module_params,
        control_conn, data_conns, log_conn,
    ))


async def _async_worker(
    module_class_path: str,
    module_id: str,
    module_params: dict[str, Any],
    control_conn: multiprocessing.connection.Connection,
    data_conns: dict[str, multiprocessing.connection.Connection],
    log_conn: multiprocessing.connection.Connection | None = None,
) -> None:
    from core.port_system import Payload
    from core.transport import ProcessTransport

    # ── Import and instantiate the module ────────────────────────
    parts = module_class_path.rsplit(".", 1)
    mod = importlib.import_module(parts[0])
    cls = getattr(mod, parts[1])

    module = cls(module_id=module_id)
    module.params.update(module_params)
    module.define_ports()

    loop = asyncio.get_running_loop()

    # Attach ProcessTransport to each wired port.
    # _reader() handles incoming pipe -> queue data.
    # For outgoing emit() -> pipe traffic, a Wire must exist in connected_wires.
    from core.port_system import Wire

    hmac_key_hex = module_params.get("_hmac_key_hex", "")
    hmac_key = bytes.fromhex(hmac_key_hex) if hmac_key_hex else None

    transports: list[ProcessTransport] = []
    all_ports = {**module.inputs, **module.outputs}

    for port_name, conn in data_conns.items():
        port = all_ports.get(port_name)
        if port is None:
            log.warning("Worker %s: no port named %s", module_id, port_name)
            continue
        transport = ProcessTransport(
            conn=conn,
            local_port=port,
            direction="bidirectional",
            loop=loop,
            hmac_key=hmac_key,
        )
        transports.append(transport)

        # Synthetic wire so port.emit() sends through the pipe
        wire = Wire(port, port)
        wire.transport = transport
        port.connected_wires.append(wire)

    # ── Control loop: wait for commands from the parent ──────────
    def _safe_send(msg: dict) -> bool:
        """Send *msg* on the control connection; return False if pipe is gone."""
        try:
            control_conn.send(msg)
            return True
        except OSError:
            return False

    _safe_send({"status": "ready"})

    try:
        while True:
            try:
                has_cmd = await loop.run_in_executor(
                    None, control_conn.poll, 0.1
                )
            except (EOFError, OSError):
                log.warning("Worker %s: control pipe closed, shutting down", module_id)
                break
            if not has_cmd:
                # Drain any payloads that arrived on any port (outputs can
                # receive payloads on bidirectional CHANNEL wires)
                for port in {**module.inputs, **module.outputs}.values():
                    while not port._queue.empty():
                        payload = await port._queue.get()
                        if payload.signal_type.name in {"CHAT_REQUEST", "BRAIN_STREAM_PAYLOAD"}:
                            req_id = payload.data.get("request_id", "")
                            log.info(
                                "worker %s: dequeued %s on %s request_id=%s",
                                module_id, payload.signal_type.value, port.name, req_id,
                            )
                        try:
                            await module.process(payload, port)
                        except Exception:
                            log.exception(
                                "Worker %s: process() error on %s",
                                module_id, port.name,
                            )
                await asyncio.sleep(0.001)
                continue

            try:
                msg: dict = await loop.run_in_executor(None, control_conn.recv)
            except (EOFError, OSError):
                log.warning("Worker %s: control pipe closed during recv, shutting down", module_id)
                break
            cmd = msg.get("cmd")

            if cmd == "initialize":
                try:
                    await module.initialize()
                    _safe_send({"status": "initialized", "params": dict(module.params)})
                except Exception as exc:
                    _safe_send({
                        "status": "error",
                        "detail": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    })

            elif cmd == "init":
                try:
                    await module.init()
                    _safe_send({"status": "inited", "params": dict(module.params)})
                except Exception as exc:
                    _safe_send({
                        "status": "error",
                        "detail": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    })

            elif cmd == "shutdown":
                try:
                    await module.shutdown()
                except Exception:
                    log.exception("Worker %s: shutdown error", module_id)
                _safe_send({"status": "stopped"})
                break

            elif cmd == "update_params":
                module.params.update(msg.get("params", {}))
                _safe_send({"status": "params_updated"})

            elif cmd == "call_method":
                method_name = msg.get("method", "")
                method_kwargs = msg.get("kwargs", {})
                fn = getattr(module, method_name, None)
                if fn is None or not callable(fn):
                    _safe_send({
                        "status": "error",
                        "detail": f"No callable method '{method_name}'",
                    })
                else:
                    try:
                        result = await fn(**method_kwargs)
                        _safe_send({
                            "status": "ok",
                            "result": result,
                            "params": dict(module.params),
                        })
                    except Exception as exc:
                        _safe_send({
                            "status": "error",
                            "detail": str(exc),
                            "exception_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                        })

    finally:
        for t in transports:
            await t.close()
        if log_conn is not None:
            try:
                log_conn.close()
            except OSError:
                pass
        try:
            control_conn.close()
        except OSError:
            pass
