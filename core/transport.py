"""
transport.py — Pluggable Wire Transport Layer
===============================================
Abstracts how payloads travel between ports.  A Wire holds a
Transport instance that handles delivery.  Two concrete flavours:

  LocalTransport   — direct in-process await (zero-copy, default).
  ProcessTransport — serialises Payload to JSON over a
                     multiprocessing.Connection pipe for cross-
                     process delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import multiprocessing.connection
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from core.port_system import Payload

if TYPE_CHECKING:
    from core.port_system import Port

log = logging.getLogger(__name__)


class Transport(ABC):
    """Interface every wire transport must satisfy."""

    @abstractmethod
    async def send(self, payload: Payload, sender: Port) -> None:
        """Deliver *payload* to whichever end of the wire isn't *sender*."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any OS resources (pipes, sockets, etc.)."""
        ...


# ── In-process (default) ────────────────────────────────────────


class LocalTransport(Transport):
    """
    Direct in-process delivery.  Equivalent to the original
    Port.emit() → target.receive() path — just factored out so
    the Wire doesn't need to know the difference.
    """

    def __init__(self, source_port: Port, target_port: Port) -> None:
        self.source_port = source_port
        self.target_port = target_port

    async def send(self, payload: Payload, sender: Port) -> None:
        target = (
            self.target_port if sender is self.source_port else self.source_port
        )
        await target.receive(payload)

    async def close(self) -> None:
        pass


# ── Cross-process ───────────────────────────────────────────────


class ProcessTransport(Transport):
    """
    Sends payloads across a multiprocessing.Connection pipe.

    Each side of the wire lives in a different OS process.  The
    parent Hypervisor holds one Connection end; the worker holds
    the other.  A background asyncio task drains the pipe and
    feeds inbound payloads into the target port's queue.

    Parameters
    ----------
    conn          The *local* end of the multiprocessing pipe.
    local_port    The port in *this* process (may be source or target).
    direction     "outbound" if this side sends to the pipe,
                  "inbound" if this side reads from the pipe,
                  "bidirectional" for CHANNEL mode.
    loop          The running asyncio event loop (for the reader task).
    """

    def __init__(
        self,
        conn: multiprocessing.connection.Connection,
        local_port: Port,
        direction: str = "bidirectional",
        loop: Optional[asyncio.AbstractEventLoop] = None,
        hmac_key: bytes | None = None,
    ) -> None:
        self.conn = conn
        self.local_port = local_port
        self.direction = direction
        self._loop = loop or asyncio.get_event_loop()
        self._reader_task: Optional[asyncio.Task] = None
        self._closed = False
        self._hmac_key = hmac_key

        if direction in ("inbound", "bidirectional"):
            self._reader_task = self._loop.create_task(self._reader())

    def _sign(self, data: bytes) -> str:
        if self._hmac_key is None:
            return ""
        return hmac.new(self._hmac_key, data, hashlib.sha256).hexdigest()

    def _verify(self, data: bytes, signature: str) -> bool:
        if self._hmac_key is None:
            return True
        expected = hmac.new(self._hmac_key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    async def send(self, payload: Payload, sender: Port) -> None:
        if self._closed:
            log.warning("ProcessTransport.send: pipe closed, dropping %s", payload.signal_type.value)
            return
        data = payload.model_dump_json()
        log.info(
            "ProcessTransport.send: %s from=%s local_port=%s",
            payload.signal_type.value, sender.name, self.local_port.name,
        )
        data_bytes = data.encode("utf-8") if isinstance(data, str) else data
        sig = self._sign(data_bytes)
        await self._loop.run_in_executor(None, self.conn.send, (data, sig))

    async def _reader(self) -> None:
        """Background task: drain the pipe and push into local_port."""
        while not self._closed:
            try:
                has_data = await self._loop.run_in_executor(
                    None, self.conn.poll, 0.05
                )
                if has_data:
                    msg = await self._loop.run_in_executor(
                        None, self.conn.recv
                    )
                    if isinstance(msg, tuple) and len(msg) == 2:
                        raw, sig = msg
                    else:
                        raw, sig = msg, ""
                    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
                    if not self._verify(raw_bytes, sig):
                        log.error("ProcessTransport: HMAC verification failed — dropping payload")
                        continue
                    payload = Payload.model_validate_json(raw)
                    log.info(
                        "ProcessTransport._reader: received %s -> local_port=%s",
                        payload.signal_type.value, self.local_port.name,
                    )
                    await self.local_port.receive(payload)
            except (EOFError, OSError):
                log.warning("ProcessTransport pipe closed.")
                break
            except Exception:
                log.exception("ProcessTransport reader error")
                await asyncio.sleep(0.1)

    async def close(self) -> None:
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        try:
            self.conn.close()
        except OSError:
            pass


# ── Helpers ─────────────────────────────────────────────────────


def create_process_transport_pair(
    source_port: Port,
    target_port: Port,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    hmac_key: bytes | None = None,
) -> tuple[ProcessTransport, multiprocessing.connection.Connection]:
    """
    Create a pipe and return *(parent_transport, child_conn)*.

    The parent side gets a fully initialised ProcessTransport with a
    reader task.  The child_conn should be passed to the worker
    process, which will build its own ProcessTransport from it.
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    parent_transport = ProcessTransport(
        conn=parent_conn,
        local_port=source_port,
        direction="bidirectional",
        loop=loop,
        hmac_key=hmac_key,
    )
    return parent_transport, child_conn
