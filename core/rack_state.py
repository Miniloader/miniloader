"""
rack_state.py — Template Serialisation Models
==============================================
Pydantic models that describe a complete rack snapshot.  These are
the data shapes written to and read from templates/*.json files.

A RackSnapshot captures:
  - Which modules are in the rack, their stable IDs, types, and params
  - Which ports are wired together (addressed by module_id + port_name,
    not by ephemeral port UUIDs which change every session)
  - The visual layout order of cards on the rack canvas
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.base_module import BaseModule

log = logging.getLogger(__name__)

VAULT_SENTINEL = "__vault__"
NON_PERSISTED_PARAM_KEYS = {
    "_max_ram_mb",
    "_max_vram_mb",
    "_hmac_key_hex",
    "_user_data_dir",
    "_ai_backend",
    "_ai_backend_detected",
}


def _is_non_persisted_param(key: str) -> bool:
    """Return True for runtime-only keys that must never be serialized."""
    return key in NON_PERSISTED_PARAM_KEYS


class ModuleSnapshot(BaseModel):
    """
    Serialised state for a single module.

    ``module_type`` maps to ``BaseModule.MODULE_NAME`` and is used to
    look up the concrete class in ``MODULE_REGISTRY`` at load time.

    ``params`` stores the complete params dict.  On load, these are
    *overlaid* on top of ``get_default_params()`` so that any new params
    added after the template was saved receive their default values.

    ``enabled`` is always saved as ``False`` — templates load in the
    OFF (STOPPED) state; the hypervisor power button boots the rack.
    """

    module_id: str
    module_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = False


class WireSnapshot(BaseModel):
    """
    Serialised state for a single wire connection.

    Port UUIDs are ephemeral (regenerated each session), so endpoints
    are stored as ``(owner_module_id, port_name)`` pairs which remain
    stable across saves and loads.
    """

    src_module_id: str
    src_port_name: str
    dst_module_id: str
    dst_port_name: str


class RackSnapshot(BaseModel):
    """
    Complete serialised rack state.  Written to ``templates/<name>.json``.

    ``version`` guards forward/backward compatibility; increment when
    the schema changes in a breaking way.

    ``module_order`` is the list of ``module_id`` values in the visual
    slot sequence used by ``RackWindow._module_order``.
    """

    version: int = 1
    template_name: str
    modules: list[ModuleSnapshot] = Field(default_factory=list)
    wires: list[WireSnapshot] = Field(default_factory=list)
    module_order: list[str] = Field(default_factory=list)


# ── Sentinel helpers ─────────────────────────────────────────────


def strip_sensitive_params(module: BaseModule, params: dict[str, Any]) -> dict[str, Any]:
    """Replace sensitive param values with the vault sentinel before serialization."""
    sensitive = getattr(module.__class__, "SENSITIVE_PARAMS", set())
    cleaned = {
        key: value for key, value in dict(params).items()
        if not _is_non_persisted_param(key)
    }
    if not sensitive:
        return cleaned
    for key in sensitive:
        if key in cleaned and cleaned[key]:
            cleaned[key] = VAULT_SENTINEL
    return cleaned


def hydrate_sensitive_params(
    module_type: str,
    params: dict[str, Any],
    vault: Any | None,
) -> dict[str, Any]:
    """Replace vault sentinels with real secrets from the vault on load."""
    hydrated = {
        key: value for key, value in dict(params).items()
        if not _is_non_persisted_param(key)
    }
    for key, value in list(hydrated.items()):
        if value != VAULT_SENTINEL:
            continue
        if vault is None:
            log.warning(
                "load_rack: sentinel '%s.%s' cannot be hydrated (no vault)",
                module_type, key,
            )
            continue
        secret = vault.get_secret(f"{module_type}.{key}")
        if secret is not None:
            hydrated[key] = secret
        else:
            log.warning(
                "load_rack: no vault secret for '%s.%s' — leaving sentinel",
                module_type, key,
            )
    return hydrated
