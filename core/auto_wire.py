"""
auto_wire.py — Best-guess wiring solver for the Miniloader rack.

Given the set of active modules and their ports, proposes connections
that are likely what the user intended.  The caller is responsible for
calling ``hypervisor.connect_ports()`` on each proposal.

Heuristics (in priority order):
  1. Name affinity      — matching base names (BRAIN_OUT ↔ BRAIN_IN).
  2. Routing priority   — domain-specific preferred targets (e.g. tool
                          ports prefer agent_engine over terminals).
  3. Signal overlap     — proportion of shared accepted_signals.
  4. Overlap count      — raw shared signal count as final tiebreaker.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.hypervisor import Hypervisor

from core.port_system import ConnectionMode, Port, PortDirection

_SUFFIX_RE = re.compile(r"(_IN_OUT|_IN|_OUT|_IO)$")

# ── Routing priority rules ───────────────────────────────────────────────────
#
# Maps a source port base-name to an ordered list of preferred target
# module types.  When multiple valid targets exist the one whose MODULE_NAME
# appears earliest in the list (lower index = higher priority) wins.
# Module types not listed are treated as lowest priority (index = len(list)).
#
# "tools" ports should reach chat terminals first.

_TOOLS_PREFERRED_TARGETS: list[str] = [
    "gpt_terminal",
    "discord_terminal",
    "agent_engine",
]

_ROUTING_PRIORITY: dict[str, list[str]] = {
    "TOOLS": _TOOLS_PREFERRED_TARGETS,
}


def _routing_priority_score(
    src: Port,
    tgt: Port,
    module_names: dict[str, str],
) -> int:
    """
    Return a routing-priority score (higher = better).

    For source ports covered by _ROUTING_PRIORITY the score is the
    inverse position of the target's module type in the preference list
    (so index 0 → highest score).  Zero is returned when no rule applies.
    """
    base = _base_name(src.name)
    preferred = _ROUTING_PRIORITY.get(base)
    if preferred is None:
        return 0
    tgt_module = module_names.get(tgt.owner_module_id, "")
    try:
        pos = preferred.index(tgt_module)
    except ValueError:
        pos = len(preferred)
    # Invert: position 0 → highest score (len(list) points), last → 0
    return len(preferred) - pos


def _base_name(port_name: str) -> str:
    return _SUFFIX_RE.sub("", port_name)


def _can_be_source(port: Port) -> bool:
    return (
        port.direction == PortDirection.OUT
        or port.connection_mode == ConnectionMode.CHANNEL
    )


def _can_be_target(port: Port) -> bool:
    return (
        port.direction == PortDirection.IN
        or port.connection_mode == ConnectionMode.CHANNEL
    )


def _target_has_capacity(port: Port) -> bool:
    return len(port.connected_wires) < port.max_connections


def _already_wired(src: Port, tgt: Port, existing_pairs: set[tuple[str, str]]) -> bool:
    return (src.id, tgt.id) in existing_pairs


def _same_module(a: Port, b: Port) -> bool:
    return a.owner_module_id == b.owner_module_id


def _is_out_port_name(name: str) -> bool:
    return name.endswith("_OUT")


def _is_in_port_name(name: str) -> bool:
    return name.endswith("_IN")


def _allows_multi_target_matches(port: Port) -> bool:
    return port.name in {"TOOLS_IN", "MCP_IN", "MCP_UPSTREAM", "MCP_DOWNSTREAM"}


def _passes_special_name_rules(src: Port, tgt: Port) -> bool:
    """
    Enforce explicit naming constraints for ambiguous CHANNEL ports.

    WEB links are strict by convention and should only auto-wire as:
      WEB_OUT -> WEB_IN
    This prevents nonsense links like WEB_OUT -> VOICE_LINK even when
    accepted signal sets overlap.

    TOOLS consumer ports (TOOLS_IN, MCP_IN) must never feed into other
    TOOLS consumer ports — only TOOLS_OUT providers may do that.
    """
    if src.name == "WEB_OUT":
        return tgt.name == "WEB_IN"
    if tgt.name == "WEB_OUT":
        return False
    if tgt.name == "WEB_IN":
        return src.name == "WEB_OUT"

    _TOOLS_CONSUMERS = {"TOOLS_IN", "MCP_IN", "MCP_UPSTREAM", "MCP_DOWNSTREAM"}
    if src.name in _TOOLS_CONSUMERS and tgt.name in _TOOLS_CONSUMERS:
        return False

    _DOC_CONSUMERS = {"CONTEXT_IN", "FILES_IN"}
    if src.name in _DOC_CONSUMERS and tgt.name in _DOC_CONSUMERS:
        return False

    return True


def _same_module_type(
    a: Port,
    b: Port,
    module_names: dict[str, str],
) -> bool:
    a_name = module_names.get(a.owner_module_id)
    b_name = module_names.get(b.owner_module_id)
    if a_name is None or b_name is None:
        return False
    return a_name == b_name


def _signal_overlap(a: Port, b: Port) -> int:
    return len(a.accepted_signals & b.accepted_signals)


def _overlap_ratio(a: Port, b: Port) -> float:
    shared = len(a.accepted_signals & b.accepted_signals)
    total = max(len(a.accepted_signals), len(b.accepted_signals), 1)
    return shared / total


def _score(
    src: Port,
    tgt: Port,
    module_names: dict[str, str],
) -> tuple[int, int, float, int]:
    """Return a sortable score tuple (higher is better).

    (name_match, routing_priority, overlap_ratio, overlap_count)
    """
    name_match = 1 if _base_name(src.name) == _base_name(tgt.name) else 0
    routing = _routing_priority_score(src, tgt, module_names)
    ratio = _overlap_ratio(src, tgt)
    overlap = _signal_overlap(src, tgt)
    return (name_match, routing, ratio, overlap)


def _collect_existing_pairs(hypervisor: Hypervisor) -> set[tuple[str, str]]:
    return {
        (w.source_port.id, w.target_port.id)
        for w in hypervisor.active_wires
    }


def _find_proposals(
    sources: list[Port],
    targets: list[Port],
    existing_pairs: set[tuple[str, str]],
    module_names: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Greedily match sources to targets by score.

    Each target slot is consumed at most once per call so that a single
    auto-wire pass doesn't over-saturate a port.
    """
    candidates: list[tuple[tuple[int, int, float, int], Port, Port]] = []

    for src in sources:
        for tgt in targets:
            if _same_module(src, tgt):
                continue
            if _same_module_type(src, tgt, module_names):
                continue
            if not _passes_special_name_rules(src, tgt):
                continue
            if _already_wired(src, tgt, existing_pairs):
                continue
            if not _target_has_capacity(tgt):
                continue
            overlap = _signal_overlap(src, tgt)
            if overlap == 0:
                continue
            candidates.append((_score(src, tgt, module_names), src, tgt))

    candidates.sort(key=lambda c: c[0], reverse=True)

    used_targets: set[str] = set()
    target_usage: dict[str, int] = {}
    proposals: list[tuple[str, str]] = []

    for _sc, src, tgt in candidates:
        allow_multi_target = _allows_multi_target_matches(tgt)
        if not allow_multi_target and tgt.id in used_targets:
            continue
        usage_count = target_usage.get(tgt.id, 0)
        if len(tgt.connected_wires) + usage_count >= tgt.max_connections:
            continue
        if _already_wired(src, tgt, existing_pairs):
            continue
        proposals.append((src.id, tgt.id))
        target_usage[tgt.id] = usage_count + 1
        if not allow_multi_target:
            used_targets.add(tgt.id)
        existing_pairs.add((src.id, tgt.id))

    return proposals


def auto_wire_all(hypervisor: Hypervisor) -> list[tuple[str, str]]:
    """Propose wires for every unconnected-or-under-max port on the rack."""
    all_ports = hypervisor.port_registry.all_ports()
    sources = [p for p in all_ports if _can_be_source(p)]
    targets = [p for p in all_ports if _can_be_target(p) and _target_has_capacity(p)]
    existing = _collect_existing_pairs(hypervisor)
    module_names = {
        module_id: module.MODULE_NAME
        for module_id, module in hypervisor.active_modules.items()
    }
    return _find_proposals(sources, targets, existing, module_names)


def auto_wire_module(hypervisor: Hypervisor, module_id: str) -> list[tuple[str, str]]:
    """Propose wires that involve *at least one port* of ``module_id``."""
    module = hypervisor.active_modules.get(module_id)
    if module is None:
        return []

    mod_ports = list(module.inputs.values()) + list(module.outputs.values())
    mod_port_ids = {p.id for p in mod_ports}

    all_ports = hypervisor.port_registry.all_ports()
    existing = _collect_existing_pairs(hypervisor)
    module_names = {
        mod_id: mod.MODULE_NAME
        for mod_id, mod in hypervisor.active_modules.items()
    }

    proposals: list[tuple[str, str]] = []

    mod_sources = [p for p in mod_ports if _can_be_source(p)]
    rack_targets = [
        p for p in all_ports
        if _can_be_target(p) and _target_has_capacity(p) and p.id not in mod_port_ids
    ]
    proposals.extend(_find_proposals(mod_sources, rack_targets, existing, module_names))

    rack_sources = [p for p in all_ports if _can_be_source(p) and p.id not in mod_port_ids]
    mod_targets = [p for p in mod_ports if _can_be_target(p) and _target_has_capacity(p)]
    proposals.extend(_find_proposals(rack_sources, mod_targets, existing, module_names))

    return proposals
