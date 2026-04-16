"""
presets.py — Curated module stacks with pre-defined wiring.

Each Preset specifies:
  • modules   — ordered list of MODULE_NAME strings to register.
  • wires     — list of (src_module_idx, src_port, tgt_module_idx, tgt_port)
                where the indices refer to positions in ``modules``.
  • name      — human-readable label shown in the right-click menu.
  • description — one-line tooltip.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WireSpec:
    src_idx: int    # index into Preset.modules
    src_port: str
    tgt_idx: int    # index into Preset.modules
    tgt_port: str


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    modules: tuple[str, ...]
    wires: tuple[WireSpec, ...]


# ── Preset definitions ───────────────────────────────────────────────────────
#
# Module index map (per preset, in order):
#
#  LOCAL_AI_CHAT
#   0 → basic_brain   1 → gpt_server   2 → gpt_terminal
#
#  LOCAL_AI_AGENT
#   0 → basic_brain   1 → gpt_server   2 → agent_engine   3 → gpt_terminal
#
#  RTC_VOICE_AGENT
#   0 → basic_brain   1 → gpt_server   2 → agent_engine
#   3 → livekit_voice   4 → gpt_terminal

LOCAL_AI_CHAT = Preset(
    name="Local AI Chat",
    description="Local brain + AI server + chat terminal, pre-wired",
    modules=("basic_brain", "gpt_server", "gpt_terminal"),
    wires=(
        # brain → server
        WireSpec(0, "BRAIN_OUT",    1, "BRAIN_IN"),
        # server → terminal
        WireSpec(1, "LOCAL_IP_OUT", 2, "LOCAL_IP_IN"),
    ),
)

LOCAL_AI_AGENT = Preset(
    name="Local AI Agent",
    description="Local brain + AI server + agent engine + chat terminal, pre-wired",
    modules=("basic_brain", "gpt_server", "agent_engine", "gpt_terminal"),
    wires=(
        # brain → server
        WireSpec(0, "BRAIN_OUT",    1, "BRAIN_IN"),
        # server → terminal
        WireSpec(1, "LOCAL_IP_OUT", 3, "LOCAL_IP_IN"),
        # server → agent
        WireSpec(1, "LOCAL_IP_OUT", 2, "API_IN"),
        # agent → terminal
        WireSpec(2, "AGENT_OUT",    3, "AGENT_IN"),
    ),
)

RTC_VOICE_AGENT = Preset(
    name="RTC Voice Agent",
    description="Full agent stack with LiveKit voice, pre-wired",
    modules=(
        "basic_brain",
        "gpt_server",
        "agent_engine",
        "livekit_voice",
        "gpt_terminal",
    ),
    wires=(
        # brain → server
        WireSpec(0, "BRAIN_OUT",    1, "BRAIN_IN"),
        # server → terminal
        WireSpec(1, "LOCAL_IP_OUT", 4, "LOCAL_IP_IN"),
        # server → agent
        WireSpec(1, "LOCAL_IP_OUT", 2, "API_IN"),
        # agent → terminal
        WireSpec(2, "AGENT_OUT",    4, "AGENT_IN"),
        # agent → voice
        WireSpec(2, "AGENT_OUT",    3, "AGENT_IO"),
        # voice config → terminal voice link
        WireSpec(3, "VOICE_CONFIG", 4, "VOICE_LINK"),
    ),
)

REMOTE_SECRETARY = Preset(
    name="Remote Secretary",
    description=(
        "Full RTC voice agent stack with file vault, knowledge engine, "
        "Google Suite tools, and web gateway — pre-wired"
    ),
    modules=(
        # 0          1            2               3
        "basic_brain", "gpt_server", "agent_engine", "livekit_voice",
        # 4               5              6              7               8
        "gpt_terminal", "file_access", "ngrok_tunnel", "google_suite", "rag_engine",
    ),
    wires=(
        # ── Core RTC voice agent stack (mirrors RTC_VOICE_AGENT) ──
        WireSpec(0, "BRAIN_OUT",    1, "BRAIN_IN"),       # brain → server
        WireSpec(1, "LOCAL_IP_OUT", 4, "LOCAL_IP_IN"),    # server → terminal
        WireSpec(1, "LOCAL_IP_OUT", 2, "API_IN"),         # server → agent
        WireSpec(2, "AGENT_OUT",    4, "AGENT_IN"),       # agent → terminal
        WireSpec(2, "AGENT_OUT",    3, "AGENT_IO"),       # agent → voice
        WireSpec(3, "VOICE_CONFIG", 4, "VOICE_LINK"),     # voice config → terminal

        # ── File vault → knowledge engine → agent ──
        WireSpec(5, "FILES_OUT",    8, "FILES_IN"),       # file vault → RAG
        WireSpec(8, "CONTEXT_OUT",  2, "CONTEXT_IN"),     # RAG knowledge → agent

        # ── Tools ──
        # File vault tools → terminal
        WireSpec(5, "TOOLS_OUT",    4, "MCP_IN"),
        # Google Suite tools → terminal
        WireSpec(7, "TOOLS_OUT",    4, "MCP_IN"),

        # ── Web gateway ──
        WireSpec(4, "WEB_OUT",      6, "WEB_IN"),         # terminal web → ngrok
    ),
)

ALL_PRESETS: tuple[Preset, ...] = (
    LOCAL_AI_CHAT,
    LOCAL_AI_AGENT,
    RTC_VOICE_AGENT,
    REMOTE_SECRETARY,
)
