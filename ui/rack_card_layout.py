"""Registry mapping module names to card builder classes and height computation."""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)


def _optional_builder(import_path: str, class_name: str) -> type | None:
    """Return a card builder class when its module package exists."""
    try:
        module = importlib.import_module(import_path)
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        log.warning("rack_card_layout: failed to import %s (%s)", import_path, exc)
        return None
    builder_cls = getattr(module, class_name, None)
    if not isinstance(builder_cls, type):
        log.warning(
            "rack_card_layout: missing builder class %s in %s",
            class_name,
            import_path,
        )
        return None
    return builder_cls

CARD_BUILDERS: dict[str, type] = {
    name: builder
    for name, builder in [
        (
            "agent_engine",
            _optional_builder("modules.agent_engine.widget", "AgentEngineCardBuilder"),
        ),
        (
            "basic_brain",
            _optional_builder("modules.basic_brain.widget", "BasicBrainCardBuilder"),
        ),
        (
            "cloud_brain",
            _optional_builder("modules.cloud_brain.widget", "CloudBrainCardBuilder"),
        ),
        (
            "database",
            _optional_builder("modules.database.widget", "DatabaseCardBuilder"),
        ),
        (
            "discord_terminal",
            _optional_builder("modules.discord_terminal.widget", "DiscordTerminalCardBuilder"),
        ),
        (
            "file_access",
            _optional_builder("modules.file_access.widget", "FileAccessCardBuilder"),
        ),
        (
            "gap_filler",
            _optional_builder("modules.gap_filler.widget", "GapFillerCardBuilder"),
        ),
        (
            "gpt_server",
            _optional_builder("modules.gpt_server.widget", "GptServerCardBuilder"),
        ),
        (
            "google_suite",
            _optional_builder("modules.google_suite.widget", "GoogleSuiteCardBuilder"),
        ),
        (
            "gpt_terminal",
            _optional_builder("modules.gpt_terminal.widget", "GptTerminalCardBuilder"),
        ),
        (
            "mcp_bus",
            _optional_builder("modules.mcp_bus.widget", "McpBusCardBuilder"),
        ),
        (
            "ngrok_tunnel",
            _optional_builder("modules.ngrok_tunnel.widget", "NgrokTunnelCardBuilder"),
        ),
        (
            "livekit_voice",
            _optional_builder("modules.livekit_voice.widget", "LivekitVoiceCardBuilder"),
        ),
        (
            "pg_cartridge",
            _optional_builder(
                "modules.mcp_bus.cartridges.postgres.widget",
                "PostgresCartridgeCardBuilder",
            ),
        ),
        (
            "obsidian_suite",
            _optional_builder("modules.obsidian_suite.widget", "ObsidianSuiteCardBuilder"),
        ),
        (
            "rag_engine",
            _optional_builder("modules.rag_engine.widget", "RagEngineCardBuilder"),
        ),
        (
            "vr_terminal",
            _optional_builder("modules.vr_terminal.widget", "VrTerminalCardBuilder"),
        ),
        (
            "web_browser",
            _optional_builder("modules.web_browser.widget", "WebBrowserCardBuilder"),
        ),
    ]
    if builder is not None
}

_CONTROLS_HEIGHT: dict[str, float] = {
    name: cls.CONTROLS_HEIGHT for name, cls in CARD_BUILDERS.items()
}


def compute_card_height(
    module_name: str,
    port_rows: int,
    padding: float = 13.0,
    controls_height: float | None = None,
) -> float:
    """Return the card pixel height for a given module type and port count."""
    base = 70.0 + padding + port_rows * 22.0
    controls_h = controls_height
    if controls_h is None:
        controls_h = _CONTROLS_HEIGHT.get(module_name)
    if controls_h is None:
        return base
    return max(base, 56.0 + 2 * padding + controls_h + 14.0 + port_rows * 22.0)
