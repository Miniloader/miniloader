"""UI-only aliases for human-friendly module and port labels."""

from __future__ import annotations


MODULE_UI_ALIASES: dict[str, str] = {
    "basic_brain": "Local Brain",
    "cloud_brain": "Cloud Brain",
    "gpt_terminal": "Chat Terminal",
    "gpt_server": "AI Server",
    "rag_engine": "Knowledge Engine",
    "file_access": "File Vault",
    "database": "Database",
    "discord_terminal": "Discord Terminal",
    "mcp_bus": "Tool Bus",
    "ngrok_tunnel": "Web Gateway",
    "gap_filler": "Spacer",
    "pg_cartridge": "Postgres Cartridge",
}

PORT_UI_ALIASES: dict[str, str] = {
    "BRAIN_OUT": "AI_OUT",
    "BRAIN_IN": "AI_IN",
    "LOCAL_IP_OUT": "API_OUT",
    "LOCAL_IP_IN": "API_IN",
    "DB_IN_OUT": "STORAGE_LINK",
    "CONTEXT_IN": "DOCUMENT_IN",
    "CONTEXT_OUT": "KNOWLEDGE_OUT",
    "MCP_IN": "TOOLS_IN",
    "MCP_UPSTREAM": "FROM_PROVIDERS",
    "MCP_DOWNSTREAM": "TO_CONSUMERS",
}

# Per-module faceplate label overrides: space-separated, hardware-panel style.
# Keys are module names; values map internal port names to their silk-screened legend text.
# Only ports that need a custom short form need an entry — others fall through to faceplate_label().
PORT_FACEPLATE_OVERRIDES: dict[str, dict[str, str]] = {
    "gpt_terminal": {
        "SERVER_IN":   "API IN",
        "LOCAL_IP_IN": "API IN",
        "DB_IN_OUT":   "STORAGE",
        "CONTEXT_IN":  "DOC IN",
        "MCP_IN":      "TOOLS IN",
        "WEB_OUT":     "WEB OUT",
        "VOICE_IN":    "VOICE IN",
    },
    "discord_terminal": {
        "LOCAL_IP_IN": "API IN",
        "DB_IN_OUT":   "STORAGE",
        "CONTEXT_IN":  "DOC IN",
        "MCP_IN":      "TOOLS IN",
        "WEB_OUT":     "WEB OUT",
    },
    "file_access": {
        "TOOLS_OUT":   "TOOLS OUT",
    },
}


def module_ui_label(module_name: str) -> str:
    key = str(module_name or "").strip().lower()
    if not key:
        return ""
    return MODULE_UI_ALIASES.get(key, key.replace("_", " ").title())


def port_ui_label(port_name: str) -> str:
    """Returns the aliased port name (UPPER_SNAKE_CASE form) used in tooltips and status bars."""
    key = str(port_name or "").strip().upper()
    if not key:
        return ""
    return PORT_UI_ALIASES.get(key, key)


def port_faceplate_label(module_name: str, port_name: str) -> str:
    """Returns the silk-screened faceplate legend for a jack: space-separated, no underscores.

    Per the UI spec all control legends are printed as hardware-panel text — all-caps,
    spaces between words, no underscores.  Module-specific overrides in
    PORT_FACEPLATE_OVERRIDES take priority; otherwise the aliased name is used with
    underscores replaced by spaces.
    """
    mod_key = str(module_name or "").strip().lower()
    port_key = str(port_name or "").strip().upper()
    module_overrides = PORT_FACEPLATE_OVERRIDES.get(mod_key, {})
    if port_key in module_overrides:
        return module_overrides[port_key]
    aliased = port_ui_label(port_name)
    return aliased.replace("_", " ")

