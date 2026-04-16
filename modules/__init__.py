"""
modules — Plugin directory for all Miniloader rack modules.

Each sub-folder is an autonomous plugin containing:
    logic.py    — Backend class (inherits BaseModule) + register() entry-point
    widget.py   — UI drawing instructions (optional for headless)
    MODULE.md   — Self-contained documentation with JSON payload examples
    assets/     — Sprite sheets, icons, textures specific to this module

The Hypervisor auto-discovers plugins by scanning this directory at boot.
"""
