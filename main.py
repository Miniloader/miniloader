import asyncio
import importlib
import importlib.util
import logging
import multiprocessing
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any

from core.base_module import BaseModule
from core.hypervisor import Hypervisor

log = logging.getLogger(__name__)


def _load_logic_module_from_path(module_name: str, logic_path: Path) -> Any:
    """Load a logic.py file from disk under an isolated module namespace."""
    spec_name = f"miniloader.community.{module_name}.logic"
    spec = importlib.util.spec_from_file_location(spec_name, str(logic_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {logic_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_modules(
    modules_path: Path,
    *,
    community_modules_path: Path | None = None,
) -> tuple[list[dict], dict[str, type[BaseModule]], list[dict[str, str]]]:
    """
    Walk the /modules/ directory and dynamically import every plugin
    that exposes a ``register(hypervisor)`` entry-point in its logic.py.

    Returns
    -------
    plugins
        List of ``{"name": str, "register_fn": callable}`` dicts used by
        the UI right-click "New" menu to add modules at runtime.
    module_registry
        ``{MODULE_NAME: ConcreteClass}`` mapping used by ``load_rack()``
        to instantiate modules from a template snapshot.
    startup_errors
        List of startup errors captured during module discovery.
    """
    discovered = []
    registry: dict[str, type[BaseModule]] = {}
    startup_errors: list[dict[str, str]] = []

    candidate_names: set[str] = set()

    if modules_path.is_dir():
        for child in sorted(modules_path.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            has_logic = (child / "logic.py").exists()
            if not has_logic and getattr(sys, "frozen", False):
                has_logic = (child / "logic.pyc").exists()
            if has_logic:
                candidate_names.add(child.name)

    try:
        import modules as modules_pkg

        for mod_info in pkgutil.iter_modules(modules_pkg.__path__):
            if not mod_info.name.startswith("_"):
                candidate_names.add(mod_info.name)
    except Exception as exc:
        log.debug("Module package enumeration skipped: %s", exc)

    for name in sorted(candidate_names):
        if importlib.util.find_spec(f"modules.{name}.logic") is None:
            continue
        try:
            module_logic = importlib.import_module(f"modules.{name}.logic")
        except Exception as exc:
            log.warning("Skipping module '%s': import failed (%s)", name, exc)
            startup_errors.append(
                {
                    "module": name,
                    "phase": "import",
                    "error": str(exc),
                }
            )
            continue
        if not hasattr(module_logic, "register"):
            continue

        discovered.append(
            {
                "name": name,
                "register_fn": module_logic.register,
            }
        )
        # Find the BaseModule subclass defined in this logic module
        for attr_name in dir(module_logic):
            obj = getattr(module_logic, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModule)
                and obj is not BaseModule
                and obj.__module__ == module_logic.__name__
            ):
                registry[obj.MODULE_NAME] = obj
                break

    builtin_names = {str(item.get("name") or "").strip() for item in discovered}
    builtin_names = {name for name in builtin_names if name}

    if community_modules_path is not None and community_modules_path.is_dir():
        for child in sorted(community_modules_path.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            logic_py = child / "logic.py"
            logic_pyc = child / "logic.pyc"
            if not logic_py.exists() and not (getattr(sys, "frozen", False) and logic_pyc.exists()):
                continue
            module_name = child.name
            if module_name in builtin_names:
                log.info(
                    "Skipping community module '%s' (name collides with built-in module)",
                    module_name,
                )
                continue

            module_path = logic_py if logic_py.exists() else logic_pyc
            try:
                module_logic = _load_logic_module_from_path(module_name, module_path)
            except Exception as exc:
                log.warning("Skipping community module '%s': import failed (%s)", module_name, exc)
                startup_errors.append(
                    {
                        "module": module_name,
                        "phase": "community-import",
                        "error": str(exc),
                    }
                )
                continue

            register_fn = getattr(module_logic, "register", None)
            if not callable(register_fn):
                startup_errors.append(
                    {
                        "module": module_name,
                        "phase": "community-validate",
                        "error": "Missing register(hypervisor) entry point",
                    }
                )
                continue

            discovered.append({"name": module_name, "register_fn": register_fn})
            for attr_name in dir(module_logic):
                obj = getattr(module_logic, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseModule)
                    and obj is not BaseModule
                ):
                    registry[obj.MODULE_NAME] = obj
                    break

    return discovered, registry, startup_errors


def _community_modules_dir(vault: Any | None = None) -> Path:
    env_override = os.getenv("MINILOADER_COMMUNITY_MODULES", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    if vault is not None:
        try:
            return vault.ensure_user_data_dir() / "community_modules"
        except Exception:
            pass
    return Path.home() / ".miniloader" / "community_modules"


async def main(vault: Any | None = None, initial_preset: str | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    if getattr(sys, "frozen", False):
        for _h in logging.getLogger().handlers:
            if isinstance(_h, logging.StreamHandler) and hasattr(_h, "stream"):
                _h.stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]

    if getattr(sys, "frozen", False):
        _log_dir = Path.home() / ".miniloader" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(str(_log_dir / "main_debug.log"), mode="w")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(_fh)

    # 1. Boot the Hypervisor (DAG manager + resource monitor)
    hypervisor = Hypervisor(vault=vault)
    await hypervisor.initialize()

    # 1b. Load per-machine backend preference from the encrypted DB
    if vault is not None:
        try:
            from core.settings_store import SettingsStore
            store = SettingsStore(vault)
            mid = SettingsStore.get_machine_id()
            stored_backend = await store.get("selected_backend", machine_id=mid)
            hypervisor.set_stored_backend(stored_backend)
            if stored_backend:
                log.info("Using stored backend preference: %s", stored_backend)
        except Exception as exc:
            log.debug("Could not load stored backend: %s", exc)

    # 2. Discover plugins and build MODULE_REGISTRY
    modules_dir = Path(__file__).parent / "modules"
    community_modules_dir = _community_modules_dir(vault)
    community_modules_dir.mkdir(parents=True, exist_ok=True)
    plugins, module_registry, startup_errors = discover_modules(
        modules_dir,
        community_modules_path=community_modules_dir,
    )
    if startup_errors:
        log.warning("Startup completed with %d module import issue(s)", len(startup_errors))

    # 3. Load template or register discovered modules (all in OFF state).
    #    When onboarding selected a preset, skip default registration so the
    #    rack starts clean and only the preset modules are loaded by the UI.
    if initial_preset is not None:
        log.info(
            "Onboarding preset '%s' selected — skipping default module registration",
            initial_preset,
        )
    else:
        default_template = Path(__file__).parent / "templates" / "default.json"
        if default_template.exists():
            try:
                from core.rack_state import RackSnapshot
                snapshot = RackSnapshot.model_validate_json(
                    default_template.read_text(encoding="utf-8")
                )
                await hypervisor.load_rack(snapshot, module_registry)
                log.info(
                    "Loaded template '%s' — press the power button to start",
                    snapshot.template_name,
                )
            except Exception as exc:
                log.warning("Failed to load default template: %s — registering defaults", exc)
                for plugin in plugins:
                    plugin["register_fn"](hypervisor)
        else:
            for plugin in plugins:
                plugin["register_fn"](hypervisor)
            log.info(
                "%d modules registered (OFF state) — press the power button to start",
                len(hypervisor.active_modules),
            )

    isolated = [
        m.module_id for m in hypervisor.active_modules.values()
        if m.PROCESS_ISOLATION
    ]
    if isolated:
        log.info("Modules requesting process isolation: %s", isolated)

    # 4. Launch the UI canvas (non-blocking)
    from ui.main_window import launch_ui
    from PySide6.QtWidgets import QApplication

    # UI requires an active Qt app, which is created in __main__.
    if QApplication.instance() is None:
        raise RuntimeError("Qt app not initialized")
    launch_ui(
        hypervisor,
        plugins,
        module_registry=module_registry,
        vault=vault,
        startup_errors=startup_errors,
        initial_preset=initial_preset,
    )

    # 5. Enter the async pub/sub event loop
    await hypervisor.run_event_loop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    from core.backend_downloader import ensure_vendor_dll_search_path

    ensure_vendor_dll_search_path()

    # Frozen GUI apps (PyInstaller console=False) have NULL C-runtime
    # stdio handles.  Native libs that fprintf(stderr,...) will segfault.
    # Ensure fd 0/1/2 are valid before any native code is loaded.
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

    from core.crash_reporter import (
        check_for_crash_log,
        clear_crash_log,
        install_crash_hooks,
    )

    install_crash_hooks()
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication
        from qasync import QEventLoop
        from ui.login_dialog import run_vault_gate

        app = QApplication(sys.argv)

        app.setStyle("Fusion")

        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,          QColor("#13161c"))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor("#d8dce4"))
        pal.setColor(QPalette.ColorRole.Base,             QColor("#1a1e28"))
        pal.setColor(QPalette.ColorRole.AlternateBase,    QColor("#1f2432"))
        pal.setColor(QPalette.ColorRole.Text,             QColor("#d8dce4"))
        pal.setColor(QPalette.ColorRole.Button,           QColor("#1e2433"))
        pal.setColor(QPalette.ColorRole.ButtonText,       QColor("#b8c8e0"))
        pal.setColor(QPalette.ColorRole.BrightText,       QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.Highlight,        QColor("#2a4a70"))
        pal.setColor(QPalette.ColorRole.HighlightedText,  QColor("#e8f0ff"))
        pal.setColor(QPalette.ColorRole.ToolTipBase,      QColor("#1e2433"))
        pal.setColor(QPalette.ColorRole.ToolTipText,      QColor("#c8d0dc"))
        pal.setColor(QPalette.ColorRole.PlaceholderText,  QColor("#5c6878"))
        pal.setColor(QPalette.ColorRole.Link,             QColor("#5a9fd4"))
        pal.setColor(QPalette.ColorRole.LinkVisited,      QColor("#8a7cc8"))

        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,  QColor("#5c6878"))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,        QColor("#5c6878"))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,  QColor("#5c6878"))
        app.setPalette(pal)

        app.setStyleSheet(
            "QToolTip { background: #1e2433; color: #c8d0dc;"
            "  border: 1px solid #2a3a50; padding: 4px; }"
            "QMenu { background: #1a1e28; color: #d8dce4;"
            "  border: 1px solid #2a3a50; }"
            "QMenu::item:selected { background: #2a4a70; color: #e8f0ff; }"
            "QMenu::separator { height: 1px; background: #2a3a50; margin: 4px 8px; }"
            "QComboBox QAbstractItemView { background: #1a1e28; color: #d8dce4;"
            "  selection-background-color: #2a4a70; selection-color: #e8f0ff; }"
            "QScrollBar:vertical { background: #13161c; width: 10px; }"
            "QScrollBar::handle:vertical { background: #2a3a50; border-radius: 4px;"
            "  min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar:horizontal { background: #13161c; height: 10px; }"
            "QScrollBar::handle:horizontal { background: #2a3a50; border-radius: 4px;"
            "  min-width: 20px; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        )

        crash_data = check_for_crash_log()
        if crash_data:
            from ui.crash_dialog import CrashReportDialog

            dialog = CrashReportDialog(crash_data)
            dialog.exec()
            clear_crash_log()

        vault, initial_preset = run_vault_gate()
        if vault is None:
            sys.exit(0)

        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)
        try:
            with loop:
                loop.run_until_complete(main(vault=vault, initial_preset=initial_preset))
        except RuntimeError as exc:
            if "Event loop stopped before Future completed" not in str(exc):
                raise
    except ImportError as exc:
        print("[miniloader] GUI dependencies are required to run Miniloader.")
        print(f"[miniloader] Startup failed: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[miniloader] Shutting down…")
        sys.exit(0)
