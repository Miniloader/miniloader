"""
module_installer.py — validation and installation for .minimod packages.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from core.base_module import BaseModule

_REQUIRED_MANIFEST_KEYS = {
    "name",
    "version",
    "display_name",
    "description",
    "author",
    "min_app_version",
    "product_id",
}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ModuleInstallerError(Exception):
    """Base error for module package install/validation operations."""


class ModulePackageValidationError(ModuleInstallerError):
    """Raised when a .minimod package fails validation checks."""


class ModuleInstaller:
    """Validate, install, and hot-load community modules."""

    def __init__(
        self,
        target_root: Path,
        *,
        builtin_module_names: set[str] | None = None,
    ) -> None:
        self._target_root = Path(target_root)
        self._target_root.mkdir(parents=True, exist_ok=True)
        self._builtin_module_names = {n.strip() for n in (builtin_module_names or set()) if n.strip()}

    @property
    def target_root(self) -> Path:
        return self._target_root

    def list_installed_modules(self) -> list[dict[str, Any]]:
        """Return installed module metadata from manifest files on disk."""
        out: list[dict[str, Any]] = []
        if not self._target_root.is_dir():
            return out
        for child in sorted(self._target_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            manifest_path = child / "manifest.json"
            logic_path = child / "logic.py"
            if not manifest_path.exists() or not logic_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(manifest, dict):
                continue
            out.append(
                {
                    "name": str(manifest.get("name") or child.name),
                    "displayName": str(manifest.get("display_name") or child.name),
                    "version": str(manifest.get("version") or ""),
                    "description": str(manifest.get("description") or ""),
                    "productId": str(manifest.get("product_id") or ""),
                    "modulePath": str(child),
                    "installed": True,
                }
            )
        return out

    def validate_package(self, zip_path: str | Path) -> dict[str, Any]:
        """Validate package structure and return normalized manifest."""
        path = Path(zip_path)
        if not path.exists():
            raise ModulePackageValidationError(f"Package not found: {path}")
        if path.suffix.lower() not in {".minimod", ".zip"}:
            raise ModulePackageValidationError("Package must use .minimod (or .zip) extension")

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not names:
                    raise ModulePackageValidationError("Package is empty")
                self._validate_zip_member_paths(names)
                if "manifest.json" not in names:
                    raise ModulePackageValidationError("Missing required file: manifest.json")
                if "logic.py" not in names:
                    raise ModulePackageValidationError("Missing required file: logic.py")
                manifest_data = zf.read("manifest.json")
        except zipfile.BadZipFile as exc:
            raise ModulePackageValidationError("Invalid ZIP package") from exc

        try:
            manifest = json.loads(manifest_data.decode("utf-8"))
        except Exception as exc:
            raise ModulePackageValidationError("manifest.json must be valid UTF-8 JSON") from exc
        if not isinstance(manifest, dict):
            raise ModulePackageValidationError("manifest.json must contain a JSON object")

        for key in sorted(_REQUIRED_MANIFEST_KEYS):
            value = manifest.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ModulePackageValidationError(f"manifest.json missing non-empty string '{key}'")

        module_name = manifest["name"].strip()
        if not _NAME_RE.match(module_name):
            raise ModulePackageValidationError(
                "manifest.name must match ^[a-z][a-z0-9_]*$ (lowercase snake_case)"
            )
        if module_name in self._builtin_module_names:
            raise ModulePackageValidationError(
                f"manifest.name '{module_name}' collides with built-in module name"
            )
        return {
            "name": module_name,
            "version": manifest["version"].strip(),
            "display_name": manifest["display_name"].strip(),
            "description": manifest["description"].strip(),
            "author": manifest["author"].strip(),
            "min_app_version": manifest["min_app_version"].strip(),
            "product_id": manifest["product_id"].strip(),
        }

    def install_package(
        self,
        zip_path: str | Path,
        *,
        allow_upgrade: bool = False,
    ) -> dict[str, Any]:
        """Validate and install package to community module root."""
        package_path = Path(zip_path)
        manifest = self.validate_package(package_path)
        module_name = manifest["name"]
        target_dir = self._target_root / module_name
        if target_dir.exists() and not allow_upgrade:
            raise ModuleInstallerError(
                f"Module '{module_name}' is already installed. Use allow_upgrade=True to replace it."
            )

        temp_dir = self._target_root / f".install-{module_name}-{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=False)

        try:
            with zipfile.ZipFile(package_path, "r") as zf:
                zf.extractall(temp_dir)

            if not (temp_dir / "logic.py").exists():
                raise ModuleInstallerError("Installed package is missing logic.py after extraction")
            if not (temp_dir / "manifest.json").exists():
                raise ModuleInstallerError("Installed package is missing manifest.json after extraction")

            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=False)
            temp_dir.replace(target_dir)
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return {
            "module_name": module_name,
            "module_dir": target_dir,
            "manifest": manifest,
        }

    def hot_load_module(self, module_dir: str | Path) -> tuple[dict[str, Any], type[BaseModule]]:
        """Import an installed module's logic file and return plugin + class."""
        module_path = Path(module_dir)
        logic_path = module_path / "logic.py"
        if not logic_path.exists():
            raise ModuleInstallerError(f"No logic.py found in installed module: {module_path}")

        spec_name = f"miniloader.runtime.{module_path.name}.logic_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(spec_name, str(logic_path))
        if spec is None or spec.loader is None:
            raise ModuleInstallerError(f"Could not build import spec for {logic_path}")
        module_logic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_logic)

        register_fn = getattr(module_logic, "register", None)
        if not callable(register_fn):
            raise ModuleInstallerError("Installed module logic.py is missing register(hypervisor)")

        module_cls: type[BaseModule] | None = None
        for attr_name in dir(module_logic):
            obj = getattr(module_logic, attr_name)
            if isinstance(obj, type) and issubclass(obj, BaseModule) and obj is not BaseModule:
                module_cls = obj
                break
        if module_cls is None:
            raise ModuleInstallerError("Installed module logic.py has no BaseModule subclass")

        plugin = {
            "name": module_path.name,
            "register_fn": register_fn,
        }
        return plugin, module_cls

    @staticmethod
    def _validate_zip_member_paths(names: list[str]) -> None:
        for name in names:
            normalized = name.replace("\\", "/")
            if normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
                raise ModulePackageValidationError("Package contains unsafe paths")
