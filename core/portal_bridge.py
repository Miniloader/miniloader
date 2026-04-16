"""QWebChannel bridge API exposed to portal.miniloader.ai."""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    from core.download_manager import DownloadManager
    from core.entitlement_store import EntitlementStore
    from core.module_installer import ModuleInstaller
    from core.vault import VaultManager

log = logging.getLogger(__name__)

# ── Portal AWS config ─────────────────────────────────────────────────────────
# These values come from the deployed Amplify backend (portal.miniloader.ai).
# Override via environment variables if needed for dev/staging environments.
_IDENTITY_POOL_ID = os.getenv(
    "MINILOADER_IDENTITY_POOL_ID",
    "us-east-1:95062b72-ae23-4252-9945-3f2123325150",
)
_AWS_REGION = os.getenv("MINILOADER_AWS_REGION", "us-east-1")
_APPSYNC_URL = os.getenv(
    "MINILOADER_APPSYNC_URL",
    "https://y5ag5reonvgvbm5v6u6rz3nequ.appsync-api.us-east-1.amazonaws.com/graphql",
)
_VAULT_CREDS_KEY = "aws.guest_credentials"

_DOWNLOAD_URL_QUERY = """
query RequestDownloadUrl($itemId: String!) {
  requestDownloadUrl(itemId: $itemId) {
    url
    filename
    expiresAt
  }
}
"""


class PortalBridge(QObject):
    """Bridge between portal web UI and local app state/download manager."""

    downloadStarted = Signal(str)
    downloadProgress = Signal(str)
    downloadComplete = Signal(str)
    downloadFailed = Signal(str)
    moduleInstalled = Signal(str)
    moduleHotloadRequested = Signal(object)

    def __init__(
        self,
        vault: "VaultManager | None" = None,
        download_manager: "DownloadManager | None" = None,
        module_installer: "ModuleInstaller | None" = None,
        entitlement_store: "EntitlementStore | None" = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._vault = vault
        self._download_manager = download_manager
        self._module_installer = module_installer
        self._entitlement_store = entitlement_store
        self._pending_module_downloads: dict[str, dict[str, Any]] = {}
        if self._download_manager is not None:
            self._download_manager.set_callbacks(
                on_progress=self._on_download_progress,
                on_complete=self._on_download_complete,
                on_failed=self._on_download_failed,
            )

    @property
    def is_logged_in(self) -> bool:
        return self._vault is not None

    def _json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=True)

    def _to_wire(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row.get("id"),
            "repoId": row.get("repo_id"),
            "filename": row.get("filename"),
            "variant": row.get("variant"),
            "size": row.get("size"),
            "url": row.get("url"),
            "status": row.get("status"),
            "kind": row.get("kind", "file"),
            "progress": row.get("progress"),
            "localPath": row.get("local_path"),
            "error": row.get("error"),
            "createdAt": row.get("created_at"),
            "completedAt": row.get("completed_at"),
        }

    def _normalize_download_url(self, raw_url: str) -> str:
        """Return a fully-qualified HTTPS URL suitable for the download manager.

        If *raw_url* is already an http/https URL it is returned unchanged.
        Otherwise the value is treated as an S3 asset key and a presigned URL
        is requested from the AppSync backend using Identity Pool guest
        credentials — no local AWS login required.
        """
        url = str(raw_url or "").strip()
        if not url:
            return ""
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return url
        if parsed.scheme:
            return url
        if url.startswith("//"):
            return f"https:{url}"

        # Bare asset key (e.g. "prod_store_demo_free/1.0.0/store_demo-1.0.0.minimod")
        # — resolve via AppSync using Identity Pool unauthenticated credentials.
        return self._presign_asset_key(url)

    def _get_guest_credentials(self) -> dict:
        """Return valid short-lived AWS credentials from the Cognito Identity Pool.

        Credentials are cached in the vault's encrypted secret storage and
        refreshed automatically when fewer than 5 minutes remain.
        """
        import boto3  # noqa: PLC0415

        # Try to load cached credentials from vault.
        if self._vault is not None:
            raw = self._vault.get_secret(_VAULT_CREDS_KEY)
            if raw:
                try:
                    creds = json.loads(raw)
                    expiry = datetime.fromisoformat(creds["Expiration"])
                    if expiry > datetime.now(tz=timezone.utc) + timedelta(minutes=5):
                        return creds
                except Exception:
                    pass  # stale or corrupt — fall through to refresh

        # Fetch fresh credentials from the Identity Pool (no login required).
        # Use UNSIGNED so boto3 skips the local credential chain entirely —
        # GetId / GetCredentialsForIdentity are public unauthenticated calls.
        from botocore import UNSIGNED  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415

        client = boto3.client(
            "cognito-identity",
            region_name=_AWS_REGION,
            config=Config(signature_version=UNSIGNED),
        )
        identity = client.get_id(IdentityPoolId=_IDENTITY_POOL_ID)
        result = client.get_credentials_for_identity(IdentityId=identity["IdentityId"])
        creds = result["Credentials"]

        # Normalise Expiration to an ISO string so it survives JSON round-trips.
        expiry = creds.get("Expiration")
        if hasattr(expiry, "isoformat"):
            creds = {**creds, "Expiration": expiry.isoformat()}

        # Persist in vault encrypted storage.
        if self._vault is not None:
            try:
                self._vault.set_secret(_VAULT_CREDS_KEY, json.dumps(creds))
            except Exception:
                log.warning("Could not cache guest credentials in vault", exc_info=True)

        return creds

    def _presign_asset_key(self, asset_key: str) -> str:
        """Call AppSync requestDownloadUrl to get a presigned S3 URL.

        Uses Cognito Identity Pool unauthenticated (guest) credentials signed
        with IAM SigV4 — works on any machine without AWS SSO or user login.
        """
        try:
            import requests  # noqa: PLC0415
            from requests_aws4auth import AWS4Auth  # noqa: PLC0415

            # Derive item_id from the asset key path.
            # e.g. "prod_store_demo_free/1.0.0/store_demo-1.0.0.minimod" -> "prod_store_demo_free"
            item_id = asset_key.lstrip("/").split("/")[0]

            creds = self._get_guest_credentials()
            auth = AWS4Auth(
                creds["AccessKeyId"],
                creds["SecretKey"],
                _AWS_REGION,
                "appsync",
                session_token=creds["SessionToken"],
            )
            response = requests.post(
                _APPSYNC_URL,
                json={"query": _DOWNLOAD_URL_QUERY, "variables": {"itemId": item_id}},
                auth=auth,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if errors := payload.get("errors"):
                raise RuntimeError(f"AppSync errors: {errors}")
            return payload["data"]["requestDownloadUrl"]["url"]
        except Exception:
            log.exception("Failed to generate presigned URL for asset key %r", asset_key)
            return ""

    def _on_download_progress(self, payload: dict[str, Any]) -> None:
        self.downloadProgress.emit(self._json(payload))

    def _on_download_complete(self, payload: dict[str, Any]) -> None:
        self.downloadComplete.emit(self._json(payload))
        download_id = str(payload.get("id") or "")
        if not download_id:
            return
        kind = str(payload.get("kind") or "").strip().lower()
        if not kind and self._download_manager is not None:
            row = self._download_manager.get(download_id)
            kind = str((row or {}).get("kind") or "").strip().lower()
        log.info("Download complete: id=%s kind=%r local_path=%r", download_id, kind, payload.get("localPath"))
        if kind != "module":
            return
        self._install_downloaded_module(download_id=download_id, payload=payload)

    def _on_download_failed(self, payload: dict[str, Any]) -> None:
        download_id = str(payload.get("id") or "")
        if download_id:
            self._pending_module_downloads.pop(download_id, None)
        self.downloadFailed.emit(self._json(payload))

    def _install_downloaded_module(self, *, download_id: str, payload: dict[str, Any]) -> None:
        if self._module_installer is None:
            self.moduleInstalled.emit(
                self._json(
                    {
                        "ok": False,
                        "id": download_id,
                        "error": "module-installer-unavailable",
                    }
                )
            )
            return

        pending = self._pending_module_downloads.get(download_id, {})
        local_path = str(payload.get("localPath") or "").strip()
        if not local_path and self._download_manager is not None:
            row = self._download_manager.get(download_id)
            local_path = str((row or {}).get("local_path") or "").strip()
        if not local_path:
            self.moduleInstalled.emit(
                self._json(
                    {
                        "ok": False,
                        "id": download_id,
                        "error": "missing-local-path",
                    }
                )
            )
            return

        try:
            result = self._module_installer.install_package(local_path, allow_upgrade=True)
            log.info(
                "Module package installed: %s v%s -> %s",
                result["manifest"].get("name"),
                result["manifest"].get("version"),
                result["module_dir"],
            )
            plugin, module_cls = self._module_installer.hot_load_module(result["module_dir"])
            hotload_payload = {
                "plugin": plugin,
                "module_cls": module_cls,
                "manifest": result["manifest"],
                "module_dir": str(result["module_dir"]),
                "download_id": download_id,
            }
            self.moduleHotloadRequested.emit(hotload_payload)
            self.moduleInstalled.emit(
                self._json(
                    {
                        "ok": True,
                        "id": download_id,
                        "moduleName": plugin["name"],
                        "displayName": result["manifest"].get("display_name", ""),
                        "version": result["manifest"].get("version", ""),
                        "productId": result["manifest"].get("product_id", ""),
                        "modulePath": str(result["module_dir"]),
                    }
                )
            )
        except Exception as exc:
            log.exception(
                "Failed to install module from %r (download_id=%s)", local_path, download_id
            )
            self.moduleInstalled.emit(
                self._json(
                    {
                        "ok": False,
                        "id": download_id,
                        "moduleName": str(pending.get("moduleName") or ""),
                        "productId": str(pending.get("productId") or ""),
                        "error": str(exc),
                    }
                )
            )
        finally:
            self._pending_module_downloads.pop(download_id, None)

    @Slot(result=str)
    def ping(self) -> str:
        """Basic health check for JS handshake."""
        return "pong"

    @Slot(result=str)
    def getUserId(self) -> str:
        """Return the local user identifier when available."""
        if self._vault is None:
            return ""
        return self._vault.user_id

    @Slot(result=str)
    def getUsername(self) -> str:
        """Return the local username when available."""
        if self._vault is None:
            return ""
        return self._vault.username

    @Slot(str, result=str)
    def requestCheckout(self, payload_json: str) -> str:
        if not self.is_logged_in:
            return self._json({"ok": False, "error": "not-logged-in"})
        if self._entitlement_store is None:
            return self._json({"ok": False, "error": "entitlement-store-unavailable"})
        if self._vault is None:
            return self._json({"ok": False, "error": "vault-unavailable"})

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "invalid-json"})
        if not isinstance(payload, dict):
            return self._json({"ok": False, "error": "invalid-json"})

        user_id = str(payload.get("userId") or "").strip()
        if not user_id or user_id != self._vault.user_id:
            return self._json({"ok": False, "error": "user-mismatch"})

        # New contract: payload uses itemIds and bridge assigns/reuses license keys.
        raw_item_ids = payload.get("itemIds")
        if isinstance(raw_item_ids, list):
            requested_item_ids: list[str] = []
            seen_item_ids: set[str] = set()
            for raw_item_id in raw_item_ids:
                item_id = str(raw_item_id or "").strip()
                if not item_id or item_id in seen_item_ids:
                    continue
                seen_item_ids.add(item_id)
                requested_item_ids.append(item_id)
            if not requested_item_ids:
                return self._json({"ok": False, "error": "invalid-itemIds"})
            try:
                results = self._entitlement_store.assign_item_ids(requested_item_ids)
            except Exception:
                log.exception("Failed to assign entitlements from requestCheckout")
                return self._json({"ok": False, "error": "entitlement-store-write-failed"})
            return self._json({"results": results})

        # Legacy fallback: accepts explicit items [{itemId, licenseKey}] payload.
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            return self._json({"ok": False, "error": "invalid-itemIds"})

        normalized_items: list[dict[str, str]] = []
        requested_item_ids: list[str] = []
        seen_item_ids: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("itemId") or "").strip()
            license_key = str(item.get("licenseKey") or "").strip()
            if not item_id or not license_key:
                continue
            normalized_items.append({"itemId": item_id, "licenseKey": license_key})
            if item_id not in seen_item_ids:
                requested_item_ids.append(item_id)
                seen_item_ids.add(item_id)

        if not normalized_items:
            return self._json({"ok": False, "error": "invalid-itemIds"})

        try:
            persisted = self._entitlement_store.upsert(normalized_items)
        except Exception:
            log.exception("Failed to persist entitlements from requestCheckout")
            return self._json({"ok": False, "error": "entitlement-store-write-failed"})

        persisted_by_item_id = {
            str(item.get("itemId") or ""): {
                "itemId": str(item.get("itemId") or ""),
                "licenseKey": str(item.get("licenseKey") or ""),
            }
            for item in persisted
            if str(item.get("itemId") or "").strip() and str(item.get("licenseKey") or "").strip()
        }
        results = [
            persisted_by_item_id[item_id]
            for item_id in requested_item_ids
            if item_id in persisted_by_item_id
        ]
        return self._json({"results": results})

    @Slot(result=str)
    def getEntitlements(self) -> str:
        if not self.is_logged_in or self._entitlement_store is None:
            return self._json([])
        try:
            return self._json(self._entitlement_store.get_all())
        except Exception:
            log.exception("Failed to read entitlements")
            return self._json([])

    @Slot(str, result=str)
    def requestDownload(self, payload_json: str) -> str:
        if not self.is_logged_in:
            return self._json({"ok": False, "error": "not-logged-in"})
        if self._download_manager is None:
            return self._json({"ok": False, "error": "download-manager-unavailable"})

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "invalid-json"})

        required = ("url", "filename", "repoId")
        for key in required:
            if not str(payload.get(key, "")).strip():
                return self._json({"ok": False, "error": f"missing-{key}"})
        normalized_url = self._normalize_download_url(str(payload.get("url") or ""))
        if not normalized_url:
            return self._json({"ok": False, "error": "missing-url"})

        download_id = str(uuid.uuid4())
        filename = str(payload["filename"]).strip()
        self._download_manager.enqueue(
            download_id=download_id,
            repo_id=str(payload["repoId"]).strip(),
            filename=filename,
            variant=str(payload.get("variant", filename)).strip() or filename,
            size=str(payload.get("size", "")).strip(),
            url=normalized_url,
        )
        self.downloadStarted.emit(filename)
        return self._json({"ok": True, "id": download_id})

    @Slot(str, result=str)
    def requestModuleDownload(self, payload_json: str) -> str:
        if not self.is_logged_in:
            return self._json({"ok": False, "error": "not-logged-in"})
        if self._download_manager is None:
            return self._json({"ok": False, "error": "download-manager-unavailable"})
        if self._module_installer is None:
            return self._json({"ok": False, "error": "module-installer-unavailable"})

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "invalid-json"})

        required = ("url", "filename", "productId")
        for key in required:
            if not str(payload.get(key, "")).strip():
                return self._json({"ok": False, "error": f"missing-{key}"})
        normalized_url = self._normalize_download_url(str(payload.get("url") or ""))
        if not normalized_url:
            return self._json({"ok": False, "error": "missing-url"})

        download_id = str(uuid.uuid4())
        filename = str(payload["filename"]).strip()
        product_id = str(payload["productId"]).strip()
        module_name = str(payload.get("moduleName", "")).strip()
        self._pending_module_downloads[download_id] = {
            "productId": product_id,
            "moduleName": module_name,
            "filename": filename,
        }
        self._download_manager.enqueue(
            download_id=download_id,
            repo_id=product_id,
            filename=filename,
            variant=module_name or filename,
            size=str(payload.get("size", "")).strip(),
            url=normalized_url,
            kind="module",
        )
        self.downloadStarted.emit(filename)
        return self._json({"ok": True, "id": download_id})

    @Slot(str, result=str)
    def cancelDownload(self, download_id: str) -> str:
        if not self.is_logged_in:
            return self._json({"ok": False, "error": "not-logged-in"})
        if self._download_manager is None:
            return self._json({"ok": False, "error": "download-manager-unavailable"})
        self._download_manager.cancel(download_id)
        return self._json({"ok": True})

    @Slot(result=str)
    def getDownloads(self) -> str:
        if not self.is_logged_in or self._download_manager is None:
            return self._json([])
        rows = self._download_manager.get_all()
        return self._json([self._to_wire(r) for r in rows])

    @Slot(result=str)
    def getInstalledCommunityModules(self) -> str:
        if not self.is_logged_in or self._module_installer is None:
            return self._json([])
        return self._json(self._module_installer.list_installed_modules())

    @Slot(str, result=str)
    def getDownloadStatus(self, download_id: str) -> str:
        if not self.is_logged_in or self._download_manager is None:
            return self._json(None)
        return self._json(self._to_wire(self._download_manager.get(download_id)))

    @Slot(str, result=str)
    def deleteDownload(self, download_id: str) -> str:
        if not self.is_logged_in:
            return self._json({"ok": False, "error": "not-logged-in"})
        if self._download_manager is None:
            return self._json({"ok": False, "error": "download-manager-unavailable"})
        self._download_manager.delete(download_id)
        return self._json({"ok": True})

    @Slot(str, result=str)
    def revealInFolder(self, local_path: str) -> str:
        p = Path(local_path)
        if not p.exists():
            return self._json({"ok": False, "error": "not-found"})
        try:
            if sys.platform == "win32":
                os.system(f'explorer /select,"{p}"')
            elif sys.platform == "darwin":
                os.system(f'open -R "{p}"')
            else:
                os.system(f'xdg-open "{p.parent}"')
        except Exception as exc:
            return self._json({"ok": False, "error": str(exc)})
        return self._json({"ok": True})
