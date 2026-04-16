"""
vault.py — Local user vault and authentication.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets as pysecrets
from pathlib import Path
from typing import Any

import keyring
from bip_utils import Base58Encoder, Bip39EntropyGenerator, Bip39MnemonicGenerator, Bip39MnemonicValidator
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

log = logging.getLogger(__name__)


class VaultError(Exception):
    """Base exception for vault failures."""


class VaultValidationError(VaultError):
    """Raised when user-provided data is invalid."""


class VaultLoginError(VaultError):
    """Raised when vault unlock fails."""


class VaultManager:
    """Handles encrypted local account storage and unlock workflows."""

    APP_DIR = Path.home() / ".miniloader"
    VAULT_PATH = APP_DIR / "vault.json"
    KEYRING_SERVICE = "miniloader"
    KDF_ITERATIONS = 600_000
    KDF_SALT_BYTES = 16
    DEFAULT_GPT_SERVER_SECRET_KEY = "gpt_server.endpoint_password"

    def __init__(
        self,
        vault_path: Path,
        username: str,
        user_id: str,
        profile: dict[str, Any],
        fernet_key: bytes,
        kdf_salt: bytes,
    ) -> None:
        self.vault_path = Path(vault_path)
        self.username = username
        self.user_id = user_id
        self.profile = profile
        self._fernet_key = fernet_key
        self._kdf_salt = kdf_salt

    @staticmethod
    def generate_mnemonic() -> str:
        """Generate a 12-word BIP-39 mnemonic from 128-bit entropy."""
        entropy = Bip39EntropyGenerator(128).Generate()
        return str(Bip39MnemonicGenerator().FromEntropy(entropy))

    @staticmethod
    def validate_mnemonic(mnemonic: str) -> str:
        """Validate and normalize a BIP-39 mnemonic."""
        normalized = " ".join(mnemonic.strip().lower().split())
        if not normalized:
            raise VaultValidationError("Recovery phrase is required.")
        if not Bip39MnemonicValidator().IsValid(normalized):
            raise VaultValidationError("Recovery phrase is invalid.")
        return normalized

    @classmethod
    def create(
        cls,
        mnemonic: str,
        username: str,
        password: str,
        save_to_keyring: bool = False,
        vault_path: Path | None = None,
    ) -> "VaultManager":
        """Create and persist a new local encrypted vault."""
        normalized_mnemonic = cls.validate_mnemonic(mnemonic)
        clean_username = username.strip()
        cls._validate_password(password)
        if not clean_username:
            raise VaultValidationError("Username is required.")

        chosen_vault_path = Path(vault_path or cls.VAULT_PATH)
        kdf_salt = os.urandom(cls.KDF_SALT_BYTES)
        fernet_key = cls._derive_fernet_key(password, kdf_salt)

        profile = {
            "mnemonic": normalized_mnemonic,
            "custom_personas": [],
            "secrets": {
                cls.DEFAULT_GPT_SERVER_SECRET_KEY: pysecrets.token_hex(24),
            },
            "module_comm_key": os.urandom(32).hex(),
        }
        user_id = cls._user_id_from_mnemonic(normalized_mnemonic)
        manager = cls(
            vault_path=chosen_vault_path,
            username=clean_username,
            user_id=user_id,
            profile=profile,
            fernet_key=fernet_key,
            kdf_salt=kdf_salt,
        )
        manager._write_vault()
        if save_to_keyring:
            manager.store_key_in_keyring()
        return manager

    @classmethod
    def from_password(
        cls,
        vault_path: Path | None,
        password: str,
    ) -> "VaultManager":
        """Unlock an existing vault using the user's password."""
        cls._validate_password(password)
        chosen_vault_path = Path(vault_path or cls.VAULT_PATH)
        payload = cls._read_vault_payload(chosen_vault_path)
        salt = bytes.fromhex(payload["kdf_salt"])
        fernet_key = cls._derive_fernet_key(password, salt)
        profile = cls._decrypt_profile(payload["encrypted_blob"], fernet_key)
        return cls(
            vault_path=chosen_vault_path,
            username=payload["username"],
            user_id=payload["user_id"],
            profile=profile,
            fernet_key=fernet_key,
            kdf_salt=salt,
        )

    @classmethod
    def from_keyring(
        cls,
        vault_path: Path | None,
    ) -> "VaultManager | None":
        """Attempt auto-login with the Fernet key from OS keychain."""
        chosen_vault_path = Path(vault_path or cls.VAULT_PATH)
        if not chosen_vault_path.exists():
            return None
        payload = cls._read_vault_payload(chosen_vault_path)
        saved_key = keyring.get_password(cls.KEYRING_SERVICE, payload["user_id"])
        if not saved_key:
            return None
        try:
            fernet_key = saved_key.encode("utf-8")
            profile = cls._decrypt_profile(payload["encrypted_blob"], fernet_key)
        except VaultLoginError:
            return None
        return cls(
            vault_path=chosen_vault_path,
            username=payload["username"],
            user_id=payload["user_id"],
            profile=profile,
            fernet_key=fernet_key,
            kdf_salt=bytes.fromhex(payload["kdf_salt"]),
        )

    @classmethod
    def from_mnemonic(
        cls,
        mnemonic: str,
        vault_path: Path | None,
        password: str,
        username: str,
        save_to_keyring: bool = False,
    ) -> "VaultManager":
        """
        Import an account identity from a mnemonic onto this machine.

        This creates a local vault bound to the provided phrase and password.
        """
        normalized = cls.validate_mnemonic(mnemonic)
        return cls.create(
            mnemonic=normalized,
            username=username,
            password=password,
            save_to_keyring=save_to_keyring,
            vault_path=vault_path,
        )

    def save_profile(self, data: dict[str, Any]) -> None:
        """Persist updated profile data back into the encrypted vault."""
        if "mnemonic" not in data:
            data = {**data, "mnemonic": self.profile.get("mnemonic", "")}
        self.profile = data
        self._write_vault()

    def store_key_in_keyring(self) -> None:
        """Store the active vault key in the OS keychain."""
        keyring.set_password(
            self.KEYRING_SERVICE,
            self.user_id,
            self._fernet_key.decode("utf-8"),
        )

    def clear_keyring_key(self) -> None:
        """Delete auto-login key from the OS keychain."""
        try:
            keyring.delete_password(self.KEYRING_SERVICE, self.user_id)
        except keyring.errors.PasswordDeleteError:
            pass

    def clear_memory_key(self) -> None:
        """Best-effort clearing of active in-memory key material."""
        self._fernet_key = b""

    # ── Secret store ─────────────────────────────────────────────

    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by dotted key (e.g. 'cloud_brain.api_key')."""
        return self.profile.get("secrets", {}).get(key)

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret and persist the vault to disk."""
        secrets = self.profile.setdefault("secrets", {})
        secrets[key] = value
        self._write_vault()

    def get_or_create_secret(self, key: str, nbytes: int = 24) -> str:
        """Return a secret by key, creating and persisting a random default if missing."""
        existing = self.get_secret(key)
        if existing:
            return existing
        generated = pysecrets.token_hex(max(8, nbytes))
        self.set_secret(key, generated)
        return generated

    # ── Per-user data directory ──────────────────────────────────

    def get_user_data_dir(self) -> Path:
        """Return ``~/.miniloader/{user_id}/``."""
        return self.APP_DIR / self.user_id

    def ensure_user_data_dir(self) -> Path:
        """Create the per-user directory tree and return the root."""
        data_dir = self.get_user_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "chroma").mkdir(exist_ok=True)
        return data_dir

    # ── Derived cryptographic keys ───────────────────────────────

    def derive_db_key(self) -> bytes:
        """Derive a 32-byte SQLCipher key via HKDF-SHA256 from the master key."""
        raw_master = base64.urlsafe_b64decode(self._fernet_key)
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._kdf_salt,
            info=b"miniloader_sqlcipher",
        ).derive(raw_master)

    def derive_api_token(self) -> str:
        """Derive a per-session bearer token (never stored on disk)."""
        raw_master = base64.urlsafe_b64decode(self._fernet_key)
        return hmac.new(raw_master, b"miniloader_api", hashlib.sha256).hexdigest()

    # ── Module comm key ──────────────────────────────────────────

    def get_module_comm_key(self) -> bytes:
        """Return the 32-byte key used for HMAC-signing IPC payloads."""
        hex_key = self.profile.get("module_comm_key", "")
        if not hex_key:
            log.warning("module_comm_key missing from vault profile")
            return b""
        return bytes.fromhex(hex_key)

    def rotate_module_comm_key(self) -> bytes:
        """Generate a new module comm key, persist, and return it."""
        new_key = os.urandom(32)
        self.profile["module_comm_key"] = new_key.hex()
        self._write_vault()
        return new_key

    # ── Internal persistence ─────────────────────────────────────

    def _write_vault(self) -> None:
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "username": self.username,
            "user_id": self.user_id,
            "kdf_salt": self._kdf_salt.hex(),
            "encrypted_blob": self._encrypt_profile(self.profile, self._fernet_key),
        }
        self.vault_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def _read_vault_payload(cls, vault_path: Path) -> dict[str, Any]:
        if not vault_path.exists():
            raise VaultValidationError(f"Vault file not found: {vault_path}")
        try:
            payload = json.loads(vault_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VaultValidationError("Vault file is corrupted JSON.") from exc

        required = {"version", "username", "user_id", "kdf_salt", "encrypted_blob"}
        missing = required - set(payload.keys())
        if missing:
            raise VaultValidationError(f"Vault file missing fields: {sorted(missing)}")
        return payload

    @classmethod
    def _derive_fernet_key(cls, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=cls.KDF_ITERATIONS,
        )
        raw_key = kdf.derive(password.encode("utf-8"))
        return base64.urlsafe_b64encode(raw_key)

    @staticmethod
    def _encrypt_profile(profile: dict[str, Any], fernet_key: bytes) -> str:
        token = Fernet(fernet_key).encrypt(
            json.dumps(profile, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        return token.decode("utf-8")

    @staticmethod
    def _decrypt_profile(token: str, fernet_key: bytes) -> dict[str, Any]:
        try:
            raw = Fernet(fernet_key).decrypt(token.encode("utf-8"))
            profile = json.loads(raw.decode("utf-8"))
        except (InvalidToken, json.JSONDecodeError, ValueError) as exc:
            raise VaultLoginError("Unable to unlock vault with provided credentials.") from exc
        if not isinstance(profile, dict):
            raise VaultLoginError("Vault profile payload has invalid shape.")
        return profile

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < 8:
            raise VaultValidationError("Password must be at least 8 characters.")

    @staticmethod
    def _user_id_from_mnemonic(mnemonic: str) -> str:
        digest = hashlib.sha256(mnemonic.encode("utf-8")).digest()
        return Base58Encoder.Encode(digest)
