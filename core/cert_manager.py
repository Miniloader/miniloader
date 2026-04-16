"""
cert_manager.py — Self-signed TLS certificate generation
==========================================================
Generates a local CA and server certificate for encrypting HTTP
traffic between miniloader's internal servers and clients.

Certificates are stored in ``~/.miniloader/certs/`` and are valid
for 365 days.  SAN includes ``127.0.0.1`` and ``localhost``.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

log = logging.getLogger(__name__)

_CERTS_DIR = Path.home() / ".miniloader" / "certs"
_CA_KEY = _CERTS_DIR / "ca.key"
_CA_CERT = _CERTS_DIR / "ca.pem"
_SERVER_KEY = _CERTS_DIR / "server.key"
_SERVER_CERT = _CERTS_DIR / "server.pem"
_VALIDITY_DAYS = 365


class CertManager:
    """Manage self-signed CA and server certificates."""

    def __init__(self, certs_dir: Path | None = None) -> None:
        self._dir = certs_dir or _CERTS_DIR

    @property
    def ca_cert_path(self) -> Path:
        return self._dir / "ca.pem"

    @property
    def server_cert_path(self) -> Path:
        return self._dir / "server.pem"

    @property
    def server_key_path(self) -> Path:
        return self._dir / "server.key"

    def ensure_certs(self) -> tuple[Path, Path]:
        """Return ``(cert_path, key_path)``, generating if needed."""
        if self.server_cert_path.exists() and self.server_key_path.exists():
            if not self._is_expired():
                return self.server_cert_path, self.server_key_path
            log.info("Server certificate expired — regenerating")
        self._generate()
        return self.server_cert_path, self.server_key_path

    def regenerate_certs(self) -> tuple[Path, Path]:
        """Force-regenerate all certificates."""
        self._generate()
        return self.server_cert_path, self.server_key_path

    def get_cert_expiry(self) -> datetime.datetime | None:
        """Return the server cert's not-after date, or None if missing."""
        if not self.server_cert_path.exists():
            return None
        cert = x509.load_pem_x509_certificate(
            self.server_cert_path.read_bytes()
        )
        return cert.not_valid_after_utc

    # ── Internal ─────────────────────────────────────────────────

    def _is_expired(self) -> bool:
        expiry = self.get_cert_expiry()
        if expiry is None:
            return True
        return datetime.datetime.now(datetime.timezone.utc) >= expiry

    def _generate(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Miniloader Local CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Miniloader"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS * 2))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0), critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )

        srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        srv_name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        san = x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(
                __import__("ipaddress").IPv4Address("127.0.0.1")
            ),
        ])
        srv_cert = (
            x509.CertificateBuilder()
            .subject_name(srv_name)
            .issuer_name(ca_name)
            .public_key(srv_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=_VALIDITY_DAYS))
            .add_extension(san, critical=False)
            .sign(ca_key, hashes.SHA256())
        )

        (self._dir / "ca.key").write_bytes(
            ca_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        self.ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        self.server_key_path.write_bytes(
            srv_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        self.server_cert_path.write_bytes(srv_cert.public_bytes(serialization.Encoding.PEM))
        log.info("TLS certificates generated in %s (valid %d days)", self._dir, _VALIDITY_DAYS)
