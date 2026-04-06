from __future__ import annotations

import json
import base64
import hashlib
import platform
import uuid
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from config import PRIVATE_KEY_PATH, PUBLIC_KEY_PATH


def ensure_keys() -> None:
    PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if PRIVATE_KEY_PATH.exists() and PUBLIC_KEY_PATH.exists():
        return

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    PRIVATE_KEY_PATH.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    PUBLIC_KEY_PATH.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def load_private_key():
    return serialization.load_pem_private_key(PRIVATE_KEY_PATH.read_bytes(), password=None)


def load_public_key(public_key_path: str | Path | None = None):
    key_path = Path(public_key_path) if public_key_path else PUBLIC_KEY_PATH
    return serialization.load_pem_public_key(key_path.read_bytes())


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_dict(data: dict[str, Any]) -> str:
    private_key = load_private_key()
    sig = private_key.sign(canonical_json_bytes(data))
    return base64.b64encode(sig).decode("utf-8")


def verify_dict(data: dict[str, Any], signature_b64: str, public_key_path: str | Path | None = None) -> bool:
    try:
        public_key = load_public_key(public_key_path)
        sig = base64.b64decode(signature_b64)
        public_key.verify(sig, canonical_json_bytes(data))
        return True
    except Exception:
        return False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_device_identifier() -> str:
    system_name = platform.system().lower()

    if "linux" in system_name:
        for p in [
            "/etc/machine-id",
            "/var/lib/dbus/machine-id",
            "/sys/class/dmi/id/product_uuid",
            "/proc/device-tree/serial-number",
        ]:
            pp = Path(p)
            if pp.exists():
                try:
                    return pp.read_text(encoding="utf-8", errors="ignore").strip().replace("\x00", "")
                except Exception:
                    pass

    return f"MAC-{uuid.getnode():012X}"


def short_id(text: str, keep: int = 12) -> str:
    text = str(text).strip()
    return text if len(text) <= keep else text[:keep]