"""
Encryption utilities for memory storage.

Uses Fernet (AES-128-CBC with HMAC-SHA256) for symmetric encryption.
Crash-safe: data is always encrypted on disk, decrypted only in memory.
"""

import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet
from loguru import logger


class MemoryCrypto:
    """
    Handles encryption/decryption for memory storage.

    Key derivation: If a passphrase is provided, derives a Fernet key using
    SHA-256. If a raw key is provided, uses it directly.

    Usage:
        # Initialize once at startup
        MemoryCrypto.initialize("my-secret-passphrase")

        # Encrypt/decrypt strings (for SQLite content)
        encrypted = MemoryCrypto.encrypt_string("secret data")
        decrypted = MemoryCrypto.decrypt_string(encrypted)

        # Encrypt/decrypt bytes (for FAISS index)
        encrypted_bytes = MemoryCrypto.encrypt_bytes(raw_bytes)
        decrypted_bytes = MemoryCrypto.decrypt_bytes(encrypted_bytes)
    """

    _instance: Optional["MemoryCrypto"] = None
    _fernet: Optional[Fernet] = None
    _enabled: bool = False

    @classmethod
    def initialize(cls, key: Optional[str] = None) -> bool:
        """
        Initialize encryption with a key or passphrase.

        Args:
            key: Either a Fernet key (base64, 32 bytes) or a passphrase.
                 If None, encryption is disabled.

        Returns:
            True if encryption is enabled, False if disabled.
        """
        if cls._instance is not None:
            return cls._enabled

        cls._instance = cls()

        if not key:
            logger.info("Memory encryption disabled (no key provided)")
            cls._enabled = False
            return False

        # Derive Fernet key from passphrase
        # Fernet requires exactly 32 bytes, base64 encoded
        derived = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(derived)

        cls._fernet = Fernet(fernet_key)
        cls._enabled = True
        logger.info("Memory encryption enabled")
        return True

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if encryption is enabled."""
        return cls._enabled

    @classmethod
    def encrypt_string(cls, plaintext: str) -> str:
        """
        Encrypt a string, returning base64-encoded ciphertext.

        If encryption is disabled, returns the plaintext unchanged.
        """
        if not cls._enabled or cls._fernet is None:
            return plaintext

        encrypted = cls._fernet.encrypt(plaintext.encode("utf-8"))
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    @classmethod
    def decrypt_string(cls, ciphertext: str) -> str:
        """
        Decrypt a base64-encoded ciphertext string.

        If encryption is disabled, returns the ciphertext unchanged.
        """
        if not cls._enabled or cls._fernet is None:
            return ciphertext

        try:
            encrypted = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
            decrypted = cls._fernet.decrypt(encrypted)
            return decrypted.decode("utf-8")
        except Exception as e:
            # If decryption fails, might be unencrypted legacy data
            logger.warning(f"Decryption failed, returning raw value: {e}")
            return ciphertext

    @classmethod
    def encrypt_bytes(cls, plaintext: bytes) -> bytes:
        """
        Encrypt bytes, returning encrypted bytes.

        If encryption is disabled, returns the plaintext unchanged.
        """
        if not cls._enabled or cls._fernet is None:
            return plaintext

        return cls._fernet.encrypt(plaintext)

    @classmethod
    def decrypt_bytes(cls, ciphertext: bytes) -> bytes:
        """
        Decrypt bytes.

        If encryption is disabled, returns the ciphertext unchanged.
        """
        if not cls._enabled or cls._fernet is None:
            return ciphertext

        try:
            return cls._fernet.decrypt(ciphertext)
        except Exception as e:
            # If decryption fails, might be unencrypted legacy data
            logger.warning(f"Decryption failed, returning raw bytes: {e}")
            return ciphertext

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
        cls._fernet = None
        cls._enabled = False
