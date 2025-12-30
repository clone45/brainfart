"""
Configuration with sensible defaults—works without any setup.

Uses pydantic-settings for environment variable support.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemorySettings(BaseSettings):
    """Configuration with sensible defaults—works without any setup."""

    model_config = SettingsConfigDict(
        env_prefix="BRAINFART_",
        env_file=".env",
        extra="ignore",
    )

    # Gemini configuration (required for memory extraction)
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key. Also reads from GOOGLE_API_KEY.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="Gemini model for memory extraction",
    )

    # Embedding model (all-MiniLM-L6-v2 = 22M params, fast, good quality)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )

    # Storage paths
    data_dir: Path = Field(
        default=Path.home() / ".cache" / "brainfart",
        description="Directory for SQLite DB and FAISS index",
    )

    # Retrieval settings
    top_k: int = Field(default=5, ge=1, le=50, description="Number of memories to retrieve")
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity for retrieval"
    )

    # Encryption settings
    encryption_key: Optional[str] = Field(
        default=None,
        description="Encryption key/passphrase for at-rest encryption. "
        "Also reads from MEMORY_ENCRYPTION_KEY.",
    )

    # Extraction settings
    extraction_window_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of messages to include in extraction window",
    )
    extraction_trigger_interval: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Extract memories every N messages",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fall back to GOOGLE_API_KEY if specific key not set
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        # Fall back to MEMORY_ENCRYPTION_KEY
        if not self.encryption_key:
            self.encryption_key = os.getenv("MEMORY_ENCRYPTION_KEY")


@lru_cache
def get_settings() -> MemorySettings:
    """Get cached settings singleton."""
    return MemorySettings()
