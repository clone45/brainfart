"""
brainfart - Batteries-included memory for voice bots.

FAISS + SQLite + embeddings + encryption in one pip install.

Quick start:
    >>> from brainfart import MemoryProcessor
    >>> processor = MemoryProcessor(user_id="user123")  # Works with sensible defaults

Features:
    - Vector similarity search with FAISS
    - Persistent storage in SQLite
    - Automatic embeddings using sentence-transformers
    - Memory extraction with Gemini
    - Optional at-rest encryption
    - Zero-config with sensible defaults
"""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

# Lazy imports to keep package load fast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import MemoryProcessor
    from .memory import LocalMemory, MemoryResult
    from .config import MemorySettings
    from .crypto import MemoryCrypto
    from .embeddings import EmbeddingService
    from .extraction import extract_memories, extract_and_store, ExtractionResult


def __getattr__(name: str):
    """Lazy load heavy components on first access."""
    if name == "MemoryProcessor":
        from .processor import MemoryProcessor

        return MemoryProcessor
    if name == "LocalMemory":
        from .memory import LocalMemory

        return LocalMemory
    if name == "MemoryResult":
        from .memory import MemoryResult

        return MemoryResult
    if name == "MemorySettings":
        from .config import MemorySettings

        return MemorySettings
    if name == "MemoryCrypto":
        from .crypto import MemoryCrypto

        return MemoryCrypto
    if name == "EmbeddingService":
        from .embeddings import EmbeddingService

        return EmbeddingService
    if name == "extract_memories":
        from .extraction import extract_memories

        return extract_memories
    if name == "extract_and_store":
        from .extraction import extract_and_store

        return extract_and_store
    if name == "ExtractionResult":
        from .extraction import ExtractionResult

        return ExtractionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [
        "__version__",
        "MemoryProcessor",
        "LocalMemory",
        "MemoryResult",
        "MemorySettings",
        "MemoryCrypto",
        "EmbeddingService",
        "extract_memories",
        "extract_and_store",
        "ExtractionResult",
    ]


__all__ = [
    "__version__",
    "MemoryProcessor",
    "LocalMemory",
    "MemoryResult",
    "MemorySettings",
    "MemoryCrypto",
    "EmbeddingService",
    "extract_memories",
    "extract_and_store",
    "ExtractionResult",
]
