"""
FAISS vector store wrapper with encryption support.

Handles vector similarity search and persistence with optional at-rest encryption.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from ..crypto import MemoryCrypto


class FaissStore:
    """
    FAISS index wrapper with encrypted persistence.

    Uses IndexFlatIP (inner product / cosine similarity) with ID mapping.
    Supports encrypted storage for crash-safe at-rest encryption.

    Usage:
        store = FaissStore(dimension=384, index_path=Path("user.index"))
        await store.load()
        await store.add([embedding], [id])
        distances, ids = await store.search(query_embedding, k=5)
        await store.save()
    """

    def __init__(
        self,
        dimension: int,
        index_path: Optional[Path] = None,
    ):
        """
        Initialize the FAISS store.

        Args:
            dimension: Embedding dimension (e.g., 384 for MiniLM)
            index_path: Path to persist index. If None, in-memory only.
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index: Optional[faiss.IndexIDMap] = None
        self._dirty = False

    async def load(self) -> bool:
        """
        Load existing index from disk, or create fresh.

        Returns:
            True if loaded from disk, False if created fresh.
        """
        if self.index_path and self.index_path.exists():
            try:
                encrypted_bytes = await asyncio.to_thread(self.index_path.read_bytes)
                decrypted_bytes = MemoryCrypto.decrypt_bytes(encrypted_bytes)

                self.index = await asyncio.to_thread(
                    faiss.deserialize_index,
                    np.frombuffer(decrypted_bytes, dtype=np.uint8),
                )
                logger.debug(f"Loaded FAISS index: {self.index.ntotal} vectors")
                return True
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, creating fresh: {e}")

        # Create fresh index
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        logger.debug(f"Created fresh FAISS index: {self.dimension} dimensions")
        return False

    async def save(self) -> None:
        """Persist index to disk with optional encryption."""
        if not self._dirty or self.index is None or self.index_path is None:
            return

        # Serialize to bytes
        serialized = await asyncio.to_thread(faiss.serialize_index, self.index)
        plaintext_bytes = serialized.tobytes()

        # Encrypt and write
        encrypted_bytes = MemoryCrypto.encrypt_bytes(plaintext_bytes)

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self.index_path.write_bytes, encrypted_bytes)

        self._dirty = False
        logger.debug(f"Saved FAISS index: {self.index.ntotal} vectors")

    async def add(
        self,
        embeddings: np.ndarray,
        ids: List[int],
    ) -> None:
        """
        Add embeddings with IDs to the index.

        Args:
            embeddings: Array of shape (n, dimension)
            ids: List of integer IDs for each embedding
        """
        if self.index is None:
            await self.load()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        id_array = np.array(ids, dtype=np.int64)

        await asyncio.to_thread(self.index.add_with_ids, embeddings, id_array)
        self._dirty = True

    async def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Args:
            query: Query embedding of shape (dimension,) or (1, dimension)
            k: Number of results to return

        Returns:
            Tuple of (distances, ids) arrays of shape (k,)
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Limit k to number of vectors
        k = min(k, self.index.ntotal)

        distances, ids = await asyncio.to_thread(self.index.search, query, k)

        return distances[0], ids[0]

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    async def close(self) -> None:
        """Save and close the store."""
        await self.save()
        self.index = None
