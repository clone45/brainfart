"""
LocalMemory - Core memory management class.

Orchestrates FAISS vector search and SQLite storage for a user/agent pair.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from .config import MemorySettings
from .crypto import MemoryCrypto
from .embeddings import EmbeddingService
from .storage.faiss_store import FaissStore
from .storage.sqlite_store import SqliteStore, MemoryRecord


@dataclass
class MemoryResult:
    """A memory with similarity score."""

    id: int
    content: str
    category: str
    importance: int
    timestamp: float
    similarity: float  # Cosine similarity (0-1)


class LocalMemory:
    """
    Local memory store combining FAISS vector search with SQLite metadata.

    Each user/agent pair gets isolated storage:
    - {data_dir}/{agent_id}/{user_id}.index — FAISS vectors
    - {data_dir}/{agent_id}/{user_id}.db — SQLite metadata

    Usage:
        memory = LocalMemory(settings, user_id="user123", agent_id="bot1")
        await memory.load()
        await memory.store("User lives in San Francisco", category="identity", importance=5)
        results = await memory.retrieve("Where does the user live?", k=5)
        await memory.save()
    """

    def __init__(
        self,
        settings: MemorySettings,
        user_id: str,
        agent_id: str = "default",
    ):
        """
        Initialize local memory for a user/agent pair.

        Args:
            settings: Memory configuration
            user_id: User identifier
            agent_id: Agent identifier (for multi-agent isolation)
        """
        self.settings = settings
        self.user_id = user_id
        self.agent_id = agent_id

        # Build paths
        agent_dir = Path(settings.data_dir) / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = agent_dir / f"{user_id}.index"
        self.db_path = agent_dir / f"{user_id}.db"

        # Components (lazy initialized)
        self._embedder: Optional[EmbeddingService] = None
        self._faiss: Optional[FaissStore] = None
        self._sqlite: Optional[SqliteStore] = None
        self._loaded = False

    @property
    def embedder(self) -> EmbeddingService:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = EmbeddingService(self.settings.embedding_model)
        return self._embedder

    async def load(self) -> float:
        """
        Load storage from disk.

        Returns:
            Load time in milliseconds.
        """
        if self._loaded:
            return 0.0

        start = time.perf_counter()

        # Initialize encryption if key provided
        if self.settings.encryption_key:
            MemoryCrypto.initialize(self.settings.encryption_key)

        # Load FAISS
        self._faiss = FaissStore(
            dimension=self.embedder.dimension,
            index_path=self.index_path,
        )
        await self._faiss.load()

        # Load SQLite
        self._sqlite = SqliteStore(db_path=self.db_path)
        self._sqlite.open()

        self._loaded = True
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"Loaded memory for {self.agent_id}/{self.user_id}: "
            f"{self._sqlite.size} memories, {self._faiss.size} vectors "
            f"({elapsed_ms:.1f}ms)"
        )

        return elapsed_ms

    async def store(
        self,
        content: str,
        category: str = "context",
        importance: int = 3,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> int:
        """
        Store a single memory.

        Args:
            content: Memory text (e.g., "User's brother Mike works at Google")
            category: One of [identity, preference, context, relationship, surprise]
            importance: 1-5 scale (5 = core identity)
            session_id: Optional session where memory was extracted
            turn_number: Optional turn number when extracted

        Returns:
            Memory ID
        """
        if not self._loaded:
            await self.load()

        # Embed content
        embedding = await self.embedder.embed(content)

        # Store in SQLite (gets encrypted)
        memory_id = self._sqlite.add(
            content=content,
            category=category,
            importance=importance,
            session_id=session_id,
            turn_number=turn_number,
        )

        # Add to FAISS
        await self._faiss.add(embedding.reshape(1, -1), [memory_id])

        logger.debug(
            f"Stored memory {memory_id} for {self.agent_id}/{self.user_id}: "
            f"[{category}] {content[:50]}..."
        )

        return memory_id

    async def store_batch(
        self,
        memories: List[dict],
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[int]:
        """
        Store multiple memories in a batch (more efficient).

        Args:
            memories: List of dicts with keys: content, category, importance
            session_id: Optional session ID
            turn_number: Optional turn number

        Returns:
            List of memory IDs
        """
        if not memories:
            return []

        if not self._loaded:
            await self.load()

        # Embed all at once
        contents = [m["content"] for m in memories]
        embeddings = await self.embedder.embed_batch(contents)

        # Store in SQLite
        ids = self._sqlite.add_batch(
            memories=memories,
            session_id=session_id,
            turn_number=turn_number,
        )

        # Add to FAISS
        await self._faiss.add(embeddings, ids)

        logger.info(
            f"Stored {len(memories)} memories for {self.agent_id}/{self.user_id} "
            f"(total: {self._sqlite.size})"
        )

        return ids

    async def retrieve(
        self,
        query: str,
        k: int = None,
        categories: Optional[List[str]] = None,
        min_similarity: float = None,
    ) -> List[MemoryResult]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query
            k: Number of results (default: settings.top_k)
            categories: Filter by categories (optional)
            min_similarity: Minimum similarity threshold (default: settings.similarity_threshold)

        Returns:
            List of MemoryResult sorted by relevance
        """
        if not self._loaded:
            await self.load()

        if self._faiss.size == 0:
            return []

        k = k or self.settings.top_k
        min_similarity = min_similarity or self.settings.similarity_threshold

        # Embed query
        query_embedding = await self.embedder.embed(query)

        # Search FAISS
        distances, ids = await self._faiss.search(query_embedding, k=k)

        # Fetch from SQLite and filter
        results = []
        for similarity, memory_id in zip(distances, ids):
            if memory_id == -1:
                continue

            # Similarity threshold
            if similarity < min_similarity:
                continue

            record = self._sqlite.get(int(memory_id))
            if record is None:
                continue

            # Category filter
            if categories and record.category not in categories:
                continue

            results.append(
                MemoryResult(
                    id=record.id,
                    content=record.content,
                    category=record.category,
                    importance=record.importance,
                    timestamp=record.timestamp,
                    similarity=float(similarity),
                )
            )

        return results

    async def get_identity_memories(self, k: int = 10) -> List[MemoryResult]:
        """
        Retrieve identity and preference memories.

        These are facts about who the user is, not what they're working on.
        """
        if not self._loaded:
            await self.load()

        records = self._sqlite.get_by_category(
            categories=["identity", "preference"],
            limit=k,
        )

        return [
            MemoryResult(
                id=r.id,
                content=r.content,
                category=r.category,
                importance=r.importance,
                timestamp=r.timestamp,
                similarity=1.0,  # Not from search
            )
            for r in records
        ]

    async def save(self) -> None:
        """Persist to disk."""
        if self._faiss:
            await self._faiss.save()
        if self._sqlite:
            self._sqlite.commit()

        logger.debug(f"Saved memory for {self.agent_id}/{self.user_id}")

    async def close(self) -> None:
        """Save and close."""
        await self.save()

        if self._faiss:
            await self._faiss.close()
        if self._sqlite:
            self._sqlite.close()

        self._faiss = None
        self._sqlite = None
        self._loaded = False

    def get_stats(self) -> dict:
        """Get store statistics."""
        if not self._loaded:
            return {"loaded": False}

        sqlite_stats = self._sqlite.get_stats() if self._sqlite else {}

        return {
            "loaded": True,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "total_memories": self._sqlite.size if self._sqlite else 0,
            "vector_count": self._faiss.size if self._faiss else 0,
            "by_category": sqlite_stats.get("by_category", {}),
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
            "encryption_enabled": MemoryCrypto.is_enabled(),
        }
