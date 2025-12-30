"""
Lazy-loaded sentence-transformers embeddings.

The model (~85MB) is loaded once and reused for all operations.
All operations are async-safe via thread pool execution.
"""

import asyncio
from typing import List, Optional

import numpy as np
from loguru import logger


class EmbeddingService:
    """
    Singleton embedding service using sentence-transformers.

    The model is loaded lazily on first use and cached globally.
    First run downloads the model (~90MB), subsequent runs use cache.

    Usage:
        service = EmbeddingService()  # Uses default model
        embedding = await service.embed("some text")
        embeddings = await service.embed_batch(["text1", "text2"])
    """

    _model = None
    _dimension: Optional[int] = None
    _model_name: Optional[str] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Sentence-transformers model name.
                       Default is all-MiniLM-L6-v2 (fast, 384 dimensions).
        """
        self._requested_model = model_name

    @property
    def model(self):
        """Load model on first use with progress logging."""
        if (
            EmbeddingService._model is None
            or EmbeddingService._model_name != self._requested_model
        ):
            logger.info(f"Loading embedding model '{self._requested_model}'...")
            logger.info("(First run may download ~90MB model, cached for future use)")

            from sentence_transformers import SentenceTransformer

            EmbeddingService._model = SentenceTransformer(self._requested_model)
            EmbeddingService._model_name = self._requested_model
            EmbeddingService._dimension = EmbeddingService._model.get_sentence_embedding_dimension()

            logger.info(
                f"Embedding model loaded: {EmbeddingService._dimension} dimensions"
            )

        return EmbeddingService._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension (loads model if needed)."""
        if EmbeddingService._dimension is None:
            _ = self.model  # Trigger load
        return EmbeddingService._dimension  # type: ignore

    async def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Returns normalized embedding vector.
        Runs in thread pool since sentence-transformers is synchronous.
        """
        model = self.model
        embedding = await asyncio.to_thread(
            model.encode, text, normalize_embeddings=True
        )
        return embedding

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts in a batch (more efficient).

        Returns array of shape (n_texts, dimension).
        """
        if not texts:
            return np.array([])

        model = self.model
        embeddings = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings

    def embed_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding (for non-async contexts)."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous batch embedding (for non-async contexts)."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if the model is already loaded."""
        return cls._model is not None

    @classmethod
    def get_dimension(cls) -> int:
        """Get the embedding dimension (loads model if needed)."""
        if cls._dimension is None:
            # Create temp instance to load
            service = cls()
            _ = service.model
        return cls._dimension  # type: ignore
