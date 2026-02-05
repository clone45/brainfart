"""
Lazy-loaded embeddings using fastembed (ONNX-based).

Uses fastembed instead of sentence-transformers for ~4x smaller install size.
The model (~90MB) is loaded once and reused for all operations.
All operations are async-safe via thread pool execution.
"""

import asyncio
from typing import List, Optional

import numpy as np
from loguru import logger


# Model name mapping: short name -> fastembed full name
MODEL_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}

# Known dimensions for common models
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


class EmbeddingService:
    """
    Singleton embedding service using fastembed (ONNX runtime).

    The model is loaded lazily on first use and cached globally.
    First run downloads the model (~90MB), subsequent runs use cache.

    Uses ONNX runtime instead of PyTorch for ~4x smaller install size
    while maintaining the same embedding quality.

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
            model_name: Model name. Can be a short alias (e.g., "all-MiniLM-L6-v2")
                       or full fastembed name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
                       Default is all-MiniLM-L6-v2 (fast, 384 dimensions).
        """
        # Resolve alias to full name
        self._requested_model = MODEL_ALIASES.get(model_name, model_name)

    @property
    def model(self):
        """Load model on first use with progress logging."""
        if (
            EmbeddingService._model is None
            or EmbeddingService._model_name != self._requested_model
        ):
            logger.info(f"Loading embedding model '{self._requested_model}'...")
            logger.info("(First run may download ~90MB model, cached for future use)")

            from fastembed import TextEmbedding

            EmbeddingService._model = TextEmbedding(model_name=self._requested_model)
            EmbeddingService._model_name = self._requested_model
            EmbeddingService._dimension = MODEL_DIMENSIONS.get(self._requested_model, 384)

            logger.info(
                f"Embedding model loaded: {EmbeddingService._dimension} dimensions (fastembed/ONNX)"
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
        Runs in thread pool since fastembed is synchronous.
        """
        model = self.model
        # fastembed returns a generator, convert to numpy array
        embedding = await asyncio.to_thread(
            lambda: np.array(list(model.embed([text]))[0], dtype=np.float32)
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
        # fastembed returns a generator, convert to numpy array
        embeddings = await asyncio.to_thread(
            lambda: np.array(list(model.embed(texts)), dtype=np.float32)
        )
        return embeddings

    def embed_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding (for non-async contexts)."""
        return np.array(list(self.model.embed([text]))[0], dtype=np.float32)

    def embed_batch_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous batch embedding (for non-async contexts)."""
        if not texts:
            return np.array([])
        return np.array(list(self.model.embed(texts)), dtype=np.float32)

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
