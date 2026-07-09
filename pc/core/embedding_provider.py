"""Embedding provider abstraction for the Lore indexing pipeline.

Defines the ``EmbeddingProvider`` interface and a concrete implementation
backed by HuggingFace Sentence Transformers. The embedding backend can be
swapped (e.g. to an ONNX Runtime provider for Qualcomm hardware) without
any changes to downstream consumers.

Usage:
    from pc.core.embedding_provider import HFEmbeddingProvider

    provider = HFEmbeddingProvider()
    vectors = provider.embed(["hello world", "semantic search"])
    print(vectors.shape)  # (2, 384)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from pc.config import BATCH_SIZE, MODEL_NAME
from pc.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers.

    All embedding implementations must subclass this and implement
    ``embed`` and ``dimension``. No other module in the pipeline
    should depend on a concrete implementation class.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape ``(len(texts), dimension)`` with
            L2-normalized embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension (e.g. 384 for bge-small-en-v1.5).
        """


class HFEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using HuggingFace Sentence Transformers.

    Loads the model once at construction time and reuses it for all
    subsequent embed calls. ``SentenceTransformer`` is only imported
    and instantiated within this class — no other module should
    reference it.

    Args:
        model_name: HuggingFace model identifier.
            Defaults to ``config.MODEL_NAME``.
        batch_size: Number of texts per encoding batch.
            Defaults to ``config.BATCH_SIZE``.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

        logger.info("Loading embedding model: '%s'", model_name)
        start_time = time.perf_counter()

        try:
            # Import here to isolate the dependency.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded: '%s' (dim=%d, %.2fs)",
            model_name,
            self._dimension,
            elapsed,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the loaded model.

        Returns:
            Integer dimension (384 for bge-small-en-v1.5).
        """
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape ``(len(texts), dimension)`` with
            L2-normalized vectors suitable for cosine similarity.

        Raises:
            EmbeddingError: If encoding fails.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        logger.info("Embedding %d texts (batch_size=%d)", len(texts), self._batch_size)
        start_time = time.perf_counter()

        try:
            embeddings: np.ndarray = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Embedding completed: %d texts → shape %s, %.3fs",
            len(texts),
            embeddings.shape,
            elapsed,
        )

        return embeddings
