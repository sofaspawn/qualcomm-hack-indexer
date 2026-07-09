"""Indexing orchestration layer for the Lore pipeline.

The ``IndexManager`` is the single coordination point that drives documents
through the full pipeline: extract → normalize → chunk → embed → store.
No other module is allowed to orchestrate multiple pipeline stages.

Usage:
    from pc.core.embedding_provider import HFEmbeddingProvider
    from pc.core.vector_store import VectorStore
    from pc.core.index_manager import IndexManager

    provider = HFEmbeddingProvider()
    store = VectorStore(dimension=provider.dimension)
    manager = IndexManager(embedding_provider=provider, vector_store=store)

    manager.index_file("paper.pdf")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pc.core.chunker import chunk_document
from pc.core.extractor import extract
from pc.core.normalizer import normalize
from pc.core.embedding_provider import EmbeddingProvider
from pc.core.vector_store import VectorStore
from pc.config import CHUNK_OVERLAP, CHUNK_SIZE
from pc.exceptions import IndexingError
from pc.models import Document

logger = logging.getLogger(__name__)


class IndexManager:
    """Orchestrates the full document indexing pipeline.

    Coordinates extraction, normalization, chunking, embedding, and
    storage. Receives its dependencies via constructor injection so
    the embedding backend can be swapped without changes here.

    Args:
        embedding_provider: An ``EmbeddingProvider`` implementation.
        vector_store: A ``VectorStore`` instance for persistence.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between chunks.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_file(self, path: str | Path) -> None:
        """Run the full indexing pipeline on a single file.

        Pipeline: extract → normalize → chunk → embed → store.

        Args:
            path: Path to the document file.

        Raises:
            IndexingError: If any pipeline stage fails.
        """
        resolved = Path(path).resolve()
        logger.info("Indexing started: '%s'", resolved)
        start_time = time.perf_counter()

        try:
            # 1. Extract
            document = extract(resolved)

            # 2. Normalize
            normalized_text = normalize(document.text)

            # Create a new Document with normalized text.
            normalized_doc = Document(
                path=document.path,
                filename=document.filename,
                extension=document.extension,
                text=normalized_text,
                page_count=document.page_count,
                metadata=document.metadata,
            )

            # 3. Chunk
            chunks = chunk_document(
                normalized_doc,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )

            if not chunks:
                logger.warning(
                    "No chunks produced for '%s' — skipping embedding/storage",
                    resolved,
                )
                return

            # 4. Embed
            texts = [c.text for c in chunks]
            embeddings = self._embedding_provider.embed(texts)

            # 5. Store
            self._vector_store.insert_chunks(chunks, embeddings)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Indexing failed for '%s' after %.3fs: %s",
                resolved,
                elapsed,
                exc,
            )
            raise IndexingError(
                f"Indexing failed for '{resolved}': {exc}",
                path=str(resolved),
            ) from exc

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Indexing completed: '%s' — %d chunks, %.3fs",
            resolved,
            len(chunks),
            elapsed,
        )

    def reindex_file(self, path: str | Path) -> None:
        """Delete existing data for a file and re-index it.

        Equivalent to ``delete_document`` followed by ``index_file``.

        Args:
            path: Path to the document file.

        Raises:
            IndexingError: If deletion or re-indexing fails.
        """
        resolved = Path(path).resolve()
        logger.info("Re-indexing: '%s'", resolved)

        try:
            self.delete_document(resolved)
            self.index_file(resolved)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(
                f"Re-indexing failed for '{resolved}': {exc}",
                path=str(resolved),
            ) from exc

    def delete_document(self, path: str | Path) -> None:
        """Remove all indexed data for a document.

        Args:
            path: Path to the document file. Only the filename is used
                for matching against stored chunks.

        Raises:
            IndexingError: If deletion fails.
        """
        resolved = Path(path).resolve()
        filename = resolved.name
        logger.info("Deleting document: '%s'", filename)

        try:
            self._vector_store.delete_document(filename)
        except Exception as exc:
            raise IndexingError(
                f"Deletion failed for '{filename}': {exc}",
                path=str(resolved),
            ) from exc
