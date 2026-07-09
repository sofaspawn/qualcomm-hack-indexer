"""LanceDB vector store for the Lore indexing pipeline.

Wraps LanceDB operations behind a clean interface. Stores document
chunks with their embedding vectors, supporting batch insertion,
document-level deletion, and semantic (cosine similarity) search.

Usage:
    from pc.core.vector_store import VectorStore

    store = VectorStore(dimension=384)
    store.insert_chunks(chunks, embeddings)
    results = store.search(query_vector, top_k=5)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa

from pc.config import TABLE_NAME, VECTOR_DB_PATH
from pc.exceptions import VectorStoreError
from pc.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """LanceDB-backed vector store for document chunks.

    Creates or opens a LanceDB database and a single table for storing
    embedded document chunks. Supports batch insert, document-level
    delete/update, and cosine similarity search.

    Args:
        db_path: Path to the LanceDB database directory.
            Defaults to ``config.VECTOR_DB_PATH``.
        table_name: Name of the table to use.
            Defaults to ``config.TABLE_NAME``.
        dimension: Embedding vector dimensionality.
            Must match the embedding provider's output dimension.
    """

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        table_name: str = TABLE_NAME,
        dimension: int = 384,
    ) -> None:
        self._db_path = db_path
        self._table_name = table_name
        self._dimension = dimension

        logger.info("Connecting to LanceDB at '%s'", db_path)
        try:
            self._db = lancedb.connect(db_path)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to connect to LanceDB at '{db_path}': {exc}"
            ) from exc

        self._table = self._ensure_table()
        logger.info("Vector store ready: table='%s', dim=%d", table_name, dimension)

    def _ensure_table(self) -> lancedb.table.Table:
        """Create the table if it does not exist, or open it.

        Returns:
            The LanceDB table handle.
        """
        existing_tables = self._db.table_names()

        if self._table_name in existing_tables:
            logger.info("Opening existing table '%s'", self._table_name)
            return self._db.open_table(self._table_name)

        logger.info("Creating new table '%s'", self._table_name)
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
                pa.field("text", pa.utf8()),
                pa.field("chunk_id", pa.utf8()),
                pa.field("chunk_index", pa.int32()),
                pa.field("filename", pa.utf8()),
                pa.field("page", pa.int32()),
                pa.field("metadata", pa.utf8()),
            ]
        )

        # Create with empty initial data matching the schema.
        empty_table = pa.table(
            {
                "vector": pa.array([], type=pa.list_(pa.float32(), self._dimension)),
                "text": pa.array([], type=pa.utf8()),
                "chunk_id": pa.array([], type=pa.utf8()),
                "chunk_index": pa.array([], type=pa.int32()),
                "filename": pa.array([], type=pa.utf8()),
                "page": pa.array([], type=pa.int32()),
                "metadata": pa.array([], type=pa.utf8()),
            },
            schema=schema,
        )

        return self._db.create_table(self._table_name, data=empty_table)

    # -----------------------------------------------------------------
    # Insert
    # -----------------------------------------------------------------

    def insert_chunks(
        self, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None:
        """Insert a batch of chunks with their embeddings into the store.

        Args:
            chunks: List of ``Chunk`` objects to store.
            embeddings: Numpy array of shape ``(len(chunks), dimension)``.

        Raises:
            VectorStoreError: If the batch sizes mismatch or insertion fails.
        """
        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) does not match "
                f"embedding count ({embeddings.shape[0]})"
            )

        if not chunks:
            logger.warning("insert_chunks called with empty batch — skipping")
            return

        logger.info("Inserting %d chunks into '%s'", len(chunks), self._table_name)
        start_time = time.perf_counter()

        try:
            records = _chunks_to_records(chunks, embeddings)
            self._table.add(records)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to insert chunks: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Insertion completed: %d chunks, %.3fs", len(chunks), elapsed
        )

    # -----------------------------------------------------------------
    # Delete
    # -----------------------------------------------------------------

    def delete_document(self, filename: str) -> None:
        """Delete all chunks associated with a given filename.

        Args:
            filename: The filename to match against stored chunks.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        logger.info("Deleting chunks for '%s'", filename)
        start_time = time.perf_counter()

        try:
            self._table.delete(f"filename = '{filename}'")
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete chunks for '{filename}': {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        logger.info("Deletion completed for '%s', %.3fs", filename, elapsed)

    # -----------------------------------------------------------------
    # Update (delete + reinsert)
    # -----------------------------------------------------------------

    def update_document(
        self, filename: str, chunks: list[Chunk], embeddings: np.ndarray
    ) -> None:
        """Replace all chunks for a document with new versions.

        Deletes existing chunks for ``filename``, then inserts the
        new chunks and embeddings. LanceDB versioning provides
        consistency.

        Args:
            filename: The filename whose chunks should be replaced.
            chunks: New list of ``Chunk`` objects.
            embeddings: Corresponding embedding vectors.

        Raises:
            VectorStoreError: If delete or insert fails.
        """
        logger.info("Updating document '%s' (%d chunks)", filename, len(chunks))
        self.delete_document(filename)
        self.insert_chunks(chunks, embeddings)

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Perform cosine similarity search over stored chunks.

        Args:
            query_vector: 1-D numpy array of shape ``(dimension,)``.
            top_k: Number of most similar results to return.

        Returns:
            List of dictionaries, each containing chunk fields and a
            ``_distance`` score. Lower distance means higher similarity.

        Raises:
            VectorStoreError: If the search operation fails.
        """
        logger.info("Searching for top %d results", top_k)
        start_time = time.perf_counter()

        try:
            results = (
                self._table.search(query_vector.tolist())
                .metric("cosine")
                .limit(top_k)
                .to_list()
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Search failed: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time
        logger.info("Search completed: %d results, %.3fs", len(results), elapsed)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_to_records(
    chunks: list[Chunk], embeddings: np.ndarray
) -> list[dict[str, Any]]:
    """Convert chunks and embeddings to a list of dicts for LanceDB insertion.

    Args:
        chunks: List of ``Chunk`` objects.
        embeddings: Numpy array of shape ``(len(chunks), dim)``.

    Returns:
        List of record dictionaries matching the table schema.
    """
    records: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, embeddings):
        records.append(
            {
                "vector": vector.tolist(),
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "filename": chunk.source_file,
                "page": chunk.page if chunk.page is not None else 0,
                "metadata": json.dumps(chunk.metadata),
            }
        )
    return records
