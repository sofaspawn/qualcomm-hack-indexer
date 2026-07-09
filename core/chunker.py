"""Recursive text chunking stage for the Lore indexing pipeline.

Splits normalized text into overlapping chunks using a hierarchical
separator strategy (paragraph → line → sentence → word → character).
Each chunk is assigned a UUID and carries provenance metadata.

Usage:
    from pc.core.chunker import chunk_document

    chunks = chunk_document(document, chunk_size=1000, chunk_overlap=200)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pc.config import CHUNK_OVERLAP, CHUNK_SIZE
from pc.exceptions import ChunkingError
from pc.models import Chunk, Document

logger = logging.getLogger(__name__)

# Separator hierarchy: try to split on the largest semantic boundary first.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


def chunk_document(
    document: Document,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping text chunks.

    Uses a recursive character text splitting strategy that tries
    to preserve semantic boundaries (paragraphs, lines, sentences)
    whenever possible.

    Args:
        document: The extracted and normalized document.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of ``Chunk`` objects with sequential indices and UUIDs.

    Raises:
        ChunkingError: If chunking fails unexpectedly.
    """
    logger.info(
        "Chunking started: '%s' (%d chars, size=%d, overlap=%d)",
        document.filename,
        len(document.text),
        chunk_size,
        chunk_overlap,
    )
    start_time = time.perf_counter()

    try:
        text_chunks = _recursive_split(
            document.text, chunk_size, chunk_overlap, _SEPARATORS
        )
    except Exception as exc:
        raise ChunkingError(
            f"Chunking failed for '{document.filename}': {exc}",
            path=str(document.path),
        ) from exc

    base_metadata: dict[str, Any] = {
        **document.metadata,
        "extension": document.extension,
    }

    chunks: list[Chunk] = []
    for idx, text in enumerate(text_chunks):
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=idx,
            source_file=document.filename,
            text=text,
            page=_estimate_page(idx, len(text_chunks), document.page_count),
            metadata=base_metadata,
        )
        chunks.append(chunk)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Chunking completed: '%s' → %d chunks, %.3fs",
        document.filename,
        len(chunks),
        elapsed,
    )

    return chunks


# ---------------------------------------------------------------------------
# Recursive splitting logic
# ---------------------------------------------------------------------------


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using a hierarchy of separators.

    Tries the first separator. If any resulting segment is still too
    long, recursively splits it with the next separator in the list.
    The final fallback (empty string) splits character-by-character.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters to overlap between chunks.
        separators: Ordered list of separator strings to try.

    Returns:
        List of text chunks, each at most ``chunk_size`` characters.
    """
    if not text:
        return []

    # If the text already fits, return it directly.
    if len(text) <= chunk_size:
        return [text]

    # Pick the best separator: the first one that actually appears in text.
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining_separators = []
            break
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    # Split text on the chosen separator.
    if separator:
        splits = text.split(separator)
    else:
        # Character-level fallback.
        splits = list(text)

    # Merge splits into chunks that respect chunk_size.
    chunks = _merge_splits(splits, separator, chunk_size, chunk_overlap)

    # Recursively handle any chunks that are still too long.
    if remaining_separators:
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = _recursive_split(
                    chunk, chunk_size, chunk_overlap, remaining_separators
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks

    return chunks


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge small splits into chunks up to chunk_size, with overlap.

    Args:
        splits: Text fragments from splitting on a separator.
        separator: The separator string used (re-joined between fragments).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of merged text chunks.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for split in splits:
        split_len = len(split)
        # Length if we add this split to current (including separator).
        projected_length = (
            current_length + split_len + (len(separator) if current_parts else 0)
        )

        if projected_length > chunk_size and current_parts:
            # Emit current chunk.
            chunk_text = separator.join(current_parts)
            chunks.append(chunk_text)

            # Build overlap: keep trailing parts that fit within overlap budget.
            overlap_parts: list[str] = []
            overlap_length = 0
            for part in reversed(current_parts):
                part_contribution = len(part) + (
                    len(separator) if overlap_parts else 0
                )
                if overlap_length + part_contribution > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_length += part_contribution

            current_parts = overlap_parts
            current_length = (
                sum(len(p) for p in current_parts)
                + len(separator) * max(0, len(current_parts) - 1)
            )

        current_parts.append(split)
        current_length += split_len + (
            len(separator) if len(current_parts) > 1 else 0
        )

    # Emit the final chunk.
    if current_parts:
        chunk_text = separator.join(current_parts)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


def _estimate_page(
    chunk_index: int, total_chunks: int, page_count: int | None
) -> int | None:
    """Estimate the page number for a chunk based on linear distribution.

    Args:
        chunk_index: Zero-based index of the chunk.
        total_chunks: Total number of chunks in the document.
        page_count: Total pages in the source document (None if unknown).

    Returns:
        Estimated 1-based page number, or None if page_count is unknown.
    """
    if page_count is None or total_chunks == 0:
        return None
    # Linear estimate: distribute chunks evenly across pages.
    return int(chunk_index / total_chunks * page_count) + 1
