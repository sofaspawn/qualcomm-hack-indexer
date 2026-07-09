"""Chunk data model for the Lore indexing pipeline.

A Chunk represents a segment of a larger document, produced by the
chunking stage. Each chunk carries enough context to be independently
embedded, stored, and traced back to its source document.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single text chunk derived from a source document.

    Attributes:
        chunk_id: Unique identifier (UUID4 string) for this chunk.
        chunk_index: Zero-based position of this chunk within the document.
        source_file: Filename of the originating document.
        text: The text content of this chunk.
        page: Page number if known (PDF), None otherwise.
        metadata: Inherited and chunk-specific metadata.
    """

    chunk_id: str
    chunk_index: int
    source_file: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
