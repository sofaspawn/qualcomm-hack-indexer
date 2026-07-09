"""Data models for the Lore document extraction layer.

Defines the structured output produced by the extraction pipeline.
The Document dataclass serves as the canonical representation of an
extracted document, carrying both content and metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """Structured representation of an extracted document.

    Produced by the extraction layer, this object carries the full
    extracted text alongside file metadata. Designed to be consumed
    by downstream pipeline stages (chunking, embedding, indexing).

    Attributes:
        path: Absolute resolved path to the source file.
        filename: Name of the file including extension (e.g. "notes.pdf").
        extension: Lowercase file extension without the dot (e.g. "pdf").
        text: The full extracted text content of the document.
        page_count: Number of pages if applicable (PDF), None otherwise.
        char_count: Total character count of the extracted text.
        metadata: Format-specific metadata (PDF info, DOCX core properties, etc.).
    """

    path: Path
    filename: str
    extension: str
    text: str
    page_count: int | None = None
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute char_count from text after initialization."""
        # frozen=True requires object.__setattr__ for post-init assignment.
        object.__setattr__(self, "char_count", len(self.text))
