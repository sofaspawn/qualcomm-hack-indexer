"""Custom exceptions for the Lore indexing pipeline.

Provides a structured exception hierarchy covering every pipeline stage.
All exceptions inherit from ``LoreError``, allowing callers to catch
any pipeline error with a single handler or target specific stages.

Hierarchy:
    LoreError
    ├── ExtractorError
    │   ├── UnsupportedFileTypeError
    │   ├── CorruptedDocumentError
    │   └── FileNotReadableError
    ├── NormalizationError
    ├── ChunkingError
    ├── EmbeddingError
    ├── VectorStoreError
    └── IndexingError
"""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class LoreError(Exception):
    """Root exception for all Lore pipeline errors.

    Args:
        message: Human-readable description of the error.
        path: Optional file path associated with the error.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Extraction stage
# ---------------------------------------------------------------------------


class ExtractorError(LoreError):
    """Base exception for all extraction-related errors.

    All custom exceptions in the extraction layer inherit from this class,
    allowing callers to catch any extraction error with a single handler.
    """


class UnsupportedFileTypeError(ExtractorError):
    """Raised when extraction is attempted on a file with an unsupported extension.

    Args:
        extension: The unsupported file extension (e.g. ".png").
        path: The file path that triggered the error.
    """

    def __init__(self, extension: str, path: str | None = None) -> None:
        self.extension = extension
        message = f"Unsupported file type: '{extension}'"
        if path:
            message += f" (file: {path})"
        super().__init__(message, path=path)


class CorruptedDocumentError(ExtractorError):
    """Raised when a document cannot be parsed due to corruption or invalid format.

    This wraps underlying library exceptions (e.g. PyMuPDF, python-docx) to
    provide a consistent interface to the caller.

    Args:
        message: Description of the corruption or parse failure.
        path: The file path that triggered the error.
        original_error: The original exception from the underlying library.
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.original_error = original_error
        super().__init__(message, path=path)


class FileNotReadableError(ExtractorError):
    """Raised when a file does not exist or cannot be read.

    Args:
        message: Description of the access failure.
        path: The file path that triggered the error.
    """


# ---------------------------------------------------------------------------
# Normalization stage
# ---------------------------------------------------------------------------


class NormalizationError(LoreError):
    """Raised when text normalization fails.

    Args:
        message: Description of the normalization failure.
        path: Optional source file path for context.
    """


# ---------------------------------------------------------------------------
# Chunking stage
# ---------------------------------------------------------------------------


class ChunkingError(LoreError):
    """Raised when text chunking fails.

    Args:
        message: Description of the chunking failure.
        path: Optional source file path for context.
    """


# ---------------------------------------------------------------------------
# Embedding stage
# ---------------------------------------------------------------------------


class EmbeddingError(LoreError):
    """Raised when embedding generation fails.

    Args:
        message: Description of the embedding failure.
        path: Optional source file path for context.
    """


# ---------------------------------------------------------------------------
# Vector store stage
# ---------------------------------------------------------------------------


class VectorStoreError(LoreError):
    """Raised when a vector store operation fails.

    Args:
        message: Description of the vector store failure.
        path: Optional source file path for context.
    """


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class IndexingError(LoreError):
    """Raised when the indexing orchestration pipeline fails.

    Wraps errors that occur during the full index_file / reindex_file flow
    but are not covered by a more specific stage exception.

    Args:
        message: Description of the indexing failure.
        path: Optional source file path for context.
    """
