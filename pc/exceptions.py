"""Custom exceptions for the Lore document extraction layer.

Provides a structured exception hierarchy for handling extraction failures,
unsupported file types, and corrupted documents. All exceptions inherit from
a common base class to allow broad or narrow exception handling by callers.
"""


class ExtractorError(Exception):
    """Base exception for all extraction-related errors.

    All custom exceptions in the extraction layer inherit from this class,
    allowing callers to catch any extraction error with a single handler.

    Args:
        message: Human-readable description of the error.
        path: Optional file path associated with the error.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


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

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message, path=path)
