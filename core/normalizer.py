"""Text normalization stage for the Lore indexing pipeline.

Cleans and normalizes raw extracted text without altering semantic content.
Produces consistent, well-formatted text suitable for chunking and embedding.

Usage:
    from pc.core.normalizer import normalize

    clean_text = normalize(raw_text)
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata

from pc.exceptions import NormalizationError

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Normalize text for downstream processing.

    Applies the following transformations in order:
        1. Unicode NFC normalization
        2. Line ending unification (``\\r\\n`` and ``\\r`` → ``\\n``)
        3. Collapse runs of 3+ blank lines to exactly 2 newlines
        4. Strip leading/trailing whitespace from each line
        5. Collapse multiple spaces/tabs within lines to a single space
        6. Strip leading/trailing whitespace from the entire text

    Semantic content (words, sentence boundaries, paragraph breaks) is
    preserved. Only formatting noise is removed.

    Args:
        text: Raw text to normalize.

    Returns:
        Cleaned and normalized text.

    Raises:
        NormalizationError: If normalization fails unexpectedly.
    """
    logger.info("Normalization started (%d chars)", len(text))
    start_time = time.perf_counter()

    try:
        result = _apply_normalization(text)
    except Exception as exc:
        raise NormalizationError(
            f"Text normalization failed: {exc}"
        ) from exc

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Normalization completed — %d → %d chars, %.3fs",
        len(text),
        len(result),
        elapsed,
    )

    return result


def _apply_normalization(text: str) -> str:
    """Apply all normalization steps sequentially.

    Args:
        text: Raw text input.

    Returns:
        Normalized text.
    """
    # 1. Unicode NFC normalization — compose characters canonically.
    text = unicodedata.normalize("NFC", text)

    # 2. Unify line endings → \n.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse 3+ consecutive newlines to exactly 2 (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4 & 5. Per-line cleanup: strip edges and collapse interior whitespace.
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 6. Strip entire text.
    text = text.strip()

    return text
