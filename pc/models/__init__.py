"""Lore data models package.

Re-exports:
    Document — structured representation of an extracted document.
    Chunk    — a text segment derived from a document.
"""

from pc.models.chunk import Chunk
from pc.models.document import Document

__all__ = ["Chunk", "Document"]
