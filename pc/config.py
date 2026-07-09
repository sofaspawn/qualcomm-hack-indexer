"""Centralized configuration for the Lore indexing pipeline.

All tuneable parameters are defined here as module-level constants.
No other module should hardcode these values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File watching
# ---------------------------------------------------------------------------

WATCH_DIRECTORIES: list[Path] = []
"""Directories to monitor for file changes."""

SUPPORTED_EXTENSIONS: set[str] = {"txt", "pdf", "docx"}
"""Lowercase file extensions (without dot) that the pipeline can process."""

DEBOUNCE_SECONDS: float = 2.0
"""Seconds to wait after the last file event before triggering indexing."""

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 1000
"""Maximum number of characters per chunk."""

CHUNK_OVERLAP: int = 200
"""Number of overlapping characters between consecutive chunks."""

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
"""HuggingFace model identifier for the embedding model."""

BATCH_SIZE: int = 64
"""Number of texts to embed in a single batch."""

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

VECTOR_DB_PATH: str = "./lore_db"
"""Path to the LanceDB database directory."""

TABLE_NAME: str = "documents"
"""Name of the LanceDB table storing document chunks."""
