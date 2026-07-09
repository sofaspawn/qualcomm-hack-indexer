"""Lore PC indexing engine.

A privacy-first local semantic memory system. This package implements
the complete document indexing pipeline:

    Filesystem → Extract → Normalize → Chunk → Embed → Store

Public API:
    from pc.core.extractor import extract
    from pc.core.normalizer import normalize
    from pc.core.chunker import chunk_document
    from pc.core.embedding_provider import HFEmbeddingProvider
    from pc.core.vector_store import VectorStore
    from pc.core.index_manager import IndexManager
    from pc.core.watcher import FileWatcher, watch
"""
