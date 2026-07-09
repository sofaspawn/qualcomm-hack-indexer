"""Lore core pipeline modules.

Submodules:
    extractor          — document text extraction (TXT, PDF, DOCX)
    normalizer         — text normalization
    chunker            — recursive text chunking
    embedding_provider — embedding generation (abstract + HuggingFace)
    vector_store       — LanceDB vector storage
    index_manager      — pipeline orchestration
    watcher            — filesystem monitoring
"""
