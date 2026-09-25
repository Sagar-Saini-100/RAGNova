"""
Write images into ChromaDB's image_index — the write path Chapter 10 noted
was missing ("image_index has no write path into ChromaDB yet").

Chapter 8's ImageIngestionPipeline already does the hard work (load, OCR,
CLIP-embed) but stores results in an in-memory ChunkStore that disappears
when the process ends. This module is the Chapter 12 integration glue:
run that pipeline, then hand its chunks to the same add_chunks() the text
side uses (src/core/vector_store.py), into the image collection.
"""

from __future__ import annotations

from pathlib import Path

from src.core.schemas import Chunk
from src.core.text_normalize import normalize_text
from src.core.vector_store import add_chunks, get_client, get_image_collection
from src.pipelines.images.ingest import SUPPORTED_EXTENSIONS, ImageIngestionPipeline
from src.pipelines.images.models import ImageChunk


def index_image_files(paths: list[str | Path], client=None, pipeline=None) -> int:
    """Ingest the given image files and upsert them into image_index.
    Returns how many were indexed. Unreadable files are skipped by the
    pipeline itself (Chapter 8's per-file try/except), not here.

    `pipeline` defaults to a real ImageIngestionPipeline (Tesseract + CLIP);
    tests pass one built with a fake embedder so they run without
    downloading CLIP's weights.
    """
    if not paths:
        return 0
    client = client or get_client()
    collection = get_image_collection(client)
    pipeline = pipeline or ImageIngestionPipeline()

    image_chunks = pipeline.ingest_batch([Path(p) for p in paths])
    # An image with no vector can't be searched at all — only possible
    # when the pipeline was built with embedding_enabled=False.
    image_chunks = [c for c in image_chunks if c.embedding is not None]
    if not image_chunks:
        return 0

    chunks = [_to_index_chunk(c) for c in image_chunks]
    add_chunks(collection, chunks, [c.embedding for c in image_chunks])
    return len(chunks)


def index_images_directory(directory: str | Path, client=None, pipeline=None) -> int:
    """Index every supported image directly under `directory` (flat, not
    recursive — same reason index_documents_directory() gives)."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    paths = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return index_image_files(paths, client=client, pipeline=pipeline)


def _to_index_chunk(image_chunk: ImageChunk) -> Chunk:
    """Turn Chapter 8's ImageChunk into a plain, contract-shaped Chunk.

    Two real fixes happen here, both found while wiring this up:

    1. `source`. ImageIngestionPipeline stores the *absolute* path
       (`Path(...).resolve()`), e.g. "/home/alice/RAGNova/data/images/x.png".
       That bakes one person's checkout folder into the citation — the
       exact bug Chapter 7's _relative_to_cwd() already fixed for
       documents. Made relative to the project root here, the same way.

    2. OCR text. Tesseract output is full of stray newlines and double
       spaces. normalize_text() is the shared cleanup Chapter 6 made for
       exactly this ("Chapter 8's OCR output ... need[s] the identical
       cleanup" — docs/GLOSSARY.md, Normalization (of text)).
    """
    return Chunk(
        chunk_id=image_chunk.chunk_id,
        source=_relative_to_cwd(image_chunk.source),
        modality="image",
        text=normalize_text(image_chunk.text or ""),
        embedding_model=image_chunk.embedding_model,
    )


def _relative_to_cwd(source: str) -> str:
    try:
        return Path(source).resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return source
