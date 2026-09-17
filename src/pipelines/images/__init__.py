"""Image ingestion pipeline package.

Public surface
--------------
The three canonical modules required by the pipeline spec:

- ``ocr``       → :class:`TesseractOCREngine` (Tesseract OCR wrapper)
- ``embedding`` → :class:`OpenCLIPEmbedder`   (OpenCLIP embedding function)
- ``ingest``    → :class:`ImageIngestionPipeline` + :class:`ChunkStore`
                  (orchestrator that creates and stores Chunks)

Quick start
-----------
>>> from src.pipelines.images import ImageIngestionPipeline
>>> pipeline = ImageIngestionPipeline()
>>> chunks = pipeline.ingest_directory("./my_images")
"""

from src.models.chunk import Chunk

# ── canonical pipeline modules ────────────────────────────────────────────
from src.pipelines.images.embedding import OpenCLIPEmbedder  # embedding.py
from src.pipelines.images.ingest import (                     # ingest.py
    ChunkStore,
    ImageIngestionPipeline,
    ingest_directory,
    ingest_image,
)
from src.pipelines.images.models import (
    ImageIngestionConfig,
    ImageMetadata,
    OCRResult,
)
from src.pipelines.images.ocr import TesseractOCREngine       # ocr.py

__all__ = [
    # Data model
    "Chunk",
    # Core modules
    "TesseractOCREngine",
    "OpenCLIPEmbedder",
    "ImageIngestionPipeline",
    "ChunkStore",
    # Config & result types
    "ImageIngestionConfig",
    "OCRResult",
    "ImageMetadata",
    # Convenience helpers
    "ingest_image",
    "ingest_directory",
]
