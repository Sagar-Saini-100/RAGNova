"""Data models and configuration for the image ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ImageIngestionConfig:
    """Configuration for ImageIngestionPipeline."""

    # OCR Settings
    ocr_enabled: bool = True
    tesseract_cmd: Optional[str] = field(
        default_factory=lambda: os.getenv("TESSERACT_CMD")
    )
    ocr_lang: str = "eng"
    ocr_config: str = "--psm 3"
    preprocess_for_ocr: bool = True

    # OpenCLIP Embedding Settings
    embedding_enabled: bool = True
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: Optional[str] = None  # None = auto-detect ('cuda', 'mps', 'cpu')
    normalize_embeddings: bool = True

    @property
    def embedding_model_id(self) -> str:
        """Returns standard embedding model identifier string."""
        return f"{self.model_name}/{self.pretrained}"


@dataclass
class OCRResult:
    """Result of OCR text extraction."""

    text: str = ""
    confidence: float = 0.0
    words_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageMetadata:
    """Extracted metadata for an input image."""

    source_path: Optional[str] = None
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    mode: str = "RGB"
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    exif: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "exif": self.exif,
        }


__all__ = [
    "ImageIngestionConfig",
    "OCRResult",
    "ImageMetadata",
]
