"""Image Ingestion Pipeline: OCR + OpenCLIP embeddings producing one Chunk per image."""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from PIL import Image

from src.models.chunk import Chunk
from src.pipelines.images.embedder import OpenCLIPEmbedder
from src.pipelines.images.models import ImageIngestionConfig, ImageMetadata, OCRResult
from src.pipelines.images.ocr import TesseractOCREngine

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
)


class ImageIngestionPipeline:
    """Ingests images via Tesseract OCR and OpenCLIP embeddings into Chunks."""

    def __init__(
        self,
        config: Optional[ImageIngestionConfig] = None,
        ocr_engine: Optional[TesseractOCREngine] = None,
        embedder: Optional[OpenCLIPEmbedder] = None,
    ) -> None:
        self.config = config or ImageIngestionConfig()
        self.ocr_engine = ocr_engine or (
            TesseractOCREngine(self.config) if self.config.ocr_enabled else None
        )
        self.embedder = embedder or (
            OpenCLIPEmbedder(self.config) if self.config.embedding_enabled else None
        )

    def _load_image(
        self, image_input: Union[str, Path, bytes, BinaryIO, Image.Image]
    ) -> tuple[Image.Image, ImageMetadata]:
        """Loads PIL Image and extracts initial file metadata."""
        source_path: Optional[str] = None
        raw_bytes: Optional[bytes] = None

        if isinstance(image_input, (str, Path)):
            path_obj = Path(image_input).expanduser().resolve()
            if not path_obj.is_file():
                raise FileNotFoundError(f"Image file not found: {path_obj}")
            source_path = str(path_obj)
            raw_bytes = path_obj.read_bytes()
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif hasattr(image_input, "read"):
            raw_bytes = image_input.read()
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            # Compute bytes if possible
            buf = io.BytesIO()
            fmt = pil_img.format or "PNG"
            try:
                pil_img.save(buf, format=fmt)
                raw_bytes = buf.getvalue()
            except Exception:
                raw_bytes = None
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Ensure image is in RGB format for downstream models
        fmt = pil_img.format
        width, height = pil_img.size
        mode = pil_img.mode

        sha256 = (
            hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
        )
        file_size = len(raw_bytes) if raw_bytes is not None else None

        metadata = ImageMetadata(
            source_path=source_path,
            width=width,
            height=height,
            format=fmt,
            mode=mode,
            file_size_bytes=file_size,
            sha256=sha256,
        )

        return pil_img, metadata

    def ingest_image(
        self,
        image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Processes a single image and produces exactly one Chunk.

        Args:
            image_input: File path, raw bytes, file-like object, or PIL Image.
            extra_metadata: Optional dictionary of additional metadata to attach.

        Returns:
            A Chunk with modality="image" and embedding_model="ViT-B-32/laion2b_s34b_b79k".
        """
        pil_img, img_metadata = self._load_image(image_input)

        # 1. OCR text extraction
        ocr_result = OCRResult()
        if self.ocr_engine and self.config.ocr_enabled:
            ocr_result = self.ocr_engine.extract_text(pil_img)

        # 2. OpenCLIP visual embedding
        embedding: Optional[List[float]] = None
        if self.embedder and self.config.embedding_enabled:
            embedding = self.embedder.embed_image(pil_img)

        # 3. Assemble metadata
        combined_metadata: Dict[str, Any] = {
            **img_metadata.to_dict(),
            "ocr": {
                "confidence": ocr_result.confidence,
                "words_count": ocr_result.words_count,
                **ocr_result.metadata,
            },
        }
        if extra_metadata:
            combined_metadata.update(extra_metadata)

        # 4. Construct unique chunk identifier
        if img_metadata.sha256:
            chunk_id = f"img_{img_metadata.sha256[:16]}"
        elif img_metadata.source_path:
            clean_name = Path(img_metadata.source_path).stem
            chunk_id = f"img_{clean_name}_{os.urandom(4).hex()}"
        else:
            chunk_id = f"img_{os.urandom(8).hex()}"

        # 5. Produce single Chunk
        chunk = Chunk(
            id=chunk_id,
            content=ocr_result.text,
            modality="image",
            embedding=embedding,
            embedding_model=self.config.embedding_model_id,
            metadata=combined_metadata,
        )

        return chunk

    def ingest_batch(
        self,
        image_inputs: List[Union[str, Path, bytes, BinaryIO, Image.Image]],
        extra_metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Chunk]:
        """Processes a batch of images and produces one Chunk per image.

        Args:
            image_inputs: List of image paths, bytes, or PIL Images.
            extra_metadata_list: Optional list of metadata dicts corresponding to inputs.

        Returns:
            List of Chunks, one per input image.
        """
        if not image_inputs:
            return []

        loaded: List[tuple[Image.Image, ImageMetadata]] = []
        for inp in image_inputs:
            loaded.append(self._load_image(inp))

        images = [img for img, _ in loaded]

        # Batch OCR
        ocr_results: List[OCRResult] = []
        for img in images:
            if self.ocr_engine and self.config.ocr_enabled:
                ocr_results.append(self.ocr_engine.extract_text(img))
            else:
                ocr_results.append(OCRResult())

        # Batch OpenCLIP embeddings
        embeddings: List[Optional[List[float]]]
        if self.embedder and self.config.embedding_enabled:
            embeddings = [
                emb for emb in self.embedder.embed_batch(images)  # type: ignore[misc]
            ]
        else:
            embeddings = [None] * len(images)

        chunks: List[Chunk] = []
        for i, ((_, img_meta), ocr_res, emb) in enumerate(
            zip(loaded, ocr_results, embeddings)
        ):
            meta: Dict[str, Any] = {
                **img_meta.to_dict(),
                "ocr": {
                    "confidence": ocr_res.confidence,
                    "words_count": ocr_res.words_count,
                    **ocr_res.metadata,
                },
            }
            if extra_metadata_list and i < len(extra_metadata_list):
                meta.update(extra_metadata_list[i])

            chunk_id = (
                f"img_{img_meta.sha256[:16]}"
                if img_meta.sha256
                else f"img_{os.urandom(8).hex()}"
            )

            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=ocr_res.text,
                    modality="image",
                    embedding=emb,
                    embedding_model=self.config.embedding_model_id,
                    metadata=meta,
                )
            )

        return chunks

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS,
    ) -> List[Chunk]:
        """Discovers and ingests all images found in a directory.

        Args:
            directory_path: Root directory to search.
            recursive: Whether to search subdirectories.
            extensions: Tuple of file extensions to include.

        Returns:
            List of generated Chunks.
        """
        dir_obj = Path(directory_path).expanduser().resolve()
        if not dir_obj.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_obj}")

        pattern = "**/*" if recursive else "*"
        image_files = [
            f
            for f in dir_obj.glob(pattern)
            if f.is_file() and f.suffix.lower() in extensions
        ]

        logger.info("Found %d images in %s", len(image_files), dir_obj)
        return self.ingest_batch(image_files)  # type: ignore[arg-type]


def ingest_image(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    config: Optional[ImageIngestionConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:
    """Convenience function to ingest a single image into a Chunk."""
    pipeline = ImageIngestionPipeline(config=config)
    return pipeline.ingest_image(image_input, extra_metadata=extra_metadata)
