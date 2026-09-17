"""Tesseract OCR wrapper for image ingestion.

This module provides :class:`TesseractOCREngine`, a self-contained wrapper
around the ``pytesseract`` Python binding.  It handles:

- Auto-discovery of the Tesseract binary (PATH, env-var, Windows defaults).
- Optional image pre-processing to improve OCR accuracy on low-resolution or
  low-contrast images.
- Structured output via :class:`~src.pipelines.images.models.OCRResult`.

Extensibility
-------------
Swap the underlying OCR backend by subclassing :class:`TesseractOCREngine`
and overriding :meth:`extract_text`.  The :class:`ImageIngestionPipeline`
accepts any object that implements that single method signature.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.pipelines.images.models import ImageIngestionConfig, OCRResult

logger = logging.getLogger(__name__)


class TesseractOCREngine:
    """Extracts text and confidence metadata from images using Tesseract OCR.

    Parameters
    ----------
    config:
        Pipeline configuration.  Key fields consumed here:
        ``tesseract_cmd``, ``ocr_lang``, ``ocr_config``,
        ``preprocess_for_ocr``.

    Notes
    -----
    The engine degrades gracefully: if ``pytesseract`` is not installed or
    the Tesseract binary cannot be found, every call to :meth:`extract_text`
    returns an empty :class:`~src.pipelines.images.models.OCRResult` with
    ``status="tesseract_unavailable"`` so the rest of the pipeline can
    continue without raising.

    Examples
    --------
    >>> from PIL import Image
    >>> engine = TesseractOCREngine()
    >>> result = engine.extract_text(Image.open("receipt.png"))
    >>> print(result.text)
    >>> print(result.confidence)
    """

    def __init__(self, config: Optional[ImageIngestionConfig] = None) -> None:
        self.config = config or ImageIngestionConfig()
        self._pytesseract = None
        self._is_available: bool = self._init_tesseract()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_tesseract(self) -> bool:
        """Locates and configures the pytesseract binding and Tesseract binary.

        Returns
        -------
        bool
            *True* when both the Python binding and the binary are ready.
        """
        # 1. Import the Python binding
        try:
            import pytesseract  # type: ignore

            self._pytesseract = pytesseract
        except ImportError:
            logger.warning(
                "pytesseract is not installed — OCR extraction will be skipped. "
                "Install it with:  pip install pytesseract"
            )
            return False

        # 2. Resolve the Tesseract binary path
        tess_cmd: Optional[str] = self.config.tesseract_cmd or os.getenv(
            "TESSERACT_CMD"
        )

        if not tess_cmd:
            tess_cmd = shutil.which("tesseract")

        if not tess_cmd:
            # Check common Windows installation directories
            _win_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(
                    r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                ),
            ]
            for candidate in _win_paths:
                if os.path.isfile(candidate):
                    tess_cmd = candidate
                    break

        if tess_cmd:
            self._pytesseract.pytesseract.tesseract_cmd = tess_cmd
            logger.info("Configured Tesseract binary at: %s", tess_cmd)
            return True

        logger.warning(
            "Tesseract executable not found. "
            "Set 'tesseract_cmd' in ImageIngestionConfig or TESSERACT_CMD env var."
        )
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` when both pytesseract and the Tesseract binary are ready."""
        return self._is_available

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Applies image enhancement to improve OCR accuracy.

        Steps (applied only when ``config.preprocess_for_ocr`` is *True*):

        1. Up-scale if the shortest side is below 300 px.
        2. Convert to grayscale.
        3. Boost contrast by 1.8x.
        4. Apply mild sharpening.

        Parameters
        ----------
        image:
            Input PIL Image (any mode).

        Returns
        -------
        PIL.Image.Image
            Pre-processed image ready for Tesseract.
        """
        if not self.config.preprocess_for_ocr:
            return image

        img = image.convert("RGB")

        # Up-scale very small images so Tesseract has enough detail
        w, h = img.size
        if w < 300 or h < 300:
            scale = max(2.0, 600.0 / max(w, h))
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
            )

        # Grayscale → contrast enhancement → sharpening
        gray = ImageOps.grayscale(img)
        enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        return sharpened

    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extracts text and word-level confidence from a PIL Image.

        Parameters
        ----------
        image:
            PIL Image to run OCR on.

        Returns
        -------
        OCRResult
            Contains the extracted ``text``, average ``confidence`` (0–100),
            ``words_count``, and a ``metadata`` dict with status information.
        """
        if not self._is_available or self._pytesseract is None:
            return OCRResult(
                text="",
                confidence=0.0,
                words_count=0,
                metadata={"status": "tesseract_unavailable"},
            )

        try:
            processed = self.preprocess_image(image)

            # Use image_to_data for per-word confidence scores
            data = self._pytesseract.image_to_data(
                processed,
                lang=self.config.ocr_lang,
                config=self.config.ocr_config,
                output_type=self._pytesseract.Output.DICT,
            )

            words: List[str] = []
            confidences: List[float] = []

            for word, conf in zip(data["text"], data["conf"]):
                clean = str(word).strip()
                if not clean:
                    continue
                words.append(clean)
                try:
                    conf_f = float(conf)
                    if conf_f >= 0:
                        confidences.append(conf_f)
                except (ValueError, TypeError):
                    pass

            full_text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=round(avg_conf, 2),
                words_count=len(words),
                metadata={
                    "status": "success",
                    "total_tokens": len(words),
                    "avg_confidence": round(avg_conf, 2),
                },
            )

        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s", exc)
            return OCRResult(
                text="",
                confidence=0.0,
                words_count=0,
                metadata={"status": "error", "error": str(exc)},
            )
