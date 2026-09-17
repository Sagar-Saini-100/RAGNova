"""OpenCLIP embedding generation for the image ingestion pipeline.

This module wraps the ``open-clip-torch`` library and exposes a clean
:class:`OpenCLIPEmbedder` interface that accepts PIL Images and returns
normalized float embedding vectors.

Default model
-------------
``ViT-B-32`` pretrained on ``laion2b_s34b_b79k`` – a 512-dimensional
visual encoder that balances quality and inference speed.

Extensibility
-------------
Swap any OpenCLIP-supported architecture by changing ``model_name`` /
``pretrained`` fields in :class:`~src.pipelines.images.models.ImageIngestionConfig`.
Text encoding (for future multimodal retrieval) is intentionally kept as a
separate public method :meth:`OpenCLIPEmbedder.embed_text`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from src.pipelines.images.models import ImageIngestionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical embedding model identifier used across the pipeline
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class OpenCLIPEmbedder:
    """Computes dense visual (and text) embeddings using OpenCLIP.

    Parameters
    ----------
    config:
        Pipeline configuration object. If *None* the defaults are used, which
        corresponds to ``ViT-B-32/laion2b_s34b_b79k``.

    Examples
    --------
    >>> from PIL import Image
    >>> embedder = OpenCLIPEmbedder()
    >>> img = Image.open("photo.jpg")
    >>> vec = embedder.embed_image(img)          # List[float], length 512
    >>> text_vec = embedder.embed_text("cat")    # List[float], length 512
    """

    def __init__(self, config: Optional[ImageIngestionConfig] = None) -> None:
        self.config = config or ImageIngestionConfig()
        self.model_name: str = self.config.model_name
        self.pretrained: str = self.config.pretrained
        self.device_str: str = self._resolve_device()

        # Lazy-loaded internals (populated on first call to _ensure_loaded)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._torch = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Selects the best available compute device.

        Priority: explicit config value -> CUDA -> Apple MPS -> CPU.
        """
        if self.config.device:
            return self.config.device

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-loads the OpenCLIP model, preprocessor, and tokenizer.

        Raises
        ------
        ImportError
            When ``open-clip-torch`` or ``torch`` are not installed.
        """
        if self._is_loaded:
            return

        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP or PyTorch is not installed.\n"
                "Install them with:\n"
                "  pip install torch torchvision open-clip-torch"
            ) from exc

        self._torch = torch

        logger.info(
            "Loading OpenCLIP model '%s' (pretrained='%s') on device='%s' ...",
            self.model_name,
            self.pretrained,
            self.device_str,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device_str,
        )
        model.eval()

        self._model = model
        self._preprocess = preprocess
        # Tokenizer is used only for text embedding (future multimodal path)
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._is_loaded = True

        logger.info("OpenCLIP model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Canonical model identifier string (e.g. ``ViT-B-32/laion2b_s34b_b79k``)."""
        return f"{self.model_name}/{self.pretrained}"

    def embed_image(self, image: Image.Image) -> List[float]:
        """Generates a normalized embedding vector for a single PIL image.

        Parameters
        ----------
        image:
            A :class:`PIL.Image.Image` object (any mode; converted to RGB internally).

        Returns
        -------
        List[float]
            512-dimensional normalized float vector.
        """
        return self.embed_batch([image])[0]

    def embed_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """Generates normalized embeddings for a batch of PIL images.

        Batched inference is significantly faster than calling
        :meth:`embed_image` in a loop for large collections.

        Parameters
        ----------
        images:
            List of :class:`PIL.Image.Image` objects.

        Returns
        -------
        List[List[float]]
            One embedding vector per input image.
        """
        if not images:
            return []

        self._ensure_loaded()
        assert self._model is not None
        assert self._preprocess is not None
        assert self._torch is not None

        torch = self._torch

        # Convert all images to RGB and apply OpenCLIP's preprocessing transform
        rgb_images = [img.convert("RGB") for img in images]
        tensors = [self._preprocess(img) for img in rgb_images]
        batch_tensor = torch.stack(tensors).to(self.device_str)

        with torch.no_grad():
            image_features = self._model.encode_image(batch_tensor)

            if self.config.normalize_embeddings:
                # L2-normalize so cosine similarity == dot product
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

        return image_features.cpu().tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generates a normalized text embedding for multimodal retrieval.

        This is the text counterpart to :meth:`embed_image` and shares the
        same embedding space, enabling cross-modal similarity search.

        Parameters
        ----------
        text:
            Raw input string (up to 77 tokens for ViT-B-32).

        Returns
        -------
        List[float]
            512-dimensional normalized float vector.
        """
        return self.embed_text_batch([text])[0]

    def embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized text embeddings for a batch of strings.

        Parameters
        ----------
        texts:
            List of raw input strings.

        Returns
        -------
        List[List[float]]
            One embedding vector per input string.
        """
        if not texts:
            return []

        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        torch = self._torch

        tokens = self._tokenizer(texts).to(self.device_str)  # type: ignore[operator]

        with torch.no_grad():
            text_features = self._model.encode_text(tokens)

            if self.config.normalize_embeddings:
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )

        return text_features.cpu().tolist()
