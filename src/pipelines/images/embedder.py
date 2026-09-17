"""OpenCLIP embedding generation for image ingestion."""

from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

from src.pipelines.images.models import ImageIngestionConfig

logger = logging.getLogger(__name__)


class OpenCLIPEmbedder:
    """Computes dense visual embeddings using OpenCLIP."""

    def __init__(self, config: Optional[ImageIngestionConfig] = None) -> None:
        self.config = config or ImageIngestionConfig()
        self.model_name = self.config.model_name
        self.pretrained = self.config.pretrained
        self.device_str = self._resolve_device()

        self._model = None
        self._preprocess = None
        self._torch = None
        self._is_loaded = False

    def _resolve_device(self) -> str:
        """Determines computation device."""
        if self.config.device:
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        """Loads OpenCLIP model and preprocessing transforms lazily."""
        if self._is_loaded:
            return

        try:
            import open_clip  # type: ignore
            import torch

            self._torch = torch

            logger.info(
                "Loading OpenCLIP model '%s' (pretrained='%s') on %s...",
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
            self._is_loaded = True
            logger.info("OpenCLIP model loaded successfully.")

        except ImportError as exc:
            raise ImportError(
                "OpenCLIP or PyTorch is not installed. "
                "Install them using: pip install torch torchvision open-clip-torch"
            ) from exc

    @property
    def model_id(self) -> str:
        """Identifier of the embedding model."""
        return f"{self.model_name}/{self.pretrained}"

    def embed_image(self, image: Image.Image) -> List[float]:
        """Generates a normalized embedding vector for a single PIL image.

        Args:
            image: PIL Image object.

        Returns:
            List of floats representing the embedding vector.
        """
        results = self.embed_batch([image])
        return results[0]

    def embed_batch(self, images: List[Image.Image]) -> List[List[float]]:
        """Generates normalized embeddings for a batch of PIL images.

        Args:
            images: List of PIL Image objects.

        Returns:
            List of embedding vectors.
        """
        if not images:
            return []

        self._ensure_loaded()
        assert self._model is not None
        assert self._preprocess is not None
        assert self._torch is not None

        torch = self._torch

        # Convert images to RGB and apply OpenCLIP preprocess transforms
        rgb_images = [img.convert("RGB") for img in images]
        tensors = [self._preprocess(img) for img in rgb_images]
        batch_tensor = torch.stack(tensors).to(self.device_str)

        with torch.no_grad():
            image_features = self._model.encode_image(batch_tensor)

            if self.config.normalize_embeddings:
                image_features /= image_features.norm(dim=-1, keepdim=True)

            embeddings = image_features.cpu().tolist()

        return embeddings
