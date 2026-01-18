"""
Embedding Extractor for product images.

Supports:
- DINOv2 (Small, Base, Large)
- DINOv3 (Small, Base, Large)
- CLIP (ViT-B/16, ViT-B/32, ViT-L/14)
- Custom fine-tuned models

Uses the bb-models package for unified model loading.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Union
from pathlib import Path
import numpy as np
import httpx
from io import BytesIO

# Import from bb-models package
try:
    from bb_models import get_backbone, get_model_config, is_model_supported
    from bb_models.registry import MODEL_CONFIGS
    BB_MODELS_AVAILABLE = True
except ImportError:
    BB_MODELS_AVAILABLE = False
    print("Warning: bb-models package not found, using legacy extractor")

# Note: AutoImageProcessor and AutoModel are imported inside methods
# to avoid issues when bb-models is available but legacy fallback is needed


class EmbeddingExtractor:
    """
    Unified embedding extractor supporting multiple model families.

    Uses bb-models package for model loading when available,
    falls back to legacy DINOv2 loader otherwise.
    """

    def __init__(
        self,
        model_type: str = "dinov2-base",
        model_path: Optional[str] = None,
        checkpoint_url: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the extractor.

        Args:
            model_type: Model identifier (e.g., 'dinov2-base', 'dinov3-base', 'clip-vit-b-16')
            model_path: Local path to custom model weights
            checkpoint_url: URL to download custom model weights
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing Embedding Extractor...")
        print(f"  Model Type: {model_type}")
        print(f"  Device: {self.device}")
        print(f"  bb-models available: {BB_MODELS_AVAILABLE}")

        if BB_MODELS_AVAILABLE and is_model_supported(model_type):
            self._init_with_bb_models(checkpoint_url)
        else:
            self._init_legacy(model_path, checkpoint_url)

        print(f"  Embedding Dim: {self.embedding_dim}")
        print(f"  Model loaded successfully!")

    def _init_with_bb_models(self, checkpoint_url: Optional[str] = None):
        """Initialize using bb-models package."""
        # Get model config
        config = get_model_config(self.model_type)
        self.embedding_dim = config.embedding_dim

        # Load backbone
        self.backbone = get_backbone(
            model_id=self.model_type,
            checkpoint_url=checkpoint_url,
            load_pretrained=True,
        )
        self.backbone.to(self.device)
        self.backbone.eval()

        # Get preprocessing config
        self.preprocess_config = self.backbone.get_preprocessing_config()

        # For batch processing, we need a processor
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(config.hf_model_id)

        self._use_bb_models = True

    def _init_legacy(self, model_path: Optional[str], checkpoint_url: Optional[str]):
        """Legacy initialization for DINOv2 only."""
        from transformers import AutoImageProcessor, AutoModel

        self._use_bb_models = False

        # Determine model
        if self.model_type == "dinov2-large":
            model_name = "facebook/dinov2-large"
            self.embedding_dim = 1024
        else:
            model_name = "facebook/dinov2-base"
            self.embedding_dim = 768

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Load custom weights
        if checkpoint_url:
            self._load_from_url(checkpoint_url)
        elif model_path:
            self._load_from_path(model_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_from_url(self, url: str):
        """Download and load model weights from URL."""
        print(f"  Downloading weights from: {url[:50]}...")
        response = httpx.get(url, timeout=300)
        response.raise_for_status()

        buffer = BytesIO(response.content)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=False)
        print(f"  Custom weights loaded!")

    def _load_from_path(self, path: str):
        """Load model weights from local path."""
        print(f"  Loading weights from: {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=False)
        print(f"  Custom weights loaded!")

    def extract_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Extract embedding from PIL Image.

        Args:
            image: PIL Image

        Returns:
            Normalized embedding vector as numpy array
        """
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract
        with torch.no_grad():
            if self._use_bb_models:
                # Use bb-models backbone
                embedding = self.backbone(inputs["pixel_values"])
            else:
                # Legacy mode
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
                embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def extract_from_url(self, url: str) -> np.ndarray:
        """
        Extract embedding from image URL.

        Args:
            url: Image URL

        Returns:
            Normalized embedding vector as numpy array
        """
        response = httpx.get(url, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return self.extract_from_pil(image)

    def extract_from_path(self, path: Union[str, Path]) -> np.ndarray:
        """
        Extract embedding from local image path.

        Args:
            path: Path to image file

        Returns:
            Normalized embedding vector as numpy array
        """
        image = Image.open(path).convert("RGB")
        return self.extract_from_pil(image)

    def extract_batch_from_urls(
        self,
        urls: list[str],
        batch_size: int = 16,
        progress_callback=None,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Extract embeddings from multiple URLs.

        Args:
            urls: List of image URLs
            batch_size: Number of images to process at once
            progress_callback: Optional callback(processed, total)

        Returns:
            List of (url, embedding) tuples for successful extractions
        """
        results = []
        total = len(urls)

        for i in range(0, total, batch_size):
            batch_urls = urls[i : i + batch_size]
            batch_images = []
            batch_valid_urls = []

            # Download images
            for url in batch_urls:
                try:
                    response = httpx.get(url, timeout=30)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    batch_images.append(image)
                    batch_valid_urls.append(url)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    continue

            if not batch_images:
                continue

            # Process batch
            inputs = self.processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                if self._use_bb_models:
                    # Use bb-models backbone
                    embeddings = self.backbone(inputs["pixel_values"])
                else:
                    # Legacy mode
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            # Collect results
            for url, emb in zip(batch_valid_urls, embeddings):
                results.append((url, emb.cpu().numpy().flatten()))

            # Progress callback
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results


# Backward compatibility alias
DINOv2Extractor = EmbeddingExtractor


# Global singleton
_extractor = None


def get_extractor(
    model_type: str = "dinov2-base",
    model_path: Optional[str] = None,
    checkpoint_url: Optional[str] = None,
) -> EmbeddingExtractor:
    """Get or create extractor singleton."""
    global _extractor

    if _extractor is None:
        _extractor = EmbeddingExtractor(
            model_type=model_type,
            model_path=model_path,
            checkpoint_url=checkpoint_url,
        )
    elif model_type != _extractor.model_type:
        # Recreate if model type changed
        _extractor = EmbeddingExtractor(
            model_type=model_type,
            model_path=model_path,
            checkpoint_url=checkpoint_url,
        )

    return _extractor


def list_supported_models() -> list:
    """List all supported model types."""
    if BB_MODELS_AVAILABLE:
        from bb_models import list_available_models
        return list_available_models()
    return ["dinov2-base", "dinov2-large", "custom"]
