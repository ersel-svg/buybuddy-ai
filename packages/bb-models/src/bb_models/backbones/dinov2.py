"""
DINOv2 backbone implementation.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

from bb_models.base import BaseBackbone
from bb_models.registry import get_model_config


class DINOv2Backbone(BaseBackbone):
    """
    DINOv2 Vision Transformer backbone.

    Supports:
    - dinov2-small (384 dim)
    - dinov2-base (768 dim)
    - dinov2-large (1024 dim)
    """

    def __init__(self, model_id: str = "dinov2-base"):
        super().__init__(model_id)

        config = get_model_config(model_id)
        self._embedding_dim = config.embedding_dim
        self._image_size = config.image_size
        self._hf_model_id = config.hf_model_id

        # Model will be loaded in load_pretrained()
        self.model: Optional[nn.Module] = None
        self.processor: Optional[Any] = None

    def load_pretrained(self, checkpoint_url: Optional[str] = None) -> None:
        """
        Load pretrained weights from HuggingFace or custom checkpoint.

        Args:
            checkpoint_url: Optional URL/path to custom checkpoint.
        """
        print(f"Loading DINOv2 model: {self._hf_model_id}")

        # Load base model from HuggingFace
        self.model = AutoModel.from_pretrained(self._hf_model_id)
        self.processor = AutoImageProcessor.from_pretrained(self._hf_model_id)

        # Load custom checkpoint if provided
        if checkpoint_url is not None:
            self._load_custom_checkpoint(checkpoint_url)

        self._is_loaded = True
        print(f"DINOv2 model loaded: {self.model_id} ({self._embedding_dim}d)")

    def _load_custom_checkpoint(self, checkpoint_url: str) -> None:
        """Load weights from a custom checkpoint."""
        import httpx
        from io import BytesIO

        print(f"Loading custom checkpoint: {checkpoint_url[:50]}...")

        if checkpoint_url.startswith(("http://", "https://")):
            response = httpx.get(checkpoint_url, timeout=300)
            response.raise_for_status()
            buffer = BytesIO(response.content)
            state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
        else:
            state_dict = torch.load(checkpoint_url, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Filter to only backbone weights
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_state[k.replace("backbone.", "")] = v
            elif not k.startswith(("head.", "proj.", "pool.", "dropout.")):
                backbone_state[k] = v

        self.model.load_state_dict(backbone_state, strict=False)
        print("Custom checkpoint loaded successfully")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        # Forward through DINOv2
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)

        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        cls_embedding = F.normalize(cls_embedding, p=2, dim=1)

        return cls_embedding

    def forward_features(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all features including patch tokens.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W)

        Returns:
            Dict with 'cls_token', 'patch_tokens', 'last_hidden_state'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state

        return {
            "cls_token": last_hidden[:, 0, :],
            "patch_tokens": last_hidden[:, 1:, :],
            "last_hidden_state": last_hidden,
        }

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """
        Get parameter groups for LLRD.

        Returns:
            List of parameter lists, from earliest to latest layers.
        """
        if self.model is None:
            return []

        groups = []

        # Embeddings layer
        if hasattr(self.model, "embeddings"):
            groups.append(list(self.model.embeddings.parameters()))

        # Encoder layers
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
            if layers is not None:
                for layer in layers:
                    groups.append(list(layer.parameters()))

        # Layer norm at the end
        if hasattr(self.model, "layernorm"):
            groups.append(list(self.model.layernorm.parameters()))

        return groups

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration from the processor."""
        if self.processor is None:
            return super().get_preprocessing_config()

        return {
            "image_mean": list(self.processor.image_mean),
            "image_std": list(self.processor.image_std),
            "image_size": self._image_size,
        }

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.model is None:
            return 0

        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
            if layers is not None:
                return len(layers)
        return 0
