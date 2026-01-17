"""
CLIP backbone implementation.

CLIP (Contrastive Language-Image Pre-training) by OpenAI.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from bb_models.base import BaseBackbone
from bb_models.registry import get_model_config


class CLIPBackbone(BaseBackbone):
    """
    CLIP Vision Transformer backbone.

    Supports:
    - clip-vit-b-16 (512 dim)
    - clip-vit-b-32 (512 dim)
    - clip-vit-l-14 (1024 dim)

    Note: Only uses the vision encoder, not the text encoder.
    """

    def __init__(self, model_id: str = "clip-vit-b-16"):
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
        print(f"Loading CLIP model: {self._hf_model_id}")

        # Load full CLIP model from HuggingFace
        clip_model = CLIPModel.from_pretrained(self._hf_model_id)
        self.processor = CLIPProcessor.from_pretrained(self._hf_model_id)

        # Only keep the vision model
        self.model = clip_model.vision_model

        # Load custom checkpoint if provided
        if checkpoint_url is not None:
            self._load_custom_checkpoint(checkpoint_url)

        self._is_loaded = True
        print(f"CLIP model loaded: {self.model_id} ({self._embedding_dim}d)")

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

        # Filter to only vision model weights
        vision_state = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                vision_state[k.replace("backbone.", "")] = v
            elif k.startswith("vision_model."):
                vision_state[k.replace("vision_model.", "")] = v
            elif not k.startswith(("head.", "proj.", "pool.", "dropout.", "text_")):
                vision_state[k] = v

        self.model.load_state_dict(vision_state, strict=False)
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

        # Forward through CLIP vision model
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)

        # CLIP uses pooler_output which is the CLS token projected
        # But we'll use last_hidden_state[:, 0, :] for consistency
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            embedding = outputs.last_hidden_state[:, 0, :]

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_features(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all features including patch tokens.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W)

        Returns:
            Dict with 'cls_token', 'patch_tokens', 'last_hidden_state', 'pooler_output'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")

        outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state

        result = {
            "cls_token": last_hidden[:, 0, :],
            "patch_tokens": last_hidden[:, 1:, :],
            "last_hidden_state": last_hidden,
        }

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            result["pooler_output"] = outputs.pooler_output

        return result

    def get_layer_groups(self) -> List[List[nn.Parameter]]:
        """
        Get parameter groups for LLRD.

        Returns:
            List of parameter lists, from earliest to latest layers.
        """
        if self.model is None:
            return []

        groups = []

        # Embeddings
        if hasattr(self.model, "embeddings"):
            groups.append(list(self.model.embeddings.parameters()))

        # Encoder layers
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layers", None)
            if layers is not None:
                for layer in layers:
                    groups.append(list(layer.parameters()))

        # Pre-layer norm and post-layer norm
        if hasattr(self.model, "pre_layrnorm"):
            groups.append(list(self.model.pre_layrnorm.parameters()))
        if hasattr(self.model, "post_layernorm"):
            groups.append(list(self.model.post_layernorm.parameters()))

        return groups

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration from the processor."""
        if self.processor is None:
            return super().get_preprocessing_config()

        # CLIP processor has image_processor attribute
        image_processor = self.processor.image_processor

        return {
            "image_mean": list(image_processor.image_mean),
            "image_std": list(image_processor.image_std),
            "image_size": self._image_size,
        }

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.model is None:
            return 0

        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            layers = getattr(encoder, "layers", None)
            if layers is not None:
                return len(layers)
        return 0
