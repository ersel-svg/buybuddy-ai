"""
RunPod Serverless Handler for Unified Inference Worker.

Handles all model types in a single worker:
- Detection: YOLO, RT-DETR, D-FINE, YOLO-NAS, Grounding DINO
- Classification: ViT, ConvNeXt, EfficientNet, Swin
- Embedding: DINOv2, CLIP, SigLIP

Input format:
{
    "task": "detection" | "classification" | "embedding",
    "model_id": "uuid",
    "model_source": "pretrained" | "trained",
    "model_type": "yolo11n" | "rt-detr" | "vit" | "dinov2" | etc.,
    "checkpoint_url": "https://..." (for trained models),
    "class_mapping": {0: "person", 1: "car", ...},
    "image": "base64_encoded_jpeg",
    "config": {
        # Detection
        "confidence": 0.5,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "input_size": 640,
        "agnostic_nms": false,

        # Classification
        "top_k": 5,
        "threshold": 0.0,

        # Embedding
        "normalize": true,
        "pooling": "cls",
    }
}

Output format:
{
    "success": true,
    "result": {
        # Detection
        "detections": [...],
        "count": 5,
        "image_size": {"width": 1920, "height": 1080},

        # Classification
        "predictions": [...],
        "top_class": "cat",
        "top_confidence": 0.98,

        # Embedding
        "embedding": [0.1, 0.2, ...],
        "embedding_dim": 768,
        "normalized": true,
    },
    "metadata": {
        "task": "detection",
        "model_type": "yolo11n",
        "model_source": "pretrained",
        "inference_time_ms": 250,
        "model_load_time_ms": 50,
        "cached": true,
    }
}
"""

import runpod
import torch
import time
import traceback
import base64
import io
import os
import hashlib
from typing import Any, Optional, Dict, List, Tuple
from PIL import Image
import numpy as np

# Model cache (stays in memory between requests)
MODEL_CACHE: Dict[str, Any] = {}

# Checkpoint download cache
CHECKPOINT_CACHE: Dict[str, str] = {}

# Cache directory
CACHE_DIR = "/tmp/checkpoints"


def setup_cache_dir():
    """Create cache directory if not exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_image_from_base64(image_data: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    try:
        # Remove data URL prefix if present
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except Exception as e:
        raise ValueError(f"Failed to load image from base64: {e}")


def download_checkpoint(url: str) -> str:
    """
    Download checkpoint from URL to local cache.

    Args:
        url: Checkpoint URL

    Returns:
        Local file path
    """
    if url in CHECKPOINT_CACHE:
        if os.path.exists(CHECKPOINT_CACHE[url]):
            return CHECKPOINT_CACHE[url]

    print(f"Downloading checkpoint from {url}")

    import httpx

    # Generate cache filename from URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()

    # Determine extension
    ext = ".pt"
    if url.endswith(".onnx"):
        ext = ".onnx"
    elif url.endswith(".pth"):
        ext = ".pth"

    local_path = os.path.join(CACHE_DIR, f"{url_hash}{ext}")

    # Download if not cached
    if not os.path.exists(local_path):
        with httpx.Client(timeout=300) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.write(response.content)

        print(f"Downloaded checkpoint to {local_path} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
    else:
        print(f"Using cached checkpoint at {local_path}")

    CHECKPOINT_CACHE[url] = local_path
    return local_path


def get_or_load_model(
    task: str,
    model_type: str,
    model_source: str,
    checkpoint_url: Optional[str] = None,
    class_mapping: Optional[Dict] = None,
    num_classes: Optional[int] = None,
    embedding_dim: Optional[int] = None,
) -> Tuple[Any, float]:
    """
    Get model from cache or load it.

    Args:
        task: "detection", "classification", "embedding"
        model_type: Model architecture (e.g., "yolo11n", "vit", "dinov2")
        model_source: "pretrained" or "trained"
        checkpoint_url: URL to checkpoint (for trained models)
        class_mapping: Class ID to name mapping
        num_classes: Number of classes (for classification)
        embedding_dim: Embedding dimension (for embedding models)

    Returns:
        (model, load_time_ms)
    """
    cache_key = f"{task}:{model_source}:{model_type}:{checkpoint_url or 'pretrained'}"

    if cache_key in MODEL_CACHE:
        print(f"Using cached model: {cache_key}")
        return MODEL_CACHE[cache_key], 0.0

    print(f"Loading model: {cache_key}")
    start_time = time.time()

    # Load based on task
    if task == "detection":
        model = load_detection_model(model_type, model_source, checkpoint_url, num_classes)
    elif task == "classification":
        model = load_classification_model(model_type, model_source, checkpoint_url, num_classes)
    elif task == "embedding":
        model = load_embedding_model(model_type, model_source, checkpoint_url)
    elif task == "segmentation":
        model = load_sam_model(model_type, model_source, checkpoint_url)
    else:
        raise ValueError(f"Unknown task: {task}")

    load_time = (time.time() - start_time) * 1000

    MODEL_CACHE[cache_key] = model
    print(f"Model loaded in {load_time:.0f}ms")

    return model, load_time


def load_dfine_model(checkpoint_path: str, num_classes: int = 80, model_size: str = "l") -> Tuple[Any, Any]:
    """
    Load D-FINE model from checkpoint using HuggingFace transformers.

    D-FINE checkpoints contain model state dict with specific keys.
    Returns (model, processor) tuple.
    """
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    print(f"Loading D-FINE model from {checkpoint_path}")

    # D-FINE model names from HuggingFace
    MODEL_NAMES = {
        "s": "ustc-community/dfine-small-coco",
        "small": "ustc-community/dfine-small-coco",
        "m": "ustc-community/dfine-medium-coco",
        "medium": "ustc-community/dfine-medium-coco",
        "l": "ustc-community/dfine-large-coco",
        "large": "ustc-community/dfine-large-coco",
        "x": "ustc-community/dfine-xlarge-coco",
        "xlarge": "ustc-community/dfine-xlarge-coco",
    }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict from our training format
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        elif "ema" in checkpoint:
            if isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
                state_dict = checkpoint["ema"]["module"]
            else:
                state_dict = checkpoint["ema"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Try to get num_classes from checkpoint
        num_classes = checkpoint.get("num_classes", checkpoint.get("config", {}).get("num_classes", num_classes))
    else:
        state_dict = checkpoint

    # Load base model from HuggingFace
    model_name = MODEL_NAMES.get(model_size.lower(), MODEL_NAMES["l"])

    try:
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        processor = AutoImageProcessor.from_pretrained(model_name)

        # Load trained weights
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded D-FINE weights from checkpoint")

    except Exception as e:
        print(f"D-FINE not available, falling back to RT-DETR: {e}")
        # Fallback to RT-DETR
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r101vd",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_rtdetr_model(checkpoint_path: str, model_variant: str = "rtdetr-l", num_classes: int = 80) -> Tuple[Any, Any]:
    """
    Load RT-DETR model from checkpoint using HuggingFace transformers.

    Returns (model, processor) tuple.
    """
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    print(f"Loading RT-DETR model from {checkpoint_path}")

    # RT-DETR model names from HuggingFace
    MODEL_NAMES = {
        "rtdetr-l": "PekingU/rtdetr_r50vd",
        "rtdetr-x": "PekingU/rtdetr_r101vd",
        "rtdetr_r50vd": "PekingU/rtdetr_r50vd",
        "rtdetr_r101vd": "PekingU/rtdetr_r101vd",
    }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Try to get num_classes from checkpoint
        num_classes = checkpoint.get("num_classes", checkpoint.get("config", {}).get("num_classes", num_classes))
    else:
        state_dict = checkpoint

    # Load base model from HuggingFace
    model_name = MODEL_NAMES.get(model_variant.lower(), "PekingU/rtdetr_r50vd")

    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    processor = RTDetrImageProcessor.from_pretrained(model_name)

    # Load trained weights
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded RT-DETR weights from checkpoint")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_grounding_dino_model(model_variant: str = "base") -> Tuple[Any, Any]:
    """
    Load Grounding DINO model for open-vocabulary detection.

    Args:
        model_variant: "tiny", "base", or "large"

    Returns:
        (model, processor) tuple
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    MODEL_NAMES = {
        "tiny": "IDEA-Research/grounding-dino-tiny",
        "base": "IDEA-Research/grounding-dino-base",
        "large": "IDEA-Research/grounding-dino-base",  # large not available yet
        "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    }

    model_name = MODEL_NAMES.get(model_variant.lower(), "IDEA-Research/grounding-dino-base")
    print(f"Loading Grounding DINO: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_owlvit_model(model_variant: str = "base") -> Tuple[Any, Any]:
    """
    Load OWL-ViT/OWL-v2 model for open-vocabulary detection.

    Args:
        model_variant: "base-patch32", "base-patch16", "large-patch14", or owl-v2 variants

    Returns:
        (model, processor) tuple
    """
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    MODEL_NAMES = {
        "base-patch32": "google/owlvit-base-patch32",
        "base-patch16": "google/owlvit-base-patch16",
        "large-patch14": "google/owlvit-large-patch14",
        "owlvit-base-patch32": "google/owlvit-base-patch32",
        "owlvit-base-patch16": "google/owlvit-base-patch16",
        "owlvit-large-patch14": "google/owlvit-large-patch14",
        "owl-vit-base": "google/owlvit-base-patch32",
        "owl-vit-large": "google/owlvit-large-patch14",
    }

    # Try OWL-v2 first if available
    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        OWL_V2_NAMES = {
            "owlv2-base": "google/owlv2-base-patch16-ensemble",
            "owlv2-large": "google/owlv2-large-patch14-ensemble",
            "owl-v2-base": "google/owlv2-base-patch16-ensemble",
            "owl-v2-large": "google/owlv2-large-patch14-ensemble",
        }
        if model_variant.lower() in OWL_V2_NAMES:
            model_name = OWL_V2_NAMES[model_variant.lower()]
            print(f"Loading OWL-v2: {model_name}")
            processor = Owlv2Processor.from_pretrained(model_name)
            model = Owlv2ForObjectDetection.from_pretrained(model_name)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            return model, processor
    except ImportError:
        pass

    model_name = MODEL_NAMES.get(model_variant.lower(), "google/owlvit-base-patch32")
    print(f"Loading OWL-ViT: {model_name}")

    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_yolonas_model(model_variant: str = "l") -> Any:
    """
    Load YOLO-NAS model.

    DISABLED: super-gradients requires onnxruntime==1.13.1 which is incompatible
    with Python 3.11. Use RT-DETR or D-FINE instead.
    """
    raise NotImplementedError(
        "YOLO-NAS is disabled due to dependency conflicts. "
        "Use RT-DETR or D-FINE instead - they have better accuracy anyway."
    )


def load_detection_model(
    model_type: str,
    model_source: str,
    checkpoint_url: Optional[str] = None,
    num_classes: Optional[int] = None,
    text_prompt: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load detection model.

    Returns (model, processor) tuple. Processor is None for YOLO models.
    """
    from ultralytics import YOLO

    processor = None

    if model_source == "pretrained":
        # Pretrained YOLO models
        if model_type.startswith("yolo") and "nas" not in model_type.lower():
            model_path = f"{model_type}.pt"
            print(f"Loading pretrained YOLO: {model_path}")
            model = YOLO(model_path)
        elif model_type.startswith("rtdetr"):
            # Use ultralytics for pretrained RT-DETR
            model_path = f"{model_type}.pt"
            print(f"Loading pretrained RT-DETR: {model_path}")
            from ultralytics import RTDETR
            model = RTDETR(model_path)
        elif "d-fine" in model_type.lower() or "dfine" in model_type.lower():
            # Pretrained D-FINE from HuggingFace
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            MODEL_NAMES = {
                "dfine-s": "ustc-community/dfine-small-coco",
                "dfine-m": "ustc-community/dfine-medium-coco",
                "dfine-l": "ustc-community/dfine-large-coco",
                "dfine-x": "ustc-community/dfine-xlarge-coco",
            }
            model_name = MODEL_NAMES.get(model_type.lower(), "ustc-community/dfine-large-coco")
            model = AutoModelForObjectDetection.from_pretrained(model_name)
            processor = AutoImageProcessor.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.cuda()
        # Grounding DINO - open vocabulary detection
        elif "grounding" in model_type.lower() or "gdino" in model_type.lower():
            variant = model_type.replace("grounding-dino-", "").replace("grounding_dino_", "").replace("gdino-", "")
            if not variant or variant == model_type:
                variant = "base"
            model, processor = load_grounding_dino_model(variant)
        # OWL-ViT / OWL-v2 - open vocabulary detection
        elif "owl" in model_type.lower():
            variant = model_type.replace("owl-vit-", "").replace("owlvit-", "").replace("owl-v2-", "").replace("owlv2-", "")
            if not variant or variant == model_type:
                variant = "base-patch32"
            model, processor = load_owlvit_model(model_type)  # Pass full type for v2 detection
        # YOLO-NAS
        elif "nas" in model_type.lower():
            variant = model_type.replace("yolo-nas-", "").replace("yolonas-", "").replace("yolo_nas_", "")
            if not variant or variant == model_type:
                variant = "l"
            model = load_yolonas_model(variant)
        else:
            raise ValueError(f"Unsupported pretrained detection model: {model_type}")
    else:
        # Trained models
        if not checkpoint_url:
            raise ValueError("Trained model requires checkpoint_url")

        local_path = download_checkpoint(checkpoint_url)

        # Load based on model type
        if "d-fine" in model_type.lower() or "dfine" in model_type.lower():
            # D-FINE model using transformers
            model, processor = load_dfine_model(local_path, num_classes or 80)
        elif "rtdetr" in model_type.lower() or "rt-detr" in model_type.lower():
            # RT-DETR model using transformers
            model, processor = load_rtdetr_model(local_path, model_type, num_classes or 80)
        elif "yolo" in model_type.lower():
            print(f"Loading YOLO variant from {local_path}")
            model = YOLO(local_path)
        else:
            # Try YOLO first, fallback to raw checkpoint
            try:
                print(f"Trying to load as YOLO model from {local_path}")
                model = YOLO(local_path)
            except Exception as e:
                print(f"YOLO load failed: {e}, trying raw checkpoint")
                checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
                model = {"checkpoint": checkpoint, "type": "raw"}

    # Move to GPU if available (for YOLO models without processor)
    if processor is None and hasattr(model, 'to') and torch.cuda.is_available():
        model.to("cuda")

    return model, processor


def load_classification_model(
    model_type: str,
    model_source: str,
    checkpoint_url: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load classification model and processor.

    Returns:
        (model, processor)
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    # Map model types to HuggingFace names
    base_models = {
        "vit": "google/vit-base-patch16-224",
        "vit-tiny": "WinKawaks/vit-tiny-patch16-224",
        "vit-small": "WinKawaks/vit-small-patch16-224",
        "vit-base": "google/vit-base-patch16-224",
        "vit-large": "google/vit-large-patch16-224",
        "convnext": "facebook/convnext-base-224",
        "convnext-tiny": "facebook/convnext-tiny-224",
        "convnext-small": "facebook/convnext-small-224",
        "convnext-base": "facebook/convnext-base-224",
        "convnext-large": "facebook/convnext-large-224",
        "swin": "microsoft/swin-base-patch4-window7-224",
        "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
        "swin-small": "microsoft/swin-small-patch4-window7-224",
        "swin-base": "microsoft/swin-base-patch4-window7-224",
        "efficientnet": "google/efficientnet-b0",
        "efficientnet-b0": "google/efficientnet-b0",
        "efficientnet-b1": "google/efficientnet-b1",
        "efficientnet-b2": "google/efficientnet-b2",
        "efficientnet-b3": "google/efficientnet-b3",
    }

    base_model = base_models.get(model_type.lower(), "google/vit-base-patch16-224")

    if model_source == "pretrained":
        print(f"Loading pretrained classification model: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)
        model = AutoModelForImageClassification.from_pretrained(base_model)
    else:
        # Trained model
        processor = AutoImageProcessor.from_pretrained(base_model)

        if num_classes:
            model = AutoModelForImageClassification.from_pretrained(
                base_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            model = AutoModelForImageClassification.from_pretrained(base_model)

        # Load trained weights
        if checkpoint_url:
            local_path = download_checkpoint(checkpoint_url)
            state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded trained weights from {local_path}")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def load_embedding_model(
    model_type: str,
    model_source: str,
    checkpoint_url: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load embedding model and processor.

    Returns:
        (model, processor)
    """
    from transformers import AutoImageProcessor, AutoModel

    # Map model types to HuggingFace names
    base_models = {
        "dinov2": "facebook/dinov2-base",
        "dinov2-small": "facebook/dinov2-small",
        "dinov2-base": "facebook/dinov2-base",
        "dinov2-large": "facebook/dinov2-large",
        "dinov2-giant": "facebook/dinov2-giant",
        "clip": "openai/clip-vit-base-patch32",
        "clip-vit-b-32": "openai/clip-vit-base-patch32",
        "clip-vit-b-16": "openai/clip-vit-base-patch16",
        "clip-vit-l-14": "openai/clip-vit-large-patch14",
        "siglip": "google/siglip-base-patch16-224",
    }

    base_model = base_models.get(model_type.lower(), "facebook/dinov2-base")

    # Handle CLIP separately
    if "clip" in model_type.lower():
        from transformers import CLIPProcessor, CLIPModel
        print(f"Loading CLIP model: {base_model}")
        processor = CLIPProcessor.from_pretrained(base_model)
        model = CLIPModel.from_pretrained(base_model)
    else:
        print(f"Loading embedding model: {base_model}")
        processor = AutoImageProcessor.from_pretrained(base_model)
        model = AutoModel.from_pretrained(base_model)

    # Load fine-tuned weights if provided
    if model_source == "trained" and checkpoint_url:
        local_path = download_checkpoint(checkpoint_url)
        state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned weights from {local_path}")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def run_grounding_dino_detection(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    text_prompt: str,
) -> Dict[str, Any]:
    """
    Run Grounding DINO open-vocabulary detection.

    Args:
        model: Grounding DINO model
        processor: Grounding DINO processor
        image: PIL Image
        config: Detection config
        text_prompt: Text prompt describing objects to detect (e.g., "person. car. dog.")

    Returns:
        Detection results
    """
    confidence = config.get("confidence", 0.3)  # Lower default for open-vocab
    max_detections = config.get("max_detections", 300)
    box_threshold = config.get("box_threshold", confidence)
    text_threshold = config.get("text_threshold", 0.25)

    width, height = image.size

    # Preprocess
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(height, width)],
    )[0]

    detections = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]  # Text labels

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if i >= max_detections:
            break

        x1, y1, x2, y2 = box

        # Normalize coordinates
        x1_norm = float(x1) / width
        y1_norm = float(y1) / height
        x2_norm = float(x2) / width
        y2_norm = float(y2) / height

        detections.append({
            "id": i,
            "class_name": label,
            "class_id": i,  # No fixed class IDs for open-vocab
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(x1_norm, 4),
                "y1": round(y1_norm, 4),
                "x2": round(x2_norm, 4),
                "y2": round(y2_norm, 4),
            },
            "area": round((x2_norm - x1_norm) * (y2_norm - y1_norm), 6),
        })

    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": width, "height": height},
        "text_prompt": text_prompt,
    }


def run_owlvit_detection(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    text_queries: List[str],
) -> Dict[str, Any]:
    """
    Run OWL-ViT/OWL-v2 open-vocabulary detection.

    Args:
        model: OWL-ViT model
        processor: OWL-ViT processor
        image: PIL Image
        config: Detection config
        text_queries: List of text queries (e.g., ["a photo of a cat", "a photo of a dog"])

    Returns:
        Detection results
    """
    confidence = config.get("confidence", 0.1)  # OWL-ViT needs lower threshold
    max_detections = config.get("max_detections", 300)

    width, height = image.size

    # Preprocess
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([[height, width]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.cuda()

    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence,
        target_sizes=target_sizes,
    )[0]

    detections = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
        if i >= max_detections:
            break

        x1, y1, x2, y2 = box

        # Normalize coordinates
        x1_norm = float(x1) / width
        y1_norm = float(y1) / height
        x2_norm = float(x2) / width
        y2_norm = float(y2) / height

        # Get class name from queries
        label_idx = int(label_idx)
        class_name = text_queries[label_idx] if label_idx < len(text_queries) else f"class_{label_idx}"

        detections.append({
            "id": i,
            "class_name": class_name,
            "class_id": label_idx,
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(x1_norm, 4),
                "y1": round(y1_norm, 4),
                "x2": round(x2_norm, 4),
                "y2": round(y2_norm, 4),
            },
            "area": round((x2_norm - x1_norm) * (y2_norm - y1_norm), 6),
        })

    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": width, "height": height},
        "text_queries": text_queries,
    }


def run_yolonas_detection(
    model: Any,
    image: Image.Image,
    config: Dict[str, Any],
    class_mapping: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run YOLO-NAS detection.

    DISABLED: super-gradients requires onnxruntime==1.13.1 which is incompatible
    with Python 3.11. Use RT-DETR or D-FINE instead.
    """
    raise NotImplementedError(
        "YOLO-NAS is disabled due to dependency conflicts. "
        "Use RT-DETR or D-FINE instead."
    )
    confidence = config.get("confidence", 0.5)
    iou_threshold = config.get("iou_threshold", 0.45)
    max_detections = config.get("max_detections", 300)

    width, height = image.size

    # Convert PIL to numpy for YOLO-NAS
    image_np = np.array(image)

    # Run prediction
    predictions = model.predict(
        image_np,
        conf=confidence,
        iou=iou_threshold,
    )

    detections = []

    # Get prediction results
    pred = predictions[0]  # First image result

    if hasattr(pred, 'prediction'):
        # New API
        bboxes = pred.prediction.bboxes_xyxy
        confidences = pred.prediction.confidence
        labels = pred.prediction.labels
    else:
        # Legacy API
        bboxes = pred.bboxes_xyxy
        confidences = pred.confidence
        labels = pred.labels

    # Get class names
    if class_mapping:
        names = class_mapping
    elif hasattr(pred, 'class_names'):
        names = {i: name for i, name in enumerate(pred.class_names)}
    else:
        names = {}

    for i, (bbox, conf, label) in enumerate(zip(bboxes, confidences, labels)):
        if i >= max_detections:
            break

        x1, y1, x2, y2 = bbox

        # Normalize coordinates
        x1_norm = float(x1) / width
        y1_norm = float(y1) / height
        x2_norm = float(x2) / width
        y2_norm = float(y2) / height

        cls_id = int(label)
        mapping_value = names.get(cls_id) or names.get(str(cls_id))
        if mapping_value is None:
            cls_name = f"class_{cls_id}"
        elif isinstance(mapping_value, dict):
            cls_name = mapping_value.get("name", f"class_{cls_id}")
        else:
            cls_name = str(mapping_value)

        detections.append({
            "id": i,
            "class_name": cls_name,
            "class_id": cls_id,
            "confidence": round(float(conf), 4),
            "bbox": {
                "x1": round(x1_norm, 4),
                "y1": round(y1_norm, 4),
                "x2": round(x2_norm, 4),
                "y2": round(y2_norm, 4),
            },
            "area": round((x2_norm - x1_norm) * (y2_norm - y1_norm), 6),
        })

    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": width, "height": height},
    }


def run_transformers_detection(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    class_mapping: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run detection using HuggingFace transformers models (D-FINE, RT-DETR)."""
    confidence = config.get("confidence", 0.5)
    max_detections = config.get("max_detections", 300)

    width, height = image.size

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process outputs
    target_sizes = torch.tensor([[height, width]])
    if torch.cuda.is_available():
        target_sizes = target_sizes.cuda()

    results = processor.post_process_object_detection(
        outputs,
        threshold=confidence,
        target_sizes=target_sizes,
    )[0]

    detections = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if i >= max_detections:
            break

        x1, y1, x2, y2 = box

        # Normalize coordinates
        x1_norm = float(x1) / width
        y1_norm = float(y1) / height
        x2_norm = float(x2) / width
        y2_norm = float(y2) / height

        cls_id = int(label)
        if class_mapping:
            mapping_value = class_mapping.get(cls_id) or class_mapping.get(str(cls_id))
            if mapping_value is None:
                cls_name = f"class_{cls_id}"
            elif isinstance(mapping_value, dict):
                cls_name = mapping_value.get("name", f"class_{cls_id}")
            else:
                cls_name = str(mapping_value)
        else:
            cls_name = f"class_{cls_id}"

        detections.append({
            "id": i,
            "class_name": cls_name,
            "class_id": cls_id,
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(x1_norm, 4),
                "y1": round(y1_norm, 4),
                "x2": round(x2_norm, 4),
                "y2": round(y2_norm, 4),
            },
            "area": round((x2_norm - x1_norm) * (y2_norm - y1_norm), 6),
        })

    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": width, "height": height},
    }


def run_detection(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    class_mapping: Optional[Dict] = None,
    model_type: str = "yolo",
    text_prompt: Optional[str] = None,
    text_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run detection inference."""
    confidence = config.get("confidence", 0.5)
    iou_threshold = config.get("iou_threshold", 0.45)
    max_detections = config.get("max_detections", 300)
    input_size = config.get("input_size", 640)
    agnostic_nms = config.get("agnostic_nms", False)

    # Handle Grounding DINO (open-vocabulary with text prompt)
    if "grounding" in model_type.lower() or "gdino" in model_type.lower():
        # Grounding DINO requires a text prompt
        prompt = text_prompt or config.get("text_prompt") or config.get("prompt") or "object"
        return run_grounding_dino_detection(model, processor, image, config, prompt)

    # Handle OWL-ViT/OWL-v2 (open-vocabulary with text queries)
    if "owl" in model_type.lower():
        # OWL-ViT requires text queries
        queries = text_queries or config.get("text_queries") or config.get("queries") or ["object"]
        if isinstance(queries, str):
            queries = [q.strip() for q in queries.split(",")]
        return run_owlvit_detection(model, processor, image, config, queries)

    # Handle YOLO-NAS
    if "nas" in model_type.lower():
        return run_yolonas_detection(model, image, config, class_mapping)

    # Handle D-FINE/RT-DETR with transformers processor
    if processor is not None:
        return run_transformers_detection(model, processor, image, config, class_mapping)

    # Handle raw checkpoints (fallback)
    if isinstance(model, dict) and model.get("type") in ["dfine_raw", "raw"]:
        print("Raw checkpoint - cannot run inference without proper model loader")
        return {
            "detections": [],
            "count": 0,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "error": "Model not properly loaded",
        }

    # Standard YOLO/RTDETR inference
    results = model.predict(
        image,
        conf=confidence,
        iou=iou_threshold,
        max_det=max_detections,
        agnostic_nms=agnostic_nms,
        imgsz=input_size,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    detections = []

    if boxes is not None and len(boxes) > 0:
        # Get class names from model or class_mapping
        if class_mapping:
            names = class_mapping
        elif hasattr(model, 'names'):
            names = model.names
        else:
            names = {}

        width, height = image.size

        for i, box in enumerate(boxes):
            # Get box coordinates (xyxy format)
            bbox_raw = box.xyxy[0].cpu().numpy()
            conf_score = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            # Get class name - supports both simple format and nested dict format
            mapping_value = names.get(cls_id) or names.get(str(cls_id))
            if mapping_value is None:
                cls_name = f"class_{cls_id}"
            elif isinstance(mapping_value, dict):
                cls_name = mapping_value.get("name", f"class_{cls_id}")
            else:
                cls_name = str(mapping_value)

            # Normalize coordinates
            x1_norm = float(bbox_raw[0]) / width
            y1_norm = float(bbox_raw[1]) / height
            x2_norm = float(bbox_raw[2]) / width
            y2_norm = float(bbox_raw[3]) / height

            # Calculate area
            area = (x2_norm - x1_norm) * (y2_norm - y1_norm)

            detections.append({
                "id": i,
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": round(conf_score, 4),
                "bbox": {
                    "x1": round(x1_norm, 4),
                    "y1": round(y1_norm, 4),
                    "x2": round(x2_norm, 4),
                    "y2": round(y2_norm, 4),
                },
                "area": round(area, 6),
            })

    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": image.size[0], "height": image.size[1]},
    }


def run_classification(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    class_mapping: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run classification inference."""
    top_k = config.get("top_k", 5)
    threshold = config.get("threshold", 0.0)

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    # Get top-k predictions
    values, indices = torch.topk(probs, min(top_k, len(probs)))

    predictions = []
    for prob, idx in zip(values, indices):
        conf = float(prob.cpu().numpy())
        class_id = int(idx.cpu().numpy())

        if conf < threshold:
            continue

        # Get class name - supports multiple formats:
        # 1. Simple: {0: "class_a", 1: "class_b"}
        # 2. CLS webhook format: {0: {"id": "uuid", "name": "class_a", "color": "#fff"}}
        if class_mapping:
            mapping_value = class_mapping.get(class_id) or class_mapping.get(str(class_id))
            if mapping_value is None:
                class_name = f"class_{class_id}"
            elif isinstance(mapping_value, dict):
                # CLS webhook format - extract name from nested dict
                class_name = mapping_value.get("name", f"class_{class_id}")
            else:
                # Simple string format
                class_name = str(mapping_value)
        elif hasattr(model.config, 'id2label'):
            class_name = model.config.id2label.get(class_id, f"class_{class_id}")
        else:
            class_name = f"class_{class_id}"

        predictions.append({
            "class_name": class_name,
            "class_id": class_id,
            "confidence": round(conf, 4),
        })

    # Get top prediction
    if predictions:
        top_class = predictions[0]["class_name"]
        top_confidence = predictions[0]["confidence"]
    else:
        top_class = "unknown"
        top_confidence = 0.0

    return {
        "predictions": predictions,
        "top_class": top_class,
        "top_confidence": top_confidence,
    }


def run_embedding(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract image embedding."""
    normalize = config.get("normalize", True)
    pooling = config.get("pooling", "cls")

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

        # Extract embedding based on model type
        if hasattr(outputs, "image_embeds"):
            # CLIP-style
            embedding = outputs.image_embeds[0]
        elif hasattr(outputs, "last_hidden_state"):
            # ViT/DINOv2-style
            hidden = outputs.last_hidden_state

            if pooling == "cls":
                embedding = hidden[0, 0]  # CLS token
            elif pooling == "mean":
                embedding = hidden[0].mean(dim=0)  # Mean pool
            elif pooling == "gem":
                # GeM pooling
                p = 3.0
                embedding = (hidden[0].clamp(min=1e-6).pow(p).mean(dim=0)).pow(1.0 / p)
            else:
                embedding = hidden[0, 0]  # Default to CLS
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output[0]
        else:
            # Fallback
            embedding = outputs[0][0, 0]

    # Convert to numpy
    embedding_np = embedding.cpu().numpy()

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm

    return {
        "embedding": embedding_np.tolist(),
        "embedding_dim": len(embedding_np),
        "normalized": normalize,
    }


# =============================================================================
# SEGMENTATION (SAM, SAM2)
# =============================================================================

def load_sam_model(
    model_type: str,
    model_source: str = "pretrained",
    checkpoint_url: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load SAM/SAM2 model and processor.

    Args:
        model_type: Model variant (sam2-tiny, sam2-base, sam2-large, etc.)
        model_source: "pretrained" or "trained"
        checkpoint_url: URL to checkpoint (for trained models)

    Returns:
        (model, processor)
    """
    from transformers import AutoProcessor, AutoModelForMaskGeneration

    # Map model types to HuggingFace names
    model_map = {
        # SAM2 variants
        "sam2-tiny": "facebook/sam2-hiera-tiny",
        "sam2-small": "facebook/sam2-hiera-small",
        "sam2-base": "facebook/sam2-hiera-base-plus",
        "sam2-large": "facebook/sam2-hiera-large",
        # SAM 2.1 variants
        "sam2.1-tiny": "facebook/sam2.1-hiera-tiny",
        "sam2.1-small": "facebook/sam2.1-hiera-small",
        "sam2.1-base": "facebook/sam2.1-hiera-base-plus",
        "sam2.1-large": "facebook/sam2.1-hiera-large",
    }

    model_name = model_map.get(model_type.lower(), "facebook/sam2-hiera-base-plus")
    print(f"Loading SAM model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMaskGeneration.from_pretrained(model_name)

    # Load fine-tuned weights if provided
    if model_source == "trained" and checkpoint_url:
        local_path = download_checkpoint(checkpoint_url)
        state_dict = torch.load(local_path, map_location="cpu", weights_only=False)

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned SAM weights from {local_path}")

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor


def run_segmentation(
    model: Any,
    processor: Any,
    image: Image.Image,
    config: Dict[str, Any],
    input_boxes: Optional[List[List[float]]] = None,
    input_points: Optional[List[List[float]]] = None,
    input_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run SAM segmentation inference.

    Args:
        model: SAM model
        processor: SAM processor
        image: PIL Image
        config: Segmentation config
        input_boxes: List of bounding boxes [[x1,y1,x2,y2], ...] in pixel coordinates
        input_points: List of point coordinates [[x, y], ...]
        input_labels: List of point labels (1=foreground, 0=background)

    Returns:
        Segmentation results with masks
    """
    multimask_output = config.get("multimask_output", True)
    return_logits = config.get("return_logits", False)

    width, height = image.size

    # Prepare inputs based on prompts
    inputs = processor(
        image,
        input_boxes=input_boxes if input_boxes else None,
        input_points=input_points if input_points else None,
        input_labels=input_labels if input_labels else None,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    # Process outputs
    masks = outputs.pred_masks.cpu().numpy()
    iou_scores = outputs.iou_scores.cpu().numpy() if hasattr(outputs, 'iou_scores') else None

    # Convert masks to list format
    result_masks = []
    for i, mask_set in enumerate(masks[0]):  # First batch
        if multimask_output:
            # Multiple masks per prompt - select best by IoU
            best_idx = 0
            if iou_scores is not None:
                best_idx = int(np.argmax(iou_scores[0][i] if len(iou_scores[0]) > i else iou_scores[0]))

            mask = mask_set[best_idx] if len(mask_set.shape) > 2 else mask_set
            score = float(iou_scores[0][i][best_idx]) if iou_scores is not None and len(iou_scores[0]) > i else 1.0
        else:
            mask = mask_set
            score = float(iou_scores[0][i]) if iou_scores is not None else 1.0

        # Threshold mask
        binary_mask = (mask > 0).astype(np.uint8)

        # Calculate mask area
        area = int(np.sum(binary_mask))
        area_ratio = area / (width * height)

        # Encode mask as RLE for compact storage
        rle = _mask_to_rle(binary_mask)

        result_masks.append({
            "id": i,
            "score": round(score, 4),
            "area": area,
            "area_ratio": round(area_ratio, 6),
            "rle": rle,
            "bbox": _mask_to_bbox(binary_mask, width, height),
        })

    return {
        "masks": result_masks,
        "count": len(result_masks),
        "image_size": {"width": width, "height": height},
    }


def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to RLE encoding."""
    pixels = mask.flatten()
    runs = []
    run_start = 0
    run_length = 0

    for i, pixel in enumerate(pixels):
        if i == 0:
            run_start = 0
            run_length = 1
        elif pixel == pixels[i - 1]:
            run_length += 1
        else:
            runs.append(run_length)
            run_start = i
            run_length = 1

    runs.append(run_length)

    return {
        "counts": runs,
        "size": list(mask.shape),
    }


def _mask_to_bbox(mask: np.ndarray, img_width: int, img_height: int) -> Dict[str, float]:
    """Convert binary mask to normalized bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return {
        "x1": round(x_min / img_width, 4),
        "y1": round(y_min / img_height, 4),
        "x2": round((x_max + 1) / img_width, 4),
        "y2": round((y_max + 1) / img_height, 4),
    }


def handler(job: dict) -> dict:
    """
    Main handler for unified inference worker.

    Args:
        job: RunPod job dict with 'input' key

    Returns:
        Result dict with success/error
    """
    job_input = job.get("input", {})

    try:
        # Validate input
        task = job_input.get("task")
        if not task:
            return {"success": False, "error": "Missing required field: task"}

        if task not in ["detection", "classification", "embedding", "segmentation"]:
            return {"success": False, "error": f"Invalid task: {task}"}

        model_id = job_input.get("model_id")
        model_source = job_input.get("model_source", "pretrained")
        model_type = job_input.get("model_type")
        checkpoint_url = job_input.get("checkpoint_url")
        class_mapping = job_input.get("class_mapping")
        num_classes = job_input.get("num_classes")
        embedding_dim = job_input.get("embedding_dim")
        image_data = job_input.get("image")
        config = job_input.get("config", {})

        # Open-vocabulary detection parameters
        text_prompt = job_input.get("text_prompt") or config.get("text_prompt")  # For Grounding DINO
        text_queries = job_input.get("text_queries") or config.get("text_queries")  # For OWL-ViT

        if not model_type:
            return {"success": False, "error": "Missing required field: model_type"}

        if not image_data:
            return {"success": False, "error": "Missing required field: image"}

        # Load image
        print(f"Loading image from base64...")
        image = load_image_from_base64(image_data)
        print(f"Image loaded: {image.size}")

        # Load model
        print(f"Loading model: task={task}, type={model_type}, source={model_source}")
        model_result = get_or_load_model(
            task=task,
            model_type=model_type,
            model_source=model_source,
            checkpoint_url=checkpoint_url,
            class_mapping=class_mapping,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

        # Handle different return types from load functions
        # All loaders now return (model, processor) or ((model, processor), load_time) tuple
        load_time = model_result[1]
        model_data = model_result[0]

        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, processor = model_data
        else:
            model = model_data
            processor = None

        # Run inference
        print(f"Running {task} inference...")
        start_time = time.time()

        if task == "detection":
            result = run_detection(
                model, processor, image, config, class_mapping, model_type,
                text_prompt=text_prompt, text_queries=text_queries
            )
        elif task == "classification":
            result = run_classification(model, processor, image, config, class_mapping)
        elif task == "embedding":
            result = run_embedding(model, processor, image, config)
        elif task == "segmentation":
            # Get segmentation prompts from input
            input_boxes = job_input.get("input_boxes")  # [[x1,y1,x2,y2], ...]
            input_points = job_input.get("input_points")  # [[x, y], ...]
            input_labels = job_input.get("input_labels")  # [1, 0, 1, ...]
            result = run_segmentation(
                model, processor, image, config,
                input_boxes=input_boxes,
                input_points=input_points,
                input_labels=input_labels,
            )
        else:
            return {"success": False, "error": f"Unknown task: {task}"}

        inference_time = (time.time() - start_time) * 1000

        print(f"Inference complete in {inference_time:.0f}ms")

        return {
            "success": True,
            "result": result,
            "metadata": {
                "task": task,
                "model_type": model_type,
                "model_source": model_source,
                "model_id": model_id,
                "inference_time_ms": round(inference_time, 2),
                "model_load_time_ms": round(load_time, 2),
                "cached": load_time == 0,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(error_trace)

        return {
            "success": False,
            "error": error_msg,
            "traceback": error_trace,
        }


# Initialize cache directory on startup
setup_cache_dir()

# Preload models if specified via environment
# (optional, for reducing cold start time)
# if os.getenv("PRELOAD_MODELS"):
#     print("Preloading models...")
#     # Add preload logic here

# Start RunPod serverless
if __name__ == "__main__":
    print("Starting unified inference worker...")
    runpod.serverless.start({"handler": handler})
