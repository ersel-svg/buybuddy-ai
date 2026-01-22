"""
CLS Annotation Worker - CLIP & SigLIP Zero-Shot Classification

Supports:
- Single image classification
- Batch classification
- Multiple models: CLIP, SigLIP
"""

import os
import torch
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional
import time

# Global model cache
MODELS = {}

# Custom session with User-Agent
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(model_name: str = "ViT-B-32"):
    """Load CLIP model using open_clip."""
    import open_clip
    
    cache_key = f"clip_{model_name}"
    if cache_key in MODELS:
        return MODELS[cache_key]
    
    print(f"[CLIP] Loading model: {model_name}")
    device = get_device()
    
    # Map friendly names to open_clip model names
    model_map = {
        "ViT-B-32": ("ViT-B-32", "openai"),
        "ViT-B-16": ("ViT-B-16", "openai"),
        "ViT-L-14": ("ViT-L-14", "openai"),
        "ViT-H-14": ("ViT-H-14", "laion2b_s32b_b79k"),
    }
    
    if model_name in model_map:
        name, pretrained = model_map[model_name]
    else:
        name, pretrained = model_name, "openai"
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, 
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(name)
    
    model.eval()
    MODELS[cache_key] = (model, preprocess, tokenizer)
    print(f"[CLIP] Model loaded on {device}")
    
    return model, preprocess, tokenizer

def load_siglip_model(model_name: str = "google/siglip-base-patch16-224"):
    """Load SigLIP model using transformers."""
    from transformers import AutoProcessor, AutoModel
    
    cache_key = f"siglip_{model_name}"
    if cache_key in MODELS:
        return MODELS[cache_key]
    
    print(f"[SigLIP] Loading model: {model_name}")
    device = get_device()
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    MODELS[cache_key] = (model, processor)
    print(f"[SigLIP] Model loaded on {device}")
    
    return model, processor

def download_image(url: str) -> Optional[Image.Image]:
    """Download image from URL."""
    try:
        response = SESSION.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")
        return None

def classify_with_clip(
    images: List[Image.Image],
    classes: List[str],
    model_name: str = "ViT-B-32",
    top_k: int = 3,
) -> List[List[Dict[str, Any]]]:
    """Classify images using CLIP."""
    import open_clip
    
    model, preprocess, tokenizer = load_clip_model(model_name)
    device = get_device()
    
    # Prepare text prompts
    text_prompts = [f"a photo of a {c}" for c in classes]
    text_tokens = tokenizer(text_prompts).to(device)
    
    results = []
    with torch.no_grad():
        # Encode text once
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Batch process images for better throughput
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(img) for img in batch_images]).to(device)
            
            img_features = model.encode_image(batch_tensors)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity for batch
            similarity = img_features @ text_features.T
            probs = torch.softmax(similarity * 100, dim=-1)  # Temperature scaling
            
            for j in range(len(batch_images)):
                # Get top-k for this image
                top_probs, top_indices = probs[j].topk(min(top_k, len(classes)))
                
                predictions = []
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    predictions.append({
                        "class": classes[idx],
                        "confidence": float(prob)
                    })
                
                results.append(predictions)
    
    return results

def classify_with_siglip(
    images: List[Image.Image],
    classes: List[str],
    model_name: str = "google/siglip-base-patch16-224",
    top_k: int = 3,
) -> List[List[Dict[str, Any]]]:
    """Classify images using SigLIP."""
    model, processor = load_siglip_model(model_name)
    device = get_device()
    
    # Prepare text prompts
    text_prompts = [f"a photo of a {c}" for c in classes]
    
    results = []
    with torch.no_grad():
        for img in images:
            # Process inputs
            inputs = processor(
                text=text_prompts,
                images=img,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Get logits and convert to probs
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1).squeeze(0)
            
            # Get top-k
            top_probs, top_indices = probs.topk(min(top_k, len(classes)))
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    "class": classes[idx],
                    "confidence": float(prob)
                })
            
            results.append(predictions)
    
    return results

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for classification.
    
    Input:
    {
        "task": "classify",
        "model": "clip" or "siglip",
        "model_variant": "ViT-B-32" (optional),
        "images": [{"id": "xxx", "url": "https://..."}],
        "classes": ["class1", "class2", ...],
        "top_k": 3,
        "threshold": 0.1
    }
    
    Output:
    {
        "results": [
            {
                "id": "xxx",
                "predictions": [
                    {"class": "class1", "confidence": 0.85}
                ]
            }
        ]
    }
    """
    start_time = time.time()
    job_input = job.get("input", job)
    
    # Parse input
    task = job_input.get("task", "classify")
    model_type = job_input.get("model", "clip").lower()
    model_variant = job_input.get("model_variant", None)
    images_data = job_input.get("images", [])
    classes = job_input.get("classes", [])
    top_k = job_input.get("top_k", 3)
    threshold = job_input.get("threshold", 0.0)
    
    if not images_data:
        return {"error": "No images provided"}
    
    if not classes:
        return {"error": "No classes provided"}
    
    print(f"[Handler] Task: {task}, Model: {model_type}, Images: {len(images_data)}, Classes: {len(classes)}")
    
    # Download images
    images = []
    image_ids = []
    for img_data in images_data:
        img_id = img_data.get("id", "unknown")
        img_url = img_data.get("url")
        
        if not img_url:
            continue
            
        img = download_image(img_url)
        if img:
            images.append(img)
            image_ids.append(img_id)
    
    if not images:
        return {"error": "No valid images downloaded"}
    
    download_time = time.time() - start_time
    print(f"[Handler] Downloaded {len(images)} images in {download_time:.2f}s")
    
    # Run classification
    inference_start = time.time()
    if model_type == "clip":
        variant = model_variant or "ViT-B-32"
        all_predictions = classify_with_clip(images, classes, variant, top_k)
    elif model_type == "siglip":
        variant = model_variant or "google/siglip-base-patch16-224"
        all_predictions = classify_with_siglip(images, classes, variant, top_k)
    else:
        return {"error": f"Unknown model type: {model_type}"}
    
    inference_time = time.time() - inference_start
    
    # Format results
    results = []
    for img_id, predictions in zip(image_ids, all_predictions):
        # Apply threshold filter
        filtered = [p for p in predictions if p["confidence"] >= threshold]
        results.append({
            "id": img_id,
            "predictions": filtered
        })
    
    total_time = time.time() - start_time
    throughput = len(images) / inference_time if inference_time > 0 else 0
    
    print(f"[Handler] Inference: {inference_time:.2f}s, Throughput: {throughput:.1f} img/s")
    print(f"[Handler] Total: {total_time:.2f}s")
    
    return {
        "results": results,
        "model": model_type,
        "model_variant": model_variant or variant,
        "elapsed_seconds": round(total_time, 2),
        "inference_seconds": round(inference_time, 2),
        "images_processed": len(images),
        "throughput_img_per_sec": round(throughput, 1)
    }

# RunPod serverless entry point
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
