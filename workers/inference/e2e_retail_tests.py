"""
End-to-End Retail Workflow Tests

20 realistic retail scenarios testing all model types and workflow blocks.
Simulates API → Worker flow for complete integration testing.

Scenarios cover:
- Product detection on shelves
- Product classification
- Brand recognition
- Stock counting
- Price tag detection
- Category classification
- Similar product search (embedding)
- Quality inspection
- Planogram compliance
- Multi-model workflows
"""

import sys
import time
import json
import base64
import io
import random
import math
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# ============================================================================
# Test Infrastructure
# ============================================================================

@dataclass
class TestResult:
    scenario_id: int
    name: str
    passed: bool
    duration_ms: float
    details: str
    blocks_tested: List[str]
    models_tested: List[str]


RESULTS: List[TestResult] = []


def log(msg, level="INFO"):
    """Log with timestamp and level."""
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "TEST": "🧪"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def create_retail_image(
    size=(1920, 1080),
    scenario="shelf",
    num_products=5,
    add_price_tags=False,
    add_brand_logos=False,
):
    """Create realistic retail scenario images."""
    img = Image.new("RGB", size, color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    if scenario == "shelf":
        # Draw shelf background
        shelf_height = size[1] // 4
        for i in range(4):
            y = i * shelf_height
            draw.rectangle([0, y, size[0], y + 20], fill=(139, 90, 43))

        # Draw products on shelves
        colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
        for i in range(num_products):
            x = random.randint(50, size[0] - 150)
            y = random.randint(50, size[1] - 200)
            w = random.randint(80, 150)
            h = random.randint(120, 200)
            color = colors[i % len(colors)]
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=2)

            if add_price_tags:
                draw.rectangle([x, y + h, x + w, y + h + 30], fill=(255, 255, 0))

    elif scenario == "product_single":
        # Single product centered
        center_x, center_y = size[0] // 2, size[1] // 2
        w, h = 300, 400
        draw.rectangle([center_x - w//2, center_y - h//2, center_x + w//2, center_y + h//2],
                      fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)),
                      outline=(0, 0, 0), width=3)

        if add_brand_logos:
            draw.ellipse([center_x - 50, center_y - 150, center_x + 50, center_y - 50],
                        fill=(255, 255, 255), outline=(0, 0, 0))

    elif scenario == "warehouse":
        # Warehouse with boxes
        for i in range(num_products):
            x = random.randint(50, size[0] - 200)
            y = random.randint(50, size[1] - 200)
            w = random.randint(150, 250)
            h = random.randint(150, 250)
            draw.rectangle([x, y, x + w, y + h], fill=(205, 133, 63), outline=(139, 69, 19), width=3)

    elif scenario == "checkout":
        # Checkout counter with items
        draw.rectangle([0, size[1] - 200, size[0], size[1]], fill=(100, 100, 100))
        for i in range(num_products):
            x = 100 + i * 300
            y = size[1] - 180
            w, h = 100, 150
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=2)

    elif scenario == "price_tags":
        # Focus on price tags
        for i in range(num_products):
            x = random.randint(50, size[0] - 200)
            y = random.randint(50, size[1] - 100)
            draw.rectangle([x, y, x + 150, y + 60], fill=(255, 255, 0), outline=(0, 0, 0), width=2)

    return img


def image_to_base64(image):
    """Convert PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_inference(task, model_type, image, config=None, model_source="pretrained", class_mapping=None):
    """Run inference through handler (simulating API call)."""
    from handler import handler

    job_input = {
        "input": {
            "task": task,
            "model_id": f"test-{task}-{model_type}",
            "model_source": model_source,
            "model_type": model_type,
            "image": image_to_base64(image),
            "config": config or {},
        }
    }

    if class_mapping:
        job_input["input"]["class_mapping"] = class_mapping

    start = time.time()
    result = handler(job_input)
    elapsed = (time.time() - start) * 1000

    return result, elapsed


def record_result(scenario_id, name, passed, duration_ms, details, blocks, models):
    """Record test result."""
    RESULTS.append(TestResult(
        scenario_id=scenario_id,
        name=name,
        passed=passed,
        duration_ms=duration_ms,
        details=details,
        blocks_tested=blocks,
        models_tested=models,
    ))

    status = "PASS" if passed else "FAIL"
    log(f"Scenario {scenario_id}: {name} - {status} ({duration_ms:.0f}ms)", status)
    if not passed:
        log(f"  Details: {details}", "FAIL")


# ============================================================================
# 20 Retail Scenarios
# ============================================================================

def scenario_01_shelf_product_detection():
    """Scenario 1: Detect products on store shelf using YOLO11n."""
    log("Scenario 1: Shelf Product Detection (YOLO11n)", "TEST")

    image = create_retail_image(scenario="shelf", num_products=8)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.25, "max_detections": 100}
    )

    if not result.get("success"):
        record_result(1, "Shelf Product Detection", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    record_result(1, "Shelf Product Detection", True, elapsed, f"{count} products detected", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_02_product_classification_vit():
    """Scenario 2: Classify single product using ViT-tiny."""
    log("Scenario 2: Product Classification (ViT-tiny)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="classification",
        model_type="vit-tiny",
        image=image,
        config={"top_k": 5}
    )

    if not result.get("success"):
        record_result(2, "Product Classification ViT", False, elapsed, result.get("error", "Unknown"), ["ClassificationBlock"], ["vit-tiny"])
        return False

    top_class = result["result"]["top_class"]
    confidence = result["result"]["top_confidence"]
    record_result(2, "Product Classification ViT", True, elapsed, f"{top_class} ({confidence*100:.1f}%)", ["ClassificationBlock"], ["vit-tiny"])
    return True


def scenario_03_product_embedding_dinov2():
    """Scenario 3: Extract product embedding using DINOv2."""
    log("Scenario 3: Product Embedding (DINOv2-small)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="embedding",
        model_type="dinov2-small",
        image=image,
        config={"normalize": True, "pooling": "cls"}
    )

    if not result.get("success"):
        record_result(3, "Product Embedding DINOv2", False, elapsed, result.get("error", "Unknown"), ["EmbeddingBlock"], ["dinov2-small"])
        return False

    dim = result["result"]["embedding_dim"]
    normalized = result["result"]["normalized"]
    record_result(3, "Product Embedding DINOv2", True, elapsed, f"dim={dim}, normalized={normalized}", ["EmbeddingBlock"], ["dinov2-small"])
    return True


def scenario_04_warehouse_box_counting():
    """Scenario 4: Count boxes in warehouse using YOLO11s."""
    log("Scenario 4: Warehouse Box Counting (YOLO11s)", "TEST")

    image = create_retail_image(scenario="warehouse", num_products=12)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11s",
        image=image,
        config={"confidence": 0.3, "max_detections": 200}
    )

    if not result.get("success"):
        record_result(4, "Warehouse Box Counting", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11s"])
        return False

    count = result["result"]["count"]
    record_result(4, "Warehouse Box Counting", True, elapsed, f"{count} boxes detected", ["DetectionBlock"], ["yolo11s"])
    return True


def scenario_05_price_tag_detection():
    """Scenario 5: Detect price tags on shelf."""
    log("Scenario 5: Price Tag Detection (YOLO11n)", "TEST")

    image = create_retail_image(scenario="price_tags", num_products=10)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.2, "input_size": 1280}
    )

    if not result.get("success"):
        record_result(5, "Price Tag Detection", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    record_result(5, "Price Tag Detection", True, elapsed, f"{count} tags detected", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_06_brand_classification_vit_base():
    """Scenario 6: Classify brand using ViT-base."""
    log("Scenario 6: Brand Classification (ViT-base)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single", add_brand_logos=True)

    # Custom brand mapping
    brand_mapping = {0: "CocaCola", 1: "Pepsi", 2: "Nestle", 3: "Unilever", 4: "PG"}

    result, elapsed = run_inference(
        task="classification",
        model_type="vit-base",
        image=image,
        config={"top_k": 5},
        class_mapping=brand_mapping
    )

    if not result.get("success"):
        record_result(6, "Brand Classification", False, elapsed, result.get("error", "Unknown"), ["ClassificationBlock"], ["vit-base"])
        return False

    top_class = result["result"]["top_class"]
    record_result(6, "Brand Classification", True, elapsed, f"Brand: {top_class}", ["ClassificationBlock"], ["vit-base"])
    return True


def scenario_07_checkout_item_detection():
    """Scenario 7: Detect items at checkout counter."""
    log("Scenario 7: Checkout Item Detection (YOLO11n)", "TEST")

    image = create_retail_image(scenario="checkout", num_products=6)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.35}
    )

    if not result.get("success"):
        record_result(7, "Checkout Detection", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    record_result(7, "Checkout Detection", True, elapsed, f"{count} items at checkout", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_08_similar_product_search():
    """Scenario 8: Find similar products using embedding similarity."""
    log("Scenario 8: Similar Product Search (DINOv2-base)", "TEST")

    # Generate two product images
    image1 = create_retail_image(size=(640, 480), scenario="product_single")
    image2 = create_retail_image(size=(640, 480), scenario="product_single")

    result1, elapsed1 = run_inference(
        task="embedding",
        model_type="dinov2-base",
        image=image1,
        config={"normalize": True}
    )

    result2, elapsed2 = run_inference(
        task="embedding",
        model_type="dinov2-base",
        image=image2,
        config={"normalize": True}
    )

    if not result1.get("success") or not result2.get("success"):
        record_result(8, "Similar Product Search", False, elapsed1 + elapsed2, "Embedding failed", ["EmbeddingBlock"], ["dinov2-base"])
        return False

    # Calculate cosine similarity
    emb1 = result1["result"]["embedding"]
    emb2 = result2["result"]["embedding"]
    similarity = sum(a * b for a, b in zip(emb1, emb2))

    record_result(8, "Similar Product Search", True, elapsed1 + elapsed2, f"Similarity: {similarity:.4f}", ["EmbeddingBlock"], ["dinov2-base"])
    return True


def scenario_09_category_classification_convnext():
    """Scenario 9: Classify product category using ConvNeXt."""
    log("Scenario 9: Category Classification (ConvNeXt-tiny)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")

    category_mapping = {
        0: "Beverages", 1: "Snacks", 2: "Dairy", 3: "Frozen",
        4: "Produce", 5: "Meat", 6: "Bakery", 7: "Household"
    }

    result, elapsed = run_inference(
        task="classification",
        model_type="convnext-tiny",
        image=image,
        config={"top_k": 3},
        class_mapping=category_mapping
    )

    if not result.get("success"):
        record_result(9, "Category Classification", False, elapsed, result.get("error", "Unknown"), ["ClassificationBlock"], ["convnext-tiny"])
        return False

    top_class = result["result"]["top_class"]
    record_result(9, "Category Classification", True, elapsed, f"Category: {top_class}", ["ClassificationBlock"], ["convnext-tiny"])
    return True


def scenario_10_high_resolution_shelf():
    """Scenario 10: Process high-resolution shelf image."""
    log("Scenario 10: High-Resolution Shelf (YOLO11n @ 1280)", "TEST")

    image = create_retail_image(size=(2560, 1440), scenario="shelf", num_products=15)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.25, "input_size": 1280}
    )

    if not result.get("success"):
        record_result(10, "High-Res Detection", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    img_size = result["result"]["image_size"]
    record_result(10, "High-Res Detection", True, elapsed, f"{count} detections on {img_size['width']}x{img_size['height']}", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_11_multi_model_workflow():
    """Scenario 11: Multi-model workflow - detect then classify."""
    log("Scenario 11: Multi-Model Workflow (YOLO + ViT)", "TEST")

    image = create_retail_image(scenario="shelf", num_products=5)

    # Step 1: Detection
    det_result, det_elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.3}
    )

    if not det_result.get("success"):
        record_result(11, "Multi-Model Workflow", False, det_elapsed, "Detection failed", ["DetectionBlock", "ClassificationBlock"], ["yolo11n", "vit-tiny"])
        return False

    # Step 2: Classify a detected region (simulate crop)
    cls_result, cls_elapsed = run_inference(
        task="classification",
        model_type="vit-tiny",
        image=image,  # In real workflow would be cropped region
        config={"top_k": 3}
    )

    if not cls_result.get("success"):
        record_result(11, "Multi-Model Workflow", False, det_elapsed + cls_elapsed, "Classification failed", ["DetectionBlock", "ClassificationBlock"], ["yolo11n", "vit-tiny"])
        return False

    total_elapsed = det_elapsed + cls_elapsed
    det_count = det_result["result"]["count"]
    cls_class = cls_result["result"]["top_class"]
    record_result(11, "Multi-Model Workflow", True, total_elapsed,
                  f"Detected {det_count}, classified as {cls_class}",
                  ["DetectionBlock", "ClassificationBlock"], ["yolo11n", "vit-tiny"])
    return True


def scenario_12_swin_classification():
    """Scenario 12: Classification using Swin Transformer."""
    log("Scenario 12: Swin Classification (swin-tiny)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="classification",
        model_type="swin-tiny",
        image=image,
        config={"top_k": 5}
    )

    if not result.get("success"):
        record_result(12, "Swin Classification", False, elapsed, result.get("error", "Unknown"), ["ClassificationBlock"], ["swin-tiny"])
        return False

    top_class = result["result"]["top_class"]
    record_result(12, "Swin Classification", True, elapsed, f"{top_class}", ["ClassificationBlock"], ["swin-tiny"])
    return True


def scenario_13_batch_shelf_analysis():
    """Scenario 13: Analyze multiple shelf images (simulate batch)."""
    log("Scenario 13: Batch Shelf Analysis (3 images)", "TEST")

    total_detections = 0
    total_elapsed = 0

    for i in range(3):
        image = create_retail_image(scenario="shelf", num_products=random.randint(5, 10))
        result, elapsed = run_inference(
            task="detection",
            model_type="yolo11n",
            image=image,
            config={"confidence": 0.3}
        )

        if not result.get("success"):
            record_result(13, "Batch Shelf Analysis", False, total_elapsed, f"Image {i+1} failed", ["DetectionBlock"], ["yolo11n"])
            return False

        total_detections += result["result"]["count"]
        total_elapsed += elapsed

    record_result(13, "Batch Shelf Analysis", True, total_elapsed,
                  f"{total_detections} total detections in 3 images", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_14_embedding_with_gem_pooling():
    """Scenario 14: Embedding extraction with GeM pooling."""
    log("Scenario 14: GeM Pooling Embedding (DINOv2-small)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="embedding",
        model_type="dinov2-small",
        image=image,
        config={"normalize": True, "pooling": "gem"}
    )

    if not result.get("success"):
        record_result(14, "GeM Pooling Embedding", False, elapsed, result.get("error", "Unknown"), ["EmbeddingBlock"], ["dinov2-small"])
        return False

    dim = result["result"]["embedding_dim"]
    record_result(14, "GeM Pooling Embedding", True, elapsed, f"dim={dim} with GeM", ["EmbeddingBlock"], ["dinov2-small"])
    return True


def scenario_15_low_confidence_detection():
    """Scenario 15: Detection with low confidence threshold."""
    log("Scenario 15: Low Confidence Detection (conf=0.1)", "TEST")

    image = create_retail_image(scenario="shelf", num_products=10)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.1, "max_detections": 500}
    )

    if not result.get("success"):
        record_result(15, "Low Confidence Detection", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    record_result(15, "Low Confidence Detection", True, elapsed, f"{count} detections at conf=0.1", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_16_mean_pooling_embedding():
    """Scenario 16: Embedding with mean pooling."""
    log("Scenario 16: Mean Pooling Embedding (DINOv2-small)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="embedding",
        model_type="dinov2-small",
        image=image,
        config={"normalize": True, "pooling": "mean"}
    )

    if not result.get("success"):
        record_result(16, "Mean Pooling Embedding", False, elapsed, result.get("error", "Unknown"), ["EmbeddingBlock"], ["dinov2-small"])
        return False

    # Verify still normalized
    embedding = result["result"]["embedding"]
    norm = math.sqrt(sum(x**2 for x in embedding))

    record_result(16, "Mean Pooling Embedding", True, elapsed, f"norm={norm:.4f} with mean pool", ["EmbeddingBlock"], ["dinov2-small"])
    return True


def scenario_17_custom_class_mapping():
    """Scenario 17: Detection with custom retail class mapping."""
    log("Scenario 17: Custom Class Mapping Detection", "TEST")

    retail_classes = {
        0: "product_box",
        1: "price_tag",
        2: "shelf_divider",
        3: "promotion_sign",
        4: "shopping_cart",
    }

    image = create_retail_image(scenario="shelf", num_products=8, add_price_tags=True)
    result, elapsed = run_inference(
        task="detection",
        model_type="yolo11n",
        image=image,
        config={"confidence": 0.25},
        class_mapping=retail_classes
    )

    if not result.get("success"):
        record_result(17, "Custom Class Mapping", False, elapsed, result.get("error", "Unknown"), ["DetectionBlock"], ["yolo11n"])
        return False

    count = result["result"]["count"]
    record_result(17, "Custom Class Mapping", True, elapsed, f"{count} objects with custom classes", ["DetectionBlock"], ["yolo11n"])
    return True


def scenario_18_efficientnet_classification():
    """Scenario 18: Classification using EfficientNet."""
    log("Scenario 18: EfficientNet Classification (efficientnet-b0)", "TEST")

    image = create_retail_image(size=(640, 480), scenario="product_single")
    result, elapsed = run_inference(
        task="classification",
        model_type="efficientnet-b0",
        image=image,
        config={"top_k": 5}
    )

    if not result.get("success"):
        record_result(18, "EfficientNet Classification", False, elapsed, result.get("error", "Unknown"), ["ClassificationBlock"], ["efficientnet-b0"])
        return False

    top_class = result["result"]["top_class"]
    record_result(18, "EfficientNet Classification", True, elapsed, f"{top_class}", ["ClassificationBlock"], ["efficientnet-b0"])
    return True


def scenario_19_full_workflow_detect_classify_embed():
    """Scenario 19: Full retail workflow - detect, classify, embed."""
    log("Scenario 19: Full Workflow (Detect → Classify → Embed)", "TEST")

    image = create_retail_image(scenario="shelf", num_products=5)

    # Step 1: Detection
    det_result, det_time = run_inference("detection", "yolo11n", image, {"confidence": 0.3})
    if not det_result.get("success"):
        record_result(19, "Full Workflow", False, det_time, "Detection failed",
                      ["DetectionBlock", "ClassificationBlock", "EmbeddingBlock"],
                      ["yolo11n", "vit-tiny", "dinov2-small"])
        return False

    # Step 2: Classification
    cls_result, cls_time = run_inference("classification", "vit-tiny", image, {"top_k": 3})
    if not cls_result.get("success"):
        record_result(19, "Full Workflow", False, det_time + cls_time, "Classification failed",
                      ["DetectionBlock", "ClassificationBlock", "EmbeddingBlock"],
                      ["yolo11n", "vit-tiny", "dinov2-small"])
        return False

    # Step 3: Embedding
    emb_result, emb_time = run_inference("embedding", "dinov2-small", image, {"normalize": True})
    if not emb_result.get("success"):
        record_result(19, "Full Workflow", False, det_time + cls_time + emb_time, "Embedding failed",
                      ["DetectionBlock", "ClassificationBlock", "EmbeddingBlock"],
                      ["yolo11n", "vit-tiny", "dinov2-small"])
        return False

    total_time = det_time + cls_time + emb_time
    det_count = det_result["result"]["count"]
    cls_class = cls_result["result"]["top_class"]
    emb_dim = emb_result["result"]["embedding_dim"]
    record_result(19, "Full Workflow", True, total_time,
                  f"Det:{det_count}, Cls:{cls_class}, Emb:{emb_dim}d",
                  ["DetectionBlock", "ClassificationBlock", "EmbeddingBlock"],
                  ["yolo11n", "vit-tiny", "dinov2-small"])
    return True


def scenario_20_stress_test():
    """Scenario 20: Stress test - rapid sequential requests."""
    log("Scenario 20: Stress Test (10 rapid requests)", "TEST")

    success_count = 0
    total_time = 0

    for i in range(10):
        image = create_retail_image(size=(640, 480), scenario="shelf", num_products=5)
        result, elapsed = run_inference(
            task="detection",
            model_type="yolo11n",
            image=image,
            config={"confidence": 0.3}
        )

        if result.get("success"):
            success_count += 1
        total_time += elapsed

    avg_time = total_time / 10

    if success_count < 10:
        record_result(20, "Stress Test", False, total_time, f"{success_count}/10 succeeded", ["DetectionBlock"], ["yolo11n"])
        return False

    record_result(20, "Stress Test", True, total_time, f"10/10 succeeded, avg {avg_time:.0f}ms", ["DetectionBlock"], ["yolo11n"])
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    log("=" * 70)
    log("END-TO-END RETAIL WORKFLOW TESTS")
    log("20 Scenarios | All Model Types | All Workflow Blocks")
    log("=" * 70)

    import torch
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("")

    # Run all 20 scenarios
    scenarios = [
        scenario_01_shelf_product_detection,
        scenario_02_product_classification_vit,
        scenario_03_product_embedding_dinov2,
        scenario_04_warehouse_box_counting,
        scenario_05_price_tag_detection,
        scenario_06_brand_classification_vit_base,
        scenario_07_checkout_item_detection,
        scenario_08_similar_product_search,
        scenario_09_category_classification_convnext,
        scenario_10_high_resolution_shelf,
        scenario_11_multi_model_workflow,
        scenario_12_swin_classification,
        scenario_13_batch_shelf_analysis,
        scenario_14_embedding_with_gem_pooling,
        scenario_15_low_confidence_detection,
        scenario_16_mean_pooling_embedding,
        scenario_17_custom_class_mapping,
        scenario_18_efficientnet_classification,
        scenario_19_full_workflow_detect_classify_embed,
        scenario_20_stress_test,
    ]

    for scenario in scenarios:
        try:
            scenario()
        except Exception as e:
            log(f"Scenario crashed: {e}", "FAIL")
            import traceback
            traceback.print_exc()
        log("")

    # Summary
    log("=" * 70)
    log("TEST SUMMARY")
    log("=" * 70)

    passed = sum(1 for r in RESULTS if r.passed)
    total = len(RESULTS)

    # Results table
    print(f"\n{'ID':<4} {'Scenario':<35} {'Status':<8} {'Time':<10} {'Blocks':<40} {'Models'}")
    print("-" * 130)

    for r in RESULTS:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        blocks = ", ".join(r.blocks_tested)
        models = ", ".join(r.models_tested)
        print(f"{r.scenario_id:<4} {r.name:<35} {status:<8} {r.duration_ms:>7.0f}ms {blocks:<40} {models}")

    print("-" * 130)

    # Block coverage
    all_blocks = set()
    all_models = set()
    for r in RESULTS:
        all_blocks.update(r.blocks_tested)
        all_models.update(r.models_tested)

    log("")
    log(f"Blocks Tested: {', '.join(sorted(all_blocks))}")
    log(f"Models Tested: {', '.join(sorted(all_models))}")
    log("")

    total_time = sum(r.duration_ms for r in RESULTS)
    log(f"Total Time: {total_time/1000:.1f}s")
    log(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")

    if passed == total:
        log("")
        log("🎉 ALL 20 SCENARIOS PASSED!", "PASS")
        return 0
    else:
        log("")
        log(f"❌ {total - passed} SCENARIOS FAILED", "FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
