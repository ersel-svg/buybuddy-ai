"""
Product Pipeline for Video Segmentation

Video → Gemini (metadata) → SAM3 (segmentation) → 518x518 Frames → Supabase Storage

Based on: buybuddy_pipeline_v2.ipynb (reference implementation)
"""

import os
import json
import time
import re
import shutil
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import cv2
from PIL import Image
import requests
import google.generativeai as genai
from tqdm import tqdm

# Parallel upload settings
MAX_UPLOAD_WORKERS = 10

from config import (
    GEMINI_API_KEY,
    HF_TOKEN,
    SUPABASE_URL,
    SUPABASE_KEY,
    TARGET_RESOLUTION,
    GEMINI_MODEL,
    TEMP_DIR,
    OUTPUT_DIR,
    DEVICE,
    MAX_FRAMES,
    FRAME_SAMPLE_RATE,
)


class ProductPipeline:
    """
    Main pipeline for processing product videos.

    Follows the same logic as buybuddy_pipeline_v2.ipynb:
    1. Download video from URL
    2. Extract frames (with sample_rate and max_frames)
    3. Extract metadata with Gemini
    4. Segment with SAM3 (single session, single prompt)
    5. Post-process frames to 518x518
    6. Upload to Supabase Storage
    7. Update product record in database
    """

    def __init__(self):
        self.device = DEVICE
        self.target_resolution = TARGET_RESOLUTION
        self.temp_dir = TEMP_DIR

        # Initialize Gemini
        print("[Pipeline] Configuring Gemini...")
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            print("[Pipeline] Gemini ready!")
        else:
            self.gemini_model = None
            print("[Pipeline] WARNING: Gemini not configured (no API key)")

        # Initialize SAM3
        print("[Pipeline] Loading SAM3 (this may take a minute)...")
        self._init_sam3()
        print("[Pipeline] SAM3 ready!")

        # Initialize Supabase
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                from supabase import create_client
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("[Pipeline] Supabase connected!")
            except Exception as e:
                print(f"[Pipeline] Supabase not available: {e}")

    def _init_sam3(self):
        """Initialize SAM3 video predictor."""
        try:
            # Authenticate with HuggingFace for model download
            if HF_TOKEN:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
                print("[Pipeline] HuggingFace authenticated")
            else:
                print("[Pipeline] WARNING: HF_TOKEN not set, SAM3 download may fail")

            from sam3.model_builder import build_sam3_video_predictor
            self.video_predictor = build_sam3_video_predictor()
            print("[Pipeline] SAM3 loaded successfully")

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"[Pipeline] GPU Memory used: {mem:.1f} GB")
        except ImportError as e:
            print(f"[Pipeline] SAM3 import error: {e}")
            print("[Pipeline] Falling back to no segmentation mode")
            self.video_predictor = None
        except Exception as e:
            print(f"[Pipeline] SAM3 load error: {e}")
            self.video_predictor = None

    def process(
        self,
        video_url: str,
        barcode: str,
        video_id: Optional[int] = None,
        product_id: Optional[str] = None,
        sample_rate: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main processing method (same as notebook).

        Args:
            video_url: URL to download video from
            barcode: Product barcode
            video_id: External video ID
            product_id: Product UUID (required for storage)
            sample_rate: Extract every Nth frame (1 = every frame). Default from config.
            max_frames: Maximum frames to extract. Default from config (None = all).

        Returns:
            {
                "metadata": {...},
                "frame_count": 177,
                "frames_url": "https://...",
                "primary_image_url": "https://..."
            }
        """
        # Validate product_id is provided (required for storage folder)
        if not product_id:
            raise ValueError("product_id is required for storage. Create product before processing.")

        # Use config defaults if not specified
        if sample_rate is None:
            sample_rate = FRAME_SAMPLE_RATE
        if max_frames is None:
            max_frames = MAX_FRAMES

        # Create temp directory for this job
        job_dir = self.temp_dir / product_id
        job_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            # 1. Download video
            print("\n[1/7] Downloading video...")
            video_path = self._download_video(video_url, job_dir / "video.mp4")
            print(f"      Saved to: {video_path}")

            # 2. Extract frames (notebook style: sample_rate + max_frames)
            print("\n[2/7] Extracting frames...")
            all_frames, video_indices = self._extract_frames(video_path, sample_rate=sample_rate, max_frames=max_frames)
            print(f"      Extracted {len(all_frames)} frames")

            # Calculate max video frame index for SAM3 propagation
            max_video_frame_idx = video_indices[-1] if video_indices else 0

            # 3. Gemini metadata extraction
            print("\n[3/7] Extracting metadata with Gemini...")
            if self.gemini_model:
                metadata = self._extract_metadata(video_path)
                print(f"      Brand: {metadata.get('brand_info', {}).get('brand_name', 'N/A')}")
                print(f"      Product: {metadata.get('product_identity', {}).get('product_name', 'N/A')}")
            else:
                metadata = {"_note": "Gemini not configured"}
                print("      Skipped (no Gemini API key)")

            # 4. SAM3 segmentation (single session, multiple prompts with fallback)
            print("\n[4/7] Segmenting with SAM3...")
            grounding_prompts = []
            if self.video_predictor:
                grounding_prompts = self._get_grounding_prompts(metadata)
                # Pass max_video_frame_idx + 1 to ensure SAM3 propagates through all needed frames
                all_frame_outputs = self._segment_video_sam3(video_path, grounding_prompts, max_video_frame_idx + 1)
                print(f"      Segmented {len(all_frame_outputs)} frames")
            else:
                print("      Skipped (SAM3 not available, using raw frames)")
                all_frame_outputs = {}

            # 5. Post-process frames (use video_indices to map masks correctly)
            print("\n[5/7] Post-processing frames...")
            processed_frames = self._process_frames(all_frames, all_frame_outputs, video_indices)
            print(f"      Processed {len(processed_frames)} frames to {self.target_resolution}x{self.target_resolution}")

            # Add pipeline info to metadata
            metadata["_pipeline"] = {
                "barcode": barcode,
                "video_id": video_id,
                "product_id": product_id,
                "timestamp": datetime.now().isoformat(),
                "sample_rate": sample_rate,
                "max_frames": max_frames,
                "frame_count": len(processed_frames),
                "grounding_prompts": grounding_prompts if self.video_predictor else None,
                "model": "SAM3",
            }

            # 6. Save frames to storage
            print("\n[6/7] Saving frames to storage...")
            frames_url, primary_image_url = self._save_frames(
                processed_frames, barcode, product_id, frames_dir, metadata
            )
            print(f"      Frames URL: {frames_url}")
            print(f"      Primary Image: {primary_image_url}")

            # 7. Sync product and frame records to database (idempotent)
            if product_id and self.supabase:
                print("\n[7/7] Syncing to database...")
                self._sync_product_and_frames(
                    product_id=product_id,
                    barcode=barcode,
                    video_url=video_url,
                    video_id=video_id,
                    metadata=metadata,
                    frame_count=len(processed_frames),
                    frames_url=frames_url,
                    primary_image_url=primary_image_url,
                )

            return {
                "metadata": metadata,
                "frame_count": len(processed_frames),
                "frames_url": frames_url,
                "primary_image_url": primary_image_url,
            }

        finally:
            # Cleanup
            self._cleanup(job_dir)

    def _download_video(self, url: str, save_path: Path) -> Path:
        """Download video from URL."""
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return save_path

    def _extract_frames(
        self,
        video_path: Path,
        sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from video (same as notebook).

        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame (1 = every frame)
            max_frames: Maximum frames to extract (None = all)

        Returns:
            Tuple of (frames, video_indices) where:
            - frames: List of RGB frames
            - video_indices: List of actual video frame indices for each extracted frame
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        video_indices = []  # Track actual video frame indices
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        print(f"      Total video frames: {total_frames}, sample_rate: {sample_rate}, max_frames: {max_frames}")

        with tqdm(total=total_frames, desc="Extracting") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample every Nth frame (same as notebook)
                if frame_idx % sample_rate == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    video_indices.append(frame_idx)  # Store actual video frame index

                    # Stop if we reached max_frames
                    if max_frames and len(frames) >= max_frames:
                        break

                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"      Extracted frames at video indices: [0:{video_indices[-1] if video_indices else 0}] (step={sample_rate})")
        return frames, video_indices

    def _extract_metadata(self, video_path: Path, max_retries: int = 3) -> dict:
        """Extract metadata using Gemini with rate limit protection."""
        last_error = None
        video_file = None

        for attempt in range(max_retries):
            try:
                # Upload video to Gemini
                video_file = genai.upload_file(path=str(video_path))

                # Wait for processing with timeout
                wait_count = 0
                max_wait = 60  # 2 minutes max (60 * 2s)
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    wait_count += 1
                    if wait_count > max_wait:
                        raise ValueError("Gemini video processing timeout")
                    video_file = genai.get_file(video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError("Gemini video processing failed")

                # Generate metadata
                prompt = self._get_extraction_prompt()
                response = self.gemini_model.generate_content([video_file, prompt])

                # Parse JSON response
                text = response.text.strip()
                text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
                text = re.sub(r"^```\s*", "", text)
                text = re.sub(r"```\s*$", "", text)

                # Cleanup: Delete uploaded file from Gemini (prevent quota issues)
                self._cleanup_gemini_file(video_file)

                try:
                    return json.loads(text.strip())
                except json.JSONDecodeError as e:
                    print(f"      Warning: JSON parse error: {e}")
                    return {"parse_error": str(e), "raw": text[:500]}

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Cleanup on error too
                if video_file:
                    self._cleanup_gemini_file(video_file)

                # Check for rate limit errors
                if "429" in str(e) or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    base_wait = (attempt + 1) * 30
                    jitter = random.uniform(0, base_wait * 0.5)
                    wait_time = base_wait + jitter
                    print(f"      Gemini rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                # For other errors, shorter retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"      Gemini error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                raise

        raise last_error or ValueError("Gemini metadata extraction failed after retries")

    def _cleanup_gemini_file(self, video_file):
        """Delete uploaded file from Gemini to prevent quota issues."""
        try:
            genai.delete_file(video_file.name)
            print(f"      Gemini file cleaned up: {video_file.name}")
        except Exception as e:
            print(f"      Warning: Gemini file cleanup failed: {e}")

    def _get_extraction_prompt(self) -> str:
        """Get Gemini extraction prompt."""
        return """You are an expert E-commerce Cataloging AI. Your task is to analyze the provided product video and generate a structured JSON output for a product directory.

### VIDEO PROCESSING STRATEGY
1. Multi-Frame Analysis: Scan the entire video. The front usually contains Brand/Name, while the back/sides contain Nutrition/Ingredients. Aggregate data from all visible angles.
2. Best Frame Selection: Identify frames with the highest resolution text to minimize OCR errors.
3. Visual Confirmation: Only extract text that is visibly legible. Do not infer details based on shape alone.

### TEXT FORMATTING & NORMALIZATION RULES
1. Spelling: Maintain EXACT spelling as seen on the package (even if stylized).
2. Capitalization: Convert package text (often ALL CAPS) into Title Case for Brands and Product Names.
3. Empty Values: If a field is not clearly visible, return null for objects/numbers or "" for strings.

### JSON OUTPUT SCHEMA
Return ONLY the following JSON object. No markdown, no conversational text.

{
  "brand_info": {
    "brand_name": "String: Main manufacturer (e.g., 'Sony', 'Nestlé'). Title Case.",
    "sub_brand": "String: Product line or sub-brand (e.g., 'Bravia', 'KitKat Chunky'). Title Case. Empty string if none.",
    "manufacturer_country": "String: 'Made in X' if visible. Empty string if not."
  },
  "product_identity": {
    "product_name": "String: The specific item name (e.g., 'Noise Canceling Headphones', 'Salted Caramel Bar'). Title Case.",
    "variant_flavor": "String: Specific flavor/scent/color (e.g., 'Midnight Blue', 'Spicy Chili'). Title Case.",
    "product_category": "Enum: [beverages, snacks, confectionery, dairy_refrigerated, pantry_food, frozen_food, bakery, health_personal_care, household_cleaning, electronics_accessories, pet_supplies, apparel, other]",
    "container_type": "Enum: [bottle, can, box, bag, pouch, jar, tube, packet, carton, cup, tray, blister_pack, wrapper, other]"
  },
  "specifications": {
    "net_quantity_text": "String: Exact weight/volume text (e.g., '500 ml', '12 oz', '250g').",
    "pack_configuration": {
        "type": "Enum: 'single_unit' or 'multipack'. Determine based on text (e.g., '6 Pack') or visual cues (e.g., plastic rings, box holding multiple units).",
        "item_count": "Integer: Total units. Return 1 for single_unit, specific count for multipack."
    },
    "identifiers": {
      "barcode": "String: Digits only. Null if not visible.",
      "sku_model_code": "String: Visible model numbers. Null if not visible."
    }
  },
  "marketing_and_claims": {
    "claims_list": ["Array of Strings: Visible badges like 'Non-GMO', 'Keto Friendly', 'Bluetooth 5.0'. Title Case."],
    "marketing_description": "String: Brief descriptive blurb found on the package. Sentence case."
  },
  "nutrition_facts": {
    "note": "Only extract if clearly visible. Use strict numbers (float). Null if not visible.",
    "serving_size": "String: e.g., '1 cup (240ml)'",
    "calories": "Number: e.g., 150",
    "total_fat": "String: e.g., '8g'",
    "protein": "String: e.g., '12g'",
    "carbohydrates": "String: e.g., '24g'",
    "sugar": "String: e.g., '10g'"
  },
  "visual_grounding": {
    "grounding_prompts": [
      "String: Primary prompt - '{Brand} {Product Name} {Container Type}' (e.g., 'Coca-Cola Zero Sugar can')",
      "String: Simple container type (e.g., 'can', 'bottle', 'box')",
      "String: Alternative description (e.g., 'soda can', 'beverage bottle')"
    ]
  },
  "extraction_metadata": {
    "visibility_score": "Number: 0-100 confidence based on text clarity.",
    "issues_detected": ["Array of Strings: e.g., 'glare_on_text', 'barcode_cutoff', 'back_of_pack_missing'"]
  }
}"""

    # Fallback prompts for common retail packaging types
    FALLBACK_PROMPTS = [
        "can",
        "bottle",
        "box",
        "bag",
        "package",
        "container",
        "jar",
        "pouch",
        "carton",
        "tube",
        "product",
    ]

    def _get_grounding_prompts(self, metadata: dict) -> List[str]:
        """
        Get ordered list of grounding prompts from metadata.

        Returns list of prompts to try in order, e.g.:
        ["Coca-Cola Zero Sugar can", "can", "soda can", "bottle", "container"]
        """
        prompts = []

        # 1. Try grounding_prompts array from Gemini
        gemini_prompts = metadata.get("visual_grounding", {}).get("grounding_prompts", [])
        if isinstance(gemini_prompts, list):
            for p in gemini_prompts:
                if p and isinstance(p, str) and p.strip():
                    prompts.append(p.strip())

        # 2. Try legacy single grounding_prompt
        legacy_prompt = metadata.get("visual_grounding", {}).get("grounding_prompt", "")
        if legacy_prompt and legacy_prompt not in prompts and legacy_prompt != "Brand ProductName ContainerType":
            prompts.insert(0, legacy_prompt)

        # 3. Build from components if needed
        brand = metadata.get("brand_info", {}).get("brand_name", "") or ""
        prod_name = metadata.get("product_identity", {}).get("product_name", "") or ""
        container = metadata.get("product_identity", {}).get("container_type", "") or ""

        if brand and prod_name and container:
            built_prompt = f"{brand} {prod_name} {container}".strip()
            if built_prompt not in prompts:
                prompts.insert(0, built_prompt)

        # 4. Add container type as fallback
        if container and container.lower() not in [p.lower() for p in prompts]:
            prompts.append(container.lower())

        # 5. Add generic fallbacks if we have less than 3 prompts
        for fallback in self.FALLBACK_PROMPTS:
            if len(prompts) >= 5:
                break
            if fallback not in [p.lower() for p in prompts]:
                prompts.append(fallback)

        # Ensure at least one prompt
        if not prompts:
            prompts = ["product"]

        print(f"      Grounding prompts: {prompts[:5]}")
        return prompts[:5]  # Max 5 prompts

    # Minimum mask pixels to consider segmentation successful
    MIN_MASK_PIXELS = 10000

    def _check_segmentation_quality(self, all_frame_outputs: dict) -> bool:
        """Check if segmentation results have enough valid pixels across the video."""
        if not all_frame_outputs:
            return False

        frame_keys = sorted(all_frame_outputs.keys())
        total_frames = len(frame_keys)

        if total_frames == 0:
            return False

        # Sample frames distributed across the video (start, middle, end)
        if total_frames >= 3:
            sample_indices = [
                frame_keys[0],                      # Start
                frame_keys[total_frames // 2],     # Middle
                frame_keys[-1],                    # End
            ]
        else:
            sample_indices = frame_keys

        valid_frames = 0
        for frame_idx in sample_indices:
            if frame_idx not in all_frame_outputs:
                continue

            outputs = all_frame_outputs[frame_idx]
            masks = outputs.get("out_binary_masks", [])

            if len(masks) > 0:
                mask = masks[0]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                pixel_count = int(mask.sum())
                if pixel_count >= self.MIN_MASK_PIXELS:
                    valid_frames += 1

        # Need at least 2 valid frames
        return valid_frames >= 2

    def _segment_video_sam3(self, video_path: Path, grounding_prompts: List[str], num_frames: int) -> dict:
        """
        Segment video with SAM3 using efficient fallback.

        SINGLE session, multiple prompts via reset_session.
        Video is loaded ONCE, only prompts are changed between attempts.
        """
        print(f"      Starting SAM3 session...")

        # Start session (video loaded ONCE)
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]
        print(f"      Session ID: {session_id}")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"      GPU Memory after load: {mem:.1f} GB")

        all_frame_outputs = {}
        successful_prompt = None

        try:
            # Try each prompt in order (within same session)
            for i, prompt in enumerate(grounding_prompts):
                print(f"\n      [{i+1}/{len(grounding_prompts)}] Trying prompt: '{prompt}'")

                # Reset session for new prompt (keeps video loaded, clears prompts)
                if i > 0:
                    print(f"      Resetting session for new prompt...")
                    self.video_predictor.handle_request(
                        request=dict(
                            type="reset_session",
                            session_id=session_id,
                        )
                    )

                # Add text prompt on first frame
                self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=prompt,
                    )
                )

                # Propagate through video
                print(f"      Propagating through {num_frames} frames...")
                propagate_result = self.video_predictor.propagate_in_video(
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_idx=0,
                    max_frame_num_to_track=num_frames,
                )

                # Collect results
                all_frame_outputs = {}

                if hasattr(propagate_result, "__iter__") and not isinstance(propagate_result, (dict, str)):
                    for item in tqdm(propagate_result, desc="Segmenting", total=num_frames):
                        if isinstance(item, dict):
                            frame_idx = item.get("frame_index", len(all_frame_outputs))
                            all_frame_outputs[frame_idx] = item.get("outputs", item)
                        elif isinstance(item, tuple):
                            frame_idx = item[0] if len(item) > 0 else len(all_frame_outputs)
                            all_frame_outputs[frame_idx] = item[1] if len(item) > 1 else item
                        else:
                            all_frame_outputs[len(all_frame_outputs)] = item
                elif isinstance(propagate_result, dict):
                    all_frame_outputs = propagate_result

                # Check quality
                if self._check_segmentation_quality(all_frame_outputs):
                    print(f"      SUCCESS: Prompt '{prompt}' found valid segmentation!")
                    successful_prompt = prompt
                    break
                else:
                    print(f"      FAIL: Prompt '{prompt}' - insufficient mask coverage")

            if not successful_prompt:
                print(f"      WARNING: All prompts failed quality check, using last result")

            return all_frame_outputs

        finally:
            # Close session (ONCE at the end)
            print("      Closing SAM3 session...")
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"      GPU Memory after cleanup: {mem:.1f} GB")

    def _select_best_mask(
        self,
        masks: List[np.ndarray],
        frame_shape: tuple,
    ) -> np.ndarray:
        """
        Select the best mask when multiple objects are detected.

        Prefers masks that are:
        1. Larger (more area)
        2. More centered in the frame

        Args:
            masks: List of binary masks
            frame_shape: (height, width) of the frame

        Returns:
            The best mask based on area and center proximity
        """
        if len(masks) == 1:
            return masks[0]

        h, w = frame_shape[:2]
        frame_center = np.array([w / 2, h / 2])
        frame_area = h * w

        best_mask = masks[0]
        best_score = -1

        for mask in masks:
            # Convert to numpy if tensor
            if hasattr(mask, "cpu"):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            # Calculate mask area (normalized by frame area)
            mask_area = np.sum(mask_np > 0)
            area_score = mask_area / frame_area

            # Calculate mask centroid
            coords = np.where(mask_np > 0)
            if len(coords[0]) == 0:
                continue

            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            centroid = np.array([centroid_x, centroid_y])

            # Calculate distance from center (normalized)
            max_distance = np.sqrt(frame_center[0]**2 + frame_center[1]**2)
            distance = np.linalg.norm(centroid - frame_center)
            center_score = 1 - (distance / max_distance)  # 1 = perfectly centered, 0 = corner

            # Combined score: area matters more, but center is a tiebreaker
            # Weight: 70% area, 30% center proximity
            score = 0.7 * area_score + 0.3 * center_score

            if score > best_score:
                best_score = score
                best_mask = mask

        return best_mask

    # Minimum mask coverage threshold (1% of frame area)
    MIN_MASK_COVERAGE = 0.01

    def _process_frames(
        self,
        frames: List[np.ndarray],
        all_frame_outputs: dict,
        video_indices: List[int] = None,
        min_mask_coverage: float = None,
    ) -> List[np.ndarray]:
        """
        Process frames: apply mask, crop, center, resize to target resolution.
        Skips frames without valid segmentation (no mask or coverage below threshold).

        Args:
            frames: List of extracted frames
            all_frame_outputs: SAM3 segmentation outputs keyed by video frame index
            video_indices: Actual video frame indices for each extracted frame.
                           When sample_rate > 1, frames[i] corresponds to video_indices[i].
                           If None, assumes sequential indices (0, 1, 2, ...).
            min_mask_coverage: Minimum mask coverage as fraction of frame area (0.01 = 1%).
                              Frames with less coverage are skipped.
        """
        processed = []
        skipped_count = 0

        # Use class default if not specified
        if min_mask_coverage is None:
            min_mask_coverage = self.MIN_MASK_COVERAGE

        # If no video_indices provided, use sequential indices (backward compatibility)
        if video_indices is None:
            video_indices = list(range(len(frames)))

        for i, frame in enumerate(frames):
            # Use actual video frame index to look up the correct mask
            video_frame_idx = video_indices[i]

            if video_frame_idx in all_frame_outputs:
                outputs = all_frame_outputs[video_frame_idx]
                out_masks = outputs.get("out_binary_masks", [])
                if len(out_masks) > 0:
                    # Select best mask if multiple objects detected
                    if len(out_masks) > 1:
                        mask = self._select_best_mask(out_masks, frame.shape)
                    else:
                        mask = out_masks[0]
                    # Convert to numpy if tensor
                    if hasattr(mask, "cpu"):
                        mask = mask.cpu().numpy()

                    # Ensure mask is 2D for coverage calculation
                    mask_2d = mask.squeeze() if mask.ndim > 2 else mask

                    # Check mask coverage
                    frame_area = frame.shape[0] * frame.shape[1]
                    mask_area = np.sum(mask_2d > 0)
                    coverage = mask_area / frame_area

                    if coverage < min_mask_coverage:
                        # Skip frame with insufficient mask coverage
                        skipped_count += 1
                        continue

                    result = self._center_on_canvas(frame, mask)
                else:
                    # No mask detected - skip frame
                    skipped_count += 1
                    continue
            else:
                # No segmentation output for this frame - skip
                skipped_count += 1
                continue

            processed.append(result)

        if skipped_count > 0:
            print(f"      Skipped {skipped_count} frames with insufficient segmentation")

        return processed

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame maintaining aspect ratio, center on black canvas."""
        h, w = frame.shape[:2]
        scale = self.target_resolution / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center on black canvas
        canvas = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
        y_off = (self.target_resolution - new_h) // 2
        x_off = (self.target_resolution - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        return canvas

    def _center_on_canvas(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crop product using mask and center on black canvas.
        Same as notebook's process_frame_with_mask function.
        """
        # Resize mask if needed
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
            mask = mask.astype(bool)

        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Crop
        cropped = frame[y_min : y_max + 1, x_min : x_max + 1].copy()
        cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        # Apply mask (black background)
        cropped[~cropped_mask] = 0

        # Resize maintaining aspect ratio
        h, w = cropped.shape[:2]
        scale = self.target_resolution / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center on black canvas
        canvas = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
        y_off = (self.target_resolution - new_h) // 2
        x_off = (self.target_resolution - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        return canvas

    def _save_frames(
        self,
        frames: List[np.ndarray],
        barcode: str,
        product_id: Optional[str],
        frames_dir: Path,
        metadata: dict = None,
    ) -> tuple[str, str]:
        """Save frames and metadata to storage. Returns (frames_url, primary_image_url)."""
        # Production mode: Supabase available → skip local save, upload directly
        if self.supabase:
            try:
                return self._upload_to_supabase(frames, barcode, product_id, metadata)
            except Exception as e:
                print(f"      Warning: Supabase upload failed: {e}, falling back to local save")

        # Local test mode: No Supabase → save to disk
        output_dir = OUTPUT_DIR / barcode
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path, format="PNG")

        # Save metadata as JSON
        if metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(output_dir), None

    def _upload_to_supabase(
        self,
        frames: List[np.ndarray],
        barcode: str,
        product_id: str,
        metadata: dict = None,
    ) -> tuple[str, str]:
        """Upload frames and metadata to Supabase Storage with parallel uploads."""
        bucket = "frames"
        folder = product_id

        print(f"      Uploading {len(frames)} frames (parallel)...")

        # Prepare frame data for parallel upload
        frame_data: List[Tuple[int, bytes]] = []
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            frame_data.append((i, buffer.getvalue()))

        def upload_single_frame(item: Tuple[int, bytes], max_retries: int = 3) -> Tuple[int, bool]:
            """Upload a single frame with retry logic."""
            idx, data = item
            path = f"{folder}/frame_{idx:04d}.png"

            for attempt in range(max_retries):
                try:
                    self.supabase.storage.from_(bucket).upload(
                        path, data, {"content-type": "image/png"}
                    )
                    return idx, True
                except Exception as e:
                    error_str = str(e).lower()
                    if "already exists" in error_str or "duplicate" in error_str:
                        try:
                            self.supabase.storage.from_(bucket).update(
                                path, data, {"content-type": "image/png"}
                            )
                            return idx, True
                        except Exception:
                            pass
                    if attempt < max_retries - 1:
                        base_wait = 0.5 * (attempt + 1)
                        jitter = random.uniform(0, base_wait)
                        time.sleep(base_wait + jitter)
                        continue
            return idx, False

        # Parallel upload
        failed_items = list(frame_data)
        success_count = 0
        max_rounds = 3

        for round_num in range(max_rounds):
            if not failed_items:
                break

            current_failed = []
            with ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS) as executor:
                futures = {executor.submit(upload_single_frame, item): item for item in failed_items}
                for future in as_completed(futures):
                    idx, success = future.result()
                    if success:
                        success_count += 1
                    else:
                        current_failed.append(futures[future])

            failed_items = current_failed
            if failed_items and round_num < max_rounds - 1:
                print(f"      Round {round_num + 1}: {len(failed_items)} failed, retrying...")
                time.sleep(1)

        print(f"      Uploaded {success_count}/{len(frames)} frames")

        # Upload metadata JSON
        if metadata:
            metadata_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
            metadata_path = f"{folder}/metadata.json"
            try:
                self.supabase.storage.from_(bucket).upload(
                    metadata_path, metadata_bytes, {"content-type": "application/json"}
                )
            except Exception:
                try:
                    self.supabase.storage.from_(bucket).update(
                        metadata_path, metadata_bytes, {"content-type": "application/json"}
                    )
                except Exception as e:
                    print(f"      Warning: metadata upload failed: {e}")

        frames_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/"
        primary_image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/frame_0000.png"

        return frames_url, primary_image_url

    def _sync_product_and_frames(
        self,
        product_id: str,
        barcode: str,
        video_url: str,
        video_id: Optional[int],
        metadata: dict,
        frame_count: int,
        frames_url: str,
        primary_image_url: str,
    ):
        """Sync product and frame records to database."""
        if not self.supabase:
            print("      Skipping database sync - no supabase client")
            return

        bucket = "frames"
        folder = product_id

        try:
            print(f"      Upserting product {product_id}...")

            product_data = {
                "id": product_id,
                "barcode": barcode,
                "video_url": video_url,
                "frame_count": frame_count,
                "frames_path": frames_url,
                "primary_image_url": primary_image_url,
                "status": "ready",
            }

            if video_id is not None:
                product_data["video_id"] = video_id

            # Extract fields from metadata
            brand_info = metadata.get("brand_info", {})
            product_identity = metadata.get("product_identity", {})
            specs = metadata.get("specifications", {})

            if brand_info.get("brand_name"):
                product_data["brand_name"] = brand_info["brand_name"]
            if brand_info.get("sub_brand"):
                product_data["sub_brand"] = brand_info["sub_brand"]
            if product_identity.get("product_name"):
                product_data["product_name"] = product_identity["product_name"]
            if product_identity.get("variant_flavor"):
                product_data["variant_flavor"] = product_identity["variant_flavor"]
            if product_identity.get("product_category"):
                product_data["category"] = product_identity["product_category"]
            if product_identity.get("container_type"):
                product_data["container_type"] = product_identity["container_type"]
            if specs.get("net_quantity_text"):
                product_data["net_quantity"] = specs["net_quantity_text"]

            grounding = metadata.get("visual_grounding", {})
            if grounding.get("grounding_prompt"):
                product_data["grounding_prompt"] = grounding["grounding_prompt"]

            extraction = metadata.get("extraction_metadata", {})
            if extraction.get("visibility_score") is not None:
                product_data["visibility_score"] = int(extraction["visibility_score"])

            nutrition = metadata.get("nutrition_facts", {})
            if nutrition:
                product_data["nutrition_facts"] = nutrition

            self.supabase.table("products").upsert(
                product_data,
                on_conflict="id"
            ).execute()
            print(f"      Product {product_id} synced")

            # Sync frame records
            print(f"      Syncing {frame_count} frame records...")

            self.supabase.table("product_images").delete().eq(
                "product_id", product_id
            ).eq("image_type", "synthetic").execute()

            records = []
            for i in range(frame_count):
                image_path = f"{folder}/frame_{i:04d}.png"
                image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{image_path}"
                records.append({
                    "product_id": product_id,
                    "image_type": "synthetic",
                    "source": "video_frame",
                    "image_path": image_path,
                    "image_url": image_url,
                    "frame_index": i,
                })

            if records:
                self.supabase.table("product_images").insert(records).execute()
                print(f"      Inserted {len(records)} frame records")

            if video_id is not None:
                print(f"      Updating video {video_id} status...")
                self.supabase.table("videos").update({
                    "status": "completed",
                    "product_id": product_id,
                }).eq("id", video_id).execute()
                print(f"      Video {video_id} marked as completed")

            print(f"      Database sync complete!")

        except Exception as e:
            print(f"      Warning: Database sync failed: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup(self, job_dir: Path):
        """Cleanup temporary files and GPU memory."""
        # Cleanup temp files
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            print(f"      Warning: File cleanup failed: {e}")

        # Cleanup GPU memory
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"      Memory cleanup done. GPU: {mem:.1f} GB")
            except Exception as e:
                print(f"      Warning: GPU cleanup failed: {e}")
