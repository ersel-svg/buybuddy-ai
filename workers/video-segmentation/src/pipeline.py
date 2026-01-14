"""
Product Pipeline for Video Segmentation

Video → Gemini (metadata) → SAM3 (segmentation) → 518x518 Frames → Supabase Storage
"""

import os
import json
import time
import re
import shutil
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
    FRAME_SKIP,
)


class ProductPipeline:
    """
    Main pipeline for processing product videos.

    Steps:
    1. Download video from URL
    2. Extract frames
    3. Extract metadata with Gemini
    4. Segment with SAM3
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
    ) -> Dict[str, Any]:
        """
        Main processing method.

        Args:
            video_url: URL to download video from
            barcode: Product barcode
            video_id: External video ID
            product_id: Product UUID (required for storage)

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

        # Create temp directory for this job
        job_dir = self.temp_dir / product_id
        job_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            # 1. Download video
            print("\n[1/6] Downloading video...")
            video_path = self._download_video(video_url, job_dir / "video.mp4")
            print(f"      Saved to: {video_path}")

            # 2. Extract frames
            print("\n[2/6] Extracting frames...")
            all_frames = self._extract_frames(video_path)
            print(f"      Extracted {len(all_frames)} frames")

            # 3. Gemini metadata extraction
            print("\n[3/6] Extracting metadata with Gemini...")
            if self.gemini_model:
                metadata = self._extract_metadata(video_path)
                print(f"      Brand: {metadata.get('brand_info', {}).get('brand_name', 'N/A')}")
                print(f"      Product: {metadata.get('product_identity', {}).get('product_name', 'N/A')}")
            else:
                metadata = {"_note": "Gemini not configured"}
                print("      Skipped (no Gemini API key)")

            # 4. SAM3 segmentation
            print("\n[4/6] Segmenting with SAM3...")
            if self.video_predictor:
                grounding_prompts = self._get_grounding_prompts(metadata)
                all_frame_outputs = self._segment_video_sam3(video_path, grounding_prompts, len(all_frames))
                print(f"      Segmented {len(all_frame_outputs)} frames")
            else:
                print("      Skipped (SAM3 not available, using raw frames)")
                all_frame_outputs = {}

            # 5. Post-process frames
            print("\n[5/6] Post-processing frames...")
            processed_frames = self._process_frames(all_frames, all_frame_outputs)
            print(f"      Processed {len(processed_frames)} frames to {self.target_resolution}x{self.target_resolution}")

            # Add pipeline info to metadata
            metadata["_pipeline"] = {
                "barcode": barcode,
                "video_id": video_id,
                "product_id": product_id,
                "timestamp": datetime.now().isoformat(),
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

    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = min(total_frames, MAX_FRAMES) if MAX_FRAMES else total_frames

        frame_idx = 0
        for _ in tqdm(range(total_frames), desc="Extracting"):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if needed
            if frame_idx % FRAME_SKIP == 0 and len(frames) < target_frames:
                # BGR -> RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_idx += 1

        cap.release()
        return frames

    def _extract_metadata(self, video_path: Path) -> dict:
        """Extract metadata using Gemini."""
        # Upload video to Gemini
        video_file = genai.upload_file(path=str(video_path))

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
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

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"      Warning: JSON parse error: {e}")
            return {"parse_error": str(e), "raw": text[:500]}

    def _get_extraction_prompt(self) -> str:
        """Get Gemini extraction prompt."""
        return """You are an expert E-commerce Cataloging AI. Analyze this product video and return JSON only.

### RULES
1. Only extract text that is visibly legible
2. Use Title Case for brand and product names
3. Return null for missing values
4. grounding_prompts: List 3-5 simple object descriptions for SAM3 segmentation, ordered by likelihood
   - Use simple nouns: "bottle", "can", "box", "pouch", "bag", "jar", "tube", "carton", "package"
   - First should be the most specific match (e.g., "pouch" for a pouch)
   - Include alternatives (e.g., "bag", "package" as fallbacks for pouch)

### OUTPUT (JSON only, no markdown)
{
  "brand_info": {"brand_name": "", "sub_brand": "", "manufacturer_country": ""},
  "product_identity": {"product_name": "", "variant_flavor": "", "product_category": "", "container_type": ""},
  "specifications": {"net_quantity_text": "", "pack_configuration": {"type": "", "item_count": 1}, "identifiers": {"barcode": null, "sku_model_code": null}},
  "marketing_and_claims": {"claims_list": [], "marketing_description": ""},
  "nutrition_facts": {"serving_size": "", "calories": null, "total_fat": "", "protein": "", "carbohydrates": "", "sugar": ""},
  "visual_grounding": {"grounding_prompts": ["bottle", "container", "package"]},
  "extraction_metadata": {"visibility_score": 0, "issues_detected": []}
}"""

    # Fallback prompts for retail packaging types (ordered by commonality)
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
    ]

    # Minimum mask pixels to consider segmentation successful
    MIN_MASK_PIXELS = 10000

    def _get_grounding_prompts(self, metadata: dict) -> list[str]:
        """Get ordered list of grounding prompts from metadata."""
        prompts = []
        invalid_values = ["", "product", "unknown", "n/a", None]

        # 1. Try grounding_prompts array from Gemini (new format)
        gemini_prompts = metadata.get("visual_grounding", {}).get("grounding_prompts", [])
        if isinstance(gemini_prompts, list):
            for p in gemini_prompts:
                if p and str(p).strip().lower() not in invalid_values:
                    prompts.append(str(p).strip().lower())

        # 2. Add container_type if not already in list
        container = metadata.get("product_identity", {}).get("container_type", "")
        if container and container.strip().lower() not in invalid_values:
            container_lower = container.strip().lower()
            if container_lower not in prompts:
                prompts.insert(0, container_lower)  # Prioritize container_type

        # 3. Legacy: try old grounding_prompt field
        old_prompt = metadata.get("visual_grounding", {}).get("grounding_prompt", "")
        if old_prompt and old_prompt.strip() not in ["Brand ProductName ContainerType", ""] + invalid_values:
            # Extract container type keywords from old format
            prompt_lower = old_prompt.strip().lower()
            for fallback in self.FALLBACK_PROMPTS:
                if fallback in prompt_lower and fallback not in prompts:
                    prompts.append(fallback)

        # 4. Default if nothing found
        if not prompts:
            prompts = ["can"]

        print(f"      Gemini prompts: {prompts}")
        return prompts

    def _clear_cuda_cache(self):
        """Clear CUDA cache to free GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _check_segmentation_quality(self, all_frame_outputs: dict) -> bool:
        """Check if segmentation results have enough valid pixels."""
        if not all_frame_outputs:
            print("      Quality check: No frame outputs!")
            return False

        # Check first few frames for valid masks
        valid_frames = 0
        frame_indices = sorted(all_frame_outputs.keys())[:5]

        for frame_idx in frame_indices:
            outputs = all_frame_outputs[frame_idx]
            masks = outputs.get("out_binary_masks", [])

            if len(masks) > 0:
                mask = masks[0]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                pixel_count = int(mask.sum())
                is_valid = pixel_count >= self.MIN_MASK_PIXELS
                print(f"      Frame {frame_idx}: {pixel_count} pixels {'✓' if is_valid else '✗'}")
                if is_valid:
                    valid_frames += 1
            else:
                print(f"      Frame {frame_idx}: No masks found")

        result = valid_frames >= 3
        print(f"      Quality check: {valid_frames}/5 valid frames -> {'PASS' if result else 'FAIL'}")
        return result

    def _run_segmentation_with_prompt(self, video_path: Path, prompt: str, num_frames: int) -> dict:
        """Run segmentation with a single prompt. Returns frame outputs."""
        print(f"      Starting SAM3 session with prompt: '{prompt}'")
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]
        print(f"      Session ID: {session_id}")

        try:
            # Add text prompt on first frame
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )

            # Propagate through video (forward from frame 0)
            print("      Propagating through video...")
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

            return all_frame_outputs

        finally:
            # Close session and clear CUDA cache
            print("      Closing SAM3 session...")
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            self._clear_cuda_cache()

    def _segment_video_sam3(self, video_path: Path, gemini_prompts: list[str], num_frames: int) -> dict:
        """
        Segment video with SAM3 using smart fallback strategy.

        Strategy:
        1. Try Gemini's suggested prompts in order (usually 3-5 options)
        2. If all fail, try generic fallback prompts
        3. Return best result (or last attempt if all fail)

        Args:
            gemini_prompts: Ordered list of prompts from Gemini (e.g., ["pouch", "bag", "package"])
        """
        tried_prompts = set()
        all_frame_outputs = {}

        # Phase 1: Try Gemini's suggested prompts
        for i, prompt in enumerate(gemini_prompts):
            prompt_lower = prompt.lower()
            if prompt_lower in tried_prompts:
                continue
            tried_prompts.add(prompt_lower)

            label = "primary" if i == 0 else f"Gemini alt {i}"
            print(f"      Trying {label}: '{prompt}'")
            all_frame_outputs = self._run_segmentation_with_prompt(video_path, prompt, num_frames)

            if self._check_segmentation_quality(all_frame_outputs):
                print(f"      {label.capitalize()} prompt succeeded!")
                return all_frame_outputs

        print(f"      All Gemini prompts failed, trying generic fallbacks...")

        # Phase 2: Try generic fallback prompts
        for fallback_prompt in self.FALLBACK_PROMPTS:
            if fallback_prompt.lower() in tried_prompts:
                continue  # Skip already tried
            tried_prompts.add(fallback_prompt.lower())

            print(f"      Trying fallback: '{fallback_prompt}'")
            all_frame_outputs = self._run_segmentation_with_prompt(video_path, fallback_prompt, num_frames)

            if self._check_segmentation_quality(all_frame_outputs):
                print(f"      Fallback '{fallback_prompt}' succeeded!")
                return all_frame_outputs

        # If all fail, return the last attempt (better than nothing)
        print(f"      All prompts failed quality check, using last result")
        return all_frame_outputs

    def _process_frames(self, frames: List[np.ndarray], all_frame_outputs: dict) -> List[np.ndarray]:
        """Process frames: apply mask, crop, center, resize to target resolution."""
        processed = []

        for frame_idx in tqdm(range(len(frames)), desc="Processing"):
            frame = frames[frame_idx]

            if frame_idx in all_frame_outputs:
                outputs = all_frame_outputs[frame_idx]
                # Get first object's binary mask
                out_masks = outputs.get("out_binary_masks", [])
                if len(out_masks) > 0:
                    mask = out_masks[0]
                    # Convert to numpy if tensor
                    if hasattr(mask, "cpu"):
                        mask = mask.cpu().numpy()
                    result = self._center_on_canvas(frame, mask)
                else:
                    result = self._resize_frame(frame)
            else:
                # No mask - just resize
                result = self._resize_frame(frame)

            processed.append(result)

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
        """Crop product using mask and center on black canvas."""
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
        # Save frames locally first
        output_dir = OUTPUT_DIR / barcode
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)

        # Save metadata as JSON
        if metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Upload to Supabase if available
        frames_url = str(output_dir)
        primary_image_url = None

        if self.supabase:
            try:
                frames_url, primary_image_url = self._upload_to_supabase(
                    frames, barcode, product_id, metadata
                )
            except Exception as e:
                print(f"      Warning: Supabase upload failed: {e}")

        return frames_url, primary_image_url

    def _upload_to_supabase(
        self,
        frames: List[np.ndarray],
        barcode: str,
        product_id: str,
        metadata: dict = None,
    ) -> tuple[str, str]:
        """Upload frames and metadata to Supabase Storage with parallel uploads."""
        bucket = "frames"
        folder = product_id  # Always use product_id for consistent storage

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
                    # File already exists - try update
                    if "already exists" in error_str or "duplicate" in error_str:
                        try:
                            self.supabase.storage.from_(bucket).update(
                                path, data, {"content-type": "image/png"}
                            )
                            return idx, True
                        except Exception:
                            pass
                    # Rate limit or server error - wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
            return idx, False

        # Parallel upload with retry for failed frames
        failed_items = list(frame_data)
        success_count = 0
        max_rounds = 3  # Try up to 3 rounds for failed frames

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
                time.sleep(1)  # Wait before retry round

        print(f"      Uploaded {success_count}/{len(frames)} frames")

        # Upload metadata JSON (single file, no need for parallel)
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

        # Get URLs (frame records will be synced in _sync_product_and_frames)
        frames_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/"

        # Primary image is the middle frame (best representation)
        middle_idx = len(frames) // 2
        primary_image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/frame_{middle_idx:04d}.png"

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
        """
        Sync product and frame records to database.

        This is idempotent - safe to retry on failure.
        Uses UPSERT for product, DELETE+INSERT for frames.
        All input fields (video_url, video_id, barcode) are included to ensure
        they are preserved even if product was created elsewhere.
        """
        if not self.supabase:
            print("      Skipping database sync - no supabase client")
            return

        bucket = "frames"
        folder = product_id

        try:
            # 1. UPSERT product (create or update atomically)
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

            # Add video_id if provided
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

            # Store grounding prompt
            grounding = metadata.get("visual_grounding", {})
            if grounding.get("grounding_prompt"):
                product_data["grounding_prompt"] = grounding["grounding_prompt"]

            # Store visibility score (convert to int for DB)
            extraction = metadata.get("extraction_metadata", {})
            if extraction.get("visibility_score") is not None:
                product_data["visibility_score"] = int(extraction["visibility_score"])

            # Store nutrition facts as JSON
            nutrition = metadata.get("nutrition_facts", {})
            if nutrition:
                product_data["nutrition_facts"] = nutrition

            # UPSERT - insert if not exists, update if exists
            self.supabase.table("products").upsert(
                product_data,
                on_conflict="id"
            ).execute()
            print(f"      Product {product_id} synced")

            # 2. Sync frame records (delete old + insert new - idempotent)
            print(f"      Syncing {frame_count} frame records...")

            # Delete existing synthetic frames
            self.supabase.table("product_images").delete().eq(
                "product_id", product_id
            ).eq("image_type", "synthetic").execute()

            # Build and insert new frame records
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

            print(f"      Database sync complete!")

        except Exception as e:
            print(f"      Warning: Database sync failed: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup(self, job_dir: Path):
        """Cleanup temporary files."""
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            print(f"      Warning: Cleanup failed: {e}")
