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
                grounding_prompt = self._get_grounding_prompt(metadata)
                print(f"      Prompt: '{grounding_prompt}'")
                all_frame_outputs = self._segment_video_sam3(video_path, grounding_prompt, len(all_frames))
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
                "grounding_prompt": self._get_grounding_prompt(metadata) if self.video_predictor else None,
                "model": "SAM3",
            }

            # 6. Save frames and metadata
            print("\n[6/6] Saving frames and metadata...")
            frames_url, primary_image_url = self._save_frames(
                processed_frames, barcode, product_id, frames_dir, metadata
            )
            print(f"      Frames URL: {frames_url}")
            print(f"      Primary Image: {primary_image_url}")

            # 7. Update product in database (if product_id provided)
            if product_id and self.supabase:
                self._update_product(product_id, metadata, len(processed_frames), frames_url, primary_image_url)

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
4. grounding_prompt must be: "{Brand} {ProductName} {ContainerType}"

### OUTPUT (JSON only, no markdown)
{
  "brand_info": {"brand_name": "", "sub_brand": "", "manufacturer_country": ""},
  "product_identity": {"product_name": "", "variant_flavor": "", "product_category": "", "container_type": ""},
  "specifications": {"net_quantity_text": "", "pack_configuration": {"type": "", "item_count": 1}, "identifiers": {"barcode": null, "sku_model_code": null}},
  "marketing_and_claims": {"claims_list": [], "marketing_description": ""},
  "nutrition_facts": {"serving_size": "", "calories": null, "total_fat": "", "protein": "", "carbohydrates": "", "sugar": ""},
  "visual_grounding": {"grounding_prompt": "Brand ProductName ContainerType"},
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

    def _get_grounding_prompt(self, metadata: dict) -> str:
        """Get grounding prompt from metadata - prefer container_type."""
        # Primary: use container_type directly (most reliable for SAM3)
        container = metadata.get("product_identity", {}).get("container_type", "")
        if container and container.strip().lower() not in ["", "product", "unknown", "n/a"]:
            return container.strip().lower()

        # Secondary: try grounding_prompt from Gemini
        prompt = metadata.get("visual_grounding", {}).get("grounding_prompt", "")
        invalid_prompts = ["Brand ProductName ContainerType", "", None, "product"]
        if prompt and prompt.strip() and prompt.strip() not in invalid_prompts:
            # Extract just the container type from the prompt if it contains one
            prompt_lower = prompt.strip().lower()
            for fallback in self.FALLBACK_PROMPTS:
                if fallback in prompt_lower:
                    return fallback
            return prompt.strip()

        # Default fallback
        return "can"

    def _test_prompt_on_first_frame(self, video_path: Path, prompt: str) -> int:
        """Test a prompt on the first frame and return mask pixel count."""
        response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=str(video_path))
        )
        session_id = response["session_id"]

        try:
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )

            propagate_result = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )

            for item in propagate_result:
                if isinstance(item, dict):
                    outputs = item.get("outputs", {})
                    masks = outputs.get("out_binary_masks", [])
                    if len(masks) > 0:
                        mask = masks[0]
                        if hasattr(mask, "cpu"):
                            mask = mask.cpu().numpy()
                        return int(mask.sum())
                break
            return 0

        finally:
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )

    def _find_best_prompt(self, video_path: Path, primary_prompt: str) -> str:
        """Find the best working prompt by testing on first frame."""
        # First try the primary prompt
        print(f"      Testing primary prompt: '{primary_prompt}'")
        pixel_count = self._test_prompt_on_first_frame(video_path, primary_prompt)
        print(f"      -> {pixel_count} pixels")

        if pixel_count >= self.MIN_MASK_PIXELS:
            return primary_prompt

        # Try fallback prompts
        print("      Primary prompt failed, trying fallbacks...")
        for fallback in self.FALLBACK_PROMPTS:
            if fallback == primary_prompt:
                continue
            print(f"      Testing fallback: '{fallback}'")
            pixel_count = self._test_prompt_on_first_frame(video_path, fallback)
            print(f"      -> {pixel_count} pixels")

            if pixel_count >= self.MIN_MASK_PIXELS:
                print(f"      Found working prompt: '{fallback}'")
                return fallback

        # If nothing works, return primary prompt anyway
        print(f"      No fallback worked, using primary: '{primary_prompt}'")
        return primary_prompt

    def _segment_video_sam3(self, video_path: Path, grounding_prompt: str, num_frames: int) -> dict:
        """Segment video using SAM3 with text prompt and fallback support."""
        # Find the best working prompt
        best_prompt = self._find_best_prompt(video_path, grounding_prompt)

        # Start session for full segmentation
        print(f"      Starting SAM3 session with prompt: '{best_prompt}'")
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
                    text=best_prompt,
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
            # Close session
            print("      Closing SAM3 session...")
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

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

        def upload_single_frame(item: Tuple[int, bytes]) -> Tuple[int, bool]:
            """Upload a single frame."""
            idx, data = item
            path = f"{folder}/frame_{idx:04d}.png"
            try:
                self.supabase.storage.from_(bucket).upload(
                    path, data, {"content-type": "image/png"}
                )
                return idx, True
            except Exception as e:
                # File might already exist, try update
                if "already exists" in str(e).lower() or "Duplicate" in str(e):
                    try:
                        self.supabase.storage.from_(bucket).update(
                            path, data, {"content-type": "image/png"}
                        )
                        return idx, True
                    except Exception:
                        return idx, False
                return idx, False

        # Parallel upload
        success_count = 0
        with ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS) as executor:
            futures = [executor.submit(upload_single_frame, item) for item in frame_data]
            for future in as_completed(futures):
                idx, success = future.result()
                if success:
                    success_count += 1

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

        # Insert frame records into product_images table
        print(f"      [DEBUG] About to call _insert_frame_records...")
        self._insert_frame_records(product_id, len(frames), bucket, folder)
        print(f"      [DEBUG] _insert_frame_records completed")

        # Get URLs
        frames_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/"

        # Primary image is the middle frame (best representation)
        middle_idx = len(frames) // 2
        primary_image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{folder}/frame_{middle_idx:04d}.png"

        return frames_url, primary_image_url

    def _insert_frame_records(
        self,
        product_id: str,
        frame_count: int,
        bucket: str,
        folder: str,
    ):
        """Insert frame records into product_images table."""
        print(f"      [DEBUG] _insert_frame_records called: product_id={product_id}, frame_count={frame_count}")
        print(f"      [DEBUG] supabase client exists: {self.supabase is not None}")

        if not self.supabase:
            print("      [DEBUG] Skipping frame records - no supabase client")
            return

        try:
            # First, delete any existing synthetic frames for this product
            # (in case of re-processing)
            print(f"      [DEBUG] Deleting existing synthetic frames for product_id={product_id}")
            delete_result = self.supabase.table("product_images").delete().eq(
                "product_id", product_id
            ).eq("image_type", "synthetic").execute()
            print(f"      [DEBUG] Delete result: {delete_result}")

            # Insert new frame records
            print(f"      [DEBUG] Building {frame_count} frame records...")
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
                # Batch insert
                print(f"      [DEBUG] Inserting {len(records)} records...")
                insert_result = self.supabase.table("product_images").insert(records).execute()
                print(f"      [DEBUG] Insert result count: {len(insert_result.data) if insert_result.data else 0}")
                print(f"      Inserted {len(records)} frame records into database")
            else:
                print("      [DEBUG] No records to insert!")

        except Exception as e:
            print(f"      Warning: Failed to insert frame records: {e}")
            import traceback
            traceback.print_exc()

    def _update_product(
        self,
        product_id: str,
        metadata: dict,
        frame_count: int,
        frames_url: str,
        primary_image_url: str,
    ):
        """Update product record in database."""
        if not self.supabase:
            return

        try:
            update_data = {
                "frame_count": frame_count,
                "frames_path": frames_url,
                "primary_image_url": primary_image_url,
                "status": "ready",
            }

            # Extract fields from metadata
            brand_info = metadata.get("brand_info", {})
            product_identity = metadata.get("product_identity", {})
            specs = metadata.get("specifications", {})

            if brand_info.get("brand_name"):
                update_data["brand_name"] = brand_info["brand_name"]
            if brand_info.get("sub_brand"):
                update_data["sub_brand"] = brand_info["sub_brand"]
            if product_identity.get("product_name"):
                update_data["product_name"] = product_identity["product_name"]
            if product_identity.get("variant_flavor"):
                update_data["variant_flavor"] = product_identity["variant_flavor"]
            if product_identity.get("product_category"):
                update_data["category"] = product_identity["product_category"]
            if product_identity.get("container_type"):
                update_data["container_type"] = product_identity["container_type"]
            if specs.get("net_quantity_text"):
                update_data["net_quantity"] = specs["net_quantity_text"]

            # Store grounding prompt
            grounding = metadata.get("visual_grounding", {})
            if grounding.get("grounding_prompt"):
                update_data["grounding_prompt"] = grounding["grounding_prompt"]

            # Store visibility score
            extraction = metadata.get("extraction_metadata", {})
            if extraction.get("visibility_score") is not None:
                update_data["visibility_score"] = extraction["visibility_score"]

            # Store nutrition facts as JSON
            nutrition = metadata.get("nutrition_facts", {})
            if nutrition:
                update_data["nutrition_facts"] = nutrition

            # Update in database
            self.supabase.table("products").update(update_data).eq("id", product_id).execute()
            print(f"      Updated product {product_id} in database")

        except Exception as e:
            print(f"      Warning: Failed to update product: {e}")

    def _cleanup(self, job_dir: Path):
        """Cleanup temporary files."""
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            print(f"      Warning: Cleanup failed: {e}")
