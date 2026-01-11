"""
Product Pipeline

Video → Gemini (metadata) → SAM3 (segmentation) → 518x518 Frames
"""

import os
import json
import time
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image
import requests
import google.generativeai as genai
from tqdm import tqdm

from config import (
    GEMINI_API_KEY, HF_TOKEN, SUPABASE_URL, SUPABASE_KEY,
    TARGET_RESOLUTION, GEMINI_MODEL, TEMP_DIR, OUTPUT_DIR, DEVICE
)


class ProductPipeline:
    """
    Main pipeline for processing product videos.
    
    1. Download video
    2. Extract metadata with Gemini
    3. Segment with SAM3
    4. Post-process frames to 518x518
    5. Upload to storage
    """
    
    def __init__(self):
        self.device = DEVICE
        self.target_resolution = TARGET_RESOLUTION
        self.temp_dir = TEMP_DIR
        
        # Initialize Gemini
        print("[Pipeline] Configuring Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print("[Pipeline] Gemini ready!")
        
        # Initialize SAM3
        print("[Pipeline] Loading SAM3 (this may take a minute)...")
        from sam3.model_builder import build_sam3_video_predictor
        self.video_predictor = build_sam3_video_predictor()
        print("[Pipeline] SAM3 ready!")
        
        # Initialize Supabase (optional)
        self.supabase = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                from supabase import create_client
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("[Pipeline] Supabase connected!")
            except Exception as e:
                print(f"[Pipeline] Supabase not available: {e}")
    
    def process(
        self,
        video_url: str,
        barcode: str,
        video_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main processing method.
        
        Returns:
            {
                "metadata": {...},
                "frame_count": 177,
                "frames_url": "https://..."
            }
        """
        # Create temp directory for this job
        job_dir = self.temp_dir / barcode
        job_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Download video
            print("\n[1/5] Downloading video...")
            video_path = self._download_video(video_url, job_dir / "video.mp4")
            print(f"      Saved to: {video_path}")
            
            # 2. Extract frames
            print("\n[2/5] Extracting frames...")
            all_frames = self._extract_frames(video_path)
            print(f"      Extracted {len(all_frames)} frames")
            
            # 3. Gemini metadata extraction
            print("\n[3/5] Extracting metadata with Gemini...")
            metadata = self._extract_metadata(video_path)
            print(f"      Brand: {metadata.get('brand_info', {}).get('brand_name', 'N/A')}")
            print(f"      Product: {metadata.get('product_identity', {}).get('product_name', 'N/A')}")
            
            # 4. SAM3 segmentation
            print("\n[4/5] Segmenting with SAM3...")
            grounding_prompt = self._get_grounding_prompt(metadata)
            print(f"      Prompt: '{grounding_prompt}'")
            masks = self._segment_video(video_path, grounding_prompt, len(all_frames))
            print(f"      Segmented {len(masks)} frames")
            
            # 5. Post-process frames
            print("\n[5/5] Post-processing frames...")
            processed_frames = self._process_frames(all_frames, masks)
            print(f"      Processed {len(processed_frames)} frames to {self.target_resolution}x{self.target_resolution}")
            
            # Add pipeline info to metadata (before saving)
            metadata['_pipeline'] = {
                'barcode': barcode,
                'video_id': video_id,
                'timestamp': datetime.now().isoformat(),
                'frame_count': len(processed_frames),
                'grounding_prompt': grounding_prompt
            }

            # 6. Save frames and metadata
            print("\n[6/6] Saving frames and metadata...")
            frames_url = self._save_frames(processed_frames, barcode, frames_dir, metadata)
            print(f"      Saved to: {frames_url}")
            
            return {
                "metadata": metadata,
                "frame_count": len(processed_frames),
                "frames_url": frames_url
            }
            
        finally:
            # Cleanup
            self._cleanup(job_dir)
    
    def _download_video(self, url: str, save_path: Path) -> Path:
        """Download video from URL."""
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    
    def _extract_frames(self, video_path: Path) -> List[np.ndarray]:
        """Extract all frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(total_frames), desc="Extracting"):
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def _extract_metadata(self, video_path: Path) -> dict:
        """Extract metadata using Gemini."""
        # Upload video
        video_file = genai.upload_file(path=str(video_path))
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError("Gemini video processing failed")
        
        # Generate
        prompt = self._get_extraction_prompt()
        response = self.gemini_model.generate_content([video_file, prompt])
        
        # Parse JSON
        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
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
    
    def _get_grounding_prompt(self, metadata: dict) -> str:
        """Get grounding prompt from metadata or generate fallback."""
        # Try to get from Gemini output
        prompt = metadata.get('visual_grounding', {}).get('grounding_prompt', '')

        # Check if it's a valid prompt (not the template placeholder)
        invalid_prompts = ['Brand ProductName ContainerType', '', None]
        if prompt and prompt.strip() and prompt.strip() not in invalid_prompts:
            return prompt.strip()

        # Fallback: build from fields
        brand = metadata.get('brand_info', {}).get('brand_name', '') or ''
        product = metadata.get('product_identity', {}).get('product_name', '') or ''
        container = metadata.get('product_identity', {}).get('container_type', 'product') or 'product'

        prompt = f"{brand} {product} {container}".strip()

        return prompt if prompt else "product"
    
    def _segment_video(self, video_path: Path, grounding_prompt: str, num_frames: int) -> dict:
        """Segment video using SAM3 with text prompt."""
        # Start session
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path)
            )
        )
        session_id = response["session_id"]
        
        try:
            # Add text prompt
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=grounding_prompt,
                )
            )
            
            # Propagate through video
            all_outputs = {}
            
            for item in tqdm(
                self.video_predictor.propagate_in_video(
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_idx=0,
                    max_frame_num_to_track=num_frames
                ),
                total=num_frames,
                desc="Segmenting"
            ):
                if isinstance(item, dict):
                    frame_idx = item.get('frame_index', len(all_outputs))
                    all_outputs[frame_idx] = item.get('outputs', item)
            
            return all_outputs
            
        finally:
            # Close session
            self.video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
    
    def _process_frames(self, frames: List[np.ndarray], masks: dict) -> List[np.ndarray]:
        """Process frames: crop, center, resize to 518x518."""
        processed = []
        
        for idx in tqdm(range(len(frames)), desc="Processing"):
            frame = frames[idx]
            
            if idx in masks:
                out_masks = masks[idx].get('out_binary_masks', [])
                mask = out_masks[0] if len(out_masks) > 0 else None
                if mask is not None:
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    result = self._center_on_canvas(frame, mask)
                else:
                    result = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
            else:
                result = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
            
            processed.append(result)
        
        return processed
    
    def _center_on_canvas(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop product using mask and center on black canvas."""
        # Resize mask if needed
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0])
            ).astype(bool)
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Crop
        cropped = frame[y_min:y_max+1, x_min:x_max+1].copy()
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
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
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        return canvas
    
    def _save_frames(self, frames: List[np.ndarray], barcode: str, frames_dir: Path, metadata: dict = None) -> str:
        """Save frames and metadata to storage."""
        # Create output directory with barcode
        output_dir = OUTPUT_DIR / barcode
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save frames locally (with barcode folder structure)
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(frame_path)

        # Save metadata as JSON
        if metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"      Metadata saved to: {metadata_path}")

        # Upload to Supabase if available
        if self.supabase:
            try:
                return self._upload_to_supabase(frames, barcode, metadata)
            except Exception as e:
                print(f"      Warning: Supabase upload failed: {e}")

        # Return local path
        return str(output_dir)
    
    def _upload_to_supabase(self, frames: List[np.ndarray], barcode: str, metadata: dict = None) -> str:
        """Upload frames and metadata to Supabase Storage."""
        bucket = "frames"
        from io import BytesIO

        # Upload frames
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            path = f"{barcode}/frame_{i:04d}.png"
            self.supabase.storage.from_(bucket).upload(path, buffer.getvalue())

        # Upload metadata JSON
        if metadata:
            metadata_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode('utf-8')
            metadata_path = f"{barcode}/metadata.json"
            self.supabase.storage.from_(bucket).upload(metadata_path, metadata_bytes)

        # Return public URL
        return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{barcode}/"
    
    def _cleanup(self, job_dir: Path):
        """Cleanup temporary files."""
        import shutil
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            print(f"      Warning: Cleanup failed: {e}")
