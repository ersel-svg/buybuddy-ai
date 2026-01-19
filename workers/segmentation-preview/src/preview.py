"""
Preview Pipeline for Single-Frame Segmentation

Downloads video, extracts first frame, runs SAM3 with provided prompts,
and returns the mask image for preview.
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
import tempfile

import torch
import numpy as np
import cv2
from PIL import Image
import requests

# HuggingFace token for SAM3 model download
HF_TOKEN = os.environ.get("HF_TOKEN", "")


class PreviewPipeline:
    """Pipeline for generating segmentation previews on a single frame."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "segmentation-preview"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SAM3
        print("[Preview] Loading SAM3...")
        self._init_sam3()
        print("[Preview] SAM3 ready!")

    def _init_sam3(self):
        """Initialize SAM3 video predictor."""
        try:
            # Authenticate with HuggingFace for model download
            if HF_TOKEN:
                from huggingface_hub import login
                login(token=HF_TOKEN, add_to_git_credential=False)
                print("[Preview] HuggingFace authenticated")
            else:
                print("[Preview] WARNING: HF_TOKEN not set, SAM3 download may fail")

            from sam3.model_builder import build_sam3_video_predictor
            self.video_predictor = build_sam3_video_predictor()
            print("[Preview] SAM3 loaded successfully")

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                print(f"[Preview] GPU Memory used: {mem:.1f} GB")
        except ImportError as e:
            print(f"[Preview] SAM3 import error: {e}")
            raise RuntimeError("SAM3 not available")
        except Exception as e:
            print(f"[Preview] SAM3 load error: {e}")
            raise

    def preview(
        self,
        video_url: str,
        text_prompts: List[str] = None,
        points: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate segmentation preview for first frame.

        Args:
            video_url: URL to download video from
            text_prompts: List of text prompts for SAM3
            points: List of point prompts [{x, y, label}]

        Returns:
            {
                "mask_image": base64-encoded PNG,
                "mask_stats": {pixel_count, coverage_percent, width, height}
            }
        """
        text_prompts = text_prompts or []
        points = points or []

        # Create temp directory for this job
        job_id = os.urandom(8).hex()
        job_dir = self.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download video
            print("[Preview] Downloading video...")
            video_path = self._download_video(video_url, job_dir / "video.mp4")

            # 2. Get video dimensions for coordinate denormalization
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"[Preview] Video dimensions: {width}x{height}")

            # 3. Start SAM3 session
            print("[Preview] Starting SAM3 session...")
            response = self.video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(video_path),
                )
            )
            session_id = response["session_id"]
            print(f"[Preview] Session ID: {session_id}")

            try:
                # 4. Add text prompt OR use generic prompt for point-only mode
                if text_prompts:
                    prompt = text_prompts[0]
                    print(f"[Preview] Adding text prompt: '{prompt}'")
                    self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text=prompt,
                        )
                    )
                elif points:
                    # For point-only prompts, we need to initialize with a generic prompt first
                    # SAM3 requires frame cache before adding point refinements
                    print("[Preview] Initializing with generic prompt for point-based segmentation...")
                    self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            text="object",  # Generic prompt to initialize tracking
                        )
                    )

                # 5. Different flow for text-only vs point prompts
                propagate_result = None

                if points:
                    # For point prompts: need initial propagation to build cache, then add points
                    print("[Preview] Running initial propagation for point refinement...")
                    initial_propagate = self.video_predictor.propagate_in_video(
                        session_id=session_id,
                        propagation_direction="forward",
                        start_frame_idx=0,
                        max_frame_num_to_track=1,
                    )
                    # Consume the generator to execute propagation and build cache
                    for _ in initial_propagate:
                        pass

                    # Add point prompts (refinement after initial propagation)
                    print(f"[Preview] Adding {len(points)} point prompts for refinement...")
                    # Denormalize coordinates from 0-1 to pixel coordinates
                    points_list = [[p["x"] * width, p["y"] * height] for p in points]
                    labels_list = [p.get("label", 1) for p in points]

                    points_tensor = torch.tensor(points_list, dtype=torch.float32)
                    labels_tensor = torch.tensor(labels_list, dtype=torch.int32)

                    self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            obj_id=1,  # Required for point prompts
                            points=points_tensor,
                            point_labels=labels_tensor,
                        )
                    )

                    # Final propagation to get refined mask
                    print("[Preview] Running final segmentation with points...")
                    propagate_result = self.video_predictor.propagate_in_video(
                        session_id=session_id,
                        propagation_direction="forward",
                        start_frame_idx=0,
                        max_frame_num_to_track=1,
                    )
                else:
                    # For text-only: single propagation is enough
                    print("[Preview] Running segmentation...")
                    propagate_result = self.video_predictor.propagate_in_video(
                        session_id=session_id,
                        propagation_direction="forward",
                        start_frame_idx=0,
                        max_frame_num_to_track=1,
                    )

                # 6. Extract mask from result
                mask = None
                if hasattr(propagate_result, "__iter__") and not isinstance(propagate_result, (dict, str)):
                    for item in propagate_result:
                        if isinstance(item, dict):
                            outputs = item.get("outputs", item)
                            masks = outputs.get("out_binary_masks")
                            if masks is not None and hasattr(masks, 'shape') and masks.shape[0] > 0:
                                mask = masks[0]
                                break
                        elif isinstance(item, tuple) and len(item) > 1:
                            outputs = item[1]
                            if isinstance(outputs, dict):
                                masks = outputs.get("out_binary_masks")
                                if masks is not None and hasattr(masks, 'shape') and masks.shape[0] > 0:
                                    mask = masks[0]
                                    break

                if mask is None:
                    raise ValueError("No mask generated - the prompt didn't match any object in the video. Try a different prompt or add point prompts.")

                # Convert mask to numpy
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                mask = mask.squeeze().astype(np.uint8)

                # 8. Create overlay image
                print("[Preview] Creating mask overlay...")
                mask_image = self._create_mask_overlay(mask, width, height)

                # 9. Calculate stats
                pixel_count = int(mask.sum())
                total_pixels = width * height
                coverage_percent = round((pixel_count / total_pixels) * 100, 2)

                print(f"[Preview] Mask: {pixel_count} pixels ({coverage_percent}% coverage)")

                return {
                    "mask_image": mask_image,
                    "mask_stats": {
                        "pixel_count": pixel_count,
                        "coverage_percent": coverage_percent,
                        "width": width,
                        "height": height,
                    }
                }

            finally:
                # Close SAM3 session
                print("[Preview] Closing SAM3 session...")
                self.video_predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            # Cleanup temp files
            self._cleanup(job_dir)

    def _download_video(self, url: str, save_path: Path) -> Path:
        """Download video from URL."""
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return save_path

    def _create_mask_overlay(self, mask: np.ndarray, width: int, height: int) -> str:
        """Create semi-transparent mask overlay as base64 PNG."""
        # Resize mask if needed
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Create RGBA image (green overlay with alpha)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[mask > 0] = [0, 255, 0, 150]  # Green with 60% opacity

        # Convert to PIL and encode as base64 PNG
        img = Image.fromarray(overlay, mode="RGBA")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _cleanup(self, job_dir: Path):
        """Cleanup temporary files."""
        try:
            import shutil
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except Exception as e:
            print(f"[Preview] Warning: cleanup failed: {e}")
