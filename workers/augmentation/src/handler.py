"""
Runpod Serverless Handler for Augmentation.
Based on: final_augmentor.py - Full shelf scene composition

INTEGRATED WITH SUPABASE:
- Receives dataset_id from API
- Downloads images from Supabase Storage
- Downloads backgrounds from Supabase Storage
- Processes locally with full shelf scene composition
- Uploads results back to Supabase Storage
- Supports configurable augmentation settings
"""

import os
# CRITICAL: Thread limiting for stability
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import runpod
import json
import traceback
import shutil
from pathlib import Path
from augmentor import ProductAugmentor, AugmentationConfig
from supabase_client import (
    DatasetDownloader,
    ResultUploader,
    BackgroundDownloader,
    NeighborProductDownloader,
)

# Singletons - loaded once on cold start
augmentor = None
downloader = None
uploader = None
background_downloader = None
neighbor_downloader = None


def get_augmentor(config: AugmentationConfig = None):
    """Get or create augmentor singleton with optional config."""
    global augmentor
    if augmentor is None:
        print("=" * 60)
        print("COLD START - Loading Augmentor...")
        print("=" * 60)
        augmentor = ProductAugmentor(config=config)
        print("Augmentor ready!")
        print("=" * 60)
    elif config is not None:
        # Update config if provided
        augmentor.config = config
        augmentor.transforms = augmentor._get_transforms()
    return augmentor


def get_downloader():
    """Get or create downloader singleton."""
    global downloader
    if downloader is None:
        downloader = DatasetDownloader()
    return downloader


def get_uploader():
    """Get or create uploader singleton."""
    global uploader
    if uploader is None:
        uploader = ResultUploader()
    return uploader


def get_background_downloader():
    """Get or create background downloader singleton."""
    global background_downloader
    if background_downloader is None:
        background_downloader = BackgroundDownloader()
    return background_downloader


def get_neighbor_downloader():
    """Get or create neighbor downloader singleton."""
    global neighbor_downloader
    if neighbor_downloader is None:
        neighbor_downloader = NeighborProductDownloader()
    return neighbor_downloader


def parse_augmentation_config(config_dict: dict) -> AugmentationConfig:
    """
    Parse augmentation config from API request.

    Supports presets: 'clean', 'normal', 'realistic', 'extreme', 'custom'
    """
    if not config_dict:
        return AugmentationConfig()

    preset = config_dict.get("preset", "normal")

    # Start with default config
    config = AugmentationConfig()

    # Apply preset modifications
    if preset == "clean":
        config.PROB_HEAVY_AUGMENTATION = 0.1
        config.PROB_NEIGHBORING_PRODUCTS = 0.0
        config.PROB_SHADOW = 0.2
        config.PROB_LENS_DISTORTION = 0.0
        config.PROB_FLUORESCENT_BANDING = 0.1
        config.PROB_HSV_SHIFT = 0.2
        config.PROB_RGB_SHIFT = 0.1
        config.PROB_CAMERA_NOISE = 0.2

    elif preset == "realistic":
        config.PROB_HEAVY_AUGMENTATION = 0.8
        config.PROB_NEIGHBORING_PRODUCTS = 0.6
        config.PROB_SHADOW = 0.8
        config.PROB_LENS_DISTORTION = 0.5
        config.PROB_CAMERA_NOISE = 0.7
        config.MAX_NEIGHBORS = 3

    elif preset == "extreme":
        config.PROB_HEAVY_AUGMENTATION = 0.9
        config.PROB_NEIGHBORING_PRODUCTS = 0.8
        config.PROB_SHADOW = 0.9
        config.PROB_LENS_DISTORTION = 0.6
        config.PROB_CAMERA_NOISE = 0.8
        config.MAX_NEIGHBORS = 3
        config.HSV_HUE_LIMIT = 12
        config.HSV_SAT_LIMIT = 25
        config.HSV_VAL_LIMIT = 20
        config.RGB_SHIFT_LIMIT = 12

    elif preset == "custom":
        # Apply all custom settings from config_dict
        pass

    # Override with any explicit values from config_dict
    for key in [
        "PROB_HEAVY_AUGMENTATION",
        "PROB_NEIGHBORING_PRODUCTS",
        "PROB_TIPPED_OVER_NEIGHBOR",
        "PROB_PRICE_TAG",
        "PROB_SHELF_RAIL",
        "PROB_CAMPAIGN_STICKER",
        "PROB_FLUORESCENT_BANDING",
        "PROB_COLOR_TRANSFER",
        "PROB_SHELF_REFLECTION",
        "PROB_SHADOW",
        "PROB_PERSPECTIVE_CHANGE",
        "PROB_LENS_DISTORTION",
        "PROB_CHROMATIC_ABERRATION",
        "PROB_CAMERA_NOISE",
        "PROB_CONDENSATION",
        "PROB_FROST_CRYSTALS",
        "PROB_COLD_COLOR_FILTER",
        "PROB_WIRE_RACK",
        "PROB_HSV_SHIFT",
        "PROB_RGB_SHIFT",
        "PROB_MEDIAN_BLUR",
        "PROB_ISO_NOISE",
        "PROB_CLAHE",
        "PROB_SHARPEN",
        "PROB_HORIZONTAL_FLIP",
        "MIN_NEIGHBORS",
        "MAX_NEIGHBORS",
        "HSV_HUE_LIMIT",
        "HSV_SAT_LIMIT",
        "HSV_VAL_LIMIT",
        "RGB_SHIFT_LIMIT",
    ]:
        if key in config_dict:
            setattr(config, key, config_dict[key])

    # Handle tuple values
    for key in [
        "COLOR_TRANSFER_STRENGTH",
        "SHADOW_OPACITY",
        "SHADOW_BLUR_RADIUS",
        "SHADOW_OFFSET",
        "REFLECTION_OPACITY",
        "CONDENSATION_OPACITY",
    ]:
        if key in config_dict:
            value = config_dict[key]
            if isinstance(value, list):
                setattr(config, key, tuple(value))
            else:
                setattr(config, key, value)

    return config


def handler(job):
    """Main handler for Runpod serverless."""
    job_id = job.get("id", "unknown")
    local_dataset_path = None

    try:
        job_input = job.get("input", {})

        # ========================================
        # INPUT PARAMETERS
        # ========================================
        # Primary: dataset_id for Supabase integration
        dataset_id = job_input.get("dataset_id")

        # Fallback: direct dataset_path for local testing
        dataset_path = job_input.get("dataset_path")

        if not dataset_id and not dataset_path:
            return {"status": "error", "error": "dataset_id or dataset_path is required"}

        # Augmentation targets
        syn_target = job_input.get("syn_target", 600)
        real_target = job_input.get("real_target", 400)

        # Local backgrounds path (for testing)
        backgrounds_path = job_input.get("backgrounds_path")

        # Augmentation config (from UI)
        aug_config_dict = job_input.get("augmentation_config", {})

        # Whether to use diversity pyramid (random level selection)
        use_diversity_pyramid = job_input.get("use_diversity_pyramid", True)

        # Whether to include neighbor products in shelf composition
        include_neighbors = job_input.get("include_neighbors", True)

        # Frame interval for angle diversity (1 = use all frames, 20 = every 20th frame)
        frame_interval = job_input.get("frame_interval", 1)

        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job_id}")
        print(f"Dataset ID: {dataset_id or 'N/A (using local path)'}")
        print(f"SYN target: {syn_target}, REAL target: {real_target}")
        print(f"Frame Interval: {frame_interval}")
        print(f"Preset: {aug_config_dict.get('preset', 'normal')}")
        print(f"Use Diversity Pyramid: {use_diversity_pyramid}")
        print(f"Include Neighbors: {include_neighbors}")
        print(f"{'=' * 60}\n")

        # ========================================
        # PARSE CONFIG
        # ========================================
        config = parse_augmentation_config(aug_config_dict)
        print(f"Config: PROB_HEAVY={config.PROB_HEAVY_AUGMENTATION}, "
              f"PROB_NEIGHBORS={config.PROB_NEIGHBORING_PRODUCTS}, "
              f"PROB_SHADOW={config.PROB_SHADOW}")

        # ========================================
        # STEP 1: DOWNLOAD FROM SUPABASE (if needed)
        # ========================================
        if dataset_id:
            dl = get_downloader()
            up = get_uploader()

            up.update_job_progress(job_id, 10, "Downloading dataset from Supabase")
            local_dataset_path = dl.download_dataset(dataset_id)
        else:
            local_dataset_path = Path(dataset_path)

        # ========================================
        # STEP 2: DOWNLOAD BACKGROUNDS FROM SUPABASE
        # ========================================
        aug = get_augmentor(config)

        if not backgrounds_path:
            # Download backgrounds from Supabase
            bg_dl = get_background_downloader()
            up = get_uploader() if dataset_id else None

            if up:
                up.update_job_progress(job_id, 20, "Downloading background images")

            backgrounds = bg_dl.download_backgrounds()
            if backgrounds:
                aug.load_backgrounds_from_pil(backgrounds)
            else:
                print("⚠️ No backgrounds downloaded, using solid colors")
        else:
            # Use local backgrounds path
            aug.load_backgrounds(backgrounds_path)

        # ========================================
        # STEP 3: DOWNLOAD NEIGHBOR PRODUCTS (optional)
        # ========================================
        if include_neighbors and dataset_id:
            neighbor_dl = get_neighbor_downloader()
            up = get_uploader()
            up.update_job_progress(job_id, 25, "Downloading neighbor product images")

            neighbor_paths = neighbor_dl.download_neighbor_images(dataset_id, max_neighbors=50)
            if neighbor_paths:
                aug.set_neighbor_paths(neighbor_paths)
            else:
                print("⚠️ No neighbor images downloaded, skipping neighbor composition")

        # ========================================
        # STEP 4: PROCESS DATASET
        # ========================================
        if dataset_id:
            get_uploader().update_job_progress(job_id, 30, "Augmenting images")

        result = aug.process_dataset(
            dataset_path=str(local_dataset_path),
            syn_target=syn_target,
            real_target=real_target,
            backgrounds_path=None,  # Already loaded
            use_diversity_pyramid=use_diversity_pyramid,
            frame_interval=frame_interval,
        )

        # ========================================
        # STEP 5: UPLOAD RESULTS (if using Supabase)
        # ========================================
        if dataset_id:
            up = get_uploader()
            up.update_job_progress(job_id, 80, "Uploading augmented images")
            upload_stats = up.upload_augmented(local_dataset_path, dataset_id)
            result["upload_stats"] = upload_stats

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: {result['syn_produced']} syn + {result['real_produced']} real generated")
        print(f"{'=' * 60}\n")

        final_result = {
            "status": "success",
            "type": "augmentation",
            "dataset_id": dataset_id,
            "syn_produced": result["syn_produced"],
            "real_produced": result["real_produced"],
            "report": result["report"],
            "config_used": {
                "preset": aug_config_dict.get("preset", "normal"),
                "use_diversity_pyramid": use_diversity_pyramid,
                "include_neighbors": include_neighbors,
                "frame_interval": frame_interval,
            }
        }

        # Update job status to completed and send callback
        if dataset_id:
            up = get_uploader()
            up.update_job_status(job_id, "completed", result=final_result)
            up.send_callback(final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(error_trace)

        # Update job status to failed
        try:
            get_uploader().update_job_status(job_id, "failed", error=error_msg)
        except Exception:
            pass  # Best effort

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }

    finally:
        # Cleanup downloaded files to save disk space
        if local_dataset_path and dataset_id:
            try:
                shutil.rmtree(local_dataset_path.parent, ignore_errors=True)
            except Exception:
                pass

        # Clear background cache to free memory for next job
        try:
            bg_dl = get_background_downloader()
            bg_dl.clear_cache()
            print("   🧹 Background cache cleared")
        except Exception:
            pass

        # Clear neighbor paths from augmentor
        try:
            aug = get_augmentor()
            aug.neighbor_product_paths = []
            aug.backgrounds = []
            print("   🧹 Augmentor caches cleared")
        except Exception:
            pass

        # Clear neighbor files
        try:
            neighbor_dl = get_neighbor_downloader()
            if neighbor_dl.local_base.exists():
                shutil.rmtree(neighbor_dl.local_base, ignore_errors=True)
                neighbor_dl.local_base.mkdir(parents=True, exist_ok=True)
            print("   🧹 Neighbor files cleared")
        except Exception:
            pass


if __name__ == "__main__":
    print("Starting Augmentation Worker...")
    runpod.serverless.start({"handler": handler})
