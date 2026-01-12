"""
Runpod Serverless Handler for Augmentation.
Based on: final_augmentor_v3.py

INTEGRATED WITH SUPABASE:
- Receives dataset_id from API
- Downloads images from Supabase Storage
- Processes locally with all optimizations
- Uploads results back to Supabase Storage
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
from augmentor import ProductAugmentor
from supabase_client import DatasetDownloader, ResultUploader

# Singletons - loaded once on cold start
augmentor = None
downloader = None
uploader = None


def get_augmentor():
    """Get or create augmentor singleton."""
    global augmentor
    if augmentor is None:
        print("=" * 60)
        print("COLD START - Loading Augmentor...")
        print("=" * 60)
        augmentor = ProductAugmentor()
        print("Augmentor ready!")
        print("=" * 60)
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

        # Optional parameters with defaults (from original code)
        syn_target = job_input.get("syn_target", 600)
        real_target = job_input.get("real_target", 400)
        backgrounds_path = job_input.get("backgrounds_path")

        print(f"\n{'=' * 60}")
        print(f"JOB ID: {job_id}")
        print(f"Dataset ID: {dataset_id or 'N/A (using local path)'}")
        print(f"SYN target: {syn_target}, REAL target: {real_target}")
        print(f"{'=' * 60}\n")

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
        # STEP 2: PROCESS DATASET
        # ========================================
        aug = get_augmentor()

        if dataset_id:
            get_uploader().update_job_progress(job_id, 30, "Augmenting images")

        result = aug.process_dataset(
            dataset_path=str(local_dataset_path),
            syn_target=syn_target,
            real_target=real_target,
            backgrounds_path=backgrounds_path,
        )

        # ========================================
        # STEP 3: UPLOAD RESULTS (if using Supabase)
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
        }

        # Send callback if configured
        if dataset_id:
            get_uploader().send_callback(final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(error_trace)

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


if __name__ == "__main__":
    print("Starting Augmentation Worker...")
    runpod.serverless.start({"handler": handler})
