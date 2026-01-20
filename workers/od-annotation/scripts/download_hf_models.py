#!/usr/bin/env python3
"""
Pre-download HuggingFace models during Docker build.
This eliminates cold start model downloads and speeds up worker startup.

Uses snapshot_download for memory-efficient downloads (no model loading).

NOTE: SAM3 is NOT pre-downloaded here. SAM3 downloads its own weights at runtime
(same behavior as video-segmentation worker). It requires HF_TOKEN at runtime.
"""

import os
import sys
import traceback

# Set cache directories
os.environ['HF_HOME'] = '/app/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/huggingface_cache'
os.environ['HF_HUB_CACHE'] = '/app/huggingface_cache'


def download_florence2():
    """Download Florence-2 Large model (~3GB) - files only, no loading."""
    print("=" * 50)
    print("Downloading Florence-2 Large...")
    print("=" * 50)
    sys.stdout.flush()

    try:
        from huggingface_hub import snapshot_download

        model_name = "microsoft/Florence-2-large"

        print(f"Downloading: {model_name}")
        sys.stdout.flush()

        # Download all files without loading into memory
        snapshot_download(
            repo_id=model_name,
            cache_dir="/app/huggingface_cache",
            local_dir_use_symlinks=False,
        )

        print("Florence-2 Large downloaded successfully!")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"ERROR downloading Florence-2: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def download_sam_hf():
    """Download SAM (HuggingFace version) (~2.5GB) - files only, no loading."""
    print("=" * 50)
    print("Downloading SAM (HuggingFace version)...")
    print("=" * 50)
    sys.stdout.flush()

    try:
        from huggingface_hub import snapshot_download

        model_name = "facebook/sam-vit-huge"

        print(f"Downloading: {model_name}")
        sys.stdout.flush()

        # Download all files without loading into memory
        snapshot_download(
            repo_id=model_name,
            cache_dir="/app/huggingface_cache",
            local_dir_use_symlinks=False,
        )

        print("SAM HuggingFace downloaded successfully!")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"ERROR downloading SAM: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def main():
    print("=" * 50)
    print("HuggingFace Model Pre-Download Script")
    print("=" * 50)
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    print(f"Python: {sys.version}")
    print()
    print("NOTE: SAM3 weights are downloaded at runtime (requires HF_TOKEN)")
    print()
    sys.stdout.flush()

    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/app")
        print(f"Disk space: {free // (1024**3)}GB free of {total // (1024**3)}GB total")
        sys.stdout.flush()
    except Exception as e:
        print(f"Could not check disk space: {e}")
        sys.stdout.flush()

    success = True

    # Download models (files only, no memory loading)
    if not download_florence2():
        success = False
    print()

    if not download_sam_hf():
        success = False

    print()
    print("=" * 50)
    if success:
        print("All HuggingFace models cached successfully!")
    else:
        print("WARNING: Some models failed to download!")
        print("The worker may still function but with slower cold starts.")
    print("=" * 50)
    sys.stdout.flush()

    # Exit with appropriate code
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
