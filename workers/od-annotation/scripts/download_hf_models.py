#!/usr/bin/env python3
"""
Pre-download HuggingFace models during Docker build.
This eliminates cold start model downloads and speeds up worker startup.
"""

import os

# Set cache directories
os.environ['HF_HOME'] = '/app/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/app/huggingface_cache'

def download_florence2():
    """Download Florence-2 Large model (~3GB)."""
    print("=" * 50)
    print("Downloading Florence-2 Large...")
    print("=" * 50)

    from transformers import AutoModelForCausalLM, AutoProcessor

    model_name = "microsoft/Florence-2-large"

    # Download processor
    print(f"Downloading processor: {model_name}")
    AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Download model
    print(f"Downloading model: {model_name}")
    AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    print("Florence-2 Large downloaded successfully!")


def download_sam_hf():
    """Download SAM (HuggingFace version) (~2.5GB)."""
    print("=" * 50)
    print("Downloading SAM (HuggingFace version)...")
    print("=" * 50)

    from transformers import SamModel, SamProcessor

    model_name = "facebook/sam-vit-huge"

    # Download processor
    print(f"Downloading processor: {model_name}")
    SamProcessor.from_pretrained(model_name)

    # Download model
    print(f"Downloading model: {model_name}")
    SamModel.from_pretrained(model_name)

    print("SAM HuggingFace downloaded successfully!")


def main():
    print("=" * 50)
    print("HuggingFace Model Pre-Download Script")
    print("=" * 50)
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
    print()

    # Download models
    download_florence2()
    print()
    download_sam_hf()

    print()
    print("=" * 50)
    print("All HuggingFace models cached successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
