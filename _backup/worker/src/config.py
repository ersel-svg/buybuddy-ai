"""Configuration for the worker."""

import os
from pathlib import Path

# API Keys (from environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Supabase (for storage)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Processing settings
TARGET_RESOLUTION = 518  # DINOv2 optimal size
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Paths
TEMP_DIR = Path("/tmp/pipeline")
TEMP_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("/workspace/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
