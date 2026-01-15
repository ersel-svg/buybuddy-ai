"""Configuration for Video Segmentation Worker."""

import os
from pathlib import Path

# ===========================================
# API Keys (from environment)
# ===========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ===========================================
# Supabase Configuration
# ===========================================

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")  # Service role key

# ===========================================
# Backend Callback
# ===========================================

CALLBACK_URL = os.environ.get("CALLBACK_URL", "")  # e.g., https://api.buybuddy.ai

# ===========================================
# Processing Settings
# ===========================================

TARGET_RESOLUTION = 518  # DINOv2 optimal input size
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")  # Default to unlimited tier

# Frame extraction settings (same as notebook)
# MAX_FRAMES: Maximum frames to extract (None = all frames)
# FRAME_SAMPLE_RATE: Extract every Nth frame (1 = every frame, 2 = every 2nd frame)
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0")) or None  # 0 means None (all frames)
FRAME_SAMPLE_RATE = int(os.environ.get("FRAME_SAMPLE_RATE", "1"))  # 1 = every frame

# ===========================================
# Paths
# ===========================================

TEMP_DIR = Path("/tmp/pipeline")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_DIR = Path("/workspace/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ===========================================
# Device
# ===========================================

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Config] Device: {DEVICE}")
print(f"[Config] Gemini Model: {GEMINI_MODEL}")
print(f"[Config] Target Resolution: {TARGET_RESOLUTION}x{TARGET_RESOLUTION}")
