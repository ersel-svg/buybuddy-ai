"""
Product Augmentor with Full Shelf Scene Composition.
Based on: final_augmentor.py - Full product, no black borders, direct resize

Features:
- BiRefNet segmentation with GPU half-precision
- BATCH INFERENCE for 10x speedup
- Full shelf scene composition (backgrounds, neighbors, shadows, effects)
- Configurable augmentation settings
- DiversityPyramid with 4 preset levels
- Direct resize to TARGET_SIZE (no black borders)
"""

import os
import re
import cv2
import math
import copy
import torch
import random
import inspect
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import albumentations as A
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json

# ==============================
# CONFIGURATION
# ==============================
TARGET_SIZE = (384, 384)
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
BATCH_SIZE = 8  # Optimal for RTX 4090 (24GB)

# Regex patterns for output detection (idempotent)
RE_SYN_OUT = re.compile(r"^syn_(.+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)
RE_REAL_OUT_ANY = re.compile(r"^.+_aug_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def RE_REAL_OUT_FOR_STEM(stem):
    return re.compile(rf"^{re.escape(stem)}_aug_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


def list_images(directory: Path):
    """List all images in directory."""
    imgs = []
    if directory.exists():
        for ext in IMAGE_EXTENSIONS:
            imgs.extend(directory.glob(ext))
    return imgs


def ceil_div(a, b):
    return math.ceil(a / b) if b > 0 else 0


# ==============================
# Albumentations Compatibility Shims
# ==============================
def mk_Downscale(*, scale_min=None, scale_max=None, scale_range=None, p=0.5):
    cls = A.Downscale
    params = inspect.signature(cls).parameters
    kwargs = {}
    if 'scale_range' in params:
        if scale_range is not None:
            kwargs['scale_range'] = scale_range
        elif scale_min is not None and scale_max is not None:
            kwargs['scale_range'] = [scale_min, scale_max]
    else:
        if scale_min is not None and scale_max is not None:
            kwargs['scale_min'] = scale_min
            kwargs['scale_max'] = scale_max
        elif scale_range is not None:
            kwargs['scale_min'], kwargs['scale_max'] = scale_range[0], scale_range[1]
    if 'p' in params:
        kwargs['p'] = p
    return cls(**kwargs)


def mk_ImageCompression(*, quality_lower=None, quality_upper=None, quality_range=None, p=0.2):
    cls = A.ImageCompression
    params = inspect.signature(cls).parameters
    kwargs = {}
    if 'quality_range' in params:
        if quality_range is not None:
            kwargs['quality_range'] = quality_range
        elif quality_lower is not None and quality_upper is not None:
            kwargs['quality_range'] = (quality_lower, quality_upper)
    else:
        if quality_lower is not None and quality_upper is not None:
            kwargs['quality_lower'] = quality_lower
            kwargs['quality_upper'] = quality_upper
        elif quality_range is not None:
            kwargs['quality_lower'], kwargs['quality_upper'] = quality_range[0], quality_range[1]
    if 'p' in params:
        kwargs['p'] = p
    return cls(**kwargs)


def mk_CoarseDropout(*, max_holes=6, max_height=30, max_width=30, min_holes=1, fill_value=220, p=0.3):
    cls = A.CoarseDropout
    params = inspect.signature(cls).parameters
    kwargs = {}
    if 'max_holes' in params:
        kwargs['max_holes'] = max_holes
    if 'min_holes' in params:
        kwargs['min_holes'] = min_holes
    if 'holes_number_range' in params and ('max_holes' not in params or 'min_holes' not in params):
        kwargs['holes_number_range'] = (min_holes, max_holes)
    if 'max_height' in params:
        kwargs['max_height'] = max_height
    if 'max_width' in params:
        kwargs['max_width'] = max_width
    if 'min_height' in params:
        kwargs['min_height'] = 1
    if 'min_width' in params:
        kwargs['min_width'] = 1
    if 'fill_value' in params:
        kwargs['fill_value'] = fill_value
    elif 'mask_fill_value' in params:
        kwargs['mask_fill_value'] = fill_value
    if 'p' in params:
        kwargs['p'] = p
    return cls(**kwargs)


def mk_Resize(h, w, always_apply=False):
    cls = A.Resize
    params = inspect.signature(cls).parameters
    kwargs = {'height': h, 'width': w}
    if 'always_apply' in params:
        kwargs['always_apply'] = always_apply
    return cls(**kwargs)


# ==============================
# AUGMENTATION CONFIG
# ==============================
@dataclass
class AugmentationConfig:
    """All augmentation settings - configurable from UI."""
    TARGET_SIZE: Tuple[int, int] = (384, 384)

    # Transform probabilities
    PROB_HEAVY_AUGMENTATION: float = 0.5
    PROB_NEIGHBORING_PRODUCTS: float = 0.5
    PROB_TIPPED_OVER_NEIGHBOR: float = 0.2

    # Shelf elements
    PROB_PRICE_TAG: float = 0.3
    PROB_SHELF_RAIL: float = 0.4
    PROB_CAMPAIGN_STICKER: float = 0.15

    # Lighting effects
    PROB_FLUORESCENT_BANDING: float = 0.4
    PROB_COLOR_TRANSFER: float = 0.3
    PROB_SHELF_REFLECTION: float = 0.4
    PROB_SHADOW: float = 0.7

    # Camera effects
    PROB_PERSPECTIVE_CHANGE: float = 0.3
    PROB_LENS_DISTORTION: float = 0.4
    PROB_CHROMATIC_ABERRATION: float = 0.0
    PROB_CAMERA_NOISE: float = 0.6

    # Refrigerator effects
    PROB_CONDENSATION: float = 0.0
    PROB_FROST_CRYSTALS: float = 0.15
    PROB_COLD_COLOR_FILTER: float = 0.0
    PROB_WIRE_RACK: float = 0.0

    # Color adjustments
    PROB_HSV_SHIFT: float = 0.5
    PROB_RGB_SHIFT: float = 0.4
    PROB_MEDIAN_BLUR: float = 0.25
    PROB_ISO_NOISE: float = 0.4
    PROB_CLAHE: float = 0.3
    PROB_SHARPEN: float = 0.3
    PROB_HORIZONTAL_FLIP: float = 0.0

    # Effect parameters
    COLOR_TRANSFER_STRENGTH: Tuple[float, float] = (0.5, 0.8)
    SHADOW_OPACITY: Tuple[int, int] = (80, 140)
    SHADOW_BLUR_RADIUS: Tuple[int, int] = (5, 8)
    SHADOW_OFFSET: Tuple[int, int] = (5, 5)
    REFLECTION_OPACITY: Tuple[float, float] = (0.2, 0.4)
    CONDENSATION_OPACITY: Tuple[int, int] = (60, 80)

    # Neighbor settings
    MIN_NEIGHBORS: int = 0
    MAX_NEIGHBORS: int = 2

    # Color shift limits
    HSV_HUE_LIMIT: int = 8
    HSV_SAT_LIMIT: int = 15
    HSV_VAL_LIMIT: int = 12
    RGB_SHIFT_LIMIT: int = 8

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AugmentationConfig':
        """Create config from dictionary (e.g., from API request)."""
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class DiversityPyramid:
    """4 preset augmentation levels with different probabilities."""
    LEVELS = {
        'clean': (0.30, {
            'PROB_HEAVY_AUGMENTATION': 0.1,
            'PROB_NEIGHBORING_PRODUCTS': 0.0,
            'PROB_SHADOW': 0.2,
            'PROB_LENS_DISTORTION': 0.0,
            'PROB_FLUORESCENT_BANDING': 0.1,
            'PROB_CONDENSATION': 0.0,
            'PROB_HSV_SHIFT': 0.2,
            'PROB_RGB_SHIFT': 0.1,
            'PROB_CAMERA_NOISE': 0.2,
            'SHADOW_OPACITY': (40, 70),
            'REFLECTION_OPACITY': (0.1, 0.2)
        }),
        'normal': (0.50, {}),  # Use default config
        'realistic': (0.40, {
            'PROB_HEAVY_AUGMENTATION': 0.8,
            'PROB_NEIGHBORING_PRODUCTS': 0.6,
            'PROB_SHADOW': 0.8,
            'PROB_LENS_DISTORTION': 0.5,
            'PROB_CAMERA_NOISE': 0.7,
            'PROB_CONDENSATION': 0.0,
            'COLOR_TRANSFER_STRENGTH': (0.5, 0.7),
            'SHADOW_OPACITY': (110, 160),
            'REFLECTION_OPACITY': (0.3, 0.5),
            'CONDENSATION_OPACITY': (80, 120),
            'MAX_NEIGHBORS': 3
        }),
        'extreme': (0.15, {
            'PROB_HEAVY_AUGMENTATION': 0.9,
            'PROB_NEIGHBORING_PRODUCTS': 0.8,
            'PROB_SHADOW': 0.9,
            'PROB_LENS_DISTORTION': 0.6,
            'PROB_CAMERA_NOISE': 0.8,
            'PROB_CONDENSATION': 0.0,
            'COLOR_TRANSFER_STRENGTH': (0.6, 0.8),
            'SHADOW_OPACITY': (140, 190),
            'REFLECTION_OPACITY': (0.4, 0.6),
            'CONDENSATION_OPACITY': (100, 140),
            'MAX_NEIGHBORS': 3,
            'SHADOW_BLUR_RADIUS': (7, 10),
            'HSV_HUE_LIMIT': 12,
            'HSV_SAT_LIMIT': 25,
            'HSV_VAL_LIMIT': 20,
            'RGB_SHIFT_LIMIT': 12
        })
    }

    @staticmethod
    def get_random_level_config(base_config: AugmentationConfig = None) -> Tuple[AugmentationConfig, str]:
        """Get random augmentation level config."""
        levels, weights = zip(*[(k, v[0]) for k, v in DiversityPyramid.LEVELS.items()])
        selected = random.choices(levels, weights=weights, k=1)[0]

        if base_config is None:
            new_cfg = AugmentationConfig()
        else:
            new_cfg = copy.deepcopy(base_config)

        settings = DiversityPyramid.LEVELS[selected][1]
        for k, v in settings.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                if isinstance(v[0], int):
                    setattr(new_cfg, k, random.randint(v[0], v[1]))
                else:
                    setattr(new_cfg, k, random.uniform(v[0], v[1]))
            else:
                setattr(new_cfg, k, v)

        return new_cfg, selected


# ==============================
# EFFECT CLASSES
# ==============================
class RealisticLighting:
    @staticmethod
    def add_fluorescent_banding(image):
        """Add fluorescent light banding effect."""
        w, h = image.size
        banded_array = np.array(image).astype(np.float32)
        if h > 40:
            for _ in range(random.randint(max(1, h // 40), max(2, h // 20))):
                y = random.randint(0, h)
                height = random.randint(5, 20)
                intensity = random.uniform(0.9, 1.1)
                color_tint = np.array([random.uniform(0.95, 1.05) for _ in range(3)])
                banded_array[y:y+height, :] = np.clip(
                    banded_array[y:y+height, :] * intensity * color_tint, 0, 255
                )
        return Image.fromarray(banded_array.astype(np.uint8))

    @staticmethod
    def add_shelf_reflection(product, config: AugmentationConfig):
        """Add shelf reflection below product."""
        reflection = product.copy().transpose(Image.FLIP_TOP_BOTTOM)
        reflection = reflection.filter(ImageFilter.GaussianBlur(3))
        w, h = reflection.size
        if h == 0:
            return reflection

        opacity_value = config.REFLECTION_OPACITY
        if isinstance(opacity_value, (list, tuple)):
            opacity = random.uniform(opacity_value[0], opacity_value[1])
        else:
            opacity = opacity_value

        gradient = np.linspace(255, 0, h, dtype=np.uint8)
        alpha_np = np.tile(gradient, (w, 1)).T
        alpha_np = (alpha_np * opacity).astype(np.uint8)
        alpha_mask = Image.fromarray(alpha_np, 'L')
        reflection.putalpha(alpha_mask)
        return reflection


class PerspectiveCorrection:
    @staticmethod
    def apply_shelf_perspective(product_rgba, viewing_angle='eye_level'):
        """Apply perspective based on viewing angle."""
        if viewing_angle == 'eye_level':
            return product_rgba

        product_array = np.array(product_rgba)
        h, w = product_array.shape[:2]

        if viewing_angle == 'looking_up':
            dst_points = [(w * 0.05, 0), (w * 0.95, 0), (w, h), (0, h)]
        else:  # looking_down
            dst_points = [(0, 0), (w, 0), (w * 0.95, h), (w * 0.05, h)]

        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32(dst_points)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        result = cv2.warpPerspective(
            product_array, matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        return Image.fromarray(result)


class CameraArtifacts:
    @staticmethod
    def add_lens_distortion(image, strength=0.03):
        """Add barrel/pincushion lens distortion."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        k1 = strength * random.choice([-1, 1])

        camera_matrix = np.array([
            [w, 0, w/2],
            [0, h, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.array([k1, k1/2, 0, 0, 0], dtype=np.float32)

        return Image.fromarray(cv2.undistort(img_array, camera_matrix, dist_coeffs))

    @staticmethod
    def add_chromatic_aberration(image, shift=2):
        """Add chromatic aberration (color fringing)."""
        if image.mode != 'RGBA':
            image = image.convert("RGBA")
        r, g, b, a = image.split()
        r_shifted = Image.fromarray(np.roll(np.array(r), shift, axis=1))
        b_shifted = Image.fromarray(np.roll(np.array(b), -shift, axis=1))
        return Image.merge('RGBA', (r_shifted, g, b_shifted, a))

    @staticmethod
    def add_camera_noise(image):
        """Add realistic camera noise."""
        img_array = np.array(image).astype(float)
        noise = np.random.normal(0, random.uniform(3, 15), img_array.shape)
        return Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))


class RefrigeratorEffects:
    @staticmethod
    def apply_cold_color_filter(image):
        """Apply cold blue tint."""
        r, g, b = image.split()
        r = r.point(lambda i: i * 0.9)
        b = b.point(lambda i: i * 1.1)
        return Image.merge('RGB', (r, g, b))

    @staticmethod
    def add_condensation(image, config: AugmentationConfig):
        """Add condensation fog effect."""
        opacity_value = config.CONDENSATION_OPACITY
        if isinstance(opacity_value, (list, tuple)):
            opacity = random.randint(opacity_value[0], opacity_value[1])
        else:
            opacity = opacity_value

        overlay = Image.new('L', image.size, color=opacity)
        noise_shape = (image.height, image.width)
        noise = np.random.randint(0, 20, noise_shape, dtype=np.uint8)
        noise_img = Image.fromarray(noise, 'L').filter(ImageFilter.GaussianBlur(10))
        overlay = Image.blend(overlay, noise_img, 0.5)

        temp_img = image.copy()
        temp_img.putalpha(overlay)
        bg = Image.new('RGB', image.size, (240, 245, 245))
        bg.paste(temp_img, (0, 0), temp_img)
        return bg

    @staticmethod
    def add_frost_crystals(image, num_crystals=50):
        """Add frost crystal pattern."""
        if image.mode != 'RGBA':
            image = image.convert("RGBA")

        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = image.size

        for _ in range(num_crystals):
            x, y = random.randint(0, w), random.randint(0, h)
            size = random.randint(2, 8)
            opacity = random.randint(50, 150)
            for _ in range(random.randint(4, 7)):
                angle = random.uniform(0, 2 * np.pi)
                end_x = x + size * np.cos(angle)
                end_y = y + size * np.sin(angle)
                draw.line([(x, y), (end_x, end_y)], fill=(255, 255, 255, opacity), width=1)

        overlay = overlay.filter(ImageFilter.GaussianBlur(0.5))
        return Image.alpha_composite(image, overlay).convert("RGB")

    @staticmethod
    def add_wire_rack_overlay(size, viewing_angle='eye_level'):
        """Add wire rack overlay for refrigerator effect."""
        w, h = size
        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        color = (128, 128, 128, 100)

        for x in range(0, w, 25):
            draw.line([(x, 0), (x, h)], fill=color, width=2)

        if viewing_angle == 'looking_down':
            y, step = 0, 15
        elif viewing_angle == 'looking_up':
            y, step = h, 15
        else:
            y, step = 0, 25

        while (y < h if viewing_angle != 'looking_up' else y > 0):
            draw.line([(0, int(y)), (w, int(y))], fill=color, width=3)
            if viewing_angle == 'looking_up':
                y -= step
                step *= 1.05
            else:
                y += step
            if viewing_angle == 'looking_down':
                step *= 1.05

        return overlay


class ShelfElements:
    @staticmethod
    def add_price_rail(image_size):
        """Add price rail at bottom."""
        width, _ = image_size
        rail_height = random.randint(30, 50)
        main_color, line_color = random.choice([
            ((240, 240, 240, 200), (200, 200, 200)),
            ((50, 50, 50, 200), (30, 30, 30))
        ])

        rail = Image.new('RGBA', (width, rail_height), main_color)
        draw = ImageDraw.Draw(rail)
        draw.line([(0, 3), (width, 3)], fill=line_color, width=3)
        draw.line([(0, rail_height-3), (width, rail_height-3)], fill=line_color, width=3)
        return rail

    @staticmethod
    def add_realistic_price_tag():
        """Add a realistic price tag."""
        w, h = random.randint(80, 120), random.randint(50, 75)
        tag = Image.new('RGBA', (w, h), random.choice([
            (255, 255, 100, 240), (255, 255, 255, 240)
        ]))

        draw = ImageDraw.Draw(tag)
        draw.rectangle([(0, 0), (w-1, h-1)], outline=(0, 0, 0), width=1)
        price = f"₺{random.randint(5, 199)}.{random.choice(['50', '75', '99'])}"
        font_size = random.randint(20, 24)

        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
            draw.text((10, 8), price, fill=(0, 0, 0), font=font)
        except IOError:
            draw.text((10, 8), price, fill=(0, 0, 0))

        return tag

    @staticmethod
    def add_campaign_sticker():
        """Add campaign sticker (e.g., discount, new)."""
        w, h = random.randint(70, 100), random.randint(70, 100)
        sticker = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sticker)

        color = random.choice([(255, 0, 0, 220), (255, 230, 0, 220)])
        shape = random.choice(['ellipse', 'rectangle'])
        outline_width = 3

        if shape == 'ellipse':
            draw.ellipse([(0, 0), (w, h)], fill=color, outline=(255, 255, 255), width=outline_width)
        else:
            draw.rectangle([(0, 0), (w, h)], fill=color, outline=(255, 255, 255), width=outline_width)

        text = random.choice(["İNDİRİM", "YENİ!", "%25", "FIRSAT"])
        font_size = random.randint(18, 22)

        try:
            font = ImageFont.truetype("arialbd.ttf", size=font_size)
            draw.text((w/2, h/2), text, fill=(255, 255, 255), anchor="mm", font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        except IOError:
            draw.text((w/2, h/2), text, fill=(255, 255, 255), anchor="mm")

        return sticker.rotate(random.uniform(-15, 15), expand=True, fillcolor=(0, 0, 0, 0))


# ==============================
# HELPER FUNCTIONS
# ==============================
def transfer_color(source_img, target_img, strength):
    """Transfer color statistics from target to source."""
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, a_mean_src, b_mean_src), (l_std_src, a_std_src, b_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tgt, a_mean_tgt, b_mean_tgt), (l_std_tgt, a_std_tgt, b_std_tgt) = cv2.meanStdDev(target_lab)

    l, a, b = cv2.split(source_lab)
    l = (l - l_mean_src[0]) * (l_std_tgt[0] / (l_std_src[0] + 1e-6)) + l_mean_tgt[0]
    a = (a - a_mean_src[0]) * (a_std_tgt[0] / (a_std_src[0] + 1e-6)) + a_mean_tgt[0]
    b = (b - b_mean_src[0]) * (b_std_tgt[0] / (b_std_src[0] + 1e-6)) + b_mean_tgt[0]

    transfer = cv2.merge([np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)])
    transferred_bgr = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(transferred_bgr, strength, source_img, 1 - strength, 0)


def apply_additional_albumentations(image, config: AugmentationConfig):
    """Apply additional albumentations effects."""
    img_array = np.array(image)

    if random.random() < config.PROB_HSV_SHIFT:
        img_array = A.HueSaturationValue(
            hue_shift_limit=config.HSV_HUE_LIMIT,
            sat_shift_limit=config.HSV_SAT_LIMIT,
            val_shift_limit=config.HSV_VAL_LIMIT, p=1.0
        )(image=img_array)['image']

    if random.random() < config.PROB_RGB_SHIFT:
        img_array = A.RGBShift(
            r_shift_limit=config.RGB_SHIFT_LIMIT,
            g_shift_limit=config.RGB_SHIFT_LIMIT,
            b_shift_limit=config.RGB_SHIFT_LIMIT, p=1.0
        )(image=img_array)['image']

    if random.random() < config.PROB_MEDIAN_BLUR:
        img_array = A.MedianBlur(blur_limit=3, p=1.0)(image=img_array)['image']

    if random.random() < config.PROB_ISO_NOISE:
        img_array = A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.2), p=1.0)(image=img_array)['image']

    if random.random() < config.PROB_CLAHE:
        img_array = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)(image=img_array)['image']

    if random.random() < config.PROB_SHARPEN:
        img_array = A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)(image=img_array)['image']

    if random.random() < config.PROB_HORIZONTAL_FLIP:
        img_array = A.HorizontalFlip(p=1.0)(image=img_array)['image']

    return Image.fromarray(img_array)


# ==============================
# MAIN AUGMENTOR CLASS
# ==============================
class ProductAugmentor:
    """Production-grade augmentor with full shelf scene composition."""

    def __init__(self, batch_size=BATCH_SIZE, config: AugmentationConfig = None):
        self.batch_size = batch_size
        self.config = config or AugmentationConfig()

        # Version info
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")

        # Device selection with fallback
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("WARNING: CUDA not available, using CPU (slower)")
            self.batch_size = 1
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Load BiRefNet for high-quality segmentation
        print("\n📦 Loading BiRefNet...")
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet', trust_remote_code=True
        )
        self.birefnet.to(self.device).eval()

        # Use half precision on GPU for memory efficiency
        if self.device == 'cuda':
            self.birefnet.half()
        print("✅ BiRefNet loaded")

        # BiRefNet transform (reusable)
        self.birefnet_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Initialize augmentation transforms
        self.transforms = self._get_transforms()
        print(f"✅ Transforms initialized (batch_size={self.batch_size})")

        # Background images (loaded later)
        self.backgrounds = []

        # Neighbor product paths (for shelf composition)
        self.neighbor_product_paths = []

    def _get_transforms(self):
        """Get augmentation transforms - 3 different pipelines."""
        h, w = self.config.TARGET_SIZE

        # Light transforms for synthetic
        light = A.Compose([
            A.Affine(shear=(-3, 3), rotate=(-3, 3), p=0.4),
            mk_Resize(h, w, always_apply=True)
        ], is_check_shapes=False)

        # Heavy transforms for synthetic
        heavy = A.Compose([
            A.Perspective(scale=(0.01, 0.05), p=0.5),
            A.GridDistortion(p=0.2, distort_limit=0.2),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            A.Affine(shear=(-7, 7), rotate=(-7, 7), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=4, p=1),
                A.GaussianBlur(blur_limit=3, p=1)
            ], p=0.5),
            mk_Downscale(scale_range=[0.5, 0.8], p=0.5),
            mk_Resize(h, w, always_apply=True)
        ], is_check_shapes=False)

        # Real image transforms
        real = A.Compose([
            A.OneOf([
                A.Affine(shear=(-5, 5), rotate=(-5, 5), p=0.6),
                A.Perspective(scale=(0.01, 0.06), p=0.4),
                A.GridDistortion(p=0.2, distort_limit=0.2),
                A.OpticalDistortion(distort_limit=0.2, p=0.2),
            ], p=0.8),
            A.Sequential([
                mk_Downscale(scale_min=0.7, scale_max=0.95, p=0.2),
                mk_ImageCompression(quality_lower=65, quality_upper=95, p=0.2)
            ], p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.15, p=0.7),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.03, p=0.6),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.4),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.6),
                A.GaussianBlur(blur_limit=3, p=0.4)
            ], p=0.5),
            mk_CoarseDropout(max_holes=6, max_height=30, max_width=30, min_holes=1, fill_value=220, p=0.3),
            mk_Resize(h, w, always_apply=True)
        ], is_check_shapes=False)

        # Real without resize (for already-resized images)
        real_transforms_list = [t for t in real.transforms if not isinstance(t, A.Resize)]
        real_no_resize = A.Compose(real_transforms_list, is_check_shapes=False)

        return {
            'light': light,
            'heavy': heavy,
            'real': real,
            'real_no_resize': real_no_resize
        }

    def segment_with_birefnet(self, img_pil):
        """BiRefNet segmentation - single image (fallback)."""
        try:
            with torch.no_grad():
                inp = self.birefnet_transform(img_pil.copy()).unsqueeze(0).to(self.device)
                if self.device == 'cuda':
                    inp = inp.half()
                preds = self.birefnet(inp)[-1].sigmoid()
                mask = (preds[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

            mask_pil = Image.fromarray(mask).resize(img_pil.size, Image.BILINEAR)
            rgba = np.array(img_pil.convert("RGBA"))
            rgba[:, :, 3] = np.array(mask_pil)
            return Image.fromarray(rgba, 'RGBA')
        except Exception as e:
            import traceback
            print(f"  Segmentation error: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            return None

    def segment_batch(self, pil_images):
        """Wrapper for backwards compatibility."""
        results, _ = self.segment_batch_with_errors(pil_images)
        return results

    def segment_batch_with_errors(self, pil_images):
        """BiRefNet BATCH segmentation with error reporting."""
        errors = []
        if not pil_images:
            return [], errors

        print(f"    segment_batch: Processing {len(pil_images)} images on {self.device}")
        original_sizes = [img.size for img in pil_images]

        try:
            print(f"    Creating batch tensor...")
            batch_tensors = [self.birefnet_transform(img.copy()) for img in pil_images]
            batch = torch.stack(batch_tensors).to(self.device)
            if self.device == 'cuda':
                batch = batch.half()
            print(f"    Batch tensor created: {batch.shape}")

            print(f"    Running BiRefNet inference...")
            with torch.no_grad():
                preds = self.birefnet(batch)[-1].sigmoid()
            print(f"    BiRefNet inference complete: {preds.shape}")

            results = []
            success_count = 0
            for i, (img_pil, orig_size) in enumerate(zip(pil_images, original_sizes)):
                try:
                    mask = (preds[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask).resize(orig_size, Image.BILINEAR)
                    rgba = np.array(img_pil.convert("RGBA"))
                    rgba[:, :, 3] = np.array(mask_pil)
                    results.append(Image.fromarray(rgba, 'RGBA'))
                    success_count += 1
                except Exception as e:
                    err = f"Mask[{i}]: {str(e)[:100]}"
                    print(f"    {err}")
                    errors.append(err)
                    results.append(None)

            print(f"    Batch complete: {success_count}/{len(pil_images)} successful")
            return results, errors

        except Exception as e:
            import traceback
            err = f"BiRefNet batch failed: {str(e)[:200]}"
            print(f"    {err}")
            errors.append(err)
            print(f"    Traceback: {traceback.format_exc()}")
            print(f"    Falling back to sequential...")

            results = []
            for i, img in enumerate(pil_images):
                try:
                    result = self.segment_with_birefnet(img)
                    if result is not None:
                        print(f"    Sequential [{i}]: OK")
                    else:
                        print(f"    Sequential [{i}]: FAILED (returned None)")
                        errors.append(f"Sequential[{i}]: returned None")
                    results.append(result)
                except Exception as seq_e:
                    err = f"Sequential[{i}]: {str(seq_e)[:100]}"
                    print(f"    {err}")
                    errors.append(err)
                    results.append(None)
            return results, errors

    def load_backgrounds(self, backgrounds_path):
        """Load background images for synthetic augmentation."""
        self.backgrounds = []
        if backgrounds_path:
            bg_dir = Path(backgrounds_path)
            if bg_dir.exists():
                for ext in IMAGE_EXTENSIONS:
                    for p in bg_dir.glob(ext):
                        try:
                            self.backgrounds.append(Image.open(p).convert("RGB"))
                        except Exception:
                            pass
        print(f"🖼️ Background images loaded: {len(self.backgrounds)}")

    def load_backgrounds_from_pil(self, pil_images: List[Image.Image]):
        """Load background images from PIL Image list (for Supabase downloads)."""
        self.backgrounds = pil_images
        print(f"🖼️ Background images loaded from PIL: {len(self.backgrounds)}")

    def set_neighbor_paths(self, paths: List[str]):
        """Set neighbor product paths for shelf composition."""
        self.neighbor_product_paths = paths
        print(f"👥 Neighbor product paths set: {len(self.neighbor_product_paths)}")

    def get_random_background(self):
        """Get random background or generate solid color."""
        if self.backgrounds:
            return random.choice(self.backgrounds)
        return Image.new('RGB', (512, 512), (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        ))

    def add_neighboring_products(self, config: AugmentationConfig):
        """Add neighboring products to shelf scene."""
        neighbors = []
        if not self.neighbor_product_paths or len(self.neighbor_product_paths) < 2:
            return neighbors

        num_neighbors = random.randint(
            config.MIN_NEIGHBORS,
            min(config.MAX_NEIGHBORS, len(self.neighbor_product_paths) - 1)
        )

        for _ in range(num_neighbors):
            try:
                neighbor_path = random.choice(self.neighbor_product_paths)
                neighbor_img = Image.open(neighbor_path).convert("RGB")

                is_tipped = random.random() < config.PROB_TIPPED_OVER_NEIGHBOR
                rotation = random.uniform(80, 100) * random.choice([-1, 1]) if is_tipped else random.uniform(-10, 10)

                neighbor_np = A.Compose([
                    A.Rotate(limit=(rotation, rotation), p=1.0),
                    A.RandomBrightnessContrast(p=0.5)
                ])(image=np.array(neighbor_img))['image']

                neighbor_seg = self.segment_with_birefnet(Image.fromarray(neighbor_np))
                if neighbor_seg is None:
                    continue

                depth_scale = random.uniform(0.7, 0.9)
                blur_amount = (1 - depth_scale) * 5

                neighbor_bbox = neighbor_seg.getbbox()
                if neighbor_bbox:
                    neighbor_crop = neighbor_seg.crop(neighbor_bbox)
                    new_size = (int(neighbor_crop.width * depth_scale), int(neighbor_crop.height * depth_scale))
                    if new_size[0] > 0 and new_size[1] > 0:
                        neighbor_crop = neighbor_crop.resize(new_size, Image.LANCZOS)
                        if blur_amount > 0:
                            neighbor_crop = neighbor_crop.filter(ImageFilter.GaussianBlur(blur_amount))
                        neighbors.append({
                            'image': neighbor_crop,
                            'depth': depth_scale,
                            'side': random.choice(['left', 'right'])
                        })
            except Exception:
                continue

        return neighbors

    def create_enhanced_shelf_image(self, original_product_pil, background_pil, config: AugmentationConfig = None):
        """
        Create enhanced shelf scene with full composition.
        This is the main augmentation function from final_augmentor.py.
        """
        if config is None:
            config = self.config

        try:
            # Select and apply transform
            transform = self.transforms['heavy'] if random.random() < config.PROB_HEAVY_AUGMENTATION else self.transforms['light']
            augmented_pil = Image.fromarray(transform(image=np.array(original_product_pil))['image'])

            # Apply additional albumentations
            augmented_pil = apply_additional_albumentations(augmented_pil, config)

            # Segment product
            segmented_product = self.segment_with_birefnet(augmented_pil)
            if segmented_product is None:
                return None

            # Apply perspective
            viewing_angle = random.choice(['looking_up', 'looking_down']) if random.random() < config.PROB_PERSPECTIVE_CHANGE else 'eye_level'
            if viewing_angle != 'eye_level':
                segmented_product = PerspectiveCorrection.apply_shelf_perspective(segmented_product, viewing_angle)

            # Get bounding box
            bbox = segmented_product.getbbox()
            if not bbox:
                return None

            main_product_layer = segmented_product.crop(bbox)
            box_w, box_h = main_product_layer.size
            if box_w <= 1 or box_h <= 1:
                return None

            # Create composition canvas from background
            composed_box = background_pil.resize((box_w, box_h), Image.LANCZOS)

            # Add fluorescent banding
            if random.random() < config.PROB_FLUORESCENT_BANDING:
                composed_box = RealisticLighting.add_fluorescent_banding(composed_box)

            composed_rgba = composed_box.convert("RGBA")

            # Add neighboring products
            if random.random() < config.PROB_NEIGHBORING_PRODUCTS:
                neighbors = self.add_neighboring_products(config)
                for n in sorted(neighbors, key=lambda x: x['depth']):
                    img = n['image']
                    paste_x = 0 if n['side'] == 'left' else box_w - img.width
                    temp_neighbor_canvas = Image.new('RGBA', composed_rgba.size, (0, 0, 0, 0))
                    if img.width <= box_w and img.height <= box_h:
                        temp_neighbor_canvas.paste(img, (paste_x, box_h - img.height), img)
                        composed_rgba = Image.alpha_composite(composed_rgba, temp_neighbor_canvas)

            # Add shadow
            if random.random() < config.PROB_SHADOW:
                shadow_opacity_value = config.SHADOW_OPACITY
                if isinstance(shadow_opacity_value, tuple):
                    shadow_opacity_value = random.randint(shadow_opacity_value[0], shadow_opacity_value[1])

                shadow_layer = Image.new('RGBA', main_product_layer.size, (0, 0, 0, 0))
                shadow_layer.paste((0, 0, 0, shadow_opacity_value), mask=main_product_layer.getchannel('A'))

                shadow_blur_radius_value = config.SHADOW_BLUR_RADIUS
                if isinstance(shadow_blur_radius_value, tuple):
                    shadow_blur_radius_value = random.randint(shadow_blur_radius_value[0], shadow_blur_radius_value[1])

                blurred_shadow = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur_radius_value))
                temp_shadow_canvas = Image.new('RGBA', composed_rgba.size, (0, 0, 0, 0))
                temp_shadow_canvas.paste(blurred_shadow, config.SHADOW_OFFSET, blurred_shadow)
                composed_rgba = Image.alpha_composite(temp_shadow_canvas, composed_rgba)

            # Paste main product
            temp_product_canvas = Image.new('RGBA', composed_rgba.size, (0, 0, 0, 0))
            temp_product_canvas.paste(main_product_layer, (0, 0), main_product_layer)
            composed_rgba = Image.alpha_composite(composed_rgba, temp_product_canvas)
            composed_box = composed_rgba.convert("RGB")

            # Add shelf reflection
            if random.random() < config.PROB_SHELF_REFLECTION:
                reflection_image = RealisticLighting.add_shelf_reflection(main_product_layer, config)
                if reflection_image.mode == 'RGBA':
                    composed_box.paste(reflection_image, (0, box_h - main_product_layer.height), reflection_image)
                else:
                    composed_box.paste(reflection_image, (0, box_h - main_product_layer.height))

            # Add shelf rail
            if random.random() < config.PROB_SHELF_RAIL and box_h > 20:
                rail = ShelfElements.add_price_rail((box_w, box_h))
                composed_box.paste(rail, (0, box_h - rail.height - random.randint(0, 10)), rail)

            # Add price tag
            if random.random() < config.PROB_PRICE_TAG:
                tag = ShelfElements.add_realistic_price_tag()
                if box_w > tag.width + 10 and box_h > tag.height + 45:
                    composed_box.paste(tag, (
                        random.randint(5, box_w - tag.width - 5),
                        box_h - tag.height - random.randint(25, 45)
                    ), tag)

            # Add campaign sticker
            if random.random() < config.PROB_CAMPAIGN_STICKER:
                sticker = ShelfElements.add_campaign_sticker()
                if box_w > sticker.width and box_h > sticker.height:
                    composed_box.paste(sticker, (
                        random.randint(0, box_w - sticker.width),
                        random.randint(0, box_h - sticker.height)
                    ), sticker)

            # Apply refrigerator effects
            if random.random() < config.PROB_COLD_COLOR_FILTER:
                composed_box = RefrigeratorEffects.apply_cold_color_filter(composed_box)

            if random.random() < config.PROB_CONDENSATION:
                composed_box = RefrigeratorEffects.add_condensation(composed_box, config)

            if random.random() < config.PROB_FROST_CRYSTALS:
                composed_box = RefrigeratorEffects.add_frost_crystals(composed_box)

            if random.random() < config.PROB_WIRE_RACK:
                overlay = RefrigeratorEffects.add_wire_rack_overlay((box_w, box_h), viewing_angle)
                composed_box.paste(overlay, (0, 0), overlay)

            # Apply color transfer
            if random.random() < config.PROB_COLOR_TRANSFER:
                try:
                    strength_value = config.COLOR_TRANSFER_STRENGTH
                    if isinstance(strength_value, tuple):
                        strength_value = random.uniform(strength_value[0], strength_value[1])

                    source_cv = cv2.cvtColor(np.array(composed_box), cv2.COLOR_RGB2BGR)
                    target_cv = cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
                    final_cv = transfer_color(source_cv, target_cv, strength_value)
                    composed_box = Image.fromarray(cv2.cvtColor(final_cv, cv2.COLOR_BGR2RGB))
                except Exception:
                    pass

            # Apply camera artifacts
            if random.random() < config.PROB_LENS_DISTORTION:
                composed_box = CameraArtifacts.add_lens_distortion(composed_box)

            if random.random() < config.PROB_CHROMATIC_ABERRATION:
                composed_box = CameraArtifacts.add_chromatic_aberration(composed_box).convert("RGB")

            if random.random() < config.PROB_CAMERA_NOISE:
                composed_box = CameraArtifacts.add_camera_noise(composed_box)

            # FINAL RESIZE - Direct to TARGET_SIZE, no black borders
            final_image = composed_box.resize(config.TARGET_SIZE, Image.LANCZOS)
            return final_image

        except Exception as e:
            print(f"  create_enhanced_shelf_image error: {e}")
            return None

    def create_real_augmented_image(self, original_product_pil):
        """Create augmented real image with segmentation."""
        try:
            segmented_product = self.segment_with_birefnet(original_product_pil)
            if segmented_product is None:
                segmented_product = original_product_pil.convert("RGBA")

            bbox = segmented_product.getbbox()
            if not bbox:
                product_layer = original_product_pil
            else:
                product_layer = segmented_product.crop(bbox)

            augmented_array = self.transforms['real'](image=np.array(product_layer.convert("RGB")))['image']
            return Image.fromarray(augmented_array)
        except Exception as e:
            print(f"  create_real_augmented_image error: {e}")
            return None

    def max_index_for_syn_stem(self, upc_root, stem):
        """Find max index for syn_{stem}_{idx}.ext files."""
        max_idx = -1
        for p in list_images(upc_root):
            m = re.match(rf"^syn_{re.escape(stem)}_(\d+)\.(jpg|jpeg|png|webp)$", p.name, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx

    def max_index_for_real_stem(self, real_dir, stem):
        """Find max index for {stem}_aug_{idx}.ext files."""
        pat = RE_REAL_OUT_FOR_STEM(stem)
        max_idx = -1
        for p in list_images(real_dir):
            m = pat.match(p.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        return max_idx

    def process_upc(self, upc_dir, syn_target, real_target, use_diversity_pyramid=True):
        """
        Process single UPC with full shelf scene composition.
        """
        upc_root = Path(upc_dir)
        real_dir = upc_root / "real"
        real_dir.mkdir(exist_ok=True)

        # --- SYN SOURCE & COUNT ---
        root_imgs_all = [p for p in list_images(upc_root) if p.parent == upc_root]
        syn_sources = [p for p in root_imgs_all if not RE_SYN_OUT.match(p.name)]
        syn_outputs = [p for p in root_imgs_all if RE_SYN_OUT.match(p.name)]
        current_syn_total = len(syn_outputs)
        missing_syn = max(0, syn_target - current_syn_total)

        # --- REAL SOURCE & COUNT ---
        real_imgs_all = list_images(real_dir)
        real_sources = [p for p in real_imgs_all if not RE_REAL_OUT_ANY.match(p.name)]
        current_real_total = len(real_imgs_all)
        missing_real = max(0, real_target - current_real_total)

        produced_syn = 0
        produced_real = 0

        debug_info = {
            "jobs_created": 0,
            "transforms_ok": 0,
            "transforms_fail": 0,
            "segments_ok": 0,
            "segments_fail": 0,
            "composes_ok": 0,
            "composes_fail": 0,
            "saves_ok": 0,
            "saves_fail": 0,
            "errors": [],
        }

        # ========== SYN TOP-UP (with shelf scene composition) ==========
        if missing_syn > 0 and len(syn_sources) > 0:
            jobs = []
            per_src = ceil_div(missing_syn, len(syn_sources))

            print(f"  Creating jobs: missing_syn={missing_syn}, per_src={per_src}")

            for src in syn_sources:
                stem = src.stem
                start_idx = self.max_index_for_syn_stem(upc_root, stem) + 1
                quota = min(per_src, missing_syn - len(jobs))

                for k in range(quota):
                    if len(jobs) >= missing_syn:
                        break
                    bg = self.get_random_background()

                    # Get config (with diversity pyramid if enabled)
                    if use_diversity_pyramid:
                        config, level = DiversityPyramid.get_random_level_config(self.config)
                    else:
                        config = self.config

                    jobs.append((src, stem, start_idx + k, bg, config))

            debug_info["jobs_created"] = len(jobs)
            print(f"  Total jobs created: {len(jobs)}")

            # Process in batches
            for batch_start in range(0, len(jobs), self.batch_size):
                batch_jobs = jobs[batch_start:batch_start + self.batch_size]
                print(f"  Processing batch {batch_start // self.batch_size + 1}: {len(batch_jobs)} jobs")

                for src_path, stem, out_idx, bg, config in batch_jobs:
                    try:
                        img = Image.open(src_path).convert("RGB")

                        # Create enhanced shelf image with full composition
                        final = self.create_enhanced_shelf_image(img, bg, config)

                        if final is None:
                            debug_info["composes_fail"] += 1
                            continue
                        debug_info["composes_ok"] += 1

                        out_name = f"syn_{stem}_{out_idx:03d}.jpg"
                        out_path = upc_root / out_name
                        try:
                            final.save(out_path, quality=95)
                            produced_syn += 1
                            debug_info["saves_ok"] += 1
                        except Exception as e:
                            print(f"  ⚠️ Syn save error {out_name}: {e}")
                            debug_info["saves_fail"] += 1
                    except Exception as e:
                        print(f"  ⚠️ Job error: {e}")
                        debug_info["composes_fail"] += 1

        # ========== REAL TOP-UP ==========
        if missing_real > 0 and len(real_sources) > 0:
            per_src = ceil_div(missing_real, len(real_sources))
            for src in real_sources:
                if produced_real >= missing_real:
                    break
                stem = src.stem
                start_idx = self.max_index_for_real_stem(real_dir, stem) + 1
                quota = min(per_src, missing_real - produced_real)

                for k in range(quota):
                    try:
                        img = Image.open(src).convert("RGB")
                        aug = self.create_real_augmented_image(img)
                        if aug is None:
                            continue
                        out_name = f"{stem}_aug_{start_idx + k:03d}.jpg"
                        out_path = real_dir / out_name
                        try:
                            aug.save(out_path, quality=95)
                            produced_real += 1
                        except Exception as e:
                            print(f"  ⚠️ Real save error {out_name}: {e}")
                        if produced_real >= missing_real:
                            break
                    except Exception as e:
                        print(f"  ⚠️ Real aug error: {e}")

        print(f"  DEBUG: {debug_info}")
        return {
            "syn_sources": len(syn_sources),
            "real_sources": len(real_sources),
            "current_syn": current_syn_total,
            "current_real": current_real_total,
            "produced_syn": produced_syn,
            "produced_real": produced_real,
            "final_syn": current_syn_total + produced_syn,
            "final_real": current_real_total + produced_real,
            "debug": debug_info,
        }

    def process_dataset(self, dataset_path, syn_target, real_target, backgrounds_path=None, use_diversity_pyramid=True):
        """Process entire dataset - all UPCs with full shelf scene composition."""
        base = Path(dataset_path)
        if not base.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Load backgrounds
        if backgrounds_path:
            self.load_backgrounds(backgrounds_path)

        totals = defaultdict(int)
        per_upc_logs = []

        # Find all splits (train, test, valid)
        splits = [d for d in base.iterdir() if d.is_dir() and d.name in ('train', 'test', 'valid')]

        for split_dir in splits:
            print(f"\n{'=' * 28}  {split_dir.name.upper()}  {'=' * 28}")

            upc_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

            for upc in tqdm(upc_dirs, desc=f"{split_dir.name} UPC"):
                try:
                    stats = self.process_upc(upc, syn_target, real_target, use_diversity_pyramid)
                    totals[f"{split_dir.name}_syn_produced"] += stats["produced_syn"]
                    totals[f"{split_dir.name}_real_produced"] += stats["produced_real"]
                    totals["syn_produced"] += stats["produced_syn"]
                    totals["real_produced"] += stats["produced_real"]

                    per_upc_logs.append({
                        "split": split_dir.name,
                        "upc": upc.name,
                        **stats
                    })
                except Exception as e:
                    print(f"❌ UPC error: {upc.name} -> {e}")

        # Generate report
        report = {
            "totals": dict(totals),
            "items": per_upc_logs,
            "target_syn_per_upc": syn_target,
            "target_real_per_upc": real_target,
            "batch_size": self.batch_size,
            "config": {
                "TARGET_SIZE": self.config.TARGET_SIZE,
                "PROB_HEAVY_AUGMENTATION": self.config.PROB_HEAVY_AUGMENTATION,
                "PROB_NEIGHBORING_PRODUCTS": self.config.PROB_NEIGHBORING_PRODUCTS,
                "PROB_SHADOW": self.config.PROB_SHADOW,
            }
        }

        report_path = base / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Report saved: {report_path}")

        return {
            "syn_produced": totals["syn_produced"],
            "real_produced": totals["real_produced"],
            "report": report,
        }


def calculate_augmentation_plan(products: List[Dict], target_per_class: int) -> Dict:
    """
    Calculate how many augmentations needed per frame to reach target.

    Formula: augs_per_frame = ceil((target - current) / current_frames)

    Args:
        products: List of product dicts with 'id', 'frame_count', 'augmented_count'
        target_per_class: Target total images per product

    Returns:
        Dict with augmentation plan per product
    """
    plan = {}
    for product in products:
        product_id = product.get('id')
        current_frames = product.get('frame_count', 0)
        current_augmented = product.get('augmented_count', 0)
        current_total = current_frames + current_augmented

        if current_frames <= 0:
            plan[product_id] = {
                'augs_needed': 0,
                'augs_per_frame': 0,
                'message': 'No frames available'
            }
            continue

        needed = max(0, target_per_class - current_total)
        augs_per_frame = ceil_div(needed, current_frames)

        plan[product_id] = {
            'current_frames': current_frames,
            'current_augmented': current_augmented,
            'current_total': current_total,
            'target': target_per_class,
            'augs_needed': needed,
            'augs_per_frame': augs_per_frame,
            'projected_total': current_total + (augs_per_frame * current_frames)
        }

    return plan
