"""
Product Augmentor using BiRefNet + Albumentations.
Based on: final_augmentor_v3.py - IN-PLACE TOP-UP (IDEMPOTENT)

Features:
- BiRefNet segmentation with GPU half-precision
- BATCH INFERENCE for 10x speedup
- 3 augmentation pipelines (light, heavy, real)
- Idempotent: only generates missing images
- Background composition with shadows
- Border detection for resize decision
"""

import os
import re
import cv2
import math
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import albumentations as A
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm
from collections import defaultdict
import json

# ==============================
# CONFIGURATION
# ==============================
TARGET_SIZE = (384, 384)
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
BATCH_SIZE = 8  # Optimal for RTX 4090 (24GB) - adjust based on GPU memory

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


class ProductAugmentor:
    """Production-grade augmentor with batch inference optimization."""

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

        # Version info
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")

        # Device selection with fallback
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            print("WARNING: CUDA not available, using CPU (slower)")
            self.batch_size = 1  # CPU can't handle large batches
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

    def _get_transforms(self):
        """Get augmentation transforms - 3 different pipelines."""
        # Light transforms for synthetic
        light = A.Compose([
            A.Affine(shear=(-3, 3), rotate=(-3, 3), p=0.4),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
        ], is_check_shapes=False)

        # Heavy transforms for synthetic
        heavy = A.Compose([
            A.Perspective(scale=(0.01, 0.05), p=0.5),
            A.GridDistortion(p=0.2, distort_limit=0.2),
            A.OpticalDistortion(distort_limit=0.2, p=0.2),
            A.Affine(shear=(-7, 7), rotate=(-7, 7), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7))
            ], p=0.5),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
        ], is_check_shapes=False)

        # Real image transforms
        real = A.Compose([
            A.OneOf([
                A.Affine(shear=(-5, 5), rotate=(-5, 5), p=0.6),
                A.Perspective(scale=(0.01, 0.06), p=0.4),
                A.GridDistortion(p=0.2, distort_limit=0.2),
                A.OpticalDistortion(distort_limit=0.2, p=0.2),
            ], p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.15, p=0.7),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.03, p=0.6),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.4),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussianBlur(blur_limit=(3, 7))
            ], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(1, 30),
                hole_width_range=(1, 30),
                fill=220,
                p=0.3
            ),
            A.Resize(TARGET_SIZE[1], TARGET_SIZE[0])
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

    def detect_resize_by_border(self, image, border_thickness=10, black_threshold=10):
        """Detect if image needs resize by checking black borders."""
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            img = np.array(image)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

            tb = gray[0:border_thickness, :]
            bb = gray[h - border_thickness:h, :]
            lb = gray[:, 0:border_thickness]
            rb = gray[:, w - border_thickness:w]

            all_black = (np.mean(tb) < black_threshold and np.mean(bb) < black_threshold
                         and np.mean(lb) < black_threshold and np.mean(rb) < black_threshold)

            return {'needs_resize': all_black, 'is_resized': not all_black}
        except Exception as e:
            print(f"  Resize detection error: {e}")
            return {'needs_resize': False, 'is_resized': False}

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
        """
        BiRefNet BATCH segmentation with error reporting.

        Returns:
            Tuple of (results_list, errors_list)
        """
        errors = []
        if not pil_images:
            return [], errors

        print(f"    segment_batch: Processing {len(pil_images)} images on {self.device}")

        # Store original sizes for mask resizing
        original_sizes = [img.size for img in pil_images]

        try:
            # Create batch tensor
            print(f"    Creating batch tensor...")
            batch_tensors = [self.birefnet_transform(img.copy()) for img in pil_images]
            batch = torch.stack(batch_tensors).to(self.device)
            if self.device == 'cuda':
                batch = batch.half()
            print(f"    Batch tensor created: {batch.shape}")

            # Batch inference
            print(f"    Running BiRefNet inference...")
            with torch.no_grad():
                preds = self.birefnet(batch)[-1].sigmoid()
            print(f"    BiRefNet inference complete: {preds.shape}")

            # Process each result
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

            # Fallback to sequential
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

    def compose_synthetic(self, rgba_product, bg_image):
        """Compose product onto background with shadow effect."""
        try:
            bbox = rgba_product.getbbox()
            if not bbox:
                return None

            product = rgba_product.crop(bbox)  # RGBA

            # Background composition
            composed = bg_image.resize(product.size, Image.LANCZOS).convert("RGB")
            composed_rgba = composed.convert("RGBA")

            # Shadow effect (70% chance)
            if random.random() < 0.7:
                product_alpha = product.getchannel('A')
                shadow = Image.new('RGBA', product.size, (0, 0, 0, random.randint(80, 140)))
                shadow.putalpha(product_alpha)
                for _ in range(random.randint(5, 8)):
                    shadow = shadow.filter(ImageFilter.BLUR)
                temp = Image.new('RGBA', composed.size, (0, 0, 0, 0))
                temp.paste(shadow, (5, 5), shadow)
                composed_rgba = Image.alpha_composite(composed_rgba, temp)

            # Paste product
            temp_product = Image.new('RGBA', composed.size, (0, 0, 0, 0))
            temp_product.paste(product, (0, 0), product)
            composed_rgba = Image.alpha_composite(composed_rgba, temp_product)

            final = composed_rgba.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            return final
        except Exception as e:
            print(f"  Compose error: {e}")
            return None

    def augment_real_image(self, img_path):
        """Real image augmentation with resize detection."""
        try:
            img = Image.open(img_path).convert("RGB")
            resize_info = self.detect_resize_by_border(img)

            if resize_info['needs_resize']:
                # Raw image - segment + crop + augment
                rgba = self.segment_with_birefnet(img)
                if rgba is None:
                    return None

                bbox = rgba.getbbox()
                if bbox:
                    cropped = rgba.crop(bbox).convert("RGB")
                    augmented = self.transforms['real'](image=np.array(cropped))['image']
                else:
                    augmented = self.transforms['real'](image=np.array(img))['image']
            else:
                # Clean image - direct augment (no resize)
                augmented = self.transforms['real_no_resize'](image=np.array(img))['image']

            return Image.fromarray(augmented)
        except Exception as e:
            print(f"  Real aug error for {img_path.name}: {e}")
            return None

    def augment_synthetic_image(self, img_path, bg_image):
        """Synthetic image augmentation - single image (fallback)."""
        try:
            img = Image.open(img_path).convert("RGB")

            # Select and apply transform
            transform = self.transforms['heavy'] if random.random() < 0.5 else self.transforms['light']
            augmented = Image.fromarray(transform(image=np.array(img))['image'])

            # Segment + crop
            rgba = self.segment_with_birefnet(augmented)
            if rgba is None:
                return None

            return self.compose_synthetic(rgba, bg_image)
        except Exception as e:
            print(f"  Synthetic aug error for {img_path.name}: {e}")
            return None

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

    def get_random_background(self):
        """Get random background or generate solid color."""
        if self.backgrounds:
            return random.choice(self.backgrounds)
        return Image.new('RGB', (512, 512), (
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        ))

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

    def process_upc(self, upc_dir, syn_target, real_target):
        """
        Process single UPC - IDEMPOTENT TOP-UP with BATCH processing.
        Only generates missing images to reach target.
        """
        upc_root = Path(upc_dir)
        real_dir = upc_root / "real"
        real_dir.mkdir(exist_ok=True)

        # --- SYN SOURCE & COUNT ---
        root_imgs_all = [p for p in list_images(upc_root) if p.parent == upc_root]
        syn_sources = [p for p in root_imgs_all if not RE_SYN_OUT.match(p.name)]
        syn_outputs = [p for p in root_imgs_all if RE_SYN_OUT.match(p.name)]
        current_syn_total = len(syn_outputs)  # Only count actual syn_*.jpg outputs
        missing_syn = max(0, syn_target - current_syn_total)

        # --- REAL SOURCE & COUNT ---
        real_imgs_all = list_images(real_dir)
        real_sources = [p for p in real_imgs_all if not RE_REAL_OUT_ANY.match(p.name)]
        current_real_total = len(real_imgs_all)
        missing_real = max(0, real_target - current_real_total)

        produced_syn = 0
        produced_real = 0

        # Debug counters
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

        # ========== SYN TOP-UP (BATCH) ==========
        if missing_syn > 0 and len(syn_sources) > 0:
            # Prepare all augmentation jobs
            jobs = []  # [(src_path, stem, out_idx, bg_image), ...]
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
                    jobs.append((src, stem, start_idx + k, bg))

            debug_info["jobs_created"] = len(jobs)
            print(f"  Total jobs created: {len(jobs)}")

            # Process in batches
            for batch_start in range(0, len(jobs), self.batch_size):
                batch_jobs = jobs[batch_start:batch_start + self.batch_size]
                print(f"  Processing batch {batch_start // self.batch_size + 1}: {len(batch_jobs)} jobs")

                # Step 1: Load and transform images
                transformed_images = []
                for src_path, stem, out_idx, bg in batch_jobs:
                    try:
                        img = Image.open(src_path).convert("RGB")
                        transform = self.transforms['heavy'] if random.random() < 0.5 else self.transforms['light']
                        augmented = Image.fromarray(transform(image=np.array(img))['image'])
                        transformed_images.append(augmented)
                        debug_info["transforms_ok"] += 1
                    except Exception as e:
                        print(f"    Transform error: {e}")
                        transformed_images.append(None)
                        debug_info["transforms_fail"] += 1

                # Step 2: Batch segmentation
                valid_images = [img for img in transformed_images if img is not None]
                print(f"    Valid images for segmentation: {len(valid_images)}")
                if valid_images:
                    try:
                        segmented, seg_errors = self.segment_batch_with_errors(valid_images)
                        debug_info["segments_ok"] += len([s for s in segmented if s is not None])
                        debug_info["segments_fail"] += len([s for s in segmented if s is None])
                        if seg_errors:
                            debug_info["errors"].extend(seg_errors[:3])  # Limit to first 3 errors
                        print(f"    Segmented: {len([s for s in segmented if s is not None])} ok, {len([s for s in segmented if s is None])} fail")
                    except Exception as e:
                        import traceback
                        err_msg = f"Batch seg outer error: {str(e)[:200]}"
                        print(f"    {err_msg}")
                        debug_info["errors"].append(err_msg)
                        segmented = [None] * len(valid_images)
                        debug_info["segments_fail"] += len(valid_images)
                else:
                    segmented = []

                # Step 3: Compose and save
                seg_idx = 0
                for i, (src_path, stem, out_idx, bg) in enumerate(batch_jobs):
                    if transformed_images[i] is None:
                        continue

                    rgba = segmented[seg_idx] if seg_idx < len(segmented) else None
                    seg_idx += 1

                    if rgba is None:
                        debug_info["composes_fail"] += 1
                        continue

                    final = self.compose_synthetic(rgba, bg)
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

        # ========== REAL TOP-UP (sequential - usually fewer images) ==========
        if missing_real > 0 and len(real_sources) > 0:
            per_src = ceil_div(missing_real, len(real_sources))
            for src in real_sources:
                if produced_real >= missing_real:
                    break
                stem = src.stem
                start_idx = self.max_index_for_real_stem(real_dir, stem) + 1
                quota = min(per_src, missing_real - produced_real)

                for k in range(quota):
                    aug = self.augment_real_image(src)
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

    def process_dataset(self, dataset_path, syn_target, real_target, backgrounds_path=None):
        """Process entire dataset - all UPCs with batch optimization."""
        base = Path(dataset_path)
        if not base.exists():
            raise ValueError(f"Dataset path not found: {dataset_path}")

        # Load backgrounds
        self.load_backgrounds(backgrounds_path)

        totals = defaultdict(int)
        per_upc_logs = []

        # Find all splits (train, test, valid)
        splits = [d for d in base.iterdir() if d.is_dir() and d.name in ('train', 'test', 'valid')]

        for split_dir in splits:
            print(f"\n{'=' * 28}  {split_dir.name.upper()}  {'=' * 28}")

            # Find all UPC directories
            upc_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

            for upc in tqdm(upc_dirs, desc=f"{split_dir.name} UPC"):
                try:
                    stats = self.process_upc(upc, syn_target, real_target)
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
            "batch_size": self.batch_size
        }

        # Save report
        report_path = base / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Report saved: {report_path}")

        return {
            "syn_produced": totals["syn_produced"],
            "real_produced": totals["real_produced"],
            "report": report,
        }
