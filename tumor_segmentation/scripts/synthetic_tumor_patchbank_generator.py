#!/usr/bin/env python3
"""
Synthetic Tumor Patch-Bank Generator (NEW)
=========================================

Creates synthetic tumors on control PET slices by sampling real tumor patches
from patients, applying light geometric/intensity transforms, and blending them
onto the control using seamless cloning or alpha-matte blending.

This script does not modify the old generators. It keeps the same CLI shape and
outputs as `synthetic_tumor_look_generator.py` so you can swap it in.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.morphology import binary_erosion, disk
from tqdm import tqdm


def pad_to_target(img: np.ndarray, target_h: int, target_w: int, value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    if w != target_w:
        raise ValueError(f"Expected width {target_w}, got {w}.")
    if h == target_h:
        return img
    if h > target_h:
        return img[:target_h, :]
    pad_rows = target_h - h
    pad = np.full((pad_rows, w), value, dtype=img.dtype)
    return np.vstack([img, pad])


def compute_body_mask(pet_img: np.ndarray) -> np.ndarray:
    binary = pet_img > 0
    if np.sum(binary) == 0:
        return binary
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    regions.sort(key=lambda r: r.area, reverse=True)
    body = labeled == regions[0].label
    body = morphology.remove_small_holes(body, area_threshold=64)
    return body


def compute_forbidden_mask(pet_img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Mask out hot organs: top 10% within body and keep only large regions."""
    intensities = pet_img[body_mask]
    if intensities.size == 0:
        return np.zeros_like(pet_img, dtype=bool)
    cutoff = np.percentile(intensities, 90)
    hot = (pet_img >= cutoff) & body_mask
    labeled = measure.label(hot)
    forbidden = np.zeros_like(hot)
    for region in measure.regionprops(labeled):
        if region.area >= 200:
            forbidden[labeled == region.label] = True
    return forbidden


def tight_bbox(mask: np.ndarray, margin: int = 10) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0, 0, 0
    miny, maxy = ys.min(), ys.max()
    minx, maxx = xs.min(), xs.max()
    return int(miny - margin), int(minx - margin), int(maxy + margin), int(maxx + margin)


def clip_bbox(miny: int, minx: int, maxy: int, maxx: int, h: int, w: int) -> Tuple[int, int, int, int]:
    miny = max(0, miny)
    minx = max(0, minx)
    maxy = min(h - 1, maxy)
    maxx = min(w - 1, maxx)
    return miny, minx, maxy, maxx


class Patch:
    def __init__(self, image: np.ndarray, mask: np.ndarray):
        self.image = image.astype(np.float32)
        self.mask = (mask > 0).astype(np.uint8)


def build_patch_bank(patients_img_dir: Path, patients_label_dir: Path, target_h: int, target_w: int, limit: int = 180) -> List[Patch]:
    print("📚 Pairing patient images and labels by numeric ID and padding…")
    import re
    def nid(name: str) -> Optional[str]:
        m = re.findall(r"(\d+)", name)
        return m[-1] if m else None
    imgs = {nid(p.name): p for p in sorted(patients_img_dir.glob("*.png")) if nid(p.name)}
    lbls = {nid(p.name): p for p in sorted(patients_label_dir.glob("*.png")) if nid(p.name)}
    ids = sorted(set(imgs.keys()) & set(lbls.keys()), key=lambda x: int(x))[:limit]
    if not ids:
        raise RuntimeError("No matching patient image/label pairs found.")

    print("🔧 Building tumor patch bank…")
    bank: List[Patch] = []
    for pid in ids:
        img = cv2.imread(str(imgs[pid]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(lbls[pid]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img = pad_to_target(img, target_h, target_w)
        msk = pad_to_target(msk, target_h, target_w)
        m = (msk > 0).astype(np.uint8)
        if np.sum(m) == 0:
            continue
        y0, x0, y1, x1 = tight_bbox(m, margin=10)
        y0, x0, y1, x1 = clip_bbox(y0, x0, y1, x1, target_h, target_w)
        patch_img = img[y0 : y1 + 1, x0 : x1 + 1]
        patch_msk = m[y0 : y1 + 1, x0 : x1 + 1]
        # Zero out background outside
        patch_img = patch_img * patch_msk
        bank.append(Patch(patch_img, patch_msk))
    print(f"✅ Patch bank size: {len(bank)}")
    return bank


def random_affine(patch: Patch, scale: float, angle: float) -> Patch:
    h, w = patch.mask.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    m2 = cv2.warpAffine(patch.mask.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST)
    i2 = cv2.warpAffine(patch.image, M, (w, h), flags=cv2.INTER_LINEAR)
    m2 = (m2 > 127).astype(np.uint8)
    i2 = i2 * m2
    return Patch(i2, m2)


def place_mask(mask_small: np.ndarray, center_xy: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    H, W = canvas_shape
    ys, xs = np.where(mask_small > 0)
    if ys.size == 0:
        return np.zeros((H, W), dtype=np.uint8)
    cy = int((ys.min() + ys.max()) / 2)
    cx = int((xs.min() + xs.max()) / 2)
    dy = center_xy[1] - cy
    dx = center_xy[0] - cx
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(mask_small.astype(np.uint8) * 255, T, (W, H), flags=cv2.INTER_NEAREST)
    return (out > 127).astype(np.uint8)


def place_image(img_small: np.ndarray, center_xy: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    H, W = canvas_shape
    ys, xs = np.where(img_small > 0)
    if ys.size == 0:
        return np.zeros((H, W), dtype=img_small.dtype)
    cy = int((ys.min() + ys.max()) / 2)
    cx = int((xs.min() + xs.max()) / 2)
    dy = center_xy[1] - cy
    dx = center_xy[0] - cx
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img_small, T, (W, H), flags=cv2.INTER_LINEAR)
    return out

def alpha_blend_translate(dst_gray: np.ndarray, src_gray: np.ndarray, src_mask: np.ndarray, center_xy: Tuple[int, int], edge_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Blend src onto dst using soft edges after pure translation. Returns (blended, placed_mask)."""
    H, W = dst_gray.shape
    placed_mask = place_mask(src_mask, center_xy, (H, W))
    soft = cv2.GaussianBlur((placed_mask * 255).astype(np.uint8), (0, 0), sigmaX=edge_sigma)
    alpha = np.clip(soft.astype(np.float32) / 255.0, 0.0, 1.0)
    # Place source via translation with intensity preserved
    placed_src = place_image(src_gray, center_xy, (H, W)).astype(np.float32)
    dst_f = dst_gray.astype(np.float32)
    out = dst_f * (1.0 - alpha) + placed_src * alpha
    return np.clip(out, 0, 255).astype(np.uint8), placed_mask


def darken_relative(control_img: np.ndarray,
                    blended_img: np.ndarray,
                    placed_mask: np.ndarray,
                    ring_width: int = 6,
                    target_ratio: float = 0.82,
                    min_scale: float = 0.70,
                    max_scale: float = 0.95) -> np.ndarray:
    """Darken lesion relative to local background, preserving texture.

    Computes background median in ring around lesion and scales lesion pixels so
    tumor median approaches target_ratio * bg_median, clamped by [min_scale, max_scale].
    """
    m = placed_mask > 0
    if not np.any(m):
        return blended_img
    import numpy as _np
    ring = binary_erosion(binary_erosion(m, disk(1)) == False, disk(ring_width)) & (~m)
    bg_vals = control_img[ring]
    if bg_vals.size == 0:
        return blended_img
    bg_med = float(_np.median(bg_vals))
    tum_vals = blended_img[m]
    if tum_vals.size == 0:
        return blended_img
    t_med = float(_np.median(tum_vals))
    if t_med <= 1e-6:
        return blended_img
    target = target_ratio * bg_med
    scale = float(_np.clip(target / t_med, min_scale, max_scale))
    out = blended_img.astype(np.float32)
    out[m] = out[m] * scale
    return _np.clip(out, 0, 255).astype(np.uint8)


def horizontal_deband_inside(img: np.ndarray, mask: np.ndarray, sigma_x: float = 1.6, blend: float = 0.6) -> np.ndarray:
    if np.sum(mask) == 0:
        return img
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma_x, sigmaY=0)
    out = img.astype(np.float32)
    m = mask > 0
    out[m] = (1.0 - blend) * out[m] + blend * blurred[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def create_comparison_plot(control: np.ndarray, overlay: np.ndarray, mask: np.ndarray, tumor_preview: np.ndarray, output_path: Path, subject: str) -> None:
    # Ensure all inputs share the same spatial size as control
    h, w = control.shape
    if overlay.shape != control.shape:
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape != control.shape:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if tumor_preview.shape != control.shape:
        tumor_preview = cv2.resize(tumor_preview, (w, h), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(control, cmap="gray")
    axes[0].set_title(f"{subject}\nOriginal Control")
    axes[0].axis("off")
    axes[1].imshow(overlay, cmap="gray")
    axes[1].set_title(f"{subject}\nControl + Synthetic Tumors")
    axes[1].axis("off")
    base_rgb = cv2.cvtColor(control, cv2.COLOR_GRAY2RGB)
    color = base_rgb.copy().astype(np.float32)
    m = mask > 0
    color[m] = 0.65 * color[m] + 0.35 * np.array([255, 0, 0])
    axes[2].imshow(np.clip(color, 0, 255).astype(np.uint8))
    axes[2].set_title(f"{subject}\nMask (expected tumor regions)")
    axes[2].axis("off")
    axes[3].imshow(tumor_preview, cmap="gray")
    axes[3].set_title(f"{subject}\nTransformed tumor(s) only")
    axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tumors on control PET slices (Patch Bank method).")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_img_dir", required=True, type=str, help="Directory with real patient image PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    # Output folders per requested structure
    parser.add_argument("--labels_dir", type=str, default="data/controls/rough_labels_bank", help="Directory for rough binary masks.")
    parser.add_argument("--tumors_dir", type=str, default="data/control/rough_tumors_bank", help="Directory for tumor-only black images.")
    parser.add_argument("--overlays_dir", type=str, default="data/control/rough_overlay_bank", help="Directory for overlays.")
    parser.add_argument("--comparisons_dir", type=str, default="data/controls/rough_comparisons_bank", help="Directory for up to 10 random comparison plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_controls", type=int, default=None, help="Number of control subjects to process (default: all)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    controls_dir = Path(args.controls_dir)
    patients_img_dir = Path(args.patients_img_dir)
    patients_label_dir = Path(args.patients_label_dir)
    labels_dir = Path(args.labels_dir); labels_dir.mkdir(parents=True, exist_ok=True)
    tumors_dir = Path(args.tumors_dir); tumors_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = Path(args.overlays_dir); overlays_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = Path(args.comparisons_dir); comparisons_dir.mkdir(parents=True, exist_ok=True)

    # Infer canvas size from a control image
    control_paths = sorted(controls_dir.glob("*.png"))
    if not control_paths:
        raise RuntimeError("No control images found.")
    sample = cv2.imread(str(control_paths[0]), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise RuntimeError("Failed to read a control image.")
    TARGET_H, TARGET_W = sample.shape

    # Build patch bank
    bank = build_patch_bank(patients_img_dir, patients_label_dir, TARGET_H, TARGET_W, limit=180)

    print(f"🚀 Generating textured tumors for {len(control_paths) if args.num_controls is None else min(len(control_paths), args.num_controls)} control slices…")
    if args.num_controls is not None:
        control_paths = control_paths[: args.num_controls]

    # Reservoir sample for up to 10 comparisons
    comp_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path, str]] = []
    processed = 0

    for img_path in tqdm(control_paths, desc="Controls"):
        control_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if control_raw is None:
            continue
        orig_h, orig_w = control_raw.shape
        control = pad_to_target(control_raw, TARGET_H, TARGET_W)
        body = compute_body_mask(control)
        body_eroded = binary_erosion(body, disk(7))
        forbidden = compute_forbidden_mask(control, body)

        composite = control.copy().astype(np.float32)
        accum_mask = np.zeros_like(control, dtype=np.uint8)
        tumor_preview = np.full_like(control, 255, dtype=np.uint8)

        # Decide number of lesions (biased 2-4)
        n_lesions = random.randint(2, 4)

        # Candidate centers inside body mask
        ys, xs = np.where(body_eroded)
        coords = list(zip(xs.tolist(), ys.tolist()))

        for _ in range(n_lesions):
            attempts = 0
            placed_ok = False
            while attempts < 30 and not placed_ok:
                attempts += 1
                if not coords:
                    break
                cx, cy = random.choice(coords)
                patch = random.choice(bank)
                # Transform
                scale = random.uniform(0.7, 1.3)
                angle = random.uniform(-25, 25)
                suv_gain = random.uniform(0.9, 1.1)
                tp = random_affine(patch, scale=scale, angle=angle)
                src_img = np.clip(tp.image * suv_gain, 0, 255).astype(np.uint8)
                src_msk = tp.mask

                # Predict placement
                placed_mask = place_mask(src_msk, (cx, cy), composite.shape)

                # QC: within eroded body, avoid hot organs, avoid overlap, avoid image edge
                if not np.all((placed_mask & ~body_eroded) == 0):
                    continue
                if np.any(placed_mask & forbidden):
                    continue
                if np.any(placed_mask & accum_mask):
                    continue
                ys_pm, xs_pm = np.where(placed_mask > 0)
                if ys_pm.size == 0:
                    continue
                if (
                    ys_pm.min() <= 2 or ys_pm.max() >= TARGET_H - 3 or xs_pm.min() <= 2 or xs_pm.max() >= TARGET_W - 3
                ):
                    continue

                # Blend (apply after QC) using alpha-matte translation to preserve intensity
                blended, placed_mask = alpha_blend_translate(composite.astype(np.uint8), src_img, src_msk, (cx, cy), edge_sigma=2.0)
                # Optional debanding for large lesions
                if np.sum(placed_mask) > 250:
                    blended = horizontal_deband_inside(blended, placed_mask, sigma_x=1.6, blend=0.5)
                # Slightly darken relative to local background to improve visibility
                blended = darken_relative(control, blended, placed_mask, ring_width=6, target_ratio=0.82, min_scale=0.70, max_scale=0.95)
                composite = blended.astype(np.float32)
                accum_mask |= placed_mask
                # Tumor-only preview on white background
                placed_src = place_image(src_img, (cx, cy), composite.shape)
                tumor_preview[placed_mask > 0] = placed_src[placed_mask > 0]
                placed_ok = True

        subject = img_path.stem
        mask_out = labels_dir / f"{subject}.png"
        overlay_out = overlays_dir / f"{subject}.png"
        plot_out = comparisons_dir / f"{subject}_comparison.png"
        tumors_out = tumors_dir / f"{subject}.png"

        # Final QC: ensure mask dilation remains inside body
        dilated = cv2.dilate((accum_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if not np.all(dilated <= body.astype(np.uint8)):
            accum_mask = (accum_mask & body).astype(np.uint8)

        # Remove padding before saving: crop back to original control size
        mask_cropped = (accum_mask * 255).astype(np.uint8)[:orig_h, :orig_w]
        overlay_cropped = composite.astype(np.uint8)[:orig_h, :orig_w]
        preview_cropped = tumor_preview[:orig_h, :orig_w]
        control_cropped = control_raw  # original image

        cv2.imwrite(str(mask_out), mask_cropped)
        cv2.imwrite(str(overlay_out), overlay_cropped)
        cv2.imwrite(str(tumors_out), preview_cropped)

        # Reservoir sampling for at most 10 comparison plots
        processed += 1
        if len(comp_samples) < 10:
            comp_samples.append((control_cropped.copy(), overlay_cropped.copy(), (mask_cropped > 0).astype(np.uint8).copy(), preview_cropped.copy(), plot_out, subject))
        else:
            import random as _rnd
            j = _rnd.randint(1, processed)
            if j <= 10:
                replace_idx = _rnd.randint(0, 9)
                comp_samples[replace_idx] = (control_cropped.copy(), overlay_cropped.copy(), (mask_cropped > 0).astype(np.uint8).copy(), preview_cropped.copy(), plot_out, subject)

    # Emit at most 10 random comparison plots
    for control_c, overlay_c, mask_c, preview_c, plot_p, subj in comp_samples:
        create_comparison_plot(control_c, overlay_c, mask_c, preview_c, plot_p, subject=subj)

    print(f"✅ Rough labels saved to {labels_dir}")
    print(f"✅ Rough overlays saved to {overlays_dir}")
    print(f"✅ Rough tumors saved to {tumors_dir}")
    print(f"✅ Comparison plots saved to {comparisons_dir}")


if __name__ == "__main__":
    main()

