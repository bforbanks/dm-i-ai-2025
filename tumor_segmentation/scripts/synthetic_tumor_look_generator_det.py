#!/usr/bin/env python3
"""
Deterministic Synthetic Tumor Look Generator
===========================================

Goal: Duplicate EXACTLY the deterministic mask-generation pipeline from
`synthetic_mask_generator.py` (use_vae=False) and, in addition, apply the
same geometric operations to the tumour-only intensity image (patient image
multiplied by its mask). The result is an overlay image where the new tumour
intensities are pasted onto the control image at the placed mask locations.

This script does NOT modify existing files and is self-contained.

Outputs per control slice:
- Overlay image with synthetic tumour intensities
- Corresponding synthetic mask used for placement

Usage (example):
    python synthetic_tumor_look_generator_det.py \
        --controls_dir tumor_segmentation/data/controls/imgs \
        --patients_img_dir tumor_segmentation/data/patients/imgs \
        --patients_label_dir tumor_segmentation/data/patients/labels \
        --output_dir tumor_segmentation/data/controls/look_overlays_det

Requirements:
    pip install numpy scipy scikit-image opencv-python tqdm
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Tuple
import re

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure
from skimage.morphology import binary_erosion, binary_dilation, disk
from tqdm import tqdm


# Unified canvas size (match mask generator exactly)
TARGET_WIDTH = 400
TARGET_HEIGHT = 992


# ----------------------------------------------------------------------------
# Helpers copied to match the deterministic mask generator exactly
# ----------------------------------------------------------------------------


def pad_to_target(img: np.ndarray, target_h: int = TARGET_HEIGHT, target_w: int = TARGET_WIDTH, value: int = 0) -> np.ndarray:
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


def crop_to_size(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop image to original (h, w) after processing on padded canvas."""
    return img[:h, :w]


def compute_body_mask(pet_img: np.ndarray) -> np.ndarray:
    """OpenCV implementation matching the mask generator."""
    binary = (pet_img > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    body = (labels == largest_label)
    kernel = np.ones((5, 5), np.uint8)
    body = cv2.morphologyEx(body.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return (body > 127)


def compute_forbidden_mask(pet_img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Match mask generator: top 10% hottest within body, area >= 200."""
    intensities = pet_img[body_mask]
    if intensities.size == 0:
        return np.zeros_like(pet_img, dtype=bool)
    cutoff = np.percentile(intensities, 90)
    hot = ((pet_img >= cutoff) & body_mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    forbidden = np.zeros_like(hot, dtype=np.uint8)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= 200:
            forbidden[labels == label] = 1
    return forbidden.astype(bool)


def build_location_prior(patient_label_paths: List[Path], sigma: float = 25.0) -> np.ndarray:
    heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    for path in tqdm(patient_label_paths, desc="Centroid pass"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = pad_to_target(mask)
        mask = mask > 0
        labeled = measure.label(mask)
        for region in measure.regionprops(labeled):
            cy, cx = region.centroid
            heatmap[int(cy), int(cx)] += 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def build_heatmap_cdf(heatmap: np.ndarray) -> np.ndarray:
    flat = heatmap.flatten().astype(np.float64)
    s = flat.sum()
    if s <= 0:
        flat[:] = 1.0
        s = flat.sum()
    flat /= s
    cdf = np.cumsum(flat)
    cdf[-1] = 1.0
    return cdf


def empirical_distributions(patient_label_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    counts: List[int] = []
    areas: List[int] = []
    for path in tqdm(patient_label_paths, desc="Stats pass"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = pad_to_target(mask)
        binary = mask > 0
        labeled = measure.label(binary)
        counts.append(max(1, labeled.max()))
        for region in measure.regionprops(labeled):
            areas.append(region.area)
    count_values, count_counts = np.unique(counts, return_counts=True)
    count_cdf = np.cumsum(count_counts) / np.sum(count_counts)
    area_values, area_counts = np.unique(areas, return_counts=True)
    area_cdf = np.cumsum(area_counts) / np.sum(area_counts)
    return (np.vstack([count_values, count_cdf]), np.vstack([area_values, area_cdf]))


def elastic_deform_fields(shape: Tuple[int, int], alpha: float = 20.0, sigma: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate elastic displacement fields exactly like the mask generator's function."""
    random_state = np.random.RandomState(None)
    h, w = shape
    dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    return dx.astype(np.float32), dy.astype(np.float32)


def elastic_deform_mask(mask: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    shape = mask.shape
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(mask.astype(np.float32), indices, order=1, mode="reflect").reshape(shape)
    return (distorted > 0.5).astype(np.uint8)


def elastic_deform_image(image: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    shape = image.shape
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(image.astype(np.float32), indices, order=1, mode="reflect").reshape(shape)
    return distorted.astype(image.dtype)


def affine_transform_mask(mask: np.ndarray, scale: float, rotation_deg: float, shear_deg: float) -> np.ndarray:
    h, w = mask.shape
    center = (w / 2, h / 2)
    scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    matrix = rot_matrix.copy()
    matrix[:, 2] += scale_matrix[:, 2]
    shear_k = math.tan(math.radians(shear_deg))
    shear_matrix = np.array([[1, shear_k, 0], [0, 1, 0]], dtype=np.float32)
    matrix = shear_matrix @ np.vstack([matrix, [0, 0, 1]])
    matrix = matrix[:2, :]
    transformed = cv2.warpAffine(mask.astype(np.uint8) * 255, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return (transformed > 127).astype(np.uint8)


def affine_transform_image(img: np.ndarray, scale: float, rotation_deg: float, shear_deg: float) -> np.ndarray:
    h, w = img.shape
    center = (w / 2, h / 2)
    scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    matrix = rot_matrix.copy()
    matrix[:, 2] += scale_matrix[:, 2]
    shear_k = math.tan(math.radians(shear_deg))
    shear_matrix = np.array([[1, shear_k, 0], [0, 1, 0]], dtype=np.float32)
    matrix = shear_matrix @ np.vstack([matrix, [0, 0, 1]])
    matrix = matrix[:2, :]
    transformed = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return transformed


def morphology_ops(mask: np.ndarray, erode_iter: int, dilate_iter: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_iter)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def sample_from_cdf(cdf: np.ndarray) -> int:
    values, cum = cdf
    r = random.random()
    idx = np.searchsorted(cum, r)
    return int(values[idx])


def place_mask(mask: np.ndarray, center: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    mask_h, mask_w = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros(canvas_shape, dtype=np.uint8)
    curr_cy = (ys.min() + ys.max()) // 2
    curr_cx = (xs.min() + xs.max()) // 2
    dy = center[1] - curr_cy
    dx = center[0] - curr_cx
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    placed = cv2.warpAffine(mask.astype(np.uint8) * 255, translation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    return (placed > 127).astype(np.uint8)


def place_image(img: np.ndarray, center: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    ys, xs = np.where(img > 0)
    if ys.size == 0:
        return np.zeros(canvas_shape, dtype=img.dtype)
    curr_cy = (ys.min() + ys.max()) // 2
    curr_cx = (xs.min() + xs.max()) // 2
    dy = center[1] - curr_cy
    dx = center[0] - curr_cx
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    placed = cv2.warpAffine(img, translation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return placed


def eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 10:
        return 1.0
    coords = np.column_stack([xs, ys])
    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[1] == 0:
        return 1.0
    ratio = eigenvalues[0] / eigenvalues[1]
    return 1 - ratio


# ----------------------------------------------------------------------------
# Core: replicate synthesize_for_slice deterministically, plus intensity overlay
# ----------------------------------------------------------------------------


def warp_real_pair(seed_mask: np.ndarray, seed_intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the SAME geometric ops as warp_real_mask(), to both mask and intensity.

    Steps (match original):
      - affine (scale, rot, shear)
      - elastic deform (alpha=20, sigma=4)
      - morphology ops (mask only)
      - morphology close 3x3 (mask only)
    """
    # Affine (random parameters identical ranges)
    scale = random.uniform(0.7, 1.3)
    rot = random.uniform(-25, 25)
    shear = random.uniform(-10, 10)
    mask = affine_transform_mask(seed_mask, scale, rot, shear)
    img = affine_transform_image(seed_intensity, scale, rot, shear)

    # Elastic (shared fields for alignment)
    dx, dy = elastic_deform_fields(mask.shape, alpha=20, sigma=4)
    mask = elastic_deform_mask(mask, dx, dy)
    img = elastic_deform_image(img, dx, dy)

    # Morphology (mask only)
    mask = morphology_ops(mask, erode_iter=random.randint(0, 2), dilate_iter=random.randint(0, 2))

    # Gentle close (mask only), same kernel size 3×3
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = (mask > 127).astype(np.uint8)

    # Ensure image only inside mask
    img = img * (mask > 0)
    return mask, img


def synthesize_slice_with_look(
    pet_img: np.ndarray,
    body_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    heatmap: np.ndarray,
    count_cdf: np.ndarray,
    area_cdf: np.ndarray,
    seed_masks: List[np.ndarray],
    seed_intensities: List[np.ndarray],
    max_attempts: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (overlay_image, synthetic_mask) for a control slice.

    This mirrors synthesize_for_slice() exactly for geometry/placement/QC, but
    also carries the tumour-only intensity through the same operations.
    """
    h, w = pet_img.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = pet_img.copy().astype(np.float32)

    # Lesion count
    n_lesions = sample_from_cdf(count_cdf)

    body_eroded = binary_erosion(body_mask, disk(3))

    for _ in range(n_lesions):
        for _attempt in range(max_attempts):
            # Candidate from warped real mask (deterministic path)
            idx = random.randrange(len(seed_masks))
            seed_m = seed_masks[idx]
            seed_img = seed_intensities[idx]
            candidate_mask, candidate_img = warp_real_pair(seed_m, seed_img)

            # Rescale to match target area (apply same to image)
            target_area = sample_from_cdf(area_cdf)
            curr_area = int(np.sum(candidate_mask))
            if curr_area == 0:
                continue
            scale_factor = math.sqrt(target_area / curr_area)
            candidate_mask = affine_transform_mask(candidate_mask, scale_factor, 0, 0)
            candidate_img = affine_transform_image(candidate_img, scale_factor, 0, 0)

            # Sample location from heatmap (respect precomputed CDF if present)
            if hasattr(synthesize_slice_with_look, "_heatmap_cdf") and synthesize_slice_with_look._heatmap_cdf is not None:
                r = np.random.random()
                idx_flat = int(np.searchsorted(synthesize_slice_with_look._heatmap_cdf, r))
            else:
                flat_prob = heatmap.flatten()
                s = flat_prob.sum()
                if s <= 0:
                    idx_flat = np.random.randint(0, flat_prob.size)
                else:
                    flat_prob = flat_prob / s
                    idx_flat = np.random.choice(flat_prob.size, p=flat_prob)
            cy, cx = divmod(idx_flat, w)

            placed_mask = place_mask(candidate_mask, (cx, cy), (h, w))
            placed_img = place_image(candidate_img, (cx, cy), (h, w))

            # Validation (identical to mask generator)
            if not np.all((placed_mask & ~body_eroded) == 0):
                continue
            if np.any(placed_mask & forbidden_mask):
                continue
            if np.any(placed_mask & final_mask):
                continue

            # QC per lesion
            area = int(np.sum(placed_mask))
            if not (30 <= area <= 4000):
                continue
            if eccentricity(placed_mask) > 0.98:
                continue

            # Commit: update mask and overlay intensities strictly inside mask
            final_mask |= placed_mask
            overlay[placed_mask > 0] = placed_img[placed_mask > 0]
            break

    # Final QC: ensure dilated mask inside body (identical)
    dilated = binary_dilation(final_mask, disk(2))
    if not np.all(dilated <= body_mask):
        final_mask = final_mask & body_mask

    # Additional QC: exclude masks that cover more than 10% of the image area
    total_pixels = h * w
    covered = int(np.sum(final_mask))
    if covered > 0.10 * total_pixels:
        # Discard this synthesis: return original control image and empty mask
        empty = np.zeros_like(final_mask, dtype=np.uint8)
        return pet_img.astype(np.uint8), empty

    return np.clip(overlay, 0, 255).astype(np.uint8), final_mask.astype(np.uint8)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic look generator: replicate mask generator and carry tumour intensities.")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_img_dir", required=True, type=str, help="Directory with real patient image PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    # Output structure per user spec
    parser.add_argument("--labels_out_dir", type=str, default="data/controls/rough_labels_det", help="Directory to save binary masks.")
    parser.add_argument("--tumors_out_dir", type=str, default="data/control/rough_tumors_det", help="Directory to save black background with tumor intensities.")
    parser.add_argument("--overlay_out_dir", type=str, default="data/control/rough_overlay_det", help="Directory to save control overlays.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_controls", type=int, default=None, help="Limit number of controls processed.")
    parser.add_argument("--comparisons_dir", type=str, default=None, help="Optional directory to save comparison plots.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    controls_dir = Path(args.controls_dir)
    patients_img_dir = Path(args.patients_img_dir)
    patients_label_dir = Path(args.patients_label_dir)
    labels_out_dir = Path(args.labels_out_dir); labels_out_dir.mkdir(parents=True, exist_ok=True)
    tumors_out_dir = Path(args.tumors_out_dir); tumors_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir = Path(args.overlay_out_dir); overlay_out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.comparisons_dir) if args.comparisons_dir else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Robust pairing by numeric ID (e.g., patient_123.png ↔ segmentation_123.png)
    def extract_numeric_id(name: str) -> str | None:
        matches = re.findall(r"(\d+)", name)
        return matches[-1] if matches else None

    img_by_id = {extract_numeric_id(p.name): p for p in sorted(patients_img_dir.glob("*.png")) if extract_numeric_id(p.name) is not None}
    lbl_by_id = {extract_numeric_id(p.name): p for p in sorted(patients_label_dir.glob("*.png")) if extract_numeric_id(p.name) is not None}
    common_ids = sorted(set(img_by_id.keys()) & set(lbl_by_id.keys()), key=lambda x: int(x))
    if len(common_ids) == 0:
        raise RuntimeError("No matching patient image/label pairs found by numeric ID. Check directory contents.")

    # Limit to ~180 pairs to mirror original behavior
    common_ids = common_ids[:180]

    # Build global data (location prior and CDFs) from the matched label set
    patient_label_paths = [lbl_by_id[i] for i in common_ids]
    if len(patient_label_paths) < 50:
        raise RuntimeError("Not enough patient masks found to build statistics.")

    print("🔧 Pre-computing location prior…")
    heatmap = build_location_prior(patient_label_paths)
    synthesize_slice_with_look._heatmap_cdf = build_heatmap_cdf(heatmap)

    print("📊 Building empirical distributions…")
    count_cdf, area_cdf = empirical_distributions(patient_label_paths)

    print("📚 Loading seed masks and tumour-only images…")
    seed_masks: List[np.ndarray] = []
    seed_intensities: List[np.ndarray] = []
    for pid in tqdm(common_ids, desc="Patients"):
        lbl_path = lbl_by_id[pid]
        img_path = img_by_id[pid]
        mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or img is None:
            continue
        mask = pad_to_target(mask)
        img = pad_to_target(img)
        mask_bin = (mask > 0).astype(np.uint8)
        tumor_only = (img.astype(np.float32)) * (mask_bin > 0)
        seed_masks.append(mask_bin)
        seed_intensities.append(tumor_only)

    if len(seed_masks) == 0:
        raise RuntimeError("No seed mask/image pairs were built. Verify patients_img_dir and patients_label_dir naming.")

    # Iterate over controls
    control_paths = sorted(controls_dir.glob("*.png"))
    if args.num_controls is not None:
        control_paths = control_paths[: args.num_controls]
    print(f"🚀 Generating overlays for {len(control_paths)} control slices…")

    # Choose at most 10 random indices to save comparisons (if requested)
    selected_plot_indices = None
    if plots_dir is not None:
        import random as _rand
        k = min(10, len(control_paths))
        selected_plot_indices = set(_rand.sample(range(len(control_paths)), k)) if k > 0 else set()

    for idx, img_path in enumerate(tqdm(control_paths, desc="Controls")):
        pet = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pet is None:
            continue
        orig_h, orig_w = pet.shape
        pet = pad_to_target(pet)
        body = compute_body_mask(pet)
        forbidden = compute_forbidden_mask(pet, body)

        overlay_img, det_mask = synthesize_slice_with_look(
            pet_img=pet,
            body_mask=body,
            forbidden_mask=forbidden,
            heatmap=heatmap,
            count_cdf=count_cdf,
            area_cdf=area_cdf,
            seed_masks=seed_masks,
            seed_intensities=seed_intensities,
        )

        # Crop outputs back to original dimensions (remove processing padding)
        overlay_cropped = crop_to_size(overlay_img, orig_h, orig_w)
        mask_cropped = crop_to_size(det_mask, orig_h, orig_w)

        subject_name = img_path.stem
        mask_out = labels_out_dir / f"{subject_name}.png"
        overlay_out = overlay_out_dir / f"{subject_name}.png"
        tumors_out = tumors_out_dir / f"{subject_name}.png"
        cv2.imwrite(str(mask_out), (mask_cropped * 255).astype(np.uint8))
        cv2.imwrite(str(overlay_out), overlay_cropped)
        tumors_only_cropped = np.where(mask_cropped > 0, overlay_cropped, np.zeros_like(overlay_cropped))
        cv2.imwrite(str(tumors_out), tumors_only_cropped)

        if plots_dir is not None and (selected_plot_indices is None or idx in selected_plot_indices):
            # Simple 3-panel comparison: original, overlay, mask overlay
            import matplotlib.pyplot as plt

            def colored_mask_overlay(base_gray: np.ndarray, mask_bin: np.ndarray, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
                base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
                overlay_rgb = base_rgb.copy().astype(np.float32)
                color_img = np.zeros_like(base_rgb, dtype=np.float32)
                color_img[..., 0] = color[0]
                color_img[..., 1] = color[1]
                color_img[..., 2] = color[2]
                m = mask_bin > 0
                overlay_rgb[m] = (1.0 - alpha) * overlay_rgb[m] + alpha * color_img[m]
                return np.clip(overlay_rgb, 0, 255).astype(np.uint8)

            plot_path = plots_dir / f"{subject_name}_det_look_comparison.png"
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(crop_to_size(pet, orig_h, orig_w), cmap="gray")
            axes[0].set_title(f"{subject_name}\nOriginal Control")
            axes[0].axis("off")
            axes[1].imshow(overlay_cropped, cmap="gray")
            axes[1].set_title(f"{subject_name}\nOverlay (det look)")
            axes[1].axis("off")
            colored = colored_mask_overlay(crop_to_size(pet, orig_h, orig_w), mask_cropped, color=(255, 0, 0), alpha=0.35)
            axes[2].imshow(colored)
            axes[2].set_title(f"{subject_name}\nMask (det)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

    print(f"✅ Saved masks to {labels_out_dir}")
    print(f"✅ Saved tumors-only images to {tumors_out_dir}")
    print(f"✅ Saved overlays to {overlay_out_dir}")


if __name__ == "__main__":
    main()

