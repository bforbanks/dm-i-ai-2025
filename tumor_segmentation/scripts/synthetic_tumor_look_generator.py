#!/usr/bin/env python3
"""
Synthetic Tumor Look Generator (NEW)
====================================

Generates synthetic tumors on healthy PET control slices by copying not only
the mask shape but also the appearance (intensity texture) of real tumors.

This script is NEW and does not modify or affect the existing
`scripts/synthetic_mask_generator.py`.

Key ideas:
- Deterministic path: apply the same geometric operations to a cropped tumor
  region (from a real patient image) as are applied to the binary mask, then
  place both on a control image.
- VAE path: train a grayscale β-VAE on 64×64 tumor patches (mask-filled
  grayscale). At generation, sample a plausible tumor intensity patch, derive
  a mask from it, rescale, and place on a control image.

Outputs per control slice:
- A synthetic mask (`*_look_mask.png`)
- A synthetic control image overlaid with tumors (`*_look_overlay.png`)
- A comparison figure with two panels: original control vs overlaid control

Usage (example):
    python synthetic_tumor_look_generator.py \
        --controls_dir tumor_segmentation/data/controls/imgs \
        --patients_img_dir tumor_segmentation/data/patients/imgs \
        --patients_label_dir tumor_segmentation/data/patients/labels \
        --output_dir tumor_segmentation/data/controls/look_synthetics

Requirements:
    pip install numpy scipy scikit-image opencv-python tqdm matplotlib torch
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from skimage import measure, morphology
from skimage.morphology import binary_erosion, binary_dilation, disk
from tqdm import tqdm

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Keep the same target canvas as the original generator
TARGET_WIDTH = 400
TARGET_HEIGHT = 992


# ----------------------------------------------------------------------------
# Utilities (padding, helper maps, statistics) – duplicated to avoid touching the original script
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


def sample_from_cdf(cdf: np.ndarray) -> int:
    values, cum = cdf
    r = random.random()
    idx = np.searchsorted(cum, r)
    return int(values[idx])


def sample_from_cdf_tail(cdf: np.ndarray, min_quantile: float = 0.85, max_quantile: float = 0.99) -> int:
    """Biased sampling toward the upper tail of an empirical CDF.

    Picks a random quantile in [min_quantile, max_quantile] and maps it via the CDF.
    """
    values, cum = cdf
    q = random.uniform(min_quantile, max_quantile)
    idx = np.searchsorted(cum, q)
    idx = np.clip(idx, 0, len(values) - 1)
    return int(values[idx])


# ----------------------------------------------------------------------------
# Geometric transforms (shared for mask and intensity)
# ----------------------------------------------------------------------------


def affine_matrices(scale: float, rotation_deg: float, shear_deg: float, h: int, w: int) -> np.ndarray:
    center = (w / 2, h / 2)
    scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    matrix = rot_matrix.copy()
    matrix[:, 2] += scale_matrix[:, 2]
    shear_k = math.tan(math.radians(shear_deg))
    shear_matrix = np.array([[1, shear_k, 0], [0, 1, 0]], dtype=np.float32)
    matrix = shear_matrix @ np.vstack([matrix, [0, 0, 1]])
    matrix = matrix[:2, :]
    return matrix.astype(np.float32)


def warp_affine_pair(mask: np.ndarray, img: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    mask_t = cv2.warpAffine((mask > 0).astype(np.uint8) * 255, matrix, (w, h), flags=cv2.INTER_NEAREST)
    img_t = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return (mask_t > 127).astype(np.uint8), img_t


def elastic_fields(shape: Tuple[int, int], alpha: float = 20.0, sigma: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    random_state = np.random.RandomState(None)
    h, w = shape
    dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    return dx.astype(np.float32), dy.astype(np.float32)


def elastic_deform_pair(mask: np.ndarray, img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
    img_def = map_coordinates(img.astype(np.float32), indices, order=1, mode="reflect").reshape((h, w))
    mask_def = map_coordinates((mask > 0).astype(np.float32), indices, order=0, mode="nearest").reshape((h, w))
    return (mask_def > 0.5).astype(np.uint8), img_def.astype(img.dtype)


@dataclass
class CutStitchPlan:
    piece_widths: List[int]
    offsets: List[int]


def build_cut_stitch_plan(w: int, pieces: int, max_offset: int) -> CutStitchPlan:
    piece_width = w // pieces
    widths = [piece_width] * (pieces - 1) + [w - piece_width * (pieces - 1)]
    offsets = [random.randint(-max_offset, max_offset) for _ in range(pieces)]
    return CutStitchPlan(widths, offsets)


def apply_cut_stitch_plan(mask: np.ndarray, img: np.ndarray, plan: CutStitchPlan) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    new_mask = np.zeros_like(mask)
    new_img = np.zeros_like(img)
    x_start = 0
    dest_cursor = 0
    for i, (slice_w, offset) in enumerate(zip(plan.piece_widths, plan.offsets)):
        x_end = min(w, x_start + slice_w)
        slice_mask = mask[:, x_start:x_end]
        slice_img = img[:, x_start:x_end]
        dest_start = int(np.clip(dest_cursor + offset, 0, w - slice_mask.shape[1]))
        dest_end = dest_start + slice_mask.shape[1]
        # Paste with overwrite inside mask
        region = new_img[:, dest_start:dest_end]
        region_mask = new_mask[:, dest_start:dest_end]
        region[ slice_mask > 0 ] = slice_img[ slice_mask > 0 ]
        region_mask |= slice_mask
        new_img[:, dest_start:dest_end] = region
        new_mask[:, dest_start:dest_end] = region_mask
        x_start = x_end
        dest_cursor += slice_w
    return new_mask, new_img


def compute_edge_band(mask: np.ndarray, inner_width: int = 2, outer_width: int = 6) -> np.ndarray:
    if np.sum(mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    inside = distance_transform_edt(mask > 0)
    band = (inside >= inner_width) & (inside <= outer_width)
    band = band & (mask > 0)
    return band


def feather_horizontal(intensity: np.ndarray, mask: np.ndarray, sigma_x: float = 1.25, band: np.ndarray | None = None) -> np.ndarray:
    if np.sum(mask) == 0:
        return intensity
    blurred = cv2.GaussianBlur(intensity.astype(np.float32), ksize=(0, 0), sigmaX=sigma_x, sigmaY=0)
    apply_region = band if band is not None else (mask > 0)
    out = intensity.copy().astype(np.float32)
    out[apply_region] = blurred[apply_region]
    return out


def add_tumor_noise(intensity: np.ndarray, mask: np.ndarray, low_sigma: float = 4.0, low_amp: float = 0.08, high_std: float = 1.0, band: np.ndarray | None = None) -> np.ndarray:
    if np.sum(mask) == 0:
        return intensity
    h, w = intensity.shape
    out = intensity.astype(np.float32)
    # Low-frequency shading across entire lesion for subtle heterogeneity
    low = np.random.randn(h, w).astype(np.float32)
    low = gaussian_filter(low, sigma=low_sigma)
    if np.max(np.abs(low)) > 0:
        low = low / (np.max(np.abs(low)) + 1e-6)
    lesion_region = (mask > 0)
    out[lesion_region] = out[lesion_region] * (1.0 + low_amp * low[lesion_region])
    # High-frequency noise only near edges to avoid ruining core textures
    apply_region = band if band is not None else lesion_region
    hi = np.random.normal(0.0, high_std, size=(h, w)).astype(np.float32)
    out[apply_region] = out[apply_region] + hi[apply_region]
    return np.clip(out, 0, 255)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = distance_transform_edt(mask > 0)
    outside = distance_transform_edt(mask == 0)
    return inside - outside


def perturb_mask_boundary(mask: np.ndarray, amplitude_px: float = 2.5, sigma: float = 5.0) -> np.ndarray:
    """Make boundaries organic by thresholding signed distance + smooth noise."""
    if np.sum(mask) == 0:
        return mask
    sd = signed_distance(mask)
    noise = np.random.randn(*mask.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=sigma)
    # Normalize noise to [-1,1] then scale by amplitude in pixels
    if np.max(np.abs(noise)) > 0:
        noise = noise / (np.max(np.abs(noise)) + 1e-6)
    noisy_sd = sd + amplitude_px * noise
    return (noisy_sd > 0).astype(np.uint8)


def is_square_like(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return False
    minr, maxr = int(ys.min()), int(ys.max())
    minc, maxc = int(xs.min()), int(xs.max())
    h = max(1, maxr - minr + 1)
    w = max(1, maxc - minc + 1)
    area = ys.size
    extent = float(area) / float(h * w)
    aspect = float(w) / float(h)
    aspect = aspect if aspect >= 1.0 else 1.0 / aspect
    return (extent >= 0.90) and (aspect <= 1.25)


def has_rectilinear_artifacts(mask: np.ndarray,
                              angle_tol_deg: float = 12.0,
                              min_axis_px: int = 10,
                              longest_axis_frac: float = 0.18,
                              axis_total_frac: float = 0.45,
                              right_angle_tol: float = 12.0) -> bool:
    """Detect harshly any straight axis-aligned edges and 90° corners.

    - Reject if the longest axis-aligned boundary segment is > longest_axis_frac of perimeter
      or longer than min_axis_px.
    - Reject if the total axis-aligned boundary length fraction exceeds axis_total_frac.
    - Reject if any internal corner angle is within right_angle_tol of 90°.
    """
    if np.sum(mask) == 0:
        return False
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return False
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return False
    pts = cnt.squeeze(1).astype(np.float32)
    perim = float(cv2.arcLength(cnt, True))
    # Edge orientations and lengths
    diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1) + 1e-6
    angles = np.degrees(np.abs(np.arctan2(diffs[:, 1], diffs[:, 0])))  # 0=horizontal, 90=vertical
    axis_mask = (angles < angle_tol_deg) | (np.abs(angles - 90.0) < angle_tol_deg)
    axis_len_total = float(np.sum(seg_lens[axis_mask]))
    # Longest consecutive axis-aligned run
    max_run = 0.0
    run = 0.0
    for is_axis, L in zip(axis_mask, seg_lens):
        if is_axis:
            run += L
            max_run = max(max_run, run)
        else:
            run = 0.0
    longest_axis_ok = max_run < max(min_axis_px, longest_axis_frac * perim)
    total_axis_ok = (axis_len_total / perim) < axis_total_frac

    # Right-angle corners via polygonal approximation
    eps = 0.01 * perim
    approx = cv2.approxPolyDP(cnt, eps, True).squeeze(1)
    right_angle_found = False
    if approx.shape[0] >= 3:
        n = approx.shape[0]
        for i in range(n):
            p0 = approx[(i - 1) % n].astype(np.float32)
            p1 = approx[i].astype(np.float32)
            p2 = approx[(i + 1) % n].astype(np.float32)
            v1 = p0 - p1
            v2 = p2 - p1
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            if abs(ang - 90.0) <= right_angle_tol:
                right_angle_found = True
                break

    # Be harsh: any failure rejects
    if not longest_axis_ok:
        return True
    if not total_axis_ok:
        return True
    if right_angle_found:
        return True
    return False


def darken_tumor_relative(
    control_img: np.ndarray,
    placed_img: np.ndarray,
    placed_mask: np.ndarray,
    bg_ring_width: int = 6,
    target_ratio: float = 0.85,
    min_scale: float = 0.75,
    max_scale: float = 0.95,
) -> np.ndarray:
    """Darken tumor slightly relative to local background while preserving texture.

    Computes background median in an outer ring and scales tumor intensities so
    the tumor median approaches target_ratio * bg_median, clamped to [min_scale, max_scale].
    """
    mask_bool = placed_mask > 0
    if not np.any(mask_bool):
        return placed_img
    # Background ring outside the lesion
    ring = binary_dilation(mask_bool, disk(bg_ring_width)) & (~mask_bool)
    bg_vals = control_img[ring]
    if bg_vals.size == 0:
        return placed_img
    bg_med = float(np.median(bg_vals))
    tum_vals = placed_img[mask_bool]
    if tum_vals.size == 0:
        return placed_img
    t_med = float(np.median(tum_vals))
    if t_med <= 1e-6:
        return placed_img
    target = target_ratio * bg_med
    scale = np.clip(target / t_med, min_scale, max_scale)
    out = placed_img.astype(np.float32)
    out[mask_bool] = out[mask_bool] * scale
    return np.clip(out, 0, 255)


def horizontal_deband_inside(intensity: np.ndarray, mask: np.ndarray, sigma_x: float = 1.8, blend: float = 0.5) -> np.ndarray:
    if np.sum(mask) == 0:
        return intensity
    blurred = cv2.GaussianBlur(intensity.astype(np.float32), ksize=(0, 0), sigmaX=sigma_x, sigmaY=0)
    out = intensity.astype(np.float32)
    m = mask > 0
    out[m] = (1.0 - blend) * out[m] + blend * blurred[m]
    return np.clip(out, 0, 255)


def fill_missing_inside_mask(intensity: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = intensity.copy().astype(np.float32)
    # Identify missing (zero) pixels strictly inside the mask
    missing = (mask > 0) & (filled <= 0)
    if not np.any(missing):
        return intensity
    # Nearest neighbor fill from available non-zero within mask
    available = (mask > 0) & (filled > 0)
    if not np.any(available):
        return intensity  # nothing to fill with
    # Distance transform to nearest available pixel
    dist, (yy, xx) = distance_transform_edt(~available, return_indices=True)
    filled[missing] = filled[yy[missing], xx[missing]]
    return filled.astype(intensity.dtype)


def place_mask(mask: np.ndarray, center: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
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
    # Compute same translation using the non-zero region of the image
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


# ----------------------------------------------------------------------------
# Intensity adjustment to make tumors sufficiently "hot"
# ----------------------------------------------------------------------------


def _ring_mask(binary_mask: np.ndarray, width: int = 5) -> np.ndarray:
    ring = binary_dilation(binary_mask, disk(width)) & ~binary_mask
    return ring


def boost_tumor_intensity(
    control_img: np.ndarray,
    placed_tumor_img: np.ndarray,
    placed_mask: np.ndarray,
    target_peak_factor: float = 1.5,
    min_peak_add: float = 25.0,
    use_clahe: bool = False,
) -> np.ndarray:
    """Boost tumor intensities without saturating to pure white.

    Uses percentile mapping with caps: maps tumor P5->bg_high_low and
    tumor P95->bg_high_high, then optionally applies mild CLAHE. This
    avoids compressing texture and reduces saturation.
    """
    tumor = placed_tumor_img.astype(np.float32)
    mask = (placed_mask > 0)
    if not np.any(mask):
        return placed_tumor_img

    ring = _ring_mask(mask, width=5)
    bg_vals = control_img[ring]
    if bg_vals.size == 0:
        bg_vals = control_img[~mask]
    if bg_vals.size == 0:
        bg_p80 = 50.0
        bg_p95 = 70.0
        bg_p99 = 90.0
    else:
        bg_p80 = float(np.percentile(bg_vals, 80))
        bg_p95 = float(np.percentile(bg_vals, 95))
        bg_p99 = float(np.percentile(bg_vals, 99))

    tumor_vals = tumor[mask]
    if tumor_vals.size == 0:
        return placed_tumor_img
    t_p5 = float(np.percentile(tumor_vals, 5))
    t_p95 = float(np.percentile(tumor_vals, 95))
    if t_p95 <= t_p5 + 1e-3:
        t_p5 = float(np.min(tumor_vals))
        t_p95 = float(np.max(tumor_vals))
        if t_p95 <= t_p5 + 1e-3:
            return placed_tumor_img

    # Define targets: keep texture range but ensure hotness
    high_target = min(max(bg_p95 * target_peak_factor, bg_p95 + min_peak_add), 235.0)
    low_target = max(min(bg_p80, high_target - 40.0), 10.0)

    # Affine mapping
    scale = (high_target - low_target) / max(t_p95 - t_p5, 1.0)
    shift = low_target - t_p5 * scale
    tumor_mapped = tumor * scale + shift
    tumor_mapped = np.clip(tumor_mapped, 0, 255)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        patch_u8 = tumor_mapped.astype(np.uint8)
        patch_eq = clahe.apply(patch_u8)
        tumor_mapped = patch_eq.astype(np.float32)

    adjusted = placed_tumor_img.copy().astype(np.float32)
    adjusted[mask] = tumor_mapped[mask]
    return adjusted

# ----------------------------------------------------------------------------
# β-VAE for grayscale tumor patches (64×64)
# ----------------------------------------------------------------------------


class BetaVAEGrayscale(nn.Module):
    def __init__(self, latent_dim: int = 16, beta: float = 4.0):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 32×32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16×16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8×8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 4×4
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8×8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16×16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32×32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 64×64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Weighted BCE: emphasize tumor (non-zero) pixels, background near 0
        weight = torch.where(x > 0, torch.tensor(3.0, device=x.device), torch.tensor(1.0, device=x.device))
        bce = nn.functional.binary_cross_entropy(recon_x, x, weight=weight, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + self.beta * kld


_VAE_TEX_MODEL: Optional[BetaVAEGrayscale] = None
_VAE_TEX_DEVICE: Optional[torch.device] = None


def train_texture_vae(patches: List[np.ndarray], epochs: int, checkpoint: Optional[str], device: str) -> BetaVAEGrayscale:
    print("⚙️  Training grayscale β-VAE on tumor patches…")
    size = 64
    tensors = []
    for p in patches:
        # Expect p as float32 [0,1] 64×64
        if p.shape != (size, size):
            p = cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)
        tensors.append(p[None, :, :])
    data = torch.tensor(np.stack(tensors, axis=0), dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

    model = BetaVAEGrayscale().to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimiser.step()
            total += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch+1}/{epochs} loss={total / len(loader.dataset):.4f}")
    if checkpoint:
        torch.save(model.state_dict(), checkpoint)
    return model


def get_texture_vae(patches: List[np.ndarray], checkpoint_path: str, auto_train: bool, epochs: int) -> Optional[BetaVAEGrayscale]:
    global _VAE_TEX_MODEL, _VAE_TEX_DEVICE
    if _VAE_TEX_MODEL is not None:
        return _VAE_TEX_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _VAE_TEX_DEVICE = device
    model = BetaVAEGrayscale().to(device)
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            _VAE_TEX_MODEL = model
            print(f"✅ Loaded texture VAE checkpoint from {checkpoint_path}")
            return model
        except Exception as e:
            print(f"⚠️  Failed to load texture VAE checkpoint: {e}. Will retrain if allowed.")
    if not auto_train:
        print("⚠️  No texture VAE checkpoint and auto_train disabled. Skipping VAE path.")
        return None
    model = train_texture_vae(patches, epochs=epochs, checkpoint=str(ckpt), device=str(device))
    model.eval()
    _VAE_TEX_MODEL = model
    return model


# ----------------------------------------------------------------------------
# Tumor extraction helpers
# ----------------------------------------------------------------------------


def extract_lesion_regions(mask_slice: np.ndarray) -> List[measure._regionprops.RegionProperties]:
    labeled = measure.label(mask_slice > 0)
    return list(measure.regionprops(labeled))


def crop_region(img: np.ndarray, region: measure._regionprops.RegionProperties, pad: int = 4) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    minr, minc, maxr, maxc = region.bbox
    minr = max(0, minr - pad)
    minc = max(0, minc - pad)
    maxr = min(img.shape[0], maxr + pad)
    maxc = min(img.shape[1], maxc + pad)
    return img[minr:maxr, minc:maxc].copy(), (minr, minc, maxr, maxc)


def build_texture_patch(patient_img: np.ndarray, patient_mask: np.ndarray, region: measure._regionprops.RegionProperties, size: int = 64, tumor_is_dark: bool = False) -> np.ndarray:
    mask_crop, _ = crop_region(patient_mask.astype(np.uint8), region)
    img_crop, _ = crop_region(patient_img.astype(np.float32), region)
    mask_crop = (mask_crop > 0).astype(np.uint8)
    # Keep image only inside mask
    tex = img_crop * (mask_crop > 0)
    # Normalize intensities inside tumor to [0,1]
    inside = tex[mask_crop > 0]
    if inside.size > 0:
        lo, hi = np.percentile(inside, [1, 99])
        if hi > lo:
            tex = np.clip((tex - lo) / (hi - lo), 0, 1)
        else:
            tex = np.clip(tex / max(1e-6, hi), 0, 1)
    # Enforce black background
    tex[mask_crop == 0] = 0.0
    # If tumors should be dark for training (not default), invert inside mask only
    if tumor_is_dark:
        tex = (1.0 - tex) * (mask_crop > 0)
    tex = cv2.resize(tex, (size, size), interpolation=cv2.INTER_AREA)
    return tex.astype(np.float32)


# ----------------------------------------------------------------------------
# Core synthesis for a single slice (returns overlay image and mask)
# ----------------------------------------------------------------------------


def synthesize_slice_with_look(
    control_img: np.ndarray,
    body_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    heatmap: np.ndarray,
    count_cdf: np.ndarray,
    area_cdf: np.ndarray,
    patient_imgs: List[np.ndarray],
    patient_masks: List[np.ndarray],
    use_texture_vae: bool,
    texture_patches: List[np.ndarray],
    max_attempts: int = 30,
    debug_vae_output_dir: Optional[Path] = None,
    subject_name: str = "unknown",
    min_total_pixels: int = 220,
    max_lesions: int = 6,
    min_large_lesion_area: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = control_img.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = control_img.copy().astype(np.float32)

    n_lesions = min(sample_from_cdf(count_cdf), max_lesions)
    # Extra edge buffer to keep lesions a bit farther from body boundary
    body_eroded = binary_erosion(body_mask, disk(5))

    has_large_lesion = False
    for lesion_idx in range(n_lesions):
        for _attempt in range(max_attempts):
            # Choose lesion source (VAE disabled per request)
            if False and use_texture_vae and _VAE_TEX_MODEL is not None and random.random() < 0.7:
                # VAE-generated 64×64 intensity patch in [0,1], black background with tumor signal
                with torch.no_grad():
                    z = torch.randn(1, 16, device=_VAE_TEX_DEVICE)
                    out = _VAE_TEX_MODEL.decode(z).cpu().numpy()[0, 0]
                
                # Debug: Save raw VAE output if requested
                if debug_vae_output_dir is not None:
                    debug_vae_output_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = debug_vae_output_dir / f"{subject_name}_vae_raw_{_attempt}.png"
                    # Save as grayscale image to see what VAE actually produces
                    cv2.imwrite(str(debug_path), (out * 255).astype(np.uint8))
                    print(f"DEBUG: Saved raw VAE output to {debug_path}")
                    print(f"DEBUG: VAE output range: [{out.min():.3f}, {out.max():.3f}]")
                    print(f"DEBUG: VAE output mean: {out.mean():.3f}")
                
                # Output is black background with tumor intensity (high=hot)
                out_intensity = out
                # Derive binary mask from intensity
                lesion_mask_small = (out_intensity > 0.2).astype(np.uint8)
                lesion_img_small = (out_intensity * 255.0).astype(np.float32)
                # Scale to target area
                target_area = sample_from_cdf(area_cdf)
                curr_area = int(np.sum(lesion_mask_small))
                if curr_area == 0:
                    continue
                scale_factor = math.sqrt(target_area / max(1, curr_area))
                new_h = max(4, int(lesion_mask_small.shape[0] * scale_factor))
                new_w = max(4, int(lesion_mask_small.shape[1] * scale_factor))
                lesion_mask = cv2.resize(lesion_mask_small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                lesion_img = cv2.resize(lesion_img_small, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Deterministic: pick a random real lesion region and apply same transforms to mask and texture
                idx = random.randrange(len(patient_masks))
                pmask = patient_masks[idx]
                pimg = patient_imgs[idx]
                regions = extract_lesion_regions(pmask)
                if len(regions) == 0:
                    continue
                region = random.choice(regions)
                mask_crop, _ = crop_region((pmask > 0).astype(np.uint8), region, pad=2)
                img_crop, _ = crop_region(pimg.astype(np.float32), region, pad=2)
                # Keep original patient intensities; mask outside as black background
                img_crop *= (mask_crop > 0)
                img_crop[mask_crop == 0] = 0
                # Geometric parameters
                scale = random.uniform(0.7, 1.3)
                rot = random.uniform(-25, 25)
                shear = random.uniform(-10, 10)
                M = affine_matrices(scale, rot, shear, mask_crop.shape[0], mask_crop.shape[1])
                mask_t, img_t = warp_affine_pair(mask_crop, img_crop, M)
                dx, dy = elastic_fields(
                    mask_t.shape,
                    alpha=float(random.uniform(25.0, 45.0)),
                    sigma=float(random.uniform(3.0, 6.0)),
                )
                mask_t, img_t = elastic_deform_pair(mask_t, img_t, dx, dy)
                # Cut and stitch with shared plan
                pieces = random.randint(3, 6)
                plan = build_cut_stitch_plan(mask_t.shape[1], pieces, max_offset=12)
                mask_t, img_t = apply_cut_stitch_plan(mask_t, img_t, plan)
                # Morphology tweaks on mask only
                kernel = np.ones((3, 3), np.uint8)
                if random.random() < 0.5:
                    mask_t = cv2.erode(mask_t, kernel, iterations=random.randint(0, 2))
                if random.random() < 0.5:
                    mask_t = cv2.dilate(mask_t, kernel, iterations=random.randint(0, 2))
                # Enforce roundness with organic perturbations: perturb signed distance field
                if np.sum(mask_t) > 0:
                    mask_t = perturb_mask_boundary(mask_t, amplitude_px=random.uniform(2.0, 4.0), sigma=random.uniform(4.0, 7.0))
                # Backup rounding ops (applied with lower prob) to avoid sharp corners
                if random.random() < 0.5:
                    mask_t = cv2.GaussianBlur((mask_t * 255).astype(np.uint8), (0, 0), sigmaX=random.uniform(1.0, 1.8))
                    mask_t = (mask_t > 127).astype(np.uint8)
                if random.random() < 0.6:
                    ksz = random.choice([5, 7])
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                    mask_t = cv2.morphologyEx(mask_t, cv2.MORPH_CLOSE, kernel_close)
                # Micro-edge jitter: introduce tiny indentations/protrusions along the boundary
                if random.random() < 0.95 and np.sum(mask_t) > 0:
                    ys, xs = np.where(mask_t > 0)
                    miny, maxy = int(ys.min()), int(ys.max())
                    # every 2–4 pixels across rows, flip pixels at border for organic edges
                    for row in range(miny, maxy + 1, random.randint(2, 4)):
                        row_pixels = np.where(mask_t[row] > 0)[0]
                        if row_pixels.size == 0:
                            continue
                        left = int(row_pixels.min())
                        right = int(row_pixels.max())
                        if random.random() < 0.7 and left - 1 >= 0:
                            mask_t[row, max(0, left - 1)] = 1  # protrusion
                        if random.random() < 0.7 and right + 1 < mask_t.shape[1]:
                            mask_t[row, min(mask_t.shape[1]-1, right + 1)] = 1  # protrusion
                        if random.random() < 0.5:
                            mask_t[row, left] = 0  # indentation
                        if random.random() < 0.5:
                            mask_t[row, right] = 0  # indentation
                img_t = img_t * (mask_t > 0)
                img_t = fill_missing_inside_mask(img_t, mask_t)
                # Preserve core texture: only feather/noise near edges using an edge band
                band = compute_edge_band(mask_t, inner_width=2, outer_width=6)
                img_t = feather_horizontal(img_t, mask_t, sigma_x=random.uniform(1.2, 2.5), band=band)
                img_t = add_tumor_noise(img_t, mask_t, low_sigma=random.uniform(3.0, 6.0), low_amp=0.08, high_std=1.0, band=band)
                lesion_mask = mask_t.astype(np.uint8)
                lesion_img = img_t.astype(np.float32)
                
                # Rescale to match target area with stronger bias to large lesions
                if lesion_idx == 0:
                    target_area = sample_from_cdf_tail(area_cdf, 0.96, 0.997)
                else:
                    if random.random() < 0.60:
                        target_area = sample_from_cdf_tail(area_cdf, 0.92, 0.995)
                    else:
                        target_area = sample_from_cdf(area_cdf)
                curr_area = int(np.sum(lesion_mask))
                if curr_area == 0:
                    continue
                scale_factor = math.sqrt(target_area / max(1, curr_area))
                center = (lesion_mask.shape[1] / 2.0, lesion_mask.shape[0] / 2.0)
                M_scale = cv2.getRotationMatrix2D(center, 0, scale_factor)
                lesion_mask, lesion_img = warp_affine_pair(lesion_mask, lesion_img, M_scale)

            # Sample location from heatmap (reuse deterministic prior like original generator)
            flat_prob = heatmap.flatten().astype(np.float64)
            if flat_prob.sum() <= 0:
                flat_prob = np.ones_like(flat_prob) / flat_prob.size
            else:
                flat_prob /= flat_prob.sum()
            idx = np.random.choice(flat_prob.size, p=flat_prob)
            cy, cx = divmod(idx, w)

            # Place
            placed_mask = place_mask(lesion_mask, (cx, cy), (h, w))
            # To avoid blocky edges, slightly blur the placed mask edge before applying on intensity
            if np.sum(placed_mask) > 0:
                edge = cv2.GaussianBlur((placed_mask * 255).astype(np.uint8), (0, 0), sigmaX=random.uniform(1.2, 2.0))
                soft_mask = np.clip(edge.astype(np.float32) / 255.0, 0, 1)
            else:
                soft_mask = placed_mask.astype(np.float32)
            placed_img = place_image(lesion_img, (cx, cy), (h, w))
            # Deband to ensure no vertical line remains inside large tumors
            if np.sum(placed_mask) > 250:
                placed_img = horizontal_deband_inside(placed_img, placed_mask, sigma_x=random.uniform(1.4, 2.2), blend=0.6)
            # Soft-apply the tumor image using soft edges
            overlay_candidate = overlay.copy()
            overlay_candidate = overlay_candidate * (1.0 - soft_mask) + placed_img * soft_mask

            # Validate placement (and reject overly square shapes)
            if not np.all((placed_mask & ~body_eroded) == 0):
                continue
            if np.any(placed_mask & forbidden_mask):
                continue
            if np.any(placed_mask & final_mask):
                continue
            area = int(np.sum(placed_mask))
            if not (30 <= area <= 4000):
                continue
            # QC eccentricity
            ys, xs = np.where(placed_mask)
            if ys.size >= 10:
                coords = np.column_stack([xs, ys])
                cov = np.cov(coords, rowvar=False)
                evals = np.linalg.eigvalsh(cov)
                if evals[1] > 0 and (1 - evals[0] / evals[1]) > 0.98:
                    continue
            # Anti-rectangle QC: harshly reject shapes with straight 0/90° edges or 90° corners
            if is_square_like(placed_mask) or has_rectilinear_artifacts(placed_mask):
                continue

            # Commit by pasting original tumor intensities (no boosting)
            final_mask |= placed_mask
            # Slightly darken relative to local background for better contrast
            overlay = overlay_candidate
            overlay = darken_tumor_relative(control_img, overlay, placed_mask, bg_ring_width=6, target_ratio=0.82, min_scale=0.75, max_scale=0.95)
            if area >= min_large_lesion_area:
                has_large_lesion = True
            break

    # Final QC: ensure a slightly larger dilation remains inside body
    dilated = binary_dilation(final_mask, disk(3))
    if not np.all(dilated <= body_mask):
        final_mask = final_mask & body_mask

    # Ensure a minimum amount of tumour pixels overall; if not, try to add a few more lesions
    extra_trials = 0
    while int(np.sum(final_mask)) < min_total_pixels and extra_trials < 25:
        extra_trials += 1
        # Deterministic extra placement (same as main branch)
        idx = random.randrange(len(patient_masks))
        pmask = patient_masks[idx]
        pimg = patient_imgs[idx]
        regions = extract_lesion_regions(pmask)
        if len(regions) == 0:
            continue
        region = random.choice(regions)
        mask_crop, _ = crop_region((pmask > 0).astype(np.uint8), region, pad=2)
        img_crop, _ = crop_region(pimg.astype(np.float32), region, pad=2)
        img_crop *= (mask_crop > 0)
        img_crop[mask_crop == 0] = 0
        scale = random.uniform(0.7, 1.3)
        rot = random.uniform(-25, 25)
        shear = random.uniform(-10, 10)
        M = affine_matrices(scale, rot, shear, mask_crop.shape[0], mask_crop.shape[1])
        mask_t, img_t = warp_affine_pair(mask_crop, img_crop, M)
        dx, dy = elastic_fields(mask_t.shape, alpha=20, sigma=4)
        mask_t, img_t = elastic_deform_pair(mask_t, img_t, dx, dy)
        pieces = random.randint(2, 4)
        plan = build_cut_stitch_plan(mask_t.shape[1], pieces, max_offset=8)
        mask_t, img_t = apply_cut_stitch_plan(mask_t, img_t, plan)
        kernel = np.ones((3, 3), np.uint8)
        if random.random() < 0.5:
            mask_t = cv2.erode(mask_t, kernel, iterations=random.randint(0, 2))
        if random.random() < 0.5:
            mask_t = cv2.dilate(mask_t, kernel, iterations=random.randint(0, 2))
        img_t = img_t * (mask_t > 0)
        img_t = fill_missing_inside_mask(img_t, mask_t)
        band = compute_edge_band(mask_t, inner_width=2, outer_width=6)
        img_t = feather_horizontal(img_t, mask_t, sigma_x=random.uniform(1.0, 2.0), band=band)
        img_t = add_tumor_noise(img_t, mask_t, low_sigma=random.uniform(3.0, 6.0), low_amp=0.08, high_std=1.0, band=band)

        lesion_mask = mask_t.astype(np.uint8)
        lesion_img = img_t.astype(np.float32)

        # Target area rescale (bias extras to be larger)
        if random.random() < 0.60:
            target_area = sample_from_cdf_tail(area_cdf, 0.92, 0.995)
        else:
            target_area = sample_from_cdf(area_cdf)
        curr_area = int(np.sum(lesion_mask))
        if curr_area == 0:
            continue
        scale_factor = math.sqrt(target_area / max(1, curr_area))
        center = (lesion_mask.shape[1] / 2.0, lesion_mask.shape[0] / 2.0)
        M_scale = cv2.getRotationMatrix2D(center, 0, scale_factor)
        lesion_mask, lesion_img = warp_affine_pair(lesion_mask, lesion_img, M_scale)

        # Place at sampled valid location
        flat_prob = heatmap.flatten().astype(np.float64)
        flat_prob = flat_prob / flat_prob.sum() if flat_prob.sum() > 0 else np.ones_like(flat_prob) / flat_prob.size
        ridx = np.random.choice(flat_prob.size, p=flat_prob)
        cy, cx = divmod(ridx, w)
        placed_mask = place_mask(lesion_mask, (cx, cy), (h, w))
        placed_img = place_image(lesion_img, (cx, cy), (h, w))

        if not np.all((placed_mask & ~body_eroded) == 0):
            continue
        if np.any(placed_mask & forbidden_mask):
            continue
        if np.any(placed_mask & final_mask):
            continue
        area = int(np.sum(placed_mask))
        if not (30 <= area <= 4000):
            continue
        ys, xs = np.where(placed_mask)
        if ys.size >= 10:
            coords = np.column_stack([xs, ys])
            cov = np.cov(coords, rowvar=False)
            evals = np.linalg.eigvalsh(cov)
            if evals[1] > 0 and (1 - evals[0] / evals[1]) > 0.98:
                continue

        final_mask |= placed_mask
        overlay = np.where(placed_mask > 0, placed_img, overlay)

    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay_u8, final_mask.astype(np.uint8)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def save_comparison_plot(control_img: np.ndarray, overlaid_img: np.ndarray, mask: np.ndarray, output_path: Path, subject_name: str) -> None:
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(control_img, cmap="gray")
    axes[0].set_title(f"{subject_name}\nOriginal Control")
    axes[0].axis("off")
    axes[1].imshow(overlaid_img, cmap="gray")
    axes[1].set_title(f"{subject_name}\nControl + Synthetic Tumors")
    axes[1].axis("off")
    colored_overlay = colored_mask_overlay(control_img, mask, color=(255, 0, 0), alpha=0.35)
    axes[2].imshow(colored_overlay)
    axes[2].set_title(f"{subject_name}\nMask (expected tumor regions)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tumors with texture on control PET slices (NEW generator).")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_img_dir", required=True, type=str, help="Directory with real patient image PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save generated overlays and masks.")
    parser.add_argument("--plots_dir", type=str, default="comparison_plots_look", help="Directory to save comparison plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_controls", type=int, default=None, help="Number of control subjects to process (default: all)")
    parser.add_argument("--no_auto_train", action="store_true", help="Disable auto-training of texture VAE if checkpoint not found.")
    parser.add_argument("--vae_epochs", type=int, default=30, help="Epochs for texture VAE if checkpoint is missing.")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_tumor_texture.pth", help="Checkpoint path for texture VAE.")
    parser.add_argument("--debug_vae", action="store_true", help="Save debug VAE outputs to see what VAE produces before blending.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    controls_dir = Path(args.controls_dir)
    patients_img_dir = Path(args.patients_img_dir)
    patients_label_dir = Path(args.patients_label_dir)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load aligned patient images and masks (same basenames)
    patient_label_paths = sorted(patients_label_dir.glob("*.png"))
    if len(patient_label_paths) < 50:
        raise RuntimeError("Not enough patient masks found to build statistics.")

    print("🔧 Pre-computing location prior…")
    heatmap = build_location_prior(patient_label_paths)

    print("📊 Building empirical distributions…")
    count_cdf, area_cdf = empirical_distributions(patient_label_paths)

    print("📚 Pairing patient images and labels by numeric ID and padding…")
    # Build robust pairing by numeric id in filenames
    def extract_numeric_id(name: str) -> Optional[str]:
        import re
        matches = re.findall(r"(\d+)", name)
        return matches[-1] if matches else None

    img_paths = {extract_numeric_id(p.name): p for p in sorted(patients_img_dir.glob('*.png')) if extract_numeric_id(p.name) is not None}
    lbl_paths = {extract_numeric_id(p.name): p for p in sorted(patients_label_dir.glob('*.png')) if extract_numeric_id(p.name) is not None}
    common_ids = sorted(set(img_paths.keys()) & set(lbl_paths.keys()), key=lambda x: int(x))
    if len(common_ids) == 0:
        raise RuntimeError("No matching patient image/label pairs found by numeric ID. Check directory contents.")

    # Limit to first ~180 pairs to mirror original behavior
    common_ids = common_ids[:180]

    patient_imgs: List[np.ndarray] = []
    patient_masks: List[np.ndarray] = []
    texture_patches: List[np.ndarray] = []  # for VAE
    for pid in tqdm(common_ids, desc="Patients"):
        img_path = img_paths[pid]
        lbl_path = lbl_paths[pid]
        mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or img is None:
            continue
        mask = pad_to_target(mask)
        img = pad_to_target(img)
        patient_masks.append((mask > 0).astype(np.uint8))
        patient_imgs.append(img.astype(np.float32))
        # Collect texture patches for VAE
        for region in extract_lesion_regions(mask):
            patch = build_texture_patch(img, mask, region, size=64)
            if np.sum(patch) > 0:
                texture_patches.append(patch)

    # Load controls
    control_paths = sorted(controls_dir.glob("*.png"))
    if args.num_controls is not None:
        control_paths = control_paths[: args.num_controls]
    print(f"🚀 Generating textured tumors for {len(control_paths)} control slices…")

    # VAE disabled per request
    vae_model = None

    for img_path in tqdm(control_paths, desc="Controls"):
        pet = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pet is None:
            continue
        pet = pad_to_target(pet)
        body = compute_body_mask(pet)
        forbidden = compute_forbidden_mask(pet, body)

        # Define subject name early for debug and plotting
        subject_name = img_path.stem

        debug_dir = None
        if args.debug_vae:
            debug_dir = Path("debug_vae_outputs")
        
        overlaid_img, look_mask = synthesize_slice_with_look(
            control_img=pet,
            body_mask=body,
            forbidden_mask=forbidden,
            heatmap=heatmap,
            count_cdf=count_cdf,
            area_cdf=area_cdf,
            patient_imgs=patient_imgs,
            patient_masks=patient_masks,
            use_texture_vae=False,
            texture_patches=texture_patches,
            debug_vae_output_dir=debug_dir,
            subject_name=subject_name,
        )

        mask_out = output_dir / f"{subject_name}_look_mask.png"
        overlay_out = output_dir / f"{subject_name}_look_overlay.png"
        plot_out = plots_dir / f"{subject_name}_look_comparison.png"

        cv2.imwrite(str(mask_out), (look_mask * 255).astype(np.uint8))
        cv2.imwrite(str(overlay_out), overlaid_img)
        save_comparison_plot(pet, overlaid_img, look_mask, plot_out, subject_name)

    print(f"✅ Synthetic look masks and overlays saved to {output_dir}")
    print(f"✅ Comparison plots saved to {plots_dir}")


if __name__ == "__main__":
    main()

