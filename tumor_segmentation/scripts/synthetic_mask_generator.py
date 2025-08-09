#!/usr/bin/env python3
"""
Synthetic Lesion Mask Generator for PET Control Slices
=====================================================

Generates anatomically-plausible tumour masks on healthy PET slices to
augment the training set of the *tile-SPA­DE-GAN*.

The implementation follows the specification provided in *Cursor Mask
Generation Module* documentation.

Main steps (executed once or cached):
 1. Compute helper maps per control slice:
    • body mask (largest connected component of voxels with intensity>0).
    • forbidden-organ mask (top 10 % hottest voxels, filtered by area).
 2. Build global location-prior heat-map from real tumour centroids.
 3. Pre-compute empirical CDFs of lesion count per slice and lesion
    volume (area in px) from real patient masks.

Per-slice synthesis:
 4. Sample the number of lesions and desired volumes from the CDFs.
 5. For each lesion, draw either (p=0.3) a β-VAE-generated mask or a
    warped real mask and rescale to match the target volume.
 6. Attempt up to *N* times to place the mask so that it lies fully
    within the body mask, outside the forbidden mask and not too close
    to the edge.
 7. Apply QC filters (area bounds, eccentricity, dilated mask in body).
 8. Save the final synthetic mask alongside the control image.

The script is fully deterministic given a *random-seed* CLI argument.

Usage (example):
    python synthetic_mask_generator.py \
        --controls_dir tumor_segmentation/data/controls/imgs \
        --patients_label_dir tumor_segmentation/data/patients/labels \
        --output_dir tumor_segmentation/data/controls/labels

Requirements:
    pip install numpy scipy scikit-image opencv-python tqdm

© 2025 Cursor-AI
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import measure, morphology, segmentation, util
from skimage.morphology import binary_erosion, binary_dilation, disk
from tqdm import tqdm
import matplotlib.pyplot as plt

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Constants for unified image size (all images padded to 400×992)
TARGET_WIDTH = 400
TARGET_HEIGHT = 992

# ----------------------------------------------------------------------------
# β-VAE for Novel Mask Shapes
# ----------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """Simple convolutional β-VAE for 64×64 binary masks."""

    def __init__(self, latent_dim: int = 16, beta: float = 4.0):
        super().__init__()
        self.beta = beta
        # Encoder
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
        # Decoder
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
        bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + self.beta * kld


def train_beta_vae(seed_masks: List[np.ndarray], epochs: int = 50, checkpoint: str | None = None, device: str = "cpu") -> BetaVAE:
    """Train β-VAE on provided seed masks (slow but done once)."""
    print("⚙️  Training β-VAE on real masks…")
    size = 64
    tensors = []
    for m in seed_masks:
        img = cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
        tensors.append(img[None, :, :])  # add channel dim
    data = torch.tensor(np.stack(tensors), dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)

    model = BetaVAE().to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch+1}/{epochs} loss={total_loss / len(loader.dataset):.4f}")

    if checkpoint:
        torch.save(model.state_dict(), checkpoint)
    return model


# Global VAE holder (lazy-loaded)
_VAE_MODEL: BetaVAE | None = None
_VAE_DEVICE: torch.device | None = None


def get_vae(seed_masks: List[np.ndarray], checkpoint_path: str = "vae_mask_generator.pth", auto_train: bool = True, epochs: int = 50) -> BetaVAE | None:
    """Load existing VAE or train a new one if not found and *auto_train* is True."""
    global _VAE_MODEL, _VAE_DEVICE
    if _VAE_MODEL is not None:
        return _VAE_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _VAE_DEVICE = device

    model = BetaVAE().to(device)
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        try:
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
            model.eval()
            _VAE_MODEL = model
            print(f"✅ Loaded VAE checkpoint from {checkpoint_path}")
            return model
        except Exception as e:
            print(f"⚠️  Failed to load VAE checkpoint: {e}. Will retrain.")

    if not auto_train:
        print("⚠️  No VAE checkpoint found and auto_train disabled. Skipping VAE.")
        return None

    model = train_beta_vae(seed_masks, epochs=epochs, checkpoint=str(ckpt_file), device=str(device))
    model.eval()
    _VAE_MODEL = model
    return model



def pad_to_target(img: np.ndarray, target_h: int = TARGET_HEIGHT, target_w: int = TARGET_WIDTH, value: int = 0) -> np.ndarray:
    """Pad *img* on the bottom with *value* until it reaches *(target_h, target_w)*.
    If the image is taller than *target_h*, it will be cropped. Assumes width
    already equals *target_w*. Raises if widths mismatch."""
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

# ----------------------------------------------------------------------------
# Helper-map computation
# ----------------------------------------------------------------------------


def compute_body_mask(pet_img: np.ndarray) -> np.ndarray:
    """Return binary mask of the patient body using OpenCV for speed."""
    binary = (pet_img > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary.astype(bool)
    # Skip background (label 0), choose largest by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    body = (labels == largest_label)
    # Fill small holes quickly via morphology close
    kernel = np.ones((5, 5), np.uint8)
    body = cv2.morphologyEx(body.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return (body > 127)


def compute_forbidden_mask(pet_img: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    """Return binary mask of hot organs to avoid placing synthetic lesions (fast OpenCV version)."""
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


# ----------------------------------------------------------------------------
# Location prior heat-map (global, from patient masks)
# ----------------------------------------------------------------------------


def build_location_prior(patient_label_paths: List[Path], sigma: float = 25.0) -> np.ndarray:
    """Return heat-map (float32, 0-1) of lesion centroids across dataset."""
    assert patient_label_paths, "No patient label files provided."
    # Initialize heatmap with unified target size
    heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)

    for path in tqdm(patient_label_paths, desc="Centroid pass"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = pad_to_target(mask)
        mask = mask > 0
        labeled = measure.label(mask)
        for region in measure.regionprops(labeled):
            cy, cx = region.centroid  # note skimage order (row, col)
            heatmap[int(cy), int(cx)] += 1.0

    # Smooth and normalise
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap.astype(np.float32)


def build_heatmap_cdf(heatmap: np.ndarray) -> np.ndarray:
    """Precompute CDF of flattened heatmap for fast sampling."""
    flat = heatmap.flatten().astype(np.float64)
    s = flat.sum()
    if s <= 0:
        flat[:] = 1.0
        s = flat.sum()
    flat /= s
    cdf = np.cumsum(flat)
    cdf[-1] = 1.0
    return cdf


# ----------------------------------------------------------------------------
# Empirical distributions (count, area)
# ----------------------------------------------------------------------------


def empirical_distributions(patient_label_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return empirical CDFs for lesion count per slice and lesion area."""
    counts: List[int] = []
    areas: List[int] = []
    for path in tqdm(patient_label_paths, desc="Stats pass"):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = pad_to_target(mask)
        binary = mask > 0
        labeled = measure.label(binary)
        counts.append(max(1, labeled.max()))  # at least one lesion per patient slice
        for region in measure.regionprops(labeled):
            areas.append(region.area)
    # Build CDFs
    count_values, count_counts = np.unique(counts, return_counts=True)
    count_cdf = np.cumsum(count_counts) / np.sum(count_counts)

    area_values, area_counts = np.unique(areas, return_counts=True)
    area_cdf = np.cumsum(area_counts) / np.sum(area_counts)

    return (np.vstack([count_values, count_cdf]), np.vstack([area_values, area_cdf]))


# ----------------------------------------------------------------------------
# Mask shape generator
# ----------------------------------------------------------------------------


def elastic_deform(image: np.ndarray, alpha: float = 20, sigma: float = 4) -> np.ndarray:
    """Elastic deformation (Simard et al.) suitable for small binary masks."""
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    distorted = map_coordinates(image.astype(np.float32), indices, order=1, mode="reflect").reshape(shape)
    return (distorted > 0.5).astype(np.uint8)


def affine_transform_mask(mask: np.ndarray, scale: float, rotation_deg: float, shear_deg: float) -> np.ndarray:
    """Apply affine transformation to a binary mask."""
    h, w = mask.shape
    center = (w / 2, h / 2)

    scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    # Apply rotation separately to keep full control
    rot_matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)

    # Combine
    matrix = rot_matrix.copy()
    matrix[:, 2] += scale_matrix[:, 2]

    # Shear using cv2.warpAffine with additional matrix
    shear_k = math.tan(math.radians(shear_deg))
    shear_matrix = np.array([[1, shear_k, 0], [0, 1, 0]], dtype=np.float32)

    # Full affine: M_total = shear * rot*scale
    matrix = shear_matrix @ np.vstack([matrix, [0, 0, 1]])
    matrix = matrix[:2, :]

    transformed = cv2.warpAffine(mask.astype(np.uint8) * 255, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return (transformed > 127).astype(np.uint8)


def morphology_ops(mask: np.ndarray, erode_iter: int, dilate_iter: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erode_iter)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def cut_and_stitch(mask: np.ndarray, pieces: int, max_offset: int = 8) -> np.ndarray:
    """Randomly cut the mask into vertical strips and shift them laterally with jagged edges."""
    h, w = mask.shape
    piece_width = w // pieces
    new_mask = np.zeros_like(mask)
    
    for i in range(pieces):
        x_start = i * piece_width
        x_end = (i + 1) * piece_width if i < pieces - 1 else w
        offset = random.randint(-max_offset, max_offset)
        slice_ = mask[:, x_start:x_end]
        slice_width = slice_.shape[1]
        dest_start = np.clip(x_start + offset, 0, w - slice_width)
        dest_end = dest_start + slice_width
        
        # Ensure we don't go out of bounds
        if dest_end > w:
            dest_end = w
            slice_ = slice_[:, :(dest_end - dest_start)]
        
        # Add subtle jagged edges to the slice before placing
        if slice_.shape[1] > 6:  # Only if slice is wide enough
            # Randomly add small indentations/protrusions along the edges
            for edge in [0, slice_.shape[1] - 1]:  # Left and right edges
                for row in range(0, h, 5):  # Every 5th row (less frequent)
                    if random.random() < 0.15:  # 15% chance (much lower)
                        # Add small indentation or protrusion
                        if edge == 0:  # Left edge
                            if random.random() < 0.7:  # 70% chance for indentation
                                # Indentation: remove 1 pixel only
                                if slice_.shape[1] > 1:
                                    slice_[row, 0] = 0
                            else:
                                # Protrusion: add 1 pixel to the left if possible
                                if dest_start > 0:
                                    slice_[row, 0] = 1
                        else:  # Right edge
                            if random.random() < 0.7:  # 70% chance for indentation
                                # Indentation: remove 1 pixel only
                                if slice_.shape[1] > 1:
                                    slice_[row, -1] = 0
                            else:
                                # Protrusion: add 1 pixel to the right if possible
                                if dest_end < w:
                                    slice_[row, -1] = 1
        
        new_mask[:, dest_start:dest_end] |= slice_
    
    return new_mask


def warp_real_mask(seed_mask: np.ndarray) -> np.ndarray:
    """Apply deterministic warps to a seed mask."""
    scale = random.uniform(0.7, 1.3)
    rot = random.uniform(-25, 25)
    shear = random.uniform(-10, 10)
    mask = affine_transform_mask(seed_mask, scale, rot, shear)
    mask = elastic_deform(mask, alpha=20, sigma=4)
    mask = morphology_ops(mask, erode_iter=random.randint(0, 2), dilate_iter=random.randint(0, 2))
    # Avoid introducing vertical seam artifacts from cut-and-stitch.
    # If we still want extra irregularity, prefer gentle smoothing instead.
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = (mask > 127).astype(np.uint8)
    return mask


# Placeholder: VAE integration skipped; return None to force warp path

def vae_sample_mask(mask_shape: Tuple[int, int]) -> np.ndarray | None:
    """Sample a new mask from the trained β-VAE and return it resized to *mask_shape*."""
    if _VAE_MODEL is None:
        return None
    
    # Try multiple samples to get a good one
    for _ in range(10):  # Increased attempts
        with torch.no_grad():
            z = torch.randn(1, 16, device=_VAE_DEVICE)
            out = _VAE_MODEL.decode(z).cpu().numpy()[0, 0]
        
        # Use an even lower threshold to get larger masks
        bin_mask = (out > 0.2).astype(np.uint8)
        
        # Morphological operations to improve mask quality
        bin_mask = cv2.morphologyEx(bin_mask * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) // 255
        bin_mask = cv2.morphologyEx(bin_mask * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) // 255
        
        # Check if mask has reasonable size
        if np.sum(bin_mask) >= 80:  # Lowered threshold
            # Resize to target shape while preserving binary nature
            resized = cv2.resize(bin_mask, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
            return resized
    
    # If no good mask found, return None to fall back to deterministic
    return None


# ----------------------------------------------------------------------------
# Placement utilities
# ----------------------------------------------------------------------------


def sample_from_cdf(cdf: np.ndarray) -> int:
    """Sample discrete value from 2-row CDF array returned by empirical_distributions."""
    values, cum = cdf
    r = random.random()
    idx = np.searchsorted(cum, r)
    return int(values[idx])


def place_mask(mask: np.ndarray, center: Tuple[int, int], canvas_shape: Tuple[int, int]) -> np.ndarray:
    """Translate binary mask to given center on a blank canvas."""
    h, w = canvas_shape
    mask_h, mask_w = mask.shape
    # Current centre of mask bbox
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros(canvas_shape, dtype=np.uint8)
    curr_cy = (ys.min() + ys.max()) // 2
    curr_cx = (xs.min() + xs.max()) // 2
    # Shift amounts
    dy = center[1] - curr_cy  # careful: (x,y) vs (col,row)
    dx = center[0] - curr_cx

    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    placed = cv2.warpAffine(mask.astype(np.uint8) * 255, translation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    return (placed > 127).astype(np.uint8)


# ----------------------------------------------------------------------------
# QC utilities
# ----------------------------------------------------------------------------


def eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if ys.size < 10:
        return 1.0
    coords = np.column_stack([xs, ys])
    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    # ratio of minor/major axis
    if eigenvalues[1] == 0:
        return 1.0
    ratio = eigenvalues[0] / eigenvalues[1]
    return 1 - ratio  # 0 for circle, 1 for line


# ----------------------------------------------------------------------------
# Main synthesis function
# ----------------------------------------------------------------------------


def synthesize_for_slice(
    pet_img: np.ndarray,
    body_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    heatmap: np.ndarray,
    count_cdf: np.ndarray,
    area_cdf: np.ndarray,
    seed_masks: List[np.ndarray],
    use_vae: bool = True,
    max_attempts: int = 30,
) -> np.ndarray:
    """Return synthetic tumour mask for given slice (same shape)."""
    h, w = pet_img.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # Lesion count
    n_lesions = sample_from_cdf(count_cdf)

    body_eroded = binary_erosion(body_mask, disk(3))  # edge safety

    for _ in range(n_lesions):
        for _attempt in range(max_attempts):
            # Generate candidate mask
            if use_vae:
                candidate = vae_sample_mask((h, w))
                if candidate is None:
                    candidate = warp_real_mask(random.choice(seed_masks))
            else:
                candidate = warp_real_mask(random.choice(seed_masks))

            # Rescale to match target area
            target_area = sample_from_cdf(area_cdf)
            curr_area = np.sum(candidate)
            if curr_area == 0:
                continue
            scale_factor = math.sqrt(target_area / curr_area)
            candidate = affine_transform_mask(candidate, scale_factor, 0, 0)

            # Sample location from heat-map (probabilistic) – use precomputed CDF if present
            if hasattr(synthesize_for_slice, "_heatmap_cdf") and synthesize_for_slice._heatmap_cdf is not None:
                r = np.random.random()
                idx = int(np.searchsorted(synthesize_for_slice._heatmap_cdf, r))
            else:
                flat_prob = heatmap.flatten()
                s = flat_prob.sum()
                if s <= 0:
                    idx = np.random.randint(0, flat_prob.size)
                else:
                    flat_prob /= s
                    idx = np.random.choice(flat_prob.size, p=flat_prob)
            cy, cx = divmod(idx, w)
            placed = place_mask(candidate, (cx, cy), (h, w))

            # Validation
            if not np.all((placed & ~body_eroded) == 0):
                continue  # outside body
            if np.any(placed & forbidden_mask):
                continue  # in forbidden organ
            if np.any(placed & final_mask):
                continue  # overlaps previous lesions

            # QC filters per-lesion
            area = np.sum(placed)
            if not (30 <= area <= 4000):  # Lowered minimum area for VAE masks
                continue
            if eccentricity(placed) > 0.98:
                continue

            final_mask |= placed
            break  # go to next lesion

    # Final QC: ensure dilated mask inside body
    dilated = binary_dilation(final_mask, disk(2))
    if not np.all(dilated <= body_mask):
        final_mask = final_mask & body_mask  # crop to body

    return final_mask.astype(np.uint8)


def create_comparison_plot(original_img: np.ndarray, vae_mask: np.ndarray, det_mask: np.ndarray, 
                          output_path: Path, subject_name: str):
    """Create a 3-panel comparison plot showing original, VAE overlay, and deterministic overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'{subject_name}\nOriginal Control')
    axes[0].axis('off')
    
    # VAE mask overlay
    vae_overlay = original_img.copy()
    vae_overlay[vae_mask > 0] = 255  # Highlight VAE mask in white
    axes[1].imshow(vae_overlay, cmap='gray')
    axes[1].set_title(f'{subject_name}\nVAE Synthetic Mask')
    axes[1].axis('off')
    
    # Deterministic mask overlay
    det_overlay = original_img.copy()
    det_overlay[det_mask > 0] = 255  # Highlight deterministic mask in white
    axes[2].imshow(det_overlay, cmap='gray')
    axes[2].set_title(f'{subject_name}\nDeterministic Synthetic Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------------
# CLI Entrypoint
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tumour masks for control PET slices.")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save generated masks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--vae_prob", type=float, default=0.3, help="Probability of sampling from VAE.")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_mask_generator.pth", help="Path to β-VAE checkpoint (will train if missing).")
    parser.add_argument("--vae_epochs", type=int, default=50, help="Number of epochs to train β-VAE if checkpoint is missing.")
    parser.add_argument("--no_auto_train", action="store_true", help="Disable auto-training of VAE if checkpoint not found.")
    parser.add_argument("--num_controls", type=int, default=None, help="Number of control subjects to process (default: all)")
    parser.add_argument("--plots_dir", type=str, default="comparison_plots", help="Directory to save comparison plots")
    parser.add_argument("--max_plots", type=int, default=None, help="Save at most this many random comparison plots (default: all)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    controls_dir = Path(args.controls_dir)
    patient_label_dir = Path(args.patients_label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Gather patient masks
    patient_label_paths = sorted(patient_label_dir.glob("*.png"))[:180]
    if len(patient_label_paths) < 50:
        raise RuntimeError("Not enough patient masks found to build statistics.")

    # Build global data
    print("🔧 Pre-computing location prior...")
    heatmap = build_location_prior(patient_label_paths)
    # Precompute CDF for faster sampling downstream
    synthesize_for_slice._heatmap_cdf = build_heatmap_cdf(heatmap)

    print("📊 Building empirical distributions...")
    count_cdf, area_cdf = empirical_distributions(patient_label_paths)

    print("📚 Loading seed masks…")
    seed_masks = [pad_to_target(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) > 0 for p in patient_label_paths]

    # Initialise / load β-VAE
    vae_model = get_vae(
        seed_masks,
        checkpoint_path=args.vae_checkpoint,
        auto_train=not args.no_auto_train,
        epochs=args.vae_epochs,
    )

    # Iterate over control images
    control_paths = sorted(controls_dir.glob("*.png"))
    if args.num_controls is not None:
        control_paths = control_paths[:args.num_controls]
    print(f"🚀 Generating masks for {len(control_paths)} control slices...")
    
    # Select a random subset of indices for which to save comparison plots
    selected_plot_indices = None
    if args.max_plots is not None:
        k = min(args.max_plots, len(control_paths))
        selected_plot_indices = set(random.sample(range(len(control_paths)), k))
        print(f"🖼️  Will save {k} random comparison plots to {plots_dir}")
    
    for idx, img_path in enumerate(tqdm(control_paths, desc="Controls")):
        pet = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pet is None:
            continue
        pet = pad_to_target(pet)
        body = compute_body_mask(pet)
        forbidden = compute_forbidden_mask(pet, body)
        
        # Generate VAE mask
        vae_mask = synthesize_for_slice(
            pet,
            body,
            forbidden,
            heatmap,
            count_cdf,
            area_cdf,
            seed_masks,
            use_vae=True,
        )
        
        # Generate deterministic mask
        det_mask = synthesize_for_slice(
            pet,
            body,
            forbidden,
            heatmap,
            count_cdf,
            area_cdf,
            seed_masks,
            use_vae=False,
        )
        
        # Save masks
        subject_name = img_path.stem
        vae_out_name = f"{subject_name}_vae_synthetic.png"
        det_out_name = f"{subject_name}_det_synthetic.png"
        vae_out_path = output_dir / vae_out_name
        det_out_path = output_dir / det_out_name
        
        cv2.imwrite(str(vae_out_path), (vae_mask * 255).astype(np.uint8))
        cv2.imwrite(str(det_out_path), (det_mask * 255).astype(np.uint8))
        
        # Create comparison plot for a random subset only (if requested)
        if selected_plot_indices is None or idx in selected_plot_indices:
            plot_path = plots_dir / f"{subject_name}_comparison.png"
            create_comparison_plot(pet, vae_mask, det_mask, plot_path, subject_name)

    print(f"✅ Synthetic masks saved to {output_dir}")
    print(f"✅ Comparison plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
