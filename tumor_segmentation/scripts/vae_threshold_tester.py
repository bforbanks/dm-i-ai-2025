#!/usr/bin/env python3
"""
VAE Threshold Tester
===================

Tests different threshold values for VAE mask generation and saves comparison images
to help determine the optimal threshold for mask generation.

Usage:
    python vae_threshold_tester.py \
        --controls_dir data/controls/imgs \
        --patients_label_dir data/patients/labels \
        --output_dir vae_threshold_tests \
        --num_controls 5 \
        --thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
"""

import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Import functions from the main script
import sys
sys.path.append(str(Path(__file__).parent))
from synthetic_mask_generator import (
    pad_to_target, compute_body_mask, compute_forbidden_mask,
    build_location_prior, empirical_distributions, vae_sample_mask,
    get_vae, TARGET_WIDTH, TARGET_HEIGHT
)


def create_threshold_comparison_plot(original_img: np.ndarray, masks_by_threshold: dict, 
                                   output_path: Path, subject_name: str):
    """Create a comparison plot showing original and masks for different thresholds."""
    n_thresholds = len(masks_by_threshold)
    fig, axes = plt.subplots(2, n_thresholds + 1, figsize=(4 * (n_thresholds + 1), 8))
    
    # Original image (spans both rows)
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title(f'{subject_name}\nOriginal Control')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original_img, cmap='gray')
    axes[1, 0].set_title(f'{subject_name}\nOriginal Control')
    axes[1, 0].axis('off')
    
    # Masks for each threshold
    for i, (threshold, mask) in enumerate(masks_by_threshold.items()):
        # Top row: mask overlay
        overlay = original_img.copy()
        overlay[mask > 0] = 255
        axes[0, i + 1].imshow(overlay, cmap='gray')
        axes[0, i + 1].set_title(f'Threshold {threshold}\nOverlay')
        axes[0, i + 1].axis('off')
        
        # Bottom row: mask only
        axes[1, i + 1].imshow(mask, cmap='gray')
        axes[1, i + 1].set_title(f'Threshold {threshold}\nMask Only\nArea: {np.sum(mask)}')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_vae_thresholds(original_img: np.ndarray, body_mask: np.ndarray, 
                       forbidden_mask: np.ndarray, thresholds: List[float]) -> dict:
    """Test VAE mask generation with different thresholds."""
    masks_by_threshold = {}
    
    for threshold in thresholds:
        # Temporarily modify the VAE sampling function to use the test threshold
        def vae_sample_with_threshold(mask_shape: Tuple[int, int], test_threshold: float) -> np.ndarray | None:
            from synthetic_mask_generator import _VAE_MODEL, _VAE_DEVICE
            import torch
            
            if _VAE_MODEL is None:
                return None
            
            # Try multiple samples to get a good one
            for _ in range(10):
                with torch.no_grad():
                    z = torch.randn(1, 16, device=_VAE_DEVICE)
                    out = _VAE_MODEL.decode(z).cpu().numpy()[0, 0]
                
                # Use the test threshold
                bin_mask = (out > test_threshold).astype(np.uint8)
                
                # Morphological operations to improve mask quality
                bin_mask = cv2.morphologyEx(bin_mask * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) // 255
                bin_mask = cv2.morphologyEx(bin_mask * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) // 255
                
                # Check if mask has reasonable size
                if np.sum(bin_mask) >= 50:  # Lower threshold for testing
                    # Resize to target shape while preserving binary nature
                    resized = cv2.resize(bin_mask, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
                    return resized
            
            # If no good mask found, return empty mask
            return np.zeros(mask_shape, dtype=np.uint8)
        
        # Generate mask with this threshold
        mask = vae_sample_with_threshold(original_img.shape, threshold)
        if mask is None:
            mask = np.zeros(original_img.shape, dtype=np.uint8)
        
        # Ensure mask is within body boundaries
        mask = mask & body_mask
        masks_by_threshold[threshold] = mask
    
    return masks_by_threshold


def main():
    parser = argparse.ArgumentParser(description="Test different VAE thresholds for mask generation.")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save comparison plots.")
    parser.add_argument("--num_controls", type=int, default=5, help="Number of control subjects to test.")
    parser.add_argument("--thresholds", nargs='+', type=float, 
                       default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                       help="Threshold values to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_mask_generator.pth", 
                       help="Path to β-VAE checkpoint.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    controls_dir = Path(args.controls_dir)
    patient_label_dir = Path(args.patients_label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather patient masks
    patient_label_paths = sorted(patient_label_dir.glob("*.png"))[:180]
    if len(patient_label_paths) < 50:
        raise RuntimeError("Not enough patient masks found to build statistics.")

    # Build global data
    print("🔧 Pre-computing location prior...")
    heatmap = build_location_prior(patient_label_paths)

    print("📊 Building empirical distributions...")
    count_cdf, area_cdf = empirical_distributions(patient_label_paths)

    print("📚 Loading seed masks...")
    seed_masks = [pad_to_target(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) > 0 for p in patient_label_paths]

    # Initialize VAE
    vae_model = get_vae(
        seed_masks,
        checkpoint_path=args.vae_checkpoint,
        auto_train=True,
        epochs=50,
    )

    if vae_model is None:
        raise RuntimeError("Failed to initialize VAE model.")

    # Test thresholds on control images
    control_paths = sorted(controls_dir.glob("*.png"))
    if args.num_controls is not None:
        control_paths = control_paths[:args.num_controls]
    
    print(f"🧪 Testing {len(args.thresholds)} thresholds on {len(control_paths)} control subjects...")
    
    for img_path in tqdm(control_paths, desc="Testing thresholds"):
        pet = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pet is None:
            continue
        
        pet = pad_to_target(pet)
        body = compute_body_mask(pet)
        forbidden = compute_forbidden_mask(pet, body)
        
        # Test all thresholds
        masks_by_threshold = test_vae_thresholds(pet, body, forbidden, args.thresholds)
        
        # Create comparison plot
        subject_name = img_path.stem
        plot_path = output_dir / f"{subject_name}_threshold_comparison.png"
        create_threshold_comparison_plot(pet, masks_by_threshold, plot_path, subject_name)
        
        # Save individual masks for each threshold
        masks_dir = output_dir / subject_name
        masks_dir.mkdir(exist_ok=True)
        
        for threshold, mask in masks_by_threshold.items():
            mask_path = masks_dir / f"threshold_{threshold:.2f}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    print(f"✅ Threshold comparison plots saved to {output_dir}")
    print(f"📊 Tested thresholds: {args.thresholds}")
    print(f"📁 Individual masks saved in subdirectories for each subject")


if __name__ == "__main__":
    main() 