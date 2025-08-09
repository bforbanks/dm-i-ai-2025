#!/usr/bin/env python3
"""
Test Improved VAE
================

Test the improved VAE implementation and compare with the original.
"""

import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch

# Import functions from the main script
import sys
sys.path.append(str(Path(__file__).parent))
from synthetic_mask_generator import (
    pad_to_target, compute_body_mask, compute_forbidden_mask,
    build_location_prior, empirical_distributions, get_vae,
    TARGET_WIDTH, TARGET_HEIGHT
)
from improved_vae_generator import (
    train_improved_vae, sample_improved_vae, ImprovedBetaVAE
)


def create_vae_comparison_plot(original_img: np.ndarray, original_vae_mask: np.ndarray, 
                              improved_vae_mask: np.ndarray, output_path: Path, subject_name: str):
    """Create a comparison plot showing original VAE vs improved VAE."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title(f'{subject_name}\nOriginal Control')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original_img, cmap='gray')
    axes[1, 0].set_title(f'{subject_name}\nOriginal Control')
    axes[1, 0].axis('off')
    
    # Original VAE
    original_overlay = original_img.copy()
    original_overlay[original_vae_mask > 0] = 255
    axes[0, 1].imshow(original_overlay, cmap='gray')
    axes[0, 1].set_title(f'Original VAE\nOverlay')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(original_vae_mask, cmap='gray')
    axes[1, 1].set_title(f'Original VAE\nMask Only\nArea: {np.sum(original_vae_mask)}')
    axes[1, 1].axis('off')
    
    # Improved VAE
    improved_overlay = original_img.copy()
    improved_overlay[improved_vae_mask > 0] = 255
    axes[0, 2].imshow(improved_overlay, cmap='gray')
    axes[0, 2].set_title(f'Improved VAE\nOverlay')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(improved_vae_mask, cmap='gray')
    axes[1, 2].set_title(f'Improved VAE\nMask Only\nArea: {np.sum(improved_vae_mask)}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test improved VAE vs original VAE.")
    parser.add_argument("--controls_dir", required=True, type=str, help="Directory with control PET PNGs.")
    parser.add_argument("--patients_label_dir", required=True, type=str, help="Directory with real patient label PNGs.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save comparison plots.")
    parser.add_argument("--num_controls", type=int, default=3, help="Number of control subjects to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--vae_epochs", type=int, default=100, help="Number of epochs to train improved VAE.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device to use.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

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

    # Initialize original VAE (it auto-selects cuda if available inside get_vae)
    print("🔄 Loading original VAE...")
    original_vae = get_vae(
        seed_masks,
        checkpoint_path="vae_mask_generator.pth",
        auto_train=True,
        epochs=50,
    )

    # Train improved VAE
    print(f"🚀 Training improved VAE on {device}...")
    improved_vae = train_improved_vae(
        seed_masks,
        epochs=args.vae_epochs,
        checkpoint="improved_vae_mask_generator.pth",
        device=device,
    )

    # Test on control images
    control_paths = sorted(controls_dir.glob("*.png"))
    if args.num_controls is not None:
        control_paths = control_paths[:args.num_controls]
    
    print(f"🧪 Testing both VAEs on {len(control_paths)} control subjects...")
    
    for img_path in tqdm(control_paths, desc="Testing VAEs"):
        pet = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if pet is None:
            continue
        
        pet = pad_to_target(pet)
        body = compute_body_mask(pet)
        forbidden = compute_forbidden_mask(pet, body)
        
        # Generate masks with both VAEs
        original_mask = np.zeros(pet.shape, dtype=np.uint8)
        improved_mask = np.zeros(pet.shape, dtype=np.uint8)
        
        # Original VAE
        if original_vae is not None:
            # Infer model device to sample on the correct device
            try:
                original_device = next(original_vae.parameters()).device.type
            except Exception:
                original_device = device
            
            mask = sample_improved_vae(original_vae, pet.shape, device=original_device, num_attempts=10)
            if mask is not None:
                original_mask = mask
        
        # Improved VAE
        improved_mask = sample_improved_vae(improved_vae, pet.shape, device=device, num_attempts=10)
        if improved_mask is None:
            improved_mask = np.zeros(pet.shape, dtype=np.uint8)
        
        # Create comparison plot
        subject_name = img_path.stem
        plot_path = output_dir / f"{subject_name}_vae_comparison.png"
        create_vae_comparison_plot(pet, original_mask, improved_mask, plot_path, subject_name)
        
        # Save individual masks
        masks_dir = output_dir / subject_name
        masks_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(masks_dir / "original_vae_mask.png"), (original_mask * 255).astype(np.uint8))
        cv2.imwrite(str(masks_dir / "improved_vae_mask.png"), (improved_mask * 255).astype(np.uint8))

    print(f"✅ VAE comparison plots saved to {output_dir}")
    print(f"📁 Individual masks saved in subdirectories for each subject")


if __name__ == "__main__":
    main() 