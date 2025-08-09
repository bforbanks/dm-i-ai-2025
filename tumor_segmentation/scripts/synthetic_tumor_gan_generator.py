#!/usr/bin/env python3
"""
Synthetic Tumor GAN Generator
============================

Creates synthetic tumors on control images using a trained SPADE GAN model.
Processes all masks from rough_labels_bank and rough_labels_det, applies them to control images,
and saves the results with proper sequential numbering.

Usage:
    python scripts/synthetic_tumor_gan_generator.py --model_dir logs/spade_gan_full_inpaint/version_X --data_root data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

sys.path.append(str(Path(__file__).parent.parent / "trainer"))
from spade_gan_full_inpaint_module import SPADEFullInpaintGANModule


def load_model_from_checkpoint(model_dir: Path, data_root: Path, device: str = "cuda") -> SPADEFullInpaintGANModule:
    """Load the trained SPADE GAN model from a checkpoint directory."""
    # Find the checkpoint file
    checkpoint_files = list(model_dir.glob("checkpoints/*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}/checkpoints/")
    
    # Use the latest checkpoint
    checkpoint_path = sorted(checkpoint_files)[-1]
    print(f"Loading model from: {checkpoint_path}")
    
    # Determine the actual device to use
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load the model with proper device mapping
    if device == "cpu":
        map_location = torch.device('cpu')
    else:
        map_location = device
    
    model = SPADEFullInpaintGANModule.load_from_checkpoint(
        checkpoint_path,
        map_location=map_location,
        strict=False,
        data_root=str(data_root)
    )
    model.eval()
    model.to(device)
    
    # Initialize the dataset for the model to access padding dimensions
    model.setup('fit')
    
    return model


def extract_control_id(filename: str) -> Optional[int]:
    """Extract numeric control ID from filename like 'control_001.png'."""
    import re
    match = re.search(r'control_(\d+)', filename)
    return int(match.group(1)) if match else None


def _predict_tumour_only(
    model: SPADEFullInpaintGANModule,
    control_img_path: Path,
    mask_img_path: Path,
    control_imgs: List[Path],
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pred_tumour, mask) cropped to original size without mixing into background."""
    # Local imports to avoid polluting global space
    from PIL import Image

    control_img = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
    mask_img = np.array(Image.open(mask_img_path).convert("L"), dtype=np.float32) / 255.0

    original_h, original_w = control_img.shape

    # Ensure mask matches control
    if mask_img.shape != control_img.shape:
        mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(control_img.shape[::-1], Image.NEAREST)
        mask_img = np.array(mask_pil, dtype=np.float32) / 255.0

    # Pad to model training size if required
    if hasattr(model, "train_dataset") and hasattr(model.train_dataset, "target_w") and model.train_dataset.target_w:
        target_w, target_h = model.train_dataset.target_w, model.train_dataset.target_h
        if control_img.shape != (target_h, target_w):
            h, w = control_img.shape
            pad_left = (target_w - w) // 2
            pad_top = 0
            canvas = np.ones((target_h, target_w), dtype=np.float32)
            canvas[pad_top:pad_top + h, pad_left:pad_left + w] = control_img
            control_img = canvas

            canvas = np.zeros((target_h, target_w), dtype=np.float32)
            canvas[pad_top:pad_top + h, pad_left:pad_left + w] = mask_img
            mask_img = canvas

    # Choose a control context image (use the same control for conditioning)
    control_img_for_context = control_img

    # Tensors
    control_tensor = torch.from_numpy(control_img_for_context).unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to(device)
    gen_input = torch.cat([control_tensor, mask_tensor], dim=1)

    # Forward generator
    model.generator.eval()
    with torch.no_grad():
        pred_tumour = model.generator(gen_input)
        pred_tumour_np = pred_tumour.cpu().numpy()[0, 0]
    model.generator.train()

    # Crop back to original
    if pred_tumour_np.shape != (original_h, original_w):
        target_w, target_h = model.train_dataset.target_w, model.train_dataset.target_h
        h, w = original_h, original_w
        pad_left = (target_w - w) // 2
        pad_top = 0
        pred_tumour_np = pred_tumour_np[pad_top:pad_top + h, pad_left:pad_left + w]

    # Ensure mask also cropped
    mask_out = mask_img[:original_h, :original_w]

    return pred_tumour_np, mask_out


def generate_tumor_on_control(
    model: SPADEFullInpaintGANModule,
    control_img_path: Path,
    mask_img_path: Path,
    control_imgs: List[Path],
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic tumor on control image using the GAN model.
    Returns (generated_image, mask) both cropped to original size.
    """
    # Use the model's inpainting function directly
    generated_img = model.inpaint_full_image(
        patient_img_path=control_img_path,
        mask_path=mask_img_path,
        control_imgs=control_imgs,
        device=device
    )
    
    # Load the mask for returning
    mask_img = np.array(Image.open(mask_img_path).convert("L"), dtype=np.float32) / 255.0
    
    # Ensure mask matches generated image size
    if mask_img.shape != generated_img.shape:
        mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(generated_img.shape[::-1], Image.NEAREST)
        mask_img = np.array(mask_pil, dtype=np.float32) / 255.0
    
    return generated_img, mask_img


def create_directories(base_dir: Path) -> Tuple[Path, Path]:
    """Create output directories for labels and overlays."""
    labels_dir = base_dir / "controls" / "rough_labels_GAN"
    overlays_dir = base_dir / "controls" / "rough_overlay_GAN"
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    
    return labels_dir, overlays_dir


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_mask_directory(
    mask_dir: Path,
    controls_dir: Path,
    model: SPADEFullInpaintGANModule,
    control_imgs: List[Path],
    labels_out_dir: Path,
    overlays_out_dir: Path,
    start_idx: int,
    device: str = "cuda",
    tumour_preview_dir: Optional[Path] = None,
    preview_every: int = 0,
) -> int:
    """
    Process all masks in a directory and generate synthetic tumors.
    Returns the next available index for numbering.
    """
    # Get all mask files sorted by control ID
    mask_files = sorted(mask_dir.glob("control_*.png"), key=lambda x: extract_control_id(x.name) or 0)
    
    current_idx = start_idx
    
    for j, mask_path in enumerate(tqdm(mask_files, desc=f"Processing {mask_dir.name}")):
        try:
            # Extract control ID and find corresponding control image
            control_id = extract_control_id(mask_path.name)
            if control_id is None:
                print(f"Warning: Could not extract control ID from {mask_path.name}")
                continue
            
            control_img_path = controls_dir / f"control_{control_id:03d}.png"
            if not control_img_path.exists():
                print(f"Warning: Control image not found: {control_img_path}")
                continue
            
            # Generate synthetic tumor
            # IMPORTANT: condition the generator on the SAME control image as the background
            generated_img, mask_img = generate_tumor_on_control(
                model, control_img_path, mask_path, [control_img_path], device
            )
            
            # Save with new sequential numbering
            new_name = f"control_{current_idx:03d}.png"
            
            # Save mask (binary)
            mask_binary = (mask_img * 255).astype(np.uint8)
            cv2.imwrite(str(labels_out_dir / new_name), mask_binary)
            
            # Save generated image with tumor
            generated_uint8 = (np.clip(generated_img, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(overlays_out_dir / new_name), generated_uint8)

            # Optionally export tumor-only preview (generator output before mixing)
            if tumour_preview_dir is not None and (preview_every <= 0 or (j % max(1, preview_every) == 0)):
                ensure_dir(tumour_preview_dir)
                pred_tumour_np, _ = _predict_tumour_only(
                    model, control_img_path, mask_path, [control_img_path], device
                )
                tumour_uint8 = (np.clip(pred_tumour_np, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(str(tumour_preview_dir / new_name), tumour_uint8)
            
            current_idx += 1
            
        except Exception as e:
            print(f"Error processing {mask_path.name}: {str(e)}")
            continue
    
    return current_idx


def create_sample_comparison_plots(
    overlays_dir: Path,
    labels_dir: Path,
    controls_dir: Path,
    output_dir: Path,
    num_samples: int = 5
) -> None:
    """Create comparison plots for a few sample results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all generated files
    overlay_files = sorted(overlays_dir.glob("control_*.png"))
    
    # Sample a few for visualization
    sample_indices = np.linspace(0, len(overlay_files)-1, min(num_samples, len(overlay_files)), dtype=int)
    
    for i, idx in enumerate(sample_indices):
        overlay_path = overlay_files[idx]
        control_id = extract_control_id(overlay_path.name)
        
        if control_id is None:
            continue
        
        # Load images
        try:
            overlay_img = cv2.imread(str(overlay_path), cv2.IMREAD_GRAYSCALE)
            mask_path = labels_dir / overlay_path.name
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Find original control (this might not exist with the same numbering)
            # We'll use the overlay as base and create a version without tumor
            control_img = overlay_img.copy()
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(control_img, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original Control")
            axes[0].axis("off")
            
            axes[1].imshow(overlay_img, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Control + Generated Tumor")
            axes[1].axis("off")
            
            # Show mask overlay
            axes[2].imshow(control_img, cmap="gray", vmin=0, vmax=255)
            if mask_img is not None:
                mask_normalized = mask_img.astype(np.float32) / 255.0
                axes[2].imshow(mask_normalized, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
            axes[2].set_title("Tumor Mask Location")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(output_dir / f"gan_synthetic_comparison_{i:03d}.png", dpi=150, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            print(f"Error creating comparison plot for {overlay_path.name}: {str(e)}")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tumors using trained SPADE GAN model")
    parser.add_argument("--model_dir", required=True, type=str, help="Directory containing the trained model (e.g., logs/spade_gan_full_inpaint/version_X)")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory containing control images and masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--export_tumour_previews", action="store_true", help="Also save generator outputs before mixing (for inspection)")
    parser.add_argument("--num_preview_samples", type=int, default=8, help="How many tumour-only samples to export (approximate)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sample comparison plots to generate")
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    
    # Input directories
    controls_dir = data_root / "controls" / "imgs"
    bank_masks_dir = data_root / "controls" / "rough_labels_bank"
    det_masks_dir = data_root / "controls" / "rough_labels_det"
    
    # Check that all required directories exist
    for dir_path in [controls_dir, bank_masks_dir, det_masks_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Create output directories
    labels_out_dir, overlays_out_dir = create_directories(data_root)
    
    # Load the model
    print("Loading trained SPADE GAN model...")
    model = load_model_from_checkpoint(model_dir, data_root, args.device)
    
    # Get list of control images for context during generation
    control_imgs = list(controls_dir.glob("*.png"))
    if not control_imgs:
        raise FileNotFoundError(f"No control images found in {controls_dir}")
    
    print(f"Found {len(control_imgs)} control images for context")
    
    # Optional tumour-only preview directory
    tumour_preview_dir = None
    preview_every = 0
    if args.export_tumour_previews:
        tumour_preview_dir = data_root / "controls" / "rough_tumors_GAN"
        ensure_dir(tumour_preview_dir)
        # Approximate spacing to export roughly num_preview_samples from each set
        # Avoid division by zero
        preview_every = max(1, len(list(bank_masks_dir.glob('control_*.png'))) // max(1, args.num_preview_samples))

    # Process bank masks first (starting from index 1)
    print(f"Processing rough_labels_bank masks...")
    next_idx = process_mask_directory(
        bank_masks_dir, controls_dir, model, control_imgs,
        labels_out_dir, overlays_out_dir, start_idx=1, device=args.device,
        tumour_preview_dir=tumour_preview_dir, preview_every=preview_every
    )
    
    print(f"Processed bank masks, next index: {next_idx}")
    
    # Process det masks (continuing numbering from bank masks)
    print(f"Processing rough_labels_det masks...")
    final_idx = process_mask_directory(
        det_masks_dir, controls_dir, model, control_imgs,
        labels_out_dir, overlays_out_dir, start_idx=next_idx, device=args.device,
        tumour_preview_dir=tumour_preview_dir, preview_every=preview_every
    )
    
    print(f"Processed det masks, final index: {final_idx}")
    
    # Create sample comparison plots
    print("Creating sample comparison plots...")
    comparison_dir = data_root / "controls" / "rough_comparisons_GAN"
    create_sample_comparison_plots(
        overlays_out_dir, labels_out_dir, controls_dir, comparison_dir, args.num_samples
    )
    
    print(f"✅ Generated {final_idx - 1} synthetic tumor images")
    print(f"✅ Masks saved to: {labels_out_dir}")
    print(f"✅ Overlay images saved to: {overlays_out_dir}")
    print(f"✅ Sample comparisons saved to: {comparison_dir}")


if __name__ == "__main__":
    main()