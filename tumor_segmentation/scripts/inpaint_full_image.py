#!/usr/bin/env python3
"""
Script for performing full-image inpainting using the trained tiled SPADE GAN.

This script can:
1. Load a trained tiled inpainting model
2. Perform high-resolution inpainting on patient images
3. Handle arbitrary image sizes through tiling
4. Save results with seamless tile blending
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from trainer.spade_gan_tiled_inpaint_module import SPADETiledInpaintGANModule


def load_model(checkpoint_path: Path, device: str = "cuda") -> SPADETiledInpaintGANModule:
    """Load trained tiled inpainting model from checkpoint."""
    model = SPADETiledInpaintGANModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.to(device)
    model.eval()
    return model


def get_control_images(data_root: Path) -> List[Path]:
    """Get list of control images for context."""
    control_dir = data_root / "controls" / "imgs"
    return sorted(control_dir.glob("*.png"))


def inpaint_patient_image(
    model: SPADETiledInpaintGANModule,
    patient_img_path: Path,
    mask_path: Path,
    control_imgs: List[Path],
    output_path: Path,
    tile_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
):
    """Perform tiled inpainting on a patient image."""
    print(f"Inpainting {patient_img_path.name}...")
    print(f"  Mask: {mask_path.name}")
    print(f"  Tile size: {tile_size}x{tile_size}")
    print(f"  Overlap: {overlap}px")
    
    # Perform inpainting
    result_img = model.inpaint_full_image(
        patient_img_path=patient_img_path,
        mask_path=mask_path,
        control_imgs=control_imgs,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
    )
    
    # Convert to PIL and save
    result_pil = Image.fromarray((result_img * 255).astype(np.uint8), mode="L")
    result_pil.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform full-image inpainting with tiled SPADE GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Data root containing controls/")
    parser.add_argument("--patient_img", type=str, required=True, help="Path to patient image")
    parser.add_argument("--mask", type=str, required=True, help="Path to tumor mask")
    parser.add_argument("--output", type=str, required=True, help="Output path for inpainted image")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for processing")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between tiles")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Convert to paths
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    patient_img_path = Path(args.patient_img)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    
    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not patient_img_path.exists():
        raise FileNotFoundError(f"Patient image not found: {patient_img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, args.device)
    
    # Get control images
    print("Loading control images...")
    control_imgs = get_control_images(data_root)
    if not control_imgs:
        raise RuntimeError(f"No control images found in {data_root}/controls/imgs/")
    print(f"Found {len(control_imgs)} control images")
    
    # Perform inpainting
    inpaint_patient_image(
        model=model,
        patient_img_path=patient_img_path,
        mask_path=mask_path,
        control_imgs=control_imgs,
        output_path=output_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()