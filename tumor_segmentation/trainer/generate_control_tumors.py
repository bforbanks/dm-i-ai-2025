#!/usr/bin/env python3
"""Generate seamlessly integrated tumors for all control images using trained refine model."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
import cv2

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.refine_tumor_dataset import RefineTumorSliceDataset
from spade_gan_refine_inpaint_module import SPADERefineInpaintGANModule
from pix2pix_refine_inpaint_module import Pix2PixRefineInpaintModule


def load_model_from_checkpoint(checkpoint_path: str, model_type: str = "spade") -> torch.nn.Module:
    """Load trained model from Lightning checkpoint."""
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint.get('hyper_parameters', {})
    
    if model_type.lower() == "spade":
        # Create model with hyperparameters from checkpoint
        model = SPADERefineInpaintGANModule(**hparams)
    elif model_type.lower() == "pix2pix":
        # Create model with hyperparameters from checkpoint
        model = Pix2PixRefineInpaintModule(**hparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'spade' or 'pix2pix'")
    
    # Trigger setup to initialize generator and discriminator
    model.setup(stage=None)
    
    # Load state dict with strict=False to ignore mismatched keys
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model


def get_original_dimensions(img_path: Path) -> Tuple[int, int]:
    """Get original image dimensions before padding."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def crop_to_original_size(padded_tensor: torch.Tensor, original_w: int, original_h: int, 
                         padded_w: int, padded_h: int) -> torch.Tensor:
    """Crop padded tensor back to original dimensions.
    
    Reverses the padding logic:
    - Horizontal: was centered, so remove equal amounts from left/right
    - Vertical: was bottom-only, so remove from bottom only
    """
    # Calculate padding that was applied
    pad_left = (padded_w - original_w) // 2
    pad_right = padded_w - original_w - pad_left
    # pad_top was 0, so all vertical padding was on bottom
    pad_bottom = padded_h - original_h
    
    # Crop: remove padding
    # tensor is (C, H, W), so slice [start_h:end_h, start_w:end_w]
    crop_h_start = 0  # no top padding was added
    crop_h_end = padded_h - pad_bottom if pad_bottom > 0 else padded_h
    crop_w_start = pad_left
    crop_w_end = padded_w - pad_right if pad_right > 0 else padded_w
    
    cropped = padded_tensor[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    return cropped


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (C, H, W) in [0,1] to numpy array (H, W) in [0,255]."""
    # Remove channel dimension if single channel
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and scale to [0, 255]
    img_np = tensor.detach().cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def save_image(img_array: np.ndarray, save_path: Path):
    """Save numpy image array to file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_array)


def generate_all_control_tumors(
    model: torch.nn.Module,
    data_root: Path,
    output_dir: Path,
    control_rough_labels_dir: str = "controls/rough_labels",
    control_rough_tumors_dir: str = "controls/rough_tumors", 
    device: torch.device = None
):
    """Generate tumors for all control images and save cropped results."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Create dataset to get the padding logic and control mappings
    dataset = RefineTumorSliceDataset(
        data_root, 
        auto_pad=True,
        control_rough_labels_dir=control_rough_labels_dir,
        control_rough_tumors_dir=control_rough_tumors_dir
    )
    
    print(f"Found {len(dataset.control_indices)} control images to process")
    print(f"Padded canvas size: {dataset.target_w}×{dataset.target_h}")
    
    output_dir = output_dir / "controls" / "imgs_tumor"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with torch.no_grad():
        for ctrl_idx in dataset.control_indices:
            try:
                # Get file paths
                control_img_path = dataset.control_img_map[ctrl_idx]
                control_mask_path = dataset.control_mask_map[ctrl_idx]
                control_rough_path = dataset.control_rough_map[ctrl_idx]
                
                # Get original dimensions before padding
                original_w, original_h = get_original_dimensions(control_img_path)
                
                # Load and pad images using dataset logic
                control_img = dataset._load_image(control_img_path, pad_value=255)
                control_mask = dataset._load_image(control_mask_path, pad_value=0)
                control_rough = dataset._load_image(control_rough_path, pad_value=255)
                
                # Create generator input: [rough_control, control_mask]
                gen_input = torch.cat([control_rough, control_mask], dim=0).unsqueeze(0).to(device)
                control_img = control_img.unsqueeze(0).to(device)
                control_mask = control_mask.unsqueeze(0).to(device)
                
                # Generate refined tumor
                pred_tumor = model.generator(gen_input)
                
                # Composite: control background + refined tumor
                control_background = control_img * (1 - control_mask)
                fake_full = control_background + pred_tumor * control_mask
                
                # Crop back to original size
                fake_full_cropped = crop_to_original_size(
                    fake_full.squeeze(0), original_w, original_h, 
                    dataset.target_w, dataset.target_h
                )
                
                # Convert to image and save
                final_img = tensor_to_image(fake_full_cropped)
                
                # Use same filename pattern as original control
                output_filename = control_img_path.name  # e.g., "control_123.png"
                output_path = output_dir / output_filename
                
                save_image(final_img, output_path)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(dataset.control_indices)} images...")
                
            except Exception as e:
                print(f"Failed to process control {ctrl_idx}: {e}")
                continue
    
    print(f"Successfully generated tumors for {processed_count}/{len(dataset.control_indices)} control images")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate control images with integrated tumors")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["spade", "pix2pix"], default="spade", 
                       help="Type of model (spade or pix2pix)")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated images")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--control_rough_labels_dir", type=str, default="controls/rough_labels", 
                       help="Directory name within data_root for control rough labels")
    parser.add_argument("--control_rough_tumors_dir", type=str, default="controls/rough_tumors", 
                       help="Directory name within data_root for control rough tumors")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, args.model_type)
    
    # Generate tumors
    print("Starting tumor generation...")
    generate_all_control_tumors(
        model=model,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        control_rough_labels_dir=args.control_rough_labels_dir,
        control_rough_tumors_dir=args.control_rough_tumors_dir,
        device=device
    )
    
    print("Generation complete!")


if __name__ == "__main__":
    main()