#!/usr/bin/env python3
"""
Plot BaselineModel predictions using utils.py plotting functions
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import BaselineModel
from data_module import TumorSegmentationDataModule
from utils import plot_prediction, dice_score
import cv2


def plot_baseline_predictions():
    """Plot some predictions from the BaselineModel"""

    print("Loading BaselineModel and data...")

    # Create model
    model = BaselineModel(threshold=50.0)
    model.eval()

    # Create data module
    datamodule = TumorSegmentationDataModule(
        data_dir="data", batch_size=4, num_workers=0, image_size=256, val_split=0.2
    )

    # Setup data
    datamodule.setup()

    # Get validation dataloader
    val_loader = datamodule.val_dataloader()

    print(f"Validation set has {len(val_loader)} batches")

    # Find samples with and without tumors
    patient_sample = None
    control_sample = None

    print("Searching for patient (with tumor) and control (without tumor) samples...")

    for batch_idx, batch in enumerate(val_loader):
        images, masks = batch

        # Check each sample in the batch
        for i in range(images.shape[0]):
            mask = masks[i]  # [1, H, W]
            has_tumor = torch.sum(mask) > 0  # Check if mask has any positive values

            if has_tumor and patient_sample is None:
                patient_sample = {
                    "image": images[i : i + 1],  # Keep batch dimension
                    "mask": masks[i : i + 1],
                    "type": "patient",
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                }
                print(f"✓ Found patient sample (batch {batch_idx}, sample {i})")
            elif not has_tumor and control_sample is None:
                control_sample = {
                    "image": images[i : i + 1],  # Keep batch dimension
                    "mask": masks[i : i + 1],
                    "type": "control",
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                }
                print(f"✓ Found control sample (batch {batch_idx}, sample {i})")

            # Break if we found both
            if patient_sample is not None and control_sample is not None:
                break

        if patient_sample is not None and control_sample is not None:
            break

    # Check if we found the required samples
    if patient_sample is None:
        print("❌ Could not find a patient sample with tumors!")
        return
    if control_sample is None:
        print("❌ Could not find a control sample without tumors!")
        return

    # Process both samples
    samples = [patient_sample, control_sample]

    for idx, sample_data in enumerate(samples):
        print(f"\n--- Sample {idx + 1}: {sample_data['type']} ---")

        # Get the sample data
        images = sample_data["image"]  # [1, 3, H, W]
        masks = sample_data["mask"]  # [1, 1, H, W]

        print(f"Sample shape: images={images.shape}, masks={masks.shape}")
        print(
            f"Image dtype: {images.dtype}, range: {images.min():.1f}-{images.max():.1f}"
        )
        print(f"Mask dtype: {masks.dtype}, range: {masks.min():.1f}-{masks.max():.1f}")
        print(f"Mask sum (tumor pixels): {torch.sum(masks):.1f}")

        # Make predictions
        with torch.no_grad():
            predictions = model(images)

        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: {predictions.min():.6f}-{predictions.max():.6f}")

        # Convert to numpy and process for plotting
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Get single sample (remove batch dimension)
        img = images_np[0]  # [3, H, W]
        mask = masks_np[0]  # [1, H, W]
        pred = predictions_np[0]  # [1, H, W]

        # Convert to format expected by plot_prediction: [H, W, 3]
        # Transpose from [3, H, W] to [H, W, 3]
        img_hwc = np.transpose(img, (1, 2, 0)).astype(np.uint8)

        # Convert mask from [1, H, W] to [H, W, 3]
        # The mask values are in [0, 1] range, convert to binary first
        mask_2d = mask[0]  # [H, W]
        print(
            f"  Mask values: min={mask_2d.min():.6f}, max={mask_2d.max():.6f}, unique={np.unique(mask_2d)}"
        )

        # Convert to binary mask (0 or 255)
        mask_binary = (mask_2d > 0).astype(np.uint8) * 255  # [H, W]
        mask_hwc = np.stack(
            [mask_binary, mask_binary, mask_binary], axis=2
        )  # [H, W, 3]

        # Convert prediction to binary [0,255] format like API expects
        pred_binary = (pred[0] > 0.5).astype(np.uint8) * 255  # [H, W]
        pred_hwc = np.stack(
            [pred_binary, pred_binary, pred_binary], axis=2
        )  # [H, W, 3]

        print(f"Sample {idx + 1} ({sample_data['type']}) shapes:")
        print(f"  Image: {img_hwc.shape}, range: {img_hwc.min()}-{img_hwc.max()}")
        print(f"  Mask: {mask_hwc.shape}, range: {mask_hwc.min()}-{mask_hwc.max()}")
        print(
            f"  Prediction: {pred_hwc.shape}, range: {pred_hwc.min()}-{pred_hwc.max()}"
        )

        # Calculate dice score using utils function
        dice_utils = dice_score(mask_hwc, pred_hwc)
        print(f"  Utils Dice Score: {dice_utils:.4f}")

        # Calculate dice score using BaseModel's custom implementation
        # Convert back to torch tensors in the format expected by the model
        pred_torch = (
            torch.tensor(pred_binary).unsqueeze(0).unsqueeze(0).float()
        )  # [1, 1, H, W]
        mask_torch = (
            torch.tensor(mask[0]).unsqueeze(0).unsqueeze(0).float()
        )  # [1, 1, H, W]

        dice_basemodel = model._calculate_dice_score(pred_torch, mask_torch)
        print(f"  BaseModel Dice Score: {dice_basemodel:.4f}")

        # Calculate difference
        dice_diff = abs(dice_utils - dice_basemodel.item())
        print(f"  Difference: {dice_diff:.6f}")

        # Plot using utils function
        try:
            plot_prediction(img_hwc, mask_hwc, pred_hwc)
        except Exception as e:
            print(f"  Error plotting: {e}")
            continue


if __name__ == "__main__":
    plot_baseline_predictions()
