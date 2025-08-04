#!/usr/bin/env python3
"""
Plot BaselineModel predictions using utils.py plotting functions
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SimpleUNet
from models.OrganDetector.model import OrganDetector
from data.data_default import TumorSegmentationDataModule
from data.tiled_data import TiledTumorDataModule
from utils import dice_score
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_enhanced_prediction(
    mip, seg, seg_pred, see_organs=False, intensity_threshold=85
):
    """
    Enhanced visualization function that can show either tumor or organ predictions.

    Args:
        mip: Input image [H, W, 3]
        seg: True segmentation [H, W, 3]
        seg_pred: Predicted segmentation [H, W, 3]
        see_organs: If True, interpret predictions as organ detection. If False, as tumor detection.
        intensity_threshold: Threshold for dark pixels (used when see_organs=True)
    """
    score = dice_score(seg, seg_pred)

    if see_organs:
        title_prefix = "Organ Detection (Threshold-Masked):"
        pred_title = "Predicted Organs"
        analysis_title = f"Organ Analysis (dice = {score:.02f})"

        # For organ detection, we interpret predictions differently
        print(f"Organ Detection - Dice Score: {score:.4f}")

        # Show dark pixels for context
        if len(mip.shape) == 3:
            mip_gray = cv2.cvtColor(mip, cv2.COLOR_RGB2GRAY)
        else:
            mip_gray = mip
        dark_pixels = (mip_gray < intensity_threshold).sum()
        total_pixels = mip_gray.size

        # For organ detection, count predictions only in dark regions
        if len(seg_pred.shape) == 3:
            pred_gray = cv2.cvtColor(seg_pred, cv2.COLOR_RGB2GRAY)
        else:
            pred_gray = seg_pred

        dark_mask = mip_gray < intensity_threshold
        bright_mask = mip_gray >= intensity_threshold
        predicted_in_dark = (pred_gray > 127)[dark_mask].sum()
        predicted_in_bright = (pred_gray > 127)[bright_mask].sum()

        print("Threshold Analysis:")
        print(
            f"  Dark pixels (<{intensity_threshold}): {dark_pixels:,} / {total_pixels:,} ({dark_pixels / total_pixels * 100:.1f}%)"
        )
        print(
            f"  Predicted organs in dark regions: {predicted_in_dark:,} ({predicted_in_dark / max(dark_pixels, 1) * 100:.1f}% of dark pixels)"
        )
        print(
            f"  ⚠️  Predicted organs in BRIGHT regions: {predicted_in_bright:,} (should be 0!)"
        )
        if predicted_in_bright > 0:
            print(
                "    🚨 ERROR: Model is predicting in bright areas - threshold masking failed!"
            )

    else:
        title_prefix = "Tumor Detection:"
        pred_title = "Predicted Tumors"
        analysis_title = f"Tumor Analysis (dice = {score:.02f})"
        print(f"Tumor Detection - Dice Score: {score:.4f}")

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(mip)
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(seg)
    plt.axis("off")
    if see_organs:
        plt.title("Reference\n(Dark regions)")
    else:
        plt.title("True Segmentation")

    plt.subplot(1, 4, 3)
    plt.imshow(seg_pred)
    plt.axis("off")
    plt.title(pred_title)

    # Analysis subplot - same logic but different interpretation
    TP = ((seg_pred > 0) & (seg > 0))[:, :, :1]
    FP = ((seg_pred > 0) & (seg == 0))[:, :, :1]
    FN = ((seg_pred == 0) & (seg > 0))[:, :, :1]
    img = np.concatenate((FP, TP, FN), axis=2).astype(np.uint8) * 255

    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.axis("off")
    plt.title(analysis_title)

    # Create legend with appropriate labels
    if see_organs:
        green_label = "Correctly identified organs"
        red_label = "False organ predictions"
        blue_label = "Missed organs"
    else:
        green_label = "True Positives (TP)"
        red_label = "False Positives (FP)"
        blue_label = "False Negatives (FN)"

    green_square = mpatches.Patch(color="green", label=green_label)
    red_square = mpatches.Patch(color="red", label=red_label)
    blue_square = mpatches.Patch(color="blue", label=blue_label)

    plt.legend(handles=[green_square, red_square, blue_square], loc="lower right")

    # Add title
    plt.suptitle(title_prefix, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_baseline_predictions(see_organs=False, model_checkpoint=None):
    """
    Plot predictions from either BaselineModel (tumor detection) or OrganDetector (organ detection).

    Args:
        see_organs: If True, use OrganDetector for organ predictions. If False, use SimpleUNet for tumor predictions.
        model_checkpoint: Optional path to model checkpoint. If None, uses default checkpoints.
    """
    if see_organs:
        print("Loading OrganDetector for organ detection...")

        # Default OrganDetector checkpoint - update this path as needed
        if model_checkpoint is None:
            model_checkpoint = "checkpoints/organ-detector-epoch=34-val_dice_patients=0.7060.ckpt"  # Update with actual path

        try:
            model = OrganDetector.load_from_checkpoint(
                model_checkpoint, map_location="cpu", tile_height=128
            )
            model.eval()
            print(f"✅ Loaded OrganDetector from {model_checkpoint}")
        except Exception as e:
            print(f"❌ Could not load OrganDetector: {e}")
            print(
                "Please provide a valid checkpoint path or train the OrganDetector model first."
            )
            return

        # Create tiled data module
        datamodule = TiledTumorDataModule(
            data_dir="data", batch_size=4, num_workers=0, tile_height=128, val_split=0.2
        )
    else:
        print("Loading SimpleUNet for tumor detection...")

        # Default SimpleUNet checkpoint
        if model_checkpoint is None:
            model_checkpoint = "C:/Users/Benja/dev/dm-i-ai-2025/tumor_segmentation/checkpoints/simple-unet-epoch=20-val_dice=0.3963.ckpt"

        model = SimpleUNet.load_from_checkpoint(
            model_checkpoint,
            map_location="cpu",
        )
        model.eval()
        print(f"✅ Loaded SimpleUNet from {model_checkpoint}")

        # Create regular data module
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
            if see_organs:
                # For organ detection, apply threshold masking to predictions
                # The model expects individual tiles, but we need to mask predictions
                predictions = model(images)

                # CRITICAL: Apply threshold masking to organ predictions
                # Convert images back to 255 range for threshold comparison
                images_255 = (images * 255.0).clamp(0, 255)
                threshold_mask = (images_255 < model.intensity_threshold).float()

                # Apply threshold mask - zero out predictions in bright areas
                predictions = predictions * threshold_mask

                print(f"  Applied threshold masking (<{model.intensity_threshold}):")
                bright_predictions = (predictions * (1 - threshold_mask)).sum().item()
                dark_predictions = (predictions * threshold_mask).sum().item()
                print(
                    f"    Predictions in bright areas: {bright_predictions:.6f} (should be 0)"
                )
                print(f"    Predictions in dark areas: {dark_predictions:.6f}")
            else:
                # Regular full-image prediction
                predictions = model(images)

        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: {predictions.min():.6f}-{predictions.max():.6f}")

        # Convert to numpy and process for plotting
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        # Get single sample (remove batch dimension)
        img = images_np[0]  # [1, H, W] - single channel grayscale
        mask = masks_np[0]  # [1, H, W]
        pred = predictions_np[0]  # [1, H, W]

        # Convert to format expected by plot_prediction: [H, W, 3]
        # Extract single channel and convert to proper range
        img_2d = img[0]  # [H, W] - extract single channel
        img_2d = (img_2d * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
        img_hwc = np.stack(
            [img_2d, img_2d, img_2d], axis=2
        )  # [H, W, 3] - replicate to RGB

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

        # Plot using enhanced function
        try:
            plot_enhanced_prediction(img_hwc, mask_hwc, pred_hwc, see_organs=see_organs)
        except Exception as e:
            print(f"  Error plotting: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot model predictions")
    parser.add_argument(
        "--see-organs",
        action="store_true",
        default=False,
        help="Use OrganDetector to visualize organ predictions instead of tumor predictions",
    )
    parser.add_argument("--model-checkpoint", type=str, help="Path to model checkpoint")

    args = parser.parse_args()

    plot_baseline_predictions(
        see_organs=args.see_organs, model_checkpoint=args.model_checkpoint
    )
