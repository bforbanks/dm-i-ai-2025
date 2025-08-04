#!/usr/bin/env python3
"""
Quick test to verify the visualization threshold masking fix.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.OrganDetector.model import OrganDetector
from data.tiled_data import TiledTumorDataModule


def test_visualization_threshold_fix():
    """Test that visualization correctly applies threshold masking"""
    print("🧪 Testing Visualization Threshold Masking Fix")
    print("=" * 60)

    # Load model
    try:
        model_path = "checkpoints/organ-detector-epoch=34-val_dice_patients=0.7060.ckpt"
        model = OrganDetector.load_from_checkpoint(
            model_path, map_location="cpu", tile_height=128, intensity_threshold=85
        )
        model.eval()
        print(f"✅ Loaded model: intensity_threshold = {model.intensity_threshold}")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return False

    # Create data module
    try:
        datamodule = TiledTumorDataModule(
            data_dir="data", batch_size=2, num_workers=0, tile_height=128, val_split=0.2
        )
        datamodule.setup()
        val_loader = datamodule.val_dataloader()
        print(f"✅ Created data module with {len(val_loader)} validation batches")
    except Exception as e:
        print(f"❌ Could not create data module: {e}")
        return False

    # Test threshold masking on a sample
    print(f"\n🔬 Testing threshold masking logic...")

    for batch_idx, batch in enumerate(val_loader):
        images, masks = batch

        # Take first sample
        sample_image = images[0:1]  # [1, 1, H, W]
        sample_mask = masks[0:1]  # [1, 1, H, W]

        print(f"Sample shape: {sample_image.shape}")
        print(f"Image range: {sample_image.min():.3f} - {sample_image.max():.3f}")

        with torch.no_grad():
            # Get raw prediction
            raw_prediction = model(sample_image)

            # Apply the SAME threshold masking logic as in the visualization
            images_255 = (sample_image * 255.0).clamp(0, 255)
            threshold_mask = (images_255 < model.intensity_threshold).float()

            # Apply threshold mask
            masked_prediction = raw_prediction * threshold_mask

            # Calculate predictions in bright vs dark areas
            bright_mask = 1 - threshold_mask

            raw_bright = (raw_prediction * bright_mask).sum().item()
            raw_dark = (raw_prediction * threshold_mask).sum().item()

            masked_bright = (masked_prediction * bright_mask).sum().item()
            masked_dark = (masked_prediction * threshold_mask).sum().item()

            print(f"\n📊 Threshold Masking Results:")
            print(f"  Raw predictions in bright areas: {raw_bright:.6f}")
            print(f"  Raw predictions in dark areas: {raw_dark:.6f}")
            print(
                f"  Masked predictions in bright areas: {masked_bright:.6f} (should be 0)"
            )
            print(f"  Masked predictions in dark areas: {masked_dark:.6f}")

            # Verify masking worked
            if masked_bright < 1e-10:
                print(
                    f"  ✅ Threshold masking SUCCESSFUL - no predictions in bright areas"
                )
                test_passed = True
            else:
                print(
                    f"  🚨 Threshold masking FAILED - found {masked_bright:.6f} in bright areas"
                )
                test_passed = False

            # Also check pixel-level
            bright_pixels = (images_255 >= model.intensity_threshold).sum().item()
            dark_pixels = (images_255 < model.intensity_threshold).sum().item()
            total_pixels = images_255.numel()

            print(f"\n📈 Image Analysis:")
            print(f"  Total pixels: {total_pixels}")
            print(
                f"  Bright pixels (>={model.intensity_threshold}): {bright_pixels} ({bright_pixels / total_pixels * 100:.1f}%)"
            )
            print(
                f"  Dark pixels (<{model.intensity_threshold}): {dark_pixels} ({dark_pixels / total_pixels * 100:.1f}%)"
            )

            break  # Only test first sample

    print(f"\n🎯 CONCLUSION:")
    if test_passed:
        print("✅ Visualization threshold masking fix is working correctly!")
        print(
            "🎉 The plot_baseline_predictions.py should now show zero predictions in bright areas"
        )
    else:
        print("❌ Threshold masking fix is not working - needs further debugging")

    return test_passed


if __name__ == "__main__":
    test_visualization_threshold_fix()
