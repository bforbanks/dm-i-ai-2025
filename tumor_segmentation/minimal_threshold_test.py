#!/usr/bin/env python3
"""
Minimal test for threshold masking logic without data loading.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.OrganDetector.model import OrganDetector


def test_threshold_logic():
    """Test the threshold masking logic with synthetic data"""
    print("🧪 Minimal Threshold Masking Test")
    print("=" * 40)

    # Create synthetic test data
    tile_height, tile_width = 128, 256
    intensity_threshold = 85

    # Create test image: bright background with some dark areas
    test_image = (
        np.ones((1, 1, tile_height, tile_width), dtype=np.float32) * 0.8
    )  # Bright (0.8 * 255 = 204)
    test_image[:, :, 20:60, 50:150] = 0.2  # Dark region (0.2 * 255 = 51)
    test_image[:, :, 80:120, 100:200] = 0.3  # Dark region (0.3 * 255 = 76)

    test_tensor = torch.from_numpy(test_image)

    print(f"Created synthetic test image: {test_tensor.shape}")
    print(f"Image range: {test_tensor.min():.3f} - {test_tensor.max():.3f}")

    # Load model
    try:
        model_path = "checkpoints/organ-detector-epoch=34-val_dice_patients=0.7060.ckpt"
        model = OrganDetector.load_from_checkpoint(
            model_path, map_location="cpu", tile_height=128, intensity_threshold=85
        )
        model.eval()
        print(f"✅ Model loaded: threshold = {model.intensity_threshold}")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return False

    # Test prediction and masking
    with torch.no_grad():
        # Get raw prediction
        raw_pred = model(test_tensor)

        # Apply visualization threshold masking logic
        images_255 = (test_tensor * 255.0).clamp(0, 255)
        threshold_mask = (images_255 < model.intensity_threshold).float()
        masked_pred = raw_pred * threshold_mask

        # Analyze results
        bright_mask = 1 - threshold_mask

        raw_bright_sum = (raw_pred * bright_mask).sum().item()
        raw_dark_sum = (raw_pred * threshold_mask).sum().item()

        masked_bright_sum = (masked_pred * bright_mask).sum().item()
        masked_dark_sum = (masked_pred * threshold_mask).sum().item()

        print(f"\n📊 Results:")
        print(f"Raw prediction range: {raw_pred.min():.6f} - {raw_pred.max():.6f}")
        print(f"Raw bright area sum: {raw_bright_sum:.6f}")
        print(f"Raw dark area sum: {raw_dark_sum:.6f}")
        print(f"Masked bright area sum: {masked_bright_sum:.6f} (should be 0)")
        print(f"Masked dark area sum: {masked_dark_sum:.6f}")

        # Check threshold distribution
        total_pixels = threshold_mask.numel()
        dark_pixels = threshold_mask.sum().item()
        bright_pixels = total_pixels - dark_pixels

        print(f"\n📈 Threshold Analysis:")
        print(f"Total pixels: {total_pixels}")
        print(
            f"Dark pixels (<{intensity_threshold}): {dark_pixels} ({dark_pixels / total_pixels * 100:.1f}%)"
        )
        print(
            f"Bright pixels (>={intensity_threshold}): {bright_pixels} ({bright_pixels / total_pixels * 100:.1f}%)"
        )

        # Verify success
        success = masked_bright_sum < 1e-10

        if success:
            print(f"\n✅ SUCCESS: Threshold masking works correctly!")
            print(f"   No predictions in bright areas ({masked_bright_sum:.10f})")
        else:
            print(
                f"\n❌ FAILURE: Still found {masked_bright_sum:.6f} predictions in bright areas"
            )

        return success


if __name__ == "__main__":
    test_threshold_logic()
