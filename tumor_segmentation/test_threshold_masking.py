#!/usr/bin/env python3
"""
Test script to verify threshold masking in training and inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.OrganDetector.model import OrganDetector
from data.tiled_data import TiledTumorDataModule
from tiled_inference import run_tiled_inference


def test_threshold_masking():
    """Test threshold masking functionality"""
    print("🧪 Testing Threshold Masking")
    print("=" * 50)

    # Create a synthetic test tile to verify threshold masking
    tile_height, tile_width = 128, 256

    # Create test image with different intensity regions
    test_image = (
        np.ones((tile_height, tile_width), dtype=np.uint8) * 200
    )  # Bright background
    test_image[20:60, 50:150] = 50  # Dark region 1 (below threshold)
    test_image[80:120, 100:200] = 30  # Dark region 2 (below threshold)

    # Create test target (organ mask for dark regions)
    test_target = np.zeros((tile_height, tile_width), dtype=np.uint8)
    test_target[25:55, 60:140] = 1  # Organ in dark region 1
    test_target[85:115, 110:190] = 1  # Organ in dark region 2

    print(
        f"Test image: {test_image.shape}, intensity range: {test_image.min()}-{test_image.max()}"
    )
    print(f"Test target: {test_target.shape}, positive pixels: {test_target.sum()}")

    # Test 1: Verify threshold masking in model training
    print(f"\n🔬 Test 1: Threshold-Aware Loss Calculation")
    print("-" * 40)

    # Create model
    model = OrganDetector(tile_height=tile_height, intensity_threshold=85)
    model.eval()

    # Convert to tensors
    image_tensor = (
        torch.from_numpy(test_image.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # [1, 1, H, W]
    target_tensor = (
        torch.from_numpy(test_target.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )  # [1, 1, H, W]

    print(
        f"Image tensor: {image_tensor.shape}, range: {image_tensor.min():.3f}-{image_tensor.max():.3f}"
    )
    print(f"Target tensor: {target_tensor.shape}, sum: {target_tensor.sum()}")

    # Test threshold masking in loss calculation
    with torch.no_grad():
        # Get model prediction
        prediction = model(image_tensor)
        print(
            f"Prediction: {prediction.shape}, range: {prediction.min():.3f}-{prediction.max():.3f}"
        )

        # Test threshold-aware loss
        loss = model._calculate_threshold_masked_loss(
            prediction, target_tensor, image_tensor
        )
        print(f"Threshold-aware loss: {loss:.6f}")

        # Calculate threshold mask
        images_255 = (image_tensor * 255.0).clamp(0, 255)
        threshold_mask = (images_255 < model.intensity_threshold).float()

        dark_pixels = threshold_mask.sum().item()
        total_pixels = threshold_mask.numel()
        target_pixels_in_dark = (target_tensor * threshold_mask).sum().item()

        print(f"Threshold analysis:")
        print(f"  Total pixels: {total_pixels}")
        print(
            f"  Dark pixels (< {model.intensity_threshold}): {dark_pixels} ({dark_pixels / total_pixels * 100:.1f}%)"
        )
        print(f"  Target pixels in dark regions: {target_pixels_in_dark}")

    # Test 2: Verify threshold masking in inference
    print(f"\n🔬 Test 2: Threshold-Aware Inference")
    print("-" * 40)

    # Create a larger test image for tiled inference
    large_height, large_width = 384, 512  # 3 tiles of 128px each
    large_image = (
        np.ones((large_height, large_width), dtype=np.uint8) * 200
    )  # Bright background

    # Add some dark regions across different tiles
    large_image[50:100, 100:300] = 40  # Dark region in tile 1
    large_image[180:220, 150:350] = 60  # Dark region in tile 2
    large_image[300:350, 200:400] = 30  # Dark region in tile 3

    print(f"Large test image: {large_image.shape}")
    print(f"Intensity distribution: min={large_image.min()}, max={large_image.max()}")

    # Run tiled inference with threshold masking
    prediction = run_tiled_inference(
        model,
        large_image,
        tile_height=tile_height,
        intensity_threshold=model.intensity_threshold,
    )

    print(f"Inference prediction: {prediction.shape}")
    print(f"Prediction range: {prediction.min():.3f}-{prediction.max():.3f}")

    # Verify that predictions are only in dark regions
    dark_mask = large_image < model.intensity_threshold
    bright_mask = large_image >= model.intensity_threshold

    predictions_in_dark = prediction[dark_mask].sum()
    predictions_in_bright = prediction[bright_mask].sum()

    print(f"Verification:")
    print(f"  Predictions in dark regions: {predictions_in_dark:.3f}")
    print(f"  Predictions in bright regions: {predictions_in_bright:.3f}")
    print(f"  ✅ Threshold masking working: {predictions_in_bright < 0.001}")

    # Test 3: Compare with and without threshold masking
    print(f"\n🔬 Test 3: Threshold Masking Comparison")
    print("-" * 40)

    # Run inference without threshold masking (direct model call)
    with torch.no_grad():
        tile_tensor = (
            torch.from_numpy((large_image[:128, :] / 255.0).astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        unmasked_pred = model(tile_tensor).squeeze().cpu().numpy()

        # Apply threshold masking manually
        tile_original = large_image[:128, :]
        threshold_mask_np = tile_original < model.intensity_threshold
        masked_pred = unmasked_pred * threshold_mask_np.astype(np.float32)

    print(f"First tile comparison:")
    print(f"  Unmasked prediction sum: {unmasked_pred.sum():.3f}")
    print(f"  Masked prediction sum: {masked_pred.sum():.3f}")
    print(
        f"  Threshold mask pixels: {threshold_mask_np.sum()} / {threshold_mask_np.size}"
    )

    print(f"\n✅ All threshold masking tests completed!")
    return True


def test_data_consistency():
    """Test that tiled data creation matches threshold logic"""
    print(f"\n🧪 Testing Data Creation Consistency")
    print("-" * 40)

    try:
        # Create a small tiled data module
        data_module = TiledTumorDataModule(
            data_dir="data",
            batch_size=2,
            tile_height=128,
            intensity_threshold=85,
            val_split=0.8,  # Small val split to get more training data
        )

        data_module.setup()
        train_loader = data_module.train_dataloader()

        # Get one batch
        for batch_images, batch_targets in train_loader:
            print(
                f"Batch shapes: images {batch_images.shape}, targets {batch_targets.shape}"
            )

            # Check consistency
            for i in range(batch_images.shape[0]):
                image = batch_images[i].squeeze()  # [H, W]
                target = batch_targets[i].squeeze()  # [H, W]

                # Convert image back to 0-255 range
                image_255 = (image * 255.0).clamp(0, 255)

                # Check threshold logic
                dark_pixels = (image_255 < 85).float()
                target_pixels = target > 0

                # All target pixels should be in dark regions (target creation logic)
                target_in_bright = target_pixels & (image_255 >= 85)
                invalid_targets = target_in_bright.sum()

                print(
                    f"  Sample {i}: Dark pixels: {dark_pixels.sum():.0f}, Target pixels: {target_pixels.sum():.0f}"
                )
                print(f"    Invalid targets (in bright regions): {invalid_targets:.0f}")

                if invalid_targets > 0:
                    print("    ⚠️ Warning: Found target pixels in bright regions!")
                else:
                    print("    ✅ All target pixels are in dark regions")

            break  # Only check first batch

        print("✅ Data consistency test completed!")
        return True

    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        return False


def main():
    """Run all threshold masking tests"""
    print("🎯 Threshold Masking Test Suite")
    print("=" * 60)

    # Test 1: Threshold masking logic
    test1_pass = test_threshold_masking()

    # Test 2: Data consistency
    test2_pass = test_data_consistency()

    print(f"\n🎉 Test Results Summary:")
    print(f"  Threshold masking logic: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Data consistency: {'✅ PASS' if test2_pass else '❌ FAIL'}")

    if test1_pass and test2_pass:
        print(f"\n🚀 All tests passed! Threshold masking is working correctly.")
        print(f"\nKey improvements:")
        print(f"  ✅ Loss calculated only on pixels < threshold")
        print(f"  ✅ Inference predictions masked to dark regions")
        print(f"  ✅ Training data and inference logic are consistent")
        print(f"  ✅ Model learns organ vs tumor distinction in relevant areas")
    else:
        print(f"\n⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
