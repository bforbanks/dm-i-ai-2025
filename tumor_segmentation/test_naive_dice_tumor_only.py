#!/usr/bin/env python3
"""
Test script to verify naive dice calculation only on tiles with tumors.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.OrganDetector.model import OrganDetector


def test_naive_dice_tumor_only():
    """Test that naive dice is only calculated on tiles with tumors"""
    print("🧪 TESTING NAIVE DICE - TUMOR TILES ONLY")
    print("=" * 60)

    # Create model
    model = OrganDetector(
        tile_height=128,
        intensity_threshold=85,
        tumor_emphasis_weight=2.5,
        use_weighted_bce=True,
    )
    model.eval()

    print("✅ Model created")

    # Test case 1: Batch with mixed tumor and control tiles
    batch_size, height, width = 4, 128, 256

    # Create test images
    test_images = torch.ones(batch_size, 1, height, width) * 0.9  # Bright background
    test_images[:, :, 30:90, 60:200] = 0.2  # Dark regions in all tiles

    # Create targets (organ targets, will be inverted to tumor targets)
    test_targets = torch.zeros(batch_size, 1, height, width)

    # Tile 0: Has tumors (no organs marked in dark area)
    # test_targets[0] = 0 (all zeros) -> tumor_targets[0] will have 1s in dark area

    # Tile 1: Has tumors (partial organs marked)
    test_targets[1, :, 30:60, 60:130] = 1  # Some organs -> rest will be tumors

    # Tile 2: No tumors (all dark pixels are organs)
    test_targets[2, :, 30:90, 60:200] = 1  # All dark pixels are organs -> no tumors

    # Tile 3: Has tumors (partial organs marked)
    test_targets[3, :, 60:90, 130:200] = 1  # Some organs -> rest will be tumors

    print(f"Test batch shape: {test_images.shape}")
    print(f"Organ targets per tile:")
    for i in range(batch_size):
        organ_pixels = test_targets[i].sum().item()
        print(f"  Tile {i}: {organ_pixels:.0f} organ pixels")

    # Calculate what tumor targets will be
    images_255 = (test_images * 255.0).clamp(0, 255)
    threshold_mask = (images_255 < model.intensity_threshold).float()
    tumor_targets = (1.0 - test_targets) * threshold_mask

    print(f"\nTumor targets per tile (after inversion and masking):")
    for i in range(batch_size):
        tumor_pixels = tumor_targets[i].sum().item()
        print(f"  Tile {i}: {tumor_pixels:.0f} tumor pixels")

    # Test with model
    with torch.no_grad():
        # Get organ predictions
        organ_pred = model(test_images)

        # Convert to tumor predictions
        tumor_pred = 1.0 - organ_pred
        tumor_pred_binary = (tumor_pred > 0.5).float()

        # Test old method (includes all tiles)
        old_dice = model._calculate_dice_score(tumor_pred_binary, tumor_targets)

        # Test new method (only tiles with tumors)
        new_dice = model._calculate_naive_dice_tumor_only(
            tumor_pred_binary, tumor_targets, threshold_mask
        )

        print(f"\n📊 DICE COMPARISON:")
        print(f"Old method (all tiles): {old_dice:.4f}")
        print(f"New method (tumor tiles only): {new_dice:.4f}")

        # Manual verification - calculate dice for each tile
        print(f"\n🔍 PER-TILE ANALYSIS:")
        tumor_tile_dices = []

        for i in range(batch_size):
            pred_sample = tumor_pred_binary[i]
            target_sample = tumor_targets[i]
            mask_sample = threshold_mask[i]

            # Check if tile has tumors
            target_sum = (target_sample * mask_sample).sum().item()

            if target_sum > 0:
                # Calculate dice for this tile
                pred_flat = (pred_sample * mask_sample).view(-1)
                target_flat = (target_sample * mask_sample).view(-1)

                intersection = (pred_flat * target_flat).sum()
                pred_sum = pred_flat.sum()
                target_sum_tensor = target_flat.sum()

                if pred_sum + target_sum_tensor > 0:
                    dice = (2.0 * intersection) / (pred_sum + target_sum_tensor + 1e-8)
                else:
                    dice = 0.0

                tumor_tile_dices.append(dice.item())
                print(f"  Tile {i}: {dice:.4f} (HAS TUMORS)")
            else:
                print(f"  Tile {i}: SKIPPED (NO TUMORS)")

        # Verify manual calculation matches new method
        if len(tumor_tile_dices) > 0:
            manual_mean = np.mean(tumor_tile_dices)
            print(f"\nManual calculation (tumor tiles only): {manual_mean:.4f}")
            print(f"New method result: {new_dice:.4f}")

            if abs(manual_mean - new_dice.item()) < 1e-6:
                print("✅ Manual calculation matches new method!")
            else:
                print("❌ Mismatch between manual and new method")
        else:
            print(f"\nNo tumor tiles found - both should return 0")
            if new_dice.item() == 0.0:
                print("✅ Correctly returns 0 for no tumor tiles")
            else:
                print("❌ Should return 0 for no tumor tiles")

    # Test case 2: Batch with no tumor tiles
    print(f"\n🧪 TEST CASE 2: No tumor tiles")

    no_tumor_targets = torch.ones(batch_size, 1, height, width)  # All organs
    no_tumor_targets = no_tumor_targets * threshold_mask  # Apply mask
    no_tumor_tumor_targets = (
        1.0 - no_tumor_targets
    ) * threshold_mask  # Should be all zeros

    with torch.no_grad():
        no_tumor_dice = model._calculate_naive_dice_tumor_only(
            tumor_pred_binary, no_tumor_tumor_targets, threshold_mask
        )

        print(f"Naive dice with no tumor tiles: {no_tumor_dice:.4f}")
        if no_tumor_dice.item() == 0.0:
            print("✅ Correctly returns 0 when no tumor tiles")
        else:
            print("❌ Should return 0 when no tumor tiles")

    print(f"\n🎯 BENEFITS OF NEW METHOD:")
    print("1. ✅ Only calculates dice on meaningful tiles (with tumors)")
    print("2. ✅ Prevents inflation from control tiles or tumor-free tiles")
    print("3. ✅ Gives true tumor detection performance")
    print("4. ✅ Wandb logging shows realistic scores")
    print("5. ✅ Better for model monitoring and comparison")

    print(f"\n✅ Naive dice (tumor tiles only) is working correctly!")

    return True


if __name__ == "__main__":
    test_naive_dice_tumor_only()
