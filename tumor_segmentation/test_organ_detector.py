#!/usr/bin/env python3
"""
Test script for the OrganDetector model with tiled data.
"""

import torch
from models.OrganDetector.model import OrganDetector
from data.tiled_data import TiledTumorDataModule


def test_model_architecture():
    """Test the OrganDetector model architecture with tiled dimensions"""
    print("🧠 Testing OrganDetector Model for Tiled Data")
    print("=" * 50)

    # Create model
    model = OrganDetector(in_channels=1, num_classes=1, tile_height=128, lr=1e-3)

    print(f"📐 Model Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test with different tile widths
    test_widths = [256, 512, 768, 1024]

    for width in test_widths:
        print(f"\n🔧 Testing with width {width}px...")
        success = model.test_tiled_input(width=width, batch_size=2)
        if not success:
            print(f"❌ Failed with width {width}")
            return False

    print(f"\n✅ All tile dimension tests passed!")
    return True


def test_with_real_data():
    """Test model with actual tiled dataset"""
    print(f"\n🏥 Testing with Real Tiled Dataset")
    print("-" * 30)

    try:
        # Create data module
        data_module = TiledTumorDataModule(
            data_dir="data",
            batch_size=4,
            tile_height=128,
            intensity_threshold=85,
        )

        # Setup data
        data_module.setup()
        train_loader = data_module.train_dataloader()

        # Create model
        model = OrganDetector(tile_height=128)
        model.eval()

        # Test with one real batch
        for batch_images, batch_targets in train_loader:
            print(f"📊 Real data batch:")
            print(f"   Images shape: {batch_images.shape}")
            print(f"   Targets shape: {batch_targets.shape}")

            with torch.no_grad():
                predictions = model(batch_images)

            print(f"   Predictions shape: {predictions.shape}")
            print(f"   Target pixels in batch: {batch_targets.sum().item():.0f}")
            print(
                f"   Predicted pixels in batch: {(predictions > 0.5).sum().item():.0f}"
            )

            break

        print(f"✅ Real data test successful!")
        return True

    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🎯 OrganDetector Tiled Data Compatibility Test")
    print("=" * 60)

    # Test 1: Architecture
    arch_success = test_model_architecture()

    # Test 2: Real data (if available)
    if arch_success:
        data_success = test_with_real_data()

        if arch_success and data_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ OrganDetector is ready for tiled tumor segmentation!")
            print(f"\n📋 Model Summary:")
            print(f"   - Optimized for 128px height tiles")
            print(f"   - Handles variable width inputs")
            print(f"   - 3-level encoder/decoder architecture")
            print(f"   - Predicts non-tumor dark pixels (organs)")
        else:
            print(f"\n⚠️ Some tests failed. Check the output above.")
    else:
        print(f"\n❌ Architecture test failed. Model needs fixes.")


if __name__ == "__main__":
    main()
