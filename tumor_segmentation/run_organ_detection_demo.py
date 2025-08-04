#!/usr/bin/env python3
"""
Demo script for organ detection using tiled inference.

This script demonstrates how to:
1. Train the OrganDetector model
2. Run tiled inference on full images
3. Visualize organ predictions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tiled_inference import demo_tiled_inference
from plot_baseline_predictions import plot_baseline_predictions


def main():
    """Run organ detection demonstrations"""
    print("🧠 Organ Detection Demo Suite")
    print("=" * 60)

    # Check if we have a trained model
    checkpoint_paths = [
        "checkpoints/organ-detector-latest.ckpt",
        "checkpoints/tiled-organ-detector-latest.ckpt",
        Path("checkpoints").glob("*organ*detector*.ckpt").__next__()
        if list(Path("checkpoints").glob("*organ*detector*.ckpt"))
        else None,
    ]

    model_checkpoint = None
    for path in checkpoint_paths:
        if path and Path(path).exists():
            model_checkpoint = str(path)
            break

    if model_checkpoint is None:
        print("❌ No OrganDetector checkpoint found!")
        print("\nTo run this demo, you need to train the OrganDetector model first:")
        print(
            "  python trainer.py fit --config models/config_base.yaml --config models/OrganDetector/config.yaml --config models/OrganDetector/wandb.yaml"
        )
        print("\nOr provide a checkpoint path manually.")
        return

    print(f"✅ Found model checkpoint: {model_checkpoint}")

    # Demo 1: Tiled inference on a full image
    print(f"\n🔬 Demo 1: Tiled Inference on Full Image")
    print("-" * 40)

    try:
        demo_tiled_inference(
            model_checkpoint=model_checkpoint,
            image_path=None,  # Will use first patient image
            see_organs=True,
            tile_height=128,
            intensity_threshold=85,
        )
        print("✅ Tiled inference demo completed!")
    except Exception as e:
        print(f"❌ Tiled inference demo failed: {e}")

    # Demo 2: Tile-by-tile predictions from dataloader
    print(f"\n📊 Demo 2: Tile Predictions from DataLoader")
    print("-" * 40)

    try:
        plot_baseline_predictions(see_organs=True, model_checkpoint=model_checkpoint)
        print("✅ Tile predictions demo completed!")
    except Exception as e:
        print(f"❌ Tile predictions demo failed: {e}")

    # Comparison demo
    print(f"\n🆚 Demo 3: Comparison with Tumor Detection")
    print("-" * 40)
    print(
        "This shows the difference between organ detection (green) and tumor detection (red)"
    )

    try:
        # First show tumor detection (if SimpleUNet checkpoint exists)
        print("\nTumor Detection View:")
        plot_baseline_predictions(
            see_organs=False,  # Show tumor predictions
            model_checkpoint=None,  # Use default SimpleUNet
        )

        print("\nOrgan Detection View:")
        plot_baseline_predictions(
            see_organs=True,  # Show organ predictions
            model_checkpoint=model_checkpoint,
        )

        print("✅ Comparison demo completed!")
    except Exception as e:
        print(f"❌ Comparison demo failed: {e}")

    print(f"\n🎉 All demos completed!")
    print(f"\n💡 Key Insights:")
    print(f"   - Organ detection targets non-tumor dark pixels (< 85 intensity)")
    print(f"   - Green areas show predicted organs (liver, heart, etc.)")
    print(f"   - Tiled approach allows processing of full-resolution images")
    print(f"   - Controls provide positive examples of normal organ tissue")


if __name__ == "__main__":
    main()
