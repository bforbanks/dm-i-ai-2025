#!/usr/bin/env python3
"""
Quick analysis of organ:tumor ratio for cost function optimization.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tiled_data import TiledTumorDataModule


def quick_organ_tumor_analysis(intensity_threshold=85, max_batches=5):
    """Quick analysis with limited batches for faster results"""
    print("🔍 QUICK ORGAN:TUMOR RATIO ANALYSIS")
    print("=" * 50)

    data_module = TiledTumorDataModule(
        data_dir="data",
        batch_size=16,
        tile_height=128,
        intensity_threshold=intensity_threshold,
        val_split=0.2,
    )
    data_module.setup()

    stats = {
        "total_dark_pixels": 0,
        "total_organ_pixels": 0,
        "total_tumor_pixels": 0,
        "tiles_analyzed": 0,
    }

    train_loader = data_module.train_dataloader()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break

        batch_size = images.shape[0]

        for i in range(batch_size):
            image = images[i].squeeze()
            target = targets[i].squeeze()

            image_255 = (image * 255.0).clamp(0, 255).numpy()
            target_np = target.numpy()

            dark_mask = image_255 < intensity_threshold
            dark_pixels = dark_mask.sum()

            organ_pixels = (target_np > 0.5).sum()
            tumor_pixels = dark_pixels - organ_pixels

            stats["total_dark_pixels"] += dark_pixels
            stats["total_organ_pixels"] += organ_pixels
            stats["total_tumor_pixels"] += tumor_pixels
            stats["tiles_analyzed"] += 1

    organ_ratio = stats["total_organ_pixels"] / stats["total_dark_pixels"]
    tumor_ratio = stats["total_tumor_pixels"] / stats["total_dark_pixels"]
    organ_tumor_ratio = stats["total_organ_pixels"] / max(
        stats["total_tumor_pixels"], 1
    )

    print(f"Analyzed {stats['tiles_analyzed']} tiles")
    print(f"Dark pixels: {stats['total_dark_pixels']:,}")
    print(f"Organ pixels: {stats['total_organ_pixels']:,} ({organ_ratio:.1%})")
    print(f"Tumor pixels: {stats['total_tumor_pixels']:,} ({tumor_ratio:.1%})")
    print(f"Organ:Tumor ratio = {organ_tumor_ratio:.2f}:1")

    # Calculate optimal weights
    tumor_weight = 1.0 / tumor_ratio  # Inverse frequency
    organ_weight = 1.0 / organ_ratio

    # Normalize
    total = tumor_weight + organ_weight
    tumor_normalized = tumor_weight / total
    organ_normalized = organ_weight / total

    print(f"\n🎯 RECOMMENDED WEIGHTS:")
    print(f"Tumor emphasis weight: {tumor_normalized:.4f}")
    print(f"Organ weight: {organ_normalized:.4f}")

    return {
        "organ_ratio": organ_ratio,
        "tumor_ratio": tumor_ratio,
        "tumor_weight": tumor_normalized,
        "organ_weight": organ_normalized,
    }


if __name__ == "__main__":
    results = quick_organ_tumor_analysis()
