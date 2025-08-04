#!/usr/bin/env python3
"""
Analyze the ratio of organ pixels to tumor pixels under the intensity threshold.
This analysis will inform the optimal cost function weighting.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tiled_data import TiledTumorDataModule


def analyze_organ_tumor_ratio(intensity_threshold=85):
    """
    Analyze the ratio of organ pixels to tumor pixels in dark regions.

    This analysis will help us determine the optimal weighting for the cost function
    to maximize tumor detection performance.
    """
    print("🔍 ANALYZING ORGAN:TUMOR PIXEL RATIO")
    print("=" * 60)

    # Create data module to get the tiled data
    data_module = TiledTumorDataModule(
        data_dir="data",
        batch_size=32,
        tile_height=128,
        intensity_threshold=intensity_threshold,
        val_split=0.2,
    )
    data_module.setup()

    # Statistics collection
    stats = {
        "total_dark_pixels": 0,
        "total_organ_pixels": 0,  # Dark pixels that are NOT tumors
        "total_tumor_pixels": 0,  # Dark pixels that ARE tumors
        "total_bright_pixels": 0,
        "patient_tiles": 0,
        "control_tiles": 0,
        "tiles_analyzed": 0,
    }

    # Per-tile statistics for distribution analysis
    tile_stats = []

    print(f"Analyzing training data with threshold = {intensity_threshold}")

    # Analyze training data
    train_loader = data_module.train_dataloader()

    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_size = images.shape[0]

        for i in range(batch_size):
            image = images[i].squeeze()  # [H, W]
            target = targets[i].squeeze()  # [H, W]

            # Convert image back to 0-255 range
            image_255 = (image * 255.0).clamp(0, 255).numpy()
            target_np = target.numpy()

            # Calculate masks
            dark_mask = image_255 < intensity_threshold
            bright_mask = image_255 >= intensity_threshold

            # In our setup:
            # - target = 1 means "organ" (dark pixel that is NOT tumor)
            # - target = 0 in dark areas means "tumor" (dark pixel that IS tumor)
            # - target = 0 in bright areas means "irrelevant" (bright pixel)

            dark_pixels = dark_mask.sum()
            bright_pixels = bright_mask.sum()

            # Organ pixels: target = 1 (these are dark non-tumor pixels)
            organ_pixels = (target_np > 0.5).sum()

            # Tumor pixels: dark pixels that are NOT organs
            tumor_pixels = dark_pixels - organ_pixels

            # Update global statistics
            stats["total_dark_pixels"] += dark_pixels
            stats["total_bright_pixels"] += bright_pixels
            stats["total_organ_pixels"] += organ_pixels
            stats["total_tumor_pixels"] += tumor_pixels
            stats["tiles_analyzed"] += 1

            # Determine tile type (patient vs control)
            # If there are any tumor pixels, it's a patient tile
            is_patient = tumor_pixels > 0
            if is_patient:
                stats["patient_tiles"] += 1
            else:
                stats["control_tiles"] += 1

            # Store per-tile statistics
            tile_stats.append(
                {
                    "dark_pixels": dark_pixels,
                    "organ_pixels": organ_pixels,
                    "tumor_pixels": tumor_pixels,
                    "bright_pixels": bright_pixels,
                    "is_patient": is_patient,
                    "organ_ratio": organ_pixels / max(dark_pixels, 1),
                    "tumor_ratio": tumor_pixels / max(dark_pixels, 1),
                }
            )

        # Progress update
        if batch_idx % 10 == 0:
            print(f"  Processed batch {batch_idx}/{len(train_loader)}")

    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 40)

    # Global statistics
    total_pixels = stats["total_dark_pixels"] + stats["total_bright_pixels"]
    organ_tumor_ratio = stats["total_organ_pixels"] / max(
        stats["total_tumor_pixels"], 1
    )

    print(f"Total tiles analyzed: {stats['tiles_analyzed']}")
    print(f"  Patient tiles (with tumors): {stats['patient_tiles']}")
    print(f"  Control tiles (no tumors): {stats['control_tiles']}")

    print(f"\nPixel Distribution:")
    print(f"  Total pixels: {total_pixels:,}")
    print(
        f"  Dark pixels (<{intensity_threshold}): {stats['total_dark_pixels']:,} ({stats['total_dark_pixels'] / total_pixels * 100:.1f}%)"
    )
    print(
        f"  Bright pixels (>={intensity_threshold}): {stats['total_bright_pixels']:,} ({stats['total_bright_pixels'] / total_pixels * 100:.1f}%)"
    )

    print(f"\nDark Pixel Breakdown:")
    print(
        f"  Organ pixels (dark non-tumor): {stats['total_organ_pixels']:,} ({stats['total_organ_pixels'] / stats['total_dark_pixels'] * 100:.1f}%)"
    )
    print(
        f"  Tumor pixels (dark tumor): {stats['total_tumor_pixels']:,} ({stats['total_tumor_pixels'] / stats['total_dark_pixels'] * 100:.1f}%)"
    )

    print(f"\n🎯 KEY METRICS:")
    print(f"  Organ:Tumor ratio = {organ_tumor_ratio:.3f}:1")
    print(f"  Tumor:Organ ratio = {1 / organ_tumor_ratio:.3f}:1")

    # Class imbalance analysis
    organ_weight = stats["total_dark_pixels"] / (2 * stats["total_organ_pixels"])
    tumor_weight = stats["total_dark_pixels"] / (2 * stats["total_tumor_pixels"])

    print(f"\n⚖️  RECOMMENDED WEIGHTS for balanced BCE:")
    print(f"  Organ weight (pos_weight): {organ_weight:.4f}")
    print(f"  Tumor weight (neg_weight): {tumor_weight:.4f}")
    print(
        f"  Normalized organ weight: {organ_weight / (organ_weight + tumor_weight):.4f}"
    )
    print(
        f"  Normalized tumor weight: {tumor_weight / (organ_weight + tumor_weight):.4f}"
    )

    # Distribution analysis
    tile_stats = np.array(
        [(t["organ_ratio"], t["tumor_ratio"], t["is_patient"]) for t in tile_stats]
    )

    patient_mask = tile_stats[:, 2] == 1
    control_mask = tile_stats[:, 2] == 0

    print(f"\n📈 DISTRIBUTION ANALYSIS:")
    if patient_mask.sum() > 0:
        print(f"  Patient tiles:")
        print(
            f"    Avg organ ratio: {tile_stats[patient_mask, 0].mean():.3f} ± {tile_stats[patient_mask, 0].std():.3f}"
        )
        print(
            f"    Avg tumor ratio: {tile_stats[patient_mask, 1].mean():.3f} ± {tile_stats[patient_mask, 1].std():.3f}"
        )

    if control_mask.sum() > 0:
        print(f"  Control tiles:")
        print(
            f"    Avg organ ratio: {tile_stats[control_mask, 0].mean():.3f} ± {tile_stats[control_mask, 0].std():.3f}"
        )
        print(
            f"    Avg tumor ratio: {tile_stats[control_mask, 1].mean():.3f} ± {tile_stats[control_mask, 1].std():.3f}"
        )

    # Calculate optimal weights for final tumor detection task
    print(f"\n🎯 OPTIMIZATION FOR TUMOR DETECTION:")

    # For tumor detection, we want to maximize tumor recall while maintaining precision
    # Since organs are more common, we need to weight tumor detection higher

    # Method 1: Inverse frequency weighting
    tumor_frequency = stats["total_tumor_pixels"] / stats["total_dark_pixels"]
    organ_frequency = stats["total_organ_pixels"] / stats["total_dark_pixels"]

    tumor_inv_freq_weight = 1.0 / tumor_frequency
    organ_inv_freq_weight = 1.0 / organ_frequency

    # Normalize to sum to 2 (standard for binary classification)
    total_weight = tumor_inv_freq_weight + organ_inv_freq_weight
    tumor_normalized = 2 * tumor_inv_freq_weight / total_weight
    organ_normalized = 2 * organ_inv_freq_weight / total_weight

    print(f"  Inverse frequency weighting:")
    print(f"    Tumor weight: {tumor_normalized:.4f}")
    print(f"    Organ weight: {organ_normalized:.4f}")

    # Method 2: Focal loss inspired weighting (emphasize hard examples)
    # Since tumors are minority class, use higher weight
    focal_tumor_weight = 2.0
    focal_organ_weight = 0.5

    print(f"  Focal-inspired weighting (emphasize tumors):")
    print(f"    Tumor weight: {focal_tumor_weight:.4f}")
    print(f"    Organ weight: {focal_organ_weight:.4f}")

    # Visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Overall pixel distribution
    plt.subplot(2, 3, 1)
    labels = [
        "Bright\n(Irrelevant)",
        "Organs\n(Dark Non-Tumor)",
        "Tumors\n(Dark Tumor)",
    ]
    sizes = [
        stats["total_bright_pixels"],
        stats["total_organ_pixels"],
        stats["total_tumor_pixels"],
    ]
    colors = ["lightgray", "green", "red"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Overall Pixel Distribution")

    # Plot 2: Dark pixel breakdown
    plt.subplot(2, 3, 2)
    labels = ["Organs", "Tumors"]
    sizes = [stats["total_organ_pixels"], stats["total_tumor_pixels"]]
    colors = ["green", "red"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Dark Pixels: Organs vs Tumors")

    # Plot 3: Organ ratio distribution
    plt.subplot(2, 3, 3)
    organ_ratios = [t["organ_ratio"] for t in tile_stats]
    plt.hist(organ_ratios, bins=20, alpha=0.7, color="green", edgecolor="black")
    plt.xlabel("Organ Ratio per Tile")
    plt.ylabel("Frequency")
    plt.title("Distribution of Organ Ratios")

    # Plot 4: Tumor ratio distribution
    plt.subplot(2, 3, 4)
    tumor_ratios = [t["tumor_ratio"] for t in tile_stats]
    plt.hist(tumor_ratios, bins=20, alpha=0.7, color="red", edgecolor="black")
    plt.xlabel("Tumor Ratio per Tile")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tumor Ratios")

    # Plot 5: Patient vs Control comparison
    plt.subplot(2, 3, 5)
    if patient_mask.sum() > 0 and control_mask.sum() > 0:
        patient_organ_ratios = tile_stats[patient_mask, 0]
        control_organ_ratios = tile_stats[control_mask, 0]

        plt.hist(
            control_organ_ratios,
            bins=15,
            alpha=0.7,
            label="Control",
            color="blue",
            edgecolor="black",
        )
        plt.hist(
            patient_organ_ratios,
            bins=15,
            alpha=0.7,
            label="Patient",
            color="orange",
            edgecolor="black",
        )
        plt.xlabel("Organ Ratio")
        plt.ylabel("Frequency")
        plt.title("Organ Ratios: Patient vs Control")
        plt.legend()

    # Plot 6: Recommended weights
    plt.subplot(2, 3, 6)
    methods = ["Current\n(Equal)", "Inv Freq\n(Balanced)", "Focal\n(Tumor Focus)"]
    tumor_weights = [1.0, tumor_normalized, focal_tumor_weight]
    organ_weights = [1.0, organ_normalized, focal_organ_weight]

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(
        x - width / 2,
        tumor_weights,
        width,
        label="Tumor Weight",
        color="red",
        alpha=0.7,
    )
    plt.bar(
        x + width / 2,
        organ_weights,
        width,
        label="Organ Weight",
        color="green",
        alpha=0.7,
    )

    plt.xlabel("Weighting Method")
    plt.ylabel("Weight")
    plt.title("Recommended Loss Weights")
    plt.xticks(x, methods)
    plt.legend()

    plt.tight_layout()
    plt.savefig("organ_tumor_ratio_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n💾 Analysis saved to: organ_tumor_ratio_analysis.png")

    return {
        "organ_tumor_ratio": organ_tumor_ratio,
        "tumor_frequency": tumor_frequency,
        "organ_frequency": organ_frequency,
        "inv_freq_tumor_weight": tumor_normalized,
        "inv_freq_organ_weight": organ_normalized,
        "focal_tumor_weight": focal_tumor_weight,
        "focal_organ_weight": focal_organ_weight,
        "stats": stats,
    }


if __name__ == "__main__":
    results = analyze_organ_tumor_ratio()

    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"1. Use weighted BCE loss instead of BCE + Dice")
    print(
        f"2. Set pos_weight = {results['inv_freq_tumor_weight']:.4f} for tumor emphasis"
    )
    print(f"3. Model outputs organ probabilities, invert for tumor probabilities")
    print(f"4. This creates probabilistic output suitable for downstream models")
