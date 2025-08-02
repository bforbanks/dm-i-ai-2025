#!/usr/bin/env python3
"""
Script to analyze the distribution, dimensions, and tumor locations in the dataset.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

import seaborn as sns

# Configuration
TALL_IMAGE_ASPECT_RATIO_THRESHOLD = (
    0.5  # Images with aspect ratio > this are considered "tall"
)


def load_image_dimensions(image_paths):
    """Load dimensions of images without loading full images"""
    dimensions = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            dimensions.append(
                {
                    "path": str(img_path),
                    "height": h,
                    "width": w,
                    "aspect_ratio": h / w,
                    "total_pixels": h * w,
                }
            )
    return dimensions


def analyze_tumor_locations(patient_images, patient_masks):
    """Analyze where tumors are located within images"""
    tumor_locations = []

    for img_path, mask_path in zip(patient_images, patient_masks):
        if mask_path is None:
            continue

        # Load image and mask
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        h, w = img.shape
        mask_binary = (mask > 0).astype(np.uint8)

        # Find tumor regions
        tumor_pixels = np.where(mask_binary > 0)

        if len(tumor_pixels[0]) > 0:
            # Calculate tumor bounding box and center
            min_y, max_y = tumor_pixels[0].min(), tumor_pixels[0].max()
            min_x, max_x = tumor_pixels[1].min(), tumor_pixels[1].max()

            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2

            # Calculate relative positions (0 = top/left, 1 = bottom/right)
            rel_center_y = center_y / h
            rel_center_x = center_x / w
            rel_min_y = min_y / h
            rel_max_y = max_y / h

            tumor_area = np.sum(mask_binary)
            tumor_coverage = tumor_area / (h * w)

            tumor_locations.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "image_height": h,
                    "image_width": w,
                    "aspect_ratio": h / w,
                    "tumor_center_y_rel": rel_center_y,
                    "tumor_center_x_rel": rel_center_x,
                    "tumor_min_y_rel": rel_min_y,
                    "tumor_max_y_rel": rel_max_y,
                    "tumor_area": tumor_area,
                    "tumor_coverage": tumor_coverage,
                    "tumor_height_span": max_y - min_y,
                    "tumor_width_span": max_x - min_x,
                }
            )

    return tumor_locations


def load_data_paths(data_dir="data"):
    """Load paths to images and masks"""
    data_path = Path(data_dir)

    # Patient images and masks
    patient_img_dir = data_path / "patients" / "imgs"
    patient_mask_dir = data_path / "patients" / "labels"

    patient_images = sorted(list(patient_img_dir.glob("*.png")))
    patient_masks = []

    for img_path in patient_images:
        # Find corresponding mask
        img_name = img_path.stem
        mask_name = img_name.replace("patient_", "segmentation_") + ".png"
        mask_path = patient_mask_dir / mask_name

        if mask_path.exists():
            patient_masks.append(str(mask_path))
        else:
            patient_masks.append(None)

    # Control images (no tumors)
    control_img_dir = data_path / "controls" / "imgs"
    control_images = sorted(list(control_img_dir.glob("*.png")))

    # Filter out patients with no masks
    valid_patients = [
        (str(p), m) for p, m in zip(patient_images, patient_masks) if m is not None
    ]
    patient_images, patient_masks = zip(*valid_patients) if valid_patients else ([], [])

    return list(patient_images), list(patient_masks), [str(p) for p in control_images]


def create_comprehensive_analysis():
    """Create comprehensive analysis of the dataset"""
    print("🔍 Loading dataset...")
    patient_images, patient_masks, control_images = load_data_paths()

    print(f"Found {len(patient_images)} patient images with masks")
    print(f"Found {len(control_images)} control images")

    # Analyze dimensions
    print("\n📐 Analyzing image dimensions...")
    patient_dims = load_image_dimensions(patient_images)
    control_dims = load_image_dimensions(control_images)

    # Convert to DataFrames for easier analysis
    patient_df = pd.DataFrame(patient_dims)
    control_df = pd.DataFrame(control_dims)

    patient_df["type"] = "Patient"
    control_df["type"] = "Control"
    combined_df = pd.concat([patient_df, control_df], ignore_index=True)

    # Print dimension statistics
    print("\n📊 DIMENSION STATISTICS")
    print("=" * 50)

    for img_type in ["Patient", "Control"]:
        subset = combined_df[combined_df["type"] == img_type]
        print(f"\n{img_type} Images ({len(subset)} total):")
        print(
            f"  Height: {subset['height'].min()}-{subset['height'].max()} (mean: {subset['height'].mean():.1f})"
        )
        print(
            f"  Width:  {subset['width'].min()}-{subset['width'].max()} (mean: {subset['width'].mean():.1f})"
        )
        print(
            f"  Aspect Ratio: {subset['aspect_ratio'].min():.2f}-{subset['aspect_ratio'].max():.2f} (mean: {subset['aspect_ratio'].mean():.2f})"
        )
        print(
            f"  Total Pixels: {subset['total_pixels'].min()}-{subset['total_pixels'].max()}"
        )

    # Analyze tumor locations
    print("\n🎯 Analyzing tumor locations...")
    tumor_data = analyze_tumor_locations(patient_images, patient_masks)
    tumor_df = pd.DataFrame(tumor_data)

    if len(tumor_df) > 0:
        print("\n📍 TUMOR LOCATION STATISTICS")
        print("=" * 50)
        print(f"Images with tumors: {len(tumor_df)}")
        print("\nTumor Center Positions (relative to image):")
        print(
            f"  Y-center (0=top, 1=bottom): {tumor_df['tumor_center_y_rel'].min():.3f}-{tumor_df['tumor_center_y_rel'].max():.3f} (mean: {tumor_df['tumor_center_y_rel'].mean():.3f})"
        )
        print(
            f"  X-center (0=left, 1=right): {tumor_df['tumor_center_x_rel'].min():.3f}-{tumor_df['tumor_center_x_rel'].max():.3f} (mean: {tumor_df['tumor_center_x_rel'].mean():.3f})"
        )

        print("\nTumor Coverage:")
        print(
            f"  Coverage: {tumor_df['tumor_coverage'].min():.4f}-{tumor_df['tumor_coverage'].max():.4f} (mean: {tumor_df['tumor_coverage'].mean():.4f})"
        )
        print(
            f"  Area (pixels): {tumor_df['tumor_area'].min()}-{tumor_df['tumor_area'].max()} (mean: {tumor_df['tumor_area'].mean():.1f})"
        )

        # Check hypothesis about tall images having tumors in upper part
        tall_images = tumor_df[
            tumor_df["aspect_ratio"] > TALL_IMAGE_ASPECT_RATIO_THRESHOLD
        ]  # Images taller than threshold
        if len(tall_images) > 0:
            print(
                f"\n🏗️ TALL IMAGES ANALYSIS (aspect ratio > {TALL_IMAGE_ASPECT_RATIO_THRESHOLD}):"
            )
            print(f"  Number of tall images: {len(tall_images)}")
            print(
                f"  Average tumor Y-center in tall images: {tall_images['tumor_center_y_rel'].mean():.3f}"
            )
            print(
                f"  Percentage of tumors in upper half: {(tall_images['tumor_center_y_rel'] < 0.5).mean() * 100:.1f}%"
            )

    # Create visualizations
    create_visualizations(combined_df, tumor_df)

    return combined_df, tumor_df


def create_visualizations(combined_df, tumor_df):
    """Create visualization plots"""
    plt.style.use("default")

    # Create figure with multiple subplots
    _fig = plt.figure(figsize=(20, 15))

    # 1. Dimension comparison
    plt.subplot(3, 3, 1)
    sns.boxplot(data=combined_df, x="type", y="height")
    plt.title("Image Heights: Patient vs Control")
    plt.ylabel("Height (pixels)")

    plt.subplot(3, 3, 2)
    sns.boxplot(data=combined_df, x="type", y="width")
    plt.title("Image Widths: Patient vs Control")
    plt.ylabel("Width (pixels)")

    plt.subplot(3, 3, 3)
    sns.boxplot(data=combined_df, x="type", y="aspect_ratio")
    plt.title("Aspect Ratios: Patient vs Control")
    plt.ylabel("Aspect Ratio (H/W)")

    # 2. Scatter plot of dimensions
    plt.subplot(3, 3, 4)
    for img_type in ["Patient", "Control"]:
        subset = combined_df[combined_df["type"] == img_type]
        plt.scatter(subset["width"], subset["height"], alpha=0.6, label=img_type)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Image Dimensions Distribution")
    plt.legend()

    if len(tumor_df) > 0:
        # 3. Tumor location heatmap
        plt.subplot(3, 3, 5)
        plt.scatter(
            tumor_df["tumor_center_x_rel"],
            tumor_df["tumor_center_y_rel"],
            c=tumor_df["tumor_coverage"],
            cmap="viridis",
            alpha=0.7,
        )
        plt.xlabel("X Position (0=left, 1=right)")
        plt.ylabel("Y Position (0=top, 1=bottom)")
        plt.title("Tumor Center Positions")
        plt.colorbar(label="Tumor Coverage")
        plt.gca().invert_yaxis()  # Invert Y axis so 0,0 is top-left

        # 4. Tumor coverage vs image size
        plt.subplot(3, 3, 6)
        plt.scatter(
            tumor_df["aspect_ratio"],
            tumor_df["tumor_center_y_rel"],
            c=tumor_df["tumor_coverage"],
            cmap="plasma",
            alpha=0.7,
        )
        plt.xlabel("Aspect Ratio (H/W)")
        plt.ylabel("Tumor Y-Center (0=top, 1=bottom)")
        plt.title("Tumor Position vs Image Aspect Ratio")
        plt.colorbar(label="Tumor Coverage")

        # 5. Tumor size distribution
        plt.subplot(3, 3, 7)
        plt.hist(tumor_df["tumor_coverage"], bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Tumor Coverage (fraction of image)")
        plt.ylabel("Number of Images")
        plt.title("Tumor Coverage Distribution")

        # 6. Y-position distribution for different aspect ratios
        plt.subplot(3, 3, 8)
        tall_images = tumor_df[
            tumor_df["aspect_ratio"] > TALL_IMAGE_ASPECT_RATIO_THRESHOLD
        ]
        normal_images = tumor_df[
            tumor_df["aspect_ratio"] <= TALL_IMAGE_ASPECT_RATIO_THRESHOLD
        ]

        if len(tall_images) > 0:
            plt.hist(
                tall_images["tumor_center_y_rel"],
                bins=20,
                alpha=0.7,
                label=f"Tall images (>{TALL_IMAGE_ASPECT_RATIO_THRESHOLD} ratio, n={len(tall_images)})",
                color="red",
            )
        if len(normal_images) > 0:
            plt.hist(
                normal_images["tumor_center_y_rel"],
                bins=20,
                alpha=0.7,
                label=f"Normal images (≤{TALL_IMAGE_ASPECT_RATIO_THRESHOLD} ratio, n={len(normal_images)})",
                color="blue",
            )

        plt.xlabel("Tumor Y-Center (0=top, 1=bottom)")
        plt.ylabel("Number of Images")
        plt.title("Tumor Y-Position: Tall vs Normal Images")
        plt.legend()
        plt.axvline(
            x=0.5, color="black", linestyle="--", alpha=0.5, label="Image center"
        )

        # 7. Aspect ratio distribution
        plt.subplot(3, 3, 9)
        plt.hist(
            combined_df[combined_df["type"] == "Patient"]["aspect_ratio"],
            bins=30,
            alpha=0.7,
            label="Patient",
            color="red",
        )
        plt.hist(
            combined_df[combined_df["type"] == "Control"]["aspect_ratio"],
            bins=30,
            alpha=0.7,
            label="Control",
            color="blue",
        )
        plt.xlabel("Aspect Ratio (H/W)")
        plt.ylabel("Number of Images")
        plt.title("Aspect Ratio Distribution")
        plt.legend()
        plt.axvline(
            x=TALL_IMAGE_ASPECT_RATIO_THRESHOLD,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Tall threshold",
        )

    plt.tight_layout()
    plt.savefig("data_analysis_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n📈 Visualization saved as: data_analysis_comprehensive.png")


def main():
    """Main analysis function"""
    print("🔬 MEDICAL IMAGE DATASET ANALYSIS")
    print("=" * 60)

    combined_df, tumor_df = create_comprehensive_analysis()

    # Additional specific analysis for your hypothesis
    if len(tumor_df) > 0:
        print("\n🔍 HYPOTHESIS TESTING: Tumors in upper part of tall images")
        print("=" * 60)

        # Define "tall" images (you can adjust this threshold at the top of the script)
        tall_images = tumor_df[
            tumor_df["aspect_ratio"] > TALL_IMAGE_ASPECT_RATIO_THRESHOLD
        ]

        if len(tall_images) > 0:
            upper_half_tumors = tall_images[tall_images["tumor_center_y_rel"] < 0.5]
            upper_third_tumors = tall_images[tall_images["tumor_center_y_rel"] < 0.33]

            print(
                f"Tall images (aspect ratio > {TALL_IMAGE_ASPECT_RATIO_THRESHOLD}): {len(tall_images)}"
            )
            print(
                f"Tumors in upper half (y < 0.5): {len(upper_half_tumors)} ({len(upper_half_tumors) / len(tall_images) * 100:.1f}%)"
            )
            print(
                f"Tumors in upper third (y < 0.33): {len(upper_third_tumors)} ({len(upper_third_tumors) / len(tall_images) * 100:.1f}%)"
            )

            print(
                f"\nAverage tumor Y-position in tall images: {tall_images['tumor_center_y_rel'].mean():.3f}"
            )
            print(
                f"Average tumor Y-position in normal images: {tumor_df[tumor_df['aspect_ratio'] <= TALL_IMAGE_ASPECT_RATIO_THRESHOLD]['tumor_center_y_rel'].mean():.3f}"
            )
        else:
            print("No tall images found in dataset")

    print("\n✅ Analysis complete!")
    return combined_df, tumor_df


if __name__ == "__main__":
    main()
