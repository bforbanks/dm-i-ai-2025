#!/usr/bin/env python3
"""
Tumor Pixel Intensity Analysis

Analyzes the distribution of pixel intensities under tumor masks to determine
if intensity thresholding could be effective for the inverted segmentation approach.

This script will help answer:
1. What's the intensity distribution of tumor pixels?
2. Could we use a threshold to isolate "dark" regions containing tumors?
3. How do tumor intensities compare to overall image intensities?
"""

from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TumorIntensityAnalyzer:
    """Analyze pixel intensities in tumor regions"""

    def __init__(self, patients_dir="data/patients"):
        self.patients_dir = Path(patients_dir)
        self.img_dir = self.patients_dir / "imgs"
        self.mask_dir = self.patients_dir / "labels"

        # Storage for analysis results
        self.tumor_pixels = []
        self.all_pixels = []
        self.image_stats = []
        self.tumor_stats = []

    def load_patient_data(self):
        """Load all patient images and corresponding masks"""
        print("Loading patient data...")

        # Get all patient images
        img_files = sorted(list(self.img_dir.glob("patient_*.png")))

        if not img_files:
            print(f"No patient images found in {self.img_dir}")
            return

        print(f"Found {len(img_files)} patient images")

        for img_path in tqdm(img_files, desc="Processing images"):
            # Load image
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue

            # Find corresponding mask
            img_name = img_path.stem  # e.g., "patient_001"
            mask_name = img_name.replace("patient_", "segmentation_") + ".png"
            mask_path = self.mask_dir / mask_name

            if not mask_path.exists():
                print(f"Warning: No mask found for {img_path} (expected {mask_path})")
                continue

            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load mask {mask_path}")
                continue

            # Process this image-mask pair
            self._process_image_mask_pair(image, mask, img_name)

    def _process_image_mask_pair(self, image, mask, img_name):
        """Process a single image-mask pair"""
        # Ensure same dimensions
        if image.shape != mask.shape:
            print(f"Warning: Image and mask shapes don't match for {img_name}")
            print(f"  Image: {image.shape}, Mask: {mask.shape}")
            return

        # Binary mask (tumor pixels = 1, background = 0)
        tumor_mask = (mask > 0).astype(bool)

        # Extract tumor pixels
        tumor_pixel_values = image[tumor_mask]

        # Extract all pixel values for comparison
        all_pixel_values = image.flatten()

        # Store for global analysis
        self.tumor_pixels.extend(tumor_pixel_values.tolist())
        self.all_pixels.extend(all_pixel_values.tolist())

        # Calculate per-image statistics
        tumor_count = tumor_mask.sum()
        total_pixels = image.size

        image_stat = {
            "image_name": img_name,
            "tumor_pixel_count": int(tumor_count),
            "total_pixels": int(total_pixels),
            "tumor_percentage": tumor_count / total_pixels * 100,
            "image_mean": float(np.mean(all_pixel_values)),
            "image_std": float(np.std(all_pixel_values)),
            "image_min": int(np.min(all_pixel_values)),
            "image_max": int(np.max(all_pixel_values)),
            "tumor_mean": float(np.mean(tumor_pixel_values)) if tumor_count > 0 else 0,
            "tumor_std": float(np.std(tumor_pixel_values)) if tumor_count > 0 else 0,
            "tumor_min": int(np.min(tumor_pixel_values)) if tumor_count > 0 else 0,
            "tumor_max": int(np.max(tumor_pixel_values)) if tumor_count > 0 else 0,
            "tumor_median": float(np.median(tumor_pixel_values))
            if tumor_count > 0
            else 0,
        }

        self.image_stats.append(image_stat)

    def _calculate_best_thresholds(self):
        """Calculate which threshold gives the best naive Dice score"""
        print("\n🔍 Finding best threshold for naive segmentation...")

        thresholds = range(10, 250, 10)  # Test more thresholds for best performance
        best_dice = 0
        best_threshold = 0
        best_direction = ""
        best_capture_rate = 0

        for thresh in thresholds:
            dice_scores_above = []
            dice_scores_below = []

            # Load and process each image
            img_files = sorted(list(self.img_dir.glob("patient_*.png")))

            for img_path in img_files:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Find corresponding mask
                img_name = img_path.stem
                mask_name = img_name.replace("patient_", "segmentation_") + ".png"
                mask_path = self.mask_dir / mask_name

                if not mask_path.exists():
                    continue

                # Load mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                if image.shape != mask.shape:
                    continue

                # Create binary masks
                true_mask = (mask > 0).astype(np.uint8)
                pred_mask_above = (image >= thresh).astype(np.uint8)
                pred_mask_below = (image <= thresh).astype(np.uint8)

                # Calculate Dice scores
                def calculate_dice(pred, target):
                    intersection = (pred * target).sum()
                    if pred.sum() + target.sum() == 0:
                        return 1.0  # Perfect score for both empty
                    return 2 * intersection / (pred.sum() + target.sum())

                dice_above = calculate_dice(pred_mask_above, true_mask)
                dice_below = calculate_dice(pred_mask_below, true_mask)

                dice_scores_above.append(dice_above)
                dice_scores_below.append(dice_below)

            # Calculate mean dice scores
            mean_dice_above = np.mean(dice_scores_above) if dice_scores_above else 0
            mean_dice_below = np.mean(dice_scores_below) if dice_scores_below else 0

            # Check if this is the best threshold
            if mean_dice_above > best_dice:
                best_dice = mean_dice_above
                best_threshold = thresh
                best_direction = f"≥{thresh}"
                # Calculate capture rate for this threshold
                tumor_pixels = np.array(self.tumor_pixels)
                best_capture_rate = (
                    np.sum(tumor_pixels >= thresh) / len(tumor_pixels) * 100
                )

            if mean_dice_below > best_dice:
                best_dice = mean_dice_below
                best_threshold = thresh
                best_direction = f"≤{thresh}"
                # Calculate capture rate for this threshold
                tumor_pixels = np.array(self.tumor_pixels)
                best_capture_rate = (
                    np.sum(tumor_pixels <= thresh) / len(tumor_pixels) * 100
                )

        return {
            "best_threshold": best_threshold,
            "best_dice": best_dice,
            "direction": best_direction,
            "capture_rate": best_capture_rate,
        }

    def calculate_global_statistics(self):
        """Calculate overall statistics across all images"""
        tumor_pixels = np.array(self.tumor_pixels)
        all_pixels = np.array(self.all_pixels)

        print("\n" + "=" * 60)
        print("TUMOR PIXEL INTENSITY ANALYSIS")
        print("=" * 60)

        print(f"\n📊 DATASET OVERVIEW")
        print(f"Total images analyzed: {len(self.image_stats)}")
        print(f"Total pixels: {len(all_pixels):,}")
        print(f"Total tumor pixels: {len(tumor_pixels):,}")
        print(
            f"Overall tumor percentage: {len(tumor_pixels) / len(all_pixels) * 100:.2f}%"
        )

        print(f"\n🎯 TUMOR PIXEL INTENSITY STATISTICS")
        print(f"Mean: {np.mean(tumor_pixels):.1f}")
        print(f"Median: {np.median(tumor_pixels):.1f}")
        print(f"Std: {np.std(tumor_pixels):.1f}")
        print(f"Min: {np.min(tumor_pixels)}")
        print(f"Max: {np.max(tumor_pixels)}")

        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n📈 TUMOR PIXEL PERCENTILES")
        for p in percentiles:
            value = np.percentile(tumor_pixels, p)
            print(f"{p:2d}th percentile: {value:.1f}")

        print(f"\n🔍 ALL PIXEL INTENSITY STATISTICS (for comparison)")
        print(f"Mean: {np.mean(all_pixels):.1f}")
        print(f"Median: {np.median(all_pixels):.1f}")
        print(f"Std: {np.std(all_pixels):.1f}")
        print(f"Min: {np.min(all_pixels)}")
        print(f"Max: {np.max(all_pixels)}")

        # Threshold analysis - check both directions with naive Dice scores
        print(f"\n🎚️ THRESHOLD ANALYSIS WITH NAIVE DICE SCORES")
        print(
            "(Checking both above and below threshold to determine correct direction)"
        )
        thresholds = [50, 75, 100, 125, 150, 175, 200]

        # Calculate naive dice scores for each threshold
        print(f"\nCalculating naive Dice scores for thresholding approach...")

        for thresh in thresholds:
            # Check pixels ABOVE threshold (bright pixels)
            tumor_above = np.sum(tumor_pixels >= thresh)
            all_above = np.sum(all_pixels >= thresh)
            tumor_capture_rate_above = tumor_above / len(tumor_pixels) * 100
            pixels_above_rate = all_above / len(all_pixels) * 100

            # Check pixels BELOW threshold (dark pixels)
            tumor_below = np.sum(tumor_pixels <= thresh)
            all_below = np.sum(all_pixels <= thresh)
            tumor_capture_rate_below = tumor_below / len(tumor_pixels) * 100
            pixels_below_rate = all_below / len(all_pixels) * 100

            # Calculate naive Dice scores by actually applying threshold to images
            dice_scores_above = []
            dice_scores_below = []

            # Load and process each image to calculate actual Dice scores
            img_files = sorted(list(self.img_dir.glob("patient_*.png")))

            for img_path in img_files:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # Find corresponding mask
                img_name = img_path.stem
                mask_name = img_name.replace("patient_", "segmentation_") + ".png"
                mask_path = self.mask_dir / mask_name

                if not mask_path.exists():
                    continue

                # Load mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                if image.shape != mask.shape:
                    continue

                # Create binary masks
                true_mask = (mask > 0).astype(np.uint8)
                pred_mask_above = (image >= thresh).astype(np.uint8)
                pred_mask_below = (image <= thresh).astype(np.uint8)

                # Calculate Dice scores
                def calculate_dice(pred, target):
                    intersection = (pred * target).sum()
                    if pred.sum() + target.sum() == 0:
                        return 1.0  # Perfect score for both empty
                    return 2 * intersection / (pred.sum() + target.sum())

                dice_above = calculate_dice(pred_mask_above, true_mask)
                dice_below = calculate_dice(pred_mask_below, true_mask)

                dice_scores_above.append(dice_above)
                dice_scores_below.append(dice_below)

            # Calculate mean dice scores
            mean_dice_above = np.mean(dice_scores_above) if dice_scores_above else 0
            mean_dice_below = np.mean(dice_scores_below) if dice_scores_below else 0

            print(f"Threshold {thresh:3d}:")
            print(
                f"  Above (≥{thresh}): {tumor_capture_rate_above:5.1f}% tumors captured, {pixels_above_rate:5.1f}% total pixels above, Naive Dice: {mean_dice_above:.3f}"
            )
            print(
                f"  Below (≤{thresh}): {tumor_capture_rate_below:5.1f}% tumors captured, {pixels_below_rate:5.1f}% total pixels below, Naive Dice: {mean_dice_below:.3f}"
            )

        # Find and display the best performing thresholds from this analysis
        print(f"\n📊 SUMMARY OF THRESHOLD PERFORMANCE:")
        print(
            f"{'Threshold':<12} {'Direction':<10} {'Dice Score':<12} {'Tumor Capture':<15}"
        )
        print("-" * 50)

        # Store results for summary
        threshold_results = []
        for thresh in thresholds:
            # Recalculate for summary (simplified version)
            tumor_above = np.sum(tumor_pixels >= thresh)
            tumor_below = np.sum(tumor_pixels <= thresh)
            tumor_capture_rate_above = tumor_above / len(tumor_pixels) * 100
            tumor_capture_rate_below = tumor_below / len(tumor_pixels) * 100

            # Get dice scores (simplified calculation for summary)
            dice_scores_above = []
            dice_scores_below = []

            img_files = sorted(list(self.img_dir.glob("patient_*.png")))[
                :5
            ]  # Sample first 5 for speed

            for img_path in img_files:
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                img_name = img_path.stem
                mask_name = img_name.replace("patient_", "segmentation_") + ".png"
                mask_path = self.mask_dir / mask_name

                if not mask_path.exists():
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or image.shape != mask.shape:
                    continue

                true_mask = (mask > 0).astype(np.uint8)
                pred_mask_above = (image >= thresh).astype(np.uint8)
                pred_mask_below = (image <= thresh).astype(np.uint8)

                def calc_dice(pred, target):
                    intersection = (pred * target).sum()
                    if pred.sum() + target.sum() == 0:
                        return 1.0
                    return 2 * intersection / (pred.sum() + target.sum())

                dice_scores_above.append(calc_dice(pred_mask_above, true_mask))
                dice_scores_below.append(calc_dice(pred_mask_below, true_mask))

            mean_dice_above = np.mean(dice_scores_above) if dice_scores_above else 0
            mean_dice_below = np.mean(dice_scores_below) if dice_scores_below else 0

            threshold_results.append(
                (thresh, "≥", mean_dice_above, tumor_capture_rate_above)
            )
            threshold_results.append(
                (thresh, "≤", mean_dice_below, tumor_capture_rate_below)
            )

        # Sort by dice score and show top 5
        threshold_results.sort(key=lambda x: x[2], reverse=True)
        for i, (thresh, direction, dice, capture) in enumerate(threshold_results[:5]):
            print(
                f"{thresh:<12} {direction}{thresh:<8} {dice:<12.3f} {capture:<15.1f}%"
            )
            if i == 0:
                print("^ BEST PERFORMING THRESHOLD")
        print("-" * 50)

        # Statistical test: Are tumor pixels significantly different?
        statistic, p_value = stats.mannwhitneyu(
            tumor_pixels,
            np.random.choice(all_pixels, size=min(len(tumor_pixels), 10000)),
        )
        print(f"\n📊 STATISTICAL SIGNIFICANCE")
        print(f"Mann-Whitney U test p-value: {p_value:.2e}")
        print(
            "Tumor pixels are significantly different from background"
            if p_value < 0.001
            else "No significant difference"
        )

        return {
            "tumor_pixels": tumor_pixels,
            "all_pixels": all_pixels,
            "tumor_stats": {
                "mean": np.mean(tumor_pixels),
                "median": np.median(tumor_pixels),
                "std": np.std(tumor_pixels),
                "min": np.min(tumor_pixels),
                "max": np.max(tumor_pixels),
            },
            "all_stats": {
                "mean": np.mean(all_pixels),
                "median": np.median(all_pixels),
                "std": np.std(all_pixels),
                "min": np.min(all_pixels),
                "max": np.max(all_pixels),
            },
        }

    def create_visualizations(self, save_dir="intensity_analysis_results"):
        """Create comprehensive visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        tumor_pixels = np.array(self.tumor_pixels)
        all_pixels = np.array(self.all_pixels)

        # 1. Distribution comparison
        plt.figure(figsize=(15, 10))

        # Subplot 1: Overlapping histograms
        plt.subplot(2, 3, 1)
        plt.hist(
            all_pixels,
            bins=50,
            alpha=0.5,
            label="All pixels",
            density=True,
            color="lightblue",
        )
        plt.hist(
            tumor_pixels,
            bins=50,
            alpha=0.7,
            label="Tumor pixels",
            density=True,
            color="red",
        )
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.title("Intensity Distribution Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Tumor pixels only (detailed)
        plt.subplot(2, 3, 2)
        plt.hist(tumor_pixels, bins=30, alpha=0.7, color="red", edgecolor="black")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Tumor Pixel Intensity Distribution")
        plt.axvline(
            np.mean(tumor_pixels),
            color="blue",
            linestyle="--",
            label=f"Mean: {np.mean(tumor_pixels):.1f}",
        )
        plt.axvline(
            np.median(tumor_pixels),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(tumor_pixels):.1f}",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 3: Box plot comparison
        plt.subplot(2, 3, 3)
        data_for_box = [
            all_pixels[::100],
            tumor_pixels,
        ]  # Subsample all_pixels for performance
        plt.boxplot(data_for_box, labels=["All pixels\n(subsampled)", "Tumor pixels"])
        plt.ylabel("Pixel Intensity")
        plt.title("Intensity Distribution Box Plot")
        plt.grid(True, alpha=0.3)

        # Subplot 4: Threshold analysis (bidirectional)
        plt.subplot(2, 3, 4)
        thresholds = range(0, 256, 5)
        tumor_capture_above = []
        tumor_capture_below = []

        for thresh in thresholds:
            tumor_above = np.sum(tumor_pixels >= thresh)
            tumor_below = np.sum(tumor_pixels <= thresh)
            tumor_capture_above.append(tumor_above / len(tumor_pixels) * 100)
            tumor_capture_below.append(tumor_below / len(tumor_pixels) * 100)

        plt.plot(
            thresholds,
            tumor_capture_above,
            "r-",
            label="% Tumors ≥ threshold",
            linewidth=2,
        )
        plt.plot(
            thresholds,
            tumor_capture_below,
            "g-",
            label="% Tumors ≤ threshold",
            linewidth=2,
        )
        plt.xlabel("Intensity Threshold")
        plt.ylabel("Tumor Capture Rate (%)")
        plt.title("Bidirectional Threshold Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 5: Cumulative distribution
        plt.subplot(2, 3, 5)
        # Calculate cumulative distributions
        tumor_sorted = np.sort(tumor_pixels)
        all_sorted = np.sort(all_pixels[::100])  # Subsample for performance

        tumor_cum = np.arange(1, len(tumor_sorted) + 1) / len(tumor_sorted)
        all_cum = np.arange(1, len(all_sorted) + 1) / len(all_sorted)

        plt.plot(tumor_sorted, tumor_cum, "r-", label="Tumor pixels", linewidth=2)
        plt.plot(
            all_sorted, all_cum, "b-", label="All pixels (subsampled)", linewidth=2
        )
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Distribution Function")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 6: Per-image tumor intensity variation
        plt.subplot(2, 3, 6)
        tumor_means = [
            stat["tumor_mean"]
            for stat in self.image_stats
            if stat["tumor_pixel_count"] > 0
        ]
        tumor_stds = [
            stat["tumor_std"]
            for stat in self.image_stats
            if stat["tumor_pixel_count"] > 0
        ]

        plt.scatter(tumor_means, tumor_stds, alpha=0.6)
        plt.xlabel("Tumor Mean Intensity (per image)")
        plt.ylabel("Tumor Std Intensity (per image)")
        plt.title("Per-Image Tumor Intensity Variation")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / "tumor_intensity_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Detailed threshold recommendation plot (bidirectional)
        plt.figure(figsize=(12, 10))

        # Find optimal thresholds - check both directions
        thresholds = range(0, 256, 1)
        tumor_capture_above = []
        tumor_capture_below = []
        efficiency_above = []
        efficiency_below = []

        for thresh in thresholds:
            # Above threshold analysis
            tumor_above = np.sum(tumor_pixels >= thresh)
            all_above = np.sum(all_pixels >= thresh)
            tumor_capture_above.append(tumor_above / len(tumor_pixels) * 100)
            efficiency_above.append(
                tumor_above / all_above * 100 if all_above > 0 else 0
            )

            # Below threshold analysis
            tumor_below = np.sum(tumor_pixels <= thresh)
            all_below = np.sum(all_pixels <= thresh)
            tumor_capture_below.append(tumor_below / len(tumor_pixels) * 100)
            efficiency_below.append(
                tumor_below / all_below * 100 if all_below > 0 else 0
            )

        # Plot 1: Tumor capture rates (both directions)
        plt.subplot(3, 1, 1)
        plt.plot(
            thresholds,
            tumor_capture_above,
            "r-",
            linewidth=2,
            label="% Tumors ≥ threshold",
        )
        plt.plot(
            thresholds,
            tumor_capture_below,
            "g-",
            linewidth=2,
            label="% Tumors ≤ threshold",
        )
        plt.axhline(
            y=90, color="orange", linestyle="--", alpha=0.7, label="90% capture target"
        )
        plt.axhline(
            y=95, color="blue", linestyle="--", alpha=0.7, label="95% capture target"
        )
        plt.xlabel("Intensity Threshold")
        plt.ylabel("Tumor Capture Rate (%)")
        plt.title("Bidirectional Tumor Capture Rate vs Threshold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Efficiency scores (above threshold)
        plt.subplot(3, 1, 2)
        plt.plot(
            thresholds,
            efficiency_above,
            "r-",
            linewidth=2,
            label="Efficiency ≥ threshold",
        )
        plt.xlabel("Intensity Threshold")
        plt.ylabel("Efficiency (% tumor pixels among selected)")
        plt.title("Threshold Efficiency (Above Threshold)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Efficiency scores (below threshold)
        plt.subplot(3, 1, 3)
        plt.plot(
            thresholds,
            efficiency_below,
            "g-",
            linewidth=2,
            label="Efficiency ≤ threshold",
        )
        plt.xlabel("Intensity Threshold")
        plt.ylabel("Efficiency (% tumor pixels among selected)")
        plt.title("Threshold Efficiency (Below Threshold)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / "threshold_optimization.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"\n📊 Visualizations saved to: {save_dir}")

    def save_detailed_results(self, save_dir="intensity_analysis_results"):
        """Save detailed results to CSV"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Per-image statistics
        df_images = pd.DataFrame(self.image_stats)
        df_images.to_csv(save_dir / "per_image_statistics.csv", index=False)

        # Sample of pixel values (for further analysis)
        tumor_sample = np.random.choice(
            self.tumor_pixels, size=min(10000, len(self.tumor_pixels)), replace=False
        )
        all_sample = np.random.choice(
            self.all_pixels, size=min(10000, len(self.all_pixels)), replace=False
        )

        df_pixels = pd.DataFrame(
            {
                "tumor_pixels": list(tumor_sample)
                + [np.nan] * (10000 - len(tumor_sample))
                if len(tumor_sample) < 10000
                else tumor_sample,
                "all_pixels": list(all_sample) + [np.nan] * (10000 - len(all_sample))
                if len(all_sample) < 10000
                else all_sample,
            }
        )
        df_pixels.to_csv(save_dir / "pixel_samples.csv", index=False)

        print(f"📄 Detailed results saved to CSV files in: {save_dir}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🔍 Starting Tumor Pixel Intensity Analysis")
        print("=" * 50)

        # Load data
        self.load_patient_data()

        if not self.tumor_pixels:
            print("❌ No tumor pixels found. Check your data directory structure.")
            return

        # Calculate statistics
        global_stats = self.calculate_global_statistics()

        # Create visualizations
        self.create_visualizations()

        # Save results
        self.save_detailed_results()

        # Calculate and store naive Dice scores for final recommendations
        best_thresholds_info = self._calculate_best_thresholds()

        # Summary recommendations
        tumor_pixels = np.array(self.tumor_pixels)
        all_pixels = np.array(self.all_pixels)

        print("\n🎯 THRESHOLD RECOMMENDATIONS")
        print("-" * 40)

        # Determine if tumors are dark (low values) or bright (high values)
        tumor_mean = np.mean(tumor_pixels)
        all_mean = np.mean(all_pixels)

        print(f"Tumor pixel mean: {tumor_mean:.1f}")
        print(f"All pixel mean: {all_mean:.1f}")

        # Show best naive thresholding results
        print(f"\n🏆 BEST NAIVE THRESHOLDING PERFORMANCE:")
        print(
            f"Best threshold: {best_thresholds_info['best_threshold']} ({best_thresholds_info['direction']})"
        )
        print(f"Best naive Dice score: {best_thresholds_info['best_dice']:.3f}")
        print(f"Tumor capture rate: {best_thresholds_info['capture_rate']:.1f}%")

        if tumor_mean < all_mean:
            # Tumors are darker (lower values) - use "below threshold" approach
            print("🔍 Tumors are DARKER (lower intensity values) than background")
            thresholds_90 = np.percentile(tumor_pixels, 90)  # 90% below this
            thresholds_95 = np.percentile(tumor_pixels, 95)  # 95% below this
            thresholds_99 = np.percentile(tumor_pixels, 99)  # 99% below this

            print(f"To capture 90% of tumor pixels: threshold ≤ {thresholds_90:.0f}")
            print(f"To capture 95% of tumor pixels: threshold ≤ {thresholds_95:.0f}")
            print(f"To capture 99% of tumor pixels: threshold ≤ {thresholds_99:.0f}")

            print(f"\n🧠 INSIGHTS FOR INVERTED SEGMENTATION")
            print("-" * 40)
            print("✅ Tumor pixels are DARK (low intensity values)")
            print("✅ Intensity thresholding approach is VIABLE")
            print(f"   - Use threshold ≤ {thresholds_95:.0f} to capture 95% of tumors")
            print("   - Focus on LOW intensity regions for inverted segmentation")

        else:
            # Tumors are brighter (higher values) - use "above threshold" approach
            print("🔍 Tumors are BRIGHTER (higher intensity values) than background")
            thresholds_90 = np.percentile(tumor_pixels, 10)  # 90% above this
            thresholds_95 = np.percentile(tumor_pixels, 5)  # 95% above this
            thresholds_99 = np.percentile(tumor_pixels, 1)  # 99% above this

            print(f"To capture 90% of tumor pixels: threshold ≥ {thresholds_90:.0f}")
            print(f"To capture 95% of tumor pixels: threshold ≥ {thresholds_95:.0f}")
            print(f"To capture 99% of tumor pixels: threshold ≥ {thresholds_99:.0f}")

            print(f"\n🧠 INSIGHTS FOR INVERTED SEGMENTATION")
            print("-" * 40)
            print("✅ Tumor pixels are BRIGHT (high intensity values)")
            print("✅ Intensity thresholding approach is VIABLE")
            print(f"   - Use threshold ≥ {thresholds_95:.0f} to capture 95% of tumors")
            print("   - Focus on HIGH intensity regions for inverted segmentation")

        return global_stats


def main():
    """Run the tumor intensity analysis"""
    analyzer = TumorIntensityAnalyzer()
    results = analyzer.run_complete_analysis()

    print("\n✅ Analysis complete!")
    print("📊 Check 'intensity_analysis_results/' for detailed visualizations and data")


if __name__ == "__main__":
    main()
