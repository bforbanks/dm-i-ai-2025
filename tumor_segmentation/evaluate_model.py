#!/usr/bin/env python3
"""
Model Evaluation Script for Tumor Segmentation

This script evaluates a tumor segmentation model on:
1. Validation set (with ground truth) - computes standard metrics
2. Evaluation set (without ground truth) - analyzes prediction statistics

Usage:
    python evaluate_model.py --data_dir data/
    python evaluate_model.py --data_dir data/ --batch_size 8
"""

import os
from pathlib import Path
import sys

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from sklearn.metrics import precision_score, recall_score, f1_score
from dotenv import load_dotenv

# Import project modules
from data.data_default import TumorSegmentationDataModule
from models.NNUNetStyle.model import NNUNetStyle


class ModelEvaluator:
    """Comprehensive model evaluation for tumor segmentation"""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Storage for results
        self.val_results = []
        self.eval_results = []

    def evaluate_validation_set(self, val_dataloader):
        """Evaluate model on validation set with ground truth"""
        print("Evaluating on validation set...")

        all_dice_scores = []
        all_ious = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_dataloader)):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Convert to numpy for metric calculation
                pred_np = predictions.cpu().numpy()
                mask_np = masks.cpu().numpy()

                # Calculate metrics for each image in batch
                for i in range(pred_np.shape[0]):
                    pred_binary = (pred_np[i, 0] > 0.5).astype(np.uint8)
                    mask_binary = (mask_np[i, 0] > 0.5).astype(np.uint8)

                    # Calculate metrics
                    dice = self._calculate_dice(pred_binary, mask_binary)
                    iou = self._calculate_iou(pred_binary, mask_binary)

                    # Flatten for sklearn metrics
                    pred_flat = pred_binary.flatten()
                    mask_flat = mask_binary.flatten()

                    if len(np.unique(mask_flat)) > 1:  # Only if there are both classes
                        precision = precision_score(
                            mask_flat, pred_flat, zero_division=0
                        )
                        recall = recall_score(mask_flat, pred_flat, zero_division=0)
                        f1 = f1_score(mask_flat, pred_flat, zero_division=0)
                    else:
                        precision = recall = f1 = 0.0 if mask_flat.sum() == 0 else 1.0

                    all_dice_scores.append(dice)
                    all_ious.append(iou)
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1s.append(f1)

                    # Store detailed results
                    self.val_results.append(
                        {
                            "batch_idx": batch_idx,
                            "image_idx": i,
                            "dice_score": dice,
                            "iou": iou,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "has_tumor_gt": mask_binary.sum() > 0,
                            "has_tumor_pred": pred_binary.sum() > 0,
                            "tumor_pixels_gt": mask_binary.sum(),
                            "tumor_pixels_pred": pred_binary.sum(),
                        }
                    )

        # Calculate summary statistics
        val_summary = {
            "mean_dice": np.mean(all_dice_scores),
            "std_dice": np.std(all_dice_scores),
            "mean_iou": np.mean(all_ious),
            "std_iou": np.std(all_ious),
            "mean_precision": np.mean(all_precisions),
            "mean_recall": np.mean(all_recalls),
            "mean_f1": np.mean(all_f1s),
            "total_images": len(all_dice_scores),
        }

        return val_summary

    def evaluate_evaluation_set(self, eval_dataloader):
        """Evaluate model on evaluation set (no ground truth) - analyze predictions"""
        print("Evaluating on evaluation set...")

        image_resolutions = []
        tumor_detections = []
        cluster_counts = []
        tumor_pixel_counts = []

        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(tqdm(eval_dataloader)):
                images = images.to(self.device)

                # Forward pass
                predictions = self.model(images)

                # Convert to numpy
                pred_np = predictions.cpu().numpy()

                # Analyze each image in batch
                for i in range(pred_np.shape[0]):
                    pred_binary = (pred_np[i, 0] > 0.5).astype(np.uint8)
                    filename = filenames[i]

                    # Get original image resolution
                    # Note: pred_binary is resized to model input size, get original from file
                    original_img = cv2.imread(
                        str(
                            Path(
                                eval_dataloader.dataset.image_paths[
                                    batch_idx * eval_dataloader.batch_size + i
                                ]
                            )
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if original_img is not None:
                        h, w = original_img.shape
                        image_resolutions.append((w, h))
                    else:
                        image_resolutions.append((256, 256))  # Default model size

                    # Tumor detection analysis
                    has_tumor = pred_binary.sum() > 0
                    tumor_detections.append(has_tumor)
                    tumor_pixel_counts.append(pred_binary.sum())

                    # Cluster analysis (connected components)
                    if has_tumor:
                        labeled_array, num_clusters = ndimage.label(pred_binary)
                        cluster_counts.append(num_clusters)
                    else:
                        cluster_counts.append(0)

                    # Store detailed results
                    self.eval_results.append(
                        {
                            "filename": filename,
                            "batch_idx": batch_idx,
                            "image_idx": i,
                            "has_tumor_pred": has_tumor,
                            "tumor_pixels": pred_binary.sum(),
                            "num_clusters": cluster_counts[-1],
                            "resolution_w": image_resolutions[-1][0],
                            "resolution_h": image_resolutions[-1][1],
                            "resolution_area": image_resolutions[-1][0]
                            * image_resolutions[-1][1],
                        }
                    )

        # Calculate summary statistics
        eval_summary = {
            "total_images": len(tumor_detections),
            "images_with_tumors": sum(tumor_detections),
            "images_without_tumors": len(tumor_detections) - sum(tumor_detections),
            "tumor_detection_rate": sum(tumor_detections) / len(tumor_detections),
            "mean_clusters_per_image": np.mean(cluster_counts),
            "mean_clusters_when_tumor_detected": np.mean(
                [c for c in cluster_counts if c > 0]
            )
            if any(cluster_counts)
            else 0,
            "mean_tumor_pixels": np.mean(tumor_pixel_counts),
            "std_tumor_pixels": np.std(tumor_pixel_counts),
            "resolution_stats": {
                "mean_width": np.mean([r[0] for r in image_resolutions]),
                "mean_height": np.mean([r[1] for r in image_resolutions]),
                "std_width": np.std([r[0] for r in image_resolutions]),
                "std_height": np.std([r[1] for r in image_resolutions]),
                "unique_resolutions": len(set(image_resolutions)),
                "most_common_resolution": max(
                    set(image_resolutions), key=image_resolutions.count
                ),
            },
        }

        return eval_summary

    def _calculate_dice(self, pred, target):
        """Calculate Dice coefficient"""
        intersection = (pred * target).sum()
        if pred.sum() + target.sum() == 0:
            return 1.0  # Perfect score for both empty
        return 2 * intersection / (pred.sum() + target.sum())

    def _calculate_iou(self, pred, target):
        """Calculate Intersection over Union"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        if union == 0:
            return 1.0  # Perfect score for both empty
        return intersection / union

    def generate_report(self, val_summary, eval_summary, save_dir="."):
        """Generate comprehensive evaluation report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 60)
        print("TUMOR SEGMENTATION MODEL EVALUATION REPORT")
        print("=" * 60)

        # Validation Set Results
        print("\n📊 VALIDATION SET RESULTS (with ground truth)")
        print("-" * 40)
        print(f"Total images: {val_summary['total_images']}")
        print(
            f"Mean Dice Score: {val_summary['mean_dice']:.4f} ± {val_summary['std_dice']:.4f}"
        )
        print(f"Mean IoU: {val_summary['mean_iou']:.4f} ± {val_summary['std_iou']:.4f}")
        print(f"Mean Precision: {val_summary['mean_precision']:.4f}")
        print(f"Mean Recall: {val_summary['mean_recall']:.4f}")
        print(f"Mean F1-Score: {val_summary['mean_f1']:.4f}")

        # Evaluation Set Results
        print("\n🔍 EVALUATION SET RESULTS (prediction analysis)")
        print("-" * 40)
        print(f"Total images: {eval_summary['total_images']}")
        print(
            f"Images with tumor predictions: {eval_summary['images_with_tumors']} ({eval_summary['tumor_detection_rate']:.1%})"
        )
        print(
            f"Images without tumor predictions: {eval_summary['images_without_tumors']}"
        )
        print(
            f"Average clusters per image: {eval_summary['mean_clusters_per_image']:.2f}"
        )
        print(
            f"Average clusters when tumor detected: {eval_summary['mean_clusters_when_tumor_detected']:.2f}"
        )
        print(
            f"Average tumor pixels per image: {eval_summary['mean_tumor_pixels']:.1f} ± {eval_summary['std_tumor_pixels']:.1f}"
        )

        # Resolution Statistics
        print("\n📐 IMAGE RESOLUTION STATISTICS")
        print("-" * 40)
        res_stats = eval_summary["resolution_stats"]
        print(
            f"Mean resolution: {res_stats['mean_width']:.0f}×{res_stats['mean_height']:.0f}"
        )
        print(
            f"Resolution std: {res_stats['std_width']:.0f}×{res_stats['std_height']:.0f}"
        )
        print(f"Unique resolutions: {res_stats['unique_resolutions']}")
        print(
            f"Most common resolution: {res_stats['most_common_resolution'][0]}×{res_stats['most_common_resolution'][1]}"
        )

        # Generate visualizations
        self._create_visualizations(val_summary, eval_summary, save_dir)

        # Save detailed results
        self._save_detailed_results(save_dir)

        print(f"\n📄 Detailed results saved to: {save_dir}")
        print("=" * 60)

    def _create_visualizations(self, val_summary, eval_summary, save_dir):
        """Create visualization plots"""
        # Validation metrics distribution
        if self.val_results:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Validation Set Metrics Distribution", fontsize=16)

            metrics = ["dice_score", "iou", "precision", "recall", "f1_score"]
            for i, metric in enumerate(metrics):
                row, col = i // 3, i % 3
                values = [r[metric] for r in self.val_results]
                axes[row, col].hist(values, bins=20, alpha=0.7, edgecolor="black")
                axes[row, col].set_title(f"{metric.replace('_', ' ').title()}")
                axes[row, col].set_xlabel("Score")
                axes[row, col].set_ylabel("Frequency")
                axes[row, col].axvline(
                    np.mean(values),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {np.mean(values):.3f}",
                )
                axes[row, col].legend()

            # Remove empty subplot
            axes[1, 2].remove()
            plt.tight_layout()
            plt.savefig(
                save_dir / "validation_metrics_distribution.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Evaluation set statistics
        if self.eval_results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Evaluation Set Prediction Analysis", fontsize=16)

            # Tumor detection pie chart
            tumor_counts = [
                eval_summary["images_with_tumors"],
                eval_summary["images_without_tumors"],
            ]
            axes[0, 0].pie(
                tumor_counts,
                labels=["With Tumors", "Without Tumors"],
                autopct="%1.1f%%",
                startangle=90,
            )
            axes[0, 0].set_title("Tumor Detection Distribution")

            # Cluster count histogram
            cluster_counts = [r["num_clusters"] for r in self.eval_results]
            axes[0, 1].hist(
                cluster_counts,
                bins=max(cluster_counts) + 1 if cluster_counts else 1,
                alpha=0.7,
                edgecolor="black",
            )
            axes[0, 1].set_title("Number of Tumor Clusters per Image")
            axes[0, 1].set_xlabel("Number of Clusters")
            axes[0, 1].set_ylabel("Frequency")

            # Resolution scatter plot
            widths = [r["resolution_w"] for r in self.eval_results]
            heights = [r["resolution_h"] for r in self.eval_results]
            axes[1, 0].scatter(widths, heights, alpha=0.6)
            axes[1, 0].set_title("Image Resolution Distribution")
            axes[1, 0].set_xlabel("Width (pixels)")
            axes[1, 0].set_ylabel("Height (pixels)")

            # Tumor pixel count histogram (for images with tumors)
            tumor_pixels = [
                r["tumor_pixels"] for r in self.eval_results if r["has_tumor_pred"]
            ]
            if tumor_pixels:
                axes[1, 1].hist(tumor_pixels, bins=20, alpha=0.7, edgecolor="black")
                axes[1, 1].set_title(
                    "Tumor Pixel Count Distribution\n(Images with Tumors)"
                )
                axes[1, 1].set_xlabel("Number of Tumor Pixels")
                axes[1, 1].set_ylabel("Frequency")
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No tumor predictions found",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Tumor Pixel Count Distribution")

            plt.tight_layout()
            plt.savefig(
                save_dir / "evaluation_analysis.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _save_detailed_results(self, save_dir):
        """Save detailed results to CSV files"""
        if self.val_results:
            val_df = pd.DataFrame(self.val_results)
            val_df.to_csv(save_dir / "validation_detailed_results.csv", index=False)

        if self.eval_results:
            eval_df = pd.DataFrame(self.eval_results)
            eval_df.to_csv(save_dir / "evaluation_detailed_results.csv", index=False)


def load_model():
    """Load model - simple initialization like in api.py"""
    print("Initializing SimpleUNet model...")

    load_dotenv()
    # Simple model initialization (same as api.py)
    # Load the PyTorch Lightning model from checkpoint (class method)
    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH",
    )
    if not checkpoint_path:
        raise ValueError("CHECKPOINT_PATH environment variable is not set")
    model = NNUNetStyle.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )  # Force CPU loading
    model.eval()  # Set to evaluation mode

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate tumor segmentation model")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size for model input"
    )

    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model (simple initialization like api.py)
    model = load_model()
    evaluator = ModelEvaluator(model, device)

    # Set up data module
    data_module = TumorSegmentationDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=False,  # No augmentation for evaluation
    )

    # Setup datasets
    data_module.setup()
    data_module.setup_evaluation_set()

    # Get dataloaders
    val_dataloader = data_module.val_dataloader()
    eval_dataloader = data_module.eval_dataloader()

    # Run evaluations
    val_summary = evaluator.evaluate_validation_set(val_dataloader)

    if eval_dataloader is not None:
        eval_summary = evaluator.evaluate_evaluation_set(eval_dataloader)
    else:
        print("Warning: No evaluation set found")
        eval_summary = {
            "total_images": 0,
            "images_with_tumors": 0,
            "images_without_tumors": 0,
            "tumor_detection_rate": 0,
            "mean_clusters_per_image": 0,
            "mean_clusters_when_tumor_detected": 0,
            "mean_tumor_pixels": 0,
            "std_tumor_pixels": 0,
            "resolution_stats": {
                "mean_width": 0,
                "mean_height": 0,
                "std_width": 0,
                "std_height": 0,
                "unique_resolutions": 0,
                "most_common_resolution": (0, 0),
            },
        }

    # Generate report
    evaluator.generate_report(val_summary, eval_summary, args.save_dir)


if __name__ == "__main__":
    main()
