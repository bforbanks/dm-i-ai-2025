# import os  # Unused
import torch
import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd  # Unused
from pathlib import Path
from typing import List, Dict, Tuple
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import wandb


class FixedDiceAnalysisCallback(Callback):
    """
    FIXED Callback for OrganDetector with optimized cost function.

    Key fixes:
    1. Applies threshold masking during inference
    2. Converts organ predictions to tumor predictions
    3. Handles inverted segmentation targets correctly
    4. Only analyzes pixels under intensity threshold
    """

    def __init__(
        self,
        analysis_every_n_epochs: int = 5,
        save_top_k: int = 7,
        verbose: bool = True,
    ):
        super().__init__()
        self.analysis_every_n_epochs = analysis_every_n_epochs
        self.save_top_k = save_top_k
        self.verbose = verbose
        self.validation_predictions = []
        self.validation_targets = []
        self.validation_images = []
        self.validation_names = []
        self.trainer = None

        # Create visualization directory
        self.viz_dir = Path("visualizations/fixed_worstdice")
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV tracking
        self.csv_path = self.viz_dir / "patient_dice_scores_fixed.csv"
        self.patient_dice_history = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Tuple,
        batch_idx: int,
    ):
        """Store validation batch data for analysis."""
        if self.trainer is None:
            self.trainer = trainer

        if trainer.current_epoch % self.analysis_every_n_epochs == 0:
            images, targets = batch

            # CRITICAL FIX: Use tiled inference with threshold masking
            with torch.no_grad():
                # Get organ predictions from model
                organ_predictions = pl_module(images)

                # Apply threshold masking (like in training)
                images_255 = (images * 255.0).clamp(0, 255)
                threshold_mask = (images_255 < pl_module.intensity_threshold).float()

                # Convert organ predictions to tumor predictions
                tumor_predictions = 1.0 - organ_predictions

                # Apply threshold masking to tumor predictions
                tumor_predictions_masked = tumor_predictions * threshold_mask

                # Convert organ targets to tumor targets
                tumor_targets = (1.0 - targets) * threshold_mask

            # Store the TUMOR data (not organ data)
            self.validation_predictions.extend(tumor_predictions_masked.cpu().numpy())
            self.validation_targets.extend(tumor_targets.cpu().numpy())
            self.validation_images.extend(images.cpu().numpy())

            # Generate sample names
            batch_size = images.shape[0]
            for i in range(batch_size):
                try:
                    dataset = trainer.val_dataloaders[0].dataset
                    global_idx = batch_idx * trainer.val_dataloaders[0].batch_size + i
                    if hasattr(dataset, "all_images") and global_idx < len(
                        dataset.all_images
                    ):
                        img_path = dataset.all_images[global_idx]
                        sample_name = Path(img_path).stem
                    else:
                        sample_name = f"batch_{batch_idx}_sample_{i}"
                except Exception:
                    sample_name = f"batch_{batch_idx}_sample_{i}"

                self.validation_names.append(sample_name)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Analyze worst performing images for TUMOR DETECTION."""
        if (
            trainer.current_epoch % self.analysis_every_n_epochs == 0
            and self.validation_predictions
        ):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(
                    f"EPOCH {trainer.current_epoch} - FIXED TUMOR DETECTION DICE ANALYSIS"
                )
                print(f"{'=' * 60}")

            # Convert to numpy arrays
            predictions = np.array(self.validation_predictions)  # TUMOR predictions
            targets = np.array(self.validation_targets)  # TUMOR targets
            images = np.array(self.validation_images)
            names = self.validation_names

            # Calculate dice scores for each sample and filter for samples with tumor targets
            dice_scores = []
            fp_counts = []
            fn_counts = []
            patient_indices = []
            patient_names = []

            for i in range(len(predictions)):
                pred_values = predictions[i, 0]
                target_binary = targets[i, 0].astype(np.float32)

                # Check if this sample has tumor targets (target_sum > 0 means tumors present)
                target_sum = np.sum(target_binary)
                if target_sum == 0:
                    # No tumor targets in this sample, skip
                    continue

                # This sample has tumors, include it
                patient_indices.append(i)
                patient_names.append(names[i])

                # Clamp and binarize predictions
                pred_values = np.clip(pred_values, 0.0, 1.0)
                pred_binary = (pred_values > 0.5).astype(np.float32)

                # Debug: print statistics for first few samples
                if self.verbose and len(patient_indices) <= 3:
                    pred_min, pred_max = pred_values.min(), pred_values.max()
                    pred_mean = pred_values.mean()
                    target_pixels = target_sum
                    pred_pixels = pred_binary.sum()
                    print(
                        f"TUMOR sample {len(patient_indices)}: pred [{pred_min:.4f}, {pred_max:.4f}], mean={pred_mean:.4f}"
                    )
                    print(
                        f"  Target pixels: {target_pixels:.0f}, Pred pixels: {pred_pixels:.0f}"
                    )

                # Calculate dice score for TUMOR detection
                intersection = np.sum(pred_binary * target_binary)
                union = np.sum(pred_binary) + np.sum(target_binary)
                dice = (2.0 * intersection) / (union + 1e-6)
                dice_scores.append(dice)

                # Calculate FP and FN for tumor detection
                fp = np.sum(
                    pred_binary * (1 - target_binary)
                )  # Predicted tumor but no tumor
                fn = np.sum((1 - pred_binary) * target_binary)  # Missed tumor
                fp_counts.append(fp)
                fn_counts.append(fn)

            # Check if we have any samples with tumors
            if len(patient_indices) == 0:
                if self.verbose:
                    print("⚠️  No samples with tumor targets found!")
                return

            # Find worst, best performing
            dice_scores = np.array(dice_scores)

            # Worst performing (lowest dice scores)
            worst_patient_indices = np.argsort(dice_scores)[
                : min(self.save_top_k, len(dice_scores))
            ]
            worst_indices = [patient_indices[idx] for idx in worst_patient_indices]

            # Best performing (highest dice scores)
            best_patient_indices = np.argsort(dice_scores)[::-1][
                : min(self.save_top_k, len(dice_scores))
            ]
            best_indices = [patient_indices[idx] for idx in best_patient_indices]

            # Report results
            if self.verbose:
                print(f"\nTUMOR DETECTION RESULTS:")
                print(f"Samples with tumors: {len(patient_indices)}")
                print(f"Average tumor dice: {np.mean(dice_scores):.4f}")
                print(f"Intensity threshold: <{pl_module.intensity_threshold}")

                print(f"\nTop {len(worst_indices)} worst TUMOR detection:")
                print("-" * 50)
                for i, idx in enumerate(worst_indices):
                    name = names[idx]
                    dice = dice_scores[worst_patient_indices[i]]
                    fp = fp_counts[worst_patient_indices[i]]
                    fn = fn_counts[worst_patient_indices[i]]
                    print(f"{i + 1}. {name}: Dice={dice:.4f}, FP={fp:.0f}, FN={fn:.0f}")

                print(f"\nTop {len(best_indices)} best TUMOR detection:")
                print("-" * 50)
                for i, idx in enumerate(best_indices):
                    name = names[idx]
                    dice = dice_scores[best_patient_indices[i]]
                    fp = fp_counts[best_patient_indices[i]]
                    fn = fn_counts[best_patient_indices[i]]
                    print(f"{i + 1}. {name}: Dice={dice:.4f}, FP={fp:.0f}, FN={fn:.0f}")

            # Save visualization with FIXED logic
            self._save_fixed_visualization(
                images,
                targets,
                predictions,
                names,
                worst_indices,
                best_indices,
                worst_patient_indices,
                best_patient_indices,
                dice_scores,
                trainer.current_epoch,
                pl_module.intensity_threshold,
            )

            # Clear stored data
            self.validation_predictions = []
            self.validation_targets = []
            self.validation_images = []
            self.validation_names = []

            if self.verbose:
                print(f"✅ FIXED analysis complete - showing TUMOR detection results")
                print(f"{'=' * 60}\n")

    def _save_fixed_visualization(
        self,
        images: np.ndarray,
        targets: np.ndarray,  # TUMOR targets
        predictions: np.ndarray,  # TUMOR predictions (masked)
        names: List[str],
        worst_indices: List[int],
        best_indices: List[int],
        worst_patient_indices: List[int],
        best_patient_indices: List[int],
        dice_scores: np.ndarray,
        epoch: int,
        intensity_threshold: int,
    ):
        """Save FIXED visualization showing tumor detection with threshold masking."""
        k = len(worst_indices)
        fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))

        if k == 1:
            axes = axes.reshape(1, -1)

        # Helper function to create overlay with threshold awareness
        def create_tumor_overlay(pred_binary, target_binary, image, threshold):
            # Create threshold mask
            threshold_mask = image < threshold

            # Only consider pixels under threshold
            pred_masked = pred_binary * threshold_mask
            target_masked = target_binary * threshold_mask

            TP = (pred_masked > 0) & (target_masked > 0)
            FP = (pred_masked > 0) & (target_masked == 0)
            FN = (pred_masked == 0) & (target_masked > 0)

            overlay = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
            overlay[TP] = [0, 255, 0]  # Green for TP (correct tumor detection)
            overlay[FP] = [255, 0, 0]  # Red for FP (false tumor)
            overlay[FN] = [0, 0, 255]  # Blue for FN (missed tumor)

            # Show threshold mask boundary
            boundary = threshold_mask.astype(np.uint8) * 50
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], boundary)
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], boundary)
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], boundary)

            return overlay

        # Column 1: Worst performing - Original images
        for i in range(k):
            idx = worst_indices[i]
            axes[i, 0].imshow(images[idx, 0], cmap="gray")
            axes[i, 0].set_title(f"Worst {i + 1}: {names[idx]}")
            axes[i, 0].axis("off")

        # Column 2: Worst performing - TUMOR detection overlays
        for i in range(k):
            idx = worst_indices[i]
            dice = dice_scores[worst_patient_indices[i]]

            pred_values = np.clip(predictions[idx, 0], 0.0, 1.0)
            pred_binary = (pred_values > 0.5).astype(np.float32)
            overlay = create_tumor_overlay(
                pred_binary, targets[idx, 0], images[idx, 0], intensity_threshold
            )
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Worst {i + 1}: TUMOR Dice = {dice:.4f}")
            axes[i, 1].axis("off")

        # Column 3: Best performing - Original images
        for i in range(k):
            idx = best_indices[i]
            axes[i, 2].imshow(images[idx, 0], cmap="gray")
            axes[i, 2].set_title(f"Best {i + 1}: {names[idx]}")
            axes[i, 2].axis("off")

        # Column 4: Best performing - TUMOR detection overlays
        for i in range(k):
            idx = best_indices[i]
            dice = dice_scores[best_patient_indices[i]]

            pred_values = np.clip(predictions[idx, 0], 0.0, 1.0)
            pred_binary = (pred_values > 0.5).astype(np.float32)
            overlay = create_tumor_overlay(
                pred_binary, targets[idx, 0], images[idx, 0], intensity_threshold
            )
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f"Best {i + 1}: TUMOR Dice = {dice:.4f}")
            axes[i, 3].axis("off")

        # Add FIXED legend
        import matplotlib.patches as mpatches

        green_patch = mpatches.Patch(color="green", label="TP (Correct Tumor)")
        red_patch = mpatches.Patch(color="red", label="FP (False Tumor)")
        blue_patch = mpatches.Patch(color="blue", label="FN (Missed Tumor)")
        axes[0, 1].legend(
            handles=[green_patch, red_patch, blue_patch], loc="lower right", fontsize=8
        )
        axes[0, 3].legend(
            handles=[green_patch, red_patch, blue_patch], loc="lower right", fontsize=8
        )

        # Add threshold info and epoch for clarity
        fig.suptitle(
            f"FIXED Tumor Detection Analysis (Threshold < {intensity_threshold}) - Epoch {epoch}",
            fontsize=16,
            y=0.98,
        )

        plt.tight_layout()

        # Save locally
        save_path = self.viz_dir / f"fixed_tumor_dice_epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Log to wandb if available
        if (
            hasattr(self, "trainer")
            and self.trainer
            and self.trainer.logger
            and hasattr(self.trainer.logger, "experiment")
        ):
            # Use global_step for proper wandb monotonicity
            current_step = self.trainer.global_step
            self.trainer.logger.experiment.log(
                {"fixed_tumor_dice_visualization": wandb.Image(fig)}, step=current_step
            )

        plt.close()

        if self.verbose:
            print(f"📊 Saved FIXED tumor detection visualization to: {save_path}")
            print(
                f"📊 Logged to wandb as: fixed_tumor_dice_visualization (epoch {epoch})"
            )
