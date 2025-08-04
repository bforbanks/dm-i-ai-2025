import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import wandb


class DiceAnalysisCallback(Callback):
    """
    Callback to analyze worst performing images every n epochs.
    Shows FP/FN problems and saves visualization plots.
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
        self.trainer = None  # Store trainer reference for wandb logging

        # Create visualization directory
        self.viz_dir = Path("visualizations/worstdice")
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV tracking
        self.csv_path = self.viz_dir / "patient_dice_scores.csv"
        self.patient_dice_history = {}  # {patient_name: {epoch: dice_score}}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Tuple,
        batch_idx: int,
    ):
        """Store validation batch data for analysis."""
        # Store trainer reference for wandb logging
        if self.trainer is None:
            self.trainer = trainer

        if trainer.current_epoch % self.analysis_every_n_epochs == 0:
            images, targets = batch

            # Get predictions directly from the model
            # The validation_step only returns the loss, so we need to call the model directly
            with torch.no_grad():
                predictions = pl_module(images)
                # Note: NNUNetStyle forward method already applies sigmoid, so don't apply it again

            # Store data
            self.validation_predictions.extend(predictions.cpu().numpy())
            self.validation_targets.extend(targets.cpu().numpy())
            self.validation_images.extend(images.cpu().numpy())

            # Try to get actual image names from dataset if possible
            batch_size = images.shape[0]
            for i in range(batch_size):
                try:
                    # Try to get actual image name from dataset
                    dataset = trainer.val_dataloaders[0].dataset
                    global_idx = batch_idx * trainer.val_dataloaders[0].batch_size + i
                    if hasattr(dataset, "all_images") and global_idx < len(
                        dataset.all_images
                    ):
                        img_path = dataset.all_images[global_idx]
                        sample_name = Path(
                            img_path
                        ).stem  # Get filename without extension
                    else:
                        sample_name = f"batch_{batch_idx}_sample_{i}"
                except:
                    sample_name = f"batch_{batch_idx}_sample_{i}"

                self.validation_names.append(sample_name)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Analyze worst performing images and save visualizations."""
        if (
            trainer.current_epoch % self.analysis_every_n_epochs == 0
            and self.validation_predictions
        ):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"EPOCH {trainer.current_epoch} - DICE ANALYSIS")
                print(f"{'=' * 60}")

            # Convert to numpy arrays
            predictions = np.array(self.validation_predictions)
            targets = np.array(self.validation_targets)
            images = np.array(self.validation_images)
            names = self.validation_names

            # Calculate dice scores for each sample and filter for patient images only
            dice_scores = []
            fp_counts = []
            fn_counts = []
            patient_indices = []  # Store indices of patient images only
            patient_names = []  # Store names of patient images only

            for i in range(len(predictions)):
                # Check if predictions are in reasonable range (0-1)
                pred_values = predictions[i, 0]
                if pred_values.max() > 1.0 or pred_values.min() < 0.0:
                    if self.verbose:
                        print(
                            f"⚠️  Warning: Sample {i} has predictions outside [0,1] range: [{pred_values.min():.4f}, {pred_values.max():.4f}]"
                        )
                    # Clamp predictions to [0,1] for analysis
                    pred_values = np.clip(pred_values, 0.0, 1.0)

                pred_binary = (pred_values > 0.5).astype(np.float32)
                target_binary = targets[i, 0].astype(np.float32)

                # Check if this is a patient image (has tumors in ground truth)
                target_sum = np.sum(target_binary)
                if target_sum == 0:
                    # This is a control image, skip it
                    continue

                # This is a patient image, include it
                patient_indices.append(i)
                patient_names.append(names[i])

                # Debug: print prediction statistics for first few patient samples
                if self.verbose and len(patient_indices) <= 3:
                    pred_min, pred_max = pred_values.min(), pred_values.max()
                    pred_mean = pred_values.mean()
                    print(
                        f"Patient sample {len(patient_indices)}: pred range [{pred_min:.4f}, {pred_max:.4f}], mean={pred_mean:.4f}"
                    )

                # Calculate dice score
                intersection = np.sum(pred_binary * target_binary)
                union = np.sum(pred_binary) + np.sum(target_binary)
                dice = (2.0 * intersection) / (union + 1e-6)
                dice_scores.append(dice)

                # Calculate FP and FN
                fp = np.sum(
                    pred_binary * (1 - target_binary)
                )  # Predicted but not in target
                fn = np.sum(
                    (1 - pred_binary) * target_binary
                )  # In target but not predicted
                fp_counts.append(fp)
                fn_counts.append(fn)

            # Check if we have any patient images
            if len(patient_indices) == 0:
                if self.verbose:
                    print("⚠️  No patient images found in validation set!")
                return

            # Find worst, best, and median performing patient images
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

            # Median performing (closest to median)
            median_dice = np.median(dice_scores)
            median_distances = np.abs(dice_scores - median_dice)
            median_patient_indices = np.argsort(median_distances)[
                : min(self.save_top_k, len(dice_scores))
            ]
            median_indices = [patient_indices[idx] for idx in median_patient_indices]

            # Analyze problems for worst performing
            if self.verbose:
                print(f"\nTop {len(worst_indices)} worst performing PATIENT images:")
                print("-" * 50)

            total_fp_worst = 0
            total_fn_worst = 0

            for i, idx in enumerate(worst_indices):
                name = names[idx]
                dice = dice_scores[worst_patient_indices[i]]
                fp = fp_counts[worst_patient_indices[i]]
                fn = fn_counts[worst_patient_indices[i]]

                if self.verbose:
                    print(f"{i + 1}. {name}: Dice={dice:.4f}, FP={fp:.0f}, FN={fn:.0f}")
                total_fp_worst += fp
                total_fn_worst += fn

            # Show best performing
            if self.verbose:
                print(f"\nTop {len(best_indices)} best performing PATIENT images:")
                print("-" * 50)

            for i, idx in enumerate(best_indices):
                name = names[idx]
                dice = dice_scores[best_patient_indices[i]]
                fp = fp_counts[best_patient_indices[i]]
                fn = fn_counts[best_patient_indices[i]]

                if self.verbose:
                    print(f"{i + 1}. {name}: Dice={dice:.4f}, FP={fp:.0f}, FN={fn:.0f}")

            if self.verbose:
                print(f"\nTotal problems in worst {len(worst_indices)} images:")
                print(f"False Positives (FP): {total_fp_worst:.0f}")
                print(f"False Negatives (FN): {total_fn_worst:.0f}")

            # Calculate average dice scores
            avg_dice_worst = np.mean(dice_scores[worst_patient_indices])
            avg_dice_best = np.mean(dice_scores[best_patient_indices])
            avg_dice_median = np.mean(dice_scores[median_patient_indices])
            avg_dice_all = np.mean(dice_scores)

            if self.verbose:
                print(f"Average Dice Scores:")
                print(f"  - Worst {len(worst_indices)}: {avg_dice_worst:.4f}")
                print(f"  - Best {len(best_indices)}: {avg_dice_best:.4f}")
                print(f"  - Median {len(median_indices)}: {avg_dice_median:.4f}")
                print(f"  - All patient images: {avg_dice_all:.4f}")

            if self.verbose:
                if total_fp_worst > total_fn_worst:
                    print(
                        f"⚠️  MAIN PROBLEM: Too many False Positives ({total_fp_worst:.0f} vs {total_fn_worst:.0f} FN)"
                    )
                    print("   → Model is predicting tumors where there aren't any")
                    if total_fp_worst > 10000:  # Very high FP count
                        print(
                            "   → NOTE: Very high FP count is normal in early training epochs"
                        )
                        print("   → The model will improve as training progresses")
                    print(
                        "   → Consider: Increase threshold, add regularization, or review data"
                    )
                elif total_fn_worst > total_fp_worst:
                    print(
                        f"⚠️  MAIN PROBLEM: Too many False Negatives ({total_fn_worst:.0f} vs {total_fp_worst:.0f} FP)"
                    )
                    print("   → Model is missing actual tumors")
                    print(
                        "   → Consider: Lower threshold, improve feature extraction, or review data"
                    )
                else:
                    print(
                        f"⚠️  BALANCED PROBLEMS: Similar FP ({total_fp_worst:.0f}) and FN ({total_fn_worst:.0f})"
                    )
                    print("   → Model has general segmentation issues")
                    print(
                        "   → Consider: Architecture improvements or more training data"
                    )

            # Save visualization (worst, best, and median patient images)
            self._save_comprehensive_visualization(
                images,
                targets,
                predictions,
                names,
                worst_indices,
                best_indices,
                median_indices,
                worst_patient_indices,
                best_patient_indices,
                median_patient_indices,
                dice_scores,
                trainer.current_epoch,
            )

            # Save CSV with all patient dice scores
            self._save_dice_csv(patient_names, dice_scores, trainer.current_epoch)

            # Log dice statistics to wandb
            self._log_dice_statistics_to_wandb(
                dice_scores,
                worst_patient_indices,
                best_patient_indices,
                median_patient_indices,
                total_fp_worst,
                total_fn_worst,
                trainer.current_epoch,
            )

            # Clear stored data
            self.validation_predictions = []
            self.validation_targets = []
            self.validation_images = []
            self.validation_names = []

            if self.verbose:
                print(f"{'=' * 60}\n")

    def _save_comprehensive_visualization(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        names: List[str],
        worst_indices: List[int],
        best_indices: List[int],
        median_indices: List[int],
        worst_patient_indices: List[int],
        best_patient_indices: List[int],
        median_patient_indices: List[int],
        dice_scores: np.ndarray,
        epoch: int,
    ):
        """Save comprehensive visualization with 4 columns: worst original, worst overlay, best original, best overlay."""
        k = len(worst_indices)
        fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))

        if k == 1:
            axes = axes.reshape(1, -1)

        # Helper function to create overlay
        def create_overlay(pred_binary, target_binary):
            TP = (pred_binary > 0) & (target_binary > 0)
            FP = (pred_binary > 0) & (target_binary == 0)
            FN = (pred_binary == 0) & (target_binary > 0)

            overlay = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
            overlay[TP] = [0, 255, 0]  # Green for TP
            overlay[FP] = [255, 0, 0]  # Red for FP
            overlay[FN] = [0, 0, 255]  # Blue for FN
            return overlay

        # Column 1: Worst performing - Original images
        for i in range(k):
            idx = worst_indices[i]
            dice = dice_scores[worst_patient_indices[i]]

            axes[i, 0].imshow(images[idx, 0], cmap="gray")
            axes[i, 0].set_title(f"Worst {i + 1}: {names[idx]}")
            axes[i, 0].axis("off")

        # Column 2: Worst performing - Overlays
        for i in range(k):
            idx = worst_indices[i]
            dice = dice_scores[worst_patient_indices[i]]

            pred_values = np.clip(predictions[idx, 0], 0.0, 1.0)
            pred_binary = (pred_values > 0.5).astype(np.float32)
            overlay = create_overlay(pred_binary, targets[idx, 0])
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Worst {i + 1}: Dice = {dice:.4f}")
            axes[i, 1].axis("off")

        # Column 3: Best performing - Original images
        for i in range(k):
            idx = best_indices[i]
            dice = dice_scores[best_patient_indices[i]]

            axes[i, 2].imshow(images[idx, 0], cmap="gray")
            axes[i, 2].set_title(f"Best {i + 1}: {names[idx]}")
            axes[i, 2].axis("off")

        # Column 4: Best performing - Overlays
        for i in range(k):
            idx = best_indices[i]
            dice = dice_scores[best_patient_indices[i]]

            pred_values = np.clip(predictions[idx, 0], 0.0, 1.0)
            pred_binary = (pred_values > 0.5).astype(np.float32)
            overlay = create_overlay(pred_binary, targets[idx, 0])
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f"Best {i + 1}: Dice = {dice:.4f}")
            axes[i, 3].axis("off")

        # Add legend to the first row
        import matplotlib.patches as mpatches

        green_patch = mpatches.Patch(color="green", label="TP")
        red_patch = mpatches.Patch(color="red", label="FP")
        blue_patch = mpatches.Patch(color="blue", label="FN")
        axes[0, 1].legend(
            handles=[green_patch, red_patch, blue_patch], loc="lower right", fontsize=8
        )
        axes[0, 3].legend(
            handles=[green_patch, red_patch, blue_patch], loc="lower right", fontsize=8
        )

        plt.tight_layout()

        # Save locally
        save_path = self.viz_dir / f"worst_dice_epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Log to wandb if available
        if (
            hasattr(self, "trainer")
            and self.trainer
            and self.trainer.logger
            and hasattr(self.trainer.logger, "experiment")
        ):
            # Use current global step instead of epoch to avoid step ordering issues
            current_step = self.trainer.global_step
            self.trainer.logger.experiment.log(
                {f"worst_dice_visualization": wandb.Image(fig)}, step=current_step
            )

        plt.close()

        if self.verbose:
            print(f"📊 Saved comprehensive visualization to: {save_path}")
            print(f"📊 Logged to wandb as: worst_dice_visualization_epoch_{epoch}")

    def _save_dice_csv(
        self, patient_names: List[str], dice_scores: np.ndarray, epoch: int
    ):
        """Save CSV with all patient dice scores across epochs."""
        # Update the dice history
        for i, name in enumerate(patient_names):
            if name not in self.patient_dice_history:
                self.patient_dice_history[name] = {}
            self.patient_dice_history[name][epoch] = dice_scores[i]

        # Create DataFrame
        all_epochs = sorted(
            set(
                epoch
                for patient_data in self.patient_dice_history.values()
                for epoch in patient_data.keys()
            )
        )

        # Create data for DataFrame
        data = {"Patient_Name": []}
        for e in all_epochs:
            data[f"Epoch_{e:03d}"] = []

        # Fill in the data
        for patient_name in sorted(self.patient_dice_history.keys()):
            data["Patient_Name"].append(patient_name)
            for e in all_epochs:
                if e in self.patient_dice_history[patient_name]:
                    data[f"Epoch_{e:03d}"].append(
                        f"{self.patient_dice_history[patient_name][e]:.6f}"
                    )
                else:
                    data[f"Epoch_{e:03d}"].append(
                        ""
                    )  # Empty cell if patient not present in that epoch

        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)

        if self.verbose:
            print(f"📊 Saved patient dice scores CSV to: {self.csv_path}")
            print(
                f"   → Tracking {len(self.patient_dice_history)} patients across {len(all_epochs)} epochs"
            )

    def _log_dice_statistics_to_wandb(
        self,
        dice_scores: np.ndarray,
        worst_patient_indices: List[int],
        best_patient_indices: List[int],
        median_patient_indices: List[int],
        total_fp_worst: float,
        total_fn_worst: float,
        epoch: int,
    ):
        """Log dice statistics to wandb for monitoring."""
        if not (
            self.trainer
            and self.trainer.logger
            and hasattr(self.trainer.logger, "experiment")
        ):
            return

        # Calculate statistics
        avg_dice_worst = np.mean(dice_scores[worst_patient_indices])
        avg_dice_best = np.mean(dice_scores[best_patient_indices])
        avg_dice_median = np.mean(dice_scores[median_patient_indices])
        avg_dice_all = np.mean(dice_scores)

        # Log to wandb
        wandb_logs = {
            f"dice_analysis/avg_dice_worst_{len(worst_patient_indices)}": avg_dice_worst,
            f"dice_analysis/avg_dice_best_{len(best_patient_indices)}": avg_dice_best,
            f"dice_analysis/avg_dice_median_{len(median_patient_indices)}": avg_dice_median,
            f"dice_analysis/avg_dice_all_patients": avg_dice_all,
            f"dice_analysis/total_fp_worst_{len(worst_patient_indices)}": total_fp_worst,
            f"dice_analysis/total_fn_worst_{len(worst_patient_indices)}": total_fn_worst,
            f"dice_analysis/fp_fn_ratio": total_fp_worst
            / (total_fn_worst + 1e-6),  # Avoid division by zero
            f"dice_analysis/num_patient_images": len(dice_scores),
        }

        # Use current global step instead of epoch to avoid step ordering issues
        current_step = self.trainer.global_step
        self.trainer.logger.experiment.log(wandb_logs, step=current_step)
        if self.verbose:
            print(
                f"📊 Logged dice statistics to wandb for epoch {epoch} at step {current_step}"
            )
