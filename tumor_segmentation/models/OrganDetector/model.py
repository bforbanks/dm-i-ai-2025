import torch
import torch.nn as nn
from tumor_segmentation.models.base_model import BaseModel


class OrganDetector(BaseModel):
    """
    Organ Detector for tiled tumor segmentation.

    Optimized for horizontal tiles (128px height, variable width).
    Predicts non-tumor dark pixels (organs) vs tumor pixels.

    Inherits from BaseModel for consistent training/validation logic.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,  # Kept for compatibility, but we'll use weighted BCE instead
        tile_height: int = 128,
        intensity_threshold: int = 85,
        tumor_emphasis_weight: float = 2.5,  # Weight to emphasize tumor detection
        use_weighted_bce: bool = True,  # Use weighted BCE optimized for tumor detection
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight
        )

        self.tile_height = tile_height
        self.intensity_threshold = intensity_threshold
        self.tumor_emphasis_weight = tumor_emphasis_weight
        self.use_weighted_bce = use_weighted_bce

        # Based on medical domain knowledge:
        # - Organs (brain, kidneys, heart, liver) are ~70% of dark pixels
        # - Tumors are ~30% of dark pixels (minority class)
        # - For final tumor detection task, we need to emphasize tumor recall

        # Calculate pos_weight for BCE loss
        # pos_weight = weight for positive class (organs in our case)
        # Since tumors are minority and more important for final task,
        # we give higher relative weight to tumor prediction errors
        estimated_organ_ratio = 0.7  # 70% of dark pixels are organs
        estimated_tumor_ratio = 0.3  # 30% of dark pixels are tumors

        # Inverse frequency weighting with tumor emphasis
        self.organ_weight = 1.0 / estimated_organ_ratio
        self.tumor_weight = tumor_emphasis_weight / estimated_tumor_ratio

        # Normalize weights
        total_weight = self.organ_weight + self.tumor_weight
        self.organ_weight_normalized = self.organ_weight / total_weight
        self.tumor_weight_normalized = self.tumor_weight / total_weight

        print(f"💡 OrganDetector Cost Function Optimization:")
        print(f"   Estimated organ ratio: {estimated_organ_ratio:.1%}")
        print(f"   Estimated tumor ratio: {estimated_tumor_ratio:.1%}")
        print(f"   Tumor emphasis factor: {tumor_emphasis_weight:.1f}x")
        print(
            f"   Final weights - Organ: {self.organ_weight_normalized:.3f}, Tumor: {self.tumor_weight_normalized:.3f}"
        )
        print(f"   Using weighted BCE: {use_weighted_bce}")

        # Optimized encoder for smaller tiles (less deep than full image segmentation)
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)

        # Bottleneck (removed one encoder level since tiles are smaller)
        self.bottleneck = self._make_layer(128, 256)

        # Decoder (adjusted for 3-level encoder)
        self.dec3 = self._make_layer(256 + 128, 128)  # +128 for skip connection
        self.dec2 = self._make_layer(128 + 64, 64)  # +64 for skip connection
        self.dec1 = self._make_layer(64 + 32, 32)  # +32 for skip connection

        # Final output with slight regularization
        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(32, num_classes, 1)
        )

        # Pooling operations
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a simple convolutional layer with dropout for regularization"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout after first conv block
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout after second conv block
        )

    def forward(self, x):
        # Validate input dimensions for tile processing
        batch_size, channels, height, width = x.shape
        assert height == self.tile_height, (
            f"Expected height {self.tile_height}, got {height}"
        )
        assert height % 8 == 0, (
            f"Height must be divisible by 8 for 3-level pooling, got {height}"
        )

        # Encoder path (3 levels for smaller tiles)
        enc1 = self.enc1(x)  # [B, 32, 128, W]
        enc2 = self.enc2(self.pool(enc1))  # [B, 64, 64, W/2]
        enc3 = self.enc3(self.pool(enc2))  # [B, 128, 32, W/4]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # [B, 256, 16, W/8]

        # Decoder path with skip connections
        dec3 = self.dec3(
            torch.cat([self.upsample(bottleneck), enc3], dim=1)
        )  # [B, 128, 32, W/4]
        dec2 = self.dec2(
            torch.cat([self.upsample(dec3), enc2], dim=1)
        )  # [B, 64, 64, W/2]
        dec1 = self.dec1(
            torch.cat([self.upsample(dec2), enc1], dim=1)
        )  # [B, 32, 128, W]

        # Final output
        output = self.final_conv(dec1)  # [B, 1, 128, W]

        return torch.sigmoid(output)

    def _calculate_optimized_loss(self, pred, target, images):
        """
        Calculate optimized weighted BCE loss for tumor detection task.

        This loss function is designed to:
        1. Focus only on dark pixels (< intensity_threshold)
        2. Use weighted BCE to handle organ:tumor class imbalance
        3. Emphasize tumor detection for the final task
        4. Output organ probabilities (1-prob = tumor probabilities)

        Args:
            pred: Model predictions [B, 1, H, W] (sigmoid output, organ probabilities)
            target: Ground truth [B, 1, H, W] (1=organ, 0=tumor in dark areas)
            images: Original images [B, 1, H, W] (normalized 0-1)

        Returns:
            Weighted BCE loss optimized for tumor detection
        """
        # Convert normalized images back to 0-255 range for threshold comparison
        images_255 = (images * 255.0).clamp(0, 255)

        # Create mask for pixels below intensity threshold
        threshold_mask = (images_255 < self.intensity_threshold).float()

        # Apply threshold mask to predictions and targets
        pred_masked = pred * threshold_mask
        target_masked = target * threshold_mask

        # Calculate number of valid pixels for proper normalization
        valid_pixels = threshold_mask.sum()

        if valid_pixels == 0:
            # No valid pixels in batch, return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.use_weighted_bce:
            # Use weighted BCE loss optimized for tumor detection

            # Create sample weights based on class (organ vs tumor)
            # Higher weight for tumor pixels (target=0) to emphasize tumor detection
            sample_weights = torch.where(
                target_masked > 0.5,  # Organ pixels
                self.organ_weight_normalized,
                self.tumor_weight_normalized,  # Tumor pixels (more important)
            )

            # Apply threshold mask to sample weights
            sample_weights = sample_weights * threshold_mask

            # Calculate weighted BCE loss
            bce_loss = (
                torch.nn.functional.binary_cross_entropy(
                    pred_masked, target_masked, weight=sample_weights, reduction="sum"
                )
                / valid_pixels
            )

            return bce_loss

        else:
            # Fallback to standard BCE on thresholded pixels
            bce_loss = (
                torch.nn.functional.binary_cross_entropy(
                    pred_masked, target_masked, reduction="sum"
                )
                / valid_pixels
            )

            return bce_loss

    def _calculate_threshold_dice_loss(
        self, pred_masked, target_masked, threshold_mask
    ):
        """Calculate Dice loss only on threshold-masked pixels"""
        # Calculate intersection and union only for valid pixels
        intersection = (pred_masked * target_masked * threshold_mask).sum()
        pred_sum = (pred_masked * threshold_mask).sum()
        target_sum = (target_masked * threshold_mask).sum()

        # Dice coefficient with smoothing
        dice = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        # Return Dice loss (1 - Dice)
        return 1 - dice

    def _calculate_naive_dice_tumor_only(
        self, tumor_pred_binary, tumor_targets, threshold_mask
    ):
        """
        Calculate naive dice ONLY on samples that have tumor targets.

        This prevents inflated scores from tiles without tumors and gives
        a meaningful measure of tumor detection performance.

        Args:
            tumor_pred_binary: Binary tumor predictions [B, 1, H, W]
            tumor_targets: Binary tumor targets [B, 1, H, W]
            threshold_mask: Threshold mask [B, 1, H, W]

        Returns:
            Naive dice score only for samples with tumors, or 0 if no tumor samples
        """
        batch_size = tumor_pred_binary.size(0)
        tumor_dice_scores = []

        for i in range(batch_size):
            # Get single sample
            pred_sample = tumor_pred_binary[i]  # [1, H, W]
            target_sample = tumor_targets[i]  # [1, H, W]
            mask_sample = threshold_mask[i]  # [1, H, W]

            # Check if this sample has tumor targets
            target_sum = (target_sample * mask_sample).sum()

            if target_sum > 0:  # Only calculate dice for samples with tumors
                # Apply threshold mask and flatten
                pred_flat = (pred_sample * mask_sample).view(-1)
                target_flat = (target_sample * mask_sample).view(-1)

                # Calculate dice score for this tumor sample
                intersection = (pred_flat * target_flat).sum()
                pred_sum = pred_flat.sum()
                target_sum_flat = target_flat.sum()

                if pred_sum + target_sum_flat > 0:
                    dice_sample = (2.0 * intersection) / (
                        pred_sum + target_sum_flat + 1e-8
                    )
                else:
                    dice_sample = torch.tensor(0.0, device=tumor_pred_binary.device)

                tumor_dice_scores.append(dice_sample)

        # Return mean dice for tumor samples only, or 0 if no tumor samples
        if len(tumor_dice_scores) > 0:
            return torch.stack(tumor_dice_scores).mean()
        else:
            return torch.tensor(0.0, device=tumor_pred_binary.device)

    def _calculate_and_log_metrics(self, pred, target, prefix: str):
        """
        Override base model metrics calculation to use threshold-aware loss.

        For OrganDetector, we only want to calculate loss and metrics on pixels
        below the intensity threshold, matching the training data logic.
        """
        # Get the original images from the current batch
        # Note: This assumes we have access to the original images
        # We'll need to modify the training step to pass images

        # For now, let's use the base implementation but add threshold masking
        # We'll need to get the images from the training step
        return super()._calculate_and_log_metrics(pred, target, prefix)

    def training_step(self, batch, batch_idx):
        """Override training step to use optimized weighted BCE loss"""
        images, masks = batch
        pred = self(images)

        # Use optimized weighted BCE loss for tumor detection
        loss = self._calculate_optimized_loss(pred, masks, images)

        # Calculate metrics optimized for tumor detection task
        with torch.no_grad():
            # Organ detection metrics (standard)
            pred_binary = (pred > 0.5).float()
            organ_dice_score = self._calculate_dice_score(pred_binary, masks)
            organ_patient_dice = self._calculate_patient_dice_score(pred_binary, masks)

            # TUMOR detection metrics (inverted predictions for final task)
            # Since pred = organ probability, tumor_prob = 1 - organ_prob
            tumor_pred = 1.0 - pred  # Tumor probabilities
            tumor_pred_binary = (tumor_pred > 0.5).float()  # Tumor predictions

            # Tumor targets: 1 where target=0 (non-organ pixels in dark areas)
            images_255 = (images * 255.0).clamp(0, 255)
            threshold_mask = (images_255 < self.intensity_threshold).float()
            tumor_targets = (1.0 - masks) * threshold_mask  # Invert organ targets

            # Calculate tumor detection dice score (the final task metric!)
            tumor_dice_score = self._calculate_dice_score(
                tumor_pred_binary, tumor_targets
            )
            tumor_patient_dice = self._calculate_patient_dice_score(
                tumor_pred_binary, tumor_targets
            )

            # Calculate naive dice: ONLY on tiles with tumors (meaningful tumor detection)
            # This directly measures tumor detection performance for wandb logging
            naive_dice_score = self._calculate_naive_dice_tumor_only(
                tumor_pred_binary, tumor_targets, threshold_mask
            )

            # Threshold-specific metrics
            threshold_pixels = threshold_mask.sum()
            threshold_organ_predictions = (pred_binary * threshold_mask).sum()
            threshold_organ_targets = (masks * threshold_mask).sum()
            threshold_tumor_predictions = (tumor_pred_binary * threshold_mask).sum()
            threshold_tumor_targets = (tumor_targets * threshold_mask).sum()

        # Log metrics - prioritize TUMOR detection metrics (final task)
        batch_size = masks.size(0)

        # Primary loss
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # 🎯 TUMOR DETECTION METRICS (FINAL TASK) - Most Important!
        self.log(
            "train_tumor_dice",
            tumor_dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,  # Show in progress bar
            batch_size=batch_size,
        )
        self.log(
            "train_tumor_dice_patients",
            tumor_patient_dice,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Organ detection metrics (intermediate task)
        self.log(
            "train_organ_dice",
            organ_dice_score,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_organ_dice_patients",
            organ_patient_dice,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Threshold analysis metrics
        self.log(
            "train_threshold_pixels",
            threshold_pixels,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_tumor_predictions",
            threshold_tumor_predictions,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_tumor_targets",
            threshold_tumor_targets,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Log naive dice (tumor detection under threshold) for wandb
        self.log(
            "train_naive_dice",
            naive_dice_score,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Override validation step to use optimized weighted BCE loss"""
        images, masks = batch
        pred = self(images)

        # Use optimized weighted BCE loss for tumor detection
        loss = self._calculate_optimized_loss(pred, masks, images)

        # Calculate metrics optimized for tumor detection task
        with torch.no_grad():
            # Organ detection metrics (standard)
            pred_binary = (pred > 0.5).float()
            organ_dice_score = self._calculate_dice_score(pred_binary, masks)
            organ_patient_dice = self._calculate_patient_dice_score(pred_binary, masks)

            # TUMOR detection metrics (inverted predictions for final task)
            tumor_pred = 1.0 - pred  # Tumor probabilities
            tumor_pred_binary = (tumor_pred > 0.5).float()  # Tumor predictions

            # Tumor targets: 1 where target=0 (non-organ pixels in dark areas)
            images_255 = (images * 255.0).clamp(0, 255)
            threshold_mask = (images_255 < self.intensity_threshold).float()
            tumor_targets = (1.0 - masks) * threshold_mask  # Invert organ targets

            # Calculate tumor detection dice score (the final task metric!)
            tumor_dice_score = self._calculate_dice_score(
                tumor_pred_binary, tumor_targets
            )
            tumor_patient_dice = self._calculate_patient_dice_score(
                tumor_pred_binary, tumor_targets
            )

            # Calculate naive dice: ONLY on tiles with tumors (meaningful tumor detection)
            # This directly measures tumor detection performance for wandb logging
            naive_dice_score = self._calculate_naive_dice_tumor_only(
                tumor_pred_binary, tumor_targets, threshold_mask
            )

            # Threshold-specific metrics
            threshold_pixels = threshold_mask.sum()
            threshold_organ_predictions = (pred_binary * threshold_mask).sum()
            threshold_organ_targets = (masks * threshold_mask).sum()
            threshold_tumor_predictions = (tumor_pred_binary * threshold_mask).sum()
            threshold_tumor_targets = (tumor_targets * threshold_mask).sum()

        # Log metrics - prioritize TUMOR detection metrics (final task)
        batch_size = masks.size(0)

        # Primary loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # 🎯 TUMOR DETECTION METRICS (FINAL TASK) - Most Important!
        self.log(
            "val_tumor_dice",
            tumor_dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,  # Show in progress bar
            batch_size=batch_size,
        )
        self.log(
            "val_tumor_dice_patients",
            tumor_patient_dice,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Organ detection metrics (intermediate task)
        self.log(
            "val_organ_dice",
            organ_dice_score,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_organ_dice_patients",
            organ_patient_dice,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Threshold analysis metrics
        self.log(
            "val_threshold_pixels",
            threshold_pixels,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_tumor_predictions",
            threshold_tumor_predictions,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_tumor_targets",
            threshold_tumor_targets,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Log naive dice (tumor detection under threshold) for wandb
        self.log(
            "val_naive_dice",
            naive_dice_score,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for OrganDetector.

        Override base model to monitor val_naive_dice instead of val_dice.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get("weight_decay", 1e-5),
        )

        # Learning rate scheduler - monitor naive dice (tumor detection performance)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Monitor naive dice score (higher is better)
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_naive_dice",  # Monitor tumor detection performance
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def test_tiled_input(self, width: int = 512, batch_size: int = 4):
        """
        Test the model with tiled input dimensions to verify compatibility.

        Args:
            width: Width of the tile (variable, default 512)
            batch_size: Batch size for testing

        Returns:
            bool: True if model works with tiled inputs
        """
        try:
            # Create sample tiled input
            x = torch.randn(batch_size, 1, self.tile_height, width)

            # Test forward pass
            with torch.no_grad():
                output = self.forward(x)

            # Verify output shape
            expected_shape = (batch_size, 1, self.tile_height, width)
            assert output.shape == expected_shape, (
                f"Output shape {output.shape} != expected {expected_shape}"
            )

            print("✅ Model successfully processes tiled input:")
            print(f"   Input: {x.shape} -> Output: {output.shape}")
            print(f"   Tile height: {self.tile_height}px, Width: {width}px")

            return True

        except Exception as e:
            print(f"❌ Model failed with tiled input: {e}")
            return False
