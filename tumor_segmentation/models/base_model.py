import torch
import lightning.pytorch as pl
import numpy as np
import cv2


class BaseModel(pl.LightningModule):
    """
    Base class for tumor segmentation models with common training/validation/logging logic

    Key design decisions:
    - Dice loss for segmentation (better than BCE for imbalanced data)
    - Dice score as primary metric (same as competition evaluation)
    - Per-epoch logging for training curves
    - Support for both patient and control data
    """

    def __init__(self, lr=1e-3, weight_decay=1e-5, bce_loss_weight=0.5, **kwargs):
        super().__init__()

        self.bce_loss_weight = bce_loss_weight
        self.save_hyperparameters()

        # Initialize loss function - we'll use the _calculate_dice_loss method
        self.smooth = 1e-6

    def _calculate_and_log_metrics(self, pred, target, prefix: str):
        """
        Calculate and log segmentation metrics consistently.

        Args:
            pred: Predicted segmentation [B, 1, H, W] (sigmoid output)
            target: Target segmentation [B, 1, H, W] (binary)
            prefix: Prefix for logging ('train' or 'val')
        """
        # Convert predictions to binary for Dice score calculation
        pred_binary = (pred > 0.5).float()

        # Calculate BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy(pred, target)

        # Calculate Dice score using custom implementation (matches utils.py)
        dice_score = self._calculate_dice_score(pred_binary, target)

        # print("bce_loss", self.bce_loss)

        # Calculate loss using raw sigmoid predictions
        loss = bce_loss * self.bce_loss_weight + (
            1 - self.bce_loss_weight
        ) * self._calculate_dice_loss(pred, target)

        # Log metrics
        batch_size = target.size(0)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.log(
            f"{prefix}_dice",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def _calculate_dice_per_sample(self, pred, target, smooth=1e-6):
        """
        Calculate Dice coefficient per sample in the batch.
        This is the core function used by both loss and metric calculations.

        Args:
            pred: Predictions [B, 1, H, W] (can be sigmoid or binary)
            target: Binary targets [B, 1, H, W] (values 0 or 1)
            smooth: Smoothing factor for numerical stability

        Returns:
            List of dice scores for each sample in the batch
        """
        batch_size = pred.size(0)
        dice_scores = []

        # Calculate dice score for each sample in the batch
        for i in range(batch_size):
            # Get single sample [1, H, W]
            pred_sample = pred[i]  # [1, H, W]
            target_sample = target[i]  # [1, H, W]

            # Flatten predictions and targets for this sample
            pred_flat = pred_sample.view(-1)
            target_flat = target_sample.view(-1)

            # Calculate intersection and union
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()

            # Handle edge case where both masks are empty (perfect match)
            if union == 0:
                dice_sample = torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
            else:
                # Calculate Dice score: 2 * intersection / union
                dice_sample = (2.0 * intersection + smooth) / (union + smooth)

            dice_scores.append(dice_sample)

        return dice_scores

    def _calculate_dice_score(self, pred_binary, target):
        """
        Calculate Dice score using binary predictions (for metrics).

        Args:
            pred_binary: Binary predictions [B, 1, H, W] (values 0 or 1)
            target: Binary targets [B, 1, H, W] (values 0 or 1)

        Returns:
            Mean Dice score as tensor
        """
        dice_scores = self._calculate_dice_per_sample(pred_binary, target, smooth=0)
        return torch.stack(dice_scores).mean()

    def _calculate_dice_loss(self, pred, target):
        """
        Calculate Dice loss using sigmoid predictions.

        Args:
            pred: Model predictions [B, 1, H, W] (sigmoid output, values 0-1)
            target: Ground truth [B, 1, H, W] (binary, values 0 or 1)

        Returns:
            Mean Dice loss (1 - Dice coefficient) across the batch
        """
        # Ensure target shape matches prediction shape
        if pred.shape != target.shape:
            # If target has extra dimensions, squeeze them
            while len(target.shape) > len(pred.shape):
                target = target.squeeze(-1)
            # If target needs channel dimension, add it
            if len(target.shape) == 3 and len(pred.shape) == 4:
                target = target.unsqueeze(1)

        # Use the shared dice calculation function
        dice_scores = self._calculate_dice_per_sample(pred, target, smooth=self.smooth)

        # Convert dice scores to losses (1 - dice)
        dice_losses = [1 - dice for dice in dice_scores]

        # Return mean dice loss across the batch
        return torch.stack(dice_losses).mean()

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, masks = batch
        pred = self(images)
        return self._calculate_and_log_metrics(pred, masks, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, masks = batch
        pred = self(images)
        return self._calculate_and_log_metrics(pred, masks, "val")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Monitor dice score (higher is better)
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Inference method for segmentation.

        Args:
            img: Input image as numpy array (H, W, C) where C can be 1 or 3

        Returns:
            Segmentation mask as grayscale numpy array (H, W) with values 0-255
        """
        # Store original shape for resizing back
        original_shape = img.shape[:2]

        # Preprocess image
        img_tensor = self.preprocess_image(img)

        # Ensure tensor is on same device as model
        img_tensor = img_tensor.to(self.device)

        # Run inference
        self.eval()
        with torch.no_grad():
            output = self(img_tensor)  # Forward pass

        # Postprocess output
        segmentation = self.postprocess_segmentation(output, original_shape)

        return segmentation

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            img: Input image (H, W, C) where C can be 1 or 3

        Returns:
            Preprocessed tensor (1, 1, H, W) - PyTorch standard format (single channel)
        """
        # Handle different input formats
        if len(img.shape) == 3 and img.shape[2] == 1:
            # Grayscale with channel dimension (H, W, 1)
            img_2d = img.squeeze(2)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # RGB format (H, W, 3) - convert to grayscale
            img_2d = np.mean(img, axis=2)
        else:
            # Already 2D grayscale (H, W)
            img_2d = img

        # Normalize to [0, 1]
        img_2d = img_2d.astype(np.float32) / 255.0

        # Resize to model input size (assuming 256x256)
        img_resized = cv2.resize(img_2d, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Convert to PyTorch standard format: (H, W) -> (1, H, W) -> (1, 1, H, W)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)

        return img_tensor

    def postprocess_segmentation(
        self, output: torch.Tensor, original_shape: tuple
    ) -> np.ndarray:
        """
        Postprocess model output to final segmentation mask.

        Args:
            output: Model output tensor (1, 1, H, W)
            original_shape: Original image shape (H, W)

        Returns:
            Binary segmentation mask (H, W, 3) with values 0-255 (RGB format with identical channels)
        """
        # Apply sigmoid to get probabilities
        prob = output

        # Convert to numpy and remove batch/channel dimensions
        prob_np = prob.squeeze().cpu().numpy()  # (H, W)

        # Apply threshold to get binary mask
        binary_mask = (prob_np > 0.5).astype(np.uint8) * 255

        # Resize back to original dimensions if needed
        if binary_mask.shape != original_shape:
            binary_mask = cv2.resize(
                binary_mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Invert colors (black becomes white, white becomes black)
        # binary_mask = cv2.bitwise_not(binary_mask)

        # Convert to RGB format with identical channels (H, W, 3) for validation compatibility
        binary_mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)

        return binary_mask_rgb
