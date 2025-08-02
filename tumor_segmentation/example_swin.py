#!/usr/bin/env python3
"""
Example script demonstrating the enhanced SwinUNet model for tumor segmentation.

This script shows how to:
1. Use the new SwinUNet model with Swin Transformer encoder
2. Train with attention mechanisms
3. Compare performance with the baseline SimpleUNet

Usage:
    python example_swin.py [--model swin_base|swin_tiny|simple]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import cv2
import os

# Import our models
from models.SimpleUNet.model import SimpleUNet
from dtos import PredictRequest

try:
    from models.SimpleUNet.model import SwinUNet

    SWIN_AVAILABLE = True
except ImportError as e:
    print(f"SwinUNet not available: {e}")
    print("Install timm with: pip install timm")
    SWIN_AVAILABLE = False


class TumorDataset(torch.utils.data.Dataset):
    """Simple dataset for demonstration purposes"""

    def __init__(self, data_dir="data/patients", subset="train"):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "imgs")
        self.mask_dir = os.path.join(data_dir, "masks")

        # Get list of image files
        if os.path.exists(self.image_dir):
            self.image_files = [
                f for f in os.listdir(self.image_dir) if f.endswith(".png")
            ]
        else:
            # Create dummy data for demonstration
            self.image_files = [f"dummy_{i:03d}.png" for i in range(100)]

        print(f"Found {len(self.image_files)} images for {subset}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # For real data, load actual images
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])

        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Load real data
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize to model input size
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            # Normalize
            image = image.astype(np.float32) / 255.0
            mask = (mask > 128).astype(np.float32)  # Binary mask

        else:
            # Create dummy data for demonstration
            image = np.random.rand(256, 256).astype(np.float32)
            mask = (np.random.rand(256, 256) > 0.9).astype(np.float32)  # Sparse tumors

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        return image, mask


def create_model(model_type="swin_base"):
    """Create and return the specified model"""

    if model_type == "simple":
        model = SimpleUNet(in_channels=1, num_classes=1, lr=1e-3, weight_decay=1e-5)
        print("Created SimpleUNet model")

    elif model_type == "swin_base" and SWIN_AVAILABLE:
        model = SwinUNet(
            in_channels=1,
            num_classes=1,
            lr=1e-4,
            weight_decay=1e-5,
            use_pretrained=True,
            model_name="swin_base_patch4_window7_224",
        )
        print("Created SwinUNet with Swin-Base encoder")

    elif model_type == "swin_tiny" and SWIN_AVAILABLE:
        model = SwinUNet(
            in_channels=1,
            num_classes=1,
            lr=1e-4,
            weight_decay=1e-5,
            use_pretrained=True,
            model_name="swin_tiny_patch4_window7_224",
        )
        print("Created SwinUNet with Swin-Tiny encoder")

    else:
        if not SWIN_AVAILABLE:
            print("Swin models not available, falling back to SimpleUNet")
            return create_model("simple")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(model, train_loader, val_loader, max_epochs=50):
    """Train the model using PyTorch Lightning"""

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_dice:.3f}",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_dice", mode="max", patience=10, verbose=True
    )

    # Logger
    logger = TensorBoardLogger("logs", name="tumor_segmentation")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        model = model.__class__.load_from_checkpoint(best_model_path)

    return model, trainer


def test_inference(model):
    """Test inference with dummy data"""
    print("\nTesting inference...")

    # Create dummy image data
    dummy_image = np.random.rand(400, 300).astype(np.uint8) * 255

    # Add some high-intensity "tumor" regions
    dummy_image[100:150, 100:150] = 200
    dummy_image[200:250, 180:230] = 220

    print(f"Input image shape: {dummy_image.shape}")

    # Test prediction
    try:
        segmentation = model.predict(dummy_image)
        print(f"Output segmentation shape: {segmentation.shape}")
        print(f"Segmentation value range: {segmentation.min()}-{segmentation.max()}")
        print("✓ Inference test passed!")

        # Count tumor pixels
        tumor_pixels = np.sum(segmentation[:, :, 0] > 128)
        total_pixels = segmentation.shape[0] * segmentation.shape[1]
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        print(f"Predicted tumor coverage: {tumor_percentage:.2f}%")

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

    return True


def compare_models():
    """Compare different model architectures"""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    models_to_test = ["simple"]
    if SWIN_AVAILABLE:
        models_to_test.extend(["swin_tiny", "swin_base"])

    for model_type in models_to_test:
        print(f"\n--- Testing {model_type.upper()} ---")

        try:
            model = create_model(model_type)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            # Test forward pass
            dummy_input = torch.randn(1, 1, 256, 256)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(
                f"Output range: {output.min().item():.3f} - {output.max().item():.3f}"
            )

            # Test inference method
            test_inference(model)

        except Exception as e:
            print(f"Failed to test {model_type}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train tumor segmentation model")
    parser.add_argument(
        "--model",
        choices=["simple", "swin_tiny", "swin_base"],
        default="swin_base",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--compare-only", action="store_true", help="Only compare models, don't train"
    )

    args = parser.parse_args()

    print("Tumor Segmentation with Enhanced Models")
    print("=" * 50)

    if args.compare_only:
        compare_models()
        return

    # Create dataset
    print("Creating dataset...")
    full_dataset = TumorDataset()

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(args.model)

    # Train model
    trained_model, trainer = train_model(model, train_loader, val_loader, args.epochs)

    # Final test
    print("\nRunning final inference test...")
    test_inference(trained_model)

    print("\nTraining completed!")
    print(f"Best model saved to: logs/tumor_segmentation/")
    print(
        "\nTo use the trained model in your API, update example.py to load the checkpoint:"
    )
    print("model = YourModel.load_from_checkpoint('path/to/checkpoint.ckpt')")


if __name__ == "__main__":
    main()
