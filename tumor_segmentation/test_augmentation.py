#!/usr/bin/env python3
"""
Script to test and visualize data augmentation pipeline.
Shows original images/masks alongside augmented versions.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from data.data_default import get_transforms, get_standard_transforms
import random


def load_sample_data(data_dir="data"):
    """Load a random sample image and mask for testing"""
    data_path = Path(data_dir)

    # Find patient images and corresponding masks
    patient_img_dir = data_path / "patients" / "imgs"
    patient_mask_dir = data_path / "patients" / "labels"

    patient_images = sorted(list(patient_img_dir.glob("*.png")))

    if not patient_images:
        raise FileNotFoundError(f"No images found in {patient_img_dir}")

    # Pick a random image
    img_path = random.choice(patient_images)

    # Find corresponding mask
    img_name = img_path.stem
    mask_name = img_name.replace("patient_", "segmentation_") + ".png"
    mask_path = patient_mask_dir / mask_name

    if not mask_path.exists():
        raise FileNotFoundError(f"No corresponding mask found: {mask_path}")

    print(f"Loading sample: {img_path.name}")
    print(f"Corresponding mask: {mask_path.name}")

    # Load image and mask
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)  # Binary mask

    return image, mask, img_path.name


def apply_transforms(image, mask, transform):
    """Apply transforms and return processed image/mask"""
    augmented = transform(image=image, mask=mask)
    aug_image = augmented["image"]
    aug_mask = augmented["mask"]

    return aug_image, aug_mask


def tensor_to_display(tensor):
    """Convert tensor to displayable numpy array"""
    if torch.is_tensor(tensor):
        # Remove channel dimension and convert to numpy
        if len(tensor.shape) == 3:  # [C, H, W]
            tensor = tensor.squeeze(0)  # Remove channel dimension
        return tensor.cpu().numpy()
    return tensor


def visualize_augmentation_comparison(image, mask, filename, num_examples=3):
    """
    Visualize original vs augmented images and masks

    Args:
        image: Original image (numpy array)
        mask: Original mask (numpy array)
        filename: Name of the sample file
        num_examples: Number of augmented examples to show
    """

    # Get transforms
    standard_transform = get_standard_transforms(image_size=256)
    augmentation_transform = get_transforms(augmentation=True, image_size=256)

    # Apply standard transform (no augmentation)
    std_image, std_mask = apply_transforms(image, mask, standard_transform)

    # Create the plot
    fig, axes = plt.subplots(2, num_examples + 1, figsize=(4 * (num_examples + 1), 8))
    fig.suptitle(f"Data Augmentation Test: {filename}", fontsize=16, fontweight="bold")

    # Column 0: Original (with standard preprocessing)
    axes[0, 0].imshow(tensor_to_display(std_image), cmap="gray")
    axes[0, 0].set_title("Original Image\n(Standard Transform)", fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(tensor_to_display(std_mask), cmap="gray")
    axes[1, 0].set_title("Original Mask\n(Standard Transform)", fontweight="bold")
    axes[1, 0].axis("off")

    # Columns 1+: Augmented examples
    for i in range(num_examples):
        # Apply augmentation
        aug_image, aug_mask = apply_transforms(image, mask, augmentation_transform)

        # Display augmented image
        axes[0, i + 1].imshow(tensor_to_display(aug_image), cmap="gray")
        axes[0, i + 1].set_title(f"Augmented Image #{i + 1}", fontweight="bold")
        axes[0, i + 1].axis("off")

        # Display augmented mask
        axes[1, i + 1].imshow(tensor_to_display(aug_mask), cmap="gray")
        axes[1, i + 1].set_title(f"Augmented Mask #{i + 1}", fontweight="bold")
        axes[1, i + 1].axis("off")

    plt.tight_layout()

    # Save the plot instead of showing it
    output_path = f"augmentation_test_{filename.replace('.png', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    print(f"Visualization saved as: {output_path}")

    # Print some statistics
    print(f"\nImage Statistics:")
    print(f"Original shape: {image.shape}")
    print(f"Standard transform shape: {tensor_to_display(std_image).shape}")
    print(
        f"Standard transform range: [{tensor_to_display(std_image).min():.3f}, {tensor_to_display(std_image).max():.3f}]"
    )

    print(f"\nMask Statistics:")
    print(f"Original mask unique values: {np.unique(mask)}")
    print(
        f"Standard transform mask unique values: {np.unique(tensor_to_display(std_mask))}"
    )
    print(
        f"Mask tumor coverage: {tensor_to_display(std_mask).mean():.3f} (0=no tumor, 1=all tumor)"
    )


def test_multiple_samples(data_dir="data", num_samples=3):
    """Test augmentation on multiple random samples"""
    print("Testing data augmentation pipeline...")
    print("=" * 50)

    for i in range(num_samples):
        print(f"\nSample {i + 1}/{num_samples}")
        print("-" * 30)

        try:
            # Load random sample
            image, mask, filename = load_sample_data(data_dir)

            # Show augmentation comparison
            visualize_augmentation_comparison(image, mask, filename, num_examples=3)

            # Add separator between samples
            if i < num_samples - 1:
                print("\n" + "=" * 30 + "\n")

        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}")
            continue


def main():
    """Main function to run augmentation tests"""

    print("Data Augmentation Testing Script")
    print("=" * 40)
    print("This script will show you:")
    print("- Original images and masks (with standard preprocessing)")
    print("- Multiple augmented versions")
    print("- Statistics about the transformations")
    print("- Visualizations saved as PNG files")
    print()

    # Check if data directory exists
    if not Path("data").exists():
        print("ERROR: 'data' directory not found!")
        print(
            "Make sure you're running this script from the tumor_segmentation directory."
        )
        return

    # Test with multiple samples
    try:
        test_multiple_samples(data_dir="data", num_samples=2)
        print("\n" + "=" * 50)
        print("Augmentation testing completed!")
        print("Check the generated PNG files to review the augmentations.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Images in data/patients/imgs/")
        print("2. Masks in data/patients/labels/")
        print("3. Required packages: opencv-python, matplotlib, albumentations")


if __name__ == "__main__":
    main()
