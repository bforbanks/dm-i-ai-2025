#!/usr/bin/env python3
"""
Tiled inference script for OrganDetector model.

Splits images into horizontal tiles, runs inference, and reassembles predictions.
"""

import torch
import numpy as np
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.OrganDetector.model import OrganDetector


def split_image_into_tiles(
    image: np.ndarray, tile_height: int = 128
) -> List[np.ndarray]:
    """
    Split an image into horizontal tiles of specified height.

    Args:
        image: Input image [H, W] grayscale
        tile_height: Height of each tile

    Returns:
        List of tile images, each with shape [tile_height, W]
    """
    height, width = image.shape
    tiles = []

    y = 0
    while y < height:
        tile_start_y = y
        tile_end_y = min(y + tile_height, height)
        actual_tile_height = tile_end_y - tile_start_y

        # Extract tile
        tile = image[tile_start_y:tile_end_y, :]

        # Pad with white if needed
        if actual_tile_height < tile_height:
            padding_height = tile_height - actual_tile_height
            tile = np.pad(
                tile,
                ((0, padding_height), (0, 0)),
                mode="constant",
                constant_values=255,
            )

        tiles.append(tile)
        y += tile_height

    return tiles


def reassemble_tiles(
    tile_predictions: List[np.ndarray], original_height: int, tile_height: int = 128
) -> np.ndarray:
    """
    Reassemble tile predictions back into full image.

    Args:
        tile_predictions: List of prediction tiles [tile_height, W]
        original_height: Original image height
        tile_height: Height of each tile

    Returns:
        Reassembled prediction [original_height, W]
    """
    if not tile_predictions:
        return np.array([])

    width = tile_predictions[0].shape[1]
    full_prediction = np.zeros((original_height, width), dtype=np.float32)

    y = 0
    for tile_idx, tile_pred in enumerate(tile_predictions):
        tile_start_y = y
        tile_end_y = min(y + tile_height, original_height)
        actual_tile_height = tile_end_y - tile_start_y

        # Take only the relevant part of the tile (remove padding if any)
        tile_pred_cropped = tile_pred[:actual_tile_height, :]

        # Place in full prediction
        full_prediction[tile_start_y:tile_end_y, :] = tile_pred_cropped

        y += tile_height

    return full_prediction


def run_tiled_inference(
    model: OrganDetector,
    image: np.ndarray,
    tile_height: int = 128,
    intensity_threshold: int = 85,
    return_tumor_probabilities: bool = True,
) -> np.ndarray:
    """
    Run inference on an image using tiled approach with threshold masking.

    Args:
        model: Trained OrganDetector model (outputs organ probabilities)
        image: Input image [H, W] grayscale
        tile_height: Height of tiles
        intensity_threshold: Intensity threshold used during training
        return_tumor_probabilities: If True, return tumor probabilities (1 - organ_prob)
                                   If False, return organ probabilities

    Returns:
        Full prediction [H, W] with values in [0, 1], masked for threshold
        If return_tumor_probabilities=True: tumor probabilities for final task
        If return_tumor_probabilities=False: organ probabilities (intermediate)
    """
    original_height, width = image.shape

    # Split into tiles
    tiles = split_image_into_tiles(image, tile_height)

    print(
        f"Split {original_height}x{width} image into {len(tiles)} tiles of {tile_height}px height"
    )

    # Run inference on each tile
    tile_predictions = []
    model.eval()

    with torch.no_grad():
        for i, tile in enumerate(tiles):
            # Normalize to [0,1] range like training data
            tile_normalized = tile.astype(np.float32) / 255.0

            # Convert to tensor [1, 1, H, W]
            tile_tensor = torch.from_numpy(tile_normalized).unsqueeze(0).unsqueeze(0)

            # Run inference
            prediction = model(tile_tensor)  # [1, 1, H, W]

            # Convert back to numpy [H, W]
            pred_np = prediction.squeeze().cpu().numpy()

            # CRITICAL: Apply threshold masking - ONLY predict on dark pixels
            # This matches the training logic where targets were only created for pixels < threshold
            threshold_mask = tile < intensity_threshold

            # Ensure absolute zero predictions in bright areas
            pred_masked = np.where(threshold_mask, pred_np, 0.0)

            tile_predictions.append(pred_masked)

            if i == 0:
                print(
                    f"Tile {i + 1}: Input {tile_tensor.shape} -> Output {prediction.shape}"
                )

                # Debug info for first tile
                bright_pixels = np.sum(~threshold_mask)
                dark_pixels = np.sum(threshold_mask)
                total_pixels = tile.size
                pred_in_bright = (
                    pred_np[~threshold_mask].sum() if bright_pixels > 0 else 0
                )
                pred_in_dark = pred_np[threshold_mask].sum() if dark_pixels > 0 else 0

                print("  First tile threshold analysis:")
                print(f"    Total pixels: {total_pixels}")
                print(
                    f"    Bright pixels (>={intensity_threshold}): {bright_pixels} ({bright_pixels / total_pixels * 100:.1f}%)"
                )
                print(
                    f"    Dark pixels (<{intensity_threshold}): {dark_pixels} ({dark_pixels / total_pixels * 100:.1f}%)"
                )
                print(f"    Original pred in bright areas: {pred_in_bright:.4f}")
                print(f"    Original pred in dark areas: {pred_in_dark:.4f}")
                print(
                    f"    ⚠️  Masked pred in bright areas: {pred_masked[~threshold_mask].sum():.4f} (MUST be 0.0)"
                )
                print(
                    f"    ✅ Masked pred in dark areas: {pred_masked[threshold_mask].sum():.4f}"
                )

    # Reassemble predictions
    full_prediction = reassemble_tiles(tile_predictions, original_height, tile_height)

    # Apply final threshold mask to full prediction for absolute consistency
    # This is redundant if tile masking worked correctly, but ensures no bright-area predictions
    full_threshold_mask = image < intensity_threshold
    full_prediction = np.where(full_threshold_mask, full_prediction, 0.0)

    # Verify masking worked correctly
    bright_areas = ~full_threshold_mask
    predictions_in_bright = (
        full_prediction[bright_areas].sum() if bright_areas.any() else 0.0
    )
    if predictions_in_bright > 1e-10:  # More strict threshold
        print(
            f"⚠️  WARNING: Found {predictions_in_bright:.6f} prediction values in bright areas!"
        )
        print("    This should be exactly 0.0 - threshold masking may have failed!")
    else:
        print("✅ Threshold masking verified: No predictions in bright areas")

    # Convert to tumor probabilities if requested (for final tumor detection task)
    if return_tumor_probabilities:
        # Model outputs organ probabilities, convert to tumor probabilities
        full_prediction = 1.0 - full_prediction
        # Re-apply threshold mask to ensure consistency
        full_prediction = np.where(full_threshold_mask, full_prediction, 0.0)
        prediction_type = "tumor"
    else:
        prediction_type = "organ"

    print(f"Reassembled {prediction_type} prediction: {full_prediction.shape}")
    print(
        f"Prediction range: {full_prediction.min():.3f} - {full_prediction.max():.3f}"
    )

    # Calculate and print threshold statistics
    total_dark_pixels = full_threshold_mask.sum()
    total_pixels = image.size
    predicted_pixels = (
        full_prediction > 0.1
    ).sum()  # Count pixels with meaningful predictions

    print("Threshold analysis:")
    print(
        f"  Dark pixels (<{intensity_threshold}): {total_dark_pixels:,}/{total_pixels:,} ({total_dark_pixels / total_pixels * 100:.1f}%)"
    )
    print(
        f"  Predicted {prediction_type} pixels: {predicted_pixels:,} ({predicted_pixels / max(total_dark_pixels, 1) * 100:.1f}% of dark pixels)"
    )

    return full_prediction


def plot_organ_prediction(
    image, true_mask, prediction, see_organs=True, intensity_threshold=85
):
    """
    Enhanced visualization function for organ detection.

    Args:
        image: Original image [H, W, 3] in 0-255 range
        true_mask: True segmentation [H, W, 3] (not used for organ detection)
        prediction: Model prediction [H, W, 3] in 0-255 range
        see_organs: If True, show organ predictions (green). If False, show tumor predictions (red)
        intensity_threshold: Threshold used to define dark pixels
    """
    # Convert to single channel if needed
    if len(image.shape) == 3:
        img_2d = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_2d = image.copy()

    if len(prediction.shape) == 3:
        pred_2d = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
    else:
        pred_2d = prediction.copy()

    # Calculate some statistics
    dark_pixels = (img_2d < intensity_threshold).sum()
    predicted_pixels = (pred_2d > 127).sum()  # Assuming binary prediction
    total_pixels = img_2d.size

    if see_organs:
        title_suffix = "Organ Detection (Non-tumor dark pixels)"
        color_info = "Green = Predicted organs (dark non-tumor areas)"
        overlay_color = [0, 255, 0]  # Green for organs
    else:
        title_suffix = "Tumor Detection"
        color_info = "Red = Predicted tumors"
        overlay_color = [255, 0, 0]  # Red for tumors

    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.axis("off")
    plt.title("Original Image")

    # Dark pixel visualization
    plt.subplot(1, 4, 2)
    dark_overlay = np.zeros((*img_2d.shape, 3), dtype=np.uint8)
    dark_mask = img_2d < intensity_threshold
    dark_overlay[dark_mask] = [100, 100, 255]  # Light blue for all dark pixels

    # Blend with original
    img_3ch = np.stack([img_2d, img_2d, img_2d], axis=2)
    blended = cv2.addWeighted(img_3ch, 0.7, dark_overlay, 0.3, 0)

    plt.imshow(blended)
    plt.axis("off")
    plt.title(
        f"Dark Pixels (<{intensity_threshold})\n{dark_pixels:,} / {total_pixels:,} pixels"
    )

    # Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(prediction, cmap="gray" if len(prediction.shape) == 2 else None)
    plt.axis("off")
    plt.title(f"Prediction\n{predicted_pixels:,} pixels")

    # Overlay visualization
    plt.subplot(1, 4, 4)
    overlay = np.zeros((*img_2d.shape, 3), dtype=np.uint8)

    # Create prediction mask
    pred_mask = pred_2d > 127
    overlay[pred_mask] = overlay_color

    # Blend with original
    blended_pred = cv2.addWeighted(img_3ch, 0.7, overlay, 0.3, 0)

    plt.imshow(blended_pred)
    plt.axis("off")
    plt.title(f"{title_suffix}\n{color_info}")

    # Add statistics as text
    plt.figtext(
        0.02,
        0.02,
        f"Stats: {dark_pixels:,} dark pixels, {predicted_pixels:,} predicted pixels\n"
        f"Dark pixel ratio: {dark_pixels / total_pixels * 100:.1f}%, "
        f"Prediction ratio: {predicted_pixels / total_pixels * 100:.1f}%",
        fontsize=10,
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.show()


def load_test_image(image_path: str) -> Tuple[np.ndarray, str]:
    """
    Load a test image for inference.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (image array [H, W], image_type)
    """
    # Load image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Determine image type from filename
    filename = Path(image_path).name.lower()
    if "patient" in filename:
        image_type = "patient"
    elif "control" in filename:
        image_type = "control"
    else:
        image_type = "unknown"

    print(f"Loaded {image_type} image: {image.shape} from {Path(image_path).name}")

    return image, image_type


def demo_tiled_inference(
    model_checkpoint: str,
    image_path: str = None,
    see_organs: bool = True,
    tile_height: int = 128,
    intensity_threshold: int = 85,
):
    """
    Demo function for tiled organ detection inference.

    Args:
        model_checkpoint: Path to trained OrganDetector checkpoint
        image_path: Path to test image (if None, uses first patient image)
        see_organs: If True, visualize organ predictions. If False, visualize as tumor predictions
        tile_height: Height of tiles for inference
        intensity_threshold: Intensity threshold used during training
    """
    print("🧠 Tiled Organ Detection Inference Demo")
    print("=" * 50)

    # Load model
    print(f"Loading model from: {model_checkpoint}")
    model = OrganDetector.load_from_checkpoint(
        model_checkpoint, map_location="cpu", tile_height=tile_height
    )
    model.eval()

    print("✅ Model loaded successfully")

    # Load test image
    if image_path is None:
        # Use first patient image as default
        data_dir = Path("data")
        patient_imgs = sorted(list((data_dir / "patients" / "imgs").glob("*.png")))
        if not patient_imgs:
            raise ValueError("No patient images found in data/patients/imgs/")
        image_path = str(patient_imgs[0])

    image, image_type = load_test_image(image_path)

    # Run tiled inference
    print("\n🔬 Running tiled inference...")
    prediction = run_tiled_inference(
        model, image, tile_height=tile_height, intensity_threshold=intensity_threshold
    )

    # Convert for visualization
    # Image: grayscale to RGB [H, W, 3]
    img_rgb = np.stack([image, image, image], axis=2)

    # Prediction: probability to binary [H, W, 3]
    pred_binary = (prediction > 0.5).astype(np.uint8) * 255
    pred_rgb = np.stack([pred_binary, pred_binary, pred_binary], axis=2)

    # Create dummy true mask (not used for organ detection)
    true_mask_rgb = np.zeros_like(img_rgb)

    # Visualize results
    print(f"\n📊 Visualization (see_organs={see_organs})...")
    plot_organ_prediction(
        img_rgb,
        true_mask_rgb,
        pred_rgb,
        see_organs=see_organs,
        intensity_threshold=intensity_threshold,
    )

    # Print summary
    dark_pixels = (image < intensity_threshold).sum()
    predicted_pixels = (prediction > 0.5).sum()
    total_pixels = image.size

    print(f"\n📈 Results Summary:")
    print(f"  Image type: {image_type}")
    print(f"  Image size: {image.shape}")
    print(
        f"  Dark pixels (<{intensity_threshold}): {dark_pixels:,} ({dark_pixels / total_pixels * 100:.1f}%)"
    )
    print(
        f"  Predicted {'organ' if see_organs else 'tumor'} pixels: {predicted_pixels:,} ({predicted_pixels / total_pixels * 100:.1f}%)"
    )

    if see_organs:
        coverage = predicted_pixels / max(dark_pixels, 1) * 100
        print(f"  Organ coverage of dark pixels: {coverage:.1f}%")

    return prediction


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Tiled Organ Detection Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--image", type=str, help="Path to test image (optional)")
    parser.add_argument(
        "--see-organs",
        action="store_true",
        default=True,
        help="Visualize organ predictions",
    )
    parser.add_argument(
        "--see-tumors", action="store_true", help="Visualize as tumor predictions"
    )
    parser.add_argument("--tile-height", type=int, default=128, help="Tile height")
    parser.add_argument(
        "--intensity-threshold", type=int, default=85, help="Intensity threshold"
    )

    args = parser.parse_args()

    # Handle conflicting flags
    see_organs = args.see_organs and not args.see_tumors

    try:
        prediction = demo_tiled_inference(
            model_checkpoint=args.checkpoint,
            image_path=args.image,
            see_organs=see_organs,
            tile_height=args.tile_height,
            intensity_threshold=args.intensity_threshold,
        )
        print("\n🎉 Inference completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        sys.exit(1)
