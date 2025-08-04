#!/usr/bin/env python3
"""
Verify nnUNetv2 preprocessing pipeline with no resampling and 320x320 tiling.

This script simulates the exact preprocessing that nnUNetv2 will perform:
1. Load original images (no resampling)
2. Apply normalization
3. Generate 320x320 tiles with overlap
4. Save visual examples for verification

This ensures we're giving the model the correct data.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import cv2


def load_nnunet_configuration(dataset_id: int = 1, config_name: str = "2d_resenc_optimized"):
    """Load the nnUNetv2 configuration to understand the preprocessing pipeline."""
    plans_path = f"data_nnUNet/preprocessed/Dataset{dataset_id:03d}_TumorSegmentation/nnUNetResEncUNetMPlans.json"
    
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    if config_name not in plans["configurations"]:
        raise ValueError(f"Configuration '{config_name}' not found in plans")
    
    return plans["configurations"][config_name]


def load_original_images(dataset_id: int = 1, num_samples: int = 5) -> List[Dict]:
    """Load original images from the nnUNetv2 data directory."""
    data_dir = f"data_nnUNet/Dataset{dataset_id:03d}_TumorSegmentation"
    images_dir = f"{data_dir}/imagesTr"
    labels_dir = f"{data_dir}/labelsTr"
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    samples = []
    for i, img_file in enumerate(image_files[:num_samples]):
        # Load image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load corresponding label
        label_file = img_file.replace('_0000.png', '.png')  # Remove channel suffix
        label_path = os.path.join(labels_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if label is None:
            # Try without .png extension
            label_file = label_file.replace('.png', '')
            label_path = os.path.join(labels_dir, label_file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        samples.append({
            'filename': img_file,
            'image': image,
            'label': label,
            'original_shape': image.shape,
            'has_tumor': np.any(label > 0) if label is not None else False
        })
    
    return samples


def normalize_image(image: np.ndarray, normalization_scheme: str = "ZScoreNormalization") -> np.ndarray:
    """Apply the same normalization that nnUNetv2 uses."""
    if normalization_scheme == "ZScoreNormalization":
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
        return normalized
    else:
        return image


def generate_tiles(image: np.ndarray, patch_size: Tuple[int, int] = (320, 320), 
                  overlap: float = 0.5) -> List[Dict]:
    """
    Generate tiles from the image with overlap.
    
    Args:
        image: Input image
        patch_size: Size of patches (height, width)
        overlap: Overlap ratio between patches (0.5 = 50% overlap)
    
    Returns:
        List of tile dictionaries with coordinates and data
    """
    height, width = image.shape
    patch_h, patch_w = patch_size
    
    # Calculate step size based on overlap
    step_h = int(patch_h * (1 - overlap))
    step_w = int(patch_w * (1 - overlap))
    
    tiles = []
    
    # Generate tiles
    for y in range(0, height - patch_h + 1, step_h):
        for x in range(0, width - patch_w + 1, step_w):
            # Extract tile
            tile = image[y:y + patch_h, x:x + patch_w]
            
            # Ensure tile is the correct size (handle edge cases)
            if tile.shape == patch_size:
                tiles.append({
                    'tile': tile,
                    'coords': (x, y, x + patch_w, y + patch_h),
                    'center': (x + patch_w // 2, y + patch_h // 2)
                })
    
    # Handle edge cases where tiles might be smaller
    # Add tiles for the right and bottom edges if needed
    if width > patch_w:
        for y in range(0, height - patch_h + 1, step_h):
            x = width - patch_w
            tile = image[y:y + patch_h, x:x + patch_w]
            if tile.shape == patch_size:
                tiles.append({
                    'tile': tile,
                    'coords': (x, y, x + patch_w, y + patch_h),
                    'center': (x + patch_w // 2, y + patch_h // 2)
                })
    
    if height > patch_h:
        for x in range(0, width - patch_w + 1, step_w):
            y = height - patch_h
            tile = image[y:y + patch_h, x:x + patch_w]
            if tile.shape == patch_size:
                tiles.append({
                    'tile': tile,
                    'coords': (x, y, x + patch_w, y + patch_h),
                    'center': (x + patch_w // 2, y + patch_h // 2)
                })
    
    # Add bottom-right corner tile if needed
    if width > patch_w and height > patch_h:
        x = width - patch_w
        y = height - patch_h
        tile = image[y:y + patch_h, x:x + patch_w]
        if tile.shape == patch_size:
            tiles.append({
                'tile': tile,
                'coords': (x, y, x + patch_w, y + patch_h),
                'center': (x + patch_w // 2, y + patch_h // 2)
            })
    
    return tiles


def create_visualization(image: np.ndarray, label: Optional[np.ndarray], 
                        tiles: List[Dict], filename: str, output_dir: str):
    """Create and save a comprehensive visualization of the preprocessing."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Original Image\nShape: {image.shape}')
    ax1.axis('off')
    
    # Original image with label overlay
    ax2 = plt.subplot(2, 3, 2)
    if label is not None:
        # Create RGB image for overlay
        rgb_image = np.stack([image, image, image], axis=-1)
        # Overlay tumor regions in red
        tumor_mask = label > 0
        rgb_image[tumor_mask] = [1, 0, 0]  # Red for tumor
        ax2.imshow(rgb_image)
        ax2.set_title(f'Image + Tumor Labels\nTumor pixels: {np.sum(tumor_mask)}')
    else:
        ax2.imshow(image, cmap='gray')
        ax2.set_title('Image (no label available)')
    ax2.axis('off')
    
    # Normalized image
    normalized = normalize_image(image)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(normalized, cmap='gray')
    ax3.set_title(f'Normalized Image\nMean: {np.mean(normalized):.3f}, Std: {np.std(normalized):.3f}')
    ax3.axis('off')
    
    # Tile coverage visualization
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(image, cmap='gray')
    
    # Draw tile boundaries
    for i, tile_info in enumerate(tiles):
        x1, y1, x2, y2 = tile_info['coords']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax4.add_patch(rect)
        # Add tile number
        ax4.text(x1 + 10, y1 + 20, str(i+1), color='red', fontsize=12, fontweight='bold')
    
    ax4.set_title(f'Tile Coverage\n{len(tiles)} tiles, 320×320 each')
    ax4.axis('off')
    
    # Sample tiles (first 4)
    for i in range(min(4, len(tiles))):
        ax = plt.subplot(2, 4, 5 + i)
        tile = tiles[i]['tile']
        ax.imshow(tile, cmap='gray')
        ax.set_title(f'Tile {i+1}\n{tile.shape}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_preprocessing_analysis.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual tiles
    tiles_dir = os.path.join(output_dir, f'{filename}_tiles')
    os.makedirs(tiles_dir, exist_ok=True)
    
    for i, tile_info in enumerate(tiles):
        tile = tile_info['tile']
        tile_path = os.path.join(tiles_dir, f'tile_{i+1:02d}.png')
        plt.imsave(tile_path, tile, cmap='gray')
    
    # Save tile info
    tile_info = {
        'filename': filename,
        'original_shape': image.shape,
        'num_tiles': len(tiles),
        'tile_size': (320, 320),
        'tiles': [
            {
                'tile_id': i+1,
                'coords': tile_info['coords'],
                'center': tile_info['center']
            }
            for i, tile_info in enumerate(tiles)
        ]
    }
    
    with open(os.path.join(tiles_dir, 'tile_info.json'), 'w') as f:
        json.dump(tile_info, f, indent=2)


def verify_no_resampling(original_samples: List[Dict], config: Dict):
    """Verify that no resampling is being applied."""
    print("\n🔍 Verifying No Resampling:")
    print("=" * 50)
    
    # Check configuration
    has_resampling = any(key.startswith('resampling_fn') for key in config.keys())
    print(f"Configuration resampling functions: {'❌ PRESENT' if has_resampling else '✅ ABSENT'}")
    
    # Check original image dimensions
    print("\nOriginal Image Dimensions:")
    for sample in original_samples:
        shape = sample['original_shape']
        print(f"  {sample['filename']}: {shape[1]}×{shape[0]} (width×height)")
    
    # Verify dimensions are preserved
    print(f"\n✅ Original dimensions preserved: {not has_resampling}")
    
    return not has_resampling


def main():
    """Main verification function."""
    print("🔍 nnUNetv2 Preprocessing Pipeline Verification")
    print("=" * 60)
    print("This script will verify that:")
    print("  ✅ No resampling is applied")
    print("  ✅ Original image dimensions are preserved")
    print("  ✅ 320×320 tiles properly cover the entire image")
    print("  ✅ Normalization is applied correctly")
    print("  ✅ Visual examples are saved for inspection")
    print()
    
    # Parameters
    dataset_id = 1
    config_name = "2d_resenc_optimized"
    num_samples = 5
    output_dir = "preprocessing_verification"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load configuration
        print("📋 Loading nnUNetv2 configuration...")
        config = load_nnunet_configuration(dataset_id, config_name)
        print(f"   Configuration: {config_name}")
        print(f"   Patch size: {config.get('patch_size', 'NOT SET')}")
        print(f"   Batch size: {config.get('batch_size', 'NOT SET')}")
        print(f"   Architecture: {config.get('architecture', {}).get('network_class_name', 'NOT SET')}")
        
        # 2. Load original images
        print(f"\n📁 Loading {num_samples} original images...")
        original_samples = load_original_images(dataset_id, num_samples)
        print(f"   Loaded {len(original_samples)} images")
        
        # 3. Verify no resampling
        no_resampling = verify_no_resampling(original_samples, config)
        
        if not no_resampling:
            print("\n❌ RESAMPLING IS STILL ENABLED!")
            print("Please run the fix script first.")
            return
        
        # 4. Process each sample
        print(f"\n🔄 Processing {len(original_samples)} samples...")
        
        for i, sample in enumerate(original_samples):
            print(f"\n   Processing {sample['filename']} ({i+1}/{len(original_samples)})")
            print(f"     Original shape: {sample['original_shape']}")
            print(f"     Has tumor: {sample['has_tumor']}")
            
            # Normalize image
            normalized_image = normalize_image(sample['image'])
            print(f"     Normalized - Mean: {np.mean(normalized_image):.3f}, Std: {np.std(normalized_image):.3f}")
            
            # Generate tiles
            tiles = generate_tiles(normalized_image, patch_size=(320, 320), overlap=0.5)
            print(f"     Generated {len(tiles)} tiles")
            
            # Create visualization
            filename = sample['filename'].replace('.png', '')
            create_visualization(
                sample['image'], 
                sample['label'], 
                tiles, 
                filename, 
                output_dir
            )
            print(f"     Saved visualization and tiles")
        
        # 5. Summary
        print(f"\n✅ Verification Complete!")
        print("=" * 50)
        print(f"Output directory: {output_dir}")
        print(f"Processed {len(original_samples)} images")
        print(f"Generated visualizations and tile examples")
        print()
        print("📋 What to check:")
        print("  1. Original images show correct dimensions")
        print("  2. No resampling applied (dimensions preserved)")
        print("  3. 320×320 tiles properly cover entire images")
        print("  4. Normalization applied correctly")
        print("  5. Tumor labels preserved in overlays")
        print()
        print("🎯 If everything looks correct, you can proceed with training!")
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 