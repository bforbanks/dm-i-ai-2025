#!/usr/bin/env python3
"""
Validation Set Analysis Script

This script processes all images in the validation set (data/full_val_set):
1. Loads each validation image
2. Makes predictions using the API
3. Creates side-by-side visualizations (original + mask)
4. Saves results to organized output folders

No dice score calculations are performed - this is purely for visualization.
"""

import requests
import base64
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tqdm import tqdm

def load_image(image_path):
    """Load image from file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image

def encode_image(image):
    """Encode image to base64 string"""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', image_rgb)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def decode_segmentation(segmentation_base64):
    """Decode base64 segmentation to numpy array"""
    segmentation_bytes = base64.b64decode(segmentation_base64)
    segmentation = cv2.imdecode(np.frombuffer(segmentation_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return segmentation

def predict_segmentation(image, api_url="http://localhost:9052"):
    """Make prediction using the API"""
    # Encode image
    image_base64 = encode_image(image)
    
    # Make request
    try:
        response = requests.post(
            f"{api_url}/predict",
            json={"img": image_base64},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            segmentation = decode_segmentation(result["img"])
            return segmentation
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        return None

def create_side_by_side_comparison(image, segmentation, output_path):
    """Create side-by-side comparison of original image and segmentation mask"""
    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Segmentation mask
    ax2.imshow(segmentation, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def get_validation_images(validation_dir):
    """Get list of all validation image files"""
    image_pattern = os.path.join(validation_dir, "image_*.png")
    image_files = sorted(glob.glob(image_pattern))
    return image_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze entire validation set with API predictions")
    parser.add_argument("--api-url", default="http://localhost:9052", 
                       help="API URL (default: http://localhost:9052)")
    parser.add_argument("--output-dir", default="validation_results", 
                       help="Output directory (default: validation_results)")
    parser.add_argument("--validation-dir", default="data/full_val_set", 
                       help="Validation directory (default: data/full_val_set)")
    
    args = parser.parse_args()
    
    # Get all validation images
    validation_images = get_validation_images(args.validation_dir)
    num_images = len(validation_images)
    
    print("🎯 Validation Set Analysis - API Predictions")
    print("=" * 60)
    print(f"Found {num_images} validation images in: {args.validation_dir}")
    print(f"API URL: {args.api_url}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_dir = os.path.join(args.output_dir, "comparison")
    mask_dir = os.path.join(args.output_dir, "mask")
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    successful_predictions = 0
    failed_predictions = 0
    
    try:
        # Create progress bar
        pbar = tqdm(validation_images, desc="Processing validation images", unit="image")
        
        for image_path in pbar:
            # Extract image name for output file
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]  # Remove .png extension
            
            # Update progress bar description
            pbar.set_description(f"Processing {image_name}")
            
            # Load image
            image = load_image(image_path)
            
            # Make prediction
            segmentation = predict_segmentation(image, args.api_url)
            
            if segmentation is not None:
                # Create side-by-side comparison
                comparison_path = os.path.join(comparison_dir, f"{base_name}_comparison.png")
                create_side_by_side_comparison(image, segmentation, comparison_path)
                
                # Save standalone mask with _initial suffix
                mask_path = os.path.join(mask_dir, f"{base_name}_initial.png")
                cv2.imwrite(mask_path, segmentation)
                
                successful_predictions += 1
                pbar.set_postfix({"Status": "Success"})
            else:
                failed_predictions += 1
                pbar.set_postfix({"Status": "Failed"})
        
        pbar.close()
        
        # Print summary
        print("📊 Processing Summary:")
        print(f"   Total images: {num_images}")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Failed predictions: {failed_predictions}")
        print(f"   Success rate: {(successful_predictions/num_images)*100:.1f}%")
        
        if successful_predictions > 0:
            print()
            print("🎉 Validation set analysis completed successfully!")
            print(f"   Results saved in: {args.output_dir}/")
            print(f"   Comparison images: {comparison_dir}/")
            print(f"   Mask images: {mask_dir}/")
        else:
            print("❌ No successful predictions to save!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 