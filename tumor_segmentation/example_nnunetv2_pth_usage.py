#!/usr/bin/env python3
"""
Example usage of the nnUNetV2 .pth API.

This script demonstrates how to:
1. Load specific patient images from the patients folder
2. Load corresponding ground truth labels
3. Make predictions using the API
4. Create a 2-column visualization similar to worstdice plots with TP/FP/FN coloring

CONFIGURATION:
Edit the PATIENT_NUMBERS list below to specify which patients you want to process.
Example: PATIENT_NUMBERS = [0, 1, 2, 3, 4]  # Process patients 0-4
Example: PATIENT_NUMBERS = [42, 67, 123]    # Process only patients 42, 67, and 123
"""

# =============================================================================
# CONFIGURATION - Edit this list to specify which patients to process
# =============================================================================
PATIENT_NUMBERS = [42, 67, 123, 3, 4]  # Change this list to your desired patient numbers
# =============================================================================

import requests
import base64
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def load_image(image_path):
    """Load image from file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"📸 Loaded image: {image_path}")
    print(f"   Shape: {image.shape}")
    print(f"   Data type: {image.dtype}")
    print(f"   Value range: [{image.min()}, {image.max()}]")
    
    return image

def load_ground_truth(label_path):
    """Load ground truth label from file"""
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    # Load label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f"Failed to load label: {label_path}")
    
    # Binarize the label (ensure it's 0 or 1)
    label_binary = (label > 0).astype(np.float32)
    
    print(f"📋 Loaded ground truth: {label_path}")
    print(f"   Shape: {label_binary.shape}")
    print(f"   Unique values: {np.unique(label_binary)}")
    
    return label_binary

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
    print(f"🚀 Making prediction request to {api_url}...")
    
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
            
            print(f"✅ Prediction successful!")
            print(f"   Output shape: {segmentation.shape}")
            print(f"   Output range: [{segmentation.min()}, {segmentation.max()}]")
            
            return segmentation
        else:
            print(f"❌ Prediction failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def create_overlay_with_ground_truth(pred_binary, target_binary):
    """Create overlay visualization with TP/FP/FN coloring like DiceAnalysisCallback"""
    # Create RGB overlay
    overlay = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    
    # Define regions
    TP = ((pred_binary > 0) & (target_binary > 0))  # True Positives
    FP = ((pred_binary > 0) & (target_binary == 0))  # False Positives  
    FN = ((pred_binary == 0) & (target_binary > 0))  # False Negatives
    
    # Color coding
    overlay[TP] = [0, 255, 0]    # Green for TP
    overlay[FP] = [255, 0, 0]    # Red for FP
    overlay[FN] = [0, 0, 255]    # Blue for FN
    
    return overlay

def calculate_dice_score(pred_binary, target_binary):
    """Calculate Dice score between prediction and ground truth"""
    intersection = np.sum((pred_binary > 0) & (target_binary > 0))
    union = np.sum((pred_binary > 0) | (target_binary > 0))
    
    if union == 0:
        return 1.0  # Perfect score if both are empty
    
    dice = (2.0 * intersection) / (np.sum(pred_binary > 0) + np.sum(target_binary > 0))
    return dice

def create_visualization(images, predictions, ground_truths, names, dice_scores, output_path):
    """Create 2-column visualization similar to worstdice plots with TP/FP/FN coloring"""
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Column 1: Original images
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f'Original: {names[i]}')
        axes[i, 0].axis('off')
        
        # Column 2: Predictions with TP/FP/FN overlay
        pred_binary = (predictions[i] > 0.5).astype(np.float32)
        overlay = create_overlay_with_ground_truth(pred_binary, ground_truths[i])
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'Prediction: {names[i]} (Dice = {dice_scores[i]:.4f})')
        axes[i, 1].axis('off')
    
    # Add legend to the first row
    green_patch = mpatches.Patch(color='green', label='TP')
    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color='blue', label='FN')
    axes[0, 1].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization to: {output_path}")

def generate_patient_data(patient_numbers):
    """Generate list of (image_path, label_path) tuples for specified patient numbers"""
    patient_data = []
    for patient_num in patient_numbers:
        image_path = f"data/patients/imgs/patient_{patient_num:03d}.png"
        label_path = f"data/patients/labels/segmentation_{patient_num:03d}.png"
        patient_data.append((image_path, label_path))
    return patient_data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Example usage of nnUNetV2 .pth API with specific patient images")
    parser.add_argument("--api-url", default="http://localhost:9052", 
                       help="API URL (default: http://localhost:9052)")
    parser.add_argument("--output-dir", default="prediction_results", 
                       help="Output directory (default: prediction_results)")
    
    args = parser.parse_args()
    
    # Generate patient data from the configuration list
    patient_data = generate_patient_data(PATIENT_NUMBERS)
    num_patients = len(patient_data)
    
    print("🎯 nnUNetV2 .pth API Example - Patient Images with Ground Truth")
    print("=" * 70)
    print(f"Processing {num_patients} patients: {PATIENT_NUMBERS}")
    print(f"API URL: {args.api_url}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    images = []
    predictions = []
    ground_truths = []
    names = []
    dice_scores = []
    
    try:
        for i, (image_path, label_path) in enumerate(patient_data):
            patient_num = PATIENT_NUMBERS[i]
            print(f"Processing patient {i+1}/{num_patients}: patient_{patient_num:03d}")
            print("-" * 50)
            
            # Load image and ground truth
            image = load_image(image_path)
            ground_truth = load_ground_truth(label_path)
            print()
            
            # Make prediction
            segmentation = predict_segmentation(image, args.api_url)
            
            if segmentation is not None:
                # Calculate Dice score
                pred_binary = (segmentation > 0.5).astype(np.float32)
                dice_score = calculate_dice_score(pred_binary, ground_truth)
                
                # Store results
                images.append(image)
                predictions.append(segmentation)
                ground_truths.append(ground_truth)
                names.append(f"patient_{patient_num:03d}")
                dice_scores.append(dice_score)
                
                print(f"📊 Dice Score: {dice_score:.4f}")
                
                # Save individual results
                save_results_with_ground_truth(image, segmentation, ground_truth, args.output_dir, f"patient_{patient_num:03d}", dice_score)
                print()
            else:
                print(f"❌ Failed to get prediction for patient_{patient_num:03d}")
                print()
        
        # Create combined visualization
        if len(images) > 0:
            print("Creating combined visualization with TP/FP/FN coloring...")
            viz_path = os.path.join(args.output_dir, "patient_predictions_with_ground_truth.png")
            create_visualization(images, predictions, ground_truths, names, dice_scores, viz_path)
            
            # Print summary statistics
            print("\n📊 Summary Statistics:")
            print(f"   Average Dice Score: {np.mean(dice_scores):.4f}")
            best_idx = np.argmax(dice_scores)
            worst_idx = np.argmin(dice_scores)
            print(f"   Best Dice Score: {np.max(dice_scores):.4f} ({names[best_idx]})")
            print(f"   Worst Dice Score: {np.min(dice_scores):.4f} ({names[worst_idx]})")
            
            print()
            print("🎉 Example completed successfully!")
            print(f"   Results saved in: {args.output_dir}/")
            print(f"   Visualization: {viz_path}")
        else:
            print("❌ No successful predictions to visualize!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def save_results_with_ground_truth(image, segmentation, ground_truth, output_dir, base_name, dice_score):
    """Save prediction results with ground truth comparison"""
    # Only save the TP/FP/FN overlay for individual analysis if needed
    pred_binary = (segmentation > 0.5).astype(np.float32)
    overlay = create_overlay_with_ground_truth(pred_binary, ground_truth)
    overlay_path = os.path.join(output_dir, f"{base_name}_tp_fp_fn_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"   TP/FP/FN Overlay: {overlay_path}")

if __name__ == "__main__":
    main() 