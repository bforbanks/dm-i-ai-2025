#!/usr/bin/env python3
"""
Analyze all patient images and create worstdice-style visualization.

This script:
1. Loops through ALL patient images in the data/patients/imgs folder
2. Gets predictions for each patient using the API
3. Calculates Dice scores against ground truth
4. Creates a visualization showing the n worst and n best performing patients
5. Uses the same 4-column layout as DiceAnalysisCallback (worst original, worst overlay, best original, best overlay)
"""

import requests
import base64
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import glob

def load_image(image_path):
    """Load image from file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
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
            print(f"❌ Prediction failed with status code: {response.status_code}")
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
    
    if intersection == 0:
        return 0.0  # No overlap
    
    dice = (2.0 * intersection) / (np.sum(pred_binary > 0) + np.sum(target_binary > 0))
    return dice

def find_all_patient_files():
    """Find all patient image files and their corresponding labels"""
    img_pattern = "data/patients/imgs/patient_*.png"
    img_files = sorted(glob.glob(img_pattern))
    
    patient_data = []
    for img_path in img_files:
        # Extract patient number from filename
        filename = Path(img_path).name
        patient_num = int(filename.replace("patient_", "").replace(".png", ""))
        
        # Check if corresponding label exists
        label_path = f"data/patients/labels/segmentation_{patient_num:03d}.png"
        if os.path.exists(label_path):
            patient_data.append((patient_num, img_path, label_path))
        else:
            print(f"⚠️  Warning: No label found for patient_{patient_num:03d}")
    
    return patient_data

def create_worstdice_visualization(images, predictions, ground_truths, names, dice_scores, worst_indices, best_indices, output_path):
    """Create 4-column visualization exactly like DiceAnalysisCallback"""
    k = len(worst_indices)  # Number of worst/best to show
    fig, axes = plt.subplots(k, 4, figsize=(20, 5 * k))
    
    if k == 1:
        axes = axes.reshape(1, -1)
    
    # Column 1: Worst performing - Original images
    for i in range(k):
        idx = worst_indices[i]
        dice = dice_scores[idx]
        
        axes[i, 0].imshow(images[idx], cmap='gray')
        axes[i, 0].set_title(f'Worst {i+1}: {names[idx]}')
        axes[i, 0].axis('off')
    
    # Column 2: Worst performing - Overlays
    for i in range(k):
        idx = worst_indices[i]
        dice = dice_scores[idx]
        
        pred_binary = (predictions[idx] > 0.5).astype(np.float32)
        overlay = create_overlay_with_ground_truth(pred_binary, ground_truths[idx])
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f'Worst {i+1}: Dice = {dice:.4f}')
        axes[i, 1].axis('off')
    
    # Column 3: Best performing - Original images
    for i in range(k):
        idx = best_indices[i]
        dice = dice_scores[idx]
        
        axes[i, 2].imshow(images[idx], cmap='gray')
        axes[i, 2].set_title(f'Best {i+1}: {names[idx]}')
        axes[i, 2].axis('off')
    
    # Column 4: Best performing - Overlays
    for i in range(k):
        idx = best_indices[i]
        dice = dice_scores[idx]
        
        pred_binary = (predictions[idx] > 0.5).astype(np.float32)
        overlay = create_overlay_with_ground_truth(pred_binary, ground_truths[idx])
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Best {i+1}: Dice = {dice:.4f}')
        axes[i, 3].axis('off')
    
    # Add legend to the first row
    green_patch = mpatches.Patch(color='green', label='TP')
    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color='blue', label='FN')
    axes[0, 1].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right', fontsize=8)
    axes[0, 3].legend(handles=[green_patch, red_patch, blue_patch], loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved worstdice visualization to: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze all patients and create worstdice visualization")
    parser.add_argument("--api-url", default="http://localhost:9052", 
                       help="API URL (default: http://localhost:9052)")
    parser.add_argument("--output-dir", default="worstdice_analysis", 
                       help="Output directory (default: worstdice_analysis)")
    parser.add_argument("--top-k", type=int, default=7, 
                       help="Number of worst/best patients to show (default: 7)")
    
    args = parser.parse_args()
    
    print("🎯 All Patients Analysis - Worstdice Style")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top K worst/best: {args.top_k}")
    print()
    
    # Find all patient files
    print("🔍 Finding all patient files...")
    patient_data = find_all_patient_files()
    print(f"Found {len(patient_data)} patients with both images and labels")
    print()
    
    if len(patient_data) == 0:
        print("❌ No patient files found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all patients
    images = []
    predictions = []
    ground_truths = []
    names = []
    dice_scores = []
    successful_patients = []
    
    print("🚀 Processing all patients...")
    for i, (patient_num, img_path, label_path) in enumerate(patient_data):
        print(f"Processing {i+1}/{len(patient_data)}: patient_{patient_num:03d}", end=" ")
        
        try:
            # Load image and ground truth
            image = load_image(img_path)
            ground_truth = load_ground_truth(label_path)
            
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
                successful_patients.append(patient_num)
                
                print(f"✅ Dice = {dice_score:.4f}")
            else:
                print("❌ Failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Successfully processed {len(successful_patients)} patients")
    
    if len(images) == 0:
        print("❌ No successful predictions to analyze!")
        return
    
    # Convert to numpy arrays
    images = np.array(images)
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    dice_scores = np.array(dice_scores)
    
    # Find worst and best performers
    k = min(args.top_k, len(dice_scores))
    worst_indices = np.argsort(dice_scores)[:k]
    best_indices = np.argsort(dice_scores)[-k:][::-1]  # Reverse to get best first
    
    # Create worstdice visualization
    print(f"\n📊 Creating worstdice visualization with {k} worst and {k} best patients...")
    viz_path = os.path.join(args.output_dir, "worstdice_analysis.png")
    create_worstdice_visualization(images, predictions, ground_truths, names, dice_scores, worst_indices, best_indices, viz_path)
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Total patients processed: {len(successful_patients)}")
    print(f"   Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"   Median Dice Score: {np.median(dice_scores):.4f}")
    print(f"   Standard Deviation: {np.std(dice_scores):.4f}")
    print(f"   Best Dice Score: {np.max(dice_scores):.4f} ({names[best_indices[0]]})")
    print(f"   Worst Dice Score: {np.min(dice_scores):.4f} ({names[worst_indices[0]]})")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, "dice_scores.csv")
    with open(results_path, 'w') as f:
        f.write("Patient,Dice_Score\n")
        for i, (name, score) in enumerate(zip(names, dice_scores)):
            f.write(f"{name},{score:.6f}\n")
    
    print(f"\n💾 Detailed results saved to: {results_path}")
    print(f"🎉 Analysis completed successfully!")
    print(f"   Visualization: {viz_path}")

if __name__ == "__main__":
    main() 