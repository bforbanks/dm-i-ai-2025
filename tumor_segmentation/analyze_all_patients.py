#!/usr/bin/env python3
"""
Analyze all patient images and create worstdice-style visualization.

This script:
1. Loops through ALL patient images in the data/patients/imgs folder
2. Gets predictions for each patient using the API
3. Calculates Dice scores against ground truth
4. Creates a visualization showing the n worst and n best performing patients
5. Uses the same 4-column layout as DiceAnalysisCallback (worst original, worst overlay, best original, best overlay)

Postprocess mode:
1. Saves all predicted masks in postprocess_dataset folder
2. Creates JSON files with dice scores and false positive ratios
3. Creates stratified train/validation splits
4. Creates hard/easy image subsets for both patients and controls
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
import json
import shutil

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
    label_binary = (label > 0).astype(np.float32)  # type: ignore
    
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
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # type: ignore
    return image_base64

def decode_segmentation(segmentation_base64):
    """Decode base64 segmentation to numpy array"""
    segmentation_bytes = base64.b64decode(segmentation_base64)
    segmentation = cv2.imdecode(np.frombuffer(segmentation_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return segmentation

def predict_segmentation(image, api_url="http://localhost:9052"):
    """Make prediction using the API - returns binary mask"""
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


def predict_segmentation_with_probabilities(image, api_url="http://localhost:9052"):
    """Make prediction using the API - returns probability maps"""
    # Encode image
    image_base64 = encode_image(image)
    
    # Make request
    try:
        response = requests.post(
            f"{api_url}/predict_with_probabilities",
            json={"img": image_base64},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            probabilities = decode_segmentation(result["img"])
            return probabilities
        else:
            print(f"❌ Probability prediction failed with status code: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Probability request failed: {e}")
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

def load_nnunet_fold_split(nnunet_preprocessed_dir: str, fold: int):
    """Load nnUNet fold splits to ensure alignment with trained model.
    
    Args:
        nnunet_preprocessed_dir: Path to nnUNet preprocessed directory 
        fold: Fold number (e.g., 2 for fold_2)
        
    Returns:
        Dict with 'train' and 'val' keys containing patient lists
    """
    splits_file = os.path.join(nnunet_preprocessed_dir, "splits_final.json")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"nnUNet splits file not found: {splits_file}")
    
    try:
        with open(splits_file, 'r') as f:
            nnunet_splits = json.load(f)
        
        if fold >= len(nnunet_splits):
            raise ValueError(f"Fold {fold} not found. Available folds: 0-{len(nnunet_splits)-1}")
        
        fold_data = nnunet_splits[fold]
        print(f"📂 Loaded nnUNet fold {fold} splits:")
        print(f"   Training cases: {len(fold_data['train'])}")
        print(f"   Validation cases: {len(fold_data['val'])}")
        
        return fold_data
    except Exception as e:
        print(f"❌ Error loading nnUNet fold {fold} from {splits_file}: {e}")
        return None


def load_split_data(split_path="data_nnUNet/split.json"):
    """Load train/val split data from JSON file"""
    try:
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        return split_data[0]  # The file contains a list with one dict
    except Exception as e:
        print(f"❌ Error loading split data from {split_path}: {e}")
        return None

def filter_patients_from_split(split_list):
    """Filter only patients from a split list (exclude controls)"""
    return [item for item in split_list if item.startswith("patient_")]

def extract_patient_number(patient_id):
    """Extract patient number from patient_XXX format"""
    return int(patient_id.replace("patient_", ""))

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

def find_patient_files_by_split(split_patients):
    """Find patient files based on a specific split list"""
    patient_data = []
    
    for patient_id in split_patients:
        patient_num = extract_patient_number(patient_id)
        
        # Construct paths
        img_path = f"data/patients/imgs/patient_{patient_num:03d}.png"
        label_path = f"data/patients/labels/segmentation_{patient_num:03d}.png"
        
        # Check if files exist
        if os.path.exists(img_path) and os.path.exists(label_path):
            patient_data.append((patient_num, img_path, label_path))
        else:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Image not found for {patient_id}")
            if not os.path.exists(label_path):
                print(f"⚠️  Warning: Label not found for {patient_id}")
    
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
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved worstdice visualization to: {output_path}")

def analyze_patient_set(patient_data, set_name, args):
    """Analyze a specific set of patients (train or val) and return results"""
    print(f"\n🚀 Processing {set_name} set ({len(patient_data)} patients)...")
    
    # Process all patients
    images = []
    predictions = []
    ground_truths = []
    names = []
    dice_scores = []
    successful_patients = []
    
    for i, (patient_num, img_path, label_path) in enumerate(patient_data):
        print(f"Processing {set_name} {i+1}/{len(patient_data)}: patient_{patient_num:03d}", end=" ")
        
        try:
            # Load image and ground truth
            image = load_image(img_path)
            ground_truth = load_ground_truth(label_path)
            
            # Make prediction
            segmentation = predict_segmentation(image, args.api_url)
            
            if segmentation is not None:
                # Calculate Dice score
                pred_binary = (segmentation > 0.5).astype(np.float32)  # type: ignore
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
    
    print(f"\n📊 Successfully processed {len(successful_patients)} {set_name} patients")
    
    if len(images) == 0:
        print(f"❌ No successful predictions to analyze for {set_name} set!")
        return None
    
    # Convert dice_scores to numpy array for sorting
    dice_scores = np.array(dice_scores)
    
    # Find worst and best performers
    k = min(args.top_k, len(dice_scores))
    worst_indices = np.argsort(dice_scores)[:k]
    best_indices = np.argsort(dice_scores)[-k:][::-1]  # Reverse to get best first
    
    # Create worstdice visualization
    print(f"\n📊 Creating {set_name} worstdice visualization with {k} worst and {k} best patients...")
    viz_path = os.path.join(args.output_dir, f"worstdice_analysis_{set_name}.png")
    create_worstdice_visualization(images, predictions, ground_truths, names, dice_scores, worst_indices, best_indices, viz_path)
    
    # Print summary statistics
    print(f"\n📊 {set_name.capitalize()} Set Summary Statistics:")
    print(f"   Total patients processed: {len(successful_patients)}")
    print(f"   Average Dice Score: {np.mean(dice_scores):.4f}")
    print(f"   Median Dice Score: {np.median(dice_scores):.4f}")
    print(f"   Standard Deviation: {np.std(dice_scores):.4f}")
    print(f"   Best Dice Score: {np.max(dice_scores):.4f} ({names[best_indices[0]]})")
    print(f"   Worst Dice Score: {np.min(dice_scores):.4f} ({names[worst_indices[0]]})")
    
    # Save detailed results
    results_path = os.path.join(args.output_dir, f"dice_scores_{set_name}.csv")
    
    # Ensure output directory exists
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("Patient,Dice_Score\n")
        for i, (name, score) in enumerate(zip(names, dice_scores)):
            f.write(f"{name},{score:.6f}\n")
    
    print(f"\n💾 {set_name.capitalize()} results saved to: {results_path}")
    
    return {
        'set_name': set_name,
        'viz_path': viz_path,
        'results_path': results_path,
        'stats': {
            'total': len(successful_patients),
            'mean': np.mean(dice_scores),
            'median': np.median(dice_scores),
            'std': np.std(dice_scores),
            'best': np.max(dice_scores),
            'worst': np.min(dice_scores)
        }
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze all patients and create worstdice visualization")
    parser.add_argument("--api-url", default="http://localhost:9052", 
                       help="API URL (default: http://localhost:9052)")
    parser.add_argument("--output-dir", default="worstdice_analysis", 
                       help="Output directory (default: worstdice_analysis)")
    parser.add_argument("--top-k", type=int, default=7, 
                       help="Number of worst/best patients to show (default: 7)")
    parser.add_argument("--max-patients", type=int, default=None, 
                       help="Maximum number of patients to process (default: all)")
    parser.add_argument("--use-val-split", action="store_true",
                       help="Use train/val split from split.json and analyze each set separately")
    parser.add_argument("--nnunet-fold", type=int, default=None,
                       help="Use nnUNet fold splits instead of split.json (e.g., 2 for fold_2)")
    parser.add_argument("--nnunet-preprocessed-dir", default="data_nnUNet/preprocessed/Dataset001_TumorSegmentation",
                       help="Path to nnUNet preprocessed directory")
    parser.add_argument("--analyze-for-postprocess", action="store_true",
                       help="Analyze patients for postprocessing (saves masks, calculates metrics, creates splits)")
    parser.add_argument("--control-patient-split", type=float, default=0.3,
                       help="Ratio of controls to patients in training folder (default: 0.3 for 70%% patients, 30%% controls)")
    
    args = parser.parse_args()
    
    print("🎯 All Patients Analysis - Worstdice Style")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top K worst/best: {args.top_k}")
    print(f"Use validation split: {args.use_val_split}")
    if args.nnunet_fold is not None:
        print(f"nnUNet fold: {args.nnunet_fold}")
        print(f"nnUNet preprocessed dir: {args.nnunet_preprocessed_dir}")
    if args.max_patients:
        print(f"Max patients to process: {args.max_patients}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.analyze_for_postprocess:
        print("🚀 Analyzing patients for postprocessing...")
        
        # Create postprocess dataset directory
        postprocess_dir = "postprocess_dataset"
        os.makedirs(postprocess_dir, exist_ok=True)
        
        # Find all patient files
        patient_data = find_all_patient_files()
        print(f"Found {len(patient_data)} patients with both images and labels")
        
        # Limit patients if requested
        if args.max_patients and args.max_patients < len(patient_data):
            print(f"Limiting analysis to first {args.max_patients} patients")
            patient_data = patient_data[:args.max_patients]
        
        # Process patients and save masks
        patient_results = {}
        successful_patients = []
        
        print(f"\n1. Processing {len(patient_data)} patients and saving masks...")
        for i, (patient_num, img_path, label_path) in enumerate(patient_data):
            print(f"Processing {i+1}/{len(patient_data)}: patient_{patient_num:03d}", end=" ")
            
            try:
                # Load image and ground truth
                image = load_image(img_path)
                ground_truth = load_ground_truth(label_path)
                
                # Make predictions - both binary and probability
                segmentation = predict_segmentation(image, args.api_url)
                probabilities = predict_segmentation_with_probabilities(image, args.api_url)
                
                if segmentation is not None and probabilities is not None:
                    # Calculate Dice score using binary mask
                    pred_binary = (segmentation > 0.5).astype(np.float32)
                    dice_score = calculate_dice_score(pred_binary, ground_truth)
                    
                    # Save binary prediction mask
                    pred_mask_path = os.path.join(postprocess_dir, f"patient_{patient_num:03d}_prediction.png")
                    cv2.imwrite(pred_mask_path, (pred_binary * 255).astype(np.uint8))
                    
                    # Convert probabilities from [0-255] range back to [0-1] range
                    prob_normalized = probabilities.astype(np.float32) / 255.0
                    
                    # Save probability map (scaled to 0-255 for visualization)
                    prob_map_path = os.path.join(postprocess_dir, f"patient_{patient_num:03d}_probabilities.png")
                    cv2.imwrite(prob_map_path, probabilities.astype(np.uint8))
                    
                    # Save raw probability map as numpy array for later use
                    prob_raw_path = os.path.join(postprocess_dir, f"patient_{patient_num:03d}_probabilities.npy")
                    np.save(prob_raw_path, prob_normalized)
                    
                    # Store results
                    patient_results[f"patient_{patient_num:03d}"] = {
                        "dice_score": dice_score,
                        "mask_path": pred_mask_path,
                        "prob_map_path": prob_map_path,
                        "prob_raw_path": prob_raw_path
                    }
                    successful_patients.append(patient_num)
                    
                    print(f"✅ Dice = {dice_score:.4f}")
                else:
                    print("❌ Failed")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Successfully processed {len(successful_patients)} patients")
        
        # Find control files
        print(f"\n2. Processing control subjects...")
        control_pattern = "data/controls/imgs/control_*.png"
        control_files = sorted(glob.glob(control_pattern))
        
        control_results = {}
        successful_controls = []
        
        for i, control_path in enumerate(control_files):
            control_num = int(Path(control_path).name.replace("control_", "").replace(".png", ""))
            print(f"Processing control {i+1}/{len(control_files)}: control_{control_num:03d}", end=" ")
            
            try:
                # Load control image
                image = load_image(control_path)
                
                # Make predictions - both binary and probability
                segmentation = predict_segmentation(image, args.api_url)
                probabilities = predict_segmentation_with_probabilities(image, args.api_url)
                
                if segmentation is not None and probabilities is not None:
                    # Calculate false positive ratio using binary mask
                    pred_binary = (segmentation > 0.5).astype(np.float32)
                    fp_pixels = np.sum(pred_binary > 0)
                    total_pixels = pred_binary.size
                    fp_ratio = fp_pixels / total_pixels if total_pixels > 0 else 0.0
                    
                    # Save binary prediction mask
                    pred_mask_path = os.path.join(postprocess_dir, f"control_{control_num:03d}_prediction.png")
                    cv2.imwrite(pred_mask_path, (pred_binary * 255).astype(np.uint8))
                    
                    # Convert probabilities from [0-255] range back to [0-1] range
                    prob_normalized = probabilities.astype(np.float32) / 255.0
                    
                    # Save probability map (scaled to 0-255 for visualization)
                    prob_map_path = os.path.join(postprocess_dir, f"control_{control_num:03d}_probabilities.png")
                    cv2.imwrite(prob_map_path, probabilities.astype(np.uint8))
                    
                    # Save raw probability map as numpy array for later use
                    prob_raw_path = os.path.join(postprocess_dir, f"control_{control_num:03d}_probabilities.npy")
                    np.save(prob_raw_path, prob_normalized)
                    
                    # Store results
                    control_results[f"control_{control_num:03d}"] = {
                        "fp_ratio": fp_ratio,
                        "fp_pixels": int(fp_pixels),
                        "total_pixels": int(total_pixels),
                        "mask_path": pred_mask_path,
                        "prob_map_path": prob_map_path,
                        "prob_raw_path": prob_raw_path
                    }
                    successful_controls.append(control_num)
                    
                    print(f"✅ FP ratio = {fp_ratio:.6f}")
                else:
                    print("❌ Failed")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Successfully processed {len(successful_controls)} controls")
        
        # Save JSON files
        print(f"\n3. Saving JSON files...")
        
        # Save patient dice scores
        with open(os.path.join(postprocess_dir, "patient_dice_scores.json"), 'w') as f:
            json.dump(patient_results, f, indent=2)
        
        # Save control FP ratios
        with open(os.path.join(postprocess_dir, "control_fp_ratios.json"), 'w') as f:
            json.dump(control_results, f, indent=2)
        
        # Create stratified splits for patients
        print(f"\n4. Creating stratified patient splits...")
        patient_names = list(patient_results.keys())
        dice_scores = [patient_results[name]["dice_score"] for name in patient_names]
        
        # Sort by dice score for stratification
        sorted_indices = np.argsort(dice_scores)
        sorted_patients = [patient_names[i] for i in sorted_indices]
        
        # Create validation set (20% with stratification)
        val_size = max(1, int(0.2 * len(sorted_patients)))
        val_patients = []
        train_patients = []
        
        # Stratified sampling: take every 5th patient for validation
        for i, patient in enumerate(sorted_patients):
            if i % 5 == 0 and len(val_patients) < val_size:
                val_patients.append(patient)
            else:
                train_patients.append(patient)
        
        # Create folders - we'll create the binary and probability versions below
        
        # Create parallel folder structures for binary masks and probability maps
        train_binary_dir = os.path.join(postprocess_dir, "train_binary")
        val_binary_dir = os.path.join(postprocess_dir, "validation_binary")
        train_prob_dir = os.path.join(postprocess_dir, "train_probabilities")
        val_prob_dir = os.path.join(postprocess_dir, "validation_probabilities")
        
        os.makedirs(train_binary_dir, exist_ok=True)
        os.makedirs(val_binary_dir, exist_ok=True)
        os.makedirs(train_prob_dir, exist_ok=True)
        os.makedirs(val_prob_dir, exist_ok=True)
        
        # Move patient files to appropriate folders (both binary and probability)
        for patient in train_patients:
            # Binary masks
            src_binary = patient_results[patient]["mask_path"]
            dst_binary = os.path.join(train_binary_dir, os.path.basename(src_binary))
            shutil.copy2(src_binary, dst_binary)
            
            # Probability maps
            src_prob = patient_results[patient]["prob_map_path"]
            dst_prob = os.path.join(train_prob_dir, os.path.basename(src_prob).replace("_prediction.png", "_probabilities.png"))
            shutil.copy2(src_prob, dst_prob)
        
        for patient in val_patients:
            # Binary masks
            src_binary = patient_results[patient]["mask_path"]
            dst_binary = os.path.join(val_binary_dir, os.path.basename(src_binary))
            shutil.copy2(src_binary, dst_binary)
            
            # Probability maps
            src_prob = patient_results[patient]["prob_map_path"]
            dst_prob = os.path.join(val_prob_dir, os.path.basename(src_prob).replace("_prediction.png", "_probabilities.png"))
            shutil.copy2(src_prob, dst_prob)
        
        # Calculate median dice on training set only
        train_dice_scores = [patient_results[name]["dice_score"] for name in train_patients]
        median_dice = np.median(train_dice_scores)
        
        # Create hard/easy subsets for patients
        hard_images_patients = [name for name in train_patients if patient_results[name]["dice_score"] < median_dice]
        easy_images_patients = [name for name in train_patients if patient_results[name]["dice_score"] >= median_dice]
        
        # Process controls  
        print(f"\n5. Processing control splits...")
        control_names = list(control_results.keys())
        fp_ratios = [control_results[name]["fp_ratio"] for name in control_names]
        
        # Count controls with FP > 0 from ENTIRE control set
        controls_with_fp_all = [name for name in control_names if control_results[name]["fp_ratio"] > 0]
        print(f"\n📊 Controls with FP > 0 (from entire set): {len(controls_with_fp_all)}")
        
        # Calculate how many controls we need for 70-30 split in training folder
        num_train_patients = len(train_patients)
        # If we want 70% patients and 30% controls in training:
        # patients / (patients + controls) = 0.7
        # controls / (patients + controls) = 0.3
        # So: controls = patients * 0.3 / 0.7
        num_controls_for_train = int(num_train_patients * args.control_patient_split / (1 - args.control_patient_split))
        num_controls_for_train = min(num_controls_for_train, len(control_names))  # Don't exceed available controls
        
        print(f"📊 Training folder target: {num_train_patients} patients + {num_controls_for_train} controls")
        
        # Sort controls by FP ratio for stratified sampling
        sorted_control_indices = np.argsort(fp_ratios)
        sorted_controls = [control_names[i] for i in sorted_control_indices]
        
        # Put ALL controls with FP > 0 in training folder
        control_train = controls_with_fp_all.copy()
        
        # Add more controls to reach the target ratio (stratified sampling)
        controls_without_fp = [name for name in control_names if control_results[name]["fp_ratio"] == 0]
        additional_controls_needed = num_controls_for_train - len(control_train)
        
        if additional_controls_needed > 0 and len(controls_without_fp) > 0:
            # Take additional controls stratified by their order (even though FP=0)
            step = max(1, len(controls_without_fp) // additional_controls_needed)
            for i in range(0, len(controls_without_fp), step):
                if len(control_train) < num_controls_for_train:
                    control_train.append(controls_without_fp[i])
        
        # Remaining controls go to control_rest
        control_rest = [name for name in control_names if name not in control_train]
        
        # Create folders (remove control_rest_dir creation as we don't need it anymore)
        # Control rest will just be the remaining controls not in training
        
        # Copy control files to training folders (both binary and probability)
        for control in control_train:
            # Binary masks
            src_binary = control_results[control]["mask_path"]
            dst_binary = os.path.join(train_binary_dir, os.path.basename(src_binary))
            shutil.copy2(src_binary, dst_binary)
            
            # Probability maps
            src_prob = control_results[control]["prob_map_path"]
            dst_prob = os.path.join(train_prob_dir, os.path.basename(src_prob).replace("_prediction.png", "_probabilities.png"))
            shutil.copy2(src_prob, dst_prob)
        
        # Create hard/easy subsets for controls using ALL controls with FP > 0
        if len(controls_with_fp_all) > 0:
            control_fp_ratios_all = [control_results[name]["fp_ratio"] for name in controls_with_fp_all]
            median_fp = np.median(control_fp_ratios_all)
            
            hard_images_control = [name for name in controls_with_fp_all if control_results[name]["fp_ratio"] >= median_fp]
            easy_images_control = [name for name in controls_with_fp_all if control_results[name]["fp_ratio"] < median_fp]
        else:
            hard_images_control = []
            easy_images_control = []
        
        # Create comprehensive JSON with all splits and subsets
        comprehensive_results = {
            "patient_splits": {
                "train": train_patients,
                "validation": val_patients,
                "hard_images_patients": hard_images_patients,
                "easy_images_patients": easy_images_patients
            },
            "control_splits": {
                "train": control_train,
                "control_rest": control_rest,
                "hard_images_control": hard_images_control
            },
            "statistics": {
                "total_patients": len(patient_results),
                "total_controls": len(control_results),
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "train_controls": len(control_train),
                "control_rest": len(control_rest),
                "controls_with_fp": len(controls_with_fp_all),
                "median_dice_train": float(median_dice),
                "median_fp_controls": float(np.median([control_results[name]["fp_ratio"] for name in controls_with_fp_all])) if controls_with_fp_all else 0.0
            }
        }
        
        with open(os.path.join(postprocess_dir, "comprehensive_splits.json"), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\n🎉 Postprocessing analysis completed successfully!")
        print("=" * 60)
        print(f"📁 Postprocess dataset saved to: {postprocess_dir}")
        print(f"📊 Patient dice scores: {len(patient_results)} patients")
        print(f"📊 Control FP ratios: {len(control_results)} controls")
        print(f"📊 Controls with FP > 0 (from entire set): {len(controls_with_fp_all)}")
        print(f"📁 Train folders: {len(train_patients)} patients + {len(control_train)} controls")
        print(f"   ├── train_binary/: Binary masks")
        print(f"   └── train_probabilities/: Probability maps")
        print(f"   └── Patient:Control ratio = {len(train_patients)/(len(train_patients)+len(control_train))*100:.1f}%:{len(control_train)/(len(train_patients)+len(control_train))*100:.1f}%")
        print(f"📁 Validation folders: {len(val_patients)} patients")
        print(f"   ├── validation_binary/: Binary masks")
        print(f"   └── validation_probabilities/: Probability maps")
        print(f"📁 Control rest (not in training): {len(control_rest)} controls")
        print(f"📊 Hard patient images: {len(hard_images_patients)}")
        print(f"📊 Easy patient images: {len(easy_images_patients)}")
        print(f"📊 Hard control images: {len(hard_images_control)}")

    elif args.use_val_split:
        # Load split data
        if args.nnunet_fold is not None:
            print(f"🔍 Loading nnUNet fold {args.nnunet_fold} split data...")
            split_data = load_nnunet_fold_split(args.nnunet_preprocessed_dir, args.nnunet_fold)
        else:
            print("🔍 Loading train/val split data from split.json...")
            split_data = load_split_data()
        
        if split_data is None:
            print("❌ Failed to load split data! Exiting.")
            return
        
        # Extract patient lists from splits
        train_patients = filter_patients_from_split(split_data['train'])
        val_patients = filter_patients_from_split(split_data['val'])
        
        print(f"Found {len(train_patients)} training patients and {len(val_patients)} validation patients in split")
        
        # Limit patients if requested
        if args.max_patients:
            if args.max_patients < len(train_patients):
                print(f"Limiting training patients to first {args.max_patients}")
                train_patients = train_patients[:args.max_patients]
            if args.max_patients < len(val_patients):
                print(f"Limiting validation patients to first {args.max_patients}")
                val_patients = val_patients[:args.max_patients]
        
        # Find patient files for each split
        train_data = find_patient_files_by_split(train_patients)
        val_data = find_patient_files_by_split(val_patients)
        
        print(f"Found files for {len(train_data)} train patients and {len(val_data)} val patients")
        
        if len(train_data) == 0 and len(val_data) == 0:
            print("❌ No patient files found for any split!")
            return
        
        results = []
        
        # Analyze validation set
        if len(val_data) > 0:
            val_result = analyze_patient_set(val_data, "val", args)
            if val_result:
                results.append(val_result)
        
        # Analyze training set
        if len(train_data) > 0:
            train_result = analyze_patient_set(train_data, "train", args)
            if train_result:
                results.append(train_result)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 Split-based Analysis Completed Successfully!")
        print("=" * 60)
        for result in results:
            print(f"\n{result['set_name'].upper()} SET RESULTS:")
            print(f"   Visualization: {result['viz_path']}")
            print(f"   CSV Results: {result['results_path']}")
            print(f"   Total patients: {result['stats']['total']}")
            print(f"   Average Dice: {result['stats']['mean']:.4f}")
            print(f"   Best Dice: {result['stats']['best']:.4f}")
            print(f"   Worst Dice: {result['stats']['worst']:.4f}")
        
    else:
        # Original analysis (all patients)
        print("🔍 Finding all patient files...")
        patient_data = find_all_patient_files()
        print(f"Found {len(patient_data)} patients with both images and labels")
        
        # Limit number of patients if requested
        if args.max_patients and args.max_patients < len(patient_data):
            print(f"Limiting analysis to first {args.max_patients} patients")
            patient_data = patient_data[:args.max_patients]
        
        print(f"Will process {len(patient_data)} patients")
        print()
        
        if len(patient_data) == 0:
            print("❌ No patient files found!")
            return
        
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
                    pred_binary = (segmentation > 0.5).astype(np.float32)  # type: ignore
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
        
        # Convert dice_scores to numpy array for sorting (images stay as lists due to different sizes)
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