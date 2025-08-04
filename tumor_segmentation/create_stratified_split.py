#!/usr/bin/env python3
"""
Create a stratified train/validation split for nnUNet v2 with class balance control.
This ensures tumor cases are properly distributed and allows control over class balance.
"""

import os
import json
import random
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import numpy as np
import cv2


def create_stratified_split(dataset_id: int = 1, seed: int = 42, tumor_percentage: float = 0.75):
    """
    Create a stratified train/validation split with class balance control.
    
    Args:
        dataset_id: Dataset ID
        seed: Random seed for reproducibility
        tumor_percentage: Percentage of tumor cases in training set (0.0-1.0)
                         If 0.75, then 75% of training cases will be patients with tumors
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"🔧 Creating stratified split for Dataset{dataset_id:03d}")
    print(f"   Target tumor percentage in training: {tumor_percentage*100:.1f}%")
    
    # Set up paths
    nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
    if not nnUNet_preprocessed:
        # Set environment variables if not already set
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_preprocessed = str(project_dir / "tumor_segmentation" / "data_nnUNet" / "preprocessed")
        os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
        print(f"⚠️  Set nnUNet_preprocessed to: {nnUNet_preprocessed}")
    
    # Dataset paths
    dataset_name = f"Dataset{dataset_id:03d}_TumorSegmentation"
    preprocessed_path = Path(str(nnUNet_preprocessed)) / dataset_name
    
    if not preprocessed_path.exists():
        print(f"⚠️  Preprocessed directory not found: {preprocessed_path}")
        print("   This is expected if preprocessing hasn't been run yet.")
    
    # Get dataset path for raw data
    nnUNet_raw = os.environ.get('nnUNet_raw')
    if not nnUNet_raw:
        # Set environment variables if not already set
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_raw = str(project_dir / "tumor_segmentation" / "data_nnUNet")
        os.environ['nnUNet_raw'] = nnUNet_raw
        print(f"⚠️  Set nnUNet_raw to: {nnUNet_raw}")
    
    dataset_path = Path(str(nnUNet_raw)) / dataset_name
    
    # Get all training cases from the imagesTr directory
    images_tr_dir = dataset_path / "imagesTr"
    if not images_tr_dir.exists():
        raise FileNotFoundError(f"ImagesTr directory not found: {images_tr_dir}")
    
    # Get all image files and extract case names
    image_files = list(images_tr_dir.glob("*_0000.png"))
    all_cases = []
    
    for img_file in image_files:
        # Extract case name: patient_001_0000.png -> patient_001
        case_name = img_file.stem.replace("_0000", "")
        all_cases.append(case_name)
    
    # Sort for reproducibility
    all_cases.sort()
    
    print(f"📊 Dataset info:")
    print(f"   Total cases: {len(all_cases)}")
    
    # Separate patients and controls
    patients = [case for case in all_cases if case.startswith('patient_')]
    controls = [case for case in all_cases if case.startswith('control_')]
    
    print(f"   Patients: {len(patients)}")
    print(f"   Controls: {len(controls)}")
    
    # Check which patients actually have tumors
    labels_dir = dataset_path / "labelsTr"
    patients_with_tumors = []
    patients_without_tumors = []
    
    for patient in patients:
        label_file = labels_dir / f"{patient}.png"
        if label_file.exists():
            label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
            if label is not None:
                tumor_pixels = np.sum(label > 0)
                if tumor_pixels > 0:
                    patients_with_tumors.append(patient)
                else:
                    patients_without_tumors.append(patient)
    
    print(f"   Patients with tumors: {len(patients_with_tumors)}")
    print(f"   Patients without tumors: {len(patients_without_tumors)}")
    
    if len(patients_with_tumors) == 0:
        raise ValueError("No patients with tumors found! Cannot create meaningful split.")
    
    # Create validation set: 20% of all patients (with and without tumors)
    all_patients = patients_with_tumors + patients_without_tumors
    random.shuffle(all_patients)
    val_split_point = int(0.2 * len(all_patients))
    val_patients = all_patients[:val_split_point]
    train_patients_available = all_patients[val_split_point:]
    
    # Separate available training patients by tumor status
    train_patients_with_tumors = [p for p in train_patients_available if p in patients_with_tumors]
    train_patients_without_tumors = [p for p in train_patients_available if p in patients_without_tumors]
    
    print(f"\n📊 Patient Split:")
    print(f"   Validation patients: {len(val_patients)} (20% of all patients)")
    print(f"   Available training patients with tumors: {len(train_patients_with_tumors)}")
    print(f"   Available training patients without tumors: {len(train_patients_without_tumors)}")
    
    # Calculate how many control cases we need to achieve the target tumor percentage
    # Formula: tumor_percentage = tumor_cases / (tumor_cases + control_cases)
    # Solving for control_cases: control_cases = tumor_cases * (1 - tumor_percentage) / tumor_percentage
    
    tumor_cases_in_training = len(train_patients_with_tumors)
    if tumor_percentage > 0:
        target_control_cases = int(tumor_cases_in_training * (1 - tumor_percentage) / tumor_percentage)
    else:
        target_control_cases = 0
    
    print(f"\n🎯 Class Balance Calculation:")
    print(f"   Tumor cases available: {tumor_cases_in_training}")
    print(f"   Target control cases: {target_control_cases}")
    print(f"   Total control cases available: {len(controls)}")
    
    # Select control cases for training
    random.shuffle(controls)
    if target_control_cases > len(controls):
        print(f"⚠️  Warning: Not enough control cases. Using all {len(controls)} available.")
        train_controls = controls
    else:
        train_controls = controls[:target_control_cases]
    
    # Combine training cases
    train_cases = train_patients_with_tumors + train_patients_without_tumors + train_controls
    random.shuffle(train_cases)
    
    # Calculate actual tumor percentage
    actual_tumor_cases = len(train_patients_with_tumors)
    actual_total_cases = len(train_cases)
    actual_tumor_percentage = actual_tumor_cases / actual_total_cases if actual_total_cases > 0 else 0
    
    print(f"\n📊 Final Split:")
    print(f"   Training cases: {len(train_cases)}")
    print(f"     - Patients with tumors: {len(train_patients_with_tumors)}")
    print(f"     - Patients without tumors: {len(train_patients_without_tumors)}")
    print(f"     - Controls: {len(train_controls)}")
    print(f"     - Actual tumor percentage: {actual_tumor_percentage*100:.1f}% (target: {tumor_percentage*100:.1f}%)")
    
    print(f"   Validation cases: {len(val_patients)}")
    print(f"     - All patients (20% of total patients)")
    
    # Create the split structure
    stratified_split = [
        {
            'train': train_cases,
            'val': val_patients
        }
    ]
    
    # Save the split file
    splits_file = preprocessed_path / "splits_final.json"
    save_json(stratified_split, str(splits_file))
    
    print(f"\n✅ Stratified split file created: {splits_file}")
    print(f"   Split type: Stratified split with {tumor_percentage*100:.1f}% tumor target")
    print(f"   Validation: Patients only (20% of all patients)")
    
    return splits_file


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create stratified split for nnUNet v2 with class balance control")
    parser.add_argument("--dataset_id", type=int, default=1, help="Dataset ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tumor_percentage", type=float, default=0.75, 
                       help="Percentage of tumor cases in training set (0.0-1.0, default: 0.75)")
    
    args = parser.parse_args()
    
    # Validate tumor percentage
    if not 0.0 <= args.tumor_percentage <= 1.0:
        print(f"❌ Error: tumor_percentage must be between 0.0 and 1.0, got {args.tumor_percentage}")
        return
    
    try:
        # Create the stratified split
        splits_file = create_stratified_split(args.dataset_id, args.seed, args.tumor_percentage)
        
        print(f"\n🎉 Stratified split created successfully!")
        print(f"   You can now train with: nnUNetv2_train {args.dataset_id} 2d 0 --npz")
        print(f"   Training set has controlled class balance.")
        print(f"   Validation set contains only patients.")
        
    except Exception as e:
        print(f"❌ Error creating stratified split: {e}")
        raise


if __name__ == "__main__":
    main() 