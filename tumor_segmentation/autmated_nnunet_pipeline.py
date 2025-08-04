#!/usr/bin/env python3
"""
Automated nnUNet Pipeline Script

This script automates the complete nnUNet pipeline:
1. Create fresh ResEnc model with custom name
2. Fix ResEnc configuration to remove resampling with custom tiling
3. Run nnUNetv2_preprocess with the custom configuration
4. Create stratified split with custom percentage

All parameters are configurable at the top of the script.

MODEL SIZE OPTIONS:
- STANDARD: Uses default nnUNet architecture (7 stages, 32-512 features)
- SMALLER: Uses reduced architecture (5 stages, 16-256 features) - ~75% fewer parameters
- VERY SMALL: Uses minimal architecture (4 stages, 16-128 features) - ~90% fewer parameters

To use different sizes, modify the SMALLER_MODEL_CONFIG section below.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional
import random
import numpy as np
import cv2
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Model configuration
MODEL_NAME = "smallerv1"  # Name for the new ResEnc model
TILING_SIZE = (400, 400)  # Tiling size (width, height) in pixels
STRATIFIED_SPLIT_PERCENTAGE = 0.5  # Percentage for stratified split (0.0-1.0)


# Dataset configuration
DATASET_ID = 1  # Dataset ID (e.g., 1 for Dataset001_TumorSegmentation)
BATCH_SIZE = 24  # Batch size for training
RANDOM_SEED = 42  # Random seed for reproducibility

# =============================================================================
# MODEL SIZE CONFIGURATION - REDUCE THESE FOR SMALLER MODELS
# =============================================================================

# Smaller model architecture parameters
# Uncomment one of these configurations to use a smaller model
# SMALLER_MODEL_CONFIG = {
#     "n_stages": 5,                          # Reduced from 7 to 5 stages
#     "features_per_stage": [16, 32, 64, 128, 256],  # Reduced feature counts
#     "n_blocks_per_stage": [1, 2, 3, 4, 4],  # Reduced blocks per stage
#     "n_conv_per_stage_decoder": [1, 1, 1, 1]  # Reduced decoder convolutions
# }

# Alternative: Even smaller model (uncomment to use)
# SMALLER_MODEL_CONFIG = {
#     "n_stages": 4,                          # Very small: only 4 stages
#     "features_per_stage": [16, 32, 64, 128],  # Minimal features
#     "n_blocks_per_stage": [1, 2, 2, 3],     # Few blocks
#     "n_conv_per_stage_decoder": [1, 1, 1]   # Minimal decoder
# }

# Set to None to use standard model architecture
SMALLER_MODEL_CONFIG = None

# =============================================================================
# END OF CONFIGURABLE PARAMETERS
# =============================================================================


def create_fresh_resenc_plans(
    dataset_id: int,
    patch_size: Tuple[int, int],
    batch_size: int,
    config_name: str,
    smaller_model_config: Optional[dict] = None,
):
    """
    Create fresh ResEnc plans with optimized configuration.

    Args:
        dataset_id: Dataset ID (e.g., 1 for Dataset001_TumorSegmentation)
        patch_size: New patch size (height, width)
        batch_size: New batch size
        config_name: Name for the new configuration
        smaller_model_config: Optional dict with smaller model parameters
    """

    # Paths
    plans_dir = f"data_nnUNet/preprocessed/Dataset{dataset_id:03d}_TumorSegmentation"
    resenc_plans_path = f"{plans_dir}/nnUNetResEncUNetMPlans.json"

    # Check if ResEnc plans exist
    if not Path(resenc_plans_path).exists():
        print(f"❌ ResEnc plans not found: {resenc_plans_path}")
        print(
            "   Please run: nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM"
        )
        return False

    print(f"🔄 Creating fresh ResEnc configuration...")
    print(f"   ResEnc plans: {resenc_plans_path}")
    print(f"   New configuration: {config_name}")
    print(f"   Patch size: {patch_size}")
    print(f"   Batch size: {batch_size}")
    if smaller_model_config:
        print(f"   Architecture: ResidualEncoderUNet (SMALLER MODEL)")
        print(f"   Stages: {smaller_model_config['n_stages']}")
        print(f"   Features: {smaller_model_config['features_per_stage']}")
    else:
        print(f"   Architecture: ResidualEncoderUNet (improved)")

    # Load ResEnc plans
    with open(resenc_plans_path, "r") as f:
        plans = json.load(f)

    # Check if configuration already exists
    if config_name in plans["configurations"]:
        print(f"⚠️  Configuration '{config_name}' already exists. Overwriting...")

    # Check if original 2d configuration exists
    if "2d" not in plans["configurations"]:
        print("❌ Original '2d' configuration not found in ResEnc plans")
        return False

    original_config = plans["configurations"]["2d"]

    # Create fresh optimized configuration that inherits from ResEnc 2d
    optimized_config = {
        "inherits_from": "2d",
        "data_identifier": f"nnUNetPlans_{config_name}",
        "preprocessor_name": "DefaultPreprocessor",
        "batch_size": batch_size,
        "patch_size": list(patch_size),
        # Keep all other settings from original ResEnc config
        "median_image_size_in_voxels": original_config["median_image_size_in_voxels"],
        "spacing": original_config["spacing"],
        "normalization_schemes": original_config["normalization_schemes"],
        "use_mask_for_norm": original_config["use_mask_for_norm"],
        "batch_dice": original_config["batch_dice"],
    }

    # Create architecture configuration
    if smaller_model_config:
        # Use smaller model architecture
        architecture_config = {
            "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            "arch_kwargs": {
                "n_stages": smaller_model_config["n_stages"],
                "features_per_stage": smaller_model_config["features_per_stage"],
                "conv_op": "torch.nn.modules.conv.Conv2d",
                "kernel_sizes": [[3, 3]] * smaller_model_config["n_stages"],
                "strides": [[1, 1]] + [[2, 2]] * (smaller_model_config["n_stages"] - 1),
                "n_blocks_per_stage": smaller_model_config["n_blocks_per_stage"],
                "n_conv_per_stage_decoder": smaller_model_config[
                    "n_conv_per_stage_decoder"
                ],
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
        }
        optimized_config["architecture"] = architecture_config
    else:
        # Keep the improved ResidualEncoderUNet architecture
        optimized_config["architecture"] = original_config["architecture"]

    # Add the new configuration to ResEnc plans
    plans["configurations"][config_name] = optimized_config

    # Create backup before modifying
    backup_path = f"{plans_dir}/nnUNetResEncUNetMPlans_fresh_backup.json"
    with open(backup_path, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"✅ Backup created: {backup_path}")

    # Save modified ResEnc plans
    with open(resenc_plans_path, "w") as f:
        json.dump(plans, f, indent=2)

    print(f"✅ Fresh ResEnc configuration created successfully!")
    print(f"   New configuration '{config_name}' added to {resenc_plans_path}")
    if smaller_model_config:
        print(f"   Using SMALLER ResidualEncoderUNet architecture")
        print(
            f"   Parameters: {smaller_model_config['n_stages']} stages, {sum(smaller_model_config['features_per_stage'])} total features"
        )
    else:
        print(f"   Using improved ResidualEncoderUNet architecture")

    return True


def fix_resenc_no_resampling(
    dataset_id: int, patch_size: Tuple[int, int], batch_size: int, config_name: str
):
    """
    Fix ResEnc configuration to remove resampling.

    Args:
        dataset_id: Dataset ID (e.g., 1 for Dataset001_TumorSegmentation)
        patch_size: New patch size (height, width)
        batch_size: New batch size
        config_name: Name for the configuration to fix
    """

    # Paths
    plans_dir = f"data_nnUNet/preprocessed/Dataset{dataset_id:03d}_TumorSegmentation"
    resenc_plans_path = f"{plans_dir}/nnUNetResEncUNetMPlans.json"

    # Check if ResEnc plans exist
    if not Path(resenc_plans_path).exists():
        print(f"❌ ResEnc plans not found: {resenc_plans_path}")
        return False

    print(f"🔧 Fixing ResEnc configuration to remove resampling...")
    print(f"   ResEnc plans: {resenc_plans_path}")
    print(f"   Configuration: {config_name}")
    print(f"   Patch size: {patch_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Goal: Use original image dimensions (no resampling)")

    # Load ResEnc plans
    with open(resenc_plans_path, "r") as f:
        plans = json.load(f)

    # Check if configuration exists
    if config_name not in plans["configurations"]:
        print(f"❌ Configuration '{config_name}' not found in ResEnc plans")
        return False

    # Get the current configuration
    current_config = plans["configurations"][config_name]

    # Create fixed configuration that explicitly removes resampling
    fixed_config = {
        "inherits_from": "2d",
        "data_identifier": f"nnUNetPlans_{config_name}",
        "preprocessor_name": "DefaultPreprocessor",
        "batch_size": batch_size,
        "patch_size": list(patch_size),
        # Keep all other settings from original ResEnc config
        "median_image_size_in_voxels": current_config["median_image_size_in_voxels"],
        "spacing": current_config["spacing"],
        "normalization_schemes": current_config["normalization_schemes"],
        "use_mask_for_norm": current_config["use_mask_for_norm"],
        # Keep the improved ResidualEncoderUNet architecture
        "architecture": current_config["architecture"],
        "batch_dice": current_config["batch_dice"],
        # EXPLICITLY REMOVE RESAMPLING FUNCTIONS
        # DO NOT inherit resampling from base configuration
    }

    # Update the configuration
    plans["configurations"][config_name] = fixed_config

    # Create backup before modifying
    backup_path = f"{plans_dir}/nnUNetResEncUNetMPlans_no_resampling_backup.json"
    with open(backup_path, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"✅ Backup created: {backup_path}")

    # Save fixed ResEnc plans
    with open(resenc_plans_path, "w") as f:
        json.dump(plans, f, indent=2)

    print(f"✅ ResEnc configuration fixed successfully!")
    print(f"   Configuration '{config_name}' updated in {resenc_plans_path}")
    print(f"   ✅ Resampling functions REMOVED")
    print(f"   ✅ Original image dimensions preserved")
    print(f"   ✅ {patch_size[0]}×{patch_size[1]} patches will tile original images")

    return True


def run_nnunet_preprocess(dataset_id: int, config_name: str):
    """
    Run nnUNetv2_preprocess command.

    Args:
        dataset_id: Dataset ID
        config_name: Configuration name
    """
    print(f"🔄 Running nnUNetv2_preprocess...")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Configuration: {config_name}")
    print(f"   Plans: nnUNetResEncUNetMPlans")

    command = [
        "nnUNetv2_preprocess",
        "-d",
        str(dataset_id),
        "-c",
        config_name,
        "-p",
        "nnUNetResEncUNetMPlans",
    ]

    print(f"   Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ nnUNetv2_preprocess completed successfully!")
        print(f"   Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ nnUNetv2_preprocess failed!")
        print(f"   Error: {e.stderr}")
        return False


def create_stratified_split(dataset_id: int, seed: int, tumor_percentage: float):
    """
    Create a stratified train/validation split with class balance control.

    Args:
        dataset_id: Dataset ID
        seed: Random seed for reproducibility
        tumor_percentage: Percentage of tumor cases in training set (0.0-1.0)
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"🔧 Creating stratified split for Dataset{dataset_id:03d}")
    print(f"   Target tumor percentage in training: {tumor_percentage * 100:.1f}%")

    # Set up paths
    nnUNet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnUNet_preprocessed:
        # Set environment variables if not already set
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_preprocessed = str(
            project_dir / "tumor_segmentation" / "data_nnUNet" / "preprocessed"
        )
        os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
        print(f"⚠️  Set nnUNet_preprocessed to: {nnUNet_preprocessed}")

    # Dataset paths
    dataset_name = f"Dataset{dataset_id:03d}_TumorSegmentation"
    preprocessed_path = Path(str(nnUNet_preprocessed)) / dataset_name

    if not preprocessed_path.exists():
        print(f"⚠️  Preprocessed directory not found: {preprocessed_path}")
        print("   This is expected if preprocessing hasn't been run yet.")

    # Get dataset path for raw data
    nnUNet_raw = os.environ.get("nnUNet_raw")
    if not nnUNet_raw:
        # Set environment variables if not already set
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_raw = str(project_dir / "tumor_segmentation" / "data_nnUNet")
        os.environ["nnUNet_raw"] = nnUNet_raw
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
    patients = [case for case in all_cases if case.startswith("patient_")]
    controls = [case for case in all_cases if case.startswith("control_")]

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
        raise ValueError(
            "No patients with tumors found! Cannot create meaningful split."
        )

    # Create validation set: 20% of all patients (with and without tumors)
    all_patients = patients_with_tumors + patients_without_tumors
    random.shuffle(all_patients)
    val_split_point = int(0.2 * len(all_patients))
    val_patients = all_patients[:val_split_point]
    train_patients_available = all_patients[val_split_point:]

    # Separate available training patients by tumor status
    train_patients_with_tumors = [
        p for p in train_patients_available if p in patients_with_tumors
    ]
    train_patients_without_tumors = [
        p for p in train_patients_available if p in patients_without_tumors
    ]

    print(f"\n📊 Patient Split:")
    print(f"   Validation patients: {len(val_patients)} (20% of all patients)")
    print(
        f"   Available training patients with tumors: {len(train_patients_with_tumors)}"
    )
    print(
        f"   Available training patients without tumors: {len(train_patients_without_tumors)}"
    )

    # Calculate how many control cases we need to achieve the target tumor percentage
    tumor_cases_in_training = len(train_patients_with_tumors)
    if tumor_percentage > 0:
        target_control_cases = int(
            tumor_cases_in_training * (1 - tumor_percentage) / tumor_percentage
        )
    else:
        target_control_cases = 0

    print(f"\n🎯 Class Balance Calculation:")
    print(f"   Tumor cases available: {tumor_cases_in_training}")
    print(f"   Target control cases: {target_control_cases}")
    print(f"   Total control cases available: {len(controls)}")

    # Select control cases for training
    random.shuffle(controls)
    if target_control_cases > len(controls):
        print(
            f"⚠️  Warning: Not enough control cases. Using all {len(controls)} available."
        )
        train_controls = controls
    else:
        train_controls = controls[:target_control_cases]

    # Combine training cases
    train_cases = (
        train_patients_with_tumors + train_patients_without_tumors + train_controls
    )
    random.shuffle(train_cases)

    # Calculate actual tumor percentage
    actual_tumor_cases = len(train_patients_with_tumors)
    actual_total_cases = len(train_cases)
    actual_tumor_percentage = (
        actual_tumor_cases / actual_total_cases if actual_total_cases > 0 else 0
    )

    print(f"\n📊 Final Split:")
    print(f"   Training cases: {len(train_cases)}")
    print(f"     - Patients with tumors: {len(train_patients_with_tumors)}")
    print(f"     - Patients without tumors: {len(train_patients_without_tumors)}")
    print(f"     - Controls: {len(train_controls)}")
    print(
        f"     - Actual tumor percentage: {actual_tumor_percentage * 100:.1f}% (target: {tumor_percentage * 100:.1f}%)"
    )

    print(f"   Validation cases: {len(val_patients)}")
    print(f"     - All patients (20% of total patients)")

    # Create the split structure
    stratified_split = [{"train": train_cases, "val": val_patients}]

    # Save the split file
    splits_file = preprocessed_path / "splits_final.json"
    save_json(stratified_split, str(splits_file))

    print(f"\n✅ Stratified split file created: {splits_file}")
    print(
        f"   Split type: Stratified split with {tumor_percentage * 100:.1f}% tumor target"
    )
    print(f"   Validation: Patients only (20% of all patients)")

    return splits_file


def main():
    """Main function that runs the complete pipeline."""
    print("🚀 Automated nnUNet Pipeline")
    print("=" * 50)
    print(f"Model Name: {MODEL_NAME}")
    print(f"Tiling Size: {TILING_SIZE[0]}×{TILING_SIZE[1]} pixels")
    print(f"Stratified Split: {STRATIFIED_SPLIT_PERCENTAGE * 100:.1f}%")
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Random Seed: {RANDOM_SEED}")
    if SMALLER_MODEL_CONFIG:
        print(
            f"Model Size: SMALLER ({SMALLER_MODEL_CONFIG['n_stages']} stages, {sum(SMALLER_MODEL_CONFIG['features_per_stage'])} features)"
        )
    else:
        print(f"Model Size: STANDARD")
    print()

    # Step 1: Create fresh ResEnc model
    print("📋 Step 1: Creating fresh ResEnc model...")
    if not create_fresh_resenc_plans(
        DATASET_ID, TILING_SIZE, BATCH_SIZE, MODEL_NAME, SMALLER_MODEL_CONFIG
    ):
        print("❌ Failed to create fresh ResEnc model. Exiting.")
        return False
    print()

    # Step 2: Fix ResEnc configuration to remove resampling
    print("📋 Step 2: Fixing ResEnc configuration (no resampling)...")
    if not fix_resenc_no_resampling(DATASET_ID, TILING_SIZE, BATCH_SIZE, MODEL_NAME):
        print("❌ Failed to fix ResEnc configuration. Exiting.")
        return False
    print()

    # Step 3: Run nnUNetv2_preprocess
    print("📋 Step 3: Running nnUNetv2_preprocess...")
    if not run_nnunet_preprocess(DATASET_ID, MODEL_NAME):
        print("❌ Failed to run nnUNetv2_preprocess. Exiting.")
        return False
    print()

    # Step 4: Create stratified split
    print("📋 Step 4: Creating stratified split...")
    try:
        splits_file = create_stratified_split(
            DATASET_ID, RANDOM_SEED, STRATIFIED_SPLIT_PERCENTAGE
        )
        print(f"✅ Stratified split created: {splits_file}")
    except Exception as e:
        print(f"❌ Failed to create stratified split: {e}")
        return False
    print()

    # Success message
    print("🎉 Pipeline completed successfully!")
    print("=" * 50)
    print(f"✅ Fresh ResEnc model '{MODEL_NAME}' created")
    if SMALLER_MODEL_CONFIG:
        print(
            f"✅ SMALLER model architecture ({SMALLER_MODEL_CONFIG['n_stages']} stages, {sum(SMALLER_MODEL_CONFIG['features_per_stage'])} features)"
        )
    print(
        f"✅ Configuration fixed (no resampling, {TILING_SIZE[0]}×{TILING_SIZE[1]} tiling)"
    )
    print(f"✅ Preprocessing completed")
    print(
        f"✅ Stratified split created ({STRATIFIED_SPLIT_PERCENTAGE * 100:.1f}% tumor target)"
    )
    print()

    print("📋 Next steps:")
    print("=" * 50)
    print(f"# Train the model:")
    print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 0 -p nnUNetResEncUNetMPlans --npz")
    print()
    print(f"# Validate the model:")
    print(
        f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 0 -p nnUNetResEncUNetMPlans --val --npz"
    )
    print()
    print(f"# Find best configuration:")
    print(
        f"nnUNetv2_find_best_configuration {DATASET_ID} -c {MODEL_NAME} -p nnUNetResEncUNetMPlans"
    )
    print()
    print(f"# Run inference:")
    print(
        f"nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d {DATASET_ID} -c {MODEL_NAME} -p nnUNetResEncUNetMPlans"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
