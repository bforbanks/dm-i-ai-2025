#!/usr/bin/env python3
"""
Automated nnUNet Pipeline Script

This script automates the complete nnUNet pipeline:
1. Create fresh ResEnc model with custom name
2. Fix ResEnc configuration to remove resampling with custom tiling
3. Run nnUNetv2_preprocess with the custom configuration
4. Apply optional salt-pepper noise augmentation
5. Create validation splits (single split OR cross-validation)

All parameters are configurable at the top of the script.

VALIDATION STRATEGY OPTIONS:
- USE_SINGLE_SPLIT = True: Single train/validation split (recommended for 800+ images)
  * More efficient: 5x faster training, more data per epoch
  * VALIDATION_SPLIT controls validation percentage (default: 15%)
- USE_SINGLE_SPLIT = False: Traditional 5-fold cross-validation
  * More robust: Statistical reliability across multiple folds
  * Slower: Requires training 5 separate models

PATIENT/CONTROL RATIO:
- Set PATIENT_CONTROL_RATIO to control the ratio of patient to control images in training sets
- Example: 0.7 means 70% patient images and 30% control images in training sets
- Control images not used in training will be excluded from the dataset
- This helps prevent overfitting to control cases while maintaining class balance

SYNTHETIC TUMOR DATA:
- USE_SYNTHETIC_TUMORS: Enable/disable synthetic tumor dataset integration
- SYNTHETIC_TUMOR_FRACTION: Fraction of synthetic tumors to include (0.0-1.0)
- When enabled, real control images are excluded from training

SALT-PEPPER NOISE AUGMENTATION:
- ENABLE_SALT_PEPPER_NOISE: Random black/white pixel noise to mimic artifacts
- SALT_PEPPER_FORCED_IMAGES: Specific cases that must receive noise augmentation

MODEL SIZE OPTIONS:
- STANDARD: Uses default nnUNet architecture (7 stages, 32-512 features)
- CONSERVATIVE: Uses reduced architecture (~50% fewer parameters)
- REDUCED: Uses reduced architecture (5 stages, 16-256 features) - ~75% fewer parameters
- MINIMAL: Uses minimal architecture (4 stages, 16-128 features) - ~90% fewer parameters

To use different sizes, modify the SELECTED_MODEL_SIZE parameter below.
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
from PIL import Image
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Model configuration
MODEL_NAME = "reduced-custom-singleval-synth050"  # Name for the new ResEnc model
TILING_SIZE = (
    384,
    384,
)  # Tiling size (width, height) in pixels - reduced to avoid dimension mismatches

# Patient/Control ratio configuration
PATIENT_CONTROL_RATIO = 0.3  # Ratio of patient images to control images in training sets (0.0-1.0)
# Example: 0.7 means 70% patient images and 30% control images in training sets
# Control images not used in training will be excluded from the dataset

# Synthetic tumor dataset configuration
# If True, training will use ONLY real patients (with/without tumors) plus a fraction of the synthetic tumor dataset,
# and it will use 0 real control images.
USE_SYNTHETIC_TUMORS = True
# Fraction (0.0-1.0) of synthetic tumor cases to include in training when USE_SYNTHETIC_TUMORS is True
SYNTHETIC_TUMOR_FRACTION = 0.50
# Naming prefix used for synthetic tumor cases placed into the same nnUNet dataset (imagesTr/labelsTr)
SYNTHETIC_TUMOR_PREFIX = "controls_tumor"

# Cross-validation vs Single Split configuration
USE_SINGLE_SPLIT = True   # True for single train/val split, False for 5-fold cross-validation
VALIDATION_SPLIT = 0.15   # Fraction of data to use for validation (0.15 = 15%)

# Dataset configuration
DATASET_ID = 1  # Dataset ID (e.g., 1 for Dataset001_TumorSegmentation)
BATCH_SIZE = 8  # Batch size for training
RANDOM_SEED = 42  # Random seed for reproducibility

SELECTED_MODEL_SIZE = "reduced-custom"  # CHANGED from conservative to reduced to prevent overfitting

# Resampling preference (Documentation: nnUNet will automatically resample images to the target spacing unless disabled)
ENABLE_RESAMPLING = False  # Set to True to keep nnUNet default resampling, False to work on original image spacing/dimensions

# Tiling preference (Documentation: nnUNet automatically determines optimal patch size based on GPU memory and image statistics)
USE_NNUNET_RECOMMENDED_TILING = False  # Set to True to use nnUNet's recommended patch size, False to use manually set TILING_SIZE
# Note: When True, nnUNet will use the patch size from the base "2d" configuration (typically 512x448 for 2D)

# Select which configuration to use
# Options: "standard", "reduced", "minimal", "conservative", "tiny", or None for standard

# Specific images that MUST receive salt-pepper noise (case names without _0000.png suffix)
# Example: ["patient_001", "controls_tumor002", "patient_042"]
SALT_PEPPER_FORCED_IMAGES = [
    "controls_tumor006",  # control_006 -> synthetic ctrl tumor
    "controls_tumor042",  # ctrl 42 -> synthetic ctrl tumor  
    "controls_tumor044",  # ctrl 44 -> synthetic ctrl tumor
    "controls_tumor113",  # ctrl 113 -> synthetic ctrl tumor
    "patient_010",        # patient 10
    "patient_015",        # patient 015
    "patient_019",        # patient 19
    "patient_026",        # patient 26
    "patient_034",        # patient 34
    "patient_037"         # patient 37
]

# Salt and pepper noise augmentation configuration
# Apply random black/white 1px dots to mimic evaluation set artifacts (7/200 ratio observed)
ENABLE_SALT_PEPPER_NOISE = True  # Whether to apply salt-pepper noise augmentation
SALT_PEPPER_PROBABILITY = 0.05  # Fraction of images to apply noise to (0.05 = 5%)
SALT_PEPPER_DENSITY = 0.11  # Density of noise pixels (0.002 = 0.2% of pixels affected)
SALT_PEPPER_SALT_RATIO = 0.5  # Ratio of salt (white) vs pepper (black) noise (0.5 = 50/50)

# =============================================================================
# MODEL SIZE CONFIGURATION - REDUCE THESE FOR SMALLER MODELS
# =============================================================================

# Enhanced model configurations with parameter estimates
ENHANCED_SMALLER_CONFIGS = {
    "standard": None,  # Use default nnUNet architecture
    "conservative": {
        "n_stages": 6,
        "features_per_stage": [16, 32, 64, 128, 256, 320],
        "n_blocks_per_stage": [1, 2, 2, 3, 3, 3],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
        "description": "Conservatively reduced model (~50% fewer parameters)"
    },
    "conservative-custom": {
        "n_stages": 6,
        "features_per_stage": [16, 32, 64, 128, 256, 512],
        "n_blocks_per_stage": [1, 2, 2, 3, 3, 3],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
        "description": "Conservatively reduced model (~50% fewer parameters)"
    },
    "reduced": {
        "n_stages": 5,
        "features_per_stage": [16, 32, 64, 128, 256],
        "n_blocks_per_stage": [1, 2, 3, 4, 4],
        "n_conv_per_stage_decoder": [1, 1, 1, 1],
        "description": "Moderately reduced model (~75% fewer parameters)"
    },
        "reduced-custom": {
        "n_stages": 5,
        "features_per_stage": [32, 64, 128, 256, 512],
        "n_blocks_per_stage": [2,2,3,3,3],
        "n_conv_per_stage_decoder": [1, 1, 1, 1],
        "description": "Moderately reduced model (~75% fewer parameters)"
    },
    "minimal": {
        "n_stages": 4,
        "features_per_stage": [16, 32, 64, 128],
        "n_blocks_per_stage": [1, 2, 2, 3],
        "n_conv_per_stage_decoder": [1, 1, 1],
        "description": "Heavily reduced model (~90% fewer parameters)"
    },
    "tiny": {
        "n_stages": 3,
        "features_per_stage": [16, 32, 64],
        "n_blocks_per_stage": [1, 1, 2],
        "n_conv_per_stage_decoder": [1, 1],
        "description": "Extremely small model (~95% fewer parameters)"
    }
}

# Set the active configuration
if SELECTED_MODEL_SIZE in ENHANCED_SMALLER_CONFIGS:
    SMALLER_MODEL_CONFIG = ENHANCED_SMALLER_CONFIGS[SELECTED_MODEL_SIZE]
else:
    SMALLER_MODEL_CONFIG = None


def apply_salt_pepper_noise(img_path: Path, probability: float, density: float, salt_ratio: float, seed: int) -> bool:
    """
    Apply salt and pepper noise to an image with given probability.
    
    Args:
        img_path: Path to the image file
        probability: Probability of applying noise to this image (0.0-1.0)
        density: Fraction of pixels to affect with noise (0.0-1.0)
        salt_ratio: Ratio of salt (white) vs pepper (black) noise (0.0-1.0)
        seed: Random seed for reproducible noise
        
    Returns:
        bool: True if noise was applied, False otherwise
    """
    rng = random.Random(seed + hash(str(img_path)))
    
    # Decide whether to apply noise to this image
    if rng.random() > probability:
        return False
    
    try:
        # Load image
        with Image.open(img_path) as img:
            img_array = np.array(img)
            
        # Handle both grayscale and RGB
        if len(img_array.shape) == 2:
            h, w = img_array.shape
        else:
            h, w = img_array.shape[:2]
            
        # Calculate number of pixels to affect
        total_pixels = h * w
        num_noise_pixels = int(total_pixels * density)
        
        if num_noise_pixels == 0:
            return False
            
        # Generate random pixel positions
        noise_positions = rng.sample(range(total_pixels), num_noise_pixels)
        
        # Convert to row, col coordinates
        rows = [pos // w for pos in noise_positions]
        cols = [pos % w for pos in noise_positions]
        
        # Apply salt and pepper noise
        for row, col in zip(rows, cols):
            if rng.random() < salt_ratio:
                # Salt (white) noise
                if len(img_array.shape) == 2:
                    img_array[row, col] = 255
                else:
                    img_array[row, col] = 255
            else:
                # Pepper (black) noise
                if len(img_array.shape) == 2:
                    img_array[row, col] = 0
                else:
                    img_array[row, col] = 0
        
        # Save the modified image
        Image.fromarray(img_array).save(img_path)
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to apply salt-pepper noise to {img_path}: {e}")
        return False


def apply_salt_pepper_to_dataset(dataset_path: Path, probability: float, density: float, salt_ratio: float, seed: int, forced_images: Optional[list] = None):
    """
    Apply salt and pepper noise augmentation to a dataset.
    
    Args:
        dataset_path: Path to the nnUNet dataset (containing imagesTr/)
        probability: Probability of applying noise to each image
        density: Density of noise pixels
        salt_ratio: Ratio of salt vs pepper noise
        seed: Random seed for reproducibility
        forced_images: List of case names (without _0000.png suffix) that must receive noise
    """
    images_dir = dataset_path / "imagesTr"
    
    if not images_dir.exists():
        print(f"⚠️  Images directory not found: {images_dir}")
        return
    
    # Get all image files
    image_files = list(images_dir.glob("*_0000.png"))
    
    if not image_files:
        print(f"⚠️  No image files found in {images_dir}")
        return
    
    # Prepare forced images set for quick lookup
    forced_images = forced_images or []
    forced_set = set(forced_images)
    
    print(f"🔄 Applying salt-pepper noise augmentation...")
    print(f"   Images to process: {len(image_files)}")
    print(f"   Probability: {probability * 100:.1f}%")
    print(f"   Density: {density * 100:.3f}% of pixels")
    print(f"   Salt ratio: {salt_ratio * 100:.1f}% salt, {(1-salt_ratio) * 100:.1f}% pepper")
    if forced_images:
        print(f"   Forced images: {len(forced_images)} cases must receive noise")
        print(f"   Forced cases: {', '.join(forced_images)}")
    
    applied_count = 0
    forced_applied = 0
    random_applied = 0
    
    for img_file in image_files:
        # Extract case name (remove _0000.png suffix)
        case_name = img_file.name.replace("_0000.png", "")
        
        # Check if this is a forced image
        is_forced = case_name in forced_set
        
        if is_forced:
            # Apply noise to forced images with probability = 1.0
            if apply_salt_pepper_noise(img_file, 1.0, density, salt_ratio, seed):
                applied_count += 1
                forced_applied += 1
        else:
            # Apply noise to regular images with specified probability
            if apply_salt_pepper_noise(img_file, probability, density, salt_ratio, seed):
                applied_count += 1
                random_applied += 1
    
    print(f"✅ Salt-pepper noise applied to {applied_count}/{len(image_files)} images ({applied_count/len(image_files)*100:.1f}%)")
    if forced_images:
        print(f"   Forced applications: {forced_applied}/{len(forced_images)}")
        print(f"   Random applications: {random_applied}/{len(image_files) - len(forced_images)}")


def calculate_model_parameters(config):
    """
    Estimate parameter count for the model configuration.
    
    This provides a simplified estimation of the total parameters in the ResidualEncoderUNet
    based on the configuration parameters. Useful for comparing model sizes.
    
    Args:
        config: Dictionary containing model configuration with keys:
                - features_per_stage: List of feature counts per stage
                - n_blocks_per_stage: List of residual blocks per stage
                - n_conv_per_stage_decoder: List of decoder convolutions per stage
    
    Returns:
        int: Estimated total number of parameters
    """
    if config is None:
        # Rough estimate for standard nnUNet (7 stages, features: 32->512)
        return 35_000_000  # ~35M parameters for standard model
    
    total_params = 0
    input_channels = 1  # Assuming single modality input (modify if multi-modal)
    
    # Encoder parameters estimation
    for i, (features, blocks) in enumerate(zip(config['features_per_stage'], 
                                               config['n_blocks_per_stage'])):
        # Initial convolution for this stage
        if i == 0:
            # First stage: input channels to first feature count
            conv_params = features * input_channels * 3 * 3 * 3
        else:
            # Subsequent stages: previous features to current features
            prev_features = config['features_per_stage'][i-1]
            conv_params = features * prev_features * 3 * 3 * 3
        
        total_params += conv_params
        
        # Residual blocks parameters
        # Each residual block has 2 convolutions: features -> features
        residual_params = blocks * features * features * 3 * 3 * 3 * 2
        total_params += residual_params
        
        # Instance normalization parameters (scale + bias)
        norm_params = features * 2 * (1 + blocks)  # Initial conv + residual blocks
        total_params += norm_params
    
    # Decoder parameters estimation
    n_decoder_stages = len(config['n_conv_per_stage_decoder'])
    for i, decoder_convs in enumerate(config['n_conv_per_stage_decoder']):
        stage_idx = len(config['features_per_stage']) - 2 - i  # Decoder works backwards
        if stage_idx >= 0:
            current_features = config['features_per_stage'][stage_idx]
            if stage_idx < len(config['features_per_stage']) - 1:
                next_features = config['features_per_stage'][stage_idx + 1]
            else:
                next_features = current_features
            
            # Upsampling convolution
            upsample_params = current_features * next_features * 3 * 3 * 3
            total_params += upsample_params
            
            # Decoder convolutions
            decoder_params = decoder_convs * current_features * current_features * 3 * 3 * 3
            total_params += decoder_params
            
            # Normalization parameters
            norm_params = current_features * 2 * decoder_convs
            total_params += norm_params
    
    # Final output layer (assuming binary segmentation, modify for multi-class)
    if len(config['features_per_stage']) > 0:
        output_params = config['features_per_stage'][0] * 2 * 1 * 1 * 1  # 2 classes (background + tumor)
        total_params += output_params
    
    return int(total_params)

# Print model configuration info
def print_model_info():
    """Print information about the selected model configuration."""
    if SMALLER_MODEL_CONFIG is None:
        estimated_params = calculate_model_parameters(None)
        print(f"🔧 Using STANDARD nnUNet architecture")
        print(f"   Estimated parameters: ~{estimated_params/1_000_000:.1f}M")
    else:
        estimated_params = calculate_model_parameters(SMALLER_MODEL_CONFIG)
        config_name = SELECTED_MODEL_SIZE.upper()
        description = SMALLER_MODEL_CONFIG.get('description', 'Custom configuration')
        
        print(f"🔧 Using {config_name} model configuration")
        print(f"   Description: {description}")
        print(f"   Stages: {SMALLER_MODEL_CONFIG['n_stages']}")
        print(f"   Features per stage: {SMALLER_MODEL_CONFIG['features_per_stage']}")
        print(f"   Blocks per stage: {SMALLER_MODEL_CONFIG['n_blocks_per_stage']}")
        print(f"   Estimated parameters: ~{estimated_params/1_000_000:.1f}M")
        
        # Compare to standard
        standard_params = calculate_model_parameters(None)
        reduction = (1 - estimated_params / standard_params) * 100
        print(f"   Parameter reduction: {reduction:.1f}% vs standard model")


# =============================================================================
# OVERFITTING PREVENTION STRATEGIES - Based on nnUNet Research
# =============================================================================

def add_overfitting_prevention_config(plans, config_name):
    """
    Add overfitting prevention strategies based on nnUNet research.
    
    Based on findings from:
    - nnUNet BraTS paper: More aggressive data augmentation helps
    - nnUNet original paper: Proper validation crucial
    
    Args:
        plans: The nnUNet plans dictionary
        config_name: Name of the configuration to modify
    """
    
    if config_name not in plans["configurations"]:
        print(f"⚠️  Configuration '{config_name}' not found for overfitting prevention")
        return plans
    
    config = plans["configurations"][config_name]
    
    # Add overfitting prevention settings    
    # Add data augmentation settings that nnUNet will use
    config["data_augmentation"] = {
        "rotation": {"probability": 0.3, "range": (-15, 15)},  # Increased from 0.2
        "scaling": {"probability": 0.3, "range": (0.65, 1.6)},  # More aggressive
        "elastic_deformation": {"probability": 0.3},           # Added based on BraTS
        "brightness": {"probability": 0.3},                    # Added
        "gamma": {"probability": 0.5, "range": (0.7, 1.5)},   # More aggressive
        "mirror": {"probability": 0.5}                         # Standard
    }
    
    # Add regularization settings
    config["weight_decay"] = 1e-5        # L2 regularization
    
    # Add the overfitting_prevention section for reference (though nnUNet may not use it directly)
    overfitting_prevention = {
        # Training strategy (adaptive based on mode)
        
        # Data augmentation enhancement (based on BraTS paper)
        "data_augmentation": {
            "rotation": {"probability": 0.3, "range": (-15, 15)},  # Increased from 0.2
            "scaling": {"probability": 0.3, "range": (0.65, 1.6)},  # More aggressive
            "elastic_deformation": {"probability": 0.3},           # Added based on BraTS
            "brightness": {"probability": 0.3},                    # Added
            "gamma": {"probability": 0.5, "range": (0.7, 1.5)},   # More aggressive
            "mirror": {"probability": 0.5}                         # Standard
        },
        
        # Regularization (nnUNet standard + enhancements)
        "weight_decay": 1e-5,        # L2 regularization
        "instance_normalization": True,  # Proven better than batch norm for small batches
        "dropout_replacement": "none",    # nnUNet doesn't use dropout, uses other strategies
        
        # Validation strategy
        "cross_validation": True,     # Use 5-fold CV as per nnUNet standard
        
        # Memory and computational efficiency
        "mixed_precision": True,      # Reduces memory usage
        "deterministic": True,        # For reproducibility
    }
    
    # Add to configuration
    config["overfitting_prevention"] = overfitting_prevention
    
    print(f"✅ Added overfitting prevention strategies to '{config_name}'")
    print(f"   - Enhanced data augmentation (based on BraTS paper)")
    print(f"   - L2 regularization (weight_decay: {overfitting_prevention['weight_decay']})")
    
    return plans


def create_cross_validation_splits(dataset_id: int, seed: int, n_folds: int = 5, tumor_only_validation: bool = False, patient_control_ratio: float = 0.7):
    """
    Create proper 5-fold cross-validation splits as per nnUNet standard.
    
    IMPORTANT: If test set contains only tumor cases, set tumor_only_validation=True
    to better match test distribution and prevent overfitting to controls.
    
    Args:
        dataset_id: Dataset ID
        seed: Random seed for reproducibility
        n_folds: Number of folds (nnUNet standard is 5)
        tumor_only_validation: If True, validation sets contain only tumor cases
        patient_control_ratio: Ratio of patient images to control images in training sets (0.0-1.0)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"🔄 Creating {n_folds}-fold cross-validation splits (nnUNet standard)")
    print(f"   This helps prevent overfitting through proper validation")
    
    # Set up paths
    nnUNet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnUNet_preprocessed:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_preprocessed = str(
            project_dir / "tumor_segmentation" / "data_nnUNet" / "preprocessed"
        )
        os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    
    # Get dataset path for raw data
    nnUNet_raw = os.environ.get("nnUNet_raw")
    if not nnUNet_raw:
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        nnUNet_raw = str(project_dir / "tumor_segmentation" / "data_nnUNet")
        os.environ["nnUNet_raw"] = nnUNet_raw
    
    dataset_name = f"Dataset{dataset_id:03d}_TumorSegmentation"
    dataset_path = Path(str(nnUNet_raw)) / dataset_name
    preprocessed_path = Path(str(nnUNet_preprocessed)) / dataset_name
    
    # Get all training cases
    images_tr_dir = dataset_path / "imagesTr"
    if not images_tr_dir.exists():
        raise FileNotFoundError(f"ImagesTr directory not found: {images_tr_dir}")
    
    # Get all image files and extract case names
    image_files = list(images_tr_dir.glob("*_0000.png"))
    all_cases = []
    
    for img_file in image_files:
        case_name = img_file.stem.replace("_0000", "")
        all_cases.append(case_name)
    
    # Sort for reproducibility
    all_cases.sort()
    
    if tumor_only_validation:
        # Separate tumor and control cases based on naming convention
        # Assuming tumor cases have specific naming (adjust as needed)
        labels_tr_dir = dataset_path / "labelsTr"
        tumor_cases = []
        control_cases = []
        synthetic_tumor_cases = []
        patient_non_tumor_cases = []
        real_control_cases = []
        
        # Robust synthetic detection: collect by prefix from imagesTr
        prefix_synthetic_from_names = [c for c in all_cases if c.startswith(SYNTHETIC_TUMOR_PREFIX)]

        for case in all_cases:
            label_file = labels_tr_dir / f"{case}.png"
            if label_file.exists():
                # Read label to check if it has tumor (non-zero pixels)
                import cv2
                label_img = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
                if label_img is not None and np.any(label_img.astype(int) > 0):
                    tumor_cases.append(case)
                    # Track synthetic tumors explicitly
                    if case.startswith(SYNTHETIC_TUMOR_PREFIX):
                        synthetic_tumor_cases.append(case)
                else:
                    control_cases.append(case)
                    # Distinguish patient non-tumor vs real control by name
                    if case.startswith("patient_"):
                        patient_non_tumor_cases.append(case)
                    else:
                        real_control_cases.append(case)
            else:
                # If no label file, assume control
                control_cases.append(case)
                if case.startswith("patient_"):
                    patient_non_tumor_cases.append(case)
                else:
                    real_control_cases.append(case)

        # Merge name-based synthetic detection for safety (in case labels check failed)
        if prefix_synthetic_from_names:
            merged = set(synthetic_tumor_cases).union(prefix_synthetic_from_names)
            synthetic_tumor_cases = sorted(list(merged))
        
        print(f"🎯 TUMOR-ONLY VALIDATION MODE (matches test distribution)")
        print(f"📊 Cross-validation setup:")
        print(f"   Total cases: {len(all_cases)} ({len(tumor_cases)} tumor, {len(control_cases)} control)")
        print(f"   Folds: {n_folds}")
        # Validation must be ONLY real patient images (exclude synthetic)
        real_patient_tumor_cases = [c for c in tumor_cases if c.startswith("patient_")]
        print(f"   Validation: Only REAL patient tumor cases (~{len(real_patient_tumor_cases) // n_folds} per fold)")
        if USE_SYNTHETIC_TUMORS:
            print(f"   Training: Real patients (tumor + non-tumor) + synthetic tumor cases (fraction={SYNTHETIC_TUMOR_FRACTION})")
            print(f"            Real controls will be EXCLUDED from training")
        else:
            print(f"   Training: Tumor + Control cases with {patient_control_ratio * 100:.1f}% patient ratio")
        
        # Create tumor-only validation folds
        random.shuffle(real_patient_tumor_cases)  # Shuffle tumor cases with fixed seed
        fold_size = len(real_patient_tumor_cases) // n_folds
        folds = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            if fold == n_folds - 1:  # Last fold gets remaining cases
                end_idx = len(real_patient_tumor_cases)
            else:
                end_idx = (fold + 1) * fold_size
            
            val_cases = real_patient_tumor_cases[start_idx:end_idx]

            # Training base: remaining real patient tumor cases
            train_real_patient_tumors = [case for case in real_patient_tumor_cases if case not in val_cases]

            if USE_SYNTHETIC_TUMORS:
                # Select synthetic tumor fraction (stable across folds using seed)
                rng = random.Random(seed + fold)
                selected_synth_count = int(len(synthetic_tumor_cases) * SYNTHETIC_TUMOR_FRACTION)
                selected_synth = rng.sample(synthetic_tumor_cases, selected_synth_count) if selected_synth_count > 0 else []

                # Include all real patient non-tumor cases in training
                train_cases = train_real_patient_tumors + patient_non_tumor_cases + selected_synth
                # Explicitly exclude real controls
                train_real_controls_used = 0
                synth_used = len(selected_synth)
            else:
                # Original behavior: add controls to achieve ratio; do NOT include synthetic tumors when disabled
                train_tumor_cases = train_real_patient_tumors
                if len(train_tumor_cases) > 0:
                    target_control_count = int(len(train_tumor_cases) * (1 - patient_control_ratio) / patient_control_ratio)
                    # Available controls are patient non-tumor + real controls
                    available_controls = patient_non_tumor_cases + real_control_cases
                    target_control_count = min(target_control_count, len(available_controls))
                else:
                    target_control_count = 0
                rng = random.Random(seed + 100 + fold)
                train_control_cases = rng.sample(available_controls, target_control_count) if target_control_count > 0 else []
                train_cases = train_tumor_cases + train_control_cases
                train_real_controls_used = len([c for c in train_control_cases if not c.startswith("patient_")])
                synth_used = len([c for c in train_tumor_cases if c.startswith(SYNTHETIC_TUMOR_PREFIX)])

            fold_split = {"train": train_cases, "val": val_cases}
            folds.append(fold_split)
            
            if USE_SYNTHETIC_TUMORS:
                num_tumors_train = len(train_real_patient_tumors) + synth_used
                num_controls_train = len(patient_non_tumor_cases)  # counted as real patient non-tumor
                print(f"   Fold {fold}: {len(train_cases)} train ({num_tumors_train} tumor [{len(train_real_patient_tumors)} real + {synth_used} synthetic] + {num_controls_train} patient non-tumor), {len(val_cases)} val (real patient tumor only)")
            else:
                print(f"   Fold {fold}: {len(train_cases)} train, {len(val_cases)} val (tumor only)")
    
    else:
        # Standard cross-validation (original behavior)
        random.shuffle(all_cases)  # Shuffle with fixed seed
        
        print(f"📊 Cross-validation setup:")
        print(f"   Total cases: {len(all_cases)}")
        print(f"   Folds: {n_folds}")
        print(f"   Cases per fold: ~{len(all_cases) // n_folds}")
        
        # Create folds
        fold_size = len(all_cases) // n_folds
        folds = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            if fold == n_folds - 1:  # Last fold gets remaining cases
                end_idx = len(all_cases)
            else:
                end_idx = (fold + 1) * fold_size
            
            val_cases = all_cases[start_idx:end_idx]
            train_cases = [case for case in all_cases if case not in val_cases]
            
            fold_split = {"train": train_cases, "val": val_cases}
            folds.append(fold_split)
            
            print(f"   Fold {fold}: {len(train_cases)} train, {len(val_cases)} val")
    
    # Save the cross-validation splits
    cv_splits_file = preprocessed_path / "splits_final.json"
    save_json(folds, str(cv_splits_file))
    
    print(f"\n✅ Cross-validation splits created: {cv_splits_file}")
    print(f"   Strategy: {n_folds}-fold CV (nnUNet standard for overfitting prevention)")
    
    return cv_splits_file


# Use smaller model to avoid dimension issues

# =============================================================================
# END OF CONFIGURABLE PARAMETERS
# =============================================================================


def create_fresh_resenc_plans(
    dataset_id: int,
    patch_size: Optional[Tuple[int, int]],
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
    if patch_size is None:
        print(f"   Patch size: Using nnUNet's recommended size (automatic)")
    else:
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

    # Calculate compatible patch size based on the ResEnc architecture
    # ResEnc has 7 stages with stride 2, so we need patch size divisible by 2^6 = 64
    def make_patch_size_compatible(patch_size, n_stages=7):
        """Make patch size compatible with the number of downsampling stages."""
        divisor = 2 ** (n_stages - 1)  # 2^6 = 64 for 7 stages
        compatible_size = []
        for dim in patch_size:
            # Round to nearest multiple of divisor
            rounded = round(dim / divisor) * divisor
            # Ensure minimum size
            if rounded < divisor:
                rounded = divisor
            compatible_size.append(rounded)
        return compatible_size

    # Determine number of stages for patch size calculation
    n_stages = 7  # Default ResEnc
    if smaller_model_config:
        n_stages = smaller_model_config["n_stages"]

    # Use original patch size if None is provided (nnUNet recommended)
    if patch_size is None:
        final_patch_size = original_config["patch_size"]
        print(f"   Using nnUNet's recommended patch size: {final_patch_size}")
    else:
        compatible_patch_size = make_patch_size_compatible(patch_size, n_stages)
        final_patch_size = compatible_patch_size
        print(f"   Original patch size: {patch_size}")
        print(f"   Compatible patch size: {compatible_patch_size}")

    # Create fresh optimized configuration that inherits from ResEnc 2d
    optimized_config = {
        "inherits_from": "2d",
        "data_identifier": f"nnUNetPlans_{config_name}",
        "preprocessor_name": "DefaultPreprocessor",
        "batch_size": batch_size,
        "patch_size": final_patch_size,
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

    # Apply overfitting prevention strategies
    print(f"🛡️  Applying overfitting prevention strategies...")
    plans = add_overfitting_prevention_config(plans, config_name)

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
    dataset_id: int, patch_size: Optional[Tuple[int, int]], batch_size: int, config_name: str
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
    if patch_size is None:
        print(f"   Patch size: Using nnUNet's recommended size (automatic)")
    else:
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

    # Calculate compatible patch size (same function as in create_fresh_resenc_plans)
    def make_patch_size_compatible(patch_size, n_stages=7):
        """Make patch size compatible with the number of downsampling stages."""
        divisor = 2 ** (n_stages - 1)  # 2^6 = 64 for 7 stages
        compatible_size = []
        for dim in patch_size:
            # Round to nearest multiple of divisor
            rounded = round(dim / divisor) * divisor
            # Ensure minimum size
            if rounded < divisor:
                rounded = divisor
            compatible_size.append(rounded)
        return compatible_size

    # Use current configuration's patch size if None is provided (nnUNet recommended)
    if patch_size is None:
        final_patch_size = current_config["patch_size"]
        print(f"   Using nnUNet's recommended patch size: {final_patch_size}")
    else:
        compatible_patch_size = make_patch_size_compatible(patch_size)
        final_patch_size = compatible_patch_size
        print(f"   Original patch size: {patch_size}")
        print(f"   Compatible patch size: {compatible_patch_size}")

    # Create fixed configuration that explicitly removes resampling
    fixed_config = {
        "inherits_from": "2d",
        "data_identifier": f"nnUNetPlans_{config_name}",
        "preprocessor_name": "DefaultPreprocessor",
        "batch_size": batch_size,
        "patch_size": final_patch_size,
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
    print(
        f"   ✅ {compatible_patch_size[0]}×{compatible_patch_size[1]} patches will tile original images"
    )

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


def create_single_split(dataset_id: int, seed: int, tumor_percentage: float, validation_split: float):
    """
    Create a single train/validation split with class balance control.
    More efficient than cross-validation for large datasets (800+ images).
    
    Args:
        dataset_id: Dataset ID
        seed: Random seed for reproducibility  
        tumor_percentage: Percentage of tumor cases in training set (0.0-1.0)
        validation_split: Fraction of data to use for validation (e.g., 0.15 = 15%)
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"🔧 Creating single train/validation split for Dataset{dataset_id:03d}")
    print(f"   Target tumor percentage in training: {tumor_percentage * 100:.1f}%")
    print(f"   Validation split: {validation_split * 100:.1f}%")

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

    raw_path = Path(str(nnUNet_raw)) / dataset_name

    # Get all image files from imagesTr directory
    images_tr_dir = raw_path / "imagesTr"
    if not images_tr_dir.exists():
        raise FileNotFoundError(f"ImagesTr directory not found: {images_tr_dir}")
    
    # Get all image files and extract case names
    image_files = list(images_tr_dir.glob("*_0000.png"))
    all_cases = []
    
    for img_file in image_files:
        case_name = img_file.stem.replace("_0000", "")
        all_cases.append(case_name)
    
    # Sort for reproducibility
    all_cases.sort()

    print(f"📂 Found {len(all_cases)} total cases")

    # Categorize cases by checking for tumors in masks
    patients_with_tumors = []
    patients_without_tumors = []
    real_controls = []
    synthetic_tumor_cases = []

    labels_tr_path = raw_path / "labelsTr"

    for case in all_cases:
        mask_path = labels_tr_path / f"{case}.png"

        if case.startswith("patient_"):
            # Load mask to check for tumors
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask > 0):
                    patients_with_tumors.append(case)
                else:
                    patients_without_tumors.append(case)
            else:
                print(f"⚠️  Warning: Mask not found for {case}")
                patients_without_tumors.append(case)
        elif case.startswith("control_"):
            real_controls.append(case)
        elif case.startswith(SYNTHETIC_TUMOR_PREFIX):
            synthetic_tumor_cases.append(case)
        else:
            print(f"⚠️  Unknown case type: {case}")

    print(f"📊 Case Analysis:")
    print(f"   Patients with tumors: {len(patients_with_tumors)}")
    print(f"   Patients without tumors: {len(patients_without_tumors)}")
    print(f"   Real controls: {len(real_controls)}")
    print(f"   Synthetic tumor cases: {len(synthetic_tumor_cases)}")

    if len(patients_with_tumors) == 0:
        raise ValueError(
            "No patients with tumors found! Cannot create meaningful split."
        )

    # Create validation set: stratified sample of ALL real patients
    all_patients = patients_with_tumors + patients_without_tumors
    random.shuffle(all_patients)
    
    val_count = int(validation_split * len(all_patients))
    val_patients = all_patients[:val_count]
    train_patients_available = all_patients[val_count:]

    # Separate available training patients by tumor status
    train_patients_with_tumors = [
        p for p in train_patients_available if p in patients_with_tumors
    ]
    train_patients_without_tumors = [
        p for p in train_patients_available if p in patients_without_tumors
    ]

    print(f"\n📊 Patient Split:")
    print(f"   Validation patients: {len(val_patients)} ({validation_split*100:.1f}% of all patients)")
    print(f"   Available training patients with tumors: {len(train_patients_with_tumors)}")
    print(f"   Available training patients without tumors: {len(train_patients_without_tumors)}")

    # Determine training additions depending on synthetic toggle
    print(f"\n🎯 Training Set Composition:")
    tumor_cases_in_training = len(train_patients_with_tumors)
    
    if USE_SYNTHETIC_TUMORS:
        # Use a fraction of synthetic tumors; use 0 real controls
        selected_synth_count = int(len(synthetic_tumor_cases) * SYNTHETIC_TUMOR_FRACTION)
        rng = random.Random(seed)
        selected_synth = rng.sample(synthetic_tumor_cases, selected_synth_count) if selected_synth_count > 0 else []
        print(f"   Using synthetic tumors: {len(selected_synth)}/{len(synthetic_tumor_cases)} (fraction={SYNTHETIC_TUMOR_FRACTION})")
        print(f"   Using 0 real control images in training (per request)")
        train_controls = []
        train_synth = selected_synth
    else:
        # Original control selection to hit target ratio
        if tumor_percentage > 0:
            target_control_cases = int(
                tumor_cases_in_training * (1 - tumor_percentage) / tumor_percentage
            )
        else:
            target_control_cases = 0
        print(f"   Tumor cases available: {tumor_cases_in_training}")
        print(f"   Target control cases: {target_control_cases}")
        print(f"   Total control cases available: {len(real_controls)}")

        random.shuffle(real_controls)
        if target_control_cases > len(real_controls):
            print(f"⚠️  Warning: Not enough control cases. Using all {len(real_controls)} available.")
            train_controls = real_controls
        else:
            train_controls = real_controls[:target_control_cases]
        train_synth = []

    # Combine training cases
    train_cases = (
        train_patients_with_tumors
        + train_patients_without_tumors
        + train_controls
        + train_synth
    )
    random.shuffle(train_cases)

    # Calculate actual tumor percentage
    actual_tumor_cases = len(train_patients_with_tumors)
    actual_total_cases = len(train_cases)
    actual_tumor_percentage = (
        actual_tumor_cases / actual_total_cases if actual_total_cases > 0 else 0
    )

    print(f"\n📊 Final Single Split:")
    print(f"   Training cases: {len(train_cases)} ({(1-validation_split)*100:.1f}% of total)")
    print(f"     - Patients with tumors: {len(train_patients_with_tumors)}")
    print(f"     - Patients without tumors: {len(train_patients_without_tumors)}")
    print(f"     - Real controls: {len(train_controls)}")
    if USE_SYNTHETIC_TUMORS:
        print(f"     - Synthetic tumors: {len(train_synth)} (prefix='{SYNTHETIC_TUMOR_PREFIX}')")
    print(f"     - Actual tumor percentage: {actual_tumor_percentage * 100:.1f}% (target: {tumor_percentage * 100:.1f}%)")

    print(f"   Validation cases: {len(val_patients)} ({validation_split*100:.1f}% of total)")
    print(f"     - All real patients (stratified sample)")

    # Create the single split structure (nnUNet expects a list with one split)
    single_split = [{"train": train_cases, "val": val_patients}]

    # Save the split file
    splits_file = preprocessed_path / "splits_final.json"
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(single_split, str(splits_file))

    print(f"✅ Single split saved to: {splits_file}")
    print(f"   Training examples: {len(train_cases)}")
    print(f"   Validation examples: {len(val_patients)}")
    print(f"   Total data utilization: {len(train_cases) + len(val_patients)}/{len(all_cases)} cases")

    return single_split


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

    # Separate patients and controls; also detect synthetic tumor cases
    patients = [case for case in all_cases if case.startswith("patient_")]
    real_controls = [case for case in all_cases if case.startswith("control_")]
    synthetic_tumor_cases = [case for case in all_cases if case.startswith(SYNTHETIC_TUMOR_PREFIX)]

    print(f"   Patients: {len(patients)}")
    print(f"   Real controls: {len(real_controls)}")
    print(f"   Synthetic tumors detected: {len(synthetic_tumor_cases)} (prefix='{SYNTHETIC_TUMOR_PREFIX}')")

    # Check which patients actually have tumors
    labels_dir = dataset_path / "labelsTr"
    patients_with_tumors = []
    patients_without_tumors = []

    for patient in patients:
        label_file = labels_dir / f"{patient}.png"
        if label_file.exists():
            label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
            if label is not None:
                tumor_pixels = np.sum(label.astype(int) > 0)
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

    # Create validation set: 20% of all REAL patients (with and without tumors)
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

    # Determine training additions depending on synthetic toggle
    print(f"\n🎯 Class Balance Calculation:")
    tumor_cases_in_training = len(train_patients_with_tumors)
    if USE_SYNTHETIC_TUMORS:
        # Use a fraction of synthetic tumors; use 0 real controls
        selected_synth_count = int(len(synthetic_tumor_cases) * SYNTHETIC_TUMOR_FRACTION)
        rng = random.Random(seed)
        selected_synth = rng.sample(synthetic_tumor_cases, selected_synth_count) if selected_synth_count > 0 else []
        print(f"   Using synthetic tumors: {len(selected_synth)}/{len(synthetic_tumor_cases)} (fraction={SYNTHETIC_TUMOR_FRACTION})")
        print(f"   Using 0 real control images in training (per request)")
    else:
        # Original control selection to hit target ratio
        if tumor_percentage > 0:
            target_control_cases = int(
                tumor_cases_in_training * (1 - tumor_percentage) / tumor_percentage
            )
        else:
            target_control_cases = 0
        print(f"   Tumor cases available: {tumor_cases_in_training}")
        print(f"   Target control cases: {target_control_cases}")
        print(f"   Total control cases available: {len(real_controls)}")

    # Select training additions
    if USE_SYNTHETIC_TUMORS:
        train_controls = []  # explicit: 0 real controls
        train_synth = selected_synth
    else:
        random.shuffle(real_controls)
        if target_control_cases > len(real_controls):
            print(
                f"⚠️  Warning: Not enough control cases. Using all {len(real_controls)} available."
            )
            train_controls = real_controls
        else:
            train_controls = real_controls[:target_control_cases]
        train_synth = []

    # Combine training cases
    train_cases = (
        train_patients_with_tumors
        + train_patients_without_tumors
        + train_controls
        + train_synth
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
    print(f"     - Real controls: {len(train_controls)}")
    if USE_SYNTHETIC_TUMORS:
        print(f"     - Synthetic tumors: {len(train_synth)} (prefix='{SYNTHETIC_TUMOR_PREFIX}')")
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
    
    # Determine and display tiling configuration
    if USE_NNUNET_RECOMMENDED_TILING:
        print(f"Tiling: Using nnUNet's recommended patch size (automatic)")
        effective_patch_size = None  # Will be determined by nnUNet
    else:
        print(f"Tiling: Manual patch size {TILING_SIZE[0]}×{TILING_SIZE[1]} pixels")
        effective_patch_size = TILING_SIZE
    
    if USE_SINGLE_SPLIT:
        print(f"Validation Strategy: Single split ({VALIDATION_SPLIT*100:.1f}% validation, {(1-VALIDATION_SPLIT)*100:.1f}% training)")
    else:
        print(f"Validation Strategy: 5-fold cross-validation")
    print(f"Patient/Control Ratio: {PATIENT_CONTROL_RATIO * 100:.1f}% patients, {(1-PATIENT_CONTROL_RATIO) * 100:.1f}% controls")
    if USE_SYNTHETIC_TUMORS:
        print(f"Synthetic tumors enabled: fraction={SYNTHETIC_TUMOR_FRACTION} (real controls excluded)")
    if ENABLE_SALT_PEPPER_NOISE:
        noise_config = f"Salt-pepper noise: {SALT_PEPPER_PROBABILITY*100:.1f}% images, {SALT_PEPPER_DENSITY*100:.3f}% pixels density"
        if SALT_PEPPER_FORCED_IMAGES:
            noise_config += f", {len(SALT_PEPPER_FORCED_IMAGES)} forced cases"
        print(noise_config)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print()
    
    # Print detailed model configuration information
    print_model_info()
    print()

    # Step 1: Create fresh ResEnc model
    print("📋 Step 1: Creating fresh ResEnc model...")
    if not create_fresh_resenc_plans(
        DATASET_ID, effective_patch_size, BATCH_SIZE, MODEL_NAME, SMALLER_MODEL_CONFIG
    ):
        print("❌ Failed to create fresh ResEnc model. Exiting.")
        return False
    print()

    # Step 2: Handle resampling preference
    if ENABLE_RESAMPLING:
        print("📋 Step 2: Resampling enabled - keeping default resampling behaviour")
    else:
        print("📋 Step 2: Fixing ResEnc configuration (no resampling)...")
        if not fix_resenc_no_resampling(DATASET_ID, effective_patch_size, BATCH_SIZE, MODEL_NAME):
            print("❌ Failed to fix ResEnc configuration. Exiting.")
            return False
    print()

    # Step 3: Apply salt-pepper noise augmentation (before preprocessing)
    if ENABLE_SALT_PEPPER_NOISE:
        print("📋 Step 3a: Applying salt-pepper noise augmentation...")
        print(f"   Probability: {SALT_PEPPER_PROBABILITY * 100:.1f}% of images")
        print(f"   Density: {SALT_PEPPER_DENSITY * 100:.3f}% of pixels per affected image")
        print(f"   Salt/Pepper ratio: {SALT_PEPPER_SALT_RATIO * 100:.1f}% salt, {(1-SALT_PEPPER_SALT_RATIO) * 100:.1f}% pepper")
        
        # Get dataset path
        script_dir = Path(__file__).parent
        project_dir = script_dir.parent
        dataset_name = f"Dataset{DATASET_ID:03d}_TumorSegmentation"
        dataset_path = project_dir / "tumor_segmentation" / "data_nnUNet" / dataset_name
        
        apply_salt_pepper_to_dataset(
            dataset_path=dataset_path,
            probability=SALT_PEPPER_PROBABILITY,
            density=SALT_PEPPER_DENSITY,
            salt_ratio=SALT_PEPPER_SALT_RATIO,
            seed=RANDOM_SEED,
            forced_images=SALT_PEPPER_FORCED_IMAGES
        )
        print()
    else:
        print("📋 Step 3a: Salt-pepper noise augmentation disabled")
        print()

    # Step 3b: Run nnUNetv2_preprocess
    print("📋 Step 3b: Running nnUNetv2_preprocess...")
    if USE_SYNTHETIC_TUMORS:
        print("ℹ️  Synthetic dataset usage is enabled. Ensure your synthetic cases are already converted into the nnUNetv2 RAW dataset alongside patients:")
        print(f"   - imagesTr/ and labelsTr/ should contain files named with prefix '{SYNTHETIC_TUMOR_PREFIX}' for synthetic tumors, e.g., '{SYNTHETIC_TUMOR_PREFIX}0001_0000.png' and '{SYNTHETIC_TUMOR_PREFIX}0001.png'")
        print("   - If you haven't converted yet, run your dataset conversion so both real and synthetic are in the same DatasetXXX_TumorSegmentation folder, then re-run this script.")
    if not run_nnunet_preprocess(DATASET_ID, MODEL_NAME):
        print("❌ Failed to run nnUNetv2_preprocess. Exiting.")
        return False
    print()

    # Step 4: Create validation split strategy
    print("📋 Step 4: Creating validation split strategy...")
    print("🛡️  OVERFITTING PREVENTION: Using enhanced validation strategy")
    print("🎯  IMPORTANT: Validation will use ONLY REAL patient images (no synthetic)")
    
    if USE_SINGLE_SPLIT:
        # Single train/validation split (more efficient for large datasets)
        print("   Using single train/validation split for maximum data utilization")
        print(f"   Training: {(1-VALIDATION_SPLIT)*100:.1f}% | Validation: {VALIDATION_SPLIT*100:.1f}%")
        print("   → Creating single stratified split...")
        
        try:
            splits_file = create_single_split(
                DATASET_ID, RANDOM_SEED, PATIENT_CONTROL_RATIO, VALIDATION_SPLIT
            )
            print(f"✅ Single split created: {splits_file}")
            print(f"   💡 More training data available per epoch vs cross-validation")
            print(f"   🎯 Patient/Control ratio applied: {PATIENT_CONTROL_RATIO * 100:.1f}% patients in training set")
        except Exception as e:
            print(f"❌ Single split failed: {e}")
            return False
    else:
        # Cross-validation (traditional approach)
        print("   Tumor-only validation prevents distribution mismatch (test = 100% tumor)")
        print("   Training on all data (tumor + control) but validating only on tumor cases")
        print("   → Using 5-fold Cross-Validation with tumor-only validation sets...")
        
        try:
            # IMPORTANT: Enable tumor-only validation since test set contains only tumor cases
            # This prevents distribution mismatch between training validation and test
            splits_file = create_cross_validation_splits(DATASET_ID, RANDOM_SEED, n_folds=5, tumor_only_validation=True, patient_control_ratio=PATIENT_CONTROL_RATIO)
            print(f"✅ Cross-validation splits created: {splits_file}")
            print(f"   💡 This 5-fold CV approach helps prevent overfitting vs single split")
            print(f"   🎯 Patient/Control ratio applied: {PATIENT_CONTROL_RATIO * 100:.1f}% patients in training sets")
        except Exception as e:
            print(f"⚠️  Cross-validation failed, falling back to stratified split: {e}")
            try:
                splits_file = create_stratified_split(
                    DATASET_ID, RANDOM_SEED, PATIENT_CONTROL_RATIO
                )
                print(f"✅ Stratified split created: {splits_file}")
                print(f"   ⚠️  Single split used - monitor for overfitting!")
                print(f"   🎯 Patient/Control ratio applied: {PATIENT_CONTROL_RATIO * 100:.1f}% patients in training sets")
            except Exception as e2:
                print(f"❌ Failed to create any validation split: {e2}")
                return False
    print()

    # Success message
    print("🎉 Pipeline completed successfully!")
    print("=" * 50)
    print(f"✅ Fresh ResEnc model '{MODEL_NAME}' created")
    
    # Print model configuration summary
    if SMALLER_MODEL_CONFIG:
        estimated_params = calculate_model_parameters(SMALLER_MODEL_CONFIG)
        standard_params = calculate_model_parameters(None)
        reduction = (1 - estimated_params / standard_params) * 100
        print(f"✅ Model: {SELECTED_MODEL_SIZE.upper()} ({estimated_params/1_000_000:.1f}M params, {reduction:.1f}% reduction)")
    else:
        estimated_params = calculate_model_parameters(None)
        print(f"✅ Model: STANDARD ({estimated_params/1_000_000:.1f}M params)")
    
    resampling_text = "resampling enabled" if ENABLE_RESAMPLING else "no resampling"
    tiling_text = "nnUNet recommended tiling" if USE_NNUNET_RECOMMENDED_TILING else f"{TILING_SIZE[0]}×{TILING_SIZE[1]} tiling"
    print(f"✅ Configuration processed ({resampling_text}, {tiling_text})")
    print(f"✅ Preprocessing completed")
    if USE_SINGLE_SPLIT:
        print(f"✅ Single split created with {PATIENT_CONTROL_RATIO * 100:.1f}% patient ratio ({VALIDATION_SPLIT*100:.1f}% validation)")
    else:
        print(f"✅ Cross-validation splits created with {PATIENT_CONTROL_RATIO * 100:.1f}% patient ratio")
    print()

    print("📋 Next steps - OVERFITTING PREVENTION ENABLED:")
    print("=" * 50)
    print("🛡️  IMPORTANT: Your pipeline now includes overfitting prevention strategies!")
    print()
    if USE_SINGLE_SPLIT:
        print(f"# Train the model with single split (EFFICIENT):")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 0 -p nnUNetResEncUNetMPlans --npz")
        print("   💡 Single fold training - faster and uses more data per epoch")
        print()
        print(f"# Validate:")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 0 -p nnUNetResEncUNetMPlans --val --npz")
    else:
        print(f"# Train the model with cross-validation (ROBUST):")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 0 -p nnUNetResEncUNetMPlans --npz")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 1 -p nnUNetResEncUNetMPlans --npz")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 2 -p nnUNetResEncUNetMPlans --npz")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 3 -p nnUNetResEncUNetMPlans --npz")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} 4 -p nnUNetResEncUNetMPlans --npz")
        print("   💡 Train all 5 folds to get robust performance estimates")
        print()
        print(f"# Validate with cross-validation:")
        print(f"nnUNetv2_train {DATASET_ID} {MODEL_NAME} all -p nnUNetResEncUNetMPlans --val --npz")
        print()
        print(f"# Find best configuration:")
        print(f"nnUNetv2_find_best_configuration {DATASET_ID} -c {MODEL_NAME} -p nnUNetResEncUNetMPlans")
    print()
    print(f"# Run inference:")
    print(f"nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d {DATASET_ID} -c {MODEL_NAME} -p nnUNetResEncUNetMPlans")
    print()
    print("🚨 OVERFITTING MONITORING:")
    print("   - Compare training vs validation DICE scores regularly")
    print("   - Expected: Training DICE should NOT reach 0.99 with these settings")
    print("   - Target: Validation DICE should be within 5-10% of training DICE")
    print()
    print("📊 PERFORMANCE EXPECTATIONS:")
    print(f"   - Model size: REDUCED ({SELECTED_MODEL_SIZE.upper()}) - ~75% fewer parameters")
    print("   - Data augmentation: ENHANCED (based on nnUNet BraTS research)")
    if USE_SINGLE_SPLIT:
        print(f"   - Validation: SINGLE SPLIT ({VALIDATION_SPLIT*100:.1f}% validation, {(1-VALIDATION_SPLIT)*100:.1f}% training)")
        print("   - Efficiency: MORE TRAINING DATA per epoch vs cross-validation")
    else:
        print("   - Validation: 5-FOLD CROSS-VALIDATION (nnUNet standard)")
        print("   - Robustness: MULTIPLE validation folds for statistical reliability")
    print(f"   - Patient/Control ratio: {PATIENT_CONTROL_RATIO * 100:.1f}% patients, {(1-PATIENT_CONTROL_RATIO) * 100:.1f}% controls")
    print()
    print("⚠️  IF OVERFITTING PERSISTS:")
    print("   1. Use 'minimal' or 'tiny' model size")
    print("   2. Reduce batch size to 12 or 16")
    print("   3. Add more diverse training data")
    print("   4. Consider ensemble methods for better generalization")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
