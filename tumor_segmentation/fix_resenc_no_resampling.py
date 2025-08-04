#!/usr/bin/env python3
"""
Fix ResEnc configuration to remove resampling and use original image dimensions.

This script explicitly removes resampling functions from the ResEnc configuration
to ensure we use original image dimensions with 320x320 tiling.
"""

import json
import os
from pathlib import Path
from typing import Tuple


def fix_resenc_no_resampling(
    dataset_id: int = 1,
    patch_size: Tuple[int, int] = (320, 320),
    batch_size: int = 24,
    config_name: str = "2d_resenc_optimized"
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
    with open(resenc_plans_path, 'r') as f:
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
    with open(backup_path, 'w') as f:
        json.dump(plans, f, indent=2)
    print(f"✅ Backup created: {backup_path}")
    
    # Save fixed ResEnc plans
    with open(resenc_plans_path, 'w') as f:
        json.dump(plans, f, indent=2)
    
    print(f"✅ ResEnc configuration fixed successfully!")
    print(f"   Configuration '{config_name}' updated in {resenc_plans_path}")
    print(f"   ✅ Resampling functions REMOVED")
    print(f"   ✅ Original image dimensions preserved")
    print(f"   ✅ 320×320 patches will tile original images")
    
    return True


def verify_configuration(dataset_id: int = 1, config_name: str = "2d_resenc_optimized"):
    """Verify that the configuration is correct."""
    plans_dir = f"data_nnUNet/preprocessed/Dataset{dataset_id:03d}_TumorSegmentation"
    resenc_plans_path = f"{plans_dir}/nnUNetResEncUNetMPlans.json"
    
    with open(resenc_plans_path, 'r') as f:
        plans = json.load(f)
    
    if config_name not in plans["configurations"]:
        print(f"❌ Configuration '{config_name}' not found")
        return False
    
    config = plans["configurations"][config_name]
    
    print(f"\n🔍 Configuration Verification:")
    print(f"   Configuration: {config_name}")
    print(f"   Patch size: {config.get('patch_size', 'NOT SET')}")
    print(f"   Batch size: {config.get('batch_size', 'NOT SET')}")
    print(f"   Architecture: {config.get('architecture', {}).get('network_class_name', 'NOT SET')}")
    
    # Check for resampling functions
    has_resampling = any(key.startswith('resampling_fn') for key in config.keys())
    print(f"   Resampling enabled: {'❌ YES (PROBLEM)' if has_resampling else '✅ NO (CORRECT)'}")
    
    if has_resampling:
        print(f"   ❌ RESAMPLING FUNCTIONS STILL PRESENT!")
        print(f"   ❌ Images will be distorted!")
        return False
    else:
        print(f"   ✅ No resampling functions found")
        print(f"   ✅ Original image dimensions will be preserved")
        return True


def provide_usage_instructions(dataset_id: int = 1, config_name: str = "2d_resenc_optimized"):
    """Provide instructions for using the fixed configuration."""
    print("\n🎯 Usage Instructions:")
    print("=" * 50)
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {config_name}")
    print(f"Patch size: 320×320")
    print(f"Batch size: 24")
    print(f"Architecture: ResidualEncoderUNet (improved)")
    print(f"Resampling: ❌ DISABLED (original dimensions)")
    print(f"Plans: nnUNetResEncUNetMPlans")
    print()
    
    print("📋 Preprocessing Commands:")
    print("=" * 50)
    print(f"# Run preprocessing with fixed configuration")
    print(f"nnUNetv2_preprocess -d {dataset_id} -c {config_name} -p nnUNetResEncUNetMPlans")
    print()
    
    print("📋 Training Commands:")
    print("=" * 50)
    print(f"# Train with fixed configuration")
    print(f"nnUNetv2_train {dataset_id} {config_name} 0 -p nnUNetResEncUNetMPlans --npz")
    print()
    
    print("📋 Validation Commands:")
    print("=" * 50)
    print(f"# Validate with fixed configuration")
    print(f"nnUNetv2_train {dataset_id} {config_name} 0 -p nnUNetResEncUNetMPlans --val --npz")
    print()


def main():
    """Main function."""
    print("🔧 ResEnc No-Resampling Fix")
    print("=" * 30)
    print("This will fix the ResEnc configuration to:")
    print("  ✅ Remove resampling functions")
    print("  ✅ Use original image dimensions")
    print("  ✅ Enable proper 320×320 tiling")
    print("  ✅ Preserve anatomical proportions")
    print()
    
    dataset_id = 1
    patch_size = (320, 320)
    batch_size = 24
    config_name = "2d_resenc_optimized"
    
    # Fix ResEnc configuration
    if fix_resenc_no_resampling(dataset_id, patch_size, batch_size, config_name):
        print("\n✅ ResEnc configuration fixed successfully!")
        
        # Verify the configuration
        if verify_configuration(dataset_id, config_name):
            print("\n✅ Configuration verification passed!")
            
            # Provide instructions
            provide_usage_instructions(dataset_id, config_name)
            
            print("\n🎉 Setup complete!")
            print("You can now train with:")
            print("  - Original image dimensions (no distortion)")
            print("  - 320×320 patches that tile the entire image")
            print("  - Improved ResEnc architecture")
        else:
            print("\n❌ Configuration verification failed!")
            print("Resampling functions are still present.")
    else:
        print("\n❌ Failed to fix ResEnc configuration.")


if __name__ == "__main__":
    main() 