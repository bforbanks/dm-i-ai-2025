#!/usr/bin/env python3
"""
Create fresh ResEnc setup with optimized settings and new configuration name.

This script creates a completely new ResEnc configuration to avoid conflicts
with existing configurations.
"""

import json
import os
from pathlib import Path
from typing import Tuple


def create_fresh_resenc_plans(
    dataset_id: int = 1,
    patch_size: Tuple[int, int] = (320, 320),
    batch_size: int = 24,
    config_name: str = "2d_resenc_optimized"
):
    """
    Create fresh ResEnc plans with optimized configuration.
    
    Args:
        dataset_id: Dataset ID (e.g., 1 for Dataset001_TumorSegmentation)
        patch_size: New patch size (height, width)
        batch_size: New batch size
        config_name: Name for the new configuration
    """
    
    # Paths
    plans_dir = f"data_nnUNet/preprocessed/Dataset{dataset_id:03d}_TumorSegmentation"
    resenc_plans_path = f"{plans_dir}/nnUNetResEncUNetMPlans.json"
    
    # Check if ResEnc plans exist
    if not Path(resenc_plans_path).exists():
        print(f"❌ ResEnc plans not found: {resenc_plans_path}")
        print("   Please run: nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM")
        return False
    
    print(f"🔄 Creating fresh ResEnc configuration...")
    print(f"   ResEnc plans: {resenc_plans_path}")
    print(f"   New configuration: {config_name}")
    print(f"   Patch size: {patch_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Architecture: ResidualEncoderUNet (improved)")
    
    # Load ResEnc plans
    with open(resenc_plans_path, 'r') as f:
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
        # Keep the improved ResidualEncoderUNet architecture
        "architecture": original_config["architecture"],
        "batch_dice": original_config["batch_dice"]
    }
    
    # Add the new configuration to ResEnc plans
    plans["configurations"][config_name] = optimized_config
    
    # Create backup before modifying
    backup_path = f"{plans_dir}/nnUNetResEncUNetMPlans_fresh_backup.json"
    with open(backup_path, 'w') as f:
        json.dump(plans, f, indent=2)
    print(f"✅ Backup created: {backup_path}")
    
    # Save modified ResEnc plans
    with open(resenc_plans_path, 'w') as f:
        json.dump(plans, f, indent=2)
    
    print(f"✅ Fresh ResEnc configuration created successfully!")
    print(f"   New configuration '{config_name}' added to {resenc_plans_path}")
    print(f"   Using improved ResidualEncoderUNet architecture")
    
    return True


def provide_usage_instructions(dataset_id: int = 1, config_name: str = "2d_resenc_optimized"):
    """Provide instructions for using the fresh ResEnc configuration."""
    print("\n🎯 Usage Instructions:")
    print("=" * 50)
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {config_name}")
    print(f"Patch size: 320×320")
    print(f"Batch size: 24")
    print(f"Architecture: ResidualEncoderUNet (improved)")
    print(f"Plans: nnUNetResEncUNetMPlans")
    print()
    
    print("📋 Preprocessing Commands:")
    print("=" * 50)
    print(f"# Run preprocessing with fresh ResEnc configuration")
    print(f"nnUNetv2_preprocess -d {dataset_id} -c {config_name} -p nnUNetResEncUNetMPlans")
    print()
    
    print("📋 Training Commands:")
    print("=" * 50)
    print(f"# Train with fresh ResEnc configuration")
    print(f"nnUNetv2_train {dataset_id} {config_name} 0 -p nnUNetResEncUNetMPlans --npz")
    print()
    
    print("📋 Validation Commands:")
    print("=" * 50)
    print(f"# Validate with fresh ResEnc configuration")
    print(f"nnUNetv2_train {dataset_id} {config_name} 0 -p nnUNetResEncUNetMPlans --val --npz")
    print()
    
    print("📋 Find Best Configuration:")
    print("=" * 50)
    print(f"nnUNetv2_find_best_configuration {dataset_id} -c {config_name} -p nnUNetResEncUNetMPlans")
    print()
    
    print("📋 Inference Commands:")
    print("=" * 50)
    print(f"# Single fold inference")
    print(f"nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d {dataset_id} -c {config_name} -p nnUNetResEncUNetMPlans -f 0")
    print()
    print(f"# Ensemble inference (all folds)")
    print(f"nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d {dataset_id} -c {config_name} -p nnUNetResEncUNetMPlans")
    print()


def main():
    """Main function."""
    print("🚀 Fresh ResEnc Configuration Creator")
    print("=" * 40)
    print("This will create a fresh ResEnc configuration:")
    print("  ✅ 320×320 patch size")
    print("  ✅ 24 batch size")
    print("  ✅ New configuration: 2d_resenc_optimized")
    print("  ✅ ResidualEncoderUNet architecture (improved)")
    print("  ✅ No conflicts with existing configurations")
    print()
    
    dataset_id = 1
    patch_size = (320, 320)
    batch_size = 24
    config_name = "2d_resenc_optimized"
    
    # Create fresh ResEnc configuration
    if create_fresh_resenc_plans(dataset_id, patch_size, batch_size, config_name):
        print("\n✅ Fresh ResEnc configuration created successfully!")
        
        # Provide instructions
        provide_usage_instructions(dataset_id, config_name)
        
        print("\n🎉 Setup complete!")
        print("You can now train with the improved ResidualEncoderUNet architecture!")
        print("This is a fresh configuration with no conflicts.")
    else:
        print("\n❌ Failed to create fresh ResEnc configuration.")
        print("Please ensure ResEnc preprocessing has been run first.")


if __name__ == "__main__":
    main() 