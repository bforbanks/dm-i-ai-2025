#!/usr/bin/env python3
"""
Verify Epoch Configuration Script

This script verifies that the automated nnUNet pipeline has properly configured
the epoch settings to stop after 100 epochs in QUICK_TEST_MODE.
"""

import json
import os
from pathlib import Path

def verify_epoch_config():
    """Verify that the epoch configuration is properly set to 100 epochs."""
    
    # Path to the plans file
    plans_path = Path("data_nnUNet/preprocessed/Dataset001_TumorSegmentation/nnUNetResEncUNetMPlans.json")
    
    if not plans_path.exists():
        print("❌ Plans file not found. Run the automated pipeline first.")
        return False
    
    # Load the plans file
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    # Check if test-CV2 configuration exists (this is the one currently being used)
    if "test-CV2" not in plans["configurations"]:
        print("❌ test-CV2 configuration not found in plans file.")
        print(f"   Available configurations: {list(plans['configurations'].keys())}")
        return False
    
    config = plans["configurations"]["test-CV2"]
    
    # Check for epoch settings
    print("🔍 Checking epoch configuration in plans file for test-CV2...")
    
    # Check num_epochs
    if "num_epochs" in config:
        num_epochs = config["num_epochs"]
        print(f"   ✅ num_epochs: {num_epochs}")
        if num_epochs == 100:
            print("   🎯 CORRECT: Set to 100 epochs for QUICK_TEST_MODE")
        else:
            print(f"   ⚠️  WARNING: Expected 100, got {num_epochs}")
    else:
        print("   ❌ num_epochs not found in test-CV2 configuration")
    
    # Check patience
    if "patience" in config:
        patience = config["patience"]
        print(f"   ✅ patience: {patience}")
    else:
        print("   ❌ patience not found in test-CV2 configuration")
    
    # Check lr_patience
    if "lr_patience" in config:
        lr_patience = config["lr_patience"]
        print(f"   ✅ lr_patience: {lr_patience}")
    else:
        print("   ❌ lr_patience not found in test-CV2 configuration")
    
    # Check validation_frequency
    if "validation_frequency" in config:
        validation_frequency = config["validation_frequency"]
        print(f"   ✅ validation_frequency: {validation_frequency}")
    else:
        print("   ❌ validation_frequency not found in test-CV2 configuration")
    
    # Check weight_decay
    if "weight_decay" in config:
        weight_decay = config["weight_decay"]
        print(f"   ✅ weight_decay: {weight_decay}")
    else:
        print("   ❌ weight_decay not found in test-CV2 configuration")
    
    # Check data_augmentation
    if "data_augmentation" in config:
        print("   ✅ data_augmentation section found")
    else:
        print("   ❌ data_augmentation section not found")
    
    # Check overfitting_prevention section
    if "overfitting_prevention" in config:
        print("   ✅ overfitting_prevention section found")
        ofp = config["overfitting_prevention"]
        if "num_epochs" in ofp:
            print(f"   ✅ overfitting_prevention.num_epochs: {ofp['num_epochs']}")
    else:
        print("   ❌ overfitting_prevention section not found")
    
    # Summary
    print("\n📋 Summary for test-CV2:")
    if "num_epochs" in config and config["num_epochs"] == 100:
        print("✅ Epoch configuration is CORRECT - test-CV2 will stop after 100 epochs")
        return True
    else:
        print("❌ Epoch configuration is INCORRECT - test-CV2 may run for 1000 epochs")
        print("   Run the fix_epoch_config.py script to fix this.")
        return False

if __name__ == "__main__":
    print("🔍 Verifying nnUNet Epoch Configuration for test-CV2")
    print("=" * 60)
    verify_epoch_config() 