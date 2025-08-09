#!/usr/bin/env python3
"""
Script to duplicate checkpoint_best.pth to checkpoint_final.pth for all folds in a specific nnUNet run.
This helps resolve the "training is not finished yet" error when trying to run validation on early-stopped training.

Usage:
    python fix_checkpoints.py <run_name>
    
Example:
    python fix_checkpoints.py nnUNetTrainer__nnUNetResEncUNetMPlans__test-CV
"""

import os
import sys
import shutil
import glob
from pathlib import Path


def find_results_directory():
    """Find the nnUNet results directory."""
    possible_paths = [
        "data_nnUNet/results",
        "results",
        "../results",
        "./data_nnUNet/results"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, ask user to specify
    print("Could not automatically find results directory.")
    print("Please enter the path to your nnUNet results directory:")
    results_path = input().strip()
    
    if os.path.exists(results_path):
        return results_path
    else:
        raise FileNotFoundError(f"Results directory not found: {results_path}")


def fix_checkpoints_for_run(run_name, results_dir):
    """
    Fix checkpoints for a specific run by copying checkpoint_best.pth to checkpoint_final.pth
    
    Args:
        run_name (str): Name of the run (e.g., "nnUNetTrainer__nnUNetResEncUNetMPlans__test-CV")
        results_dir (str): Path to the results directory
    """
    print(f"Looking for run: {run_name}")
    print(f"In results directory: {os.path.abspath(results_dir)}")
    
    # Find all dataset directories
    dataset_pattern = os.path.join(results_dir, "Dataset*")
    dataset_dirs = glob.glob(dataset_pattern)
    
    if not dataset_dirs:
        print(f"No dataset directories found in {results_dir}")
        return False
    
    fixed_count = 0
    total_folds = 0
    
    for dataset_dir in dataset_dirs:
        run_path = os.path.join(dataset_dir, run_name)
        
        if not os.path.exists(run_path):
            continue
            
        print(f"\nFound run in: {run_path}")
        
        # Find all fold directories
        fold_pattern = os.path.join(run_path, "fold_*")
        fold_dirs = glob.glob(fold_pattern)
        
        if not fold_dirs:
            print(f"No fold directories found in {run_path}")
            continue
            
        fold_dirs.sort()  # Sort to process in order
        
        for fold_dir in fold_dirs:
            total_folds += 1
            fold_name = os.path.basename(fold_dir)
            
            checkpoint_best = os.path.join(fold_dir, "checkpoint_best.pth")
            checkpoint_final = os.path.join(fold_dir, "checkpoint_final.pth")
            
            if not os.path.exists(checkpoint_best):
                print(f"  {fold_name}: ❌ No checkpoint_best.pth found")
                continue
                
            if os.path.exists(checkpoint_final):
                print(f"  {fold_name}: ⚠️  checkpoint_final.pth already exists, skipping")
                continue
                
            try:
                # Copy checkpoint_best.pth to checkpoint_final.pth
                shutil.copy2(checkpoint_best, checkpoint_final)
                print(f"  {fold_name}: ✅ Created checkpoint_final.pth from checkpoint_best.pth")
                fixed_count += 1
                
            except Exception as e:
                print(f"  {fold_name}: ❌ Error copying checkpoint: {e}")
    
    print(f"\n" + "="*60)
    print(f"Summary:")
    print(f"  Total folds found: {total_folds}")
    print(f"  Checkpoints fixed: {fixed_count}")
    print(f"  Success rate: {fixed_count/total_folds*100:.1f}%" if total_folds > 0 else "  No folds found")
    
    return fixed_count > 0


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_checkpoints.py <run_name>")
        print("\nExample:")
        print("  python fix_checkpoints.py nnUNetTrainer__nnUNetResEncUNetMPlans__test-CV")
        print("\nAvailable runs:")
        
        try:
            results_dir = find_results_directory()
            # List available runs
            dataset_pattern = os.path.join(results_dir, "Dataset*")
            dataset_dirs = glob.glob(dataset_pattern)
            
            runs = set()
            for dataset_dir in dataset_dirs:
                if os.path.isdir(dataset_dir):
                    for item in os.listdir(dataset_dir):
                        item_path = os.path.join(dataset_dir, item)
                        if os.path.isdir(item_path) and "nnUNet" in item:
                            runs.add(item)
            
            for run in sorted(runs):
                print(f"  - {run}")
                
        except Exception as e:
            print(f"Could not list available runs: {e}")
        
        sys.exit(1)
    
    run_name = sys.argv[1]
    
    try:
        results_dir = find_results_directory()
        success = fix_checkpoints_for_run(run_name, results_dir)
        
        if success:
            print(f"\n🎉 Successfully fixed checkpoints for run: {run_name}")
            print("\nYou can now try running validation again:")
            print(f"nnUNetv2_train 1 test-CV all -p nnUNetResEncUNetMPlans --val --npz")
        else:
            print(f"\n❌ No checkpoints were fixed for run: {run_name}")
            print("Please check that the run name is correct and that checkpoint_best.pth files exist.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()