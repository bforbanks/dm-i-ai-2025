#!/usr/bin/env python3
"""
Complete workflow script for training the Refiner U-Net.

This script automates the entire process from postprocess_dataset to trained model.

Usage:
    python run_refiner_workflow.py --postprocess_dir postprocess_dataset --output_dir refiner_data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Complete Refiner U-Net workflow")
    parser.add_argument("--postprocess_dir", required=True, type=Path, 
                       help="Directory with probability maps from analyze_all_patients.py")
    parser.add_argument("--output_dir", required=True, type=Path,
                       help="Output directory for RefinerDataset format")
    parser.add_argument("--nnunet_fold", type=int, default=2,
                       help="nnUNet fold number to align validation set with (default: 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max_epochs", type=int, default=15, help="Max training epochs")
    parser.add_argument("--gpus", type=int, default=0, help="GPU device number")
    parser.add_argument("--dry_run", action="store_true", help="Show commands without executing")
    
    args = parser.parse_args()
    
    print("🚀 Refiner U-Net Complete Workflow")
    print("=" * 60)
    print(f"Input: {args.postprocess_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print()
    
    # Step 1: Setup RefinerDataset
    setup_cmd = [
        "python", "setup_refiner_dataset.py",
        "--postprocess_dir", str(args.postprocess_dir),
        "--output_dir", str(args.output_dir),
        "--nnunet_fold", str(args.nnunet_fold)
    ]
    
    if args.dry_run:
        setup_cmd.append("--dry_run")
    
    print("📂 Step 1: Setup RefinerDataset format")
    if args.dry_run:
        print(f"   Would run: {' '.join(setup_cmd)}")
    else:
        success = run_command(setup_cmd, "Setup RefinerDataset")
        if not success:
            print("❌ Setup failed, stopping workflow")
            return
    
    print()
    
    # Step 2: Train the model
    if not args.dry_run:
        train_cmd = [
            "python", "train_refiner.py",
            "--data_dir", str(args.output_dir),
            "--batch_size", str(args.batch_size),
            "--max_epochs", str(args.max_epochs),
            "--gpus", str(args.gpus)
        ]
        
        print("🚀 Step 2: Train Refiner U-Net")
        print(f"   Training with custom batch sampling (5/8 hard, 1/8 control, 2/8 random)")
        print(f"   Loss: DiceLoss + 0.5*BCE + 0.1*SurfaceLoss")
        print(f"   Optimizer: AdamW with CosineAnnealingLR and warmup")
        print(f"   Mixed precision: enabled, gradient clipping: 1.0")
        print()
        
        success = run_command(train_cmd, "Train Refiner U-Net")
        if not success:
            print("❌ Training failed")
            return
    else:
        print("🚀 Step 2: Train Refiner U-Net (DRY RUN)")
        print(f"   Would run: python train_refiner.py --data_dir {args.output_dir} --batch_size {args.batch_size} --max_epochs {args.max_epochs} --gpus {args.gpus}")
    
    print()
    print("🎉 Refiner U-Net workflow completed!")
    
    if not args.dry_run:
        print(f"📁 RefinerDataset created in: {args.output_dir}")
        print(f"🤖 Model checkpoints saved in: lightning_logs/")
        print(f"📊 Training logs available in Lightning logs")
        print()
        print("📋 Configuration used:")
        print(f"   Batch pattern: 5 hard + 1 control + 2 random = 8 total")
        print(f"   Hard threshold: bottom 20% of patients by Dice score")
        print(f"   Loss weights: Dice=1.0, BCE=0.5, Surface=0.1")
        print(f"   Learning rate: 1e-3 with CosineAnnealingLR")
        print(f"   Weight decay: 1e-5")
        print(f"   Warmup: 10 iterations")
        print()
        print("🔧 To modify configuration, edit the top of train_refiner.py:")
        print("   - BATCH_PATTERN: change batch composition")
        print("   - HARD_FRACTION: change hardness threshold") 
        print("   - LOSS_WEIGHTS: change loss function weights")


if __name__ == "__main__":
    main()