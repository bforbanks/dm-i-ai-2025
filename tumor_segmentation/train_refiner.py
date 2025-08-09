#!/usr/bin/env python3
"""Lightning training entry-point for the Refiner U-Net with custom batch sampling and SurfaceLoss"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import albumentations as A
import cv2
from PIL import Image

from data.refiner_dataset import RefinerDataset
from models.RefinerUNet.model import RefinerUNet


class TrainingMetricsLogger(Callback):
    """Custom callback to log training metrics to a file."""
    
    def __init__(self, dirpath: Path):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.log_file = self.dirpath / "training_metrics.log"
        self.metrics_history = []
        
        # Ensure directory exists
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write("Timestamp,Epoch,Step,Phase,Metric,Value\n")
        
        print(f"📊 TrainingMetricsLogger initialized, will log to: {self.log_file}")
    
    def _log_metric(self, trainer, pl_module, phase: str, metric_name: str, value: float):
        """Log a single metric to the file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch = trainer.current_epoch
        step = trainer.global_step
        
        log_line = f"{timestamp},{epoch},{step},{phase},{metric_name},{value:.6f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        # Also store in memory for potential analysis
        self.metrics_history.append({
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'phase': phase,
            'metric': metric_name,
            'value': value
        })
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training metrics after each batch."""
        if outputs is not None and isinstance(outputs, dict):
            for metric_name, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if isinstance(value, (int, float)):
                    self._log_metric(trainer, pl_module, "train", metric_name, value)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log validation metrics after each batch."""
        if outputs is not None and isinstance(outputs, dict):
            for metric_name, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if isinstance(value, (int, float)):
                    self._log_metric(trainer, pl_module, "val", metric_name, value)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch-level training metrics."""
        # Log any epoch-level metrics from trainer.callback_metrics
        for metric_name, value in trainer.callback_metrics.items():
            if metric_name.startswith('train_'):
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if isinstance(value, (int, float)):
                    self._log_metric(trainer, pl_module, "train_epoch", metric_name, value)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log epoch-level validation metrics."""
        # Log any epoch-level metrics from trainer.callback_metrics
        for metric_name, value in trainer.callback_metrics.items():
            if metric_name.startswith('val_'):
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if isinstance(value, (int, float)):
                    self._log_metric(trainer, pl_module, "val_epoch", metric_name, value)
    
    def on_fit_end(self, trainer, pl_module):
        """Log final training summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_file = self.dirpath / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary - {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total epochs: {trainer.current_epoch + 1}\n")
            f.write(f"Total steps: {trainer.global_step}\n")
            f.write(f"Best validation dice: {trainer.callback_metrics.get('val_dice', 'N/A')}\n")
            f.write(f"Final training loss: {trainer.callback_metrics.get('train_loss', 'N/A')}\n")
            f.write(f"Training completed successfully!\n")
        
        print(f"📋 Training summary saved to: {summary_file}")


class PthModelSaver(Callback):
    """Custom callback to save best model state_dict as refiner_best.pth file."""
    
    def __init__(self, dirpath: Path):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.best_val_dice = -1.0
        print(f"🔧 PthModelSaver initialized, will save to: {self.dirpath}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save best model if this epoch achieved the best validation dice."""
        # Debug: Print all available metrics
        print(f"🔍 Available metrics: {list(trainer.callback_metrics.keys())}")
        
        current_val_dice = trainer.callback_metrics.get("val_dice", -1.0)
        print(f"📊 Current val_dice: {current_val_dice}, Best so far: {self.best_val_dice}")
        
        # Save best model if this is the best validation dice so far
        if current_val_dice > self.best_val_dice:
            self.best_val_dice = current_val_dice
            best_path = self.dirpath / "refiner_best.pth"
            
            try:
                # Ensure directory exists
                self.dirpath.mkdir(parents=True, exist_ok=True)
                torch.save(pl_module.state_dict(), best_path)
                print(f"💾 New best refiner model saved to: {best_path}")
                print(f"✅ Validation dice improved to: {current_val_dice:.4f}")
                
                # Verify file was actually created
                if best_path.exists():
                    file_size = best_path.stat().st_size / (1024*1024)  # MB
                    print(f"📄 File saved successfully: {file_size:.1f} MB")
                else:
                    print("❌ ERROR: File was not created!")
                    
            except Exception as e:
                print(f"❌ ERROR saving refiner model: {e}")
        else:
            print(f"📈 No improvement (current: {current_val_dice:.4f} vs best: {self.best_val_dice:.4f})")

def variable_height_collate_fn(batch):
    """
    Custom collate function to handle variable height tensors in batches.
    
    Pads all tensors in the batch to the maximum height to enable batching.
    The RefinerUNet will handle individual tensor padding internally.
    """
    inputs, targets = zip(*batch)
    
    # Find maximum height in the batch
    max_height = max(inp.shape[1] for inp in inputs)  # Shape: [C, H, W]
    max_width = max(inp.shape[2] for inp in inputs)
    
    # Pad all inputs to max dimensions
    padded_inputs = []
    for inp in inputs:
        h, w = inp.shape[1], inp.shape[2]
        pad_h = max_height - h
        pad_w = max_width - w
        
        # Pad with 1.0 (white) - bottom and right padding
        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), mode='constant', value=1.0)
        else:
            padded = inp
        padded_inputs.append(padded)
    
    # Pad all targets to max dimensions
    padded_targets = []
    for tgt in targets:
        # Target shape is [C, H, W] where C=1 for binary segmentation
        h, w = tgt.shape[1], tgt.shape[2]
        pad_h = max_height - h
        pad_w = max_width - w
        
        # Pad with 0 (background class)
        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(tgt, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            padded = tgt
        padded_targets.append(padded)
    
    # Stack into batch tensors
    inputs_batch = torch.stack(padded_inputs, dim=0)
    targets_batch = torch.stack(padded_targets, dim=0)
    
    return inputs_batch, targets_batch

# ════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION SECTION - Modify these parameters as needed
# ════════════════════════════════════════════════════════════════════════════════════════

# Batch composition (must sum to 4 for reduced batch_size=4)
BATCH_PATTERN = {
    "hard": 3,     # 2/4th of batch: hardest patient images  
    "control": 0,  # 1/4th of batch: control images
    "random": 1    # 1/4th of batch: random patient images
}

# Loss function weights
LOSS_WEIGHTS = {
    "dice": 1.0,      # DiceLoss
    "bce": 0.5,       # BCELoss  
    "surface": 0.1    # SurfaceLoss
}

# Training hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_ITERS = 500
GRADIENT_CLIP_VAL = 1.0

# ════════════════════════════════════════════════════════════════════════════════════════


class RefinerBatchSampler(Sampler):
    """Custom batch sampler that ensures each batch contains the specified mix of hard/control/random samples."""
    
    def __init__(self, hard_indices: List[int], control_indices: List[int], random_indices: List[int], 
                 batch_size: int = 8, shuffle: bool = True):
        self.hard_indices = hard_indices
        self.control_indices = control_indices  
        self.random_indices = random_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Validate batch pattern
        pattern_sum = sum(BATCH_PATTERN.values())
        if pattern_sum != batch_size:
            raise ValueError(f"BATCH_PATTERN sum ({pattern_sum}) must equal batch_size ({batch_size})")
            
        # Ensure we have enough samples
        min_hard = BATCH_PATTERN["hard"]
        min_control = BATCH_PATTERN["control"] 
        min_random = BATCH_PATTERN["random"]
        
        if len(hard_indices) < min_hard:
            raise ValueError(f"Need at least {min_hard} hard samples, got {len(hard_indices)}")
        if len(control_indices) < min_control:
            raise ValueError(f"Need at least {min_control} control samples, got {len(control_indices)}")
        if len(random_indices) < min_random:
            raise ValueError(f"Need at least {min_random} random samples, got {len(random_indices)}")
    
    def __iter__(self):
        # Calculate number of batches based on smallest group to avoid repetition
        batch_calculations = []
        
        # Hard patients
        if BATCH_PATTERN["hard"] > 0:
            batch_calculations.append(len(self.hard_indices) // BATCH_PATTERN["hard"])
        else:
            batch_calculations.append(float('inf'))
        
        # Control patients
        if BATCH_PATTERN["control"] > 0:
            batch_calculations.append(len(self.control_indices) // BATCH_PATTERN["control"])
        else:
            batch_calculations.append(float('inf'))
        
        # Random patients
        if BATCH_PATTERN["random"] > 0:
            batch_calculations.append(len(self.random_indices) // BATCH_PATTERN["random"])
        else:
            batch_calculations.append(float('inf'))
        
        num_batches = min(batch_calculations)
        
        for batch_idx in range(num_batches):
            batch = []
            
            # Sample hard patients
            if BATCH_PATTERN["hard"] > 0:
                hard_start = batch_idx * BATCH_PATTERN["hard"]
                hard_end = hard_start + BATCH_PATTERN["hard"]
                batch.extend(self.hard_indices[hard_start:hard_end])
            
            # Sample controls  
            if BATCH_PATTERN["control"] > 0:
                control_start = batch_idx * BATCH_PATTERN["control"]
                control_end = control_start + BATCH_PATTERN["control"]
                batch.extend(self.control_indices[control_start:control_end])
            
            # Sample random patients
            if BATCH_PATTERN["random"] > 0:
                random_start = batch_idx * BATCH_PATTERN["random"] 
                random_end = random_start + BATCH_PATTERN["random"]
                batch.extend(self.random_indices[random_start:random_end])
            
            if self.shuffle:
                np.random.shuffle(batch)
                
            yield batch
    
    def __len__(self):
        batch_calculations = []
        
        # Hard patients
        if BATCH_PATTERN["hard"] > 0:
            batch_calculations.append(len(self.hard_indices) // BATCH_PATTERN["hard"])
        else:
            batch_calculations.append(float('inf'))
        
        # Control patients
        if BATCH_PATTERN["control"] > 0:
            batch_calculations.append(len(self.control_indices) // BATCH_PATTERN["control"])
        else:
            batch_calculations.append(float('inf'))
        
        # Random patients
        if BATCH_PATTERN["random"] > 0:
            batch_calculations.append(len(self.random_indices) // BATCH_PATTERN["random"])
        else:
            batch_calculations.append(float('inf'))
        
        return min(batch_calculations)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=Path, help="RefinerDataset directory (output from setup_refiner_dataset.py)")
    p.add_argument("--output_folder", required=True, type=str, help="Output folder path for saving models (e.g., 'nnUNetTrainer__nnUNetResEncUNetMPlans__CV-reduced/fold_2')")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    return p.parse_args()


def get_refiner_augmentations():
    """Augmentations that work with 6-channel RefinerUNet input (no color-space transforms)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),  # Reduced as medical images often have specific orientation
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.GaussNoise(noise_scale_factor=0.1, p=0.5),  # Reduced intensity for multi-channel
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        # Only use intensity transforms, skip HueSaturationValue for multi-channel
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ], additional_targets={'mask': 'mask'})


def generate_test_predictions(model: RefinerUNet, test_cases: List[str], data_dir: Path, output_dir: Path):
    """Generate predictions for test cases and save as images."""
    
    # Create test output directory
    test_dir = output_dir / "test_predictions"
    test_dir.mkdir(exist_ok=True)
    
    print(f"📸 Generating predictions for {len(test_cases)} validation cases...")
    print(f"💾 Saving to: {test_dir}")
    
    # Set model to evaluation mode
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    
    with torch.no_grad():
        for i, case_id in enumerate(test_cases):
            try:
                print(f"   Processing case {i+1}/{len(test_cases)}: {case_id}")
                
                # Load case data (same as RefinerDataset)
                case_dir = data_dir / case_id
                
                # Load input data
                softmax = np.load(case_dir / "softmax.npz")["softmax"]  # [C, H, W]
                petmr = np.load(case_dir / "petmr.npy")  # [H, W] or [C, H, W]
                
                # Load ground truth if available
                gt_path = case_dir / "mask.png"
                if gt_path.exists():
                    import cv2
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)  # [H, W]
                    gt = (gt > 0).astype(np.uint8)  # Convert to binary
                else:
                    print(f"   ⚠️  No ground truth found for {case_id}")
                    gt = None
                
                # Prepare input tensor (same logic as RefinerDataset)
                if petmr.ndim == 2:
                    petmr = petmr[np.newaxis, ...]  # Add channel dimension
                
                # Create entropy map
                entropy = -np.sum(softmax * np.log(softmax + 1e-8), axis=0, keepdims=True)
                
                # Create distance map (using argmax of softmax)
                softmax_pred = np.argmax(softmax, axis=0)
                distance = np.zeros_like(entropy)
                
                # Create y-coordinate map
                h, w = softmax.shape[1], softmax.shape[2]
                y_coords = np.linspace(0, 1, h).reshape(-1, 1)
                y_coord_map = np.tile(y_coords, (1, w))[np.newaxis, ...]
                
                # Concatenate all features: [softmax, petmr, entropy, distance, y_coord]
                input_tensor = np.concatenate([
                    softmax,           # [C, H, W]
                    petmr,             # [1 or C, H, W] 
                    entropy,           # [1, H, W]
                    distance,          # [1, H, W]
                    y_coord_map        # [1, H, W]
                ], axis=0)
                
                # Convert to tensor and add batch dimension
                input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0)  # [1, C, H, W]
                
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                # Generate prediction
                logits, probs = model(input_tensor)
                
                # Convert to numpy
                probs_np = probs.squeeze().cpu().numpy()  # [H, W] or [C, H, W]
                
                # If multi-class, take the tumor class (assuming last channel)
                if probs_np.ndim == 3:
                    probs_np = probs_np[-1]  # Take last channel (tumor)
                
                # Create binary prediction
                pred_binary = (probs_np > 0.5).astype(np.uint8) * 255
                
                # Save prediction as image
                pred_path = test_dir / f"{case_id}_prediction.png"
                Image.fromarray(pred_binary).save(pred_path)
                
                # Save probability map
                prob_map = (probs_np * 255).astype(np.uint8)
                prob_path = test_dir / f"{case_id}_probability.png"
                Image.fromarray(prob_map).save(prob_path)
                
                # Save ground truth if available
                if gt is not None:
                    gt_binary = (gt > 0).astype(np.uint8) * 255
                    gt_path = test_dir / f"{case_id}_ground_truth.png"
                    Image.fromarray(gt_binary).save(gt_path)
                
                # Create comparison image if ground truth exists
                if gt is not None:
                    # Create RGB visualization: R=GT, G=Pred, B=Background
                    h, w = pred_binary.shape
                    comparison = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Red channel: Ground truth
                    comparison[:, :, 0] = gt_binary
                    # Green channel: Prediction
                    comparison[:, :, 1] = pred_binary
                    # Blue channel: Background (where both are 0)
                    comparison[:, :, 2] = ((gt_binary == 0) & (pred_binary == 0)).astype(np.uint8) * 128
                    
                    comp_path = test_dir / f"{case_id}_comparison.png"
                    Image.fromarray(comparison).save(comp_path)
                
                print(f"   ✅ Saved: {pred_path.name}, {prob_path.name}" + 
                      (f", {gt_path.name}, {comp_path.name}" if gt is not None else ""))
                
            except Exception as e:
                print(f"   ❌ Error processing {case_id}: {e}")
                continue
    
    print(f"✅ Test predictions completed! Check: {test_dir}")


def main():
    args = parse_args()
    
    print("🚀 Refiner U-Net Training with Custom Batch Sampling")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output folder: {args.output_folder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batch pattern: {BATCH_PATTERN}")
    print(f"Loss weights: {LOSS_WEIGHTS}")
    print()

    # Load refiner splits
    splits_file = args.data_dir / "refiner_splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"Refiner splits file not found: {splits_file}")
        
    with open(splits_file, 'r') as f:
        refiner_splits = json.load(f)
    
    train_hard = refiner_splits["train_hard"]
    train_random = refiner_splits["train_random"] 
    train_controls = refiner_splits["train_control"]
    val_patients = refiner_splits["validation"]
    
    print(f"📊 Training set composition:")
    print(f"   Hard patients: {len(train_hard)}")
    print(f"   Random patients: {len(train_random)}")
    print(f"   Controls: {len(train_controls)}")
    print(f"   Validation patients: {len(val_patients)}")
    print()
    
    # Check if we have enough samples for the batch pattern
    batch_calculations = []
    
    # Hard patients
    if BATCH_PATTERN["hard"] > 0:
        batch_calculations.append(len(train_hard) // BATCH_PATTERN["hard"])
    else:
        batch_calculations.append(float('inf'))  # No hard patients required
    
    # Control patients
    if BATCH_PATTERN["control"] > 0:
        batch_calculations.append(len(train_controls) // BATCH_PATTERN["control"])
    else:
        batch_calculations.append(float('inf'))  # No control patients required
    
    # Random patients
    if BATCH_PATTERN["random"] > 0:
        batch_calculations.append(len(train_random) // BATCH_PATTERN["random"])
    else:
        batch_calculations.append(float('inf'))  # No random patients required
    
    min_batches = min(batch_calculations)
    print(f"📊 Maximum batches possible: {min_batches}")
    
    if min_batches == 0:
        raise ValueError("Not enough samples to create even one batch with the current BATCH_PATTERN")
    elif min_batches == float('inf'):
        raise ValueError("BATCH_PATTERN requires at least one non-zero category")
    
    # Create datasets
    print("📂 Creating datasets...")
    
    # All training cases (for mapping indices)
    all_train_cases = train_hard + train_random + train_controls
    train_dataset = RefinerDataset(args.data_dir, cases=all_train_cases, transform=get_refiner_augmentations())
    val_dataset = RefinerDataset(args.data_dir, cases=val_patients, transform=None)
    
    # Create index mappings for batch sampler
    case_to_idx = {case: idx for idx, case in enumerate(all_train_cases)}
    
    hard_indices = [case_to_idx[case] for case in train_hard]
    control_indices = [case_to_idx[case] for case in train_controls]
    random_indices = [case_to_idx[case] for case in train_random]
    
    # Create custom batch sampler
    print("🔄 Creating custom batch sampler...")
    batch_sampler = RefinerBatchSampler(
        hard_indices=hard_indices,
        control_indices=control_indices,
        random_indices=random_indices,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create data loaders with custom collate function for variable heights
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=variable_height_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=variable_height_collate_fn
    )
    
    print(f"📊 Data loader info:")
    print(f"   Training batches per epoch: {len(batch_sampler)}")
    print(f"   Validation batches: {len(val_loader)}")
    print()
    
    # Determine input channels from a sample
    sample_case_dir = args.data_dir / all_train_cases[0]
    softmax = np.load(sample_case_dir / "softmax.npz")["softmax"]
    petmr = np.load(sample_case_dir / "petmr.npy")
    
    if petmr.ndim == 2:
        pet_channels = 1
    else:
        pet_channels = petmr.shape[0]
    
    in_channels = softmax.shape[0] + pet_channels + 3  # softmax + petmr + entropy + distance + y_coord
    num_classes = softmax.shape[0]
    
    print(f"📊 Model configuration:")
    print(f"   Input channels: {in_channels}")
    print(f"   Number of classes: {num_classes}")
    print(f"   PET/MR channels: {pet_channels}")
    print()
    
    # Create model with loss configuration
    model = RefinerUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_iters=WARMUP_ITERS,
        max_epochs=args.max_epochs,
        dice_weight=LOSS_WEIGHTS["dice"],
        bce_weight=LOSS_WEIGHTS["bce"],
        surface_weight=LOSS_WEIGHTS["surface"]
    )
    
    # Create output directory structure
    output_dir = Path("data_nnUNet/results/Dataset001_TumorSegmentation") / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Models will be saved to: {output_dir}")
    print()
    
    # Create callback for best model saving
    best_ckpt_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="refiner_best",
        monitor="val_dice", 
        mode="max", 
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False
    )
    
    lr_cb = LearningRateMonitor(logging_interval="step")
    pth_saver_cb = PthModelSaver(dirpath=output_dir)
    metrics_logger_cb = TrainingMetricsLogger(dirpath=output_dir)
    
    # Create trainer with mixed precision and gradient clipping
    if args.gpus > 0 and torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=args.gpus,
            callbacks=[best_ckpt_cb, lr_cb, pth_saver_cb, metrics_logger_cb],
            precision="16-mixed",
            gradient_clip_val=GRADIENT_CLIP_VAL,
            log_every_n_steps=5,  # Lower for small batch count
            check_val_every_n_epoch=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="cpu",
            callbacks=[best_ckpt_cb, lr_cb, pth_saver_cb, metrics_logger_cb],
            precision="bf16-mixed",  # Use bfloat16 for CPU
            gradient_clip_val=GRADIENT_CLIP_VAL,
            log_every_n_steps=5,  # Lower for small batch count
            check_val_every_n_epoch=1
        )
    
    print("🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"📁 Models should be saved to: {output_dir}")
    
    # Check what files actually exist
    print("\n🔍 Checking saved files:")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            for file in sorted(files):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024*1024)
                    print(f"   ✅ {file.name} ({size_mb:.1f} MB)")
                else:
                    print(f"   📁 {file.name}/ (directory)")
        else:
            print("   ❌ No files found in output directory!")
    else:
        print(f"   ❌ Output directory does not exist: {output_dir}")
    
    # Specifically check for refiner files
    refiner_pth = output_dir / "refiner_best.pth"
    refiner_ckpt = output_dir / "refiner_best.ckpt"
    
    print(f"\n🎯 Refiner-specific files:")
    if refiner_pth.exists():
        size_mb = refiner_pth.stat().st_size / (1024*1024)
        print(f"   ✅ refiner_best.pth ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ refiner_best.pth NOT FOUND")
        
    if refiner_ckpt.exists():
        size_mb = refiner_ckpt.stat().st_size / (1024*1024)
        print(f"   ✅ refiner_best.ckpt ({size_mb:.1f} MB)")
    else:
        print(f"   ❌ refiner_best.ckpt NOT FOUND")
    
    # Check for training log files
    training_log = output_dir / "training_metrics.log"
    training_summary = output_dir / "training_summary.txt"
    
    print(f"\n📊 Training log files:")
    if training_log.exists():
        size_kb = training_log.stat().st_size / 1024
        print(f"   ✅ training_metrics.log ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ training_metrics.log NOT FOUND")
        
    if training_summary.exists():
        size_kb = training_summary.stat().st_size / 1024
        print(f"   ✅ training_summary.txt ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ training_summary.txt NOT FOUND")
    
    # Generate test predictions on first 3 validation patients
    print(f"\n🔮 Generating test predictions...")
    generate_test_predictions(model, val_patients[:3], args.data_dir, output_dir)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
