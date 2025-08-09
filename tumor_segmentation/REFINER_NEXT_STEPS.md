# Refiner U-Net Next Steps

You now have the `postprocess_dataset` folder with **probability maps** from your updated `analyze_all_patients.py`. Here's exactly what to do next.

## Current Implementation Status

✅ **COMPLETED:**
- `postprocess_dataset/` with **probability maps** (.npy files) and splits
- `setup_refiner_dataset.py` script to convert probability maps to RefinerDataset format
- Custom batch sampler (5/8 hard patients, 1/8 control, 2/8 random)
- RefinerUNet model with SurfaceLoss, attention gates, cross-attention
- Loss function: `DiceLoss + 0.5*BCE + 0.1*SurfaceLoss`
- AdamW optimizer with CosineAnnealingLR and warmup
- Mixed precision and gradient clipping configuration
- Complete automated workflow script

✅ **CONFIGURATION LOCATIONS:**
- `BATCH_PATTERN` (line 24 in train_refiner.py)
- `HARD_FRACTION` (line 31 in train_refiner.py) 
- `LOSS_WEIGHTS` (line 34 in train_refiner.py)
- All hyperparameters at top of train_refiner.py

## Ready to Train! 🚀

Since you've updated `analyze_all_patients.py` to save probability maps, you can now proceed directly:

### Option 1: Complete Automated Workflow (Recommended)

```bash
python run_refiner_workflow.py \
    --postprocess_dir postprocess_dataset \
    --output_dir refiner_data \
    --batch_size 8 \
    --max_epochs 100 \
    --gpus 1
```

This single command will:
1. Convert your probability maps to RefinerDataset format
2. Compute auxiliary channels (entropy, distance, y-coord)
3. Copy original images and ground truth masks
4. Train the model with custom batch sampling

### Option 2: Step-by-Step

**Step 1: Convert to RefinerDataset format**
```bash
python setup_refiner_dataset.py \
    --postprocess_dir postprocess_dataset \
    --output_dir refiner_data
```

**Step 2: Train the model**
```bash
python train_refiner.py \
    --data_dir refiner_data \
    --batch_size 8 \
    --max_epochs 100 \
    --gpus 1
```

## Expected RefinerDataset Structure

After running `setup_refiner_dataset.py`, you'll have:
```
refiner_data/
├── patient_001/
│   ├── softmax.npz      # Converted from your probability .npy files (2,H,W)
│   ├── petmr.npy        # Original PET/MR image (H,W)
│   ├── entropy.npy      # Computed entropy map (H,W)
│   ├── distance.npy     # Signed distance transform (H,W)
│   ├── y_coord.npy      # Y-coordinate map (H,W)
│   └── mask.png         # Ground truth mask from data/patients/labels/
├── control_001/
│   └── ... (same structure, empty masks)
└── refiner_splits.json  # Training splits with hardness classification
```

## Key Configuration Points

### Batch Pattern (train_refiner.py line 24)
```python
BATCH_PATTERN = {
    "hard": 5,     # 5/8th of batch: hardest patient images  
    "control": 1,  # 1/8th of batch: control images
    "random": 2    # 2/8th of batch: random patient images
}
```

### Hard Patient Threshold (train_refiner.py line 31)
```python
HARD_FRACTION = 0.20  # Bottom 20% of patients by Dice score
```

### Loss Weights (train_refiner.py line 34)
```python
LOSS_WEIGHTS = {
    "dice": 1.0,      # DiceLoss
    "bce": 0.5,       # BCELoss  
    "surface": 0.1    # SurfaceLoss
}
```

## What the Training Will Do

The trainer automatically:
- Classifies patients as "hard" vs "random" based on Dice scores
- Creates batches with exactly 5 hard + 1 control + 2 random samples
- Uses the custom loss: `L = DiceLoss + 0.5*BCE + 0.1*SurfaceLoss`
- Applies AdamW optimizer with CosineAnnealingLR and 10-iteration warmup
- Uses mixed precision (`precision="16-mixed"`) and gradient clipping (1.0)
- Handles variable height images (300-1000px) with automatic padding
- Logs individual loss components and training metrics

## Quick Test

To verify the model architecture works with variable heights:
```bash
python test_refiner_padding.py
```

## Data Flow Summary

```
postprocess_dataset/
├── patient_001_probabilities.npy    # Your probability maps
├── comprehensive_splits.json        # Train/val splits
└── patient_dice_scores.json         # Hardness classification
                ↓
    setup_refiner_dataset.py converts to:
                ↓  
refiner_data/
├── patient_001/softmax.npz + aux channels
└── refiner_splits.json
                ↓
           train_refiner.py
                ↓
      Trained RefinerUNet model!
```

You're all set to train! 🎉