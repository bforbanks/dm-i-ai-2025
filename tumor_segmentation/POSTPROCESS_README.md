# Postprocess Analysis Mode

This document explains how to use the new `--analyze-for-postprocess` option in `analyze_all_patients.py`.

## Overview

The postprocess mode analyzes all patient and control images, saves predicted masks, calculates metrics, and creates stratified splits for training and validation.

## Usage

```bash
python analyze_all_patients.py --analyze-for-postprocess [OPTIONS]
```

### Options

- `--analyze-for-postprocess`: Enable postprocess mode
- `--control-patient-split FLOAT`: Ratio of controls to patients in training folder (default: 0.3 for 70% patients, 30% controls)
- `--api-url URL`: API URL for predictions (default: http://localhost:9052)
- `--max-patients N`: Maximum number of patients to process (optional)

## What it does

1. **Processes Patients**:
   - Loads all patient images from `data/patients/imgs/`
   - Gets both binary predictions and probability maps using the API
   - Calculates Dice scores against ground truth
   - Saves predicted masks and probability maps to `postprocess_dataset/`

2. **Processes Controls**:
   - Loads all control images from `data/controls/imgs/`
   - Gets both binary predictions and probability maps using the API
   - Calculates false positive ratios (FP/ALL_PIXELS)
   - Saves predicted masks and probability maps to `postprocess_dataset/`

3. **Creates JSON Files**:
   - `patient_dice_scores.json`: Dice scores per patient
   - `control_fp_ratios.json`: False positive ratios per control
   - `comprehensive_splits.json`: All splits and subsets

4. **Creates Stratified Splits**:
   - **Patients**: 20% validation, 80% training (stratified by Dice score)
   - **Controls**: ALL controls with FP > 0 go to training, plus additional controls to achieve 70-30 patient:control ratio in training folder
   - **Folder Structure**: Creates parallel binary and probability map folders that maintain identical organization

5. **Creates Hard/Easy Subsets**:
   - **Patients**: Based on median Dice score of training set
   - **Controls**: Based on median FP ratio from ALL controls with FP > 0 (not just training controls)

## Output Structure

```
postprocess_dataset/
├── patient_001_prediction.png (binary mask)
├── patient_001_probabilities.png (probability map for visualization)
├── patient_001_probabilities.npy (raw probability data)
├── patient_002_prediction.png
├── patient_002_probabilities.png
├── patient_002_probabilities.npy
├── ...
├── control_001_prediction.png (binary mask)
├── control_001_probabilities.png (probability map for visualization)
├── control_001_probabilities.npy (raw probability data)
├── control_002_prediction.png
├── control_002_probabilities.png
├── control_002_probabilities.npy
├── ...
├── train_binary/
│   ├── patient_XXX_prediction.png (80% of patients - binary masks)
│   └── control_XXX_prediction.png (ALL with FP>0 + additional for 70:30 ratio - binary masks)
├── train_probabilities/
│   ├── patient_XXX_probabilities.png (80% of patients - probability maps)
│   └── control_XXX_probabilities.png (ALL with FP>0 + additional for 70:30 ratio - probability maps)
├── validation_binary/
│   └── patient_XXX_prediction.png (20% of patients - binary masks)
├── validation_probabilities/
│   └── patient_XXX_probabilities.png (20% of patients - probability maps)
├── patient_dice_scores.json
├── control_fp_ratios.json
└── comprehensive_splits.json
```

## JSON File Formats

### patient_dice_scores.json
```json
{
  "patient_001": {
    "dice_score": 0.8542,
    "mask_path": "postprocess_dataset/patient_001_prediction.png",
    "prob_map_path": "postprocess_dataset/patient_001_probabilities.png",
    "prob_raw_path": "postprocess_dataset/patient_001_probabilities.npy"
  },
  ...
}
```

### control_fp_ratios.json
```json
{
  "control_001": {
    "fp_ratio": 0.001234,
    "fp_pixels": 1234,
    "total_pixels": 1000000,
    "mask_path": "postprocess_dataset/control_001_prediction.png",
    "prob_map_path": "postprocess_dataset/control_001_probabilities.png",
    "prob_raw_path": "postprocess_dataset/control_001_probabilities.npy"
  },
  ...
}
```

### comprehensive_splits.json
```json
{
  "patient_splits": {
    "train": ["patient_001", "patient_002", ...],
    "validation": ["patient_005", "patient_010", ...],
    "hard_images_patients": ["patient_003", "patient_007", ...],
    "easy_images_patients": ["patient_001", "patient_002", ...]
  },
  "control_splits": {
    "train": ["control_002", "control_004", ...],
    "control_rest": ["control_001", "control_003", ...],
    "hard_images_control": ["control_005", "control_007", ...]
  },
  "statistics": {
    "total_patients": 100,
    "total_controls": 50,
    "train_patients": 80,
    "val_patients": 20,
    "train_controls": 20,
    "control_rest": 30,
    "controls_with_fp": 15,
    "median_dice_train": 0.8234,
    "median_fp_controls": 0.002345
  }
}
```

## Example Usage

```bash
# Basic usage
python analyze_all_patients.py --analyze-for-postprocess

# With custom control-patient split (e.g., 80% patients, 20% controls in training)
python analyze_all_patients.py --analyze-for-postprocess --control-patient-split 0.25

# Limit number of patients
python analyze_all_patients.py --analyze-for-postprocess --max-patients 50

# Use different API
python analyze_all_patients.py --analyze-for-postprocess --api-url http://localhost:8080
```

## API Endpoints

The postprocess analysis uses two API endpoints:

1. **`/predict`**: Returns binary segmentation masks (0/255 values)
2. **`/predict_with_probabilities`**: Returns probability maps (0-255 scaled from 0-1 probabilities)

Both endpoints use the same nnUNet model but the probability endpoint returns the raw softmax outputs before thresholding.

## File Types

- **Binary masks** (`*_prediction.png`): PNG files with 0/255 values for visualization and metrics
- **Probability maps** (`*_probabilities.png`): PNG files with 0-255 values (scaled probabilities) for visualization
- **Raw probabilities** (`*_probabilities.npy`): NumPy arrays with 0-1 values for further processing

## Folder Organization

The system creates **parallel folder structures** for binary masks and probability maps:

- **`train_binary/`** and **`train_probabilities/`**: Identical file organization, one with binary masks, one with probability maps
- **`validation_binary/`** and **`validation_probabilities/`**: Identical file organization for validation data
- All sorting/stratification decisions are based on binary mask performance (Dice scores, FP ratios)
- Probability maps follow the exact same organization as their corresponding binary masks

## Notes

- The script expects patient images in `data/patients/imgs/patient_*.png`
- The script expects patient labels in `data/patients/labels/segmentation_*.png`
- The script expects control images in `data/controls/imgs/control_*.png`
- All predicted masks are saved as PNG files (0-255 range)
- Probability maps are saved in both PNG (for visualization) and NPY (for processing) formats
- Stratification ensures balanced representation across difficulty levels
- Hard/easy subsets are created based on median scores of the training set only 