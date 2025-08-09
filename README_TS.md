## Tumor Segmentation (TS) — nnUNetv2 Pipeline

End-to-end instructions to prepare data, plan/preprocess, train, and run inference with nnUNetv2 for this project.

### 1) Install dependencies

requirements.txt in the tumor_segmentation folder

```bash
pip install -r requirements.txt
```

### 2) Set nnUNet environment variables (required by all nnUNetv2 CLIs)

Run these in every new shell (or add to your shell profile):

```bash
export nnUNet_raw=".../dm-i-ai-2025/tumor_segmentation/data_nnUNet"
export nnUNet_preprocessed=".../dm-i-ai-2025/tumor_segmentation/data_nnUNet/preprocessed"
export nnUNet_results=".../dm-i-ai-2025/tumor_segmentation/data_nnUNet/results"
```

### 3) Convert dataset to nnUNetv2 format

This prepares patients, controls, and (optionally) synthetic tumor data into a single nnUNetv2 dataset folder.

Expected input locations:
- `tumor_segmentation/data/patients/imgs`
- `tumor_segmentation/data/patients/labels`
- `tumor_segmentation/data/controls/imgs`
- `tumor_segmentation/data/controls_tumor/imgs` and `tumor_segmentation/data/controls_tumor/labels` (synthetic, optional)

Command:
```bash
python tumor_segmentation/convert_to_nnunetv2_format.py
```

This creates `Dataset001_TumorSegmentation/` under `tumor_segmentation/data_nnUNet/` with `imagesTr/`, `labelsTr/`, and `dataset.json`.

### 4) Plan and preprocess for nnUNetv2 (ResEnc planner)

```bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM
```

### 5) Automated pipeline configuration

The script `tumor_segmentation/automated_nnunet_pipeline.py` is pre-configured with the settings used in this project:
- Tiling: 384x384
- Synthetic fraction: 0.50
- Single validation split: 0.15
- Batch size: 8
- Seed: 42
- Model: conservative-custom

Run it (it will create a fresh ResEnc config, apply no-resampling, preprocess, and create the split):
```bash
python tumor_segmentation/automated_nnunet_pipeline.py
```

### 6) Train (250 epochs)

Train fold 0 with the configured setup:
```bash
nnUNetv2_train 1 conservative-custom-singleval-synth050 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
```

Notes:
- Training was run for 250 epochs; we used around epoch 220 - we can figure out the exact one if needed.
- Results folder (after/while training):
  `tumor_segmentation/data_nnUNet/results/Dataset001_TumorSegmentation/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__conservative-custom-singleval-synth050/fold_0`

### 7) Start the inference API (FastAPI)

Create a `.env` file at the project root with the following (adjust paths if different):

```bash
# Server
HOST=0.0.0.0
PORT=9052

# Model (point to the model directory, NOT a specific fold directory)
MODEL_FOLDER=.../tumor_segmentation/data_nnUNet/results/Dataset001_TumorSegmentation/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__conservative-custom-singleval-synth050
CONFIGURATION_NAME=conservative-custom-singleval-synth050
USE_FOLDS=0
CHECKPOINT_NAME=checkpoint_best.pth

# Predictor settings
TILE_STEP_SIZE=0.5
USE_MIRRORING=true
USE_GAUSSIAN=true
PERFORM_EVERYTHING_ON_DEVICE=true

# Post-processing and validation
SKIP_VALIDATION=true
```

Start the API server:
```bash
python tumor_segmentation/api_nnunetv2_pth.py
```

With the API running at `http://localhost:9052`, you can run the inference script in the next step.

### 8) Inference / Validation visualization

Use the provided script to run API-based predictions over the full validation set and save side-by-side visuals:
```bash
python tumor_segmentation/analyze_validation_set.py \
  --validation-dir tumor_segmentation/data/full_val_set \
  --output-dir validation_results
```

Outputs are saved under `validation_results/` (`comparison/` and `mask/`).

### Synthetic data used

We used synthetic tumor data included in:
- `tumor_segmentation/data/controls_tumor/imgs`
- `tumor_segmentation/data/controls_tumor/labels`

Generation process (already completed for the supplied data):
- Patch-based and heuristic masks: `tumor_segmentation/scripts/synthetic_tumor_patchbank_generator.py` and `tumor_segmentation/scripts/synthetic_tumor_look_generator_det.py` (each produced one mask per control image)
- SPADE-based inpainting model: trained with `tumor_segmentation/trainer/train_spade_gan_full_inpaint.py`, implemented in `tumor_segmentation/trainer/spade_gan_full_inpaint_module.py`
- Synthetic dataset creation via GAN on the generated masks, creating two synthetic tumor patients per control image: `tumor_segmentation/scripts/synthetic_tumor_gan_generator.py`

These synthetic cases are integrated by the conversion script so they sit alongside real patients/controls in `Dataset001_TumorSegmentation` and are used at 50% fraction during training.

