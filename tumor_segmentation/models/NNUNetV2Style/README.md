# NNUNetV2Style Model

A pure nnUNet v2 integration model for tumor segmentation that maintains compatibility with the existing codebase.

## Features

- **Pure nnUNet v2 Integration**: Uses nnUNet v2 architecture and preprocessing
- **BaseModel Compatibility**: Inherits from BaseModel for consistent training/validation logic
- **Inference Mode**: Designed for inference with trained nnUNet v2 models
- **No Fallback**: Requires nnUNet v2 to be installed and a valid model folder

## Architecture

The model uses nnUNet v2 exclusively:

- **nnUNet v2 Network**: Loads and uses trained nnUNet v2 network architecture
- **nnUNet v2 Preprocessing**: Uses nnUNet v2 preprocessing pipeline
- **nnUNet v2 Inference**: Leverages nnUNet v2 inference methods
- **No Custom Fallback**: If nnUNet v2 is not available, the model will fail to initialize

## Usage

### Inference Mode (nnUNet v2)

```python
from tumor_segmentation.models import NNUNetV2Style

# Initialize with trained nnUNet v2 model (REQUIRED)
model = NNUNetV2Style(
    model_folder="/path/to/nnunetv2/model",  # REQUIRED
    configuration_name="2d_fullres",
    use_folds=(0, 1, 2, 3, 4),  # or "all"
    checkpoint_name="checkpoint_final.pth"
)

# Predict using nnUNet v2 pipeline
segmentation = model.predict_with_nnunetv2(
    image_path="input.nii.gz",
    output_path="output.npy"  # optional
)
```

### Loading nnUNet v2 Model After Initialization

```python
# Initialize with model folder
model = NNUNetV2Style(model_folder="/path/to/nnunetv2/model")

# Or load later
model = NNUNetV2Style()
model.load_nnunetv2_model("/path/to/nnunetv2/model")
```

## Configuration

The model can be configured via the `config.yaml` file:

```yaml
model:
  class_path: models.NNUNetV2Style.model.NNUNetV2Style
  init_args:
    # Standard parameters
    bce_loss_weight: 0.4
    false_negative_penalty: 0.01
    patient_weight: 0.5
    control_weight: 0.5
    base_channels: 32
    depth: 4
    dropout_rate: 0.1
    
    # nnUNet v2 specific parameters
    model_folder: null  # Path to nnUNet v2 model folder
    configuration_name: "2d_fullres"
    use_folds: null  # Auto-detect
    checkpoint_name: "checkpoint_final.pth"
    use_mirroring: true
    tile_step_size: 0.5
```

## Dependencies

- **Required**: PyTorch, PyTorch Lightning, NumPy, OpenCV
- **Required**: nnUNet v2, batchgenerators (for nnUNet v2 functionality)

The model will fail to initialize if nnUNet v2 is not available.

## Integration with Existing Codebase

The model follows the same integration pattern as other models in the codebase:

1. **BaseModel Inheritance**: Inherits from BaseModel for consistent training/validation
2. **API Integration**: Can be used with the existing API endpoints
3. **Configuration System**: Uses the same configuration system as other models
4. **Checkpoint Loading**: Supports PyTorch Lightning checkpoint loading

## Differences from NNUNetStyle

- **Pure nnUNet v2**: Uses only nnUNet v2 components (no custom architecture)
- **No Fallback**: Requires nnUNet v2 to be installed and available
- **Inference Focused**: Designed primarily for inference with trained models
- **Advanced Preprocessing**: Leverages nnUNet v2 preprocessing pipeline

## Future Enhancements

- Support for nnUNet v2 training pipeline
- Integration with nnUNet v2 data format
- Support for 3D volumes
- Advanced post-processing methods 