# nnU-Net v2 with TotalSegmentator v2.4 Pre-trained Weights

This directory contains an implementation of nnU-Net v2 (2-D) for tumor segmentation, designed to work with TotalSegmentator v2.4 pre-trained weights.

## Features

- **nnU-Net v2 Architecture**: Follows the nnU-Net v2 design principles
- **Instance Normalization**: Uses instance normalization instead of batch normalization for better generalization
- **Leaky ReLU**: Uses Leaky ReLU activation (negative_slope=0.01) as per nnU-Net v2
- **Pre-trained Weights**: Supports loading TotalSegmentator v2.4 pre-trained weights
- **Adaptive Depth**: Configurable network depth based on input size
- **Skip Connections**: U-Net style skip connections for better feature preservation

## Architecture

The model follows the nnU-Net v2 architecture with the following key components:

- **Encoder Path**: Progressive downsampling with feature extraction
- **Bottleneck**: Deepest layer for global feature representation
- **Decoder Path**: Progressive upsampling with skip connections
- **Instance Normalization**: Normalizes each sample independently
- **Leaky ReLU**: Activation function with small negative slope

## Usage

### Basic Training

```bash
python trainer.py fit --config models/config_base.yaml --config models/nnUNetv2/config.yaml --config models/nnUNetv2/wandb.yaml
```

### Configuration

The model can be configured through the `config.yaml` file:

```yaml
model:
  class_path: models.nnUNetv2.model.nnUNetv2
  init_args:
    in_channels: 3
    num_classes: 1
    base_channels: 32
    depth: 5
    lr: 1e-3
    weight_decay: 1e-5
    use_pretrained: true
    pretrained_path: null  # Set path to pre-trained weights if available
```

### Parameters

- `in_channels`: Number of input channels (default: 3)
- `num_classes`: Number of output classes (default: 1 for binary segmentation)
- `base_channels`: Number of base channels (default: 32)
- `depth`: Network depth (default: 5)
- `lr`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay for regularization (default: 1e-5)
- `use_pretrained`: Whether to load pre-trained weights (default: true)
- `pretrained_path`: Path to pre-trained weights file (optional)

## Pre-trained Weights

The model supports loading TotalSegmentator v2.4 pre-trained weights. To use pre-trained weights:

1. **Download TotalSegmentator v2.4 weights** from the official repository
2. **Place the weights file** in one of these locations:
   - `checkpoints/totalsegmentator_v2.4.pth`
   - `models/nnUNetv2/totalsegmentator_v2.4.pth`
   - `pretrained/totalsegmentator_v2.4.pth`
3. **Set the path** in the config file if using a different location

### Weight Loading

The model automatically attempts to load compatible weights and will:
- Match layer names and shapes
- Handle different checkpoint formats
- Continue with random initialization if no compatible weights are found
- Provide detailed logging about the loading process

## Dependencies

Install the required dependencies:

```bash
pip install -r models/nnUNetv2/requirements.txt
```

Key dependencies include:
- `torch>=2.0.0`
- `lightning>=2.0.0`
- `monai>=1.3.0` (for medical imaging)
- `nibabel>=5.0.0` (for medical image formats)
- `SimpleITK>=2.2.0` (for medical image processing)

## Performance

The nnU-Net v2 model is designed to achieve better performance than SimpleUNet through:

- **Better Normalization**: Instance normalization provides better generalization
- **Improved Activations**: Leaky ReLU prevents dying ReLU problem
- **Pre-trained Weights**: TotalSegmentator v2.4 weights provide strong initialization
- **Proper Initialization**: He initialization optimized for Leaky ReLU

## Monitoring

The model integrates with Weights & Biases for experiment tracking:

- **Project**: `tumor-segmentation`
- **Run Name**: `nnunet-v2-totalsegmentator`
- **Metrics**: Dice score, loss, learning rate
- **Model Checkpoints**: Automatically saved based on validation Dice score

## Troubleshooting

### Common Issues

1. **Pre-trained weights not found**: The model will continue with random initialization
2. **Shape mismatches**: The model will only load compatible layers
3. **Memory issues**: Reduce `base_channels` or `depth` for smaller models

### Performance Tips

1. **Use pre-trained weights** when available for better initialization
2. **Adjust learning rate** based on your dataset size
3. **Monitor validation metrics** to prevent overfitting
4. **Use appropriate batch size** for your GPU memory

## Comparison with SimpleUNet

| Feature | SimpleUNet | nnU-Net v2 |
|---------|------------|------------|
| Normalization | BatchNorm | InstanceNorm |
| Activation | ReLU | Leaky ReLU |
| Pre-trained | No | Yes (TotalSegmentator) |
| Initialization | Random | He + Pre-trained |
| Architecture | Basic U-Net | nnU-Net v2 principles |

The nnU-Net v2 model should provide better performance due to these architectural improvements and pre-trained weights. 