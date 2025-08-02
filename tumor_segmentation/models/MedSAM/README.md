# MedSAM Model for Tumor Segmentation

This directory contains the MedSAM (Medical Segment Anything Model) implementation for tumor segmentation using the ViT-B/16 backbone.

## Model Architecture

The MedSAM model consists of three main components:

1. **Vision Transformer (ViT-B/16) Backbone**: Processes the input image and extracts features
2. **Prompt Encoder**: Handles user prompts (currently using dummy prompts for automatic segmentation)
3. **Mask Decoder**: Generates segmentation masks from image features and prompts

## Key Features

- **Pretrained Weights**: Uses MedSAM ViT-B/16 pretrained weights from Zenodo
- **Automatic Segmentation**: No user prompts required - automatically segments tumors
- **Binary Output**: Produces binary segmentation masks (tumor vs background)
- **Flexible Input Size**: Automatically resizes input to 256x256 for processing

## Usage

### 1. Download Pretrained Weights

First, download the pretrained MedSAM weights:

```bash
python download_medsam_weights.py
```

This will download the weights to `checkpoints/medsam_vit_b.pth`.

### 2. Train the Model

Use the same training command as other models:

```bash
python trainer.py fit --config models/config_base.yaml --config models/MedSAM/config.yaml --config models/MedSAM/wandb.yaml
```

### 3. Test the Model

Test the model with dummy data:

```bash
python test_medsam.py
```

## Configuration

The model configuration is in `config.yaml`:

- **Image Size**: 256x256 (matches data loader)
- **Batch Size**: 4 (reduced for larger model)
- **Learning Rate**: 1e-4 (lower than SimpleUNet for stable training)
- **Pretrained Weights**: Automatically loaded from `checkpoints/medsam_vit_b.pth`

## Model Parameters

- **Total Parameters**: ~86M (much larger than SimpleUNet)
- **Input Channels**: 3 (RGB)
- **Output Classes**: 1 (binary segmentation)
- **ViT Embedding Dimension**: 768
- **Transformer Dimension**: 256

## Performance Considerations

- **Memory Usage**: Higher than SimpleUNet due to larger model size
- **Training Time**: Longer training time expected
- **Batch Size**: Reduced to 4 to fit in GPU memory
- **Learning Rate**: Lower learning rate for stable training

## Expected Advantages

1. **Better Feature Extraction**: ViT backbone provides superior feature representation
2. **Medical Domain Knowledge**: Pretrained on medical data
3. **Attention Mechanism**: Better understanding of spatial relationships
4. **Transfer Learning**: Leverages pretrained weights for better performance

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `config.yaml`
- Use gradient accumulation if needed
- Consider using mixed precision training

### Weight Loading Issues
- Ensure weights are downloaded to `checkpoints/medsam_vit_b.pth`
- Check file permissions and disk space
- Verify the download completed successfully

### Training Instability
- Reduce learning rate further if needed
- Increase weight decay for regularization
- Monitor loss curves for signs of overfitting 