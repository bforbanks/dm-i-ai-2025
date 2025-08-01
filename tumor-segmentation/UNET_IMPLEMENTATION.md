# U-Net with Global Parameters and Attention Mechanisms

## Overview

This implementation extends the basic tumor segmentation approach with a sophisticated U-Net architecture that incorporates:

1. **Spatial and Channel Attention Mechanisms** in the encoder
2. **Global Parameter Extraction** for anatomical location and symmetry information
3. **Integration of Global Features** into the bottleneck layer
4. **Advanced Loss Functions** combining segmentation and global parameter supervision

## Architecture Components

### 1. Global Parameter Extractor
```python
class GlobalParameterExtractor(nn.Module):
```
- **Region Classifier**: Identifies anatomical regions (head/neck, chest, abdomen, pelvis, extremities, spine, other)
- **Symmetry Analyzer**: Extracts 32-dimensional symmetry features to detect bilateral patterns
- Specifically addresses your need for location-specific cancer detection and symmetry analysis

### 2. Attention Mechanisms
```python
class SpatialAttention(nn.Module):
class ChannelAttention(nn.Module):
```
- **Spatial Attention**: Focuses on tumor-relevant spatial regions
- **Channel Attention**: Weights feature importance across channels
- Applied in both encoder and decoder (skip connections)

### 3. U-Net with Global Integration
```python
class GlobalParameterUNet(nn.Module):
```
- Standard U-Net encoder-decoder with skip connections
- Global features are concatenated at the bottleneck level
- Attention mechanisms throughout the network

## Key Features Addressing Your Requirements

### Cancer Location Specificity
- **Anatomical Region Classification**: 7 different body regions
- **Location-aware Features**: Global features influence segmentation based on anatomical context
- **Region-specific Loss**: Supervises anatomical region prediction

### Symmetry Information
- **Bilateral Pattern Detection**: Analyzes left-right symmetry
- **Asymmetry Detection**: Identifies regions with significant asymmetric patterns
- **Symmetry Features**: 32-dimensional representation of bilateral information

### Attention in Encoder
- **Channel Attention**: SE-block style attention for feature recalibration
- **Spatial Attention**: Spatial focus on relevant regions
- **Skip Connection Attention**: Improves information flow from encoder to decoder

## Usage Guide

### 1. Training the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Start training
python train.py
```

**Training Configuration:**
- Batch size: 8 (adjust based on GPU memory)
- Learning rate: 1e-4 with ReduceLROnPlateau scheduler
- Combined loss: Dice + BCE + Region Classification
- Data augmentation: Rotation, brightness, contrast, noise

### 2. Using the Trained Model

```python
from model import TumorSegmentationModel

# Initialize model
model = TumorSegmentationModel(model_path="checkpoints/best_model.pth")

# Basic prediction
segmentation = model.predict(img)

# Detailed prediction with analysis
result = model.predict_with_analysis(img)
print(f"Detected regions: {result['global_params']['region_probs']}")
print(f"Symmetry score: {result['symmetry_analysis']['symmetry_score']}")
```

### 3. Integration with Existing API

The implementation is designed to be a drop-in replacement for the existing `example.py`:

```python
# In example.py
def predict(img: np.ndarray) -> np.ndarray:
    # Uses advanced U-Net if trained model available
    # Falls back to threshold method otherwise
    return model.predict(img)
```

## Model Architecture Details

### Input Processing
- **Multi-channel Support**: Handles grayscale and RGB inputs
- **Normalization**: ImageNet-style normalization
- **Size Handling**: Resizes to 512x384 for training, adapts to original size for inference

### Global Parameter Integration
1. **Feature Extraction**: Parallel branches for region and symmetry analysis
2. **Feature Expansion**: Global features broadcasted to spatial dimensions
3. **Concatenation**: Combined with bottleneck features
4. **Integration Layer**: 1x1 convolution to fuse global and local information

### Attention Flow
```
Input → Encoder (with attention) → Bottleneck + Global Features → Decoder (with attention) → Output
                ↓                                                        ↑
           Skip Connections ──────────────────────────── (with attention)
```

## Training Strategy

### Multi-task Learning
- **Primary Task**: Tumor segmentation (Dice + BCE loss)
- **Auxiliary Task**: Anatomical region classification
- **Regularization**: Global parameter supervision prevents overfitting

### Data Utilization
- **Patient Images**: 182 with tumor labels
- **Control Images**: 426 without tumors (negative examples)
- **Combined Training**: Learns to distinguish tumor vs. normal high-uptake regions

### Loss Function
```python
total_loss = 0.6 * dice_loss + 0.3 * bce_loss + 0.1 * region_loss
```

## Addressing Specific Challenges

### 1. Intestinal vs. Neck Cancer
- **Global region features** help the model understand anatomical context
- **Location-specific priors** learned from training data
- **Symmetry analysis** can distinguish bilateral neck patterns from intestinal tumors

### 2. False Positives from Normal Organs
- **Multi-task learning** with control images teaches normal patterns
- **Global context** helps distinguish expected high-uptake regions
- **Attention mechanisms** focus on truly abnormal patterns

### 3. Symmetry Patterns
- **Dedicated symmetry branch** analyzes bilateral patterns
- **Asymmetric detection** highlights regions breaking normal symmetry
- **Integration** with spatial features for context-aware decisions

## Performance Optimizations

### Inference Speed (10-second constraint)
- **Efficient Architecture**: Optimized U-Net design
- **GPU Acceleration**: CUDA support
- **Preprocessing Pipeline**: Minimal overhead
- **Model Size**: Balanced complexity vs. speed

### Memory Management
- **Gradient Checkpointing**: Available for training large models
- **Dynamic Resizing**: Handles variable input sizes
- **Batch Processing**: Optimized for inference

## Advanced Features

### 1. Symmetry Analysis
```python
def analyze_symmetry(self, img: np.ndarray) -> Dict[str, float]:
    # Compares left and right halves
    # Returns symmetry score and asymmetric regions
```

### 2. Confidence Estimation
- **Model uncertainty**: Based on prediction entropy
- **Global consistency**: Agreement between local and global features
- **Attention maps**: Visualization of model focus

### 3. Anatomical Region Detection
- **7-class classification**: Major body regions
- **Soft predictions**: Probability distribution over regions
- **Integration**: Influences segmentation decisions

## Future Enhancements

### 1. Advanced Symmetry
- **3D symmetry analysis** if volumetric data available
- **Organ-specific symmetry** patterns
- **Temporal symmetry** for longitudinal studies

### 2. Enhanced Global Features
- **Patient metadata** integration (age, gender, treatment history)
- **Multi-scale global** features
- **Hierarchical anatomical** understanding

### 3. Attention Improvements
- **Self-attention** mechanisms
- **Cross-attention** between encoder levels
- **Learnable attention** patterns

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**: Ensure your data follows the expected structure in `data/`

3. **Train Model**:
   ```bash
   python train.py
   ```

4. **Test Integration**:
   ```bash
   python example.py
   ```

5. **Deploy**:
   ```bash
   python api.py
   ```

## Troubleshooting

### Common Issues
- **GPU Memory**: Reduce batch size in `train.py`
- **Model Loading**: Check paths in `example.py`
- **Dependencies**: Ensure PyTorch version compatibility

### Performance Tips
- **Use GPU**: Significant speedup for training and inference
- **Data Preprocessing**: Consider caching preprocessed data
- **Model Pruning**: For deployment optimization

This implementation provides a solid foundation for incorporating global parameters and attention mechanisms into your tumor segmentation pipeline, specifically addressing the cancer location and symmetry requirements you mentioned. 