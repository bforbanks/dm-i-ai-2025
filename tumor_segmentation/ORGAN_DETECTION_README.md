# Organ Detection with Tiled Inference

This directory contains the implementation of tiled organ detection using the OrganDetector model. The approach focuses on identifying non-tumor dark pixels (organs) rather than tumor pixels.

## 🎯 **Key Concept**

Instead of detecting tumors directly, this approach:
- **Targets non-tumor pixels** in dark regions (intensity < 85)
- **Uses controls as positive examples** of normal organ tissue
- **Processes images as horizontal tiles** (128px height) for better resolution
- **Inverts the segmentation problem** from "find tumors" to "find normal organs"

## 📁 **Files Overview**

### Core Implementation
- `data/tiled_data.py` - Tiled dataset that creates 128px horizontal strips
- `models/OrganDetector/model.py` - U-Net optimized for tile dimensions
- `models/OrganDetector/config.yaml` - Configuration for training

### Inference Scripts
- `tiled_inference.py` - **Main tiled inference script**
- `plot_baseline_predictions.py` - **Enhanced visualization with organ mode**
- `run_organ_detection_demo.py` - **Complete demo suite**

### Configuration
- `models/OrganDetector/wandb.yaml` - Weights & Biases logging
- `trainer.py` - Updated to support tiled data module

## 🚀 **Quick Start**

### 1. Train the OrganDetector
```bash
python trainer.py fit \
  --config models/config_base.yaml \
  --config models/OrganDetector/config.yaml \
  --config models/OrganDetector/wandb.yaml
```

### 2. Run Tiled Inference on Full Images
```bash
# Basic organ detection
python tiled_inference.py --checkpoint checkpoints/organ-detector-latest.ckpt --see-organs

# On specific image
python tiled_inference.py \
  --checkpoint checkpoints/organ-detector-latest.ckpt \
  --image data/patients/imgs/patient_042.png \
  --see-organs
```

### 3. Visualize Tile-by-Tile Predictions
```bash
# Show organ predictions (green areas)
python plot_baseline_predictions.py --see-organs --model-checkpoint checkpoints/organ-detector-latest.ckpt

# Compare with tumor predictions (red areas)
python plot_baseline_predictions.py
```

### 4. Run Complete Demo Suite
```bash
python run_organ_detection_demo.py
```

### 5. Test Threshold Masking
```bash
# Verify threshold masking is working correctly
python test_threshold_masking.py
```

## 🧠 **How Tiled Inference Works**

### Image Processing Pipeline:
1. **Split**: Divide full image into 128px height horizontal tiles
2. **Pad**: Add white padding to last tile if needed
3. **Inference**: Run OrganDetector on each tile independently  
4. **Reassemble**: Stitch predictions back into full image
5. **Visualize**: Show organ predictions with context

### Example:
```
Original Image: [1024, 512] → 4 tiles of [128, 512]
Tile 1: [0:128, :]     → Prediction 1
Tile 2: [128:256, :]   → Prediction 2  
Tile 3: [256:384, :]   → Prediction 3
Tile 4: [384:512, :]   → Prediction 4 (padded to 128px)
Reassembled: [1024, 512] prediction
```

## 📊 **Visualization Modes**

### Organ Detection Mode (`--see-organs`)
- **Green areas**: Predicted organs (non-tumor dark pixels)
- **Blue overlay**: All dark pixels (< 85 intensity) for context
- **Statistics**: Organ coverage of dark regions

### Tumor Detection Mode (default)
- **Red areas**: Predicted tumors
- **Standard**: True positive/false positive analysis

## 🔧 **Advanced Usage**

### Custom Parameters
```bash
# Different tile height
python tiled_inference.py --checkpoint model.ckpt --tile-height 64

# Different intensity threshold  
python tiled_inference.py --checkpoint model.ckpt --intensity-threshold 100

# Specific image processing
python tiled_inference.py \
  --checkpoint model.ckpt \
  --image my_image.png \
  --see-organs \
  --tile-height 128 \
  --intensity-threshold 85
```

### Integration in Code
```python
from tiled_inference import run_tiled_inference
from models.OrganDetector.model import OrganDetector

# Load model
model = OrganDetector.load_from_checkpoint("checkpoint.ckpt")

# Run inference
prediction = run_tiled_inference(model, image, tile_height=128)

# prediction is [H, W] with values in [0, 1]
organs = prediction > 0.5  # Binary organ mask
```

## 🎨 **Understanding the Output**

### What the Model Learns:
- **Dark pixels that are NOT tumors** = organs (liver, heart, kidneys, etc.)
- **Controls provide positive examples** of what normal organs look like
- **Patients provide mixed examples** (organs vs tumors in dark regions)

### Interpretation:
- **High prediction values** (bright/green) = likely organ tissue
- **Low prediction values** (dark) = likely tumor or background
- **Focus on dark regions** since model only trained on pixels < 85 intensity

## 🔬 **Technical Details**

### Model Architecture:
- **3-level encoder/decoder** (optimized for 128px tiles)
- **Skip connections** for detail preservation
- **Dropout regularization** to prevent overfitting
- **Input validation** ensures tile height compatibility
- **Threshold-aware loss** calculation for focused learning

### Training Strategy:
- **Inverted targets**: Pixels < 85 AND not tumor = positive
- **Threshold-masked loss**: Only calculate loss on pixels < 85 intensity
- **All control data**: Provides positive organ examples
- **Oversampling support** for class balance
- **Tile-level augmentation** for robustness

### Threshold Masking:
- **Training**: Loss calculated only on dark pixels (< 85 intensity)
- **Inference**: Predictions masked to dark regions only
- **Consistency**: Ensures training and inference use same pixel selection
- **Focus**: Model learns organ vs tumor distinction where it matters most

### Performance Benefits:
- **Higher resolution**: Process full images without downsampling
- **Better memory usage**: Process small tiles instead of full images
- **More training data**: Each image becomes multiple training samples
- **Better control usage**: Controls provide positive signal instead of just background

## ✅ **Key Fix: Threshold Masking**

This implementation includes a **critical fix** that ensures training and inference consistency:

### **Problem Before Fix:**
- ❌ Model trained on targets only for pixels < 85 intensity
- ❌ But loss calculated on ALL pixels during training
- ❌ Inference ran on ALL pixels without threshold masking
- ❌ Training/inference mismatch led to poor performance

### **Solution After Fix:**
- ✅ **Training**: Loss calculated ONLY on pixels < 85 intensity
- ✅ **Inference**: Predictions masked to pixels < 85 intensity  
- ✅ **Consistency**: Training and inference use identical pixel selection
- ✅ **Focus**: Model learns organ vs tumor distinction in relevant regions only

### **Implementation:**
```python
# In training (OrganDetector model)
threshold_mask = (images_255 < self.intensity_threshold).float()
loss = calculate_loss_only_on_masked_pixels(pred, target, threshold_mask)

# In inference (tiled_inference.py)  
threshold_mask = image < intensity_threshold
prediction = model_prediction * threshold_mask
```

This ensures the model **only learns and predicts where it matters** - in dark regions where organ vs tumor distinction is relevant.

## 🚨 **Common Issues**

### 1. Model Checkpoint Not Found
```
Error: Could not load OrganDetector: [Errno 2] No such file or directory
```
**Solution**: Train the model first or provide correct checkpoint path

### 2. Tile Height Mismatch
```
AssertionError: Expected height 128, got 256
```
**Solution**: Ensure tile_height matches training configuration

### 3. Memory Issues
```
CUDA out of memory
```
**Solution**: Process tiles on CPU or reduce batch size

### 4. No Organ Predictions
```
All predictions are zero
```
**Solution**: Check intensity_threshold matches training, verify model convergence

## 📈 **Expected Results**

### Good Organ Detection:
- **Dark regions** (< 85 intensity) show organ predictions
- **Anatomically consistent** patterns (organs in expected locations)
- **Controls show more organs** than patients (since no tumors)
- **Clear distinction** between organ and tumor regions in patients

### Performance Metrics:
- **Organ coverage**: % of dark pixels identified as organs
- **Spatial consistency**: Organ predictions form coherent regions
- **Control vs Patient**: Controls should show higher organ ratios

This approach transforms tumor segmentation from "find small tumors in large images" to "distinguish normal organs from tumors in focused dark regions" - making much better use of your control data! 🚀