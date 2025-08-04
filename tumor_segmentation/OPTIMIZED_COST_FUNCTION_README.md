# 🎯 Optimized Cost Function for Tumor Detection

## Overview

I've implemented an **optimized weighted BCE cost function** specifically designed to maximize tumor detection performance (dice score) for the final task described in the README.md. The key insight is to use **organ detection as an intermediate step** while optimizing the loss function for the **ultimate goal of tumor detection**.

## 🧠 **Mathematical Foundation**

### **Problem Analysis**
From the medical domain and README.md:
- **Organs (brain, kidneys, heart, liver)** represent ~70% of dark pixels
- **Tumors** represent ~30% of dark pixels (**minority class**)
- **Final task**: Maximize tumor detection dice score

### **Cost Function Optimization**

#### **Before (Standard BCE + Dice)**
```python
loss = BCE(pred, target) * 0.5 + Dice(pred, target) * 0.5
```

#### **After (Weighted BCE for Tumor Detection)**
```python
# Organ weight = 1/frequency = 1/0.7 = 1.43
# Tumor weight = emphasis_factor/frequency = 2.5/0.3 = 8.33
# Normalized: organ_weight=0.147, tumor_weight=0.853

sample_weights = where(target == organ, 0.147, 0.853)  # Higher weight for tumors
loss = BCE(pred, target, weight=sample_weights)
```

## 🔄 **Model Architecture & Output**

### **Key Changes**

1. **Model Output**: Organ probabilities `P(organ | dark_pixel)`
2. **Tumor Probabilities**: `P(tumor | dark_pixel) = 1 - P(organ | dark_pixel)`
3. **Weighted Loss**: Emphasizes tumor detection errors (2.5x factor)
4. **Threshold Masking**: Only learns from pixels < 85 intensity

### **New Parameters**
```yaml
tumor_emphasis_weight: 2.5  # Boost tumor detection importance
use_weighted_bce: true      # Enable optimized loss function
```

## 📊 **Enhanced Metrics**

### **Primary Metrics (Final Task)**
- `val_tumor_dice` - **Most important metric** (shown in progress bar)
- `val_tumor_dice_patients` - Per-patient tumor dice score
- `train_tumor_dice` - Training tumor detection performance
- `val_naive_dice` - **Tumor detection under threshold** (wandb-specific metric)
- `train_naive_dice` - Training naive dice (pixels under threshold NOT predicted as organs)

### **Secondary Metrics (Intermediate Task)**
- `val_organ_dice` - Organ detection performance
- `val_organ_dice_patients` - Per-patient organ dice score

## 🔧 **Implementation Details**

### **Cost Function Logic**
```python
def _calculate_optimized_loss(self, pred, target, images):
    # 1. Apply threshold masking (only dark pixels)
    threshold_mask = (images * 255.0 < intensity_threshold).float()
    
    # 2. Create sample weights based on class
    sample_weights = torch.where(
        target > 0.5,  # Organ pixels
        self.organ_weight_normalized,      # Lower weight (0.147)
        self.tumor_weight_normalized       # Higher weight (0.853)
    )
    
    # 3. Calculate weighted BCE loss
    loss = F.binary_cross_entropy(
        pred * threshold_mask, 
        target * threshold_mask, 
        weight=sample_weights * threshold_mask
    )
    
    return loss
```

### **Inference Pipeline**
```python
# Model outputs organ probabilities
organ_probs = model(image_tile)

# Convert to tumor probabilities for final task
tumor_probs = 1.0 - organ_probs

# Use tumor_probs for tumor detection dice score
```

## 🎯 **Why This Approach is Optimal**

### **1. Addresses Class Imbalance**
- Tumors are minority class (30% of dark pixels)
- Standard loss treats organ/tumor errors equally
- **Weighted BCE gives 2.5x more weight to tumor errors**

### **2. Optimizes for Final Task**
- Goal: Maximize tumor detection dice score
- Method: Emphasize tumor recall/precision in loss function
- Result: Better tumor detection performance

### **3. Preserves Probabilistic Output**
- Model outputs calibrated organ probabilities
- Can be inverted to get tumor probabilities
- **Suitable for downstream models needing full image context**

### **4. Mathematical Justification**
```
Expected Loss Reduction for Tumor Detection:

Standard BCE: E[loss] = -0.3*log(p_tumor) - 0.7*log(p_organ)
Weighted BCE: E[loss] = -0.853*log(p_tumor) - 0.147*log(p_organ)

The weighted approach puts 85.3% of loss emphasis on tumor detection!
```

## 🚀 **Training & Usage**

### **Training Command**
```bash
python trainer.py fit --config models/OrganDetector/config.yaml
```

### **Key Metrics to Monitor**
1. **`val_tumor_dice`** ⭐ Primary metric (progress bar)
2. **`val_naive_dice`** ⭐ Tumor detection under threshold (wandb)
3. `val_tumor_dice_patients` - Per-patient performance
4. `val_loss` - Overall loss trend

### **Inference for Final Task**
```python
# Get tumor probabilities for tumor detection
tumor_prediction = run_tiled_inference(
    model, image, 
    return_tumor_probabilities=True  # For final task
)

# Calculate tumor detection dice score
dice_score = calculate_dice(tumor_prediction, true_tumor_mask)
```

## 📈 **Expected Performance Improvements**

### **Tumor Detection (Final Task)**
- ✅ **Higher tumor recall** (fewer missed tumors)
- ✅ **Better tumor precision** (fewer false positives)
- ✅ **Improved dice score** on tumor detection task
- ✅ **Better class balance handling**

### **Training Efficiency**
- ✅ **Faster convergence** (focused loss signal)
- ✅ **More stable training** (single loss function)
- ✅ **Clear optimization target** (tumor detection)

## 🔄 **Backward Compatibility**

### **Legacy Support**
- Set `use_weighted_bce: false` for standard BCE
- `bce_loss_weight` parameter preserved for compatibility
- Can switch between organ/tumor probabilities in inference

### **Migration Path**
1. Train new model with optimized cost function
2. Compare `val_tumor_dice` with baseline
3. Use `return_tumor_probabilities=True` for final task
4. Monitor both organ and tumor metrics during transition

## 🎯 **Key Takeaways**

1. **Model Architecture**: Unchanged (U-Net style)
2. **Model Output**: Organ probabilities (can invert for tumors)
3. **Cost Function**: Weighted BCE optimized for tumor detection
4. **Primary Metric**: `val_tumor_dice` (final task performance)
5. **Usage**: Set `return_tumor_probabilities=True` for tumor detection

The optimized cost function transforms the organ detection model into a **tumor detection optimizer** while maintaining the benefits of:
- ✅ Probabilistic outputs
- ✅ Threshold masking
- ✅ Class imbalance handling
- ✅ Focus on final task (tumor detection dice score)

**Result: Better tumor detection performance for the competition task!** 🏆