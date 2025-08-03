# Model Evaluation for Tumor Segmentation

This directory contains comprehensive evaluation tools for tumor segmentation models. The evaluation system supports both validation sets (with ground truth) and evaluation sets (without ground truth).

## 🚀 Quick Start

### Basic Evaluation
```bash
# Run complete evaluation with a trained model
python evaluate_model.py --checkpoint path/to/model.ckpt --data_dir data/

# Custom settings
python evaluate_model.py \
    --checkpoint path/to/model.ckpt \
    --data_dir data/ \
    --batch_size 8 \
    --save_dir my_evaluation_results/
```

### Example Usage
```bash
# See example statistics and run demo
python example_evaluation.py
```

## 📊 What Gets Evaluated

### Validation Set (with ground truth)
- **Standard Metrics**: Dice Score, IoU, Precision, Recall, F1-Score
- **Per-image Analysis**: Individual performance metrics
- **Statistical Analysis**: Mean, standard deviation, distributions

### Evaluation Set (without ground truth)
- **Tumor Detection Statistics**:
  - Number of images with tumor predictions
  - Number of images without tumor predictions
  - Overall tumor detection rate
  
- **Cluster Analysis**:
  - Average number of tumor clusters per image
  - Cluster distribution analysis
  - Connected component analysis
  
- **Resolution Statistics**:
  - Image resolution distribution
  - Mean and standard deviation of image sizes
  - Most common resolution
  - Unique resolution count

## 📈 Generated Outputs

### Files Created
- `validation_detailed_results.csv` - Per-image validation metrics
- `evaluation_detailed_results.csv` - Per-image evaluation analysis
- `validation_metrics_distribution.png` - Validation metrics histograms
- `evaluation_analysis.png` - Evaluation set analysis plots

### Visualization Examples
1. **Validation Metrics**: Histograms of Dice, IoU, Precision, Recall, F1
2. **Tumor Detection**: Pie chart of tumor vs. no-tumor predictions
3. **Cluster Analysis**: Histogram of cluster counts per image
4. **Resolution Analysis**: Scatter plot of image dimensions
5. **Pixel Analysis**: Distribution of tumor pixel counts

## 🔧 Technical Details

### Data Structure Expected
```
data/
├── patients/
│   ├── imgs/           # Patient images
│   └── labels/         # Ground truth masks
├── controls/
│   └── imgs/           # Control images (no tumors)
└── evaluation_set/     # Evaluation images (no ground truth)
    └── *.png           # Images for prediction analysis
```

### Model Requirements
- Models should inherit from `BaseModel` or be compatible with PyTorch Lightning
- Models should output sigmoid probabilities (0-1 range)
- Expected input: Single-channel grayscale images
- Expected output: Single-channel segmentation masks

### Statistics Calculated

#### For Validation Set (with ground truth):
```python
{
    'mean_dice': 0.845,
    'std_dice': 0.123,
    'mean_iou': 0.731,
    'mean_precision': 0.823,
    'mean_recall': 0.867,
    'mean_f1': 0.844,
    'total_images': 156
}
```

#### For Evaluation Set (prediction analysis):
```python
{
    'total_images': 100,
    'images_with_tumors': 73,
    'images_without_tumors': 27,
    'tumor_detection_rate': 0.73,
    'mean_clusters_per_image': 1.2,
    'mean_clusters_when_tumor_detected': 1.6,
    'mean_tumor_pixels': 423.5,
    'std_tumor_pixels': 289.1,
    'resolution_stats': {
        'mean_width': 512.0,
        'mean_height': 512.0,
        'std_width': 128.0,
        'std_height': 64.0,
        'unique_resolutions': 8,
        'most_common_resolution': (512, 512)
    }
}
```

## 🛠️ Customization

### Adding New Metrics
To add custom metrics to the evaluation, modify the `ModelEvaluator` class:

```python
# In evaluate_model.py
def evaluate_validation_set(self, val_dataloader):
    # Add your custom metric calculation here
    custom_metric = calculate_my_metric(pred_binary, mask_binary)
    
    # Store in results
    self.val_results.append({
        # ... existing metrics ...
        'custom_metric': custom_metric
    })
```

### Custom Visualization
```python
# Add to _create_visualizations method
def _create_visualizations(self, val_summary, eval_summary, save_dir):
    # Your custom plots here
    plt.figure(figsize=(10, 6))
    # ... your plotting code ...
    plt.savefig(save_dir / 'my_custom_plot.png')
```

## 📋 Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--checkpoint` | str | Required | Path to model checkpoint (.ckpt file) |
| `--data_dir` | str | `data` | Path to data directory |
| `--batch_size` | int | `16` | Batch size for evaluation |
| `--num_workers` | int | `4` | Number of worker threads |
| `--save_dir` | str | `evaluation_results` | Directory to save results |
| `--image_size` | int | `256` | Image size for model input |

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Error: Failed to load checkpoint
   ```
   - Check checkpoint path exists
   - Ensure model architecture matches checkpoint

2. **Data Directory Not Found**
   ```
   Warning: Evaluation directory does not exist
   ```
   - Check data directory structure
   - Ensure evaluation_set folder exists

3. **Out of Memory**
   ```
   CUDA out of memory
   ```
   - Reduce batch_size: `--batch_size 4`
   - Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

### Memory Usage
- **Validation set**: ~2GB GPU memory for 100 images (batch_size=16)
- **Evaluation set**: Memory usage depends on number of images
- **Large datasets**: Consider reducing batch_size

## 🔍 Understanding Results

### Good Model Indicators
- **High Dice Score** (>0.8): Good overlap between prediction and ground truth
- **Balanced Precision/Recall**: Neither missing tumors nor over-predicting
- **Consistent Cluster Count**: Usually 1-2 clusters per tumor image
- **Reasonable Detection Rate**: Depends on your dataset characteristics

### Red Flags
- **Very Low Dice** (<0.3): Poor segmentation quality
- **High Cluster Count** (>5): Noisy predictions
- **Extreme Detection Rates** (0% or 100%): Model bias issues

## 🤝 Contributing

To extend the evaluation system:
1. Add new metrics to `ModelEvaluator` class
2. Update visualization functions
3. Add corresponding tests
4. Update this documentation

## 📚 Related Files
- `data/data_default.py` - Data loading and evaluation dataset
- `models/base_model.py` - Base model class with inference methods
- `utils.py` - Utility functions for metrics calculation
- `evaluate_model.py` - Main evaluation script
- `example_evaluation.py` - Usage examples