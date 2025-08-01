import numpy as np
from model import TumorSegmentationModel
import os

# Initialize the model
MODEL_PATH = "checkpoints/best_model.pth" if os.path.exists("checkpoints/best_model.pth") else None
model = TumorSegmentationModel(model_path=MODEL_PATH)

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(img: np.ndarray) -> np.ndarray:
    """
    Predict tumor segmentation using U-Net with global parameters
    
    Args:
        img: Input MIP-PET image as numpy array
        
    Returns:
        segmentation: Binary segmentation mask (0,255) as RGB image
    """
    try:
        # Use the advanced model if available
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            segmentation = model.predict(img)
        else:
            # Fallback to threshold method if model not trained yet
            print("Warning: Using fallback threshold method. Train the model first for better results.")
            threshold = 50
            segmentation = get_threshold_segmentation(img, threshold)
        
        return segmentation
    
    except Exception as e:
        print(f"Error in model prediction: {e}")
        # Fallback to threshold method
        threshold = 50
        segmentation = get_threshold_segmentation(img, threshold)
        return segmentation

def predict_with_analysis(img: np.ndarray) -> dict:
    """
    Predict tumor segmentation with additional global parameter analysis
    
    Args:
        img: Input MIP-PET image as numpy array
        
    Returns:
        dict: Contains segmentation, global parameters, symmetry analysis, and confidence
    """
    try:
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            result = model.predict_with_analysis(img)
            return result
        else:
            # Fallback with basic analysis
            segmentation = get_threshold_segmentation(img, 50)
            return {
                'segmentation': segmentation,
                'global_params': None,
                'symmetry_analysis': model.analyze_symmetry(img),
                'confidence': 0.5  # Default confidence for threshold method
            }
    except Exception as e:
        print(f"Error in detailed prediction: {e}")
        segmentation = get_threshold_segmentation(img, 50)
        return {
            'segmentation': segmentation,
            'global_params': None,
            'symmetry_analysis': None,
            'confidence': 0.5
        }

### DUMMY MODEL (FALLBACK) ###
def get_threshold_segmentation(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    Simple threshold-based segmentation (fallback method)
    
    Args:
        img: Input image
        threshold: Pixel intensity threshold
        
    Returns:
        Binary segmentation mask
    """
    if len(img.shape) == 3:
        # Convert to grayscale if RGB
        gray = np.mean(img, axis=2)
    else:
        gray = img
    
    # Apply threshold (inverted logic: low intensity = tumor)
    binary_mask = (gray < threshold).astype(np.uint8) * 255
    
    # Convert to RGB format
    segmentation = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    
    return segmentation

### UTILITY FUNCTIONS ###
def get_anatomical_region_info(img: np.ndarray) -> dict:
    """
    Extract anatomical region information from image
    This is a simplified version - the full model does this automatically
    """
    h, w = img.shape[:2]
    
    # Simple heuristics based on image regions
    regions = {
        'head_neck': h < 300,  # Shorter images likely head/neck
        'chest': 300 <= h <= 600,  # Medium height likely chest
        'abdomen': 600 <= h <= 800,  # Taller images likely abdomen
        'pelvis': h > 800,  # Very tall images likely pelvis
        'full_body': h > 900  # Full body scans
    }
    
    return regions

def analyze_tumor_characteristics(segmentation: np.ndarray) -> dict:
    """
    Analyze characteristics of detected tumors
    """
    if len(segmentation.shape) == 3:
        binary_mask = segmentation[:, :, 0] > 0
    else:
        binary_mask = segmentation > 0
    
    # Calculate tumor properties
    tumor_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    
    # Find tumor regions (connected components)
    from scipy import ndimage
    try:
        labeled_array, num_features = ndimage.label(binary_mask)
        tumor_regions = num_features
    except:
        tumor_regions = 1 if tumor_pixels > 0 else 0
    
    return {
        'tumor_pixel_count': int(tumor_pixels),
        'tumor_percentage': float(tumor_percentage),
        'num_tumor_regions': int(tumor_regions),
        'has_tumor': tumor_pixels > 0
    }

# Example usage and testing
if __name__ == "__main__":
    # Test with a dummy image
    print("Testing tumor segmentation model...")
    
    # Create a test image (simulate MIP-PET data)
    test_img = np.random.randint(0, 255, size=(400, 300, 3), dtype=np.uint8)
    
    # Add some high-intensity regions (simulate organs with high sugar uptake)
    test_img[50:100, 140:160] = 200  # Brain region
    test_img[300:350, 140:160] = 180  # Bladder region
    
    # Add some low-intensity regions (simulate potential tumors)
    test_img[200:230, 100:130] = 30  # Potential tumor
    
    print(f"Test image shape: {test_img.shape}")
    
    # Test basic prediction
    segmentation = predict(test_img)
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation unique values: {np.unique(segmentation)}")
    
    # Test detailed prediction
    result = predict_with_analysis(test_img)
    print(f"Detailed analysis keys: {result.keys()}")
    
    # Analyze tumor characteristics
    tumor_analysis = analyze_tumor_characteristics(segmentation)
    print(f"Tumor analysis: {tumor_analysis}")
    
    print("Model testing completed successfully!")
    
    