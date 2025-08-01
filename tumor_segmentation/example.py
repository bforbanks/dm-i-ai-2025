import numpy as np


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(img: np.ndarray) -> np.ndarray:
    threshold = 50
    segmentation = get_threshold_segmentation(img, threshold)
    return segmentation


### DUMMY MODEL ###
def get_threshold_segmentation(img: np.ndarray, threshold: int) -> np.ndarray:
    # Handle different input formats
    if len(img.shape) == 3 and img.shape[2] == 1:
        # Grayscale with channel dimension (H, W, 1) -> (H, W)
        img_2d = img.squeeze(2)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # RGB format (H, W, 3) -> convert to grayscale (H, W)
        img_2d = np.mean(img, axis=2)
    else:
        # Already 2D grayscale (H, W)
        img_2d = img

    # Apply thresholding on 2D grayscale image
    return (img_2d < threshold).astype(np.uint8) * 255
