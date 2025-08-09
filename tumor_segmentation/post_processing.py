import numpy as np
import cv2
from typing import Tuple


def filter_disconnected_tumors(
    original_image: np.ndarray,
    tumor_mask: np.ndarray,
    white_threshold: int = 240
) -> np.ndarray:
    """
    Filter out tumor predictions that are not connected to the main body.
    
    This function:
    1. Identifies the main body as the largest connected component in non-white pixels
    2. Removes tumor predictions that don't touch the main body
    
    Args:
        original_image: Original input image (H, W) or (H, W, C)
        tumor_mask: Binary tumor segmentation mask (H, W) with values 0 or 255
        white_threshold: Pixel values >= this are considered background/white (default: 240)
        
    Returns:
        Filtered tumor mask with disconnected tumors removed
    """
    # Convert to grayscale if needed
    if len(original_image.shape) == 3:
        if original_image.shape[2] == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = original_image[:, :, 0]
    else:
        gray_image = original_image.copy()
    
    # Convert tumor mask to binary (0, 1)
    tumor_binary = (tumor_mask > 127).astype(np.uint8)
    
    # Create body mask: pixels that are NOT white (background)
    body_mask = (gray_image < white_threshold).astype(np.uint8)
    
    # Find connected components in the body mask
    num_labels, labels = cv2.connectedComponents(body_mask, connectivity=8)
    
    if num_labels <= 1:
        # No body found or only background, return empty mask
        print("⚠️  No body regions found, returning empty mask")
        return np.zeros_like(tumor_mask)
    
    # Find the largest connected component (excluding background label 0)
    component_sizes = []
    for label_id in range(1, num_labels):
        component_size = np.sum(labels == label_id)
        component_sizes.append((component_size, label_id))
    
    # Get the largest component (main body)
    largest_size, main_body_label = max(component_sizes)
    main_body_mask = (labels == main_body_label).astype(np.uint8)
    
    print(f"📊 Found {num_labels-1} body components, largest has {largest_size} pixels")
    
    # Find connected components in the tumor mask
    tumor_num_labels, tumor_labels = cv2.connectedComponents(tumor_binary, connectivity=8)
    
    if tumor_num_labels <= 1:
        # No tumors found
        print("📊 No tumor regions found")
        return tumor_mask.copy()
    
    # For each tumor component, check if it touches the main body
    filtered_tumor_mask = np.zeros_like(tumor_binary)
    connected_tumors = 0
    disconnected_tumors = 0
    
    for tumor_label_id in range(1, tumor_num_labels):
        tumor_component = (tumor_labels == tumor_label_id).astype(np.uint8)
        tumor_size = np.sum(tumor_component)
        
        # Check if this tumor component overlaps with the main body
        # We dilate the tumor component slightly to check for adjacency
        kernel = np.ones((3, 3), np.uint8)
        dilated_tumor = cv2.dilate(tumor_component, kernel, iterations=1)
        
        # Check if dilated tumor overlaps with main body
        overlap = np.sum(dilated_tumor * main_body_mask)
        
        if overlap > 0:
            # This tumor is connected to the main body, keep it
            filtered_tumor_mask = np.logical_or(filtered_tumor_mask, tumor_component).astype(np.uint8)
            connected_tumors += 1
            print(f"  ✅ Tumor component {tumor_label_id} (size: {tumor_size}) is connected to body")
        else:
            # This tumor is disconnected, remove it
            disconnected_tumors += 1
            print(f"  ❌ Tumor component {tumor_label_id} (size: {tumor_size}) is disconnected, removing")
    
    print(f"📊 Kept {connected_tumors} connected tumors, removed {disconnected_tumors} disconnected tumors")
    
    # Convert back to original mask format (0, 255)
    result_mask = (filtered_tumor_mask * 255).astype(np.uint8)
    
    return result_mask


def filter_disconnected_tumors_3d(
    original_image: np.ndarray,
    tumor_mask: np.ndarray,
    white_threshold: int = 240
) -> np.ndarray:
    """
    Filter disconnected tumors for 3D images (H, W, 3).
    
    This is a wrapper around filter_disconnected_tumors that handles 3D input/output.
    """
    if len(tumor_mask.shape) == 3:
        # Process first channel and replicate to all channels
        filtered_2d = filter_disconnected_tumors(
            original_image, tumor_mask[:, :, 0], white_threshold
        )
        # Replicate to all channels
        result = np.stack([filtered_2d] * tumor_mask.shape[2], axis=-1)
        return result
    else:
        # 2D case
        return filter_disconnected_tumors(original_image, tumor_mask, white_threshold)