#!/usr/bin/env python3
"""
Test script for the nnUNetV2 .pth API.

This script tests the API by:
1. Loading a sample image
2. Making a prediction request
3. Saving the result
4. Displaying timing information
"""

import requests
import base64
import cv2
import numpy as np
import time
import os
from pathlib import Path

# API configuration
API_URL = "http://localhost:9052"
HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Predictor loaded: {health_data.get('predictor_loaded')}")
            print(f"   Device: {health_data.get('device')}")
            print(f"   Model folder: {health_data.get('model_folder')}")
            print(f"   Configuration: {health_data.get('configuration_name')}")
            print(f"   Total predictions: {health_data.get('total_predictions')}")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def create_test_image(width=400, height=600):
    """Create a test image for prediction"""
    print(f"🖼️  Creating test image ({width}x{height})...")
    
    # Create a simple test image with some patterns
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Add some patterns to make it interesting
    # Horizontal lines
    for i in range(0, height, 50):
        image[i:i+10, :] = 128
    
    # Vertical lines
    for i in range(0, width, 50):
        image[:, i:i+10] = 128
    
    # Add some random noise
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Add some circular patterns (simulating potential tumor regions)
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
    image[mask] = 200
    
    # Add another circle
    center_y2, center_x2 = height // 4, width // 4
    mask2 = (x - center_x2)**2 + (y - center_y2)**2 <= 30**2
    image[mask2] = 180
    
    return image

def encode_image(image):
    """Encode image to base64 string"""
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', image_rgb)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def decode_segmentation(segmentation_base64):
    """Decode base64 segmentation to numpy array"""
    segmentation_bytes = base64.b64decode(segmentation_base64)
    segmentation = cv2.imdecode(np.frombuffer(segmentation_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return segmentation

def test_prediction(image):
    """Test the prediction endpoint"""
    print("🚀 Testing prediction endpoint...")
    
    # Encode image
    encode_start = time.time()
    image_base64 = encode_image(image)
    encode_time = time.time() - encode_start
    print(f"   ⏱️  Image encoding: {encode_time:.3f}s")
    
    # Make prediction request
    predict_start = time.time()
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json={"img": image_base64},
            timeout=60  # Longer timeout for prediction
        )
        
        if response.status_code == 200:
            result = response.json()
            predict_time = time.time() - predict_start
            print(f"   ⏱️  API prediction: {predict_time:.3f}s")
            
            # Decode result
            decode_start = time.time()
            segmentation = decode_segmentation(result["img"])
            decode_time = time.time() - decode_start
            print(f"   ⏱️  Result decoding: {decode_time:.3f}s")
            
            total_time = encode_time + predict_time + decode_time
            print(f"✅ Prediction successful!")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Input shape: {image.shape}")
            print(f"   Output shape: {segmentation.shape}")
            print(f"   Output range: [{segmentation.min()}, {segmentation.max()}]")
            
            return segmentation, total_time
        else:
            print(f"❌ Prediction failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return None, 0
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return None, 0

def save_results(image, segmentation, output_dir="test_results"):
    """Save test results"""
    print(f"💾 Saving results to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image
    input_path = os.path.join(output_dir, "test_input.png")
    cv2.imwrite(input_path, image)
    print(f"   Input image: {input_path}")
    
    # Save segmentation
    output_path = os.path.join(output_dir, "test_segmentation.png")
    cv2.imwrite(output_path, segmentation)
    print(f"   Segmentation: {output_path}")
    
    # Create side-by-side comparison
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    if len(segmentation.shape) == 2:
        segmentation_rgb = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2RGB)
    else:
        segmentation_rgb = segmentation
    
    # Create comparison image
    comparison = np.hstack([image_rgb, segmentation_rgb])
    comparison_path = os.path.join(output_dir, "test_comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"   Comparison: {comparison_path}")

def main():
    """Main test function"""
    print("🧪 nnUNetV2 .pth API Test")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Health check failed. Make sure the API is running.")
        return
    
    print()
    
    # Create test image
    test_image = create_test_image(400, 600)
    print(f"✅ Test image created: {test_image.shape}")
    
    # Test prediction
    segmentation, total_time = test_prediction(test_image)
    
    if segmentation is not None:
        print()
        # Save results
        save_results(test_image, segmentation)
        
        print()
        print("🎉 Test completed successfully!")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Results saved in: test_results/")
    else:
        print("❌ Test failed!")

if __name__ == "__main__":
    main() 