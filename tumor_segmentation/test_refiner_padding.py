#!/usr/bin/env python3
"""Test script to verify Refiner U-Net handles variable height images correctly."""

import torch
import numpy as np
from pathlib import Path

from models.RefinerUNet.model import RefinerUNet


def test_variable_heights():
    """Test that the model can handle different image heights."""
    
    # Test different heights that should work
    test_heights = [300, 400, 500, 600, 700, 800, 900, 1000]
    width = 400
    
    model = RefinerUNet(in_channels=6, num_classes=2)  # 2 softmax + 1 petmr + 3 aux channels
    model.eval()
    
    print("Testing variable height images...")
    print(f"Model expects input channels: {model.delta_conv.in_channels}")
    
    for height in test_heights:
        # Create dummy input: [softmax(2), petmr(1), entropy, distance, y_coord]
        dummy_input = torch.randn(1, 6, height, width)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            # Check output has same spatial dimensions as input
            expected_shape = (1, 2, height, width)  # 2 classes output
            actual_shape = output.shape
            
            if actual_shape == expected_shape:
                print(f"✅ Height {height:4d}px: {actual_shape}")
            else:
                print(f"❌ Height {height:4d}px: expected {expected_shape}, got {actual_shape}")
                
        except Exception as e:
            print(f"❌ Height {height:4d}px: Error - {e}")
    
    print("\nTesting edge cases...")
    
    # Test heights that need padding
    edge_heights = [301, 399, 401, 799, 1001]
    for height in edge_heights:
        dummy_input = torch.randn(1, 6, height, width)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✅ Edge case {height:4d}px: {output.shape}")
        except Exception as e:
            print(f"❌ Edge case {height:4d}px: Error - {e}")


def test_padding_behavior():
    """Test that padding is applied correctly."""
    
    model = RefinerUNet(in_channels=6, num_classes=2)
    model.eval()
    
    # Test a height that needs padding (e.g., 301 needs 15px padding to reach 316)
    height, width = 301, 400
    dummy_input = torch.randn(1, 6, height, width)
    
    print(f"\nTesting padding behavior for {height}x{width} image...")
    
    # The model should pad internally and return original dimensions
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Padding successful: {output.shape == dummy_input.shape}")


if __name__ == "__main__":
    test_variable_heights()
    test_padding_behavior() 