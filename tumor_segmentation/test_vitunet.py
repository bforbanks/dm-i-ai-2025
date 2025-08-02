#!/usr/bin/env python3
"""
Test script for ViTUNet model.
Verifies that the model can be instantiated and run a forward pass.
"""

import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ViTUNet.model import ViTUNet


def test_vitunet():
    """Test ViTUNet model instantiation and forward pass"""
    
    print("Testing ViTUNet model...")
    
    # Create model
    model = ViTUNet(
        in_channels=3,
        num_classes=1,
        img_size=256,
        pretrained_weights_path=None  # Don't load weights for testing
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    print(f"Input shape: {input_tensor.shape}")
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Verify output is binary (after sigmoid)
    assert output.shape == (batch_size, 1, 256, 256), f"Expected shape {(batch_size, 1, 256, 256)}, got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, f"Output should be in [0,1] range, got [{output.min():.4f}, {output.max():.4f}]"
    
    print("✅ ViTUNet test passed!")
    
    return model


if __name__ == "__main__":
    test_vitunet() 