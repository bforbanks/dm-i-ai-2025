#!/usr/bin/env python3
"""
Download script for MedSAM pretrained weights.
Downloads the ViT-B/16 weights from Zenodo and saves them to checkpoints/medsam_vit_b.pth
"""

import os
import requests
import zipfile
from pathlib import Path


def download_medsam_weights():
    """Download MedSAM pretrained weights from Zenodo"""
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # MedSAM weights URL (Zenodo)
    weights_url = "https://zenodo.org/record/7860267/files/medsam_vit_b.pth"
    weights_path = checkpoint_dir / "medsam_vit_b.pth"
    
    # Check if weights already exist
    if weights_path.exists():
        print(f"Weights already exist at {weights_path}")
        print("Skipping download.")
        return
    
    print(f"Downloading MedSAM weights from {weights_url}")
    print(f"Saving to {weights_path}")
    
    try:
        # Download the weights
        response = requests.get(weights_url, stream=True)
        response.raise_for_status()
        
        # Save the weights
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded weights to {weights_path}")
        
        # Verify file size (should be around 330MB)
        file_size = weights_path.stat().st_size
        print(f"File size: {file_size / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"Error downloading weights: {e}")
        print("Please download manually from:")
        print("https://zenodo.org/record/7860267/files/medsam_vit_b.pth")
        print("And save to checkpoints/medsam_vit_b.pth")


if __name__ == "__main__":
    download_medsam_weights() 