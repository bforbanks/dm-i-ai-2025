#!/usr/bin/env python3
"""
📥 EMBEDDING MODEL DOWNLOADER
Downloads and verifies sentence transformer models for local use.

This script ensures all models are properly downloaded and cached
before running the optimization script.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict

def check_sentence_transformers():
    """Check if sentence-transformers is available"""
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers available")
        return True
    except ImportError:
        print("❌ sentence-transformers not available")
        print("   Install with: pip install sentence-transformers")
        return False

def get_cache_dir() -> Path:
    """Get the sentence transformers cache directory"""
    cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
    return cache_dir

def get_model_path(model_name: str) -> Path:
    """Get the local path for a model"""
    cache_dir = get_cache_dir()
    return cache_dir / model_name

def download_model(model_name: str) -> bool:
    """Download a single model with verification"""
    print(f"\n📥 Downloading {model_name}...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Get expected path
        model_path = get_model_path(model_name)
        
        # Check if already exists
        if model_path.exists():
            print(f"✅ {model_name} - Already available at {model_path}")
            return True
        
        print(f"   Downloading to: {model_path}")
        
        # Download the model
        start_time = time.time()
        model = SentenceTransformer(model_name)
        download_time = time.time() - start_time
        
        # Verify download
        if model_path.exists():
            # Test the model works
            try:
                test_text = "This is a test sentence."
                embedding = model.encode([test_text])
                print(f"✅ {model_name} - Downloaded successfully in {download_time:.1f}s")
                print(f"   Model path: {model_path}")
                print(f"   Embedding shape: {embedding.shape}")
                return True
            except Exception as e:
                print(f"❌ {model_name} - Model test failed: {e}")
                return False
        else:
            print(f"❌ {model_name} - Download failed: Model path not found")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} - Download failed: {e}")
        return False

def verify_model(model_name: str) -> bool:
    """Verify a model is working correctly"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_path = get_model_path(model_name)
        if not model_path.exists():
            print(f"❌ {model_name} - Not found at {model_path}")
            return False
        
        # Load and test the model
        model = SentenceTransformer(str(model_path))
        test_texts = [
            "This is a test sentence.",
            "Another test sentence for verification.",
            "Medical emergency response protocol."
        ]
        
        embeddings = model.encode(test_texts)
        
        print(f"✅ {model_name} - Verified working")
        print(f"   Path: {model_path}")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Model dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} - Verification failed: {e}")
        return False

def list_available_models() -> List[str]:
    """List all available models in cache"""
    cache_dir = get_cache_dir()
    available_models = []
    
    if cache_dir.exists():
        for model_dir in cache_dir.iterdir():
            if model_dir.is_dir():
                available_models.append(model_dir.name)
    
    return available_models

def main():
    """Main download function"""
    print("📥 EMBEDDING MODEL DOWNLOADER")
    print("=" * 50)
    
    # Check dependencies
    if not check_sentence_transformers():
        sys.exit(1)
    
    # Models to download
    models_to_download = [
        'all-mpnet-base-v2',      # ~438MB - Most accurate
        'all-MiniLM-L6-v2',       # ~90MB - Fast, good accuracy
        'all-MiniLM-L12-v2'       # ~120MB - Balanced
    ]
    
    print(f"\n🎯 Target models: {len(models_to_download)}")
    for model in models_to_download:
        print(f"   - {model}")
    
    # Check cache directory
    cache_dir = get_cache_dir()
    print(f"\n📁 Cache directory: {cache_dir}")
    
    if not cache_dir.exists():
        print("   Creating cache directory...")
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download models
    print(f"\n🚀 DOWNLOADING MODELS")
    print("=" * 30)
    
    successful_downloads = []
    failed_downloads = []
    
    for model_name in models_to_download:
        if download_model(model_name):
            successful_downloads.append(model_name)
        else:
            failed_downloads.append(model_name)
    
    # Verify all models
    print(f"\n🔍 VERIFYING MODELS")
    print("=" * 30)
    
    verified_models = []
    for model_name in successful_downloads:
        if verify_model(model_name):
            verified_models.append(model_name)
    
    # Summary
    print(f"\n📊 DOWNLOAD SUMMARY")
    print("=" * 30)
    print(f"✅ Successfully downloaded: {len(successful_downloads)}/{len(models_to_download)}")
    print(f"✅ Verified working: {len(verified_models)}/{len(models_to_download)}")
    
    if successful_downloads:
        print(f"\n✅ Working models:")
        for model in verified_models:
            print(f"   - {model}")
    
    if failed_downloads:
        print(f"\n❌ Failed models:")
        for model in failed_downloads:
            print(f"   - {model}")
    
    # List all available models
    print(f"\n📋 ALL AVAILABLE MODELS")
    print("=" * 30)
    all_models = list_available_models()
    if all_models:
        for model in sorted(all_models):
            print(f"   - {model}")
    else:
        print("   No models found in cache")
    
    # Instructions for next steps
    print(f"\n🎯 NEXT STEPS")
    print("=" * 30)
    if verified_models:
        print("✅ Models are ready! You can now run:")
        print("   python match-and-choose-model-1/optimize_search.py")
    else:
        print("❌ No models available. Try:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have enough disk space")
        print("   3. Try downloading models individually:")
        for model in models_to_download:
            print(f"      python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model}')\"")
    
    return len(verified_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 