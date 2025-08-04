#!/usr/bin/env python3
"""
Setup script for the nnUNetV2 .pth API.

This script helps users:
1. Check if nnUNetV2 is installed
2. Verify model folder structure
3. Create .env file with correct settings
4. Test the API setup

Usage:
    python setup_nnunetv2_pth_api.py                                    # Auto-select best model
    python setup_nnunetv2_pth_api.py path/to/specific/model            # Use specific model
"""

import os
import sys
import json
from pathlib import Path
import subprocess

def check_nnunetv2_installation():
    """Check if nnUNetV2 is installed"""
    print("🔍 Checking nnUNetV2 installation...")
    
    try:
        import nnunetv2
        # print(f"✅ nnUNetV2 is installed: {nnunetv2.__version__}")
        return True
    except ImportError:
        print("❌ nnUNetV2 is not installed")
        print("   Please install nnUNetV2: pip install nnunetv2")
        return False

def find_trained_models():
    """Find available trained models"""
    print("🔍 Looking for trained models...")
    
    results_dir = Path("data_nnUNet/results/Dataset001_TumorSegmentation")
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return []
    
    models = []
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("nnUNetTrainer"):
            # Check if it has the required files
            plans_file = model_dir / "plans.json"
            dataset_file = model_dir / "dataset.json"
            fold_dir = model_dir / "fold_0"
            checkpoint_file = fold_dir / "checkpoint_best.pth"
            
            if plans_file.exists() and dataset_file.exists() and checkpoint_file.exists():
                models.append(str(model_dir))
                print(f"✅ Found model: {model_dir.name}")
            else:
                print(f"⚠️  Incomplete model: {model_dir.name}")
    
    return models

def get_model_configuration(model_folder):
    """Get configuration name from plans.json"""
    plans_file = Path(model_folder) / "plans.json"
    if not plans_file.exists():
        return None
    
    try:
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        
        # Get the first configuration that's not inherited
        configurations = plans.get("configurations", {})
        for config_name, config in configurations.items():
            if "inherits_from" not in config:
                return config_name
        
        # If all inherit, return the first one
        if configurations:
            return list(configurations.keys())[0]
        
        return None
    except Exception as e:
        print(f"⚠️  Error reading plans.json: {e}")
        return None

def create_env_file(model_folder, configuration_name):
    """Create .env file with correct settings"""
    print("📝 Creating .env file...")
    
    env_content = f"""# nnUNetV2 .pth API Configuration

# Model Configuration
MODEL_FOLDER={model_folder}
CONFIGURATION_NAME={configuration_name}
USE_FOLDS=0
CHECKPOINT_NAME=checkpoint_best.pth

# API Configuration
HOST=0.0.0.0
PORT=9057

# Performance Settings
TILE_STEP_SIZE=0.5
USE_MIRRORING=true
USE_GAUSSIAN=true
PERFORM_EVERYTHING_ON_DEVICE=true

# Validation Settings
SKIP_VALIDATION=false
VALIDATION_API_URL=https://cases.dmiai.dk/api/v1/usecases/tumor-segmentation/validate/queue
VALIDATION_TOKEN=18238e2f472643739573ad26f3680c51
PREDICT_URL=https://your-api-url.com/predict
"""
    
    env_file = Path(".env")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created .env file: {env_file}")
    return env_file

def test_api_setup():
    """Test if the API can be started"""
    print("🧪 Testing API setup...")
    
    try:
        # Try to import the API module
        import api_nnunetv2_pth
        print("✅ API module imports successfully")
        return True
    except Exception as e:
        print(f"❌ API module import failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 nnUNetV2 .pth API Setup")
    print("=" * 50)
    
    # Allow command line argument to specify model
    if len(sys.argv) > 1:
        specified_model = sys.argv[1]
        print(f"🎯 Using specified model: {specified_model}")
        if not Path(specified_model).exists():
            print(f"❌ Specified model not found: {specified_model}")
            return False
        selected_model = specified_model
        configuration_name = get_model_configuration(selected_model)
        if not configuration_name:
            print("❌ Setup failed: Could not determine configuration name")
            return False
        print(f"   Configuration: {configuration_name}")
        print()
        
        # Create .env file
        env_file = create_env_file(selected_model, configuration_name)
        print()
        
        # Test API setup
        if not test_api_setup():
            print("❌ Setup failed: API module test failed")
            return False
        
        print()
        print("🎉 Setup completed successfully!")
        print()
        print("📋 Next steps:")
        print("1. Start the API:")
        print("   python api_nnunetv2_pth.py")
        print()
        print("2. Test the API:")
        print("   python test_nnunetv2_pth_api.py")
        print()
        print("3. Use with your own images:")
        print("   python example_nnunetv2_pth_usage.py path/to/your/image.png")
        print()
        print("📖 For more information, see README_nnUNetV2_pth_API.md")
        
        return True
    
    # Check nnUNetV2 installation
    if not check_nnunetv2_installation():
        print("\n❌ Setup failed: nnUNetV2 not installed")
        return False
    
    print()
    
    # Find trained models
    models = find_trained_models()
    if not models:
        print("\n❌ Setup failed: No trained models found")
        print("   Please train a model first or check the data_nnUNet/results directory")
        return False
    
    print()
    
    # Select model - prefer the resenc_optimized model if available
    selected_model = None
    for model in models:
        if "resenc_optimized" in model:
            selected_model = model
            break
    
    # If no resenc_optimized model found, use the first one
    if selected_model is None and models:
        selected_model = models[0]
    
    if selected_model is None:
        print("❌ Setup failed: No valid models found")
        return False
        
    print(f"🎯 Using model: {Path(selected_model).name}")
    
    # Get configuration name
    configuration_name = get_model_configuration(selected_model)
    if not configuration_name:
        print("❌ Setup failed: Could not determine configuration name")
        return False
    
    print(f"   Configuration: {configuration_name}")
    print()
    
    # Create .env file
    env_file = create_env_file(selected_model, configuration_name)
    print()
    
    # Test API setup
    if not test_api_setup():
        print("❌ Setup failed: API module test failed")
        return False
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("📋 Next steps:")
    print("1. Start the API:")
    print("   python api_nnunetv2_pth.py")
    print()
    print("2. Test the API:")
    print("   python test_nnunetv2_pth_api.py")
    print()
    print("3. Use with your own images:")
    print("   python example_nnunetv2_pth_usage.py path/to/your/image.png")
    print()
    print("📖 For more information, see README_nnUNetV2_pth_API.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 