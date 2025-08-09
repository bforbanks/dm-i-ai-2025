#!/usr/bin/env python3
"""
Simple test script to verify the comparison functionality.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_api_validation import ValidationComparator

def test_paths():
    """Test that all required paths exist."""
    print("🔍 Testing paths...")
    
    try:
        comparator = ValidationComparator()
        print("✅ All paths are valid!")
        return True
    except Exception as e:
        print(f"❌ Path test failed: {e}")
        return False

def test_validation_files():
    """Test that validation files can be found."""
    print("\n🔍 Testing validation files...")
    
    try:
        comparator = ValidationComparator()
        validation_files = comparator.get_validation_files()
        print(f"✅ Found {len(validation_files)} validation files")
        
        if validation_files:
            print(f"   First few files: {validation_files[:5]}")
            
            # Test loading first validation mask
            first_patient = validation_files[0]
            mask = comparator.load_validation_mask(first_patient)
            print(f"   ✅ Successfully loaded mask for {first_patient}: {mask.shape}, range: {mask.min()}-{mask.max()}")
            
            # Test loading first original image
            image = comparator.load_original_image(first_patient)
            print(f"   ✅ Successfully loaded image for {first_patient}: {image.shape}")
            
        return True
    except Exception as e:
        print(f"❌ Validation files test failed: {e}")
        return False

def test_mask_normalization():
    """Test mask normalization functionality."""
    print("\n🔍 Testing mask normalization...")
    
    try:
        comparator = ValidationComparator()
        
        # Test with 0-255 range mask
        mask_255 = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        norm_255 = comparator.normalize_mask(mask_255, (0, 1))
        print(f"   ✅ 0-255 mask normalized: range {norm_255.min()}-{norm_255.max()}")
        
        # Test with 0-1 range mask
        mask_1 = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
        norm_1 = comparator.normalize_mask(mask_1, (0, 1))
        print(f"   ✅ 0-1 mask normalized: range {norm_1.min()}-{norm_1.max()}")
        
        return True
    except Exception as e:
        print(f"❌ Mask normalization test failed: {e}")
        return False

def main():
    print("🧪 Testing comparison functionality...")
    
    tests = [
        test_paths,
        test_validation_files,
        test_mask_normalization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The comparison script should work correctly.")
    else:
        print("❌ Some tests failed. Please check the configuration.")

if __name__ == "__main__":
    import numpy as np
    main() 