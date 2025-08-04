#!/usr/bin/env python3
"""
Test the fixed threshold masking implementation.
Run this to verify that threshold masking is working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from debug_threshold_masking import debug_threshold_masking_detailed
from plot_baseline_predictions import plot_baseline_predictions


def main():
    """Test the fixed threshold masking implementation"""
    print("🚀 Testing Fixed Threshold Masking Implementation")
    print("=" * 70)

    # Run detailed debugging
    print("Step 1: Running detailed threshold masking analysis...")
    debug_threshold_masking_detailed()

    print(f"\n" + "=" * 70)
    print("Step 2: Testing with visualization...")

    # Run with organ detection visualization
    try:
        plot_baseline_predictions(
            see_organs=True,
            model_checkpoint="checkpoints/organ-detector-epoch=34-val_dice_patients=0.7060.ckpt",
        )
        print("✅ Visualization test completed!")
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")

    print(f"\n🎯 KEY CHECKS TO VERIFY:")
    print("1. ✅ No predictions in bright areas (should be exactly 0.0)")
    print("2. ✅ All predictions confined to dark pixels (<85 intensity)")
    print("3. ✅ Training and inference use same threshold logic")
    print("4. ✅ Visual overlay shows predictions only in dark regions")

    print(f"\n📊 If you see any predictions in bright areas:")
    print("   🚨 This indicates threshold masking is still failing")
    print("   🔧 Check the tiled_inference.py masking logic")
    print("   🔧 Verify model training uses threshold-aware loss")


if __name__ == "__main__":
    main()
