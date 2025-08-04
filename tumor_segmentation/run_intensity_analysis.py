#!/usr/bin/env python3
"""
Quick script to run tumor intensity analysis

Simply run: python run_intensity_analysis.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_tumor_intensity import TumorIntensityAnalyzer


def main():
    """Run intensity analysis with default settings"""
    print("🔍 Tumor Pixel Intensity Analysis")
    print("=" * 40)
    print("📁 Analyzing data in: data/patients/")
    print("📊 This will examine pixel intensities under tumor masks")
    print(
        "🎯 Goal: Determine if intensity thresholding is viable for inverted segmentation"
    )
    print("")

    try:
        analyzer = TumorIntensityAnalyzer()
        results = analyzer.run_complete_analysis()

        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 60)
        print("📊 Results saved in: intensity_analysis_results/")
        print("📈 Visualizations: tumor_intensity_analysis.png")
        print("🎚️  Thresholds: threshold_optimization.png")
        print("📄 Data: per_image_statistics.csv, pixel_samples.csv")

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure 'data/patients/imgs/' contains patient_*.png files")
        print("2. Make sure 'data/patients/labels/' contains segmentation_*.png files")
        print("3. Ensure image and mask files have matching names")
        print("   (patient_001.png → segmentation_001.png)")


if __name__ == "__main__":
    main()
