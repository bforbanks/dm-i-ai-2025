#!/usr/bin/env python3
"""
Simple script to run model evaluation without any configuration needed.

Just run: python run_evaluation.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_model import load_model, ModelEvaluator
from data.data_default import TumorSegmentationDataModule


def main():
    """Run evaluation with default settings"""
    print("🚀 Running Tumor Segmentation Model Evaluation")
    print("=" * 50)

    # Default configuration
    data_dir = "data"
    batch_size = 8  # Smaller batch size for stability
    image_size = 256
    save_dir = "evaluation_results"

    print(f"📁 Data directory: {data_dir}")
    print(f"🔧 Batch size: {batch_size}")
    print(f"📏 Image size: {image_size}")
    print(f"💾 Save directory: {save_dir}")

    # Set device
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Load model
    print("\n📦 Loading model...")
    model = load_model()
    evaluator = ModelEvaluator(model, device)

    # Set up data module
    print("📂 Setting up data loaders...")
    data_module = TumorSegmentationDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        augmentation=False,  # No augmentation for evaluation
    )

    # Setup datasets
    data_module.setup()
    data_module.setup_evaluation_set()

    # Get dataloaders
    val_dataloader = data_module.val_dataloader()
    eval_dataloader = data_module.eval_dataloader()

    print(f"✅ Validation set: {len(val_dataloader.dataset)} images")
    if eval_dataloader:
        print(f"✅ Evaluation set: {len(eval_dataloader.dataset)} images")
    else:
        print("⚠️  No evaluation set found")

    # Run evaluations
    print("\n📊 Running validation set evaluation...")
    val_summary = evaluator.evaluate_validation_set(val_dataloader)

    if eval_dataloader is not None:
        print("🔍 Running evaluation set analysis...")
        eval_summary = evaluator.evaluate_evaluation_set(eval_dataloader)
    else:
        # Create empty summary if no evaluation set
        eval_summary = {
            "total_images": 0,
            "images_with_tumors": 0,
            "images_without_tumors": 0,
            "tumor_detection_rate": 0,
            "mean_clusters_per_image": 0,
            "mean_clusters_when_tumor_detected": 0,
            "mean_tumor_pixels": 0,
            "std_tumor_pixels": 0,
            "resolution_stats": {
                "mean_width": 0,
                "mean_height": 0,
                "std_width": 0,
                "std_height": 0,
                "unique_resolutions": 0,
                "most_common_resolution": (0, 0),
            },
        }

    # Generate report
    print("📄 Generating comprehensive report...")
    evaluator.generate_report(val_summary, eval_summary, save_dir)

    print(f"\n🎉 Evaluation complete! Results saved to: {save_dir}")
    print("\n📋 Summary of what was analyzed:")
    print("  ✅ Validation set: Dice, IoU, Precision, Recall, F1-Score")
    print("  ✅ Evaluation set: Tumor detection, cluster analysis, resolution stats")
    print("  ✅ Visualizations: Metric distributions and analysis plots")
    print("  ✅ Detailed CSV files: Per-image results for further analysis")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure 'data/' directory exists with the correct structure")
        print("2. Check that evaluation_set/ folder contains PNG images")
        print("3. Ensure all dependencies are installed")
        print("\nFor detailed usage, see EVALUATION_README.md")
