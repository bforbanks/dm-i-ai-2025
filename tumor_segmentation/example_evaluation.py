#!/usr/bin/env python3
"""
Example script showing how to use the evaluation functionality.

This demonstrates:
1. Loading a trained model
2. Running evaluation on validation and evaluation sets
3. Generating comprehensive statistics and visualizations
"""

from pathlib import Path
import sys

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from pathlib import Path
from data.data_default import TumorSegmentationDataModule
from evaluate_model import ModelEvaluator, load_model


def example_evaluation():
    """Example of how to run model evaluation"""

    # Configuration
    data_dir = "data"
    batch_size = 8
    image_size = 256

    print("🚀 Starting Tumor Segmentation Model Evaluation")
    print("-" * 50)

    # Load model (simple initialization like api.py)
    model = load_model()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create evaluator
    evaluator = ModelEvaluator(model, device)

    # Set up data module
    print("Setting up data loaders...")
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

    print(f"Validation set: {len(val_dataloader.dataset)} images")
    if eval_dataloader:
        print(f"Evaluation set: {len(eval_dataloader.dataset)} images")
    else:
        print("No evaluation set found")

    # Run evaluation on validation set (with ground truth)
    print("\n📊 Evaluating on validation set...")
    val_summary = evaluator.evaluate_validation_set(val_dataloader)

    # Run evaluation on evaluation set (no ground truth - prediction analysis)
    if eval_dataloader:
        print("\n🔍 Analyzing predictions on evaluation set...")
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

    # Generate comprehensive report
    print("\n📄 Generating evaluation report...")
    save_dir = "evaluation_results_example"
    evaluator.generate_report(val_summary, eval_summary, save_dir)

    print(f"\n✅ Evaluation complete! Results saved to: {save_dir}")
    print("\nGenerated files:")
    print("  - validation_detailed_results.csv")
    print("  - evaluation_detailed_results.csv")
    print("  - validation_metrics_distribution.png")
    print("  - evaluation_analysis.png")


def quick_stats_example():
    """Quick example of the statistics that will be generated"""

    print("\n📈 EXAMPLE STATISTICS THAT WILL BE GENERATED:")
    print("=" * 60)

    print("\n📊 VALIDATION SET (with ground truth):")
    print("  - Dice Score: 0.845 ± 0.123")
    print("  - IoU Score: 0.731 ± 0.156")
    print("  - Precision: 0.823")
    print("  - Recall: 0.867")
    print("  - F1-Score: 0.844")

    print("\n🔍 EVALUATION SET (prediction analysis):")
    print("  - Total images: 100")
    print("  - Images with tumor predictions: 73 (73.0%)")
    print("  - Images without tumor predictions: 27")
    print("  - Average tumor clusters per image: 1.2")
    print("  - Average clusters when tumor detected: 1.6")
    print("  - Average tumor pixels per image: 423.5 ± 289.1")

    print("\n📐 IMAGE RESOLUTION STATISTICS:")
    print("  - Mean resolution: 512×512")
    print("  - Resolution std: 128×64")
    print("  - Unique resolutions: 8")
    print("  - Most common resolution: 512×512")

    print("\n📊 VISUALIZATIONS GENERATED:")
    print("  - Histogram of Dice scores")
    print("  - Histogram of IoU scores")
    print("  - Tumor detection pie chart")
    print("  - Cluster count distribution")
    print("  - Image resolution scatter plot")
    print("  - Tumor pixel count histogram")


if __name__ == "__main__":
    # Show example statistics first
    quick_stats_example()

    # Run actual evaluation
    try:
        example_evaluation()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        print("\nThis is likely because:")
        print("1. Data directory structure is different")
        print("2. Missing dependencies")
        print("3. Model architecture issues")
        print("\nTo run evaluation:")
        print("python evaluate_model.py --data_dir data/")
