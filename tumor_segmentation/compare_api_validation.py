#!/usr/bin/env python3
"""
Script to compare API-generated masks with validation masks from nnUNetv2 training.

This script:
1. Loads validation masks from the validation folder (PNG files)
2. Loads corresponding original images from the dataset
3. Runs the images through the API
4. Compares the API-generated masks with the validation masks
5. Reports differences and statistics

Usage:
    python compare_api_validation.py [--model-path MODEL_PATH] [--fold FOLD] [--api-url API_URL]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import requests
import json
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import encode_request, decode_request, dice_score


class ValidationComparator:
    def __init__(self, 
                 model_name: str = "CV-reduced-custom-128x128",
                 fold: int = 2,
                 api_url: str = "http://localhost:9052/predict",
                 dataset_path: str = "data_nnUNet/Dataset001_TumorSegmentation"):
        """
        Initialize the validation comparator.
        
        Args:
            model_name: Name of the model (e.g., "CV-reduced-custom-128x128")
            fold: Fold number to use for validation
            api_url: URL of the API endpoint
            dataset_path: Path to the dataset directory
        """
        # Construct the full model path from the model name
        dataset_root = Path(dataset_path)
        self.results_root = dataset_root.parent / "results" / dataset_root.name
        self.model_path = self.results_root / f"nnUNetTrainer__nnUNetResEncUNetMPlans__{model_name}"
        self.fold = fold
        self.api_url = api_url
        self.dataset_path = Path(dataset_path)
        
        # Construct paths
        self.validation_path = self.model_path / f"fold_{fold}" / "validation"
        self.images_path = self.dataset_path / "imagesTr"
        self.labels_path = self.dataset_path / "labelsTr"
        
        # Validate paths
        if not self.validation_path.exists():
            raise ValueError(f"Validation path does not exist: {self.validation_path}")
        if not self.images_path.exists():
            raise ValueError(f"Images path does not exist: {self.images_path}")
        if not self.labels_path.exists():
            raise ValueError(f"Labels path does not exist: {self.labels_path}")
            
        print(f"✅ Validation path: {self.validation_path}")
        print(f"✅ Images path: {self.images_path}")
        print(f"✅ Labels path: {self.labels_path}")
        print(f"✅ API URL: {self.api_url}")
        
    def get_validation_files(self) -> List[str]:
        """Get list of validation PNG files."""
        png_files = list(self.validation_path.glob("*.png"))
        return [f.stem for f in png_files]  # Return just the patient names
    
    def load_validation_mask(self, patient_name: str) -> np.ndarray:
        """Load validation mask for a patient."""
        mask_path = self.validation_path / f"{patient_name}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Validation mask not found: {mask_path}")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load validation mask: {mask_path}")
        
        return mask
    
    def load_original_image(self, patient_name: str) -> np.ndarray:
        """Load original image for a patient."""
        # The validation files are named like "patient_004.png" but the original images are "patient_004_0000.png"
        image_path = self.images_path / f"{patient_name}_0000.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Original image not found: {image_path}")
        
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load original image: {image_path}")
        
        return image
    
    def call_api(self, image: np.ndarray) -> np.ndarray:
        """Call the API with an image and return the predicted mask."""
        # Encode the image
        encoded_image = encode_request(image)
        
        # Prepare request
        request_data = {
            "img": encoded_image
        }
        
        # Make API call
        try:
            response = requests.post(self.api_url, json=request_data, timeout=60)
            response.raise_for_status()
            
            # Decode response
            response_data = response.json()
            
            # The API returns a TumorPredictResponseDto with an 'img' field
            if 'img' in response_data:
                # Create a simple object with img attribute for decode_request
                class ResponseObject:
                    def __init__(self, img_data):
                        self.img = img_data
                
                response_obj = ResponseObject(response_data['img'])
                predicted_mask = decode_request(response_obj)
            else:
                # Debug: print the actual response structure
                print(f"   ⚠️  Unexpected API response structure: {list(response_data.keys())}")
                print(f"   📄 Response data: {response_data}")
                raise RuntimeError(f"API response missing 'img' field. Available keys: {list(response_data.keys())}")
            
            return predicted_mask
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to process API response: {e}")
    
    def normalize_mask(self, mask: np.ndarray, target_range: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """Normalize mask to target range."""
        if mask.max() == 0:
            return mask
        
        # Convert to float for normalization
        mask_float = mask.astype(np.float32)
        
        if mask.max() > 1:
            # Assume 0-255 range, normalize to 0-1
            mask_float = mask_float / 255.0
        
        if target_range != (0, 1):
            # Scale to target range
            mask_float = mask_float * (target_range[1] - target_range[0]) + target_range[0]
        
        return mask_float
    
    def compare_masks(self, mask1: np.ndarray, mask2: np.ndarray, patient_name: str) -> Dict:
        """Compare two masks and return metrics."""
        # Normalize both masks to 0-1 range for comparison
        mask1_norm = self.normalize_mask(mask1, (0, 1))
        mask2_norm = self.normalize_mask(mask2, (0, 1))
        
        # Ensure both masks are 2D for comparison
        if len(mask1_norm.shape) == 3:
            mask1_norm = mask1_norm[:, :, 0]  # Take first channel
        if len(mask2_norm.shape) == 3:
            mask2_norm = mask2_norm[:, :, 0]  # Take first channel
        
        # Ensure same shape
        if mask1_norm.shape != mask2_norm.shape:
            # Resize mask2 to match mask1
            mask2_norm = cv2.resize(mask2_norm, (mask1_norm.shape[1], mask1_norm.shape[0]))
        
        # Calculate metrics
        dice = dice_score(mask1_norm, mask2_norm)
        
        # Calculate pixel-wise differences
        diff = np.abs(mask1_norm - mask2_norm)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # Calculate binary metrics
        mask1_bin = mask1_norm > 0.5
        mask2_bin = mask2_norm > 0.5
        
        # True positives, false positives, false negatives
        tp = np.sum(mask1_bin & mask2_bin)
        fp = np.sum(~mask1_bin & mask2_bin)
        fn = np.sum(mask1_bin & ~mask2_bin)
        tn = np.sum(~mask1_bin & ~mask2_bin)
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "patient_name": patient_name,
            "dice_score": dice,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "mask1_shape": mask1.shape,
            "mask2_shape": mask2.shape,
            "mask1_range": (mask1.min(), mask1.max()),
            "mask2_range": (mask2.min(), mask2.max())
        }
    
    def save_comparison_plot(self, original_image: np.ndarray, validation_mask: np.ndarray, 
                           api_mask: np.ndarray, comparison_result: Dict, output_dir: Path):
        """Save a comparison plot showing original image, validation mask, and API mask."""
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Validation mask
        axes[1].imshow(validation_mask, cmap='gray')
        axes[1].set_title(f"Validation Mask\nRange: {comparison_result['mask1_range']}")
        axes[1].axis('off')
        
        # API mask
        axes[2].imshow(api_mask, cmap='gray')
        axes[2].set_title(f"API Mask\nRange: {comparison_result['mask2_range']}")
        axes[2].axis('off')
        
        # Difference
        mask1_norm = self.normalize_mask(validation_mask, (0, 1))
        mask2_norm = self.normalize_mask(api_mask, (0, 1))
        
        if len(mask1_norm.shape) == 3:
            mask1_norm = mask1_norm[:, :, 0]
        if len(mask2_norm.shape) == 3:
            mask2_norm = mask2_norm[:, :, 0]
        
        if mask1_norm.shape != mask2_norm.shape:
            mask2_norm = cv2.resize(mask2_norm, (mask1_norm.shape[1], mask1_norm.shape[0]))
        
        diff = np.abs(mask1_norm - mask2_norm)
        axes[3].imshow(diff, cmap='hot')
        axes[3].set_title(f"Difference\nMean: {comparison_result['mean_diff']:.3f}")
        axes[3].axis('off')
        
        # Add metrics as text
        fig.suptitle(f"Patient: {comparison_result['patient_name']}\n"
                    f"Dice: {comparison_result['dice_score']:.3f}, "
                    f"F1: {comparison_result['f1_score']:.3f}", 
                    fontsize=12)
        
        # Save plot
        plot_path = output_dir / f"{comparison_result['patient_name']}_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def run_comparison(self, max_patients: Optional[int] = None, 
                      save_plots: bool = True, 
                      output_dir: str = "comparison_results") -> Dict:
        """Run the full comparison between API and validation masks."""
        print(f"\n🔍 Starting comparison between API and validation masks...")
        
        # Get validation files
        validation_files = self.get_validation_files()
        if max_patients:
            validation_files = validation_files[:max_patients]
        
        print(f"📊 Found {len(validation_files)} validation files to compare")
        
        # Setup output directory
        output_path = Path(output_dir)
        if save_plots:
            output_path.mkdir(exist_ok=True)
        
        # Results storage
        results = []
        failed_patients = []
        
        # Process each patient
        for patient_name in tqdm(validation_files, desc="Comparing patients"):
            try:
                print(f"\n📋 Processing {patient_name}...")
                
                # Load validation mask
                validation_mask = self.load_validation_mask(patient_name)
                print(f"   ✅ Loaded validation mask: {validation_mask.shape}, range: {validation_mask.min()}-{validation_mask.max()}")
                
                # Load original image
                original_image = self.load_original_image(patient_name)
                print(f"   ✅ Loaded original image: {original_image.shape}")
                
                # Call API
                print(f"   🔄 Calling API...")
                api_start_time = time.time()
                api_mask = self.call_api(original_image)
                api_time = time.time() - api_start_time
                print(f"   ✅ API response: {api_mask.shape}, range: {api_mask.min()}-{api_mask.max()}, time: {api_time:.2f}s")
                
                # Compare masks
                comparison_result = self.compare_masks(validation_mask, api_mask, patient_name)
                comparison_result['api_time'] = api_time
                results.append(comparison_result)
                
                print(f"   📊 Comparison results:")
                print(f"      Dice: {comparison_result['dice_score']:.3f}")
                print(f"      F1: {comparison_result['f1_score']:.3f}")
                print(f"      Mean diff: {comparison_result['mean_diff']:.3f}")
                
                # Save comparison plot
                if save_plots:
                    plot_path = self.save_comparison_plot(original_image, validation_mask, api_mask, comparison_result, output_path)
                    print(f"   💾 Saved comparison plot: {plot_path}")
                
            except Exception as e:
                print(f"   ❌ Failed to process {patient_name}: {e}")
                failed_patients.append(patient_name)
                continue
        
        # Calculate summary statistics
        if results:
            summary = self.calculate_summary_statistics(results)
        else:
            summary = {"error": "No successful comparisons"}
        
        # Save results
        results_file = output_path / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_results": results,
                "failed_patients": failed_patients
            }, f, indent=2)
        
        print(f"\n📈 Summary:")
        print(f"   Total patients: {len(validation_files)}")
        print(f"   Successful comparisons: {len(results)}")
        print(f"   Failed comparisons: {len(failed_patients)}")
        
        if results:
            print(f"   Mean Dice Score: {summary['mean_dice']:.3f}")
            print(f"   Mean F1 Score: {summary['mean_f1']:.3f}")
            print(f"   Mean API Time: {summary['mean_api_time']:.2f}s")
        
        print(f"   Results saved to: {results_file}")
        
        return {
            "summary": summary,
            "detailed_results": results,
            "failed_patients": failed_patients
        }
    
    def calculate_summary_statistics(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics from comparison results."""
        dice_scores = [r['dice_score'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        api_times = [r['api_time'] for r in results]
        mean_diffs = [r['mean_diff'] for r in results]
        
        return {
            "mean_dice": np.mean(dice_scores),
            "std_dice": np.std(dice_scores),
            "min_dice": np.min(dice_scores),
            "max_dice": np.max(dice_scores),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "min_f1": np.min(f1_scores),
            "max_f1": np.max(f1_scores),
            "mean_api_time": np.mean(api_times),
            "std_api_time": np.std(api_times),
            "mean_diff": np.mean(mean_diffs),
            "std_diff": np.std(mean_diffs),
            "total_patients": len(results)
        }


def main():
    parser = argparse.ArgumentParser(description="Compare API-generated masks with validation masks")
    parser.add_argument("--model-name", type=str, 
                       default="CV-reduced-custom-128x128",
                       help="Name of the model (e.g., 'CV-reduced-custom-128x128')")
    parser.add_argument("--fold", type=int, default=2, help="Fold number to use for validation")
    parser.add_argument("--api-url", type=str, default="http://localhost:9052/predict",
                       help="URL of the API endpoint")
    parser.add_argument("--dataset-path", type=str, 
                       default="data_nnUNet/Dataset001_TumorSegmentation",
                       help="Path to the dataset directory")
    parser.add_argument("--max-patients", type=int, default=None,
                       help="Maximum number of patients to process (for testing)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip saving comparison plots")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Create comparator
        comparator = ValidationComparator(
            model_name=args.model_name,
            fold=args.fold,
            api_url=args.api_url,
            dataset_path=args.dataset_path
        )
        
        # Run comparison
        results = comparator.run_comparison(
            max_patients=args.max_patients,
            save_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Comparison completed successfully!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 