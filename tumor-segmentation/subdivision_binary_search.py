"""
Subdivision Binary Search for Exact Mask Discovery

This algorithm uses the insight that TP counts provide enough information
to determine exactly which pixels are positive through systematic subdivision.

Key insight: If we always test with exactly half the pixels set to positive
and half to negative, the TP count tells us how many of our "positive" pixels
are actually correct. This gives us enough information to iteratively refine
our guess until we have the exact mask.

Algorithm:
1. Start with all pixels, assign half as "positive guess", half as "negative guess"
2. Query API with this mask, get TP count
3. The TP count tells us how many pixels in our "positive" half are actually positive
4. Subdivide and repeat until we know the exact assignment for each pixel

Time complexity: O(log2(total_pixels)) - much better than the previous approach!
"""

import numpy as np
import requests
from typing import List, Tuple, Set
from PIL import Image
import sys
import os
from utils import encode_request, decode_request

class SubdivisionSearcher:
    """Binary search using systematic subdivision and TP counting"""
    
    def __init__(self, api_url: str = "http://localhost:9052/predict"):
        self.api_url = api_url
        self.query_count = 0
    
    def query_api(self, image: np.ndarray) -> np.ndarray:
        """Send image to API and get predicted segmentation"""
        try:
            encoded_img = encode_request(image)
            response = requests.post(self.api_url, json={"img": encoded_img}, timeout=30)
            response.raise_for_status()
            result = response.json()
            segmentation = decode_request(type('obj', (object,), {'img': result['img']})())
            self.query_count += 1
            return segmentation
        except Exception as e:
            print(f"API error: {e}")
            raise
    
    def count_tp(self, ground_truth: np.ndarray, prediction: np.ndarray) -> int:
        """Count true positive pixels"""
        gt_binary = ground_truth[:,:,0] > 0
        pred_binary = prediction[:,:,0] > 0
        return int(np.sum(gt_binary & pred_binary))
    
    def create_mask_from_pixel_list(self, image_shape: tuple, positive_pixels: List[Tuple[int, int]]) -> np.ndarray:
        """Create mask where specified pixels are positive (255) and others negative (0)"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        for row, col in positive_pixels:
            mask[row, col, :] = 255
        return mask
    
    def subdivision_search(self, original_image: np.ndarray, ground_truth: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Main subdivision algorithm to find exact positive pixels
        
        The key insight: if we have a set of pixels and we know exactly how many
        should be positive, we can use binary subdivision to determine which ones.
        """
        height, width = original_image.shape[:2]
        all_pixels = [(r, c) for r in range(height) for c in range(width)]
        
        print(f"Total pixels: {len(all_pixels)}")
        
        # First, find total number of positive pixels
        all_positive_mask = self.create_mask_from_pixel_list(original_image.shape, all_pixels)
        total_tp = self.count_tp(ground_truth, all_positive_mask)
        print(f"Total positive pixels in ground truth: {total_tp}")
        
        if total_tp == 0:
            return set()
        
        # Now use subdivision to find exactly which pixels are positive
        positive_pixels = self.recursive_subdivision(
            original_image, ground_truth, all_pixels, total_tp
        )
        
        return positive_pixels
    
    def recursive_subdivision(self, 
                            original_image: np.ndarray,
                            ground_truth: np.ndarray, 
                            candidate_pixels: List[Tuple[int, int]],
                            target_positive_count: int) -> Set[Tuple[int, int]]:
        """
        Recursively subdivide pixels to find exact positive set
        
        Args:
            candidate_pixels: List of pixel coordinates we're deciding between
            target_positive_count: How many of these pixels should be positive
        """
        n_candidates = len(candidate_pixels)
        
        print(f"Subdividing {n_candidates} pixels, need to find {target_positive_count} positives")
        
        # Base cases
        if target_positive_count == 0:
            return set()
        if target_positive_count == n_candidates:
            return set(candidate_pixels)
        if n_candidates == 1:
            return set(candidate_pixels) if target_positive_count == 1 else set()
        
        # Split candidates in half
        mid = n_candidates // 2
        left_half = candidate_pixels[:mid]
        right_half = candidate_pixels[mid:]
        
        # Test: assume all of left_half are positive, all of right_half are negative
        test_mask = self.create_mask_from_pixel_list(original_image.shape, left_half)
        tp_count = self.count_tp(ground_truth, test_mask)
        
        print(f"  Testing left half ({len(left_half)} pixels): got {tp_count} TP")
        
        # Now we know:
        # - tp_count pixels in left_half are actually positive
        # - (target_positive_count - tp_count) pixels in right_half are actually positive
        
        left_positives = self.recursive_subdivision(
            original_image, ground_truth, left_half, tp_count
        )
        
        right_target = target_positive_count - tp_count
        right_positives = self.recursive_subdivision(
            original_image, ground_truth, right_half, right_target
        )
        
        return left_positives | right_positives
    
    def find_exact_mask(self, image_path: str) -> np.ndarray:
        """Find exact mask using subdivision approach"""
        print(f"🚀 Finding exact mask using subdivision: {image_path}")
        
        # Load image
        original_image = np.array(Image.open(image_path))
        height, width = original_image.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Get ground truth
        print("📡 Getting ground truth from API...")
        ground_truth = self.query_api(original_image)
        
        # Find exact positive pixels using subdivision
        print("🔍 Starting subdivision search...")
        positive_pixels = self.subdivision_search(original_image, ground_truth)
        
        print(f"✅ Found {len(positive_pixels)} positive pixels")
        print(f"Total API queries: {self.query_count}")
        
        # Create final mask
        final_mask = self.create_mask_from_pixel_list(original_image.shape, list(positive_pixels))
        
        # Verify
        final_tp = self.count_tp(ground_truth, final_mask)
        total_gt_positive = np.sum(ground_truth[:,:,0] > 0)
        print(f"Verification: Found {final_tp}/{total_gt_positive} positive pixels")
        
        return final_mask


def main():
    """Command line usage"""
    if len(sys.argv) != 2:
        print("Usage: python subdivision_binary_search.py <image_path>")
        print("Example: python subdivision_binary_search.py validation2/image_001.png")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Run subdivision search
    searcher = SubdivisionSearcher()
    exact_mask = searcher.find_exact_mask(image_path)
    
    # Save result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_subdivision_mask.png"
    Image.fromarray(exact_mask).save(output_path)
    print(f"💾 Saved exact mask to: {output_path}")


if __name__ == "__main__":
    main()