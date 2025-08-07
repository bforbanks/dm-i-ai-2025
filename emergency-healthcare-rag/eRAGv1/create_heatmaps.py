#!/usr/bin/env python3
"""
Create heatmaps for BM25 grid search results
Shows accuracy across b and k1 parameters for different chunk size/overlap combinations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path(".cache/grid_search_results.json")
with open(results_file, 'r') as f:
    results = json.load(f)

# Extract unique parameter values
chunk_sizes = sorted(list(set(r['chunk_size'] for r in results)))
chunk_overlaps = sorted(list(set(r['overlap'] for r in results)))
k1_values = sorted(list(set(r['k1'] for r in results)))
b_values = sorted(list(set(r['b'] for r in results)))

print(f"Chunk sizes: {chunk_sizes}")
print(f"Chunk overlaps: {chunk_overlaps}")
print(f"k1 values: {k1_values}")
print(f"b values: {b_values}")

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('BM25 Accuracy Heatmaps: b vs k1 for Different Chunk Configurations', fontsize=16, fontweight='bold')

# Create heatmap for each chunk size/overlap combination
for i, chunk_size in enumerate(chunk_sizes):
    for j, overlap in enumerate(chunk_overlaps):
        ax = axes[i, j]
        
        # Create matrix for this configuration
        heatmap_data = np.zeros((len(b_values), len(k1_values)))
        
        # Fill matrix with accuracy values
        for result in results:
            if result['chunk_size'] == chunk_size and result['overlap'] == overlap:
                b_idx = b_values.index(result['b'])
                k1_idx = k1_values.index(result['k1'])
                heatmap_data[b_idx, k1_idx] = result['accuracy']
        
        # Create heatmap using matplotlib
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=0.95)
        
        # Add text annotations
        for b_idx in range(len(b_values)):
            for k1_idx in range(len(k1_values)):
                text = ax.text(k1_idx, b_idx, f'{heatmap_data[b_idx, k1_idx]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        # Set labels
        ax.set_xticks(range(len(k1_values)))
        ax.set_yticks(range(len(b_values)))
        ax.set_xticklabels(k1_values)
        ax.set_yticklabels(b_values)
        ax.set_xlabel('k1')
        ax.set_ylabel('b')
        ax.set_title(f'Chunk Size: {chunk_size}, Overlap: {overlap}')
        
        # Highlight best configuration
        best_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1, 
                                 fill=False, edgecolor='black', lw=3))

# Add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label('Accuracy', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('.cache/bm25_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# Print best configurations for each chunk size/overlap
print("\n=== BEST CONFIGURATIONS BY CHUNK SIZE/OVERLAP ===")
for chunk_size in chunk_sizes:
    for overlap in chunk_overlaps:
        config_results = [r for r in results if r['chunk_size'] == chunk_size and r['overlap'] == overlap]
        if config_results:
            best = max(config_results, key=lambda x: x['accuracy'])
            print(f"Size {chunk_size}, Overlap {overlap}: k1={best['k1']}, b={best['b']} → {best['accuracy']:.3f}")

# Print overall best
overall_best = max(results, key=lambda x: x['accuracy'])
print(f"\n=== OVERALL BEST ===")
print(f"Size {overall_best['chunk_size']}, Overlap {overall_best['overlap']}: k1={overall_best['k1']}, b={overall_best['b']} → {overall_best['accuracy']:.3f}") 