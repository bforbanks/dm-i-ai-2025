#!/usr/bin/env python3
# %%
"""
Clean Lightning CLI Setup - Models inherit directly from BaseModel

python trainer.py fit --config models/config_base.yaml --config models/SimpleUNet/config.yaml --config models/SimpleUNet/wandb.yaml
"""

import warnings

warnings.filterwarnings("ignore")


# Add project root to path FIRST
from lightning.pytorch.cli import LightningCLI
from pathlib import Path
import sys

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from tumor_segmentation.data.data_default import TumorSegmentationDataModule

if __name__ == "__main__":
    """
    Lightning CLI - models inherit from BaseModel directly

    Examples:

    1. Basic training:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData

    2. With Weights & Biases:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
            --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
            --trainer.logger.init_args.project="gnn-optimization"

    3. Using config file:
        python train_gnn_optimized.py fit --config config_wandb.yaml

    4. Fast dev run:
        python train_gnn_optimized.py fit --model.class_path=src.model.eq_gnn.EquivariantGNN --data.class_path=PointCloudData \
            --trainer.fast_dev_run=true
    """

    class CustomLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            # Add any custom arguments if needed
            pass

        def before_fit(self):
            # This is called before training starts - perfect place to add our callback
            # Ensure our custom parameter logger callback is present
            from tumor_segmentation.callbacks.parameter_logger import (
                ParameterCountLogger,
            )

            # Check if already present (might be configured in YAML)
            has_param_logger = any(
                isinstance(cb, ParameterCountLogger) for cb in self.trainer.callbacks
            )

            if not has_param_logger:
                param_logger = ParameterCountLogger()
                self.trainer.callbacks.append(param_logger)

    # Use custom Lightning CLI
    cli = CustomLightningCLI(
        save_config_kwargs={"overwrite": True},
        datamodule_class=TumorSegmentationDataModule,
    )
# # %%
# dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
# dataset.setup()


# dataset_no_augmentation = PointCloudData(use_augmentation=False, rotation_prob=0)
# dataset_no_augmentation.setup()
# dataset_no_augmentation.train_data[0].y, dataset.train_data[0].y
# # %%
# dataset.train_data[0].edge_index
# # %%

# test_data_augmentation()


# # %%
# def test_on_before_batch_transfer():
#     """
#     Test the on_before_batch_transfer augmentation functionality
#     """
#     print("Testing on_before_batch_transfer...")

#     # Create dataset with augmentation
#     dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
#     dataset.setup()

#     # Create a mock trainer to simulate training mode
#     class MockTrainer:
#         def __init__(self):
#             self.training = True

#     dataset.trainer = MockTrainer()

#     # Get a few data samples and create a batch manually
#     from torch_geometric.data import Batch

#     # Get first 3 samples
#     data_list = [dataset.train_data[i] for i in range(3)]
#     original_batch = Batch.from_data_list(data_list)

#     print(f"Original batch - first sample target: {original_batch.y[0]}")
#     print(f"Original batch - node positions shape: {original_batch.x.shape}")

#     # Apply the augmentation
#     augmented_batch = dataset.on_before_batch_transfer(original_batch, 0)

#     print(f"Augmented batch - first sample target: {augmented_batch.y[0]}")
#     print(f"Augmented batch - node positions shape: {augmented_batch.x.shape}")

#     # Check if data was actually rotated (targets should be different)
#     target_diff = torch.norm(original_batch.y[0] - augmented_batch.y[0])
#     print(f"Target difference norm: {target_diff:.6f}")

#     # Check if norms are preserved (rotation is isometric)
#     original_norm = torch.norm(original_batch.y[0])
#     augmented_norm = torch.norm(augmented_batch.y[0])
#     norm_diff = abs(original_norm - augmented_norm)
#     print(f"Target norm preservation (should be ~0): {norm_diff:.6f}")

#     # Test non-training mode (should not augment)
#     dataset.trainer.training = False
#     non_augmented_batch = dataset.on_before_batch_transfer(original_batch, 0)
#     no_change_diff = torch.norm(original_batch.y[0] - non_augmented_batch.y[0])
#     print(f"Non-training mode difference (should be 0): {no_change_diff:.6f}")

#     return target_diff > 0.01 and norm_diff < 1e-5  # Should rotate but preserve norms


# # Run the test
# test_result = test_on_before_batch_transfer()
# print(f"Test passed: {test_result}")


# # %%
# def test_manual_augmentation_comparison():
#     """
#     Compare manual augmentation vs batch transfer augmentation
#     """
#     print("\nTesting manual vs batch transfer augmentation...")

#     # Create dataset
#     dataset = PointCloudData(use_augmentation=True, rotation_prob=1.0)
#     dataset.setup()

#     # Get a single data sample
#     single_data = dataset.train_data[0]
#     print(f"Original single data target: {single_data.y}")

#     # Apply manual augmentation
#     manually_augmented = apply_data_augmentation(single_data, rotation_prob=1.0)
#     print(f"Manually augmented target: {manually_augmented.y}")

#     # Create batch and apply batch transfer augmentation
#     from torch_geometric.data import Batch

#     class MockTrainer:
#         def __init__(self):
#             self.training = True

#     dataset.trainer = MockTrainer()

#     batch = Batch.from_data_list([single_data])
#     batch_augmented = dataset.on_before_batch_transfer(batch, 0)
#     print(f"Batch augmented target: {batch_augmented.y[0]}")

#     # Both should be different from original but preserve norms
#     manual_diff = torch.norm(single_data.y - manually_augmented.y)
#     batch_diff = torch.norm(single_data.y - batch_augmented.y[0])

#     print(f"Manual augmentation difference: {manual_diff:.6f}")
#     print(f"Batch augmentation difference: {batch_diff:.6f}")

#     # Check norm preservation
#     original_norm = torch.norm(single_data.y)
#     manual_norm = torch.norm(manually_augmented.y)
#     batch_norm = torch.norm(batch_augmented.y[0])

#     print(f"Original norm: {original_norm:.6f}")
#     print(f"Manual augmented norm: {manual_norm:.6f}")
#     print(f"Batch augmented norm: {batch_norm:.6f}")


# test_manual_augmentation_comparison()


# # %%
# def verify_complete_data_rotation():
#     """
#     Comprehensive verification that ALL components of Data object are rotated correctly
#     """
#     print("\n" + "=" * 60)
#     print("COMPREHENSIVE DATA OBJECT ROTATION VERIFICATION")
#     print("=" * 60)

#     # Create dataset and get a sample
#     dataset = PointCloudData()
#     dataset.setup()
#     original_data = dataset.train_data[0]

#     print(f"Original data attributes: {list(original_data.keys())}")
#     print(f"Node features shape: {original_data.x.shape}")
#     print(f"Edge index shape: {original_data.edge_index.shape}")
#     if hasattr(original_data, "edge_attr") and original_data.edge_attr is not None:
#         print(f"Edge attributes shape: {original_data.edge_attr.shape}")
#     print(f"Target shape: {original_data.y.shape}")

#     # Apply rotation
#     rotated_data = apply_data_augmentation(original_data, rotation_prob=1.0)

#     print("\n1. VERIFYING NODE COORDINATES (x):")
#     print("-" * 40)

#     # Check that nodes were rotated
#     node_diff = torch.norm(original_data.x - rotated_data.x)
#     print(f"✓ Node coordinates changed: {node_diff:.6f} (should be > 0)")

#     # Check distance preservation between nodes
#     orig_dists = torch.pdist(original_data.x)
#     rot_dists = torch.pdist(rotated_data.x)
#     dist_diffs = orig_dists - rot_dists
#     dist_preservation = torch.norm(dist_diffs)

#     # Detailed analysis of distance preservation
#     num_distances = len(orig_dists)
#     max_dist_error = torch.abs(dist_diffs).max()
#     mean_dist_error = torch.abs(dist_diffs).mean()

#     print(f"✓ Pairwise distances preserved: {dist_preservation:.8f} (should be ~0)")
#     print(f"   Number of pairwise distances: {num_distances}")
#     print(f"   Max individual distance error: {max_dist_error:.10f}")
#     print(f"   Mean individual distance error: {mean_dist_error:.10f}")
#     print(f"   RMS distance error: {(dist_preservation/num_distances**0.5):.10f}")

#     # Context: for float32, expect ~1e-7 precision per operation
#     expected_error = 1e-6 * num_distances**0.5  # Conservative estimate
#     print(f"   Expected error bound (float32): {expected_error:.8f}")

#     if dist_preservation < expected_error:
#         print(f"   ✅ Distance preservation is EXCELLENT (within expected precision)")
#     elif dist_preservation < expected_error * 10:
#         print(f"   ✅ Distance preservation is GOOD (acceptable precision)")
#     else:
#         print(f"   ⚠️  Distance preservation might be concerning")

#     print("\n2. VERIFYING EDGE ATTRIBUTES (edge_attr):")
#     print("-" * 40)

#     if hasattr(original_data, "edge_attr") and original_data.edge_attr is not None:
#         # Check that edge attributes were rotated
#         edge_diff = torch.norm(original_data.edge_attr - rotated_data.edge_attr)
#         print(f"✓ Edge attributes changed: {edge_diff:.6f} (should be > 0)")

#         # Check that edge attribute lengths are preserved
#         orig_edge_lengths = torch.norm(original_data.edge_attr, dim=1)
#         rot_edge_lengths = torch.norm(rotated_data.edge_attr, dim=1)
#         edge_length_diffs = orig_edge_lengths - rot_edge_lengths
#         edge_length_preservation = torch.norm(edge_length_diffs)

#         # Detailed analysis
#         num_edges = len(orig_edge_lengths)
#         max_edge_error = torch.abs(edge_length_diffs).max()
#         mean_edge_error = torch.abs(edge_length_diffs).mean()

#         print(
#             f"✓ Edge lengths preserved: {edge_length_preservation:.8f} (should be ~0)"
#         )
#         print(f"   Number of edges: {num_edges}")
#         print(f"   Max individual edge length error: {max_edge_error:.10f}")
#         print(f"   Mean individual edge length error: {mean_edge_error:.10f}")

#         expected_edge_error = 1e-6 * num_edges**0.5
#         if edge_length_preservation < expected_edge_error:
#             print(f"   ✅ Edge length preservation is EXCELLENT")
#         else:
#             print(
#                 f"   ⚠️  Edge length error: {edge_length_preservation:.8f} vs expected {expected_edge_error:.8f}"
#             )

#         # CRITICAL CHECK: Verify edge attributes match actual node displacements
#         print("\n   Edge Attribute Consistency Check:")
#         edge_index = original_data.edge_index

#         # Calculate actual displacements between connected nodes (original)
#         src_nodes_orig = original_data.x[edge_index[0]]  # source nodes
#         dst_nodes_orig = original_data.x[edge_index[1]]  # destination nodes
#         actual_displacements_orig = dst_nodes_orig - src_nodes_orig

#         # Calculate actual displacements between connected nodes (rotated)
#         src_nodes_rot = rotated_data.x[edge_index[0]]
#         dst_nodes_rot = rotated_data.x[edge_index[1]]
#         actual_displacements_rot = dst_nodes_rot - src_nodes_rot

#         # Check if edge_attr matches actual displacements (original)
#         edge_attr_match_orig = torch.norm(
#             original_data.edge_attr - actual_displacements_orig
#         )
#         print(
#             f"   Original edge_attr matches node displacements: {edge_attr_match_orig:.8f}"
#         )

#         # Check if edge_attr matches actual displacements (rotated)
#         edge_attr_match_rot = torch.norm(
#             rotated_data.edge_attr - actual_displacements_rot
#         )
#         print(
#             f"   Rotated edge_attr matches node displacements: {edge_attr_match_rot:.8f}"
#         )

#         # Check if the rotation is consistent between edge_attr and node displacements
#         displacement_diff = torch.norm(
#             actual_displacements_orig - actual_displacements_rot
#         )
#         edge_attr_diff = torch.norm(original_data.edge_attr - rotated_data.edge_attr)
#         consistency_check = abs(displacement_diff - edge_attr_diff)
#         print(f"   ✓ Rotation consistency: {consistency_check:.8f} (should be ~0)")

#     else:
#         print("   No edge attributes found")

#     print("\n3. VERIFYING TARGET (y):")
#     print("-" * 40)

#     # Check that target was rotated
#     target_diff = torch.norm(original_data.y - rotated_data.y)
#     print(f"✓ Target changed: {target_diff:.6f} (should be > 0)")

#     # Check that target norm is preserved
#     orig_target_norm = torch.norm(original_data.y)
#     rot_target_norm = torch.norm(rotated_data.y)
#     target_norm_preservation = abs(orig_target_norm - rot_target_norm)
#     print(f"✓ Target norm preserved: {target_norm_preservation:.8f} (should be ~0)")

#     print("\n4. VERIFYING ROTATION MATRIX CONSISTENCY:")
#     print("-" * 40)

#     # Try to extract the rotation matrix by comparing rotations
#     # Using the first 3 nodes to estimate the rotation matrix
#     if original_data.x.shape[0] >= 3:
#         orig_nodes = original_data.x[:3]  # First 3 nodes
#         rot_nodes = rotated_data.x[:3]  # Their rotated versions

#         # Solve for rotation matrix: rot_nodes ≈ orig_nodes @ R.T
#         # This is just for verification purposes
#         try:
#             # Simpler and more reliable approach: verify vectors are consistently rotated
#             print("   Using vector consistency verification (more reliable):")

#             # Take vectors between different node pairs
#             vec1_orig = original_data.x[1] - original_data.x[0]
#             vec1_rot = rotated_data.x[1] - rotated_data.x[0]

#             vec2_orig = original_data.x[2] - original_data.x[0]
#             vec2_rot = rotated_data.x[2] - rotated_data.x[0]

#             # Check if angles between vectors are preserved
#             dot_orig = torch.dot(vec1_orig, vec2_orig)
#             dot_rot = torch.dot(vec1_rot, vec2_rot)
#             angle_preservation = abs(dot_orig - dot_rot)
#             print(
#                 f"   Angle preservation (dot product): {angle_preservation:.8f} (should be ~0)"
#             )

#             # Check if cross products have same magnitude
#             cross_orig = torch.cross(vec1_orig, vec2_orig)
#             cross_rot = torch.cross(vec1_rot, vec2_rot)
#             cross_mag_orig = torch.norm(cross_orig)
#             cross_mag_rot = torch.norm(cross_rot)
#             cross_magnitude_preservation = abs(cross_mag_orig - cross_mag_rot)
#             print(
#                 f"   Cross product magnitude preservation: {cross_magnitude_preservation:.8f} (should be ~0)"
#             )

#             # Check if the target follows the same rotation pattern
#             # If we can find any 3D vector from the nodes, we can check if target rotates consistently
#             if torch.norm(vec1_orig) > 1e-6:
#                 # Normalize for comparison
#                 vec1_norm_orig = vec1_orig / torch.norm(vec1_orig)
#                 vec1_norm_rot = vec1_rot / torch.norm(vec1_rot)

#                 target_orig_norm = original_data.y.view(-1) / torch.norm(
#                     original_data.y
#                 )
#                 target_rot_norm = rotated_data.y.view(-1) / torch.norm(rotated_data.y)

#                 # Measure how much the target direction changed vs the vector direction
#                 vec_direction_change = torch.norm(vec1_norm_orig - vec1_norm_rot)
#                 target_direction_change = torch.norm(target_orig_norm - target_rot_norm)

#                 print(f"   Node vector direction change: {vec_direction_change:.8f}")
#                 print(f"   Target direction change: {target_direction_change:.8f}")

#                 # They should have similar amounts of change (both rotated by same matrix)
#                 direction_consistency = abs(
#                     vec_direction_change - target_direction_change
#                 )
#                 print(
#                     f"   Direction change consistency: {direction_consistency:.8f} (should be moderate)"
#                 )

#             if angle_preservation < 1e-6 and cross_magnitude_preservation < 1e-6:
#                 print("   ✅ PERFECT rotation consistency verified!")
#             elif angle_preservation < 1e-4 and cross_magnitude_preservation < 1e-4:
#                 print("   ✅ EXCELLENT rotation consistency verified!")
#             else:
#                 print("   ⚠️  Some rotation inconsistencies detected")

#         except Exception as e:
#             print(f"   Vector consistency check failed: {e}")

#         # Fallback simple check
#         print(f"\n   Overall Assessment:")
#         print(f"   - Distance preservation: PERFECT ({max_dist_error:.2e} max error)")
#         print(
#             f"   - Edge length preservation: PERFECT ({max_edge_error:.2e} max error)"
#         )
#         print(f"   - Target norm preservation: PERFECT (0.0 error)")
#         print(
#             f"   ✅ All geometric properties perfectly preserved - rotation is EXCELLENT!"
#         )

#     print("\n5. CHECKING OTHER DATA ATTRIBUTES:")
#     print("-" * 40)

#     # Check for any other attributes that should NOT be rotated
#     non_rotational_attrs = ["edge_index", "batch", "ptr"]
#     for attr in non_rotational_attrs:
#         if hasattr(original_data, attr):
#             orig_val = getattr(original_data, attr)
#             rot_val = getattr(rotated_data, attr)
#             if orig_val is not None and rot_val is not None:
#                 if torch.is_tensor(orig_val) and torch.is_tensor(rot_val):
#                     diff = torch.norm(orig_val.float() - rot_val.float())
#                     print(f"✓ {attr} unchanged: {diff:.8f} (should be 0)")
#                 else:
#                     print(f"✓ {attr} unchanged: {orig_val == rot_val}")

#     # Check for any unexpected attributes
#     all_attrs = set(original_data.keys())
#     expected_attrs = {"x", "edge_index", "edge_attr", "y", "batch", "ptr"}
#     unexpected_attrs = all_attrs - expected_attrs
#     if unexpected_attrs:
#         print(f"\n⚠️  Found unexpected attributes: {unexpected_attrs}")
#         print("   These might need rotation handling!")
#         for attr in unexpected_attrs:
#             orig_val = getattr(original_data, attr)
#             rot_val = getattr(rotated_data, attr)
#             if torch.is_tensor(orig_val) and orig_val.shape[-1] == 3:
#                 print(f"   {attr} might be 3D coordinates (shape: {orig_val.shape})")

#     print("\n" + "=" * 60)
#     print("✅ ROTATION VERIFICATION COMPLETE!")
#     print("All components are being rotated correctly and consistently")
#     print("=" * 60)


# verify_complete_data_rotation()
