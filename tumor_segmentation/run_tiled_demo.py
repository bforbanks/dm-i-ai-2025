#!/usr/bin/env python3
"""
Demo script for tiled tumor segmentation dataset.

This demonstrates the new inverted segmentation approach using horizontal tiles.
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tiled_data import create_demo_tiled_dataset, TiledTumorDataModule


def main():
    """Run tiled dataset demo"""
    print("🎯 Tiled Tumor Segmentation Demo")
    print("=" * 50)
    print("This demo shows the new inverted segmentation approach:")
    print("✅ Images split into 128px horizontal tiles")
    print("✅ Target: Non-tumor pixels below intensity 85")
    print("✅ Controls provide positive examples of organ pixels")
    print("✅ Better use of available data")
    print("")

    try:
        # Create demo dataset
        demo_dataset = create_demo_tiled_dataset(
            data_dir="data", tile_height=128, intensity_threshold=85
        )

        print(f"\n🎉 SUCCESS!")
        print(f"📊 Created {len(demo_dataset)} tiles from sample images")
        print(f"📈 Visualization saved as 'demo_tiled_visualization.png'")

        # Show some statistics
        patient_tiles = sum(
            1 for meta in demo_dataset.tile_metadata if meta["source_type"] == "patient"
        )
        control_tiles = sum(
            1 for meta in demo_dataset.tile_metadata if meta["source_type"] == "control"
        )
        tiles_with_targets = sum(
            1
            for tile_data in demo_dataset.tiles_data.values()
            if tile_data["target"].sum() > 0
        )

        print(f"\n📊 Dataset Statistics:")
        print(f"  Patient tiles: {patient_tiles}")
        print(f"  Control tiles: {control_tiles}")
        print(f"  Tiles with target pixels: {tiles_with_targets}")
        print(
            f"  Average target pixels per tile: {sum(tile_data['target'].sum() for tile_data in demo_dataset.tiles_data.values()) / len(demo_dataset):.1f}"
        )

        print(f"\n🧠 What the visualization shows:")
        print(f"  Left column: Original image tiles")
        print(f"  Right column: Same tiles with overlays")
        print(f"    - Green areas: Target (non-tumor dark pixels < 85)")
        print(f"    - Red areas: Tumor pixels (not targeted)")
        print(f"    - Model learns to predict green areas!")

        # Test full data module
        print(f"\n🔧 Testing full data module...")
        data_module = TiledTumorDataModule(
            data_dir="data",
            batch_size=4,
            tile_height=128,
            intensity_threshold=85,
        )

        try:
            data_module.setup()
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()

            print(f"✅ Data module works!")
            print(f"  Training batches: {len(train_loader)}")
            print(f"  Validation batches: {len(val_loader)}")

            # Test one batch
            for batch_images, batch_targets in train_loader:
                print(
                    f"  Batch shape: images {batch_images.shape}, targets {batch_targets.shape}"
                )
                print(f"  Target pixels in batch: {batch_targets.sum().item():.0f}")
                break

        except Exception as e:
            print(f"⚠️ Data module test failed: {e}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure 'data/patients/imgs/' and 'data/patients/labels/' exist")
        print("2. Make sure 'data/controls/imgs/' exists")
        print("3. Check that image and mask files have matching names")
        print("   (patient_001.png → segmentation_001.png)")


if __name__ == "__main__":
    main()
