import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class TiledTumorSegmentationDataset(Dataset):
    """
    Tiled dataset for inverted tumor segmentation.

    Creates horizontal tiles of 128px height from images and focuses on predicting
    non-tumor pixels in dark regions (pixel value < 85).

    Key changes:
    1. Images are split into 128px height tiles
    2. Target is inverted: predict pixels < 85 that are NOT tumors
    3. Controls provide positive examples of non-tumor dark pixels
    4. Each tile becomes its own training sample
    """

    def __init__(
        self,
        patient_image_paths: List[str] = None,
        patient_mask_paths: List[str] = None,
        control_image_paths: List[str] = None,
        tile_height: int = 128,
        intensity_threshold: int = 85,
        transform=None,
        store_in_ram: bool = True,
    ):
        self.patient_image_paths = patient_image_paths or []
        self.patient_mask_paths = patient_mask_paths or []
        self.control_image_paths = control_image_paths or []
        self.tile_height = tile_height
        self.intensity_threshold = intensity_threshold
        self.transform = transform
        self.store_in_ram = store_in_ram

        # Storage for tiles in RAM
        self.tiles_data = {}  # Dict to store all tiles
        self.tile_metadata = []  # List of metadata for each tile

        print(f"Creating tiled dataset with {tile_height}px tiles...")
        print(f"Targeting non-tumor pixels below intensity {intensity_threshold}")

        self._create_tiles()

    def _create_tiles(self):
        """Create tiles from all images and store in RAM"""
        tile_idx = 0

        # Process patient images
        for i, (img_path, mask_path) in enumerate(
            zip(self.patient_image_paths, self.patient_mask_paths)
        ):
            print(
                f"Processing patient {i + 1}/{len(self.patient_image_paths)}: {Path(img_path).name}"
            )

            # Load image and mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Warning: Could not load {img_path} or {mask_path}")
                continue

            if image.shape != mask.shape:
                print(f"Warning: Shape mismatch for {img_path}")
                continue

            # Create tiles from this image
            tiles = self._create_tiles_from_image(
                image, mask, is_control=False, source_idx=i
            )

            for tile_data in tiles:
                self.tiles_data[tile_idx] = tile_data
                self.tile_metadata.append(
                    {
                        "tile_idx": tile_idx,
                        "source_type": "patient",
                        "source_idx": i,
                        "source_file": Path(img_path).name,
                        "tile_y": tile_data["tile_y"],
                        "has_tumor_pixels": tile_data["target"].sum()
                        < (
                            tile_data["target"].size - tile_data["target"].sum()
                        ),  # Some pixels are NOT target (i.e., tumor pixels exist)
                    }
                )
                tile_idx += 1

        # Process control images
        for i, img_path in enumerate(self.control_image_paths):
            print(
                f"Processing control {i + 1}/{len(self.control_image_paths)}: {Path(img_path).name}"
            )

            # Load image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Could not load {img_path}")
                continue

            # Create zero mask for controls (no tumors)
            mask = np.zeros_like(image)

            # Create tiles from this image
            tiles = self._create_tiles_from_image(
                image, mask, is_control=True, source_idx=i
            )

            for tile_data in tiles:
                self.tiles_data[tile_idx] = tile_data
                self.tile_metadata.append(
                    {
                        "tile_idx": tile_idx,
                        "source_type": "control",
                        "source_idx": i,
                        "source_file": Path(img_path).name,
                        "tile_y": tile_data["tile_y"],
                        "has_tumor_pixels": False,  # Controls never have tumors
                    }
                )
                tile_idx += 1

        print(f"Created {len(self.tiles_data)} tiles total:")
        print(
            f"  - Patient tiles: {sum(1 for m in self.tile_metadata if m['source_type'] == 'patient')}"
        )
        print(
            f"  - Control tiles: {sum(1 for m in self.tile_metadata if m['source_type'] == 'control')}"
        )
        print(
            f"  - Tiles with tumor pixels: {sum(1 for m in self.tile_metadata if m['has_tumor_pixels'])}"
        )

    def _create_tiles_from_image(
        self, image: np.ndarray, mask: np.ndarray, is_control: bool, source_idx: int
    ) -> List[Dict]:
        """Create horizontal tiles from a single image"""
        height, width = image.shape
        tiles = []

        y = 0
        while y < height:
            # Calculate tile boundaries
            tile_start_y = y
            tile_end_y = min(y + self.tile_height, height)
            actual_tile_height = tile_end_y - tile_start_y

            # Extract tile
            tile_image = image[tile_start_y:tile_end_y, :]
            tile_mask = mask[tile_start_y:tile_end_y, :]

            # Pad with white if needed (white = 255 for grayscale)
            if actual_tile_height < self.tile_height:
                padding_height = self.tile_height - actual_tile_height

                # Pad image with white
                tile_image = np.pad(
                    tile_image,
                    ((0, padding_height), (0, 0)),
                    mode="constant",
                    constant_values=255,
                )

                # Pad mask with zeros (no tumor/no target in padding)
                tile_mask = np.pad(
                    tile_mask,
                    ((0, padding_height), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # Create inverted target: predict non-tumor pixels in dark regions
            target = self._create_inverted_target(tile_image, tile_mask)

            tiles.append(
                {
                    "image": tile_image.copy(),
                    "target": target.copy(),
                    "original_mask": tile_mask.copy(),
                    "tile_y": tile_start_y,
                    "is_control": is_control,
                    "source_idx": source_idx,
                }
            )

            y += self.tile_height

        return tiles

    def _create_inverted_target(
        self, image: np.ndarray, tumor_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create inverted segmentation target.

        Target = 1 for pixels that are:
        - Below intensity threshold (dark pixels)
        - AND not tumor pixels

        This teaches the model to identify non-tumor dark regions.
        """
        # Dark pixels (below threshold)
        dark_pixels = image < self.intensity_threshold

        # Tumor pixels
        tumor_pixels = tumor_mask > 0

        # Target: dark pixels that are NOT tumors
        target = dark_pixels & (~tumor_pixels)

        return target.astype(np.uint8)

    def __len__(self):
        return len(self.tiles_data)

    def __getitem__(self, idx):
        tile_data = self.tiles_data[idx]

        image = tile_data["image"].copy()
        target = tile_data["target"].copy()

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

        # Convert target to float
        target = (
            target.float() if torch.is_tensor(target) else target.astype(np.float32)
        )

        # Ensure target has correct shape [C, H, W] for PyTorch
        if torch.is_tensor(target):
            if len(target.shape) == 2:  # [H, W]
                target = target.unsqueeze(0)  # [1, H, W]
        else:
            if len(target.shape) == 2:  # [H, W]
                target = target[np.newaxis, ...]  # [1, H, W]

        return image, target

    def get_tile_metadata(self, idx):
        """Get metadata for a specific tile"""
        return self.tile_metadata[idx]

    def get_tiles_by_source(self, source_type: str, source_idx: int) -> List[int]:
        """Get all tile indices from a specific source image"""
        return [
            i
            for i, meta in enumerate(self.tile_metadata)
            if meta["source_type"] == source_type and meta["source_idx"] == source_idx
        ]


class TiledTumorDataModule(pl.LightningDataModule):
    """Lightning DataModule for tiled tumor segmentation"""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = None,  # Not used for tiled approach
        val_split: float = 0.2,
        random_state: int = 42,
        augmentation: bool = True,
        tile_height: int = 128,
        intensity_threshold: int = 85,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_state = random_state
        self.augmentation = augmentation
        self.tile_height = tile_height
        self.intensity_threshold = intensity_threshold

        # Datasets
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets"""
        print(f"Loading tiled data from {self.data_dir}...")

        # Load data paths
        patient_images, patient_masks, control_images = self._load_data_paths()

        print(
            f"Found {len(patient_images)} patient images and {len(control_images)} control images"
        )

        # Split data at the image level (not tile level)
        train_patient_imgs, val_patient_imgs, train_patient_masks, val_patient_masks = (
            train_test_split(
                patient_images,
                patient_masks,
                test_size=self.val_split,
                random_state=self.random_state,
            )
        )

        train_control_imgs, val_control_imgs = train_test_split(
            control_images, test_size=self.val_split, random_state=self.random_state
        )

        # Create tiled datasets
        print("Creating training tiles...")
        self.train_dataset = TiledTumorSegmentationDataset(
            patient_image_paths=train_patient_imgs,
            patient_mask_paths=train_patient_masks,
            control_image_paths=train_control_imgs,
            tile_height=self.tile_height,
            intensity_threshold=self.intensity_threshold,
            transform=self._get_transforms(augmentation=self.augmentation),
        )

        print("Creating validation tiles...")
        self.val_dataset = TiledTumorSegmentationDataset(
            patient_image_paths=val_patient_imgs,
            patient_mask_paths=val_patient_masks,
            control_image_paths=val_control_imgs,
            tile_height=self.tile_height,
            intensity_threshold=self.intensity_threshold,
            transform=self._get_transforms(augmentation=False),
        )

        print(f"Training tiles: {len(self.train_dataset)}")
        print(f"Validation tiles: {len(self.val_dataset)}")

    def _load_data_paths(self) -> Tuple[List[str], List[str], List[str]]:
        """Load paths to images and masks"""
        # Patient images and masks
        patient_img_dir = self.data_dir / "patients" / "imgs"
        patient_mask_dir = self.data_dir / "patients" / "labels"

        patient_images = sorted(list(patient_img_dir.glob("*.png")))
        patient_masks = []

        for img_path in patient_images:
            # Find corresponding mask
            img_name = img_path.stem
            mask_name = img_name.replace("patient_", "segmentation_") + ".png"
            mask_path = patient_mask_dir / mask_name

            if mask_path.exists():
                patient_masks.append(str(mask_path))
            else:
                patient_masks.append(None)

        # Control images (no tumors)
        control_img_dir = self.data_dir / "controls" / "imgs"
        control_images = sorted(list(control_img_dir.glob("*.png")))

        return (
            [
                str(p)
                for p in patient_images
                if patient_masks[patient_images.index(p)] is not None
            ],
            [m for m in patient_masks if m is not None],
            [str(p) for p in control_images],
        )

    def _get_transforms(self, augmentation: bool = True):
        """Get transforms for tiled data"""
        if augmentation:
            return A.Compose(
                [
                    # Light augmentations only (tiles are already small)
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.5
                    ),
                    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
                    ToTensorV2(),
                ]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def visualize_tiled_dataset(
    dataset: TiledTumorSegmentationDataset,
    num_tiles: int = 10,
    save_path: str = "tiled_visualization.png",
):
    """
    Visualize tiled dataset with side-by-side comparison.

    Left column: Original tiles
    Right column: Tiles with green overlay for target (non-tumor dark pixels) and red for tumors
    """
    print(f"Creating visualization of {num_tiles} tiles...")

    # Select diverse tiles for visualization
    tile_indices = []

    # Try to get a mix of patient and control tiles
    patient_tiles = [
        i
        for i, meta in enumerate(dataset.tile_metadata)
        if meta["source_type"] == "patient"
    ]
    control_tiles = [
        i
        for i, meta in enumerate(dataset.tile_metadata)
        if meta["source_type"] == "control"
    ]

    # Get half patient, half control tiles
    n_patient = min(num_tiles // 2, len(patient_tiles))
    n_control = min(num_tiles - n_patient, len(control_tiles))

    import random

    random.seed(42)
    tile_indices.extend(random.sample(patient_tiles, n_patient))
    tile_indices.extend(random.sample(control_tiles, n_control))

    # If we still need more, add whatever we can
    remaining = num_tiles - len(tile_indices)
    if remaining > 0:
        all_remaining = [i for i in range(len(dataset)) if i not in tile_indices]
        tile_indices.extend(
            random.sample(all_remaining, min(remaining, len(all_remaining)))
        )

    tile_indices = tile_indices[:num_tiles]

    # Create figure
    fig, axes = plt.subplots(num_tiles, 2, figsize=(12, 2 * num_tiles))
    fig.suptitle(
        "Tiled Dataset Visualization\nLeft: Original | Right: Target Overlay (Green=Non-tumor dark, Red=Tumor)",
        fontsize=14,
    )

    spacing = 0.02  # Spacing between tiles

    for i, tile_idx in enumerate(tile_indices):
        tile_data = dataset.tiles_data[tile_idx]
        metadata = dataset.tile_metadata[tile_idx]

        # Get original data (before transforms)
        image = tile_data["image"]
        target = tile_data["target"]
        tumor_mask = tile_data["original_mask"]

        # Left column: Original image
        axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(
            f"Tile {i + 1}: {metadata['source_type']} - {metadata['source_file']}"
        )
        axes[i, 0].axis("off")

        # Right column: Image with overlays
        axes[i, 1].imshow(image, cmap="gray", vmin=0, vmax=255)

        # Green overlay for target pixels (non-tumor dark pixels)
        target_overlay = np.zeros((*target.shape, 4))  # RGBA
        target_mask_bool = target > 0
        target_overlay[target_mask_bool] = [0, 1, 0, 0.6]  # Green with alpha

        # Red overlay for tumor pixels
        tumor_overlay = np.zeros((*tumor_mask.shape, 4))  # RGBA
        tumor_mask_bool = tumor_mask > 0
        tumor_overlay[tumor_mask_bool] = [1, 0, 0, 0.6]  # Red with alpha

        # Apply overlays
        if target_mask_bool.any():
            axes[i, 1].imshow(target_overlay, alpha=0.4)
        if tumor_mask_bool.any():
            axes[i, 1].imshow(tumor_overlay, alpha=0.6)

        axes[i, 1].set_title(f"Overlay - Y: {metadata['tile_y']}")
        axes[i, 1].axis("off")

        # Add spacing
        if i < num_tiles - 1:
            fig.subplots_adjust(hspace=spacing)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="green", alpha=0.4, label="Target: Non-tumor dark pixels (< 85)"
        ),
        Patch(facecolor="red", alpha=0.6, label="Tumor pixels (not targeted)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {save_path}")

    # Print statistics
    print("\nVisualization Statistics:")
    print(f"Total tiles shown: {len(tile_indices)}")
    print(
        f"Patient tiles: {sum(1 for i in tile_indices if dataset.tile_metadata[i]['source_type'] == 'patient')}"
    )
    print(
        f"Control tiles: {sum(1 for i in tile_indices if dataset.tile_metadata[i]['source_type'] == 'control')}"
    )
    print(
        f"Tiles with target pixels: {sum(1 for i in tile_indices if dataset.tiles_data[i]['target'].sum() > 0)}"
    )
    print(
        f"Tiles with tumor pixels: {sum(1 for i in tile_indices if dataset.tiles_data[i]['original_mask'].sum() > 0)}"
    )


def create_demo_tiled_dataset(
    data_dir: str = "data", tile_height: int = 128, intensity_threshold: int = 85
):
    """
    Create a demo tiled dataset and visualize it.
    """
    print("🚀 Creating Demo Tiled Dataset")
    print("=" * 50)

    # Load data paths
    data_path = Path(data_dir)
    patient_img_dir = data_path / "patients" / "imgs"
    patient_mask_dir = data_path / "patients" / "labels"
    control_img_dir = data_path / "controls" / "imgs"

    # Get a few sample files
    patient_images = sorted(list(patient_img_dir.glob("*.png")))[:3]  # First 3 patients
    control_images = sorted(list(control_img_dir.glob("*.png")))[:2]  # First 2 controls

    patient_masks = []
    for img_path in patient_images:
        img_name = img_path.stem
        mask_name = img_name.replace("patient_", "segmentation_") + ".png"
        mask_path = patient_mask_dir / mask_name
        if mask_path.exists():
            patient_masks.append(str(mask_path))

    print(
        f"Using {len(patient_images)} patient images and {len(control_images)} control images"
    )

    # Create tiled dataset
    dataset = TiledTumorSegmentationDataset(
        patient_image_paths=[str(p) for p in patient_images],
        patient_mask_paths=patient_masks,
        control_image_paths=[str(p) for p in control_images],
        tile_height=tile_height,
        intensity_threshold=intensity_threshold,
        transform=None,  # No transforms for demo
    )

    # Create visualization
    visualize_tiled_dataset(
        dataset, num_tiles=10, save_path="demo_tiled_visualization.png"
    )

    return dataset


if __name__ == "__main__":
    # Create demo
    demo_dataset = create_demo_tiled_dataset()
    print(f"\n✅ Demo complete! Created {len(demo_dataset)} tiles total.")
    print("📊 Check 'demo_tiled_visualization.png' for the visualization")
