import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TumorSegmentationDataset(Dataset):
    """
    Dataset for tumor segmentation with support for both patient and control data.

    Patient data: Images with tumor masks
    Control data: Images without tumors (all zeros masks)
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        control_paths: List[str] = None,
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.control_paths = control_paths or []
        self.transform = transform

        # Combine patient and control data
        self.all_images = image_paths + self.control_paths
        self.all_masks = mask_paths + [None] * len(
            self.control_paths
        )  # Controls have no tumors

        print(
            f"Dataset initialized with {len(image_paths)} patient images and {len(self.control_paths)} control images"
        )

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.all_images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.all_masks[idx]
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Binary mask
        else:
            # Control image - no tumor
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert mask to float (already binary 0/1, no need to divide by 255)
        mask = mask.float() if torch.is_tensor(mask) else mask.astype(np.float32)

        # Ensure mask has correct shape [C, H, W] for PyTorch
        if torch.is_tensor(mask):
            if len(mask.shape) == 2:  # [H, W]
                mask = mask.unsqueeze(0)  # [1, H, W]
        else:
            if len(mask.shape) == 2:  # [H, W]
                mask = mask[np.newaxis, ...]  # [1, H, W]

        return image, mask


def get_transforms(is_training: bool = True, image_size: int = 256):
    """
    Get data augmentation transforms.

    Args:
        is_training: Whether to apply training augmentations
        image_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    if is_training:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.5),
                        A.GaussianBlur(p=0.5),
                    ],
                    p=0.2,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                # Convert to float32 and normalize to [0-1] range for neural networks
                # This prevents uint8/float32 mismatch errors
                A.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )


class TumorSegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for tumor segmentation.

    Handles:
    - Loading patient and control data
    - Train/validation split
    - Data augmentation
    - Dataloader creation
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 256,
        val_split: float = 0.2,
        random_state: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.random_state = random_state

        # Data paths
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Load and split data"""
        print(f"Loading data from {self.data_dir}...")

        # Load data paths
        patient_images, patient_masks, control_images = self._load_data_paths()

        print(
            f"Found {len(patient_images)} patient images and {len(control_images)} control images"
        )

        # Create balanced train/validation split
        train_imgs, val_imgs, train_masks, val_masks, train_controls, val_controls = (
            self._create_balanced_split(patient_images, patient_masks, control_images)
        )

        # Create datasets
        self.train_dataset = TumorSegmentationDataset(
            train_imgs,
            train_masks,
            train_controls,
            transform=get_transforms(is_training=True, image_size=self.image_size),
        )

        self.val_dataset = TumorSegmentationDataset(
            val_imgs,
            val_masks,
            val_controls,
            transform=get_transforms(is_training=False, image_size=self.image_size),
        )

        print(f"Train set: {len(self.train_dataset)} samples")
        print(f"Validation set: {len(self.val_dataset)} samples")

    def _create_balanced_split(self, patient_images, patient_masks, control_images):
        """
        Create a balanced train/validation split ensuring equal distribution of patient/control samples.

        Strategy:
        1. Balance the overall dataset first by limiting the larger class
        2. Then split both train and validation sets to maintain balance
        3. Ensure both sets have approximately 50/50 patient/control distribution
        """
        available_patients = len(patient_images)
        available_controls = len(control_images)

        print(
            f"Original data: {available_patients} patients, {available_controls} controls"
        )

        # Balance the dataset by limiting to the smaller class size
        # This prevents class imbalance in both training and validation
        balanced_size = min(available_patients, available_controls)

        print(f"Balancing dataset to {balanced_size} samples per class to prevent bias")

        # Randomly sample to balance classes
        if available_patients > balanced_size:
            # Too many patients, sample subset
            patient_indices = np.random.RandomState(self.random_state).choice(
                available_patients, balanced_size, replace=False
            )
            balanced_patient_images = [patient_images[i] for i in patient_indices]
            balanced_patient_masks = [patient_masks[i] for i in patient_indices]
        else:
            balanced_patient_images = patient_images
            balanced_patient_masks = patient_masks

        if available_controls > balanced_size:
            # Too many controls, sample subset
            control_indices = np.random.RandomState(self.random_state).choice(
                available_controls, balanced_size, replace=False
            )
            balanced_control_images = [control_images[i] for i in control_indices]
        else:
            balanced_control_images = control_images

        # Now split the balanced data
        # Split patients
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            balanced_patient_images,
            balanced_patient_masks,
            test_size=self.val_split,
            random_state=self.random_state,
        )

        # Split controls
        train_controls, val_controls = train_test_split(
            balanced_control_images,
            test_size=self.val_split,
            random_state=self.random_state,
        )

        # Print final distribution
        total_train = len(train_imgs) + len(train_controls)
        total_val = len(val_imgs) + len(val_controls)

        print(
            f"Final Training split: {len(train_imgs)} patients, {len(train_controls)} controls"
        )
        print(
            f"Training distribution: {len(train_imgs) / total_train * 100:.1f}% patients, {len(train_controls) / total_train * 100:.1f}% controls"
        )
        print(
            f"Final Validation split: {len(val_imgs)} patients, {len(val_controls)} controls"
        )
        print(
            f"Validation distribution: {len(val_imgs) / total_val * 100:.1f}% patients, {len(val_controls) / total_val * 100:.1f}% controls"
        )

        return (
            train_imgs,
            val_imgs,
            train_masks,
            val_masks,
            train_controls,
            val_controls,
        )

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

    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
