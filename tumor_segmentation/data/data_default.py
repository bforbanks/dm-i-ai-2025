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
        # Load image as grayscale for single-channel input
        img_path = self.all_images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

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


class OversamplingTumorSegmentationDataset(Dataset):
    """
    Dataset for tumor segmentation with oversampling to achieve desired patient/control ratio.
    
    This dataset uses ALL available data and oversamples the minority class to achieve
    the desired patient_control_ratio. It always uses all data from one class and oversamples
    the other class as needed.
    """

    def __init__(
        self,
        patient_image_paths: List[str],
        patient_mask_paths: List[str],
        control_image_paths: List[str],
        patient_control_ratio: float = 0.8,
        transform=None,
        random_state: int = 42,
    ):
        self.patient_image_paths = patient_image_paths
        self.patient_mask_paths = patient_mask_paths
        self.control_image_paths = control_image_paths
        self.patient_control_ratio = patient_control_ratio
        self.transform = transform
        self.random_state = random_state
        
        num_patients = len(patient_image_paths)
        num_controls = len(control_image_paths)
        
        # Determine which class to oversample based on the desired ratio
        if patient_control_ratio >= 0.5:
            # We want more patients than controls, so oversample patients
            # Use ALL control data and oversample patients
            target_patients = int(num_controls * patient_control_ratio / (1 - patient_control_ratio))
            
            if target_patients > num_patients:
                # Need to oversample patients
                self.patient_indices = []
                
                # Add all original patient indices
                self.patient_indices.extend(range(num_patients))
                
                # Add oversampled indices
                rng = np.random.RandomState(random_state)
                additional_samples = target_patients - num_patients
                oversampled_indices = rng.choice(num_patients, size=additional_samples, replace=True)
                self.patient_indices.extend(oversampled_indices)
            else:
                # No oversampling needed, use all patients
                self.patient_indices = list(range(num_patients))
            
            # Use all controls
            self.control_indices = list(range(num_controls))
            self.oversample_patients = True
            
        else:
            # We want more controls than patients, so oversample controls
            # Use ALL patient data and oversample controls
            target_controls = int(num_patients * (1 - patient_control_ratio) / patient_control_ratio)
            
            if target_controls > num_controls:
                # Need to oversample controls
                self.control_indices = []
                
                # Add all original control indices
                self.control_indices.extend(range(num_controls))
                
                # Add oversampled indices
                rng = np.random.RandomState(random_state)
                additional_samples = target_controls - num_controls
                oversampled_indices = rng.choice(num_controls, size=additional_samples, replace=True)
                self.control_indices.extend(oversampled_indices)
            else:
                # No oversampling needed, use all controls
                self.control_indices = list(range(num_controls))
            
            # Use all patients
            self.patient_indices = list(range(num_patients))
            self.oversample_patients = False
        
        # Shuffle the indices for randomness
        rng = np.random.RandomState(random_state)
        rng.shuffle(self.patient_indices)
        rng.shuffle(self.control_indices)
        
        print(f"Oversampling dataset initialized:")
        print(f"  - Original patients: {num_patients}")
        print(f"  - Original controls: {num_controls}")
        if self.oversample_patients:
            print(f"  - Target patients for {patient_control_ratio:.1%} ratio: {len(self.patient_indices)}")
            print(f"  - Final ratio: {len(self.patient_indices) / (len(self.patient_indices) + len(self.control_indices)):.1%} patients")
        else:
            print(f"  - Target controls for {patient_control_ratio:.1%} ratio: {len(self.control_indices)}")
            print(f"  - Final ratio: {len(self.patient_indices) / (len(self.patient_indices) + len(self.control_indices)):.1%} patients")
    
    def regenerate_oversampling(self, epoch: int = 0):
        """
        Regenerate oversampling indices for a new epoch to ensure different patterns.
        This helps with training stability and prevents overfitting to specific oversampling patterns.
        """
        num_patients = len(self.patient_image_paths)
        num_controls = len(self.control_image_paths)
        
        # Use epoch number as additional seed for randomness
        seed = self.random_state + epoch
        rng = np.random.RandomState(seed)
        
        if self.oversample_patients:
            # Regenerate patient oversampling
            target_patients = int(num_controls * self.patient_control_ratio / (1 - self.patient_control_ratio))
            
            if target_patients > num_patients:
                self.patient_indices = []
                self.patient_indices.extend(range(num_patients))
                
                additional_samples = target_patients - num_patients
                oversampled_indices = rng.choice(num_patients, size=additional_samples, replace=True)
                self.patient_indices.extend(oversampled_indices)
            else:
                self.patient_indices = list(range(num_patients))
            
            # Shuffle patient indices
            rng.shuffle(self.patient_indices)
        else:
            # Regenerate control oversampling
            target_controls = int(num_patients * (1 - self.patient_control_ratio) / self.patient_control_ratio)
            
            if target_controls > num_controls:
                self.control_indices = []
                self.control_indices.extend(range(num_controls))
                
                additional_samples = target_controls - num_controls
                oversampled_indices = rng.choice(num_controls, size=additional_samples, replace=True)
                self.control_indices.extend(oversampled_indices)
            else:
                self.control_indices = list(range(num_controls))
            
            # Shuffle control indices
            rng.shuffle(self.control_indices)

    def __len__(self):
        return len(self.patient_indices) + len(self.control_indices)

    def __getitem__(self, idx):
        # Determine if this is a patient or control sample
        if idx < len(self.patient_indices):
            # Patient sample
            patient_idx = self.patient_indices[idx]
            img_path = self.patient_image_paths[patient_idx]
            mask_path = self.patient_mask_paths[patient_idx]
            
            # Load image and mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # Binary mask
        else:
            # Control sample
            control_idx = idx - len(self.patient_indices)
            control_idx = self.control_indices[control_idx]  # Map to actual control index
            img_path = self.control_image_paths[control_idx]
            mask_path = None
            
            # Load image and create zero mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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


class OversamplingDataLoader(DataLoader):
    """
    Custom DataLoader that regenerates oversampling indices at the beginning of each epoch.
    This ensures different oversampling patterns across epochs for better training stability.
    """
    
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.epoch = 0
    
    def __iter__(self):
        # Regenerate oversampling indices for the new epoch
        if isinstance(self.dataset, OversamplingTumorSegmentationDataset):
            self.dataset.regenerate_oversampling(self.epoch)
        self.epoch += 1
        return super().__iter__()


def get_augmentations_transforms(image_size: int = 256):
    """
    Get comprehensive training augmentations for tumor segmentation.

    These augmentations help the model become more robust to:
    - Lighting variations (brightness, contrast, gamma)
    - Noise and blur
    - Geometric variations (rotation, scaling, shifting)
    - Grid distortion and elastic deformation
    - Small occlusions (cutout)

    All transforms are applied consistently to both image and mask.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            # Brightness and contrast variations (common in medical imaging)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15, contrast_limit=0.15, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                ],
                p=0.8,
            ),
            # Very subtle blur only (minimal for medical images)
            A.GaussianBlur(blur_limit=(3, 3), p=0.8),
            # Conservative geometric transforms
            A.ShiftScaleRotate(
                shift_limit=0.03,  # Reduced: 3% shift
                scale_limit=0.05,  # Reduced: 5% zoom
                rotate_limit=3,  # Reduced: max ±3 degrees
                border_mode=cv2.BORDER_REFLECT,  # reflect border for natural padding
                p=0.8,
            ),
            # Remove the aggressive distortions that cause weird artifacts
            # GridDistortion and ElasticTransform removed for medical image quality
            # Convert to float32 and normalize to [0-1] range for neural networks
            # Single channel (grayscale) normalization
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_standard_transforms(image_size: int = 256):
    """
    Get validation transforms (no augmentation, only preprocessing).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            # Convert to float32 and normalize to [0-1] range for neural networks
            # Single channel (grayscale) normalization
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_transforms(augmentation: bool = True, image_size: int = 256):
    """
    Get data augmentation transforms.

    Args:
        is_training: Whether to apply training augmentations
        image_size: Target image size

    Returns:
        Albumentations transform pipeline
    """
    if augmentation:
        return get_augmentations_transforms(image_size)
    else:
        return get_standard_transforms(image_size)


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
        augmentation: bool = True,
        patient_control_ratio: float = 0.5,  # 0.5 = 50/50, 0.7 = 70% patients, 0.3 = 30% controls
        use_oversampling: bool = True,  # Toggle between oversampling and previous dataloading
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.random_state = random_state
        self.augmentation = augmentation
        self.patient_control_ratio = patient_control_ratio
        self.use_oversampling = use_oversampling
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

        if self.use_oversampling:
            # Use oversampling approach
            print("Using OVERSAMPLING approach (uses all data)")
            train_imgs, val_imgs, train_masks, val_masks, train_controls, val_controls = (
                self._create_oversampled_split(patient_images, patient_masks, control_images)
            )

            # Create oversampling datasets for training
            self.train_dataset = OversamplingTumorSegmentationDataset(
                train_imgs,
                train_masks,
                train_controls,
                patient_control_ratio=self.patient_control_ratio,
                transform=get_transforms(
                    augmentation=self.augmentation, image_size=self.image_size
                ),
                random_state=self.random_state,
            )

            # Use regular dataset for validation (no oversampling needed)
            self.val_dataset = TumorSegmentationDataset(
                val_imgs,
                val_masks,
                val_controls,
                transform=get_transforms(augmentation=False, image_size=self.image_size),
            )

            print(f"Train set: {len(self.train_dataset)} samples (with oversampling)")
            print(f"Validation set: {len(self.val_dataset)} samples")
        else:
            # Use previous balanced approach
            print("Using PREVIOUS BALANCED approach (may waste data)")
            train_imgs, val_imgs, train_masks, val_masks, train_controls, val_controls = (
                self._create_balanced_split(patient_images, patient_masks, control_images)
            )

            # Create regular datasets for both training and validation
            self.train_dataset = TumorSegmentationDataset(
                train_imgs,
                train_masks,
                train_controls,
                transform=get_transforms(
                    augmentation=self.augmentation, image_size=self.image_size
                ),
            )

            self.val_dataset = TumorSegmentationDataset(
                val_imgs,
                val_masks,
                val_controls,
                transform=get_transforms(augmentation=False, image_size=self.image_size),
            )

            print(f"Train set: {len(self.train_dataset)} samples (balanced)")
            print(f"Validation set: {len(self.val_dataset)} samples")

    def _create_oversampled_split(self, patient_images, patient_masks, control_images):
        """
        Create a train/validation split that uses ALL data and applies oversampling during training.
        
        Strategy:
        1. Use ALL available data (no undersampling)
        2. Split both patient and control data into train/validation
        3. During training, the OversamplingTumorSegmentationDataset will handle oversampling
        4. Validation uses the regular split without oversampling
        """
        available_patients = len(patient_images)
        available_controls = len(control_images)

        print(
            f"Original data: {available_patients} patients, {available_controls} controls"
        )
        print(f"Using ALL data with oversampling to achieve {self.patient_control_ratio:.1%} patient ratio")

        # Split patients into train/validation
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            patient_images,
            patient_masks,
            test_size=self.val_split,
            random_state=self.random_state,
        )

        # Split controls into train/validation
        train_controls, val_controls = train_test_split(
            control_images,
            test_size=self.val_split,
            random_state=self.random_state,
        )

        # Print final distribution
        total_train = len(train_imgs) + len(train_controls)
        total_val = len(val_imgs) + len(val_controls)

        print(
            f"Training split: {len(train_imgs)} patients, {len(train_controls)} controls"
        )
        print(
            f"Training distribution (before oversampling): {len(train_imgs) / total_train * 100:.1f}% patients, {len(train_controls) / total_train * 100:.1f}% controls"
        )
        print(
            f"Validation split: {len(val_imgs)} patients, {len(val_controls)} controls"
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

    def _create_balanced_split(self, patient_images, patient_masks, control_images):
        """
        Create a balanced train/validation split ensuring equal distribution of patient/control samples.
        This is the previous approach that may waste data through undersampling.

        Strategy:
        1. Balance the overall dataset first by limiting the larger class
        2. Then split both train and validation sets to maintain balance
        3. Ensure both sets have approximately the desired patient/control distribution
        """
        available_patients = len(patient_images)
        available_controls = len(control_images)

        print(
            f"Original data: {available_patients} patients, {available_controls} controls"
        )

        # Calculate target sizes based on the desired ratio
        total_target_size = min(
            available_patients / self.patient_control_ratio,
            available_controls / (1 - self.patient_control_ratio),
        )
        target_patients = int(total_target_size * self.patient_control_ratio)
        target_controls = int(total_target_size * (1 - self.patient_control_ratio))

        print(
            f"Target distribution: {target_patients} patients ({self.patient_control_ratio * 100:.1f}%), {target_controls} controls ({(1 - self.patient_control_ratio) * 100:.1f}%)"
        )

        # Randomly sample to achieve target distribution
        if available_patients > target_patients:
            # Too many patients, sample subset
            patient_indices = np.random.RandomState(self.random_state).choice(
                available_patients, target_patients, replace=False
            )
            balanced_patient_images = [patient_images[i] for i in patient_indices]
            balanced_patient_masks = [patient_masks[i] for i in patient_indices]
        else:
            balanced_patient_images = patient_images
            balanced_patient_masks = patient_masks

        if available_controls > target_controls:
            # Too many controls, sample subset
            control_indices = np.random.RandomState(self.random_state).choice(
                available_controls, target_controls, replace=False
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
        """Training dataloader with optional epoch-based oversampling regeneration"""
        if self.use_oversampling and isinstance(self.train_dataset, OversamplingTumorSegmentationDataset):
            return OversamplingDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
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
