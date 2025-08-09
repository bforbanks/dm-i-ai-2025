import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import scipy.ndimage as ndi


def compute_entropy(softmax: np.ndarray) -> np.ndarray:
    """Compute voxel-wise entropy from softmax probabilities.
    softmax: (C,H,W)
    Returns (H,W) float32"""
    entropy = -np.sum(softmax * np.log(softmax + 1e-8), axis=0)
    return entropy.astype(np.float32)


def compute_signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform: positive outside, negative inside the mask."""
    dist_out = ndi.distance_transform_edt(mask == 0)
    dist_in = ndi.distance_transform_edt(mask == 1)
    return (dist_out - dist_in).astype(np.float32)


def compute_y_coord(shape):
    H, W = shape
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    return np.broadcast_to(y, (H, W)).copy()


class RefinerDataset(Dataset):
    """Dataset for training the Refiner U-Net.

    Each *case* lives in a directory that must at least contain:
        ├── petmr.npy           # original input image (C,H,W) or (H,W) - variable height 300-1000px
        ├── softmax.npz         # nnUNet soft-max logits; stored under key "softmax" (C,H,W)
        ├── entropy.npy         # pre-computed entropy map (H,W)
        ├── distance.npy        # signed distance transform (H,W)
        ├── y_coord.npy         # normalised y-coordinate map (H,W)
        └── mask.png            # ground-truth binary mask 0/255 (H,W) (any format cv2 can read)

    The model handles variable height images (300-1000px) by padding to make dimensions
    divisible by 16. No resizing is performed - padding is handled in the model forward pass.

    Parameters
    ----------
    root_dir : str | Path
        Directory that contains one sub-folder per patient.
    cases : Optional[List[str]]
        If given, only these case sub-folders are used; otherwise all folders in *root_dir*.
    modalities_first : bool
        Whether original modalities should be placed before or after the auxiliary channels.
    transform : callable, optional
        Albumentations/PyTorch style transform that operates on image + mask (expects keys "image", "mask").
    """

    def __init__(
        self,
        root_dir: str | Path,
        cases: Optional[List[str]] = None,
        modalities_first: bool = True,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        if cases is None:
            # assume every directory is a case
            self.cases = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        else:
            self.cases = cases
        self.modalities_first = modalities_first
        self.transform = transform

    # ---------------------------------------------------------
    def __len__(self):
        return len(self.cases)

    # ---------------------------------------------------------
    def _load_petmr(self, case_dir: Path) -> np.ndarray:
        pet_file = case_dir / "petmr.npy"
        if not pet_file.exists():
            raise FileNotFoundError(pet_file)
        arr = np.load(pet_file)  # (H,W) or (C,H,W)
        if arr.ndim == 2:
            arr = arr[None, :, :]  # add channel dim
        return arr.astype(np.float32)

    def _load_softmax(self, case_dir: Path) -> np.ndarray:
        sm_file = case_dir / "softmax.npz"
        if not sm_file.exists():
            raise FileNotFoundError(sm_file)
        probs = np.load(sm_file)["softmax"].astype(np.float32)
        return probs  # (C,H,W)

    # ---------------------------------------------------------
    def __getitem__(self, idx):
        case_name = self.cases[idx]
        case_dir = self.root_dir / case_name

        # Load mandatory channels
        petmr = self._load_petmr(case_dir)  # (M,H,W)
        softmax = self._load_softmax(case_dir)  # (C,H,W)
        entropy = np.load(case_dir / "entropy.npy").astype(np.float32)
        distance = np.load(case_dir / "distance.npy").astype(np.float32)
        y_coord = np.load(case_dir / "y_coord.npy").astype(np.float32)

        # Stack channels
        extra = np.stack([entropy, distance, y_coord], axis=0)  # (3,H,W)

        if self.modalities_first:
            img = np.concatenate([softmax, petmr, extra], axis=0)
        else:
            img = np.concatenate([petmr, softmax, extra], axis=0)

        # Ground truth mask
        mask_path = case_dir / "mask.png"
        if mask_path.suffix.lower() in {".npy", ".npz"}:
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = mask[0]  # assume first channel
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # binary 0/1

        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=img.transpose(1, 2, 0), mask=mask)
            img = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()  # Albumentations returns H,W,C -> C,H,W
            mask = torch.from_numpy(augmented["mask"]).unsqueeze(0).float()
        else:
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
