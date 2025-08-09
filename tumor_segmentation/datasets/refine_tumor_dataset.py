from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RefineTumorSliceDataset(Dataset):
    """Dataset for unpaired refinement using control-side rough tumors.

    Expected directories under ``data_root``:

        controls/imgs/*.png                  # original healthy controls
        controls/rough_labels/*.png          # rough binary masks for controls (control-side)
        controls/rough_tumors/*.png          # control images with rough tumors inpainted

        patients/imgs/patient_XXX.png        # patient images
        patients/labels/segmentation_XXX.png # ground truth patient masks

    Note: patient masks are never used with control images. They are only used
    for the discriminator 'real' path.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        transform: Optional[callable] = None,
        mask_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        auto_pad: bool = True,
        control_rough_labels_dir: str = "controls/rough_labels",
        control_rough_tumors_dir: str = "controls/rough_tumors",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.transform = transform  # applied to rough-control (image domain transforms)
        self.mask_transform = mask_transform or transform
        self.target_transform = target_transform or transform
        self.auto_pad = auto_pad
        # Use configurable paths for control directories
        self.controls_dir = self.data_root / "controls" / "imgs"
        self.controls_rough_masks_dir = self.data_root / control_rough_labels_dir
        self.controls_rough_tumors_dir = self.data_root / control_rough_tumors_dir

        # Gather patient image and mask pairs
        patient_imgs_dir = self.data_root / "patients" / "imgs"
        patient_labels_dir_std = self.data_root / "patients" / "labels"
        patient_labels_dir_typo = self.data_root / "patiengs" / "labels"
        patient_labels_dir = patient_labels_dir_std if patient_labels_dir_std.exists() else patient_labels_dir_typo
        self.patient_imgs: List[Path] = sorted(patient_imgs_dir.glob("patient_*.png"))

        self.patient_pairs: List[Tuple[Path, Path]] = []
        for img_path in self.patient_imgs:
            stem = img_path.stem  # e.g., patient_123
            idx = stem.split("_")[-1]
            seg_path = patient_labels_dir / f"segmentation_{idx}.png"
            if not seg_path.exists():
                raise RuntimeError(
                    f"Missing mask for {img_path.name}: expected segmentation_{idx}.png"
                )
            self.patient_pairs.append((img_path, seg_path))

        if not self.patient_pairs:
            raise RuntimeError("No patient image/mask pairs found.")

        # Build control-side maps by index from filenames 'control_XXX.png'
        def index_from_stem(stem: str) -> Optional[str]:
            # expected patterns: control_XXX or any trailing digits
            parts = stem.split("_")
            for p in reversed(parts):
                if p.isdigit():
                    return p
            digits = "".join(ch for ch in stem if ch.isdigit())
            return digits if digits else None

        control_images = sorted(self.controls_dir.glob("*.png"))
        control_rough_masks = sorted(self.controls_rough_masks_dir.glob("*.png")) if self.controls_rough_masks_dir.exists() else []
        control_rough_tumors = sorted(self.controls_rough_tumors_dir.glob("*.png")) if self.controls_rough_tumors_dir.exists() else []

        self.control_img_map: Dict[str, Path] = {}
        for p in control_images:
            idx = index_from_stem(p.stem)
            if idx:
                self.control_img_map[idx] = p

        self.control_mask_map: Dict[str, Path] = {}
        for p in control_rough_masks:
            idx = index_from_stem(p.stem)
            if idx:
                self.control_mask_map[idx] = p

        self.control_rough_map: Dict[str, Path] = {}
        for p in control_rough_tumors:
            idx = index_from_stem(p.stem)
            if idx:
                self.control_rough_map[idx] = p

        # Use only indices available in all required control maps (image, rough mask, rough tumor)
        common_indices = set(self.control_img_map.keys()) & set(self.control_mask_map.keys()) & set(self.control_rough_map.keys())
        if not common_indices:
            print(f"DEBUG: Control images found: {len(self.control_img_map)} in {self.controls_dir}")
            print(f"DEBUG: Control masks found: {len(self.control_mask_map)} in {self.controls_rough_masks_dir}")
            print(f"DEBUG: Control rough tumors found: {len(self.control_rough_map)} in {self.controls_rough_tumors_dir}")
            print(f"DEBUG: Control image indices: {sorted(self.control_img_map.keys())[:10]}")
            print(f"DEBUG: Control mask indices: {sorted(self.control_mask_map.keys())[:10]}")
            print(f"DEBUG: Control rough indices: {sorted(self.control_rough_map.keys())[:10]}")
            raise RuntimeError(
                f"No overlapping control indices found across:\n"
                f"  - {self.controls_dir} ({len(self.control_img_map)} files)\n"
                f"  - {self.controls_rough_masks_dir} ({len(self.control_mask_map)} files)\n"
                f"  - {self.controls_rough_tumors_dir} ({len(self.control_rough_map)} files)"
            )
        self.control_indices: List[str] = sorted(common_indices)

        # Auto-detect optimal padding dimensions common to all images
        if self.auto_pad:
            self.target_w, self.target_h = self._compute_optimal_padding()
        else:
            self.target_w = self.target_h = None

    def __len__(self) -> int:
        return len(self.patient_pairs)

    def _compute_optimal_padding(self) -> Tuple[int, int]:
        """Scan images to determine a common canvas size (multiple of 32)."""
        all_paths = []
        all_paths.extend([p for p, _ in self.patient_pairs])
        all_paths.extend([m for _, m in self.patient_pairs])
        all_paths.extend([self.control_img_map[i] for i in self.control_indices])
        all_paths.extend([self.control_mask_map[i] for i in self.control_indices])
        all_paths.extend([self.control_rough_map[i] for i in self.control_indices])

        max_w = max_h = 0
        for path in all_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            except Exception:
                continue

        target_w = ((max_w + 31) // 32) * 32
        target_h = ((max_h + 31) // 32) * 32
        print(f"[RefineDataset] Auto padding to {target_w}×{target_h} (max {max_w}×{max_h})")
        return target_w, target_h

    def _load_image(self, path: Path, pad_value: int) -> torch.Tensor:
        """Load single-channel image and pad to target canvas.

        pad_value = 255 for image-like inputs, 0 for masks.
        """
        img = Image.open(path).convert("L")
        if self.auto_pad and self.target_w and self.target_h:
            w, h = img.size
            pad_left = (self.target_w - w) // 2
            pad_top = 0
            canvas = Image.new("L", (self.target_w, self.target_h), pad_value)
            canvas.paste(img, (pad_left, pad_top))
            img = canvas
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, index: int):
        # Control side (by control index)
        ctrl_idx = self.control_indices[index % len(self.control_indices)]
        control_img_path = self.control_img_map[ctrl_idx]
        control_mask_path = self.control_mask_map[ctrl_idx]
        control_rough_path = self.control_rough_map[ctrl_idx]

        control_img = self._load_image(control_img_path, pad_value=255)
        control_mask = self._load_image(control_mask_path, pad_value=0)
        control_rough = self._load_image(control_rough_path, pad_value=255)

        # Patient side (random unpaired sample for real path)
        patient_img_path, patient_mask_path = self.patient_pairs[np.random.randint(0, len(self.patient_pairs))]
        patient_img = self._load_image(patient_img_path, pad_value=255)
        patient_mask = self._load_image(patient_mask_path, pad_value=0)

        # Apply transforms
        if self.transform:
            control_img = self.transform(control_img)
            control_rough = self.transform(control_rough)
        if self.mask_transform:
            control_mask = self.mask_transform(control_mask)
            patient_mask = self.mask_transform(patient_mask)
        if self.target_transform:
            patient_img = self.target_transform(patient_img)

        # Generator input: rough control + control mask
        gen_input = torch.cat([control_rough, control_mask], dim=0)

        return {
            # Generator/composite on control domain
            "gen_input": gen_input,                 # (2, H, W)
            "control_image": control_img,           # (1, H, W)
            "control_mask": control_mask,           # (1, H, W)
            "rough_control": control_rough,         # (1, H, W)

            # Discriminator real path (patient domain)
            "patient_image": patient_img,           # (1, H, W)
            "patient_mask": patient_mask,           # (1, H, W)

            # Paths
            "control_img_path": str(control_img_path),
            "control_mask_path": str(control_mask_path),
            "rough_control_path": str(control_rough_path),
            "patient_path": str(patient_img_path),
            "patient_mask_path": str(patient_mask_path),
        }

