import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SyntheticTumorSliceDataset(Dataset):
    """PyTorch `Dataset` yielding (control_slice, mask, tumour_slice).

    The folder structure is assumed to be

    root/
        controls/imgs/*.png            # healthy slices
        patients/imgs/*.png            # patient slices containing tumours
        patients/labels/segmentation_*.png  # corresponding binary masks

    Each patient slice *n* has image file name pattern ``patient_{n:03d}.png``
    and its mask is ``segmentation_{n:03d}.png``.  Control slices follow
    ``control_{n:03d}.png`` or similar with unique indices.

    On every `__getitem__`, a random control slice is paired with the given
    patient slice + tumour mask.  This lets the GAN learn to paint a tumour on
    arbitrary healthy anatomy guided solely by the mask.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        transform: Optional[callable] = None,
        control_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        auto_pad: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.transform = transform
        self.control_transform = control_transform or transform
        self.target_transform = target_transform or transform
        self.auto_pad = auto_pad

        # gather file lists
        self.control_imgs: List[Path] = sorted(
            (self.data_root / "controls" / "imgs").glob("*.png")
        )
        if not self.control_imgs:
            raise RuntimeError("No control images found. Expected under controls/imgs/*.png")

        patient_imgs_dir = self.data_root / "patients" / "imgs"
        patient_labels_dir = self.data_root / "patients" / "labels"
        self.patient_imgs: List[Path] = sorted(patient_imgs_dir.glob("patient_*.png"))

        # Build mapping to corresponding segmentation file
        self.patient_pairs: List[Tuple[Path, Path]] = []
        for img_path in self.patient_imgs:
            stem = img_path.stem  # e.g., patient_123
            idx = stem.split("_")[-1]
            seg_name = f"segmentation_{idx}.png"
            seg_path = patient_labels_dir / seg_name
            if not seg_path.exists():
                raise RuntimeError(f"Missing mask for {img_path.name}: expected {seg_name}")
            self.patient_pairs.append((img_path, seg_path))

        if not self.patient_pairs:
            raise RuntimeError("No patient image/mask pairs found.")

        # Auto-detect optimal padding dimensions
        if self.auto_pad:
            self.target_w, self.target_h = self._compute_optimal_padding()
        else:
            self.target_w = self.target_h = None

    def __len__(self) -> int:
        return len(self.patient_pairs)

    def _compute_optimal_padding(self) -> Tuple[int, int]:
        """Scan all images to find optimal padding dimensions."""
        all_paths = list(self.control_imgs) + [p for p, _ in self.patient_pairs] + [m for _, m in self.patient_pairs]
        
        max_w = max_h = 0
        for path in all_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)
            except Exception:
                continue
        
        # Round up to nearest multiple of 32 for clean downsampling
        target_w = ((max_w + 31) // 32) * 32
        target_h = ((max_h + 31) // 32) * 32
        
        print(f"Auto-detected padding: {target_w}×{target_h} (from max {max_w}×{max_h})")
        return target_w, target_h

    def _load_image(self, path: Path, pad_value: int = 255) -> torch.Tensor:
        """Load a grayscale image and pad to the dataset target size.

        Padding policy:
        • Horizontal padding is symmetrical (left & right) so the slice is centred.
        • Vertical padding is applied *only at the bottom* of the canvas.
        • Padding pixels are filled with *white* (255) by default, but a custom
          `pad_value` can be given (e.g. 0 for masks).
        """
        img = Image.open(path).convert("L")

        if self.auto_pad and self.target_w and self.target_h:
            w, h = img.size
            pad_left = (self.target_w - w) // 2  # centre horizontally
            pad_top = 0  # no padding on top, only bottom

            canvas = Image.new("L", (self.target_w, self.target_h), pad_value)
            canvas.paste(img, (pad_left, pad_top))
            img = canvas

        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, index: int):
        patient_img_path, mask_path = self.patient_pairs[index]

        # pick a random control slice each time for diversity
        control_img_path = np.random.choice(self.control_imgs)

        # load tensors with correct padding colours
        control_tensor = self._load_image(control_img_path, pad_value=255)
        mask_tensor = self._load_image(mask_path, pad_value=0)
        patient_tensor = self._load_image(patient_img_path, pad_value=255)

        # apply transforms if any
        if self.control_transform:
            control_tensor = self.control_transform(control_tensor)
        if self.transform:
            mask_tensor = self.transform(mask_tensor)
        if self.target_transform:
            patient_tensor = self.target_transform(patient_tensor)

        # Concatenate control + mask along channel dimension for generator input
        gen_input = torch.cat([control_tensor, mask_tensor], dim=0)  # (2, H, W)

        return {
            "gen_input": gen_input,
            "real_image": patient_tensor,
            "mask": mask_tensor,
            "control": control_tensor,
            "patient_path": str(patient_img_path),
            "control_path": str(control_img_path),
        }
