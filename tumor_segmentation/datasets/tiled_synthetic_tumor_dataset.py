import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TiledSyntheticTumorSliceDataset(Dataset):
    """PyTorch Dataset for tiled tumor inpainting training.
    
    Instead of padding images to a common size, this dataset:
    1. Keeps images at their original resolution
    2. Extracts random tiles of fixed size for training
    3. Only uses tiles that contain tumor pixels (efficiency)
    4. Provides original patient background for inpainting
    
    This allows training on much higher effective resolutions while
    maintaining efficient memory usage.
    """

    def __init__(
        self,
        data_root: str | os.PathLike,
        tile_size: int = 256,
        transform: Optional[callable] = None,
        control_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.tile_size = tile_size
        self.transform = transform
        self.control_transform = control_transform or transform
        self.target_transform = target_transform or transform

        # Gather file lists
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

        # Pre-compute valid tile locations for each patient image
        self.valid_tiles = self._precompute_valid_tiles()
        
        print(f"Tiled dataset initialized:")
        print(f"  - {len(self.patient_pairs)} patient images")
        print(f"  - {len(self.control_imgs)} control images")
        print(f"  - {len(self.valid_tiles)} total tiles (tumor + background)")
        print(f"  - Tile size: {tile_size}x{tile_size}")
        print(f"  - Strategy: Include ALL tumor tiles (any size) + balanced background")

    def _precompute_valid_tiles(self) -> List[Tuple[int, int, int, int, int]]:
        """Pre-compute valid tile locations with balanced tumor representation.
        
        Strategy:
        1. Include ALL tiles that contain ANY tumor pixels (no minimum threshold)
        2. Ensure representation of both large and small tumors
        3. Balance between tumor-containing tiles and background tiles
        
        Returns:
            List of (patient_idx, x, y, width, height) tuples
        """
        valid_tiles = []
        
        for patient_idx, (img_path, mask_path) in enumerate(self.patient_pairs):
            # Load mask to find tumor regions
            mask = np.array(Image.open(mask_path).convert("L"))
            img_h, img_w = mask.shape
            
            # Find all tumor-containing tiles first
            tumor_tiles = []
            background_tiles = []
            
            # Systematic tile extraction to ensure coverage
            for y in range(0, img_h, self.tile_size // 2):  # 50% overlap for better coverage
                for x in range(0, img_w, self.tile_size // 2):
                    # Adjust tile size if near image boundaries
                    tile_w = min(self.tile_size, img_w - x)
                    tile_h = min(self.tile_size, img_h - y)
                    
                    # Extract tile mask
                    tile_mask = mask[y:y+tile_h, x:x+tile_w]
                    tumor_pixels = np.sum(tile_mask > 127)  # Binary threshold
                    
                    if tumor_pixels > 0:
                        # Any tumor pixels - include this tile
                        tumor_tiles.append((patient_idx, x, y, tile_w, tile_h, tumor_pixels))
                    else:
                        # Background tile - store for potential inclusion
                        background_tiles.append((patient_idx, x, y, tile_w, tile_h, 0))
            
            # Balance tile selection
            if tumor_tiles:
                # Sort tumor tiles by tumor size for balanced sampling
                tumor_tiles.sort(key=lambda x: x[5])  # Sort by tumor pixel count
                
                # Include all tumor tiles (both small and large)
                for tile_info in tumor_tiles:
                    valid_tiles.append(tile_info[:5])  # Remove tumor_pixel count
                
                # Add some background tiles for context (but fewer than tumor tiles)
                if background_tiles:
                    num_background = min(len(background_tiles), len(tumor_tiles) // 2)
                    selected_background = random.sample(background_tiles, num_background)
                    for tile_info in selected_background:
                        valid_tiles.append(tile_info[:5])
            else:
                # No tumors in this image - include some background tiles for context
                # Use systematic sampling to ensure good coverage
                num_background = min(len(background_tiles), 8)  # Reasonable default for context
                selected_background = random.sample(background_tiles, num_background)
                for tile_info in selected_background:
                    valid_tiles.append(tile_info[:5])
        
        return valid_tiles

    def __len__(self) -> int:
        return len(self.valid_tiles)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load a grayscale image as numpy array."""
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def _extract_tile(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> torch.Tensor:
        """Extract a tile from the image and convert to tensor.
        
        If the tile is smaller than tile_size (near image boundaries),
        pad with zeros to maintain consistent dimensions.
        """
        tile = image[y:y+h, x:x+w]
        
        # Pad to consistent tile size if needed
        if tile.shape != (self.tile_size, self.tile_size):
            padded_tile = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            padded_tile[:h, :w] = tile
            tile = padded_tile
        
        return torch.from_numpy(tile).unsqueeze(0)  # Add channel dimension

    def __getitem__(self, index: int):
        patient_idx, x, y, tile_w, tile_h = self.valid_tiles[index]
        patient_img_path, mask_path = self.patient_pairs[patient_idx]

        # Pick a random control slice for context
        control_img_path = np.random.choice(self.control_imgs)

        # Load full images
        patient_img = self._load_image(patient_img_path)
        mask_img = self._load_image(mask_path)
        control_img = self._load_image(control_img_path)

        # Extract tiles
        patient_tile = self._extract_tile(patient_img, x, y, tile_w, tile_h)
        mask_tile = self._extract_tile(mask_img, x, y, tile_w, tile_h)
        
        # For control, extract from same relative position (if possible)
        control_h, control_w = control_img.shape
        control_x = min(x, max(0, control_w - tile_w))
        control_y = min(y, max(0, control_h - tile_h))
        control_tile = self._extract_tile(control_img, control_x, control_y, tile_w, tile_h)

        # Apply transforms if any
        if self.control_transform:
            control_tile = self.control_transform(control_tile)
        if self.transform:
            mask_tile = self.transform(mask_tile)
        if self.target_transform:
            patient_tile = self.target_transform(patient_tile)

        # Concatenate control + mask for generator input
        gen_input = torch.cat([control_tile, mask_tile], dim=0)  # (2, H, W)

        return {
            "gen_input": gen_input,
            "real_image": patient_tile,
            "mask": mask_tile,
            "control": control_tile,
            "patient_path": str(patient_img_path),
            "control_path": str(control_img_path),
            "tile_coords": (x, y, tile_w, tile_h),
        }


class TiledInferenceDataset(Dataset):
    """Dataset for inference on full images using tiling."""
    
    def __init__(
        self,
        patient_img_path: Path,
        mask_path: Path,
        control_imgs: List[Path],
        tile_size: int = 256,
        overlap: int = 32,
    ):
        self.patient_img_path = patient_img_path
        self.mask_path = mask_path
        self.control_imgs = control_imgs
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Load images
        self.patient_img = np.array(Image.open(patient_img_path).convert("L"), dtype=np.float32) / 255.0
        self.mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        # Ensure mask matches patient image dimensions
        if self.mask_img.shape != self.patient_img.shape:
            mask_pil = Image.fromarray((self.mask_img * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(self.patient_img.shape[::-1], Image.NEAREST)
            self.mask_img = np.array(mask_pil, dtype=np.float32) / 255.0
        
        # Generate tile coordinates
        self.tile_coords = self._generate_tile_coords()
        
    def _generate_tile_coords(self) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates for full image coverage."""
        coords = []
        img_h, img_w = self.patient_img.shape
        stride = self.tile_size - self.overlap
        
        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                tile_w = min(self.tile_size, img_w - x)
                tile_h = min(self.tile_size, img_h - y)
                coords.append((x, y, tile_w, tile_h))
                
        return coords
    
    def __len__(self) -> int:
        return len(self.tile_coords)
    
    def __getitem__(self, index: int):
        x, y, tile_w, tile_h = self.tile_coords[index]
        
        # Extract patient and mask tiles
        patient_tile = self.patient_img[y:y+tile_h, x:x+tile_w]
        mask_tile = self.mask_img[y:y+tile_h, x:x+tile_w]
        
        # Pad to consistent size if needed
        if patient_tile.shape != (self.tile_size, self.tile_size):
            padded_patient = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            padded_mask = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            padded_patient[:tile_h, :tile_w] = patient_tile
            padded_mask[:tile_h, :tile_w] = mask_tile
            patient_tile = padded_patient
            mask_tile = padded_mask
        
        # Random control image for context
        control_img_path = np.random.choice(self.control_imgs)
        control_img = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
        
        # Extract control tile from same relative position if possible
        control_h, control_w = control_img.shape
        control_x = min(x, max(0, control_w - tile_w))
        control_y = min(y, max(0, control_h - tile_h))
        control_tile = control_img[control_y:control_y+tile_h, control_x:control_x+tile_w]
        
        # Pad control tile if needed
        if control_tile.shape != (self.tile_size, self.tile_size):
            padded_control = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            padded_control[:tile_h, :tile_w] = control_tile
            control_tile = padded_control
        
        # Convert to tensors
        patient_tensor = torch.from_numpy(patient_tile).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_tile).unsqueeze(0)
        control_tensor = torch.from_numpy(control_tile).unsqueeze(0)
        
        gen_input = torch.cat([control_tensor, mask_tensor], dim=0)
        
        return {
            "gen_input": gen_input,
            "real_image": patient_tensor,
            "mask": mask_tensor,
            "control": control_tensor,
            "tile_coords": (x, y, tile_w, tile_h),
        }