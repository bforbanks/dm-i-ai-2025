#!/usr/bin/env python3
"""Prepare auxiliary channels (entropy, distance transform, y-coordinate) for the Refiner U-Net.

Example:
    python scripts/prepare_refiner_dataset.py \
        --nnunet_results_dir data_nnUNet/results/Dataset001_TumorSegmentation/.../inference_raw \
        --output_root refiner_data

For every "*_softmax.npz" file found under *nnunet_results_dir* the script will:
    1. Copy / link the softmax file.
    2. Compute entropy.npy, distance.npy, y_coord.npy .
    3. Copy the original PET/MR image (assumed to be *caseid_image.npz* or similar if provided).
    4. Copy the ground-truth mask if available.

You can then train the refiner with train_refiner.py pointing to --data_dir refiner_data.
"""

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi


def compute_entropy(softmax: np.ndarray) -> np.ndarray:
    """softmax shape: (C,H,W)"""
    entropy = -np.sum(softmax * np.log(softmax + 1e-8), axis=0)
    return entropy.astype(np.float32)


def compute_signed_distance(mask: np.ndarray) -> np.ndarray:
    # mask binary (H,W)
    dist_out = ndi.distance_transform_edt(mask == 0)
    dist_in = ndi.distance_transform_edt(mask == 1)
    signed = dist_out - dist_in
    return signed.astype(np.float32)


def compute_y_coord(shape):
    """Compute normalized y-coordinate map for variable height images."""
    H, W = shape
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    return np.broadcast_to(y, (H, W)).copy()


def process_case(softmax_file: Path, output_dir: Path, petmr_file: Path | None, mask_file: Path | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy / link softmax
    target_softmax = output_dir / "softmax.npz"
    if target_softmax.exists():
        target_softmax.unlink()
    shutil.copy(softmax_file, target_softmax)

    softmax = np.load(target_softmax)["softmax"].astype(np.float32)
    entropy = compute_entropy(softmax)
    np.save(output_dir / "entropy.npy", entropy)

    # Compute distance map using class-1 logits
    mask = (softmax[1] > 0.5).astype(np.uint8) if softmax.shape[0] > 1 else (softmax[0] > 0.5).astype(np.uint8)
    distance = compute_signed_distance(mask)
    np.save(output_dir / "distance.npy", distance)

    # y-coordinate (works with variable height images)
    y_coord = compute_y_coord(entropy.shape)
    np.save(output_dir / "y_coord.npy", y_coord)

    # PET/MR copy if provided
    if petmr_file and petmr_file.exists():
        shutil.copy(petmr_file, output_dir / "petmr.npy")

    # mask copy
    if mask_file and mask_file.exists():
        shutil.copy(mask_file, output_dir / mask_file.name)



def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Refiner U-Net")
    parser.add_argument("--nnunet_results_dir", required=True, type=Path, help="Directory with nnUNet softmax .npz files")
    parser.add_argument("--output_root", required=True, type=Path, help="Where to write per-case folders")
    parser.add_argument("--pet_pattern", default="{caseid}_petmr.npy", help="Pattern to locate PET/MR image relative to softmax dir. {caseid} placeholder is replaced.")
    parser.add_argument("--mask_pattern", default="{caseid}.png", help="Pattern to locate ground-truth mask. {caseid} placeholder is replaced.")
    args = parser.parse_args()

    softmax_files = sorted(list(Path(args.nnunet_results_dir).rglob("*_softmax.npz")))
    if not softmax_files:
        raise RuntimeError(f"No *_softmax.npz files found in {args.nnunet_results_dir}")

    print(f"Found {len(softmax_files)} softmax files")

    for sf in softmax_files:
        caseid = sf.stem.replace("_softmax", "")
        out_dir = args.output_root / caseid

        pet_file = sf.parent / args.pet_pattern.format(caseid=caseid)
        mask_file = sf.parent / args.mask_pattern.format(caseid=caseid)
        try:
            process_case(sf, out_dir, pet_file, mask_file)
            print(f"[OK] processed {caseid}")
        except Exception as e:
            print(f"[WARN] failed {caseid}: {e}")


if __name__ == "__main__":
    main()
