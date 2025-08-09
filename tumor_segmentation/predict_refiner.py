#!/usr/bin/env python3
"""Inference helper that chains nnUNet logits with the Refiner U-Net to produce final segmentation masks."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from tumor_segmentation.models.RefinerUNet.model import RefinerUNet
from tumor_segmentation.data.refiner_dataset import compute_entropy, compute_signed_distance, compute_y_coord  # reuse functions


def build_input(petmr_path: Path, softmax_path: Path) -> torch.Tensor:
    softmax = np.load(softmax_path)["softmax"].astype(np.float32)  # (C,H,W)
    petmr = np.load(petmr_path).astype(np.float32)  # (H,W) or (M,H,W)
    if petmr.ndim == 2:
        petmr = petmr[None]

    entropy = compute_entropy(softmax)
    if softmax.shape[0] > 1:
        mask = (softmax[1] > 0.5).astype(np.uint8)
    else:
        mask = (softmax[0] > 0.5).astype(np.uint8)
    distance = compute_signed_distance(mask)
    y_coord = compute_y_coord(entropy.shape)

    extra = np.stack([entropy, distance, y_coord], axis=0)
    inp = np.concatenate([softmax, petmr, extra], axis=0)
    
    # Model will handle padding internally for variable height images
    return torch.from_numpy(inp).unsqueeze(0)  # B=1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--softmax", required=True, type=Path)
    parser.add_argument("--petmr", required=True, type=Path)
    parser.add_argument("--out_mask", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint just to get model args
    ckpt = torch.load(args.checkpoint, map_location=device)
    hparams = ckpt["hyper_parameters"]
    in_channels = hparams.get("in_channels", None)
    num_classes = hparams.get("num_classes", 1)
    model = RefinerUNet(in_channels=in_channels or 8, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)

    inp = build_input(args.petmr, args.softmax).to(device)
    with torch.no_grad():
        pred = model(inp)[0, 0]  # assume single class, remove batch & channel
    mask = (pred.cpu().numpy() > args.threshold).astype(np.uint8) * 255
    args.out_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_mask, mask)
    print(f"Refined mask saved to {args.out_mask}")


if __name__ == "__main__":
    main()
