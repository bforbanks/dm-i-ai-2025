#!/usr/bin/env python3
"""Training script for SPADE GAN with full-image masked inpainting."""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_full_inpaint_module import SPADEFullInpaintGANModule


def main():
    parser = argparse.ArgumentParser(description="Train SPADE GAN with full-image inpainting")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--g_lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=50.0)
    parser.add_argument("--lambda_fft", type=float, default=2.0)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--blend_border_px", type=int, default=6, help="Feather width (pixels) to softly blend tumour edges during inference/visualisation")
    args = parser.parse_args()

    module = SPADEFullInpaintGANModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_g=args.g_lr,
        lr_d=args.d_lr,
        lambda_l1=args.lambda_l1,
        lambda_fft=args.lambda_fft,
        g_steps=args.g_steps,
        blend_border_px=args.blend_border_px,
    )

    logger = CSVLogger("logs", name="spade_gan_full_inpaint")
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=25,
        val_check_interval=0.25,  # Check validation every 25% of epoch
    )

    trainer.fit(module)


if __name__ == "__main__":
    main() 