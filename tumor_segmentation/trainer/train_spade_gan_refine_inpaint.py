#!/usr/bin/env python3
"""Training script for SPADE GAN that refines rough tumors within masks.

This does not modify any existing training setups.
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_refine_inpaint_module import SPADERefineInpaintGANModule


def main():
    parser = argparse.ArgumentParser(description="Train SPADE GAN (refine rough tumors inpaint)")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--g_lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--lambda_identity", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=10.0)
    parser.add_argument("--lambda_style", type=float, default=5.0)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--control_rough_labels_dir", type=str, default="controls/rough_labels", 
                       help="Directory containing rough control masks")
    parser.add_argument("--control_rough_tumors_dir", type=str, default="controls/rough_tumors",
                       help="Directory containing controls with rough tumors")
    args = parser.parse_args()

    module = SPADERefineInpaintGANModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_g=args.g_lr,
        lr_d=args.d_lr,
        lambda_identity=args.lambda_identity,
        lambda_edge=args.lambda_edge,
        lambda_style=args.lambda_style,
        g_steps=args.g_steps,
        control_rough_labels_dir=args.control_rough_labels_dir,
        control_rough_tumors_dir=args.control_rough_tumors_dir,
    )

    logger = CSVLogger("logs", name="spade_gan_refine_inpaint")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=25,
        val_check_interval=0.25,
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()

