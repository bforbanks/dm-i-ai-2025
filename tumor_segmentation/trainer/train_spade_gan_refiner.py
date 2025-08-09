#!/usr/bin/env python3
"""Train a post-generator masked refiner to improve body–tumor blending."""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_refiner_module import SPADERefinerModule


def main():
    parser = argparse.ArgumentParser(description="Train SPADE generator refiner (masked)")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr_refiner", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=50.0)
    parser.add_argument("--lambda_fft", type=float, default=2.0)
    parser.add_argument("--lambda_grad", type=float, default=5.0)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--freeze_generator", action="store_true", default=True)
    parser.add_argument("--gen_ckpt_path", type=str, default=None, help="Optional path to load generator weights")
    args = parser.parse_args()

    module = SPADERefinerModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_refiner=args.lr_refiner,
        lambda_l1=args.lambda_l1,
        lambda_fft=args.lambda_fft,
        lambda_grad=args.lambda_grad,
        residual_scale=args.residual_scale,
        freeze_generator=args.freeze_generator,
        gen_ckpt_path=args.gen_ckpt_path,
    )

    logger = CSVLogger("logs", name="spade_gan_refiner")

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

