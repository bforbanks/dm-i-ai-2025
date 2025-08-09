#!/usr/bin/env python3
"""Train adversarial refiner focused on local tumor realism using patient crops."""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_refiner_adv_module import SPADERefinerAdvModule


def main():
    parser = argparse.ArgumentParser(description="Train adversarial SPADE refiner (masked, local realism)")
    parser.add_argument("--data_root", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr_refiner", type=float, default=2e-4)
    parser.add_argument("--lr_disc", type=float, default=2e-4)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_tv", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=1.0)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--gen_ckpt_path", type=str, default=None, help="Optional path to load generator weights")
    args = parser.parse_args()

    module = SPADERefinerAdvModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_refiner=args.lr_refiner,
        lr_disc=args.lr_disc,
        lambda_adv=args.lambda_adv,
        lambda_tv=args.lambda_tv,
        lambda_delta=args.lambda_delta,
        residual_scale=args.residual_scale,
        crop_size=args.crop_size,
        gen_ckpt_path=args.gen_ckpt_path,
    )

    logger = CSVLogger("logs", name="spade_gan_refiner_adv")

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

