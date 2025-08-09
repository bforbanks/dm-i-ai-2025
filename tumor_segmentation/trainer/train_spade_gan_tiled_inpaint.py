import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_tiled_inpaint_module import SPADETiledInpaintGANModule


def main():
    parser = argparse.ArgumentParser(description="Train SPADE tiled inpainting GAN")
    parser.add_argument(
        "--data_root",
        type=str,
        default="tumor_segmentation/data",
        help="Path containing controls/ and patients/ folders",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--g_lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--lambda_fft", type=float, default=10.0)  # Enable FFT loss
    parser.add_argument("--lambda_perceptual", type=float, default=0.0)  # Enable perceptual loss
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=256, help="Size of training tiles")
    args = parser.parse_args()

    module = SPADETiledInpaintGANModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_g=args.g_lr,
        lr_d=args.d_lr,
        lambda_l1=args.lambda_l1,
        lambda_fft=args.lambda_fft,
        lambda_perceptual=args.lambda_perceptual,
        g_steps=args.g_steps,
        tile_size=args.tile_size,
    )

    logger = CSVLogger("logs", name="spade_gan_tiled_inpaint")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        precision=16,
        logger=logger,
        log_every_n_steps=25,
        default_root_dir="spade_gan_tiled_inpaint_ckpts",
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()