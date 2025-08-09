import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from spade_gan_inpaint_module import SPADEInpaintGANModule


def main():
    parser = argparse.ArgumentParser(description="Train SPADE masked-inpainting GAN")
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
    parser.add_argument("--lambda_l1", type=float, default=50.0)
    parser.add_argument("--lambda_fft", type=float, default=2.0)
    parser.add_argument("--g_steps", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)
    args = parser.parse_args()

    module = SPADEInpaintGANModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr_g=args.g_lr,
        lr_d=args.d_lr,
        lambda_l1=args.lambda_l1,
        lambda_fft=args.lambda_fft,
        g_steps=args.g_steps,
        crop_size=args.crop_size,
    )

    logger = CSVLogger("logs", name="spade_gan_inpaint")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        precision=16,
        logger=logger,
        log_every_n_steps=25,
        default_root_dir="spade_gan_inpaint_ckpts",
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
