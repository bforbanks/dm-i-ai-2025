from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets import SyntheticTumorSliceDataset  # noqa: E402
from models.gan_components import GANBuilder  # noqa: E402

try:
    from kornia.losses import FFTConsistencyLoss  # type: ignore
except ImportError:  # pragma: no cover
    FFTConsistencyLoss = None  # noqa: N816


class SPADEInpaintGANModule(pl.LightningModule):
    """SPADE GAN variant that performs masked inpainting on patient images.

    The generator predicts *only* the tumour region. The final fake image is
    created by taking the patient image background and replacing the tumor
    region with generated content:

        fake_full = patient * (1 − mask) + gen_out * mask

    During training, the discriminator compares the inpainted patient image
    against the original patient image, forcing the generator to focus
    solely on synthesising realistic tumour textures that match the original.
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_l1: float = 50.0,
        lambda_fft: float = 2.0,
        g_steps: int = 1,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        # Prepare dataset path first so we can inspect image dimensions
        self.data_root = Path(data_root)

        # Probe dataset once to get padding dimensions
        tmp_ds = SyntheticTumorSliceDataset(self.data_root)
        if hasattr(tmp_ds, "target_w") and tmp_ds.target_w:
            gan_kwargs.update({
                "crop_size_w": tmp_ds.target_w,
                "crop_size_h": tmp_ds.target_h,
            })

        # Will build networks later once DataLoader is prepared
        self.builder: GANBuilder | None = None
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.gan_kwargs = gan_kwargs

        # Losses
        self.criterion_l1 = nn.L1Loss()
        self.criterion_fft = (
            FFTConsistencyLoss(reduction="mean") if FFTConsistencyLoss else None
        )

        self.batch_size = batch_size

        # Manual optimisation
        self.automatic_optimization = False
        self.g_steps = g_steps
        self._d_updates = 0

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SyntheticTumorSliceDataset(self.data_root, auto_pad=True)

            if self.builder is None:
                # Ensure GANBuilder knows the exact dimensions
                self.gan_kwargs.update(
                    {
                        "crop_size_w": self.train_dataset.target_w,
                        "crop_size_h": self.train_dataset.target_h,
                    }
                )
                self.builder = GANBuilder(**self.gan_kwargs)
                self.generator, self.discriminator = self.builder.build()

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    # ---------------------------------------------------------------------
    # Optimisers
    # ---------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.0, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.0, 0.999)
        )
        return [opt_g, opt_d]

    # ---------------------------------------------------------------------
    # Training step (manual optimisation)
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        real_img: torch.Tensor = batch["real_image"]  # (B,1,H,W)
        control: torch.Tensor = batch["control"]  # (B,1,H,W)
        mask: torch.Tensor = batch["mask"]  # (B,1,H,W)
        gen_input: torch.Tensor = batch["gen_input"]  # (B,2,H,W)

        # --------------------------------------------------
        # Train Discriminator
        # --------------------------------------------------
        with torch.no_grad():
            pred_tumour = self.generator(gen_input)  # (B,1,H,W)
        
        # Use patient image as background, inpaint only tumor region
        patient_background = real_img * (1 - mask)  # Patient tissue outside tumor
        fake_full = patient_background + pred_tumour * mask  # Patient background + generated tumor
        real_full = real_img  # Original patient image (ground truth)

        fake_combined = torch.cat([mask, fake_full], dim=1)
        real_combined = torch.cat([mask, real_full], dim=1)

        fake_pred = self.discriminator(fake_combined)
        real_pred = self.discriminator(real_combined)

        d_loss_fake = sum(F.relu(1.0 + fp[-1]).mean() for fp in fake_pred) / len(fake_pred)
        d_loss_real = sum(F.relu(1.0 - rp[-1]).mean() for rp in real_pred) / len(real_pred)
        d_loss = 0.5 * (d_loss_fake + d_loss_real)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # --------------------------------------------------
        # Train Generator
        # --------------------------------------------------
        if self._d_updates % self.g_steps == 0:
            pred_tumour = self.generator(gen_input)
            
            # Use patient image as background, inpaint only tumor region
            patient_background = real_img * (1 - mask)
            fake_full = patient_background + pred_tumour * mask
            pred_fake = self.discriminator(torch.cat([mask, fake_full], dim=1))

            g_gan = -sum(p[-1].mean() for p in pred_fake) / len(pred_fake)
            
            # Focus losses on tumor region only (most important part)
            g_l1 = self.criterion_l1(pred_tumour * mask, real_img * mask)
            if self.criterion_fft is not None:
                g_fft = self.criterion_fft(pred_tumour * mask, real_img * mask)
            else:
                g_fft = torch.tensor(0.0, device=self.device)

            g_loss = g_gan + self.hparams.lambda_l1 * g_l1 + self.hparams.lambda_fft * g_fft

            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(g_loss)
            opt_g.step()

            self.log_dict(
                {
                    "g_loss": g_loss,
                    "g_gan": g_gan,
                    "g_l1": g_l1,
                    "g_fft": g_fft,
                },
                prog_bar=True,
                on_step=True,
            )

        self._d_updates += 1

    # ---------------------------------------------------------------------
    # Visualisation helper (optional)
    # ---------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Save a few example inpainted images for qualitative inspection."""
        import os
        from torchvision.utils import save_image

        vis_dir = Path(self.logger.log_dir).parent / "gan_inpaint_vis"
        os.makedirs(vis_dir, exist_ok=True)

        self.generator.eval()
        with torch.no_grad():
            for i in range(5):
                sample = self.train_dataset[i]
                gen_input = sample["gen_input"].unsqueeze(0).to(self.device)
                mask = sample["mask"].unsqueeze(0).to(self.device)
                control = sample["control"].unsqueeze(0).to(self.device)
                pred_tumour = self.generator(gen_input).clamp(0, 1)
                fake_full = control * (1 - mask) + pred_tumour * mask

                save_image(fake_full.cpu(), vis_dir / f"fake_full_{i}.png")
                save_image(pred_tumour.cpu(), vis_dir / f"pred_tumour_{i}.png")
                save_image(mask.cpu(), vis_dir / f"mask_{i}.png")

        self.generator.train()
