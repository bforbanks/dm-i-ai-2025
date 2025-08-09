from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets import SyntheticTumorSliceDataset
from models.gan_components import GANBuilder

try:
    from kornia.losses import FFTConsistencyLoss  # type: ignore
except ImportError:  # pragma: no cover
    FFTConsistencyLoss = None  # noqa: N816


class SPADEGANModule(pl.LightningModule):
    """LightningModule wrapping SPADE generator + multi-scale discriminator.

    Training uses manual optimization to update D once per batch and G every
    *g_steps* batches (default 1).
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_l1: float = 50.0,
        lambda_fft: float = 2.0,
        g_steps: int = 2,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        # Dataset / dataloader parameters - set this first
        self.data_root = Path(data_root)

        # Auto-detect dimensions from dataset first
        temp_ds = SyntheticTumorSliceDataset(self.data_root)
        if hasattr(temp_ds, 'target_w') and temp_ds.target_w:
            gan_kwargs.update({
                'crop_size_w': temp_ds.target_w,
                'crop_size_h': temp_ds.target_h,
            })
        
        # Build networks - will be created after dataset setup
        self.builder = None
        self.generator = None 
        self.discriminator = None
        self.gan_kwargs = gan_kwargs

        # Loss helpers
        self.criterion_l1 = nn.L1Loss()
        self.criterion_fft = (
            FFTConsistencyLoss(reduction="mean") if FFTConsistencyLoss else None
        )
        self.batch_size = batch_size

        # Manual opt flag
        self.automatic_optimization = False
        self.g_steps = g_steps
        self._d_updates = 0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        if stage == "fit" or stage is None:
            self.train_dataset = SyntheticTumorSliceDataset(self.data_root, auto_pad=True)
            
            # Now build networks with detected dimensions
            if self.builder is None:
                self.gan_kwargs.update({
                    'crop_size_w': self.train_dataset.target_w,
                    'crop_size_h': self.train_dataset.target_h,
                })
                self.builder = GANBuilder(**self.gan_kwargs)
                self.generator, self.discriminator = self.builder.build()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.0, 0.999))
        return [opt_g, opt_d]

    # ------------------------------------------------------------------
    # Training step (manual optimization)
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        real_img: torch.Tensor = batch["real_image"]  # shape (B,1,H,W)
        gen_input: torch.Tensor = batch["gen_input"]   # (B,2,H,W)
        mask: torch.Tensor = batch["mask"]

        # ------------------ Train D ------------------
        with torch.no_grad():
            fake_img = self.generator(gen_input)
        fake_combined = torch.cat([mask, fake_img], dim=1)
        real_combined = torch.cat([mask, real_img], dim=1)

        fake_pred = self.discriminator(fake_combined)
        real_pred = self.discriminator(real_combined)

        d_loss_fake = sum(F.relu(1.0 + fp[-1]).mean() for fp in fake_pred) / len(fake_pred)
        d_loss_real = sum(F.relu(1.0 - rp[-1]).mean() for rp in real_pred) / len(real_pred)
        d_loss = 0.5 * (d_loss_fake + d_loss_real)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # ------------------ Train G ------------------
        if self._d_updates % self.g_steps == 0:
            fake_img = self.generator(gen_input)
            fake_combined = torch.cat([mask, fake_img], dim=1)
            pred_fake = self.discriminator(fake_combined)

            g_gan = -sum(p[-1].mean() for p in pred_fake) / len(pred_fake)
            g_l1 = self.criterion_l1(fake_img * mask, real_img * mask)
            g_fft = (
                self.criterion_fft(fake_img, real_img)
                if self.criterion_fft is not None
                else torch.tensor(0.0, device=self.device)
            )

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

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Generate a few example outputs and save to gan_vis/ folder."""
        import os
        from torchvision.utils import save_image

        vis_dir = Path(self.logger.log_dir).parent / "gan_vis"
        os.makedirs(vis_dir, exist_ok=True)

        self.generator.eval()
        with torch.no_grad():
            for i in range(5):
                sample = self.train_dataset[i]
                gen_input = sample["gen_input"].unsqueeze(0).to(self.device)
                fake_img = self.generator(gen_input).cpu().clamp(0, 1)
                mask = sample["mask"].unsqueeze(0)

                save_image(fake_img, vis_dir / f"fake_{i}.png")
                save_image(mask, vis_dir / f"mask_{i}.png")

        self.generator.train()

    # ------------------------------------------------------------------
    # Validation / test could be added similarly if needed
    # ------------------------------------------------------------------
