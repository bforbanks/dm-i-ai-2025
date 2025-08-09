from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets import SyntheticTumorSliceDataset  # noqa: E402
from models.gan_components import GANBuilder  # noqa: E402

try:  # Optional FFT loss
    from kornia.losses import FFTConsistencyLoss  # type: ignore
except ImportError:  # pragma: no cover
    FFTConsistencyLoss = None  # noqa: N816


class SPADEFullInpaintGANModule(pl.LightningModule):
    """SPADE GAN variant that performs masked inpainting on full patient images.

    The generator predicts only the tumour region. The final fake image is:
        fake_full = patient * (1 − mask) + gen_out * mask

    The discriminator compares the inpainted patient image against the original
    patient image, focusing learning on realistic tumour textures.
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
        blend_border_px: int = 6,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        # Dataset path for dimension probing and visualisation
        self.data_root = Path(data_root)

        # Probe dataset once to pass exact dims to SPADE generator
        tmp_ds = SyntheticTumorSliceDataset(self.data_root)
        if hasattr(tmp_ds, "target_w") and tmp_ds.target_w:
            gan_kwargs.update({
                "crop_size_w": tmp_ds.target_w,
                "crop_size_h": tmp_ds.target_h,
            })

        # Will build networks after DataLoader is prepared
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

        # Edge blending width (in pixels) to feather tumour borders during inference/visualisation
        self.blend_border_px = max(0, int(blend_border_px))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SyntheticTumorSliceDataset(self.data_root, auto_pad=True)

            if self.builder is None:
                # Ensure GANBuilder knows exact dimensions
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

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.0, 0.999))
        return [opt_g, opt_d]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        real_img: torch.Tensor = batch["real_image"]
        control: torch.Tensor = batch["control"]
        mask: torch.Tensor = batch["mask"]
        gen_input: torch.Tensor = batch["gen_input"]

        # ---------------------- Train D ----------------------
        with torch.no_grad():
            pred_tumour = self.generator(gen_input)

        patient_background = real_img * (1 - mask)
        fake_full = patient_background + pred_tumour * mask
        real_full = real_img

        fake_combined = torch.cat([mask, fake_full], dim=1)
        real_combined = torch.cat([mask, real_full], dim=1)

        fake_pred = self.discriminator(fake_combined)
        real_pred = self.discriminator(real_combined)

        # Hinge loss (as in original good setup)
        d_loss_fake = sum(F.relu(1.0 + fp[-1]).mean() for fp in fake_pred) / len(fake_pred)
        d_loss_real = sum(F.relu(1.0 - rp[-1]).mean() for rp in real_pred) / len(real_pred)
        d_loss = 0.5 * (d_loss_fake + d_loss_real)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # ---------------------- Train G ----------------------
        if self._d_updates % self.g_steps == 0:
            pred_tumour = self.generator(gen_input)
            patient_background = real_img * (1 - mask)
            fake_full = patient_background + pred_tumour * mask
            pred_fake = self.discriminator(torch.cat([mask, fake_full], dim=1))

            # Hinge generator term
            g_gan = -sum(p[-1].mean() for p in pred_fake) / len(pred_fake)

            # Focus on tumour region only
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
                {"g_loss": g_loss, "g_gan": g_gan, "g_l1": g_l1, "g_fft": g_fft},
                prog_bar=True,
                on_step=True,
            )

        self._d_updates += 1

    # ------------------------------------------------------------------
    # Inference on a full image (pads then crops back)
    # ------------------------------------------------------------------
    def inpaint_full_image(
        self,
        patient_img_path: Path,
        mask_path: Path,
        control_imgs: List[Path],
        device: str = "cuda",
    ) -> np.ndarray:
        """Inpaint a full image using a random control image for context.
        Returns output with the SAME size as the original patient/control image.
        """
        import numpy as np
        from PIL import Image

        patient_img = np.array(Image.open(patient_img_path).convert("L"), dtype=np.float32) / 255.0
        mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        original_h, original_w = patient_img.shape

        # Ensure mask matches patient size
        if mask_img.shape != patient_img.shape:
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(patient_img.shape[::-1], Image.NEAREST)
            mask_img = np.array(mask_pil, dtype=np.float32) / 255.0

        # Pad to training dimensions if required (white=1 for images, black=0 for mask)
        if hasattr(self.train_dataset, "target_w") and self.train_dataset.target_w:
            target_w, target_h = self.train_dataset.target_w, self.train_dataset.target_h
            if patient_img.shape != (target_h, target_w):
                h, w = patient_img.shape
                pad_left = (target_w - w) // 2
                pad_top = 0
                canvas = np.ones((target_h, target_w), dtype=np.float32)
                canvas[pad_top:pad_top + h, pad_left:pad_left + w] = patient_img
                patient_img = canvas

                canvas = np.zeros((target_h, target_w), dtype=np.float32)
                canvas[pad_top:pad_top + h, pad_left:pad_left + w] = mask_img
                mask_img = canvas

        # Pick a control
        control_img_path = np.random.choice(control_imgs)
        control_img = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
        if control_img.shape != patient_img.shape:
            h, w = control_img.shape
            pad_left = (patient_img.shape[1] - w) // 2
            pad_top = 0
            canvas = np.ones(patient_img.shape, dtype=np.float32)
            canvas[pad_top:pad_top + h, pad_left:pad_left + w] = control_img
            control_img = canvas

        # To tensors
        patient_tensor = torch.from_numpy(patient_img).unsqueeze(0).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to(device)
        control_tensor = torch.from_numpy(control_img).unsqueeze(0).unsqueeze(0).to(device)
        gen_input = torch.cat([control_tensor, mask_tensor], dim=1)

        # Generate
        self.generator.eval()
        with torch.no_grad():
            pred_tumour = self.generator(gen_input)
            # Optional edge-only blending using a soft mask
            soft_mask = mask_tensor
            if self.blend_border_px > 0:
                # Build a small separable Gaussian kernel; radius ~ blend_border_px
                radius = int(self.blend_border_px)
                kernel_size = 2 * radius + 1
                # 1D Gaussian
                x = torch.arange(kernel_size, device=device, dtype=patient_tensor.dtype) - radius
                sigma = max(1.0, radius / 2.0)
                g = torch.exp(-(x**2) / (2 * sigma * sigma))
                g = (g / g.sum()).view(1, 1, 1, -1)
                # Horizontal then vertical blur to approximate Gaussian 2D
                sm = F.conv2d(mask_tensor, g, padding=(0, radius))
                gT = g.transpose(2, 3)
                sm = F.conv2d(sm, gT, padding=(radius, 0))
                # Clamp to [0,1]
                soft_mask = torch.clamp(sm, 0.0, 1.0)

            patient_background = patient_tensor * (1 - soft_mask)
            output_img = patient_background + pred_tumour * soft_mask
            output_img = output_img.cpu().numpy()[0, 0]
        self.generator.train()

        # Crop back to original (undo padding)
        if output_img.shape != (original_h, original_w):
            target_w, target_h = self.train_dataset.target_w, self.train_dataset.target_h
            h, w = original_h, original_w
            pad_left = (target_w - w) // 2
            pad_top = 0
            output_img = output_img[pad_top:pad_top + h, pad_left:pad_left + w]
        return output_img

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Save comparison plots (original control, generated, mask) under version dir."""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        vis_dir = Path(self.logger.log_dir) / "gan_full_image_vis"
        os.makedirs(vis_dir, exist_ok=True)

        control_imgs = list((self.data_root / "controls" / "imgs").glob("*.png"))
        patient_pairs = [(p, m) for p, m in self.train_dataset.patient_pairs]

        self.generator.eval()
        with torch.no_grad():
            for i in range(min(5, len(control_imgs))):
                control_img_path = control_imgs[i]
                _, mask_path = patient_pairs[i % len(patient_pairs)]

                synthetic_img = self.inpaint_full_image(
                    patient_img_path=control_img_path,
                    mask_path=mask_path,
                    control_imgs=[control_img_path],
                    device=self.device,
                )

                control_orig = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
                mask_orig = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

                if mask_orig.shape != control_orig.shape:
                    mask_pil = Image.fromarray((mask_orig * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(control_orig.shape[::-1], Image.NEAREST)
                    mask_orig = np.array(mask_pil, dtype=np.float32) / 255.0

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(control_orig, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Original Control"); axes[0].axis("off")
                axes[1].imshow(synthetic_img, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Control + Generated Tumor"); axes[1].axis("off")
                axes[2].imshow(control_orig, cmap="gray", vmin=0, vmax=1)
                axes[2].imshow(mask_orig, cmap="Reds", alpha=0.5, vmin=0, vmax=1); axes[2].set_title("Tumor Mask Location"); axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(vis_dir / f"synthetic_tumor_comparison_{i}.png", dpi=150, bbox_inches="tight")
                plt.close()

        self.generator.train()
        print(f"Synthetic tumor visualizations saved to: {vis_dir}")