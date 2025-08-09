from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets import SyntheticTumorSliceDataset  # noqa: E402
from models.gan_components import GANBuilder  # noqa: E402


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward."""
        return self.net(x)


class MaskedRefinerUNet(nn.Module):
    """Small U-Net refiner predicting in-mask residual on top of generator output."""

    def __init__(self, base_ch: int = 32, in_ch: int = 3) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(base_ch * 2 + base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.tanh(self.out_conv(d1)) * 0.2


class PatchDiscriminator(nn.Module):
    """Lightweight PatchGAN discriminator, conditioned on mask.

    Input channels: 2 (mask + image)
    Output: patch logits (B, 1, H', W')
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = base_ch
        layers += [nn.utils.spectral_norm(nn.Conv2d(in_ch, ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(1, n_layers):
            prev = ch
            ch = min(ch * 2, 512)
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(prev, ch, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.utils.spectral_norm(nn.Conv2d(ch, 1, 3, stride=1, padding=1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def get_bbox_from_mask(mask: torch.Tensor, pad: int = 16) -> Tuple[int, int, int, int]:
    """Compute bounding box (y0, x0, y1, x1) from binary mask (1,H,W) with padding.

    Returns full-image box if mask is empty.
    """
    c, h, w = mask.shape
    assert c == 1
    ys, xs = torch.where(mask[0] > 0.5)
    if ys.numel() == 0:
        return 0, 0, h, w
    y0 = max(int(ys.min().item()) - pad, 0)
    y1 = min(int(ys.max().item()) + pad + 1, h)
    x0 = max(int(xs.min().item()) - pad, 0)
    x1 = min(int(xs.max().item()) + pad + 1, w)
    return y0, x0, y1, x1


def resize_pair(img: torch.Tensor, mask: torch.Tensor, size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize image and mask to square size using appropriate interpolation."""
    img_r = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    mask_r = F.interpolate(mask, size=(size, size), mode="nearest")
    return img_r, mask_r


class SPADERefinerAdvModule(pl.LightningModule):
    """Adversarially train a refiner to make tumors look real using real patient crops.

    - Generator is frozen; it synthesizes a tumor on the control canvas.
    - Refiner adjusts only within the mask to improve realism.
    - Discriminator sees local crops (mask-conditioned) from real patients vs refined fakes.
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_refiner: float = 2e-4,
        lr_disc: float = 2e-4,
        lambda_adv: float = 1.0,
        lambda_tv: float = 0.1,
        lambda_delta: float = 1.0,
        residual_scale: float = 0.3,
        crop_size: int = 128,
        gen_ckpt_path: str | None = None,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.residual_scale = residual_scale
        self.crop_size = crop_size
        self.gen_ckpt_path = gen_ckpt_path

        self.builder: GANBuilder | None = None
        self.generator: nn.Module | None = None
        # Refiner takes [initial_fake_on_control, control, mask] => 3ch
        self.refiner = MaskedRefinerUNet(base_ch=32, in_ch=3)
        # Discriminator on [mask, crop] => 2ch
        self.discriminator = PatchDiscriminator(in_ch=2, base_ch=64, n_layers=3)

        # Manual optimization with two optims
        self.automatic_optimization = False

        self.gan_kwargs = gan_kwargs

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SyntheticTumorSliceDataset(self.data_root, auto_pad=True)
            if self.builder is None:
                self.gan_kwargs.update(
                    {
                        "crop_size_w": self.train_dataset.target_w,
                        "crop_size_h": self.train_dataset.target_h,
                    }
                )
                self.builder = GANBuilder(**self.gan_kwargs)
                self.generator, _ = self.builder.build()
                for p in self.generator.parameters():
                    p.requires_grad = False
                if self.gen_ckpt_path:
                    self._try_load_generator_weights(self.gen_ckpt_path)

    def _try_load_generator_weights(self, ckpt_path: str) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to load checkpoint at {ckpt_path}: {e}")
            return

        state_dict: Dict[str, torch.Tensor] | None = None
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt

        if state_dict is None:
            print(f"Warning: No state_dict found in {ckpt_path}")
            return

        gen_state = {}
        for k, v in state_dict.items():
            if k.startswith("generator."):
                gen_state[k[len("generator."):]] = v
            elif k.startswith("G."):
                gen_state[k[len("G."):]] = v
            elif k.startswith("module.generator."):
                gen_state[k[len("module.generator."):]] = v
        if not gen_state:
            gen_state = state_dict
        try:
            self.generator.load_state_dict(gen_state, strict=False)
            print(f"Loaded generator weights from {ckpt_path}")
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to load generator weights: {e}")

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
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
        opt_g = torch.optim.Adam(self.refiner.parameters(), lr=self.hparams.lr_refiner, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_disc, betas=(0.0, 0.999))
        return [opt_g, opt_d]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        patient_img: torch.Tensor = batch["real_image"]  # (B,1,H,W)
        mask: torch.Tensor = batch["mask"]               # (B,1,H,W)
        control_img: torch.Tensor = batch["control"]     # (B,1,H,W)
        gen_input: torch.Tensor = batch["gen_input"]     # (B,2,H,W)

        with torch.no_grad():
            pred_residual = self.generator(gen_input) * self.residual_scale
            # Compose on control canvas (as in inference)
            initial_fake = torch.clamp(control_img + pred_residual * mask, 0.0, 1.0)

        # Refiner forward (constrained by mask later)
        refiner_in = torch.cat([initial_fake, control_img, mask], dim=1)
        delta = self.refiner(refiner_in)
        refined = torch.clamp(initial_fake + delta * mask, 0.0, 1.0)

        # Build local crops (real patient tumor vs refined fake on control)
        real_crops: List[torch.Tensor] = []
        real_masks: List[torch.Tensor] = []
        fake_crops: List[torch.Tensor] = []
        fake_masks: List[torch.Tensor] = []
        for b in range(patient_img.shape[0]):
            y0, x0, y1, x1 = get_bbox_from_mask(mask[b])
            real_patch = patient_img[b : b + 1, :, y0:y1, x0:x1]
            real_mask = mask[b : b + 1, :, y0:y1, x0:x1]
            fake_patch = refined[b : b + 1, :, y0:y1, x0:x1]
            fake_mask = real_mask  # same box

            real_patch, real_mask = resize_pair(real_patch, real_mask, size=self.crop_size)
            fake_patch, fake_mask = resize_pair(fake_patch, fake_mask, size=self.crop_size)

            real_crops.append(real_patch)
            real_masks.append(real_mask)
            fake_crops.append(fake_patch)
            fake_masks.append(fake_mask)

        real_crops_t = torch.cat(real_crops, dim=0)
        real_masks_t = torch.cat(real_masks, dim=0)
        fake_crops_t = torch.cat(fake_crops, dim=0)
        fake_masks_t = torch.cat(fake_masks, dim=0)

        # -------------------- Train D --------------------
        self.toggle_optimizer(opt_d)
        d_real_in = torch.cat([real_masks_t, real_crops_t], dim=1)
        d_fake_in = torch.cat([fake_masks_t, fake_crops_t.detach()], dim=1)
        pred_real = self.discriminator(d_real_in)
        pred_fake = self.discriminator(d_fake_in)

        d_loss_real = F.relu(1.0 - pred_real).mean()
        d_loss_fake = F.relu(1.0 + pred_fake).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # -------------------- Train Refiner (G) --------------------
        self.toggle_optimizer(opt_g)
        d_fake_in = torch.cat([fake_masks_t, fake_crops_t], dim=1)
        pred_fake_for_g = self.discriminator(d_fake_in)
        g_adv = -pred_fake_for_g.mean()

        # Regularizers: keep deltas small and smooth inside mask
        tv = (torch.mean(torch.abs(delta[:, :, :, :-1] - delta[:, :, :, 1:])) +
              torch.mean(torch.abs(delta[:, :, :-1, :] - delta[:, :, 1:, :])))
        delta_l2 = torch.mean((delta * mask) ** 2)

        g_loss = self.hparams.lambda_adv * g_adv + self.hparams.lambda_tv * tv + self.hparams.lambda_delta * delta_l2

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log_dict({
            "g_loss": g_loss,
            "g_adv": g_adv,
            "tv": tv,
            "delta_l2": delta_l2,
        }, prog_bar=True, on_step=True)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Save comparison crops: real tumor vs refined fake, plus full images."""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        vis_dir = Path(self.logger.log_dir) / "refiner_adv_vis"
        os.makedirs(vis_dir, exist_ok=True)

        control_imgs = list((self.data_root / "controls" / "imgs").glob("*.png"))
        patient_pairs = [(p, m) for p, m in self.train_dataset.patient_pairs]

        self.generator.eval()
        self.refiner.eval()
        with torch.no_grad():
            for i in range(min(5, len(control_imgs))):
                control_img_path = control_imgs[i]
                patient_img_path, mask_path = patient_pairs[i % len(patient_pairs)]

                control = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
                patient = np.array(Image.open(patient_img_path).convert("L"), dtype=np.float32) / 255.0
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

                # Ensure same size and pad if needed
                target_w, target_h = self.train_dataset.target_w, self.train_dataset.target_h
                def pad_to(arr, pad_val):
                    if arr.shape == (target_h, target_w):
                        return arr
                    h, w = arr.shape
                    pad_left = (target_w - w) // 2
                    pad_top = 0
                    canvas = np.full((target_h, target_w), pad_val, dtype=np.float32)
                    canvas[pad_top:pad_top+h, pad_left:pad_left+w] = arr
                    return canvas

                control = pad_to(control, 1.0)
                patient = pad_to(patient, 1.0)
                mask = pad_to(mask, 0.0)

                control_t = torch.from_numpy(control).unsqueeze(0).unsqueeze(0).to(self.device)
                patient_t = torch.from_numpy(patient).unsqueeze(0).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
                gen_input = torch.cat([control_t, mask_t], dim=1)

                pred_residual = self.generator(gen_input) * self.residual_scale
                initial_fake = torch.clamp(control_t + pred_residual * mask_t, 0.0, 1.0)
                refiner_in = torch.cat([initial_fake, control_t, mask_t], dim=1)
                delta = self.refiner(refiner_in)
                refined = torch.clamp(initial_fake + delta * mask_t, 0.0, 1.0)

                # Crop box
                y0, x0, y1, x1 = get_bbox_from_mask(mask_t[0])
                real_crop = patient_t[0:1, :, y0:y1, x0:x1]
                fake_crop = refined[0:1, :, y0:y1, x0:x1]
                real_crop, _ = resize_pair(real_crop, mask_t[0:1, :, y0:y1, x0:x1], size=self.crop_size)
                fake_crop, _ = resize_pair(fake_crop, mask_t[0:1, :, y0:y1, x0:x1], size=self.crop_size)

                real_np = real_crop[0, 0].cpu().numpy()
                fake_np = fake_crop[0, 0].cpu().numpy()
                init_np = initial_fake[0, 0].cpu().numpy()
                ref_np = refined[0, 0].cpu().numpy()

                fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                axes[0,0].imshow(control, cmap="gray", vmin=0, vmax=1); axes[0,0].set_title("Control (full)"); axes[0,0].axis("off")
                axes[0,1].imshow(init_np, cmap="gray", vmin=0, vmax=1); axes[0,1].set_title("Initial fake (full)"); axes[0,1].axis("off")
                axes[0,2].imshow(ref_np, cmap="gray", vmin=0, vmax=1); axes[0,2].set_title("Refined (full)"); axes[0,2].axis("off")
                axes[1,0].imshow(patient, cmap="gray", vmin=0, vmax=1); axes[1,0].set_title("Patient (full)"); axes[1,0].axis("off")
                axes[1,1].imshow(real_np, cmap="gray", vmin=0, vmax=1); axes[1,1].set_title("Real tumor crop"); axes[1,1].axis("off")
                axes[1,2].imshow(fake_np, cmap="gray", vmin=0, vmax=1); axes[1,2].set_title("Refined fake crop"); axes[1,2].axis("off")
                plt.tight_layout()
                plt.savefig(vis_dir / f"refiner_adv_comparison_{i}.png", dpi=150, bbox_inches="tight")
                plt.close()

        print(f"Refiner-ADV visualizations saved to: {vis_dir}")

