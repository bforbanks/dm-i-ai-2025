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

from datasets.refine_tumor_dataset import RefineTumorSliceDataset  # noqa: E402
from models.pix2pix_components import Pix2PixBuilder  # noqa: E402


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edges for single-channel images with numerical safety."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    edges_x = F.conv2d(x, sobel_x, padding=1)
    edges_y = F.conv2d(x, sobel_y, padding=1)
    mag2 = edges_x.pow(2) + edges_y.pow(2)
    return torch.sqrt(torch.clamp(mag2, min=1e-12))


def create_boundary_ring(mask: torch.Tensor, ring_width: int = 3) -> torch.Tensor:
    """Create boundary ring around mask for edge consistency."""
    kernel = torch.ones(1, 1, 2*ring_width+1, 2*ring_width+1, device=mask.device, dtype=mask.dtype)
    dilated = F.conv2d(mask, kernel, padding=ring_width) > 0
    ring = dilated.float() - mask
    return torch.clamp(ring, 0, 1)


def binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is strictly binary in {0,1} and detached from gradients."""
    return (mask > 0.5).to(mask.dtype)


def masked_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute per-sample masked standard deviation over HxW with safe denominator."""
    mask = binarize_mask(mask)
    num = mask.sum(dim=(-2, -1), keepdim=True) + eps
    mean = (x * mask).sum(dim=(-2, -1), keepdim=True) / num
    var = (((x - mean) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / num
    return torch.sqrt(torch.clamp(var, min=0.0) + eps)


class Pix2PixRefineInpaintModule(pl.LightningModule):
    """pix2pix-style refinement on control domain with unpaired patient realism.

    Fake: [control_mask, control_image*(1-control_mask) + G(rough_control, control_mask)*control_mask]
    Real: [patient_mask, patient_image]
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_identity: float = 1.0,
        lambda_edge: float = 10.0,
        lambda_style: float = 5.0,
        g_steps: int = 1,
        control_rough_labels_dir: str = "controls/rough_labels",
        control_rough_tumors_dir: str = "controls/rough_tumors",
        **builder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["builder_kwargs"])

        self.data_root = Path(data_root)
        self.batch_size = batch_size

        self.builder_kwargs = builder_kwargs
        self.builder: Pix2PixBuilder | None = None
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None

        self.criterion_l1 = nn.L1Loss()
        
        # Simple feature extractor for style loss
        self.style_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )

        self.automatic_optimization = False
        self.g_steps = g_steps
        self._d_updates = 0

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = RefineTumorSliceDataset(
                self.data_root, 
                auto_pad=True,
                control_rough_labels_dir=self.hparams.control_rough_labels_dir,
                control_rough_tumors_dir=self.hparams.control_rough_tumors_dir
            )

            if self.builder is None:
                if hasattr(self.train_dataset, "target_w") and self.train_dataset.target_w:
                    self.builder_kwargs.update(
                        {
                            "crop_size_w": self.train_dataset.target_w,
                            "crop_size_h": self.train_dataset.target_h,
                        }
                    )
                self.builder = Pix2PixBuilder(**self.builder_kwargs)
                self.generator, self.discriminator = self.builder.build()

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def configure_optimizers(self):  # type: ignore[override]
        assert self.generator is not None and self.discriminator is not None, "Models not built yet. Call setup('fit') first."
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.0, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        assert self.generator is not None and self.discriminator is not None, "Models not built yet."
        opt_g, opt_d = self.optimizers()

        def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
            # Replace NaN/Inf and clamp to [0, 1] for images/masks
            return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

        def _sanitize_preds(pred_list):
            sanitized = []
            for p in pred_list:
                # Some discriminators return list of feature lists
                if isinstance(p, (list, tuple)):
                    sanitized.append([_sanitize_tensor(t) for t in p])
                else:
                    sanitized.append(_sanitize_tensor(p))
            return sanitized

        def _is_finite(*tensors: torch.Tensor) -> bool:
            return all(torch.isfinite(t).all().item() for t in tensors)

        control_img: torch.Tensor = _sanitize_tensor(batch["control_image"]).clamp(0.0, 1.0)
        control_mask: torch.Tensor = _sanitize_tensor(batch["control_mask"]).clamp(0.0, 1.0)
        gen_input: torch.Tensor = _sanitize_tensor(batch["gen_input"]).clamp(0.0, 1.0)
        rough_control: torch.Tensor = _sanitize_tensor(batch["rough_control"]).clamp(0.0, 1.0)
        patient_img: torch.Tensor = _sanitize_tensor(batch["patient_image"]).clamp(0.0, 1.0)
        patient_mask: torch.Tensor = _sanitize_tensor(batch["patient_mask"]).clamp(0.0, 1.0)

        # D update
        with torch.no_grad():
            pred_tumor = self.generator(gen_input)
        # Binarize mask defensively to avoid fractional leakage
        control_mask_b = binarize_mask(control_mask)
        fake_full = control_img * (1 - control_mask_b) + pred_tumor * control_mask_b
        fake_combined = torch.cat([control_mask, fake_full], dim=1)
        real_combined = torch.cat([patient_mask, patient_img], dim=1)

        fake_pred = _sanitize_preds(self.discriminator(fake_combined))
        real_pred = _sanitize_preds(self.discriminator(real_combined))

        d_loss_fake = sum(F.relu(1.0 + fp[-1]).mean() for fp in fake_pred) / max(1, len(fake_pred))
        d_loss_real = sum(F.relu(1.0 - rp[-1]).mean() for rp in real_pred) / max(1, len(real_pred))
        d_loss = 0.5 * (d_loss_fake + d_loss_real)
        if not _is_finite(d_loss):
            self.log("warn/d_loss_nan", 1.0, on_step=True)
            return
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        opt_d.step()
        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # G update
        if self._d_updates % self.g_steps == 0:
            pred_tumor = self.generator(gen_input)
            control_mask_b = binarize_mask(control_mask)
            fake_full = (control_img * (1 - control_mask_b) + pred_tumor * control_mask_b).clamp(0.0, 1.0)
            pred_fake = _sanitize_preds(self.discriminator(torch.cat([control_mask, fake_full], dim=1)))

            # GAN loss (main driver for realism)
            g_gan = -sum(p[-1].mean() for p in pred_fake) / max(1, len(pred_fake))

            # Background identity loss (preserve control background outside mask)
            g_identity = self.criterion_l1(fake_full * (1 - control_mask_b), control_img * (1 - control_mask_b))

            # Edge consistency on boundary ring (smooth transitions)
            boundary_ring = create_boundary_ring(control_mask_b, ring_width=3)
            fake_edges = sobel_edges(fake_full)
            control_edges = sobel_edges(control_img)
            g_edge = self.criterion_l1(fake_edges * boundary_ring, control_edges * boundary_ring)

            # Style/texture loss for realistic tumor appearance (not tied to rough input)
            fake_tumor_region = fake_full * control_mask_b
            # Encourage texture variation for realism without tying to rough input
            fake_tumor_std = masked_std(fake_tumor_region, control_mask_b)
            target_std = torch.full_like(fake_tumor_std, 0.08)  # Slightly conservative texture target
            g_style = F.mse_loss(fake_tumor_std, target_std)

            g_loss = (
                g_gan
                + self.hparams.lambda_identity * g_identity
                + self.hparams.lambda_edge * g_edge
                + self.hparams.lambda_style * g_style
            )

            opt_g.zero_grad(set_to_none=True)
            if not _is_finite(g_loss):
                self.log("warn/g_loss_nan", 1.0, on_step=True)
                return
            self.manual_backward(g_loss)
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            opt_g.step()

            self.log_dict({"g_loss": g_loss, "g_gan": g_gan, "g_identity": g_identity, "g_edge": g_edge, "g_style": g_style}, prog_bar=True, on_step=True)

        self._d_updates += 1

