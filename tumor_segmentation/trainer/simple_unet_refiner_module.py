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


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleUNetRefiner(nn.Module):
    """Simple U-Net for tumor refinement.
    
    Input: [rough_control_with_tumor, control_mask, control_image] -> 3 channels
    Output: refined tumor region (1 channel) - a residual delta to add to rough input
    """
    
    def __init__(self, in_ch: int = 3, base_ch: int = 64) -> None:
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(base_ch * 2 + base_ch, base_ch)
        
        # Output layer - predict residual delta
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Decoder path with skip connections
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output residual delta, range [-1, 1]
        delta = torch.tanh(self.out_conv(d1))
        return delta


class PatchDiscriminator(nn.Module):
    """Simple PatchGAN discriminator for tumor realism."""
    
    def __init__(self, in_ch: int = 2, base_ch: int = 64) -> None:
        super().__init__()
        # Input: [mask, tumor_region] -> 2 channels
        layers = []
        
        # First layer without normalization
        layers.append(nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers with spectral norm
        nf_mult = 1
        for i in range(1, 4):  # 3 layers
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(base_ch * nf_mult_prev, base_ch * nf_mult, 4, stride=2, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(base_ch * nf_mult, 1, 4, stride=1, padding=1)
        ))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is strictly binary."""
    return (mask > 0.5).to(mask.dtype)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    edges_x = F.conv2d(x, sobel_x, padding=1)
    edges_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-8)


def get_bbox_from_mask(mask: torch.Tensor, pad: int = 16) -> tuple[int, int, int, int]:
    """Compute bbox (y0,x0,y1,x1) around 1-channel mask with padding."""
    _, h, w = mask.shape
    ys, xs = torch.where(mask[0] > 0.5)
    if ys.numel() == 0:
        return 0, 0, h, w
    y0 = max(int(ys.min().item()) - pad, 0)
    y1 = min(int(ys.max().item()) + pad + 1, h)
    x0 = max(int(xs.min().item()) - pad, 0)
    x1 = min(int(xs.max().item()) + pad + 1, w)
    return y0, x0, y1, x1


def resize_like(img: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)


class SimpleUNetRefinerModule(pl.LightningModule):
    """Simple U-Net refiner following pix2pix data flow but focused on adjustments.
    
    Takes rough control tumors and refines them using patient ground truth as reference.
    The key is adjustment rather than full reconstruction.
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_adv: float = 2.0,
        lambda_l1: float = 2.0,
        lambda_edge: float = 1.0,
        lambda_delta: float = 0.5,
        delta_scale: float = 0.6,
        crop_size: int = 128,
        g_steps: int = 1,
        control_rough_labels_dir: str = "controls/rough_labels",
        control_rough_tumors_dir: str = "controls/rough_tumors",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        
        # Networks
        self.generator = SimpleUNetRefiner(in_ch=3, base_ch=64)
        self.discriminator = PatchDiscriminator(in_ch=2, base_ch=64)
        
        # Loss
        self.criterion_l1 = nn.L1Loss()
        
        # Manual optimization
        self.automatic_optimization = False
        self.g_steps = g_steps
        self._d_updates = 0
        self.delta_scale = delta_scale
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = RefineTumorSliceDataset(
                self.data_root, 
                auto_pad=True,
                control_rough_labels_dir=self.hparams.control_rough_labels_dir,
                control_rough_tumors_dir=self.hparams.control_rough_tumors_dir
            )

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def configure_optimizers(self):  # type: ignore[override]
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.0, 0.999))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        # Extract batch data following pix2pix format
        control_img: torch.Tensor = batch["control_image"]      # (B,1,H,W)
        control_mask: torch.Tensor = batch["control_mask"]      # (B,1,H,W)
        rough_control: torch.Tensor = batch["rough_control"]    # (B,1,H,W)
        patient_img: torch.Tensor = batch["patient_image"]      # (B,1,H,W)
        patient_mask: torch.Tensor = batch["patient_mask"]      # (B,1,H,W)

        # Sanitize inputs
        control_img = torch.clamp(control_img, 0.0, 1.0)
        control_mask = torch.clamp(control_mask, 0.0, 1.0)
        rough_control = torch.clamp(rough_control, 0.0, 1.0)
        patient_img = torch.clamp(patient_img, 0.0, 1.0)
        patient_mask = torch.clamp(patient_mask, 0.0, 1.0)

        # -------------------- Train Discriminator --------------------
        with torch.no_grad():
            # Generator input: [rough_control, control_mask, control_image]
            gen_input = torch.cat([rough_control, control_mask, control_img], dim=1)
            delta = self.generator(gen_input)
            
            # Apply refinement only within mask
            control_mask_b = binarize_mask(control_mask)
            refined = torch.clamp(rough_control + self.delta_scale * delta * control_mask_b, 0.0, 1.0)
            
            # Build local crops around mask for stronger D supervision
            y0, x0, y1, x1 = get_bbox_from_mask(control_mask[0])
            fake_crop = refined[:, :, y0:y1, x0:x1]
            fake_mask_crop = control_mask[:, :, y0:y1, x0:x1]
            fake_crop = resize_like(fake_crop, self.crop_size)
            fake_mask_crop = F.interpolate(fake_mask_crop, size=(self.crop_size, self.crop_size), mode="nearest")
            fake_d_input = torch.cat([fake_mask_crop, fake_crop], dim=1)
        
        # Real discriminator input: [patient_mask, patient_img] (crop around patient mask)
        y0r, x0r, y1r, x1r = get_bbox_from_mask(patient_mask[0])
        real_crop = patient_img[:, :, y0r:y1r, x0r:x1r]
        real_mask_crop = patient_mask[:, :, y0r:y1r, x0r:x1r]
        real_crop = resize_like(real_crop, self.crop_size)
        real_mask_crop = F.interpolate(real_mask_crop, size=(self.crop_size, self.crop_size), mode="nearest")
        real_d_input = torch.cat([real_mask_crop, real_crop], dim=1)
        
        # Discriminator predictions
        pred_fake = self.discriminator(fake_d_input)
        pred_real = self.discriminator(real_d_input)
        
        # Hinge loss for discriminator
        d_loss_fake = F.relu(1.0 + pred_fake).mean()
        d_loss_real = F.relu(1.0 - pred_real).mean()
        d_loss = 0.5 * (d_loss_fake + d_loss_real)
        
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        opt_d.step()
        
        self.log("d_loss", d_loss, prog_bar=True, on_step=True)

        # -------------------- Train Generator --------------------
        if self._d_updates % self.g_steps == 0:
            gen_input = torch.cat([rough_control, control_mask, control_img], dim=1)
            delta = self.generator(gen_input)
            
            control_mask_b = binarize_mask(control_mask)
            refined = torch.clamp(rough_control + self.delta_scale * delta * control_mask_b, 0.0, 1.0)
            
            # Adversarial loss on local crop
            y0, x0, y1, x1 = get_bbox_from_mask(control_mask[0])
            fake_crop = refined[:, :, y0:y1, x0:x1]
            fake_mask_crop = control_mask[:, :, y0:y1, x0:x1]
            fake_crop = resize_like(fake_crop, self.crop_size)
            fake_mask_crop = F.interpolate(fake_mask_crop, size=(self.crop_size, self.crop_size), mode="nearest")
            fake_d_input = torch.cat([fake_mask_crop, fake_crop], dim=1)
            pred_fake_for_g = self.discriminator(fake_d_input)
            g_adv = -pred_fake_for_g.mean()
            
            # L1 loss between refined and rough (encourage small adjustments)
            g_l1 = self.criterion_l1(refined * control_mask_b, rough_control * control_mask_b)
            
            # Edge preservation loss (keep tumor boundaries smooth)
            refined_edges = sobel_edges(refined)
            rough_edges = sobel_edges(rough_control)
            g_edge = self.criterion_l1(refined_edges * control_mask_b, rough_edges * control_mask_b)
            
            # Delta magnitude penalty (encourage small deltas)
            g_delta = torch.mean((delta * control_mask_b) ** 2)
            
            # Combined loss (more adversarially driven)
            g_loss = (
                self.hparams.lambda_adv * g_adv +
                self.hparams.lambda_l1 * g_l1 +
                self.hparams.lambda_edge * g_edge +
                self.hparams.lambda_delta * g_delta
            )
            
            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(g_loss)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            opt_g.step()
            
            self.log_dict({
                "g_loss": g_loss,
                "g_adv": g_adv, 
                "g_l1": g_l1,
                "g_edge": g_edge,
                "g_delta": g_delta,
            }, prog_bar=True, on_step=True)

        self._d_updates += 1

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def on_train_end(self) -> None:
        """Save comparison plots showing refinement results."""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        vis_dir = Path(self.logger.log_dir) / "simple_unet_refiner_vis"
        os.makedirs(vis_dir, exist_ok=True)

        # Use first few samples from dataset
        self.generator.eval()
        with torch.no_grad():
            for i in range(min(5, len(self.train_dataset))):
                sample = self.train_dataset[i]
                
                # Move to device and add batch dimension
                control_img = sample["control_image"].unsqueeze(0).to(self.device)
                control_mask = sample["control_mask"].unsqueeze(0).to(self.device)
                rough_control = sample["rough_control"].unsqueeze(0).to(self.device)
                patient_img = sample["patient_image"].unsqueeze(0).to(self.device)
                
                # Generate refinement
                gen_input = torch.cat([rough_control, control_mask, control_img], dim=1)
                delta = self.generator(gen_input)
                control_mask_b = binarize_mask(control_mask)
                refined = torch.clamp(rough_control + self.delta_scale * delta * control_mask_b, 0.0, 1.0)
                
                # Convert to numpy for plotting
                control_np = control_img[0, 0].cpu().numpy()
                rough_np = rough_control[0, 0].cpu().numpy()
                refined_np = refined[0, 0].cpu().numpy()
                patient_np = patient_img[0, 0].cpu().numpy()
                mask_np = control_mask[0, 0].cpu().numpy()
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                axes[0].imshow(control_np, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Control")
                axes[0].axis("off")
                
                axes[1].imshow(rough_np, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Rough Tumor")
                axes[1].axis("off")
                
                axes[2].imshow(refined_np, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Refined Tumor")
                axes[2].axis("off")
                
                axes[3].imshow(patient_np, cmap="gray", vmin=0, vmax=1)
                axes[3].set_title("Patient Reference")
                axes[3].axis("off")
                
                axes[4].imshow(control_np, cmap="gray", vmin=0, vmax=1)
                axes[4].imshow(mask_np, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
                axes[4].set_title("Mask Overlay")
                axes[4].axis("off")
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"refiner_comparison_{i}.png", dpi=150, bbox_inches="tight")
                plt.close()

        print(f"Simple UNet refiner visualizations saved to: {vis_dir}")