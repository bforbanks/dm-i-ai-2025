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

try:  # Optional FFT loss
    from kornia.losses import FFTConsistencyLoss  # type: ignore
except ImportError:  # pragma: no cover
    FFTConsistencyLoss = None  # noqa: N816


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
    """A small U-Net that predicts an in-mask residual refinement.

    Inputs: [initial_fake, real_image, mask] -> 3 channels
    Output: residual delta (1 channel), which will be multiplied by mask externally
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        in_ch = 3
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

        # Restrict magnitude via tanh to keep refinement conservative
        return torch.tanh(self.out_conv(d1)) * 0.2


def erode_binary_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Binary erosion using max-pool trick.

    mask: (B, 1, H, W) in {0,1}
    returns eroded mask with same shape.
    """
    pad = kernel_size // 2
    # Erosion via min-pool == 1 - maxpool(1 - mask)
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)


def sobel_gradients(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients in x and y for a single-channel image tensor.

    Returns (gx, gy), each shape (B, 1, H, W)
    """
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return gx, gy


class SPADERefinerModule(pl.LightningModule):
    """Freeze SPADE generator, train a small refiner UNet to improve blending.

    Final composition strictly modifies only within the mask:
        refined = clamp(initial_fake + delta * mask, 0, 1)
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_refiner: float = 2e-4,
        lambda_l1: float = 50.0,
        lambda_fft: float = 2.0,
        lambda_grad: float = 5.0,
        residual_scale: float = 0.3,
        freeze_generator: bool = True,
        gen_ckpt_path: str | None = None,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.residual_scale = residual_scale
        self.freeze_generator = freeze_generator
        self.gen_ckpt_path = gen_ckpt_path

        # Build after dataset to know sizes
        self.builder: GANBuilder | None = None
        self.generator: nn.Module | None = None
        self.refiner = MaskedRefinerUNet(base_ch=32)
        self.gan_kwargs = gan_kwargs

        # Losses
        self.criterion_l1 = nn.L1Loss()
        self.criterion_fft = FFTConsistencyLoss(reduction="mean") if FFTConsistencyLoss else None

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
                self.generator, _ = self.builder.build()
                if self.freeze_generator:
                    for p in self.generator.parameters():
                        p.requires_grad = False

                # Optional: load generator weights from checkpoint
                if self.gen_ckpt_path:
                    self._try_load_generator_weights(self.gen_ckpt_path)

    def _try_load_generator_weights(self, ckpt_path: str) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to load checkpoint at {ckpt_path}: {e}")
            return

        state_dict: Dict[str, torch.Tensor] | None = None
        # Try Lightning-style checkpoint
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt  # assume plain state dict

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
        # As a fallback, if keys look like they match directly
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
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.Adam(self.refiner.parameters(), lr=self.hparams.lr_refiner, betas=(0.0, 0.999))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        real_img: torch.Tensor = batch["real_image"]  # (B,1,H,W)
        mask: torch.Tensor = batch["mask"]            # (B,1,H,W)
        gen_input: torch.Tensor = batch["gen_input"]  # (B,2,H,W)

        with torch.no_grad():
            pred_residual = self.generator(gen_input) * self.residual_scale
            initial_fake = torch.clamp(real_img + pred_residual * mask, 0.0, 1.0)

        # Refiner input: [initial_fake, real_img, mask]
        refiner_in = torch.cat([initial_fake, real_img, mask], dim=1)
        delta = self.refiner(refiner_in)
        refined = torch.clamp(initial_fake + delta * mask, 0.0, 1.0)

        # Losses (all constrained to mask)
        l1 = self.criterion_l1(refined * mask, real_img * mask)

        if self.criterion_fft is not None:
            fft = self.criterion_fft(refined * mask, real_img * mask)
        else:
            fft = torch.tensor(0.0, device=self.device)

        # Boundary gradient loss (inner 1px band)
        eroded = erode_binary_mask(mask, kernel_size=3)
        inner_band = (mask - eroded).clamp(min=0.0, max=1.0)
        gx_r, gy_r = sobel_gradients(refined)
        gx_t, gy_t = sobel_gradients(real_img)
        grad_loss = (F.l1_loss(gx_r * inner_band, gx_t * inner_band) + F.l1_loss(gy_r * inner_band, gy_t * inner_band)) * 0.5

        loss = self.hparams.lambda_l1 * l1 + self.hparams.lambda_fft * fft + self.hparams.lambda_grad * grad_loss

        self.log_dict({
            "loss": loss,
            "l1": l1,
            "fft": fft,
            "grad": grad_loss,
        }, prog_bar=True, on_step=True)

        return loss

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine_full_image(
        self,
        patient_img_path: Path,
        mask_path: Path,
        control_imgs: List[Path],
        device: str = "cuda",
    ) -> torch.Tensor:
        """One-off inference mirroring the generator's inpaint, then refine."""
        # Reuse SPADEFullInpaintGANModule.inpaint_full_image logic at a minimum
        from PIL import Image
        import numpy as np

        patient_img = np.array(Image.open(patient_img_path).convert("L"), dtype=np.float32) / 255.0
        mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        original_h, original_w = patient_img.shape

        # Ensure mask matches patient size
        if mask_img.shape != patient_img.shape:
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(patient_img.shape[::-1], Image.NEAREST)
            mask_img = np.array(mask_pil, dtype=np.float32) / 255.0

        # Pad to training dimensions if required
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

        # Generate and refine
        self.generator.eval()
        self.refiner.eval()
        pred_residual = self.generator(gen_input) * self.residual_scale
        initial_fake = torch.clamp(patient_tensor + pred_residual * mask_tensor, 0.0, 1.0)
        refiner_in = torch.cat([initial_fake, patient_tensor, mask_tensor], dim=1)
        delta = self.refiner(refiner_in)
        refined = torch.clamp(initial_fake + delta * mask_tensor, 0.0, 1.0)

        # Crop back to original
        refined_np = refined[0, 0].detach().cpu().numpy()
        if refined_np.shape != (original_h, original_w):
            target_w, target_h = self.train_dataset.target_w, self.train_dataset.target_h
            h, w = original_h, original_w
            pad_left = (target_w - w) // 2
            pad_top = 0
            refined_np = refined_np[pad_top:pad_top + h, pad_left:pad_left + w]
        return torch.from_numpy(refined_np)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Save comparison plots (control, initial fake, refined, mask overlay)."""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        vis_dir = Path(self.logger.log_dir) / "refiner_vis"
        os.makedirs(vis_dir, exist_ok=True)

        control_imgs = list((self.data_root / "controls" / "imgs").glob("*.png"))
        patient_pairs = [(p, m) for p, m in self.train_dataset.patient_pairs]

        self.generator.eval()
        self.refiner.eval()
        with torch.no_grad():
            for i in range(min(5, len(control_imgs))):
                control_img_path = control_imgs[i]
                _, mask_path = patient_pairs[i % len(patient_pairs)]

                # Build paths and arrays
                control_orig = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
                mask_orig = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

                # Ensure sizes match
                if mask_orig.shape != control_orig.shape:
                    mask_pil = Image.fromarray((mask_orig * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(control_orig.shape[::-1], Image.NEAREST)
                    mask_orig = np.array(mask_pil, dtype=np.float32) / 255.0

                # Pad to train dims
                if hasattr(self.train_dataset, "target_w") and self.train_dataset.target_w:
                    target_w, target_h = self.train_dataset.target_w, self.train_dataset.target_h
                    if control_orig.shape != (target_h, target_w):
                        h, w = control_orig.shape
                        pad_left = (target_w - w) // 2
                        pad_top = 0
                        canvas_img = np.ones((target_h, target_w), dtype=np.float32)
                        canvas_img[pad_top:pad_top + h, pad_left:pad_left + w] = control_orig
                        control_pad = canvas_img

                        canvas_mask = np.zeros((target_h, target_w), dtype=np.float32)
                        canvas_mask[pad_top:pad_top + h, pad_left:pad_left + w] = mask_orig
                        mask_pad = canvas_mask
                    else:
                        control_pad = control_orig
                        mask_pad = mask_orig
                else:
                    control_pad = control_orig
                    mask_pad = mask_orig

                control_tensor = torch.from_numpy(control_pad).unsqueeze(0).unsqueeze(0).to(self.device)
                mask_tensor = torch.from_numpy(mask_pad).unsqueeze(0).unsqueeze(0).to(self.device)
                # Use control as patient canvas (as in generator visualisation)
                patient_tensor = control_tensor.clone()
                gen_input = torch.cat([control_tensor, mask_tensor], dim=1)

                pred_residual = self.generator(gen_input) * self.residual_scale
                initial_fake = torch.clamp(patient_tensor + pred_residual * mask_tensor, 0.0, 1.0)

                refiner_in = torch.cat([initial_fake, patient_tensor, mask_tensor], dim=1)
                delta = self.refiner(refiner_in)
                refined = torch.clamp(initial_fake + delta * mask_tensor, 0.0, 1.0)

                initial_np = initial_fake[0, 0].detach().cpu().numpy()
                refined_np = refined[0, 0].detach().cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                axes[0].imshow(control_pad, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Control"); axes[0].axis("off")
                axes[1].imshow(initial_np, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Initial Fake"); axes[1].axis("off")
                axes[2].imshow(refined_np, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Refined"); axes[2].axis("off")
                axes[3].imshow(control_pad, cmap="gray", vmin=0, vmax=1)
                axes[3].imshow(mask_pad, cmap="Reds", alpha=0.4, vmin=0, vmax=1); axes[3].set_title("Mask overlay"); axes[3].axis("off")
                plt.tight_layout()
                plt.savefig(vis_dir / f"refiner_comparison_{i}.png", dpi=150, bbox_inches="tight")
                plt.close()

        print(f"Refiner visualizations saved to: {vis_dir}")

