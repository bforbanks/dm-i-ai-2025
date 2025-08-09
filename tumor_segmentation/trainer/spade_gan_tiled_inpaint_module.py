from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PIL import Image

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

from datasets.tiled_synthetic_tumor_dataset import TiledSyntheticTumorSliceDataset, TiledInferenceDataset  # noqa: E402
from models.gan_components import GANBuilder  # noqa: E402

# Custom FFT loss implementation
def perceptual_loss(pred: torch.Tensor, target: torch.Tensor, vgg_features) -> torch.Tensor:
    """Compute perceptual loss using VGG features.
    
    Args:
        pred: Predicted image tensor (B, 1, H, W)
        target: Target image tensor (B, 1, H, W)
        vgg_features: VGG feature extractor
        
    Returns:
        Perceptual loss
    """
    if vgg_features is None:
        return torch.tensor(0.0, device=pred.device)
    
    # Convert grayscale to RGB by repeating channels
    pred_rgb = pred.repeat(1, 3, 1, 1)
    target_rgb = target.repeat(1, 3, 1, 1)
    
    # Extract features
    pred_features = vgg_features(pred_rgb)
    target_features = vgg_features(target_rgb)
    
    # L2 loss between features
    perceptual_loss = F.mse_loss(pred_features, target_features)
    
    return perceptual_loss


def fft_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute FFT consistency loss between predicted and target images.
    
    Args:
        pred: Predicted image tensor (B, C, H, W)
        target: Target image tensor (B, C, H, W)
        
    Returns:
        FFT consistency loss
    """
    # Compute 2D FFT for both images
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    
    # Compute magnitude spectrum
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    # L1 loss between magnitude spectra
    fft_loss = F.l1_loss(pred_mag, target_mag)
    
    return fft_loss


class SPADETiledInpaintGANModule(pl.LightningModule):
    """SPADE GAN variant that performs tiled inpainting for high-resolution images.

    Key features:
    - Trains on random tiles containing tumors (efficient memory usage)
    - Only generates tumor pixels, preserves patient background
    - Can handle arbitrary image resolutions at inference time
    - Seamless reconstruction through overlapping tile processing
    
    Training strategy:
    - Extract random tiles with sufficient tumor content
    - Train generator to inpaint tumor regions only
    - Use patient background + generated tumor for final output
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 8,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        lambda_l1: float = 100.0,
        lambda_fft: float = 10.0,
        lambda_perceptual: float = 0.0,
        g_steps: int = 1,
        tile_size: int = 256,
        **gan_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["gan_kwargs"])

        # Store tiling parameters
        self.data_root = Path(data_root)
        self.tile_size = tile_size

        # Set tile size for GAN builder
        gan_kwargs.update({
            "crop_size": tile_size,
            "crop_size_w": tile_size,
            "crop_size_h": tile_size,
        })

        # Networks will be built in setup()
        self.builder: GANBuilder | None = None
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self.gan_kwargs = gan_kwargs

        # Losses
        self.criterion_l1 = nn.L1Loss()
        # FFT loss will be computed using our custom function
        
        # Perceptual loss using VGG features (if enabled)
        if self.hparams.lambda_perceptual > 0:
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features[:16]  # Up to conv3_3
                self.vgg_features = vgg.eval()
                for param in self.vgg_features.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: torchvision not available, perceptual loss disabled")
                self.vgg_features = None
        else:
            self.vgg_features = None

        self.batch_size = batch_size

        # Manual optimization
        self.automatic_optimization = False
        self.g_steps = g_steps
        self._d_updates = 0

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = TiledSyntheticTumorSliceDataset(
                self.data_root,
                tile_size=self.tile_size,
            )

            # Build networks now that we know tile size
            if self.builder is None:
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
    # Optimizers
    # ---------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.lr_g, betas=(0.5, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr_d, betas=(0.5, 0.999)
        )
        
        # Add gentle learning rate schedulers after warmup
        scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.5)
        scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=50, gamma=0.5)
        
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    # ---------------------------------------------------------------------
    # Training step (manual optimization)
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_g, opt_d = self.optimizers()

        real_img: torch.Tensor = batch["real_image"]  # (B,1,H,W) - patient tile
        control: torch.Tensor = batch["control"]  # (B,1,H,W) - control tile
        mask: torch.Tensor = batch["mask"]  # (B,1,H,W) - mask tile
        gen_input: torch.Tensor = batch["gen_input"]  # (B,2,H,W) - control+mask

        # --------------------------------------------------
        # Train Discriminator
        # --------------------------------------------------
        with torch.no_grad():
            pred_tumour = self.generator(gen_input)  # (B,1,H,W)

        # Use patient tile as background, inpaint only tumor region
        patient_background = real_img * (1 - mask)  # Patient tissue outside tumor
        fake_full = patient_background + pred_tumour * mask  # Patient background + generated tumor
        real_full = real_img  # Original patient tile (ground truth)

        fake_combined = torch.cat([mask, fake_full], dim=1)
        real_combined = torch.cat([mask, real_full], dim=1)

        fake_pred = self.discriminator(fake_combined)
        real_pred = self.discriminator(real_combined)

        # Use standard GAN loss (discriminator sees full image context)
        d_loss_fake = sum(F.binary_cross_entropy_with_logits(fp[-1], torch.zeros_like(fp[-1])) for fp in fake_pred) / len(fake_pred)
        d_loss_real = sum(F.binary_cross_entropy_with_logits(rp[-1], torch.ones_like(rp[-1])) for rp in real_pred) / len(real_pred)
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

            # Use patient tile as background, inpaint only tumor region
            patient_background = real_img * (1 - mask)
            fake_full = patient_background + pred_tumour * mask
            pred_fake = self.discriminator(torch.cat([mask, fake_full], dim=1))

            # Use standard GAN loss for generator (discriminator provides global context)
            g_gan = sum(F.binary_cross_entropy_with_logits(p[-1], torch.ones_like(p[-1])) for p in pred_fake) / len(pred_fake)

            # Focus losses on tumor region only (most important part)
            g_l1 = self.criterion_l1(pred_tumour * mask, real_img * mask)
            
            # FFT loss - only compute on masked regions if lambda_fft > 0
            if self.hparams.lambda_fft > 0:
                # Apply mask to both images for FFT comparison
                masked_pred = pred_tumour * mask
                masked_real = real_img * mask
                g_fft = fft_consistency_loss(masked_pred, masked_real)
            else:
                g_fft = torch.tensor(0.0, device=self.device)
            
            # Perceptual loss - only compute on masked regions if lambda_perceptual > 0
            if self.hparams.lambda_perceptual > 0:
                masked_pred = pred_tumour * mask
                masked_real = real_img * mask
                g_perceptual = perceptual_loss(masked_pred, masked_real, self.vgg_features)
            else:
                g_perceptual = torch.tensor(0.0, device=self.device)

            g_loss = g_gan + self.hparams.lambda_l1 * g_l1 + self.hparams.lambda_fft * g_fft + self.hparams.lambda_perceptual * g_perceptual

            opt_g.zero_grad(set_to_none=True)
            self.manual_backward(g_loss)
            opt_g.step()

            self.log_dict(
                {
                    "g_loss": g_loss,
                    "g_gan": g_gan,
                    "g_l1": g_l1,
                    "g_fft": g_fft,
                    "g_perceptual": g_perceptual,
                },
                prog_bar=True,
                on_step=True,
            )

        self._d_updates += 1

    # ---------------------------------------------------------------------
    # High-resolution inference methods
    # ---------------------------------------------------------------------
    def inpaint_full_image(
        self,
        patient_img_path: Path,
        mask_path: Path,
        control_imgs: List[Path],
        tile_size: int = None,
        overlap: int = 32,
        device: str = "cuda",
    ) -> np.ndarray:
        """Perform tiled inpainting on a full high-resolution image.
        
        Args:
            patient_img_path: Path to patient image
            mask_path: Path to tumor mask
            control_imgs: List of control image paths for context
            tile_size: Tile size for processing (default: training tile size)
            overlap: Overlap between tiles for seamless blending
            device: Device to run inference on
            
        Returns:
            Inpainted image as numpy array
        """
        if tile_size is None:
            tile_size = self.tile_size

        # Create inference dataset
        inference_ds = TiledInferenceDataset(
            patient_img_path=patient_img_path,
            mask_path=mask_path,
            control_imgs=control_imgs,
            tile_size=tile_size,
            overlap=overlap,
        )

        # Load original image for reconstruction
        patient_img = np.array(Image.open(patient_img_path).convert("L"), dtype=np.float32) / 255.0
        mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        
        # Ensure mask has same shape as patient image
        if mask_img.shape != patient_img.shape:
            mask_pil = Image.fromarray((mask_img * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(patient_img.shape[::-1], Image.NEAREST)
            mask_img = np.array(mask_pil, dtype=np.float32) / 255.0
        
        # Initialize output arrays
        output_img = np.copy(patient_img)
        weight_map = np.zeros_like(patient_img)

        self.generator.eval()
        with torch.no_grad():
            for i in range(len(inference_ds)):
                sample = inference_ds[i]
                gen_input = sample["gen_input"].unsqueeze(0).to(device)
                mask_tile = sample["mask"].unsqueeze(0).to(device)
                patient_tile = sample["real_image"].unsqueeze(0).to(device)
                x, y, tile_w, tile_h = sample["tile_coords"]

                # Only process tiles with tumor content
                if mask_tile.sum() > 0:
                    pred_tumour = self.generator(gen_input)
                    patient_background = patient_tile * (1 - mask_tile)
                    inpainted_tile = patient_background + pred_tumour * mask_tile
                    
                    # Convert back to numpy and extract valid region
                    inpainted_np = inpainted_tile.cpu().numpy()[0, 0, :tile_h, :tile_w]
                    
                    # Create smooth blending window to avoid seams
                    win_y = np.hanning(tile_h) if tile_h > 1 else np.ones(tile_h)
                    win_x = np.hanning(tile_w) if tile_w > 1 else np.ones(tile_w)
                    window = np.outer(win_y, win_x)
                    
                    # Apply window
                    output_img[y:y+tile_h, x:x+tile_w] += inpainted_np * window
                    weight_map[y:y+tile_h, x:x+tile_w] += window

        # Normalize overlapping regions
        weight_map[weight_map == 0] = 1.0  # Avoid division by zero
        output_img = output_img / weight_map

        # Ensure output_img has the same shape as patient_img and mask_img
        if output_img.shape != patient_img.shape:
            # Resize output_img to match patient_img shape
            output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
            output_pil = output_pil.resize(patient_img.shape[::-1], Image.BILINEAR)
            output_img = np.array(output_pil, dtype=np.float32) / 255.0
        
        # Keep original pixels where no tumor was detected
        final_mask = mask_img > 0.5
        output_img = patient_img * (~final_mask) + output_img * final_mask

        self.generator.train()
        return output_img

    # ---------------------------------------------------------------------
    # Visualization helper
    # ---------------------------------------------------------------------
    def on_train_end(self) -> None:  # noqa: D401
        """Generate full control images with synthetic tumors and save comparison plots."""
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        vis_dir = Path(self.logger.log_dir) / "gan_full_image_vis"
        os.makedirs(vis_dir, exist_ok=True)

        # Get control images and some patient masks for demonstration
        control_imgs = list((self.data_root / "controls" / "imgs").glob("*.png"))
        patient_pairs = [(p, m) for p, m in self.train_dataset.patient_pairs]
        
        self.generator.eval()
        with torch.no_grad():
            for i in range(min(5, len(control_imgs))):
                control_img_path = control_imgs[i]
                # Use a random patient mask for demonstration
                patient_img_path, mask_path = patient_pairs[i % len(patient_pairs)]
                
                # Generate full synthetic image
                synthetic_img = self.inpaint_full_image(
                    patient_img_path=control_img_path,  # Use control as "patient" background
                    mask_path=mask_path,  # Apply patient's tumor mask
                    control_imgs=[control_img_path],  # Use same control for context
                    device=self.device,
                )
                
                # Load original images for comparison
                control_orig = np.array(Image.open(control_img_path).convert("L"), dtype=np.float32) / 255.0
                mask_orig = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
                
                # Resize mask to match control image if needed
                if mask_orig.shape != control_orig.shape:
                    mask_pil = Image.fromarray((mask_orig * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(control_orig.shape[::-1], Image.NEAREST)
                    mask_orig = np.array(mask_pil, dtype=np.float32) / 255.0
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original control image
                axes[0].imshow(control_orig, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original Control')
                axes[0].axis('off')
                
                # Generated image (control + synthetic tumor)
                axes[1].imshow(synthetic_img, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Control + Generated Tumor')
                axes[1].axis('off')
                
                # Tumor mask overlay
                axes[2].imshow(control_orig, cmap='gray', vmin=0, vmax=1)
                axes[2].imshow(mask_orig, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                axes[2].set_title('Tumor Mask Location')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"synthetic_tumor_comparison_{i}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()

        self.generator.train()
        print(f"Synthetic tumor visualizations saved to: {vis_dir}")