import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class SelfAttention(nn.Module):
    """Memory-efficient channel self-attention (same as in AttentionUNet)."""

    def __init__(self, in_channels: int, reduction_ratio: int = 8, drop_rate: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        inter_channels = max(in_channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # DropBlock for regularization with small hard/control dataset
        self.drop_block = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        attended = self.gamma * x * attention + x
        
        # Apply DropBlock during training for regularization  
        return self.drop_block(attended)


class CrossAttentionBlock(nn.Module):
    """Very light cross-attention operating at bottleneck resolution. Queries come from *x*,
    keys/values come from *ctx* (e.g. down-sampled nnUNet logits)."""

    def __init__(self, channels: int, context_channels: int | None = None, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        if context_channels is None:
            context_channels = channels
        
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Use GroupNorm for query, InstanceNorm for KV (more explicit than GroupNorm(1))
        self.norm_q = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.norm_kv = nn.InstanceNorm2d(channels, affine=True)
        
        # Learnable context projection instead of hard-coded averaging
        self.ctx_proj = nn.Conv2d(context_channels, channels, 1) if context_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) bottleneck features
        # ctx: (B,C_ctx,Hc,Wc) context features (e.g. logits)
        B, C, H, W = x.shape
        
        # Apply normalization in spatial format, then reshape
        x_norm = self.norm_q(x)
        q = x_norm.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Downsample ctx to same spatial size as x, then project to same channel dimension
        ctx_ds = F.adaptive_avg_pool2d(ctx, (H, W))  # B, C_ctx, H, W
        ctx_projected = self.ctx_proj(ctx_ds)  # Learnable projection
            
        ctx_norm = self.norm_kv(ctx_projected)
        kv = ctx_norm.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Wrap MHA in autocast(enabled=False) for AMP stability
        with torch.amp.autocast('cuda', enabled=False):
            q_float = q.float()
            kv_float = kv.float()
            out, _ = self.mha(q_float, kv_float, kv_float)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return out + x


class AttentionGate(nn.Module):
    """Spatial attention on skip connections (same implementation as AttentionUNet)."""

    def __init__(self, encoder_channels: int, decoder_channels: int):
        super().__init__()
        inter_channels = encoder_channels // 2
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention_conv = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        size = encoder_feat.size()[2:]
        decoder_upsampled = F.interpolate(decoder_feat, size=size, mode="bilinear", align_corners=False)
        enc_processed = self.encoder_conv(encoder_feat)
        dec_processed = self.decoder_conv(decoder_upsampled)
        coeffs = self.attention_conv(F.relu(enc_processed + dec_processed))
        return encoder_feat * coeffs


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = x if self.res_conv is None else self.res_conv(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + residual)


class SurfaceLoss(nn.Module):
    """Optimized Surface loss using boundary weighting for fast training."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted probabilities (B, C, H, W)  
            target: binary ground truth (B, C, H, W)
        """
        # Simple but effective boundary-weighted loss
        # Use Sobel edge detection for boundary identification
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)
        
        # Apply to each channel separately
        boundaries = torch.zeros_like(target)
        for c in range(target.shape[1]):
            target_c = target[:, c:c+1]  # Keep channel dimension
            
            # Sobel edge detection
            edges_x = F.conv2d(target_c, sobel_x.view(1, 1, 3, 3), padding=1)
            edges_y = F.conv2d(target_c, sobel_y.view(1, 1, 3, 3), padding=1)
            edges = torch.sqrt(edges_x**2 + edges_y**2)
            boundaries[:, c:c+1] = edges
        
        # Normalize boundaries and create weights
        boundaries = torch.sigmoid(boundaries * 10)  # Sharp boundary weighting
        boundary_weight = 1.0 + 3.0 * boundaries  # 1x to 4x weighting
        
        # Weighted BCE loss focusing on boundaries
        pred_clamp = torch.clamp(pred, min=1e-7, max=1-1e-7)
        surface_loss = -torch.mean(
            boundary_weight * (
                target * torch.log(pred_clamp) + 
                (1 - target) * torch.log(1 - pred_clamp)
            )
        )
        
        return surface_loss


class RefinerUNet(BaseModel):
    """4-level residual U-Net with attention gates and cross-attention in the bottleneck.

    The model predicts a *delta* that is **added** to the incoming nnUNet logits.
    The final segmentation returned by `forward` is `sigmoid(logits + delta)` so it
    can be trained with standard Dice/BCE losses from BaseModel.
    
    Handles variable height images (300-1000px) by padding to make dimensions
    divisible by 16 (4 downsampling levels). Pads with white (1.0) on bottom and
    equally on left/right sides if needed.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        base_channels: int = 32,
        cross_attention_heads: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.smooth = 1e-6  # Smoothing factor for dice calculations

        # ---------- Encoder ----------
        self.enc1 = ResidualConvBlock(in_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)

        # Self-attention on deeper encoders
        self.sa3 = SelfAttention(base_channels * 4)
        self.sa4 = SelfAttention(base_channels * 8)

        # Pool / upsample
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # ---------- Bottleneck ----------
        self.bottleneck = ResidualConvBlock(base_channels * 8, base_channels * 16)
        self.cross_attn1 = CrossAttentionBlock(base_channels * 16, context_channels=num_classes, num_heads=cross_attention_heads)
        self.cross_attn2 = CrossAttentionBlock(base_channels * 16, context_channels=num_classes, num_heads=cross_attention_heads)

        # ---------- Attention gates ----------
        self.ag4 = AttentionGate(base_channels * 8, base_channels * 16)
        self.ag3 = AttentionGate(base_channels * 4, base_channels * 8)
        self.ag2 = AttentionGate(base_channels * 2, base_channels * 4)
        self.ag1 = AttentionGate(base_channels, base_channels * 2)

        # ---------- Decoder ----------
        self.dec4 = ResidualConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = ResidualConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ResidualConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ResidualConvBlock(base_channels * 2 + base_channels, base_channels)

        # Output delta
        self.delta_conv = nn.Conv2d(base_channels, num_classes, 1)
        
        # Custom loss functions
        self.surface_loss = SurfaceLoss()

    # ------------------------ Helpers ------------------------
    def _split_logits_and_rest(self, x: torch.Tensor):
        """Assume the first `self.num_classes` channels are nnUNet logits."""
        logits = x[:, : self.num_classes]
        rest = x[:, self.num_classes :]
        return logits, rest

    def _pad_to_divisible(self, x: torch.Tensor, divisor: int = 16) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Pad input to make height and width divisible by divisor.
        
        Args:
            x: Input tensor (B, C, H, W)
            divisor: Target divisor (default 16 for 4 downsampling levels)
            
        Returns:
            Padded tensor and padding values (left, right, top, bottom)
        """
        _, _, h, w = x.shape
        
        # Calculate padding needed
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        
        # Pad height on bottom only
        pad_top = 0
        pad_bottom = pad_h
        
        # Pad width equally on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left  # Handle odd padding
        
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        if any(padding):
            # Pad with white (1.0) for all channels
            x = F.pad(x, padding, mode='constant', value=1.0)
            
        return x, padding

    def _unpad(self, x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
        """Remove padding from output tensor.
        
        Args:
            x: Padded tensor (B, C, H, W)
            padding: Padding values (left, right, top, bottom)
            
        Returns:
            Unpadded tensor
        """
        left, right, top, bottom = padding
        
        # Process height first, then width to avoid double-slicing issues
        if top > 0 or bottom > 0:
            h_end = x.shape[2] - bottom if bottom > 0 else x.shape[2]
            h_start = top
            x = x[:, :, h_start:h_end, :]
            
        if left > 0 or right > 0:
            w_end = x.shape[3] - right if right > 0 else x.shape[3]
            w_start = left
            x = x[:, :, :, w_start:w_end]
            
        return x

    # ------------------------ Forward ------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to make dimensions divisible by 16
        x_padded, padding = self._pad_to_divisible(x)
        
        # Split logits so we can reuse later
        logits, _ = self._split_logits_and_rest(x_padded)

        # Encoder
        e1 = self.enc1(x_padded)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e3 = self.sa3(e3)
        e4 = self.enc4(self.pool(e3))
        e4 = self.sa4(e4)

        # Bottleneck with cross-attention (context: downsampled logits)
        b = self.pool(e4)
        b = self.bottleneck(b)
        b = self.cross_attn1(b, logits)
        b = self.cross_attn2(b, logits)

        # Decoder with attention gated skips
        d4 = self.upsample(b)
        d4 = self.dec4(torch.cat([d4, self.ag4(e4, b)], dim=1))

        d3 = self.upsample(d4)
        d3 = self.dec3(torch.cat([d3, self.ag3(e3, d4)], dim=1))

        d2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d2, self.ag2(e2, d3)], dim=1))

        d1 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d1, self.ag1(e1, d2)], dim=1))

        delta = self.delta_conv(d1)  # residual correction logits
        refined_logits = logits + delta
        refined_output = torch.sigmoid(refined_logits)
        
        # Remove padding to return to original dimensions
        refined_logits_unpadded = self._unpad(refined_logits, padding)
        refined_output_unpadded = self._unpad(refined_output, padding)
        
        # Return both logits and probabilities for later stages
        return refined_logits_unpadded, refined_output_unpadded

    def _calculate_custom_loss(self, pred: torch.Tensor, target: torch.Tensor, prefix: str = "train"):
        """Calculate custom loss: DiceLoss + 0.5*BCE + 0.1*SurfaceLoss"""
        
        # Get loss weights from configuration (will be passed via hparams)
        dice_weight = self.hparams.get('dice_weight', 1.0)
        bce_weight = self.hparams.get('bce_weight', 0.5) 
        surface_weight = self.hparams.get('surface_weight', 0.1)
        
        # Calculate individual losses
        dice_loss = self._calculate_dice_loss(pred, target)
        # Use autocast context to handle BCE safely
        with torch.amp.autocast('cuda', enabled=False):
            pred_float = pred.float()
            target_float = target.float()
            bce_loss = F.binary_cross_entropy(pred_float, target_float)
        surface_loss = self.surface_loss(pred, target)
        
        # Combined loss
        total_loss = dice_weight * dice_loss + bce_weight * bce_loss + surface_weight * surface_loss
        
        # Log individual components
        batch_size = target.size(0)
        self.log(f"{prefix}_dice_loss", dice_loss, batch_size=batch_size)
        self.log(f"{prefix}_bce_loss", bce_loss, batch_size=batch_size)
        self.log(f"{prefix}_surface_loss", surface_loss, batch_size=batch_size)
        self.log(f"{prefix}_total_loss", total_loss, batch_size=batch_size, prog_bar=True)
        
        # Calculate TUMOR-ONLY dice metrics (NO BACKGROUND!)
        pred_binary = (pred > 0.5).float()
        
        # TUMOR-ONLY dice (the ONLY metric we care about)
        tumor_dice = self._calculate_tumor_dice_score(pred_binary, target)
        
        # Log ONLY tumor dice - NO background dice anywhere!
        self.log(f"{prefix}_dice", tumor_dice, batch_size=batch_size, prog_bar=True, 
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_tumor_dice", tumor_dice, batch_size=batch_size, 
                 on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        """Training step with custom loss and gradient clipping"""
        images, masks = batch
        logits, probs = self(images)  # Now returns both
        # Crop masks to match prediction size (handle padding mismatch)
        masks_cropped = self._crop_to_match(masks, probs)
        
        # Calculate loss using probabilities
        loss = self._calculate_custom_loss(probs, masks_cropped, "train")
        
        # Manual gradient clipping for MHA stability
        if self.automatic_optimization:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step with custom loss"""
        images, masks = batch
        logits, probs = self(images)  # Now returns both
        # Crop masks to match prediction size (handle padding mismatch)
        masks_cropped = self._crop_to_match(masks, probs)
        return self._calculate_custom_loss(probs, masks_cropped, "val")
    
    def _crop_to_match(self, masks: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Crop masks to match prediction dimensions and convert to same channel format"""
        # First crop spatially if needed
        target_h, target_w = pred.shape[2], pred.shape[3]
        masks_cropped = masks[:, :, :target_h, :target_w]
        
        # Convert single-channel mask to multi-channel one-hot if needed
        if masks_cropped.shape[1] != pred.shape[1]:
            # Assuming masks are single channel binary (0/1)
            # Convert to one-hot: [B, 1, H, W] -> [B, 2, H, W]
            batch_size = masks_cropped.shape[0]
            h, w = masks_cropped.shape[2], masks_cropped.shape[3]
            
            # Create one-hot encoding
            masks_one_hot = torch.zeros(batch_size, 2, h, w, device=masks_cropped.device, dtype=masks_cropped.dtype)
            masks_one_hot[:, 0] = (masks_cropped[:, 0] == 0).float()  # Background channel
            masks_one_hot[:, 1] = (masks_cropped[:, 0] == 1).float()  # Tumor channel
            
            return masks_one_hot
        
        return masks_cropped

    def _calculate_tumor_dice_score(self, pred_binary, target):
        """Calculate Dice score for tumor channel only (channel 1)."""
        if pred_binary.shape[1] == 2 and target.shape[1] == 2:
            # Extract tumor channel (channel 1)
            pred_tumor = pred_binary[:, 1:2]  # Keep channel dimension [B, 1, H, W]
            target_tumor = target[:, 1:2]     # Keep channel dimension [B, 1, H, W]
            
            # Use the same dice calculation as base model but only on tumor channel
            dice_scores = self._calculate_dice_per_sample(pred_tumor, target_tumor, smooth=0)
            return torch.stack(dice_scores).mean()
        else:
            # Fallback for single channel case
            dice_scores = self._calculate_dice_per_sample(pred_binary, target, smooth=0)
            return torch.stack(dice_scores).mean()
    
    def _calculate_dice_loss(self, pred, target):
        """Override dice loss to use ONLY tumor channel dice - NO BACKGROUND!"""
        if pred.shape[1] == 2 and target.shape[1] == 2:
            # Extract ONLY tumor channel (channel 1) - IGNORE background completely
            pred_tumor = pred[:, 1:2]   # [B, 1, H, W]
            target_tumor = target[:, 1:2] # [B, 1, H, W]
            
            # Calculate tumor-ONLY dice loss
            tumor_dice_scores = self._calculate_dice_per_sample(pred_tumor, target_tumor, smooth=self.smooth)
            tumor_dice_losses = [1 - dice for dice in tumor_dice_scores]
            return torch.stack(tumor_dice_losses).mean()
        else:
            # Fallback for single channel case
            return super()._calculate_dice_loss(pred, target)

    # Remove the custom dice loss override - use parent implementation
    # The loss function should remain: L = DiceLoss + 0.5*BCE + 0.1*SurfaceLoss
    # Only the METRIC calculation (dice score) should focus on tumor channel

    def configure_optimizers(self):
        """Configure AdamW optimizer with CosineAnnealingLR and warmup"""
        lr = self.hparams.get('lr', 1e-3)
        weight_decay = self.hparams.get('weight_decay', 1e-5)
        warmup_iters = self.hparams.get('warmup_iters', 500)  # Increased from 10 to 500 steps
        max_epochs = self.hparams.get('max_epochs', 100)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Calculate total iterations for cosine schedule
        # This is approximate - will be corrected by trainer
        total_iters = max_epochs * 100  # rough estimate
        
        # Combined warmup + cosine annealing scheduler
        def combined_lambda(step):
            if step < warmup_iters:
                # Linear warmup
                return float(step) / float(max(1, warmup_iters))
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
            
        combined_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, combined_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr-AdamW",
            }
        }
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override to ensure correct optimizer.step() -> scheduler.step() order"""
        # Step optimizer first
        optimizer.step(closure=optimizer_closure)
        # Then step scheduler (Lightning will call this automatically after optimizer_step)
        # This fixes the "lr_scheduler.step() before optimizer.step()" warning
