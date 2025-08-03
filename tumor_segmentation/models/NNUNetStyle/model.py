import torch
import torch.nn as nn
import torch.nn.functional as F
from tumor_segmentation.models.base_model import BaseModel


class ChannelAttention(nn.Module):
    """
    Channel Attention mechanism to focus on important feature channels.
    Helps the model pay attention to channels that contain tumor-relevant information.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism to focus on important spatial regions.
    Helps the model concentrate on tumor regions in the image.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining channel and spatial attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 8, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with instance normalization and leaky ReLU.
    Helps with gradient flow and feature preservation.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion module to combine features at different scales.
    Helps capture both fine and coarse tumor details.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        # Multi-scale convolutions
        conv1 = self.conv1x1(x)
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        
        # Concatenate and fuse
        fused = torch.cat([conv1, conv3, conv5], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.norm(fused)
        fused = self.activation(fused)
        
        return fused


class DeepSupervision(nn.Module):
    """
    Deep supervision module for multi-scale predictions.
    Helps with gradient flow and provides supervision at multiple scales.
    """
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, target_size):
        x = self.conv(x)
        if x.size()[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)


class NNUNetStyle(BaseModel):
    """
    Enhanced NNUNet-style U-Net for tumor segmentation with advanced features:
    - Attention mechanisms (CBAM) for better feature focus
    - Residual connections for improved gradient flow
    - Multi-scale feature fusion for capturing different tumor scales
    - Deep supervision for multi-scale predictions
    - Deeper architecture with more sophisticated skip connections
    - Instance normalization and leaky ReLU activations
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,
        false_negative_penalty=2.0,
        false_negative_penalty_scheduler: dict | None = None,
        patient_weight: float = 2.0,
        control_weight: float = 0.5,
        base_channels: int = 32,
        depth: int = 5,  # Increased depth
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        use_residual: bool = True,
        use_multiscale: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight, 
            false_negative_penalty=false_negative_penalty, false_negative_penalty_scheduler=false_negative_penalty_scheduler,
            patient_weight=patient_weight, control_weight=control_weight
        )
        
        # Store configuration
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_multiscale = use_multiscale
        self.use_deep_supervision = use_deep_supervision
        self.depth = depth

        # Calculate channel dimensions
        self.channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else self.channels[i-1]
            out_ch = self.channels[i]
            
            if self.use_residual:
                block = ResidualBlock(in_ch, out_ch, dropout_rate)
            else:
                block = self._make_encoder_block(in_ch, out_ch)
            
            self.encoder_blocks.append(block)
            
            # Add attention after each encoder block
            if self.use_attention:
                self.encoder_blocks.append(CBAM(out_ch))

        # Bottleneck
        if self.use_residual:
            self.bottleneck = ResidualBlock(self.channels[depth-1], self.channels[depth], dropout_rate)
        else:
            self.bottleneck = self._make_encoder_block(self.channels[depth-1], self.channels[depth])
        
        if self.use_attention:
            self.bottleneck_attention = CBAM(self.channels[depth])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        for i in range(depth):
            in_ch = self.channels[depth-i] + self.channels[depth-i-1]  # Skip connection
            out_ch = self.channels[depth-i-1]
            
            if self.use_multiscale:
                block = MultiScaleFeatureFusion(in_ch, out_ch)
            elif self.use_residual:
                block = ResidualBlock(in_ch, out_ch, dropout_rate)
            else:
                block = self._make_decoder_block(in_ch, out_ch)
            
            self.decoder_blocks.append(block)
            
            # Add attention after each decoder block
            if self.use_attention:
                self.decoder_attentions.append(CBAM(out_ch))

        # Final output
        self.final_conv = nn.Conv2d(self.channels[0], num_classes, 1)

        # Deep supervision outputs
        if self.use_deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(depth - 1):  # Skip the final output
                self.deep_supervision_outputs.append(
                    DeepSupervision(self.channels[depth-i-1], num_classes)
                )

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an encoder block with NNUNet-style features"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout_rate),
        )

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a decoder block with NNUNet-style features"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(self.dropout_rate),
        )

    def forward(self, x):
        # Store original size for deep supervision
        original_size = x.size()[-2:]
        
        # Encoder path
        encoder_features = []
        current_x = x
        
        for i in range(self.depth):
            # Apply encoder block
            current_x = self.encoder_blocks[i * (2 if self.use_attention else 1)](current_x)
            
            # Apply attention if enabled
            if self.use_attention:
                current_x = self.encoder_blocks[i * 2 + 1](current_x)
            
            encoder_features.append(current_x)
            current_x = self.pool(current_x)

        # Bottleneck
        current_x = self.bottleneck(current_x)
        if self.use_attention:
            current_x = self.bottleneck_attention(current_x)

        # Decoder path with skip connections
        decoder_features = []
        for i in range(self.depth):
            # Upsample
            current_x = self.upsample(current_x)
            
            # Skip connection
            skip_features = encoder_features[-(i+1)]
            current_x = torch.cat([current_x, skip_features], dim=1)
            
            # Apply decoder block
            current_x = self.decoder_blocks[i](current_x)
            
            # Apply attention if enabled
            if self.use_attention:
                current_x = self.decoder_attentions[i](current_x)
            
            decoder_features.append(current_x)

        # Final output
        main_output = self.final_conv(current_x)
        main_output = torch.sigmoid(main_output)

        # Deep supervision outputs
        if self.use_deep_supervision and self.training:
            deep_outputs = []
            for i, decoder_feat in enumerate(decoder_features[:-1]):  # Skip the final decoder output
                deep_output = self.deep_supervision_outputs[i](decoder_feat, original_size)
                deep_outputs.append(deep_output)
            
            return main_output, deep_outputs
        
        return main_output

    def training_step(self, batch, batch_idx):
        """Override training step to handle deep supervision"""
        images, masks = batch
        outputs = self(images)
        
        if self.use_deep_supervision and isinstance(outputs, tuple):
            main_output, deep_outputs = outputs
            loss = self._calculate_and_log_metrics(main_output, masks, 'train')
            
            # Add deep supervision losses
            for i, deep_output in enumerate(deep_outputs):
                deep_loss = self._calculate_and_log_metrics(deep_output, masks, f'train_deep_{i}')
                loss += 0.1 * deep_loss  # Weight deep supervision losses
            
            return loss
        else:
            return self._calculate_and_log_metrics(outputs, masks, 'train')

    def validation_step(self, batch, batch_idx):
        """Override validation step to handle deep supervision"""
        images, masks = batch
        outputs = self(images)
        
        if self.use_deep_supervision and isinstance(outputs, tuple):
            main_output, _ = outputs  # Only use main output for validation metrics
            return self._calculate_and_log_metrics(main_output, masks, 'val')
        else:
            return self._calculate_and_log_metrics(outputs, masks, 'val') 