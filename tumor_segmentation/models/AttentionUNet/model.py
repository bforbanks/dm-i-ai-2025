import torch
import torch.nn as nn
import torch.nn.functional as F
from tumor_segmentation.models.base_model import BaseModel


class SelfAttention(nn.Module):
    """
    Simple Spatial Self-Attention mechanism for medical image segmentation.
    Allows each pixel to attend to all other pixels in the feature map.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.inter_channels = max(in_channels // reduction_ratio, 1)

        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)

        # Output projection
        self.output_conv = nn.Conv2d(self.inter_channels, in_channels, 1)

        # Normalization
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Self-attended feature map (B, C, H, W)
        """
        batch_size, channels, height, width = x.size()

        # Generate Query, Key, Value
        query = self.query_conv(x).view(
            batch_size, self.inter_channels, -1
        )  # (B, C', H*W)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)  # (B, C', H*W)
        value = self.value_conv(x).view(
            batch_size, self.inter_channels, -1
        )  # (B, C', H*W)

        # Transpose for matrix multiplication
        query = query.permute(0, 2, 1)  # (B, H*W, C')

        # Compute attention scores
        attention_scores = torch.bmm(query, key)  # (B, H*W, H*W)
        attention_weights = self.softmax(
            attention_scores
        )  # Normalize along last dimension

        # Apply attention to values
        attended_values = torch.bmm(
            attention_weights, value.permute(0, 2, 1)
        )  # (B, H*W, C')
        attended_values = attended_values.permute(0, 2, 1).view(
            batch_size, self.inter_channels, height, width
        )

        # Project back to original channel dimensions
        attended_output = self.output_conv(attended_values)

        # Residual connection with learnable scaling
        output = self.gamma * attended_output + x

        return output


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections in U-Net.
    Helps the model focus on relevant features from the encoder.
    """

    def __init__(
        self,
        encoder_channels: int,
        decoder_channels: int,
        intermediate_channels: int = None,
    ):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = encoder_channels // 2

        # Encoder feature processing
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, intermediate_channels, 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )

        # Decoder feature processing
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, intermediate_channels, 1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )

        # Attention coefficient generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, encoder_features, decoder_features):
        """
        Args:
            encoder_features: Skip connection features from encoder (B, C_enc, H, W)
            decoder_features: Upsampled features from decoder (B, C_dec, H_dec, W_dec)
        Returns:
            gated_features: Attention-gated encoder features (B, C_enc, H, W)
        """
        # Upsample decoder features to match encoder spatial dimensions
        encoder_size = encoder_features.size()[2:]
        decoder_upsampled = F.interpolate(
            decoder_features, size=encoder_size, mode="bilinear", align_corners=False
        )

        # Process encoder and decoder features
        encoder_processed = self.encoder_conv(encoder_features)
        decoder_processed = self.decoder_conv(decoder_upsampled)

        # Combine and generate attention coefficients
        combined = encoder_processed + decoder_processed
        attention_coeffs = self.attention_conv(F.relu(combined))

        # Apply attention to encoder features
        gated_features = encoder_features * attention_coeffs

        return gated_features


class AttentionUNet(BaseModel):
    """
    Attention U-Net baseline for tumor segmentation.
    Inherits from BaseModel for consistent training/validation logic.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,
        use_self_attention: bool = True,
        use_attention_gates: bool = True,
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight
        )

        self.use_self_attention = use_self_attention
        self.use_attention_gates = use_attention_gates

        # Encoder
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)

        # Self-attention modules for encoder (optional)
        if self.use_self_attention:
            self.self_attn_enc2 = SelfAttention(64, reduction_ratio=8)
            self.self_attn_enc3 = SelfAttention(128, reduction_ratio=8)
            self.self_attn_enc4 = SelfAttention(256, reduction_ratio=8)

        # Bottleneck with self-attention
        self.bottleneck = self._make_layer(256, 512)
        if self.use_self_attention:
            self.self_attn_bottleneck = SelfAttention(512, reduction_ratio=8)

        # Attention Gates for skip connections (optional)
        if self.use_attention_gates:
            self.att_gate4 = AttentionGate(encoder_channels=256, decoder_channels=512)
            self.att_gate3 = AttentionGate(encoder_channels=128, decoder_channels=256)
            self.att_gate2 = AttentionGate(encoder_channels=64, decoder_channels=128)
            self.att_gate1 = AttentionGate(encoder_channels=32, decoder_channels=64)

        # Decoder
        self.dec4 = self._make_layer(512 + 256, 256)  # +256 for skip connection
        self.dec3 = self._make_layer(256 + 128, 128)  # +128 for skip connection
        self.dec2 = self._make_layer(128 + 64, 64)  # +64 for skip connection
        self.dec1 = self._make_layer(64 + 32, 32)  # +32 for skip connection

        # Final output
        self.final_conv = nn.Conv2d(32, num_classes, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a simple convolutional layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)

        enc2 = self.enc2(self.pool(enc1))
        if self.use_self_attention:
            enc2 = self.self_attn_enc2(enc2)

        enc3 = self.enc3(self.pool(enc2))
        if self.use_self_attention:
            enc3 = self.self_attn_enc3(enc3)

        enc4 = self.enc4(self.pool(enc3))
        if self.use_self_attention:
            enc4 = self.self_attn_enc4(enc4)

        # Bottleneck with self-attention
        bottleneck = self.bottleneck(self.pool(enc4))
        if self.use_self_attention:
            bottleneck = self.self_attn_bottleneck(bottleneck)

        # Decoder path with attention-gated skip connections
        upsampled_bottleneck = self.upsample(bottleneck)
        if self.use_attention_gates:
            gated_enc4 = self.att_gate4(enc4, bottleneck)
            dec4 = self.dec4(torch.cat([upsampled_bottleneck, gated_enc4], dim=1))
        else:
            dec4 = self.dec4(torch.cat([upsampled_bottleneck, enc4], dim=1))

        upsampled_dec4 = self.upsample(dec4)
        if self.use_attention_gates:
            gated_enc3 = self.att_gate3(enc3, dec4)
            dec3 = self.dec3(torch.cat([upsampled_dec4, gated_enc3], dim=1))
        else:
            dec3 = self.dec3(torch.cat([upsampled_dec4, enc3], dim=1))

        upsampled_dec3 = self.upsample(dec3)
        if self.use_attention_gates:
            gated_enc2 = self.att_gate2(enc2, dec3)
            dec2 = self.dec2(torch.cat([upsampled_dec3, gated_enc2], dim=1))
        else:
            dec2 = self.dec2(torch.cat([upsampled_dec3, enc2], dim=1))

        upsampled_dec2 = self.upsample(dec2)
        if self.use_attention_gates:
            gated_enc1 = self.att_gate1(enc1, dec2)
            dec1 = self.dec1(torch.cat([upsampled_dec2, gated_enc1], dim=1))
        else:
            dec1 = self.dec1(torch.cat([upsampled_dec2, enc1], dim=1))

        # Final output
        output = self.final_conv(dec1)

        return torch.sigmoid(output)
