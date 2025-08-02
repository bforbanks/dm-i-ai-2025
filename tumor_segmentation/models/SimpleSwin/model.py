import torch
import torch.nn as nn
import torch.nn.functional as F
from tumor_segmentation.models.base_model import BaseModel

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print(
        "Warning: timm not available. Install with 'pip install timm' to use SwinUNet"
    )


class SelfAttentionBlock(nn.Module):
    """Self-attention block for feature enhancement"""

    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class SwinEncoder(nn.Module):
    """Swin Transformer encoder with pretrained weights"""

    def __init__(self, pretrained=True, model_name="swin_base_patch4_window7_224"):
        super(SwinEncoder, self).__init__()
        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for SwinEncoder. Install with 'pip install timm'"
            )

        # Create Swin model for feature extraction
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Get features from 4 stages
        )

        # Input adaptation layer to convert single channel to RGB
        self.input_adapter = nn.Conv2d(1, 3, 1, bias=False)
        if pretrained:
            # Initialize to grayscale conversion weights
            with torch.no_grad():
                self.input_adapter.weight.fill_(1 / 3)

    def forward(self, x):
        # Convert single channel to RGB-like for pretrained model
        x_rgb = self.input_adapter(x)

        # Get multi-scale features
        features = self.model(x_rgb)

        # Convert features back to proper format (B, C, H, W)
        features = [f.permute(0, 3, 1, 2) for f in features]
        return features


class ChannelAttention(nn.Module):
    """Channel attention module"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class DecoderBlock(nn.Module):
    """Enhanced decoder block with attention"""

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        total_channels = (
            in_channels + skip_channels if skip_channels > 0 else in_channels
        )

        self.conv1 = nn.Conv2d(total_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Attention modules
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Ensure spatial dimensions match for concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.use_attention:
            x = self.channel_attention(x)

        return x


class SimpleSwin(BaseModel):
    """
    Enhanced U-Net with Swin Transformer encoder and attention mechanisms.
    Combines the power of pretrained Swin transformers with attention-based decoding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-4,
        weight_decay=1e-5,
        use_pretrained=True,
        model_name="swin_base_patch4_window7_224",
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for SwinUNet. Install with 'pip install timm'"
            )

        # Swin Transformer encoder
        self.encoder = SwinEncoder(pretrained=use_pretrained, model_name=model_name)

        # Get feature dimensions based on model
        # Typical Swin Base dimensions: [128, 256, 512, 1024]
        if "base" in model_name:
            self.feature_dims = [128, 256, 512, 1024]
        elif "small" in model_name:
            self.feature_dims = [96, 192, 384, 768]
        elif "tiny" in model_name:
            self.feature_dims = [96, 192, 384, 768]
        else:
            # Default to base
            self.feature_dims = [128, 256, 512, 1024]

        # Bottleneck with self-attention
        self.bottleneck_attention = SelfAttentionBlock(self.feature_dims[-1])

        # Decoder blocks with attention
        self.decoder4 = DecoderBlock(self.feature_dims[-1], self.feature_dims[-2], 512)
        self.decoder3 = DecoderBlock(512, self.feature_dims[-3], 256)
        self.decoder2 = DecoderBlock(256, self.feature_dims[-4], 128)
        self.decoder1 = DecoderBlock(
            128, 0, 64
        )  # No skip connection for final upsampling

        # Final output layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        # Store input size for final upsampling
        input_size = x.shape[2:]

        # Encoder path - get multi-scale features
        features = self.encoder(x)

        # Apply self-attention to bottleneck features
        bottleneck = self.bottleneck_attention(features[-1])

        # Decoder path with skip connections
        x = self.decoder4(bottleneck, features[-2])
        x = self.decoder3(x, features[-3])
        x = self.decoder2(x, features[-4])
        x = self.decoder1(x)

        # Final convolution
        x = self.final_conv(x)

        # Ensure output matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return torch.sigmoid(x)
