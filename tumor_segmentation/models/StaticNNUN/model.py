import torch
import torch.nn as nn
from models.base_model import BaseModel


class ResidualBlock(nn.Module):
    """Residual block with instance normalization and LeakyReLU as per nnUNet config"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, eps=1e-05, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm2 = nn.InstanceNorm2d(out_channels, eps=1e-05, affine=True)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=True
                ),
                nn.InstanceNorm2d(out_channels, eps=1e-05, affine=True),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)

        return out


class EncoderStage(nn.Module):
    """Encoder stage with multiple residual blocks"""

    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(EncoderStage, self).__init__()

        # First block handles stride and channel change
        blocks = [ResidualBlock(in_channels, out_channels, stride)]

        # Remaining blocks
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class DecoderStage(nn.Module):
    """Decoder stage with upsampling and skip connections"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderStage, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        # Single conv block as per n_conv_per_stage_decoder: [1, 1, 1, 1, 1, 1]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(out_channels, eps=1e-05, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class StaticNNUN(BaseModel):
    """
    ResidualEncoderUNet following nnUNet 2d_resenc_optimized configuration.
    Architecture matches: n_stages=7, features_per_stage=[32, 64, 128, 256, 512, 512, 512]
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6], InstanceNorm2d, LeakyReLU
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight
        )

        # nnUNet ResEncUNet configuration
        features_per_stage = [32, 64, 128, 256, 512, 512, 512]
        n_blocks_per_stage = [1, 3, 4, 6, 6, 6, 6]
        strides = [
            1,
            2,
            2,
            2,
            2,
            2,
            2,
        ]  # First stage no downsampling, rest downsample by 2

        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, features_per_stage[0], kernel_size=3, padding=1, bias=True
            ),
            nn.InstanceNorm2d(features_per_stage[0], eps=1e-05, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        in_ch = features_per_stage[0]

        for i, (out_ch, n_blocks, stride) in enumerate(
            zip(features_per_stage, n_blocks_per_stage, strides)
        ):
            stage = EncoderStage(in_ch, out_ch, n_blocks, stride)
            self.encoder_stages.append(stage)
            in_ch = out_ch

        # Decoder stages (reverse order, skip last encoder stage which is bottleneck)
        self.decoder_stages = nn.ModuleList()
        decoder_features = features_per_stage[:-1][::-1]  # [512, 256, 128, 64, 32]
        encoder_features = features_per_stage[:-2][
            ::-1
        ]  # [256, 128, 64, 32] for skip connections

        for i, out_ch in enumerate(decoder_features):
            if i == 0:
                # First decoder stage (from bottleneck)
                in_ch = features_per_stage[-1]  # 512
                skip_ch = features_per_stage[-2]  # 512
            else:
                in_ch = decoder_features[i - 1]
                skip_ch = (
                    encoder_features[i - 1] if i - 1 < len(encoder_features) else 0
                )

            stage = DecoderStage(in_ch, skip_ch, out_ch)
            self.decoder_stages.append(stage)

        # Final output layer
        self.final_conv = nn.Conv2d(
            features_per_stage[0], num_classes, kernel_size=1, bias=True
        )

    def forward(self, x):
        # Input projection
        x = self.input_conv(x)

        # Encoder path - store skip connections
        skip_connections = []
        for stage in self.encoder_stages[:-1]:  # All except bottleneck
            x = stage(x)
            skip_connections.append(x)

        # Bottleneck (last encoder stage)
        x = self.encoder_stages[-1](x)

        # Decoder path with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        for i, stage in enumerate(self.decoder_stages):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = stage(x, skip)

        # Final output
        output = self.final_conv(x)

        return torch.sigmoid(output)
