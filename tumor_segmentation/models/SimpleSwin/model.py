import torch
import os
import torch.nn as nn
import timm
from tumor_segmentation.models.base_model import BaseModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only show error messages
ON_HPC = "ON_HPC" in os.environ

# Import your dataset and data loading functions
model_tag = "swin1_base_224_single_attention_augmented"


class SwinEncoder(nn.Module):
    def __init__(
        self, pretrained=True, model_name: str = "swinv2_base_window8_256.ms_in1k"
    ):
        super(SwinEncoder, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(self, x):
        features = self.model(x)
        features = [f.permute(0, 3, 1, 2) for f in features]
        # return torch.stack(features).permute(1, 0, 4, 2, 3)
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class SimpleSwin(BaseModel):
    """
    Swin Transformer U-Net for tumor segmentation.
    Inherits from BaseModel for consistent training/validation logic.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,
        model_name: str = "swinv2_base_window8_256.ms_in1k",
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight
        )
        self.in_channels = in_channels
        self.encoder = SwinEncoder(pretrained=True, model_name=model_name)

        # Add input channel conversion if needed
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            # Initialize to repeat grayscale channel
            with torch.no_grad():
                self.input_conv.weight.fill_(1.0)

        # Decoder blocks
        self.decoder4 = DecoderBlock(
            in_channels=1024, skip_channels=512, out_channels=512
        )
        self.decoder3 = DecoderBlock(
            in_channels=512, skip_channels=256, out_channels=256
        )
        self.decoder2 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )
        self.decoder1 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64)
        self.decoder0 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        # Final convolution
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Convert input channels if needed (1-channel -> 3-channel)
        if self.in_channels != 3:
            x = self.input_conv(x)

        # Encoder
        features = self.encoder(x)

        # Bottleneck
        bottleneck = features[-1]

        # Decoder
        x = self.decoder4(bottleneck, features[2])
        x = self.decoder3(x, features[1])
        x = self.decoder2(x, features[0])
        x = self.decoder1(x)
        x = self.decoder0(x)

        # Final output
        output = self.final_conv(x)

        return torch.sigmoid(output)
