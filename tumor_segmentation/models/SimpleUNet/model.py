import torch
import torch.nn as nn
from tumor_segmentation.models.base_model import BaseModel


class SimpleUNet(BaseModel):
    """
    Simple U-Net baseline for tumor segmentation.
    Inherits from BaseModel for consistent training/validation logic.
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

        # Encoder
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)

        # Bottleneck
        self.bottleneck = self._make_layer(256, 512)

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
        """Create a simple convolutional layer with dropout for regularization"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout after first conv block
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # Add dropout after second conv block
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder path with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        # Final output
        output = self.final_conv(dec1)

        return torch.sigmoid(output)
