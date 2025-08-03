import torch
import torch.nn as nn
from tumor_segmentation.models.base_model import BaseModel


class NNUNetStyle(BaseModel):
    """
    NNUNet-style U-Net for tumor segmentation.
    Implements key NNUNet architectural features:
    - Instance normalization instead of batch normalization
    - Leaky ReLU activations
    - Proper channel dimensions following NNUNet conventions
    - More sophisticated skip connections
    Inherits from BaseModel for consistent training/validation logic.
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
        depth: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight, 
            false_negative_penalty=false_negative_penalty, false_negative_penalty_scheduler=false_negative_penalty_scheduler,
            patient_weight=patient_weight, control_weight=control_weight
        )
        
        # Store dropout rate
        self.dropout_rate = dropout_rate

        # Use hardcoded channel dimensions like SimpleUNet for now
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 32)
        self.enc2 = self._make_encoder_block(32, 64)
        self.enc3 = self._make_encoder_block(64, 128)
        self.enc4 = self._make_encoder_block(128, 256)

        # Bottleneck
        self.bottleneck = self._make_encoder_block(256, 512)

        # Decoder - match SimpleUNet exactly
        self.dec4 = self._make_decoder_block(512 + 256, 256)  # +256 for skip connection
        self.dec3 = self._make_decoder_block(256 + 128, 128)  # +128 for skip connection
        self.dec2 = self._make_decoder_block(128 + 64, 64)    # +64 for skip connection
        self.dec1 = self._make_decoder_block(64 + 32, 32)     # +32 for skip connection

        # Final output
        self.final_conv = nn.Conv2d(32, num_classes, 1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

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