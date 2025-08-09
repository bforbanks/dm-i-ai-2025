from __future__ import annotations

from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn

from .networks.discriminator import MultiscaleDiscriminator


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_dropout: bool = False):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class Pix2PixUNetGenerator(nn.Module):
    """A compact U-Net generator suitable for pix2pix with 2-channel input.

    Input:  (B, 2, H, W)  -> [rough_control, mask]
    Output: (B, 1, H, W)  -> refined tumor region
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = UNetBlock(base_channels, base_channels * 2, down=True)
        self.down3 = UNetBlock(base_channels * 2, base_channels * 4, down=True)
        self.down4 = UNetBlock(base_channels * 4, base_channels * 8, down=True)
        self.down5 = UNetBlock(base_channels * 8, base_channels * 8, down=True)
        self.down6 = UNetBlock(base_channels * 8, base_channels * 8, down=True)
        self.down7 = UNetBlock(base_channels * 8, base_channels * 8, down=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up1 = UNetBlock(base_channels * 8, base_channels * 8, down=False, use_dropout=True)
        self.up2 = UNetBlock(base_channels * 16, base_channels * 8, down=False, use_dropout=True)
        self.up3 = UNetBlock(base_channels * 16, base_channels * 8, down=False, use_dropout=True)
        self.up4 = UNetBlock(base_channels * 16, base_channels * 8, down=False)
        self.up5 = UNetBlock(base_channels * 16, base_channels * 4, down=False)
        self.up6 = UNetBlock(base_channels * 8, base_channels * 2, down=False)
        self.up7 = UNetBlock(base_channels * 4, base_channels, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bn = self.bottleneck(d7)

        # Upsample with skip connections
        u1 = self.up1(bn)
        # Interpolate to match skip connection size if needed
        if u1.shape[-2:] != d7.shape[-2:]:
            u1 = torch.nn.functional.interpolate(u1, size=d7.shape[-2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d7], dim=1)
        
        u2 = self.up2(u1)
        if u2.shape[-2:] != d6.shape[-2:]:
            u2 = torch.nn.functional.interpolate(u2, size=d6.shape[-2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d6], dim=1)
        
        u3 = self.up3(u2)
        if u3.shape[-2:] != d5.shape[-2:]:
            u3 = torch.nn.functional.interpolate(u3, size=d5.shape[-2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d5], dim=1)
        
        u4 = self.up4(u3)
        if u4.shape[-2:] != d4.shape[-2:]:
            u4 = torch.nn.functional.interpolate(u4, size=d4.shape[-2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, d4], dim=1)
        
        u5 = self.up5(u4)
        if u5.shape[-2:] != d3.shape[-2:]:
            u5 = torch.nn.functional.interpolate(u5, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        u5 = torch.cat([u5, d3], dim=1)
        
        u6 = self.up6(u5)
        if u6.shape[-2:] != d2.shape[-2:]:
            u6 = torch.nn.functional.interpolate(u6, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        u6 = torch.cat([u6, d2], dim=1)
        
        u7 = self.up7(u6)
        if u7.shape[-2:] != d1.shape[-2:]:
            u7 = torch.nn.functional.interpolate(u7, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        u7 = torch.cat([u7, d1], dim=1)
        
        out = self.final(u7)

        # Map Tanh [-1,1] to [0,1] range to match image intensities
        return (out + 1.0) * 0.5


class Pix2PixBuilder:
    """Utility to build pix2pix-style generator and reuse existing discriminator."""

    def __init__(
        self,
        crop_size: int = 256,
        crop_size_w: int | None = None,
        crop_size_h: int | None = None,
        ngf: int = 64,
        ndf: int = 64,
        num_D: int = 3,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.crop_size = crop_size
        self.crop_size_w = crop_size_w or crop_size
        self.crop_size_h = crop_size_h or crop_size
        self.ngf = ngf
        self.ndf = ndf
        self.num_D = num_D
        self.device = torch.device(device)

        self.G: nn.Module | None = None
        self.D: nn.Module | None = None

    def build(self) -> Tuple[nn.Module, nn.Module]:
        # Generator: pix2pix U-Net
        self.G = Pix2PixUNetGenerator(in_channels=2, out_channels=1, base_channels=self.ngf).to(self.device)

        # Discriminator: reuse MultiscaleDiscriminator with [mask(1) + output(1)] inputs
        d_opt = SimpleNamespace(
            ndf=self.ndf,
            num_D=self.num_D,
            n_layers_D=4,
            netD_subarch="n_layer",
            norm_D="spectralinstance",
            label_nc=1,
            output_nc=1,
            contain_dontcare_label=False,
            no_instance=True,
            no_ganFeat_loss=True,
        )
        self.D = MultiscaleDiscriminator(d_opt).to(self.device)
        return self.G, self.D

