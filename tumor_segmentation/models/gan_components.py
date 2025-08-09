from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn

from .networks.generator import SPADEGenerator
from .networks.discriminator import MultiscaleDiscriminator

try:
    from kornia.losses import FFTConsistencyLoss  # type: ignore
except ImportError:  # pragma: no cover
    FFTConsistencyLoss = None  # noqa: N816


class GANBuilder:
    """Utility class to create SPADE generator + multi-scale discriminator with
    minimal dummy `opt` objects.

    Notes
    -----
    The original NVIDIA implementation relies on a large command-line
    `opt` Namespace.  We spoof just the fields that the individual
    networks actually access, keeping things delightfully simple.
    """

    def __init__(
        self,
        crop_size: int = 256,
        crop_size_w: int = None,
        crop_size_h: int = None,
        aspect_ratio: float = 1.0,
        ngf: int = 64,
        ndf: int = 64,
        num_D: int = 3,
        num_upsampling_layers: str = "normal",
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        # Store parameters for potential later reference
        self.crop_size = crop_size
        self.crop_size_w = crop_size_w or crop_size
        self.crop_size_h = crop_size_h or crop_size
        if crop_size_w is not None and crop_size_h is not None:
            self.aspect_ratio = crop_size_w / crop_size_h
        else:
            self.aspect_ratio = aspect_ratio
        self.ngf = ngf
        self.ndf = ndf
        self.num_D = num_D
        self.num_upsampling_layers = num_upsampling_layers
        self.device = torch.device(device)

        self.G: nn.Module | None = None
        self.D: nn.Module | None = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def build(self) -> Tuple[nn.Module, nn.Module]:
        """Instantiate generator & discriminator and move them to device."""

        # --- Generator options -------------------------------------------------
        g_opt = SimpleNamespace(
            norm_G='spectralspadesyncbatch3x3',
            init_type='normal',
            init_variance=0.02,
            gpu_ids=[],
            ngf=self.ngf,
            semantic_nc=2,  # control slice + mask
            crop_size=self.crop_size_h,  # Use height for legacy compatibility
            crop_size_w=self.crop_size_w,  # Add width dimension
            crop_size_h=self.crop_size_h,  # Add height dimension
            aspect_ratio=self.aspect_ratio,
            num_upsampling_layers=self.num_upsampling_layers,
            use_vae=False,
            z_dim=256,
        )
        self.G = SPADEGenerator(g_opt)
        # replace final conv to output single channel
        self.G.conv_img = nn.Conv2d(self.G.conv_img.in_channels, 1, kernel_size=3, padding=1)
        self.G = self.G.to(self.device)

        # --- Discriminator options -------------------------------------------
        d_opt = SimpleNamespace(
            ndf=self.ndf,
            num_D=self.num_D,
            n_layers_D=4,
            netD_subarch="n_layer",
            norm_D="spectralinstance",
            label_nc=1,  # mask (single-channel)
            output_nc=1,  # generated slice (single-channel)
            contain_dontcare_label=False,
            no_instance=True,
            no_ganFeat_loss=True,  # we attach only final logits
        )
        self.D = MultiscaleDiscriminator(d_opt).to(self.device)

        return self.G, self.D

    # ------------------------------------------------------------------
    # Optional loss helpers
    # ------------------------------------------------------------------
    def build_fft_loss(self, **kwargs):
        """Return FFTConsistencyLoss if kornia available."""
        if FFTConsistencyLoss is None:
            raise RuntimeError(
                "kornia not installed – install kornia>=0.7.0 to enable FFTConsistencyLoss"
            )
        return FFTConsistencyLoss(**kwargs)
