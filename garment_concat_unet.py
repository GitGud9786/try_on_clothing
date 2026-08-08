
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = min(32, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool, use_norm: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = _group_norm(out_ch) if use_norm else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm1 = _group_norm(out_ch) if use_norm else nn.Identity()
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(out_ch) if use_norm else nn.Identity()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class EncoderLayer(nn.Module):
    """One (optionally downsampling) convolution followed by two residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool, use_norm: bool) -> None:
        super().__init__()
        self.down = ConvBlock(in_ch, out_ch, downsample=downsample, use_norm=use_norm)
        self.res1 = ResidualBlock(out_ch, out_ch, use_norm)
        self.res2 = ResidualBlock(out_ch, out_ch, use_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(self.down(x)))


class GarmentEncoder(nn.Module):
   

    def __init__(
        self,
        in_channels: int = 3,
        channels: Iterable[int] = (64, 128, 256, 256, 256),
        downsample: Optional[Sequence[bool]] = None,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if not channels:
            raise ValueError("channels must have at least one entry")

        if downsample is None:
            downsample = [True] * len(channels)
        downsample = list(downsample)
        if len(downsample) != len(channels):
            raise ValueError(
                f"downsample has {len(downsample)} entries but channels has {len(channels)}"
            )

        layers, in_ch = [], in_channels
        for out_ch, ds in zip(channels, downsample):
            layers.append(EncoderLayer(in_ch, out_ch, ds, use_norm))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.downsample_factor = 2 ** sum(downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class MappingUNet(nn.Module):
  

    def __init__(self, in_ch: int, hidden_ch: int) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            _group_norm(hidden_ch),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch * 2, kernel_size=3, stride=2, padding=1),
            _group_norm(hidden_ch * 2),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch * 2, kernel_size=3, padding=1),
            _group_norm(hidden_ch * 2),
            nn.SiLU(),
        )
        self.up = nn.ConvTranspose2d(hidden_ch * 2, hidden_ch, kernel_size=4, stride=2, padding=1)
        self.dec = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, kernel_size=3, padding=1),
            _group_norm(hidden_ch),
            nn.SiLU(),
        )
        self.out_channels = hidden_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                  # (B,  h,   H,   W)
        e2 = self.enc2(e1)                 # (B, 2h, H/2, W/2)
        b = self.bottleneck(e2)            # (B, 2h, H/2, W/2)

        d = self.up(b)                     # (B,  h,   H,   W)
        if d.shape[-2:] != e1.shape[-2:]:  # only fires when H or W is odd
            d = F.interpolate(d, size=e1.shape[-2:], mode="nearest")
        return self.dec(torch.cat([d, e1], dim=1))   # (B, h, H, W)


class GarmentOcclusionLearning(nn.Module):
 

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: Iterable[int] = (64, 128, 256, 256, 256),
        encoder_downsample: Optional[Sequence[bool]] = None,
        mapping_hidden_channels: int = 256,
        use_norm: bool = True,
        attention_bias_init: float = 2.0,
        attention_floor: float = 0.0,
        share_encoder_weights: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= attention_floor < 1.0:
            raise ValueError("attention_floor must be in [0, 1)")
        self.attention_floor = attention_floor

        self.outer_encoder = GarmentEncoder(
            in_channels, encoder_channels, encoder_downsample, use_norm
        )
        if share_encoder_weights:
            # "Two identical encoders" in the paper means identical architecture.
            # Tying the weights is available but off by default -- the two inputs
            # play asymmetric roles (occluder vs occluded).
            self.inner_encoder = self.outer_encoder
        else:
            self.inner_encoder = GarmentEncoder(
                in_channels, encoder_channels, encoder_downsample, use_norm
            )

        mapping_in = self.outer_encoder.out_channels + self.inner_encoder.out_channels
        self.mapping_u = MappingUNet(mapping_in, mapping_hidden_channels)

        # "Linear" in Eq. (3): a per-pixel linear projection to a 1-channel map.
        self.to_attention = nn.Conv2d(mapping_hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.to_attention.weight)
        # Start at A = sigmoid(2) ~= 0.88 so the gate is open at init and z_iv is
        # close to z_i. Starting at sigmoid(0) = 0.5 silently halves every inner
        # latent before training has learned anything.
        nn.init.constant_(self.to_attention.bias, attention_bias_init)

    def forward(
        self,
        outer_img: torch.Tensor,
        inner_img: torch.Tensor,
        inner_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        outer_img / inner_img : (B, C_in, H, W)      garment images (or latents)
        inner_latent  (z_i)   : (B, C_lat, h, w)     frozen-VAE encoding of the inner garment

        returns (z_iv, A), on the latent grid: (B, C_lat, h, w) and (B, 1, h, w).
        """
        if outer_img.shape[-2:] != inner_img.shape[-2:]:
            raise ValueError(
                f"Garment inputs differ spatially: outer={tuple(outer_img.shape[-2:])}, "
                f"inner={tuple(inner_img.shape[-2:])}. Feed both at the same resolution."
            )

        f_o = self.outer_encoder(outer_img)
        f_i = self.inner_encoder(inner_img)

        u = self.mapping_u(torch.cat([f_o, f_i], dim=1))   # E_o(g_o) (c) E_i(g_i) -> U
        attention = torch.sigmoid(self.to_attention(u))    # (B, 1, H/32, W/32)

        if self.attention_floor > 0.0:
            # Keeps a little signal alive in fully-suppressed regions so their
            # gradients do not die permanently.
            attention = self.attention_floor + (1.0 - self.attention_floor) * attention

        if attention.shape[-2:] != inner_latent.shape[-2:]:
            attention = F.interpolate(
                attention,
                size=inner_latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attention = attention.to(inner_latent.dtype)   # survives AMP / fp16 latents

        z_iv = inner_latent * attention
        return z_iv, attention


@torch.no_grad()
def attention_stats(attention: torch.Tensor) -> dict:
    """
    Log these every epoch. A degenerate gate is invisible in the loss curve but
    obvious here: mean drifting to ~1.0 with std collapsing towards 0 means A has
    become the identity and the module is contributing nothing.
    """
    a = attention.float()
    return {
        "A/mean": a.mean().item(),
        "A/std": a.std().item(),
        "A/min": a.amin().item(),
        "A/max": a.amax().item(),
        "A/frac_below_0.5": (a < 0.5).float().mean().item(),
    }



class GarmentConditionFusion(nn.Module):
 

    def __init__(self, latent_channels: int = 4, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = out_channels or latent_channels
        self.proj = nn.Conv2d(latent_channels * 2, out_channels, kernel_size=1)
        self.out_channels = out_channels

        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for c in range(min(out_channels, latent_channels)):
                self.proj.weight[c, c, 0, 0] = 1.0

    def forward(self, outer_latent: torch.Tensor, z_iv: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([outer_latent, z_iv], dim=1))


class GarmentConcatUNet(nn.Module):
 

    def __init__(
        self,
        inner_encoder: Optional[nn.Module] = None,
        outer_encoder: Optional[nn.Module] = None,
        unet: Optional[nn.Module] = None,
        latent_channels: int = 4,
        cond_out_channels: Optional[int] = None,
        project_in: Optional[bool] = None,          # deprecated, ignored
        unet_takes_timesteps: Optional[bool] = None,  # deprecated, ignored
        **gol_kwargs,
    ) -> None:
        super().__init__()
        # nn.Identity() was the idiom for "no UNet here" and is accepted silently.
        # A real UNet is refused: the old code called it with the garment tensor as
        # `x`, which replaces the noisy latent and breaks the diffusion process.
        if unet is not None and not isinstance(unet, nn.Identity):
            raise TypeError(
            )

        self.gol = GarmentOcclusionLearning(**gol_kwargs)
        # Optional injection of pre-built encoders, matching the old call style.
        if inner_encoder is not None:
            self.gol.inner_encoder = inner_encoder
        if outer_encoder is not None:
            self.gol.outer_encoder = outer_encoder

        self.fusion = GarmentConditionFusion(latent_channels, cond_out_channels)

    def forward(
        self,
        outer_img: torch.Tensor,
        inner_img: torch.Tensor,
        outer_latent: Optional[torch.Tensor] = None,
        inner_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Under StableVITON's use_VAEdownsample path the garment inputs are already
        # VAE latents, so they double as the modulation targets.
        if outer_latent is None:
            outer_latent = outer_img
        if inner_latent is None:
            inner_latent = inner_img

        if outer_latent.shape[-2:] != inner_latent.shape[-2:]:
            raise ValueError(
                f"Latents differ spatially: outer={tuple(outer_latent.shape[-2:])}, "
                f"inner={tuple(inner_latent.shape[-2:])}."
            )
        z_iv, attention = self.gol(outer_img, inner_img, inner_latent)
        cond = self.fusion(outer_latent, z_iv)
        return cond, z_iv, attention
