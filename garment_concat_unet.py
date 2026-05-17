import torch
from torch import nn
from typing import Iterable, Optional


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
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool) -> None:
        super().__init__()
        self.down = ConvBlock(in_ch, out_ch, downsample=True, use_norm=use_norm)
        self.res1 = ResidualBlock(out_ch, out_ch, use_norm)
        self.res2 = ResidualBlock(out_ch, out_ch, use_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class GarmentEncoder(nn.Module):
    """
    Small image encoder that preserves or downsamples spatial resolution.

    channels: output channels per layer, e.g. (64, 128, 256, 256, 256)
    num_layers: number of downsampling layers (M in the paper)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Iterable[int] = (64, 128, 256, 256, 256),
        num_layers: int = 5,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if not channels:
            raise ValueError("channels must have at least one entry")
        if num_layers != len(channels):
            raise ValueError("num_layers must match the length of channels")

        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.append(EncoderLayer(in_ch, out_ch, use_norm))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GarmentConcatUNet(nn.Module):
    """
    Two encoders + occlusion mapping + U-Net.

    The two garment images are encoded separately, concatenated on channel
    dimension, processed by mapping network U and a linear layer to produce
    a spatial attention map A, then passed into the provided UNet.
    """

    def __init__(
        self,
        inner_encoder: nn.Module,
        outer_encoder: nn.Module,
        unet: nn.Module,
        project_in: bool = True,
        unet_takes_timesteps: bool = True,
        mapping_hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.inner_encoder = inner_encoder
        self.outer_encoder = outer_encoder
        self.unet = unet
        self.unet_takes_timesteps = unet_takes_timesteps

        inner_ch = getattr(inner_encoder, "out_channels", None)
        outer_ch = getattr(outer_encoder, "out_channels", None)
        unet_in = getattr(unet, "in_channels", None)

        if inner_ch is not None and outer_ch is not None:
            mapping_in = inner_ch + outer_ch
            mapping_hidden = mapping_hidden_channels or mapping_in
            self.mapping_u = MappingUNet(mapping_in, mapping_hidden)
            self.to_attention = nn.Conv2d(mapping_hidden, 1, kernel_size=1)
        else:
            self.mapping_u = nn.Identity()
            self.to_attention = nn.Conv2d(1, 1, kernel_size=1)

        if project_in and inner_ch is not None and outer_ch is not None and unet_in is not None:
            concat_ch = inner_ch + outer_ch
            if concat_ch != unet_in:
                self.project_in = nn.Conv2d(concat_ch, unet_in, kernel_size=1)
            else:
                self.project_in = nn.Identity()
        else:
            self.project_in = nn.Identity()

    def forward(
        self,
        inner_img: torch.Tensor,
        outer_img: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        inner_feat = self.inner_encoder(inner_img)
        outer_feat = self.outer_encoder(outer_img)

        if inner_feat.shape[-2:] != outer_feat.shape[-2:]:
            raise ValueError(
                f"Encoded sizes differ: inner={inner_feat.shape[-2:]}, outer={outer_feat.shape[-2:]}"
            )

        concat_feat = torch.cat([outer_feat, inner_feat], dim=1)
        u = self.mapping_u(concat_feat)
        attention = torch.sigmoid(self.to_attention(u))
        refined_inner = inner_feat * attention

        x = torch.cat([outer_feat, refined_inner], dim=1)
        x = self.project_in(x)

        if self.unet_takes_timesteps:
            if timesteps is None:
                timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            out = self.unet(x, timesteps=timesteps, context=context, **kwargs)
        else:
            out = self.unet(x)

        if return_attention:
            return out, attention

        return out

class MappingUNet(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int) -> None:
        super().__init__()

        # Encoder
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

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch * 2, kernel_size=3, padding=1),
            _group_norm(hidden_ch * 2),
            nn.SiLU(),
        )

        # Decoder
        self.dec1_up = nn.ConvTranspose2d(hidden_ch * 2, hidden_ch, kernel_size=4, stride=2, padding=1)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(hidden_ch + hidden_ch * 2, hidden_ch, kernel_size=3, padding=1),
            _group_norm(hidden_ch),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_ch + hidden_ch, hidden_ch, kernel_size=3, padding=1),
            _group_norm(hidden_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                                    # (B, hidden,   H,   W)
        e2 = self.enc2(e1)                                   # (B, hidden*2, H/2, W/2)

        b = self.bottleneck(e2)                              # (B, hidden*2, H/2, W/2)

        d1 = self.dec1_up(b)                                 # (B, hidden,   H,   W)
        d1 = self.dec1_conv(torch.cat([d1, e2], dim=1))     # skip from e2
        d2 = self.dec2(torch.cat([d1, e1], dim=1))          # skip from e1

        return d2                                            # (B, hidden, H, W)


def build_simple_garment_concat_unet(unet: nn.Module) -> GarmentConcatUNet:
    inner_encoder = GarmentEncoder(in_channels=3)
    outer_encoder = GarmentEncoder(in_channels=3)
    return GarmentConcatUNet(inner_encoder, outer_encoder, unet)
