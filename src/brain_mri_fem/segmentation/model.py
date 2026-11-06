"""A compact 3D U-Net baseline implemented in PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet3D(nn.Module):
    """Small configurable 3D U-Net for binary or multiclass segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2, channels: tuple[int, ...] = (16, 32, 64)) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least two levels")
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        previous = in_channels
        for width in channels:
            self.down.append(DoubleConv(previous, width))
            previous = width
        self.up = nn.ModuleList()
        for width, skip_width in zip(reversed(channels[:-1]), reversed(channels[:-1])):
            self.up.append(nn.ConvTranspose3d(previous, width, 2, stride=2))
            self.up.append(DoubleConv(width + skip_width, width))
            previous = width
        self.head = nn.Conv3d(previous, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down[:-1]:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.down[-1](x)
        for index in range(0, len(self.up), 2):
            x = self.up[index](x)
            skip = skips[-(index // 2 + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = self.up[index + 1](torch.cat((x, skip), dim=1))
        return self.head(x)
