"""Attention U-Net colorizer.

Encoder  : ImageNet-pretrained ResNet34, conv1 adapted to a 1-channel L input
           (new conv weights = channel-mean of the pretrained RGB conv, so the
           pretrained low-level filters are preserved).
Attention: SAGAN-style spatial self-attention at the bottleneck (8x8) and,
           optionally, at a mid-decoder stage (32x32) for longer-range colour
           consistency.
Decoder  : U-Net upsampling path with skip connections; 1x1 head -> 2 ab
           channels squashed to [-1, 1] with tanh.

For a 256x256 input the ResNet34 taps are:
    x0  relu(bn(conv1))  ->  64 ch @ 128
    x1  layer1           ->  64 ch @  64
    x2  layer2           -> 128 ch @  32
    x3  layer3           -> 256 ch @  16
    x4  layer4           -> 512 ch @   8   (bottleneck)
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SelfAttention2d(nn.Module):
    """Self-attention over spatial positions (Zhang et al., SAGAN)."""

    def __init__(self, in_ch: int, reduction: int = 8):
        super().__init__()
        inter = max(1, in_ch // reduction)
        self.query = nn.Conv2d(in_ch, inter, 1)
        self.key = nn.Conv2d(in_ch, inter, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        q = self.query(x).view(b, -1, n).permute(0, 2, 1)   # b, n, c'
        k = self.key(x).view(b, -1, n)                       # b, c', n
        attn = self.softmax(torch.bmm(q, k))                 # b, n, n
        v = self.value(x).view(b, c, n)                      # b, c, n
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Upsample to the skip's exact size, concat, double conv.

    Resizing to ``skip.shape`` (rather than a fixed 2x) keeps skip connections
    aligned even when the input size is not a multiple of 32 and the ResNet's
    floor-rounded strides make stage sizes off by one.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(
        self,
        attention_levels: List[str] = ("bottleneck",),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        attention_levels = set(attention_levels or [])

        weights = (
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        backbone = torchvision.models.resnet34(weights=weights)

        # Adapt conv1: 3-channel RGB -> 1-channel L, preserving pretrained filters.
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # ->64@128
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 64 @ 64
        self.layer2 = backbone.layer2   # 128 @ 32
        self.layer3 = backbone.layer3   # 256 @ 16
        self.layer4 = backbone.layer4   # 512 @ 8

        self.attn_bottleneck = (
            SelfAttention2d(512) if "bottleneck" in attention_levels else None
        )
        self.attn_mid = (
            SelfAttention2d(128) if "mid" in attention_levels else None
        )

        self.up1 = UpBlock(512, 256, 256)   # 8 -> 16
        self.up2 = UpBlock(256, 128, 128)   # 16 -> 32
        self.up3 = UpBlock(128, 64, 64)     # 32 -> 64
        self.up4 = UpBlock(64, 64, 64)      # 64 -> 128
        self.up5 = UpBlock(64, 0, 32)       # 128 -> 256 (no skip)
        self.head = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)              # 64 @ 128
        x1 = self.layer1(self.maxpool(x0))  # 64 @ 64
        x2 = self.layer2(x1)           # 128 @ 32
        x3 = self.layer3(x2)           # 256 @ 16
        x4 = self.layer4(x3)           # 512 @ 8

        if self.attn_bottleneck is not None:
            x4 = self.attn_bottleneck(x4)

        d = self.up1(x4, x3)           # 256 @ 16
        d = self.up2(d, x2)            # 128 @ 32
        if self.attn_mid is not None:
            d = self.attn_mid(d)
        d = self.up3(d, x1)            # 64 @ 64
        d = self.up4(d, x0)            # 64 @ 128
        d = self.up5(d)               # 32 @ 256
        return torch.tanh(self.head(d))  # ab_norm in [-1, 1]


def build_model(cfg) -> AttentionUNet:
    return AttentionUNet(
        attention_levels=cfg.attention_levels,
        pretrained_backbone=cfg.pretrained_backbone,
    )
