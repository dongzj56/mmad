
"""ImageEncoder.py
-------------------
3D‑ResNet encoder backbone for volumetric data (MRI, PET, CT).

Provides:
    * ImageEncoder  – flexible encoder with optional global pooling
    * image_encoder{18,34,50,101,152,200}  – easy factory helpers

Usage Example
-------------
import torch
from ImageEncoder import image_encoder18
model = image_encoder18(in_channels=1, global_pool=False)  # feature map
x = torch.randn(2, 1, 120, 144, 120)
eat = model(x)   # shape: [2, 512, 15, 18, 15]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = [
    "ImageEncoder",
    "image_encoder18",
    "image_encoder34",
    "image_encoder50",
    "image_encoder101",
    "image_encoder152",
    "image_encoder200",
]

# -------------------------------------------------------------------- #
# Basic ops and blocks
# -------------------------------------------------------------------- #
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3×3×3 Conv, automatically pads for dilation."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    """A‑shortcut: avg‑pool + zero‑pad along channels."""
    out = F.avg_pool3d(x, 1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# -------------------------------------------------------------------- #
# Encoder
# -------------------------------------------------------------------- #
class ImageEncoder(nn.Module):
    """3D‑ResNet encoder (no segmentation head).

    Parameters
    ----------
    block : BasicBlock or Bottleneck
    layers: list[int]  Number of blocks in each stage
    in_channels : int  Number of input channels (default 1)
    shortcut_type: str 'A' or 'B' residual shortcut
    global_pool  : bool
        True  → outputs B×C embedding via adaptive avg‑pool
        False → outputs feature map from layer4
    """

    def __init__(
        self,
        block,
        layers,
        in_channels=1,
        shortcut_type="B",
        no_cuda=False,
        global_pool=False,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.no_cuda = no_cuda
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            in_channels, 64, 7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, blocks, shortcut_type, stride=1, dilation=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers_ = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers_)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.global_pool:
            x = F.adaptive_avg_pool3d(x, 1)
            x = torch.flatten(x, 1)
        return x


# -------------------------------------------------------------------- #
# Factory helpers
# -------------------------------------------------------------------- #
def image_encoder18(**kwargs):
    return ImageEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def image_encoder34(**kwargs):
    return ImageEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)


def image_encoder50(**kwargs):
    return ImageEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def image_encoder101(**kwargs):
    return ImageEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)


def image_encoder152(**kwargs):
    return ImageEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)


def image_encoder200(**kwargs):
    return ImageEncoder(Bottleneck, [3, 24, 36, 3], **kwargs)
