from __future__ import annotations

import random

import torch.nn as nn
from torchvision import models
from torchvision.transforms import functional as F


class RandomGamma:
    """Pickle-friendly random gamma adjustment for PIL images."""

    def __init__(self, gamma_min: float, gamma_max: float):
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def __call__(self, img):
        g = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(img, gamma=g)


def build_model(name: str, pretrained: bool, num_classes: int, in_channels: int = 1) -> nn.Module:
    """
    Model factory. supports resnet18.
    - in_channels = 1 for grayscale.
    - num_classes = 5 for grade 0..4.
    """

    name = name.lower().strip()

    if name != "resnet18":
        raise ValueError(f"Unsupported model name: {name}")

    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        # print("[DEBUG] conv1 in_channels:", model.conv1.in_channels)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
