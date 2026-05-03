from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Create a ResNet model adapted for CIFAR images.

    Standard ImageNet ResNet uses 7x7 stride-2 conv + maxpool, which is too aggressive
    for 32x32 CIFAR images. Here we use a 3x3 stride-1 stem and remove maxpool.
    """
    model_name = name.lower()
    if model_name != "resnet18":
        raise ValueError("Starter project currently supports only resnet18.")

    weights = None
    if pretrained:
        # Pretrained weights are ImageNet-oriented and not the default for this baseline.
        weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    # CIFAR adaptation.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # TODO(student): Print the model and confirm spatial size is not crushed too early.
    return model
