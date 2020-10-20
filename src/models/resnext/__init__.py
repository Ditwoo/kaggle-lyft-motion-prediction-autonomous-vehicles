import torch
import torch.nn as nn
from .base import resnext18_32x4d, resnext34_32x4d, resnext50_32x4d


def _adjust_model(model, in_channels, num_classes):
    if in_channels != model.conv1.in_channels:
        model.conv1 = nn.Conv2d(
            in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnext18(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = resnext18_32x4d()
    model = _adjust_model(model, in_channels, num_classes)
    return model


def resnext34(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = resnext34_32x4d()
    model = _adjust_model(model, in_channels, num_classes)
    return model


def resnext50(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = resnext50_32x4d(pretrained=pretrained, progress=progress)
    model = _adjust_model(model, in_channels, num_classes)
    return model
