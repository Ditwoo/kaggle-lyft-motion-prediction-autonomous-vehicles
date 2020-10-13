import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

from .base import (
    ConvKX,
    BN,
    RELU,
    SuperResKXKX,
    SuperResK1KXK1,
    SuperResK1DWK1,
    AdaptiveAvgPool,
    _create_netblock_list_from_str_,
)


class PlainNet(nn.Module):
    def __init__(self, block_list, pool, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.module_list = nn.ModuleList(block_list)
        self.adptive_avg_pool = pool

        self.last_channels = self.adptive_avg_pool.out_channels
        self.fc_linear = nn.Linear(self.last_channels, self.num_classes, bias=True)

    def extract_features(self, x):
        output = x
        for the_block in self.module_list:
            output = the_block(output)
        return output

    def forward(self, x):
        output = self.extract_features(x)
        output = self.adptive_avg_pool(output)
        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output

    @staticmethod
    def from_string(num_classes, plainnet_struct):
        block_list, remaining_s = _create_netblock_list_from_str_(plainnet_struct)
        assert len(remaining_s) == 0

        if isinstance(block_list[-1], AdaptiveAvgPool):
            pool = block_list[-1]
            block_list.pop(-1)
        else:
            pool = AdaptiveAvgPool(
                out_channels=block_list[-1].out_channels, output_size=1
            )

        return PlainNet(block_list, pool, num_classes)


def genet_small(pretrained=None, num_classes=1000):
    model = PlainNet(
        block_list=[
            ConvKX(3, 13, 3, 2),
            BN(13),
            RELU(13),
            SuperResKXKX(13, 48, 3, 2, 1.0, 1),
            SuperResKXKX(48, 48, 3, 2, 1.0, 3),
            SuperResK1KXK1(48, 384, 3, 2, 0.25, 7),
            SuperResK1DWK1(384, 560, 3, 2, 3.0, 2),
            SuperResK1DWK1(560, 256, 3, 1, 3.0, 1),
            ConvKX(256, 1920, 1, 1),
            BN(1920),
            RELU(1920),
        ],
        pool=AdaptiveAvgPool(1920, 1),
        num_classes=num_classes,
    )

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


def genet_normal(pretrained=None, num_classes=1000):
    model = PlainNet(
        block_list=[
            ConvKX(3, 32, 3, 2),
            BN(32),
            RELU(32),
            SuperResKXKX(32, 128, 3, 2, 1.0, 1),
            SuperResKXKX(128, 192, 3, 2, 1.0, 2),
            SuperResK1KXK1(192, 640, 3, 2, 0.25, 6),
            SuperResK1DWK1(640, 640, 3, 2, 3.0, 4),
            SuperResK1DWK1(640, 640, 3, 1, 3.0, 1),
            ConvKX(640, 2560, 1, 1),
            BN(2560),
            RELU(2560),
        ],
        pool=AdaptiveAvgPool(2560, 1),
        num_classes=num_classes,
    )

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


def genet_large(pretrained=None, num_classes=1000):
    model = PlainNet(
        block_list=[
            ConvKX(3, 32, 3, 2),
            BN(32),
            RELU(32),
            SuperResKXKX(32, 128, 3, 2, 1.0, 1),
            SuperResKXKX(128, 192, 3, 2, 1.0, 2),
            SuperResK1KXK1(192, 640, 3, 2, 0.25, 6),
            SuperResK1DWK1(640, 640, 3, 2, 3.0, 5),
            SuperResK1DWK1(640, 640, 3, 1, 3.0, 4),
            ConvKX(640, 2560, 1, 1),
            BN(2560),
            RELU(2560),
        ],
        pool=AdaptiveAvgPool(2560, 1),
        num_classes=num_classes,
    )

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model
