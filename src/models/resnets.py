import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock, model_urls
from torchvision.models.utils import load_state_dict_from_url


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


def resnet18(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = models.resnet18(pretrained=pretrained, progress=progress)
    model = _adjust_model(model, in_channels, num_classes)
    return model

def resnet34(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = models.resnet34(pretrained=pretrained, progress=progress)
    model = _adjust_model(model, in_channels, num_classes)
    return model

def resnet50(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = models.resnet50(pretrained=pretrained, progress=progress)
    model = _adjust_model(model, in_channels, num_classes)
    return model


class Category(nn.Module):
    def __init__(self, num_categories, num_features=128):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, num_features * 2)
        self.head = nn.Sequential(
            nn.Linear(num_features * 2, num_features, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, x):
        emb = self.embedding(x)
        features = self.head(emb)
        return features


class MultiCategory(nn.Module):
    def __init__(self, categories_sizes, emb_dim=32, out_features=64):
        """
        Args:
            categories_sizes (List[int]): list with number of categories
            emb_dim (int): dimension of each of embeddings.
            out_features (int, optional): output feature dimension.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.out_features = out_features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_size, self.emb_dim) for cat_size in categories_sizes]
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.emb_dim * len(categories_sizes), out_features),
            nn.ReLU(True),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
        )

    def forward(self, *cat_feats):
        assert len(self.embeddings) == len(cat_feats)
        x = [emb(feat) for emb, feat in zip(self.embeddings, cat_feats)]
        x = torch.cat(x, dim=1)
        x = self.head(x)
        return x


class CubicResNet(ResNet):
    def __init__(self, *args, num_categories=96, num_cat_features=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = Category(num_categories, num_cat_features)
        self.fc = nn.Linear(
            self.fc.in_features + num_cat_features, self.fc.out_features
        )

    def unet_forward(self, x, c):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # conv features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # categorical features
        c = self.category(c)

        x = self.fc(torch.cat((x, c), dim=1))

        return x, (x1, x2, x3, x4)

    def forward(self, x, c):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # conv features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # categorical features
        c = self.category(c)

        x = self.fc(torch.cat((x, c), dim=1))

        return x


class MultiCategoryResNet(ResNet):
    def __init__(self, *args, cat_sizes, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = MultiCategory(cat_sizes)
        self.fc = nn.Linear(
            self.fc.in_features + self.category.out_features, self.fc.out_features
        )

    def unet_forward(self, x, *cats):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # conv features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # categorical features
        c = self.category(*cats)

        x = self.fc(torch.cat((x, c), dim=1))

        return x, (x1, x2, x3, x4)

    def forward(self, x, *cats):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # conv features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # categorical features
        c = self.category(*cats)

        x = self.fc(torch.cat((x, c), dim=1))

        return x


def resnet18_cubic(pretrained=False, in_channels=3, num_classes=1000, progress=True):
    model = CubicResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model = _adjust_model(model, in_channels, num_classes)
    return model


def resnet18_cat(
    pretrained=False,
    in_channels=3,
    num_classes=1000,
    progress=True,
    cat_sizes=[96, 12, 7, 24],
):
    model = MultiCategoryResNet(
        BasicBlock, [2, 2, 2, 2], cat_sizes=cat_sizes, num_classes=num_classes
    )
    model = _adjust_model(model, in_channels, num_classes)
    return model


# NOTE: should be used with resnets from torchvision
class ResnetCustom(ResNet):
    def unet_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, (x1, x2, x3, x4)


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        mid_channels = 32

        self.conv2dT = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.conv2d = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2dT(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class SegmentationResnet(nn.Module):
    def __init__(self, encoder):
        super(SegmentationResnet, self).__init__()

        fn = getattr(encoder, "unet_forward", None)
        if not callable(fn):
            raise ValueError("encoder doesn't have a 'unet_forward' function!")

        self.encoder = encoder

        layers_outputs = [
            layer[-1].conv2.out_channels
            for layer in (
                encoder.layer4,
                encoder.layer3,
                encoder.layer2,
                encoder.layer1,
            )
        ]

        self.up4 = UpLayer(
            in_channels=layers_outputs[0], out_channels=layers_outputs[1] // 2,
        )
        self.up3 = UpLayer(
            in_channels=layers_outputs[1] + layers_outputs[1] // 2,
            out_channels=layers_outputs[2] // 2,
        )

        self.up2 = UpLayer(
            in_channels=layers_outputs[2] + layers_outputs[2] // 2,
            out_channels=layers_outputs[3] // 2,
        )

        self.up1 = UpLayer(
            in_channels=layers_outputs[3] + layers_outputs[3] // 2, out_channels=32
        )

        self.up = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding_mode="zeros",
        )

    def forward(self, *args):
        logits, (x1, x2, x3, x4) = self.encoder.unet_forward(*args)

        u4 = self.up4(x4)
        u3 = self.up3(torch.cat((u4, x3), dim=1))
        u2 = self.up2(torch.cat((u3, x2), dim=1))
        u1 = self.up1(torch.cat((u2, x1), dim=1))

        u = self.up(u1)

        return logits, u


def resnet18_unet(
    pretrained=False, in_channels=3, num_classes=1000, progress=True, **kwargs
):
    encoder = ResnetCustom(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=progress)
        encoder.load_state_dict(state_dict)
    encoder = _adjust_model(encoder, in_channels, num_classes)
    model = SegmentationResnet(encoder)
    return model

