import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock


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
