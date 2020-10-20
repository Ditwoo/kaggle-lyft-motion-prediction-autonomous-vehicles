from torchvision.models.resnet import Bottleneck, _resnet, resnext50_32x4d


def resnext18_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext18_32x4d", Bottleneck, [2, 2, 2, 2], False, False, **kwargs)


def resnext34_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext18_32x4d", Bottleneck, [3, 4, 6, 3], False, False, **kwargs)
