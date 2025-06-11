import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Dict, List, Any

# Pretrained URLs omitted for brevity


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class EarlyExitResNet(nn.Module):
    """
    ResNet with two early exits after layer1 and layer2.
    Returns a list of logits [exit1, exit2, final].
    """
    def __init__(self, block, layers: List[int], num_classes: int = 10):
        super(EarlyExitResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # Backbone layers
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Early exit classifiers
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16 * block.expansion, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32 * block.expansion, num_classes)
        )

        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Layer1 + exit1
        x1 = self.layer1(x)
        logits1 = self.exit1(x1)

        # Layer2 + exit2
        x2 = self.layer2(x1)
        logits2 = self.exit2(x2)

        # Layer3 + final
        x3 = self.layer3(x2)
        out = self.avgpool(x3)
        out = torch.flatten(out, 1)
        logits3 = self.fc(out)

        return [logits1, logits2, logits3]


def _early_exit_resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str]=None,
    pretrained: bool = False,
    num_classes: int = 10,
    progress: bool = True,
    **kwargs: Any
) -> EarlyExitResNet:
    model = EarlyExitResNet(BasicBlock, layers, num_classes=num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def cifar10_eeresnet20(*args, **kwargs) -> EarlyExitResNet: pass
def cifar10_eeresnet32(*args, **kwargs) -> EarlyExitResNet: pass
def cifar10_eeresnet44(*args, **kwargs) -> EarlyExitResNet: pass
def cifar10_eeresnet56(*args, **kwargs) -> EarlyExitResNet: pass

def cifar100_eeresnet20(*args, **kwargs) -> EarlyExitResNet: pass
def cifar100_eeresnet32(*args, **kwargs) -> EarlyExitResNet: pass
def cifar100_eeresnet44(*args, **kwargs) -> EarlyExitResNet: pass
def cifar100_eeresnet56(*args, **kwargs) -> EarlyExitResNet: pass

def tinyimagenet_eeresnet20(*args, **kwargs) -> EarlyExitResNet: pass
def tinyimagenet_eeresnet32(*args, **kwargs) -> EarlyExitResNet: pass
def tinyimagenet_eeresnet44(*args, **kwargs) -> EarlyExitResNet: pass
def tinyimagenet_eeresnet56(*args, **kwargs) -> EarlyExitResNet: pass

# Register constructors
thismodule = sys.modules[__name__]
datasets = {
    "cifar10": {"num_classes": 10,  "constructor": _early_exit_resnet},
    "cifar100": {"num_classes": 100, "constructor": _early_exit_resnet},
    "tinyimagenet": {"num_classes": 200, "constructor": _early_exit_resnet}
}

for dataset, params in datasets.items():
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["eeresnet20", "eeresnet32", "eeresnet44", "eeresnet56"]):
        method_name = f"{dataset}_{model_name}"
        setattr(
            thismodule,
            method_name,
            partial(
                params["constructor"],
                arch=model_name,
                layers=layers,
                num_classes=params["num_classes"]
            )
        )

