import sys
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional
from slimmable_networks.models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear

# cifar10_pretrained_weight_urls = {
#     'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
#     'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
#     'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
#     'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
# }

# cifar100_pretrained_weight_urls = {
#     'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
#     'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
#     'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
#     'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
# }


def conv3x3(in_planes, out_planes, stride=1, us=[True, True]):
    """3x3 convolution with padding"""
    return USConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, us=us)


def conv1x1(in_planes, out_planes, stride=1, us=[True, True]):
    """1x1 convolution"""
    return USConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, us=us)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = USBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = USBatchNorm2d(planes)
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
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, img_size=32):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16, us=[False, True])
        self.bn1 = USBatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = USLinear(64 * block.expansion, num_classes, us=[True, False])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, USBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                USBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    layers: List[int],
    # model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    # TODO: Add functionality for loading pretrained models
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def cifar10_usresnet20(*args, **kwargs) -> CifarResNet: pass
def cifar10_usresnet32(*args, **kwargs) -> CifarResNet: pass
def cifar10_usresnet44(*args, **kwargs) -> CifarResNet: pass
def cifar10_usresnet56(*args, **kwargs) -> CifarResNet: pass


def cifar100_usresnet20(*args, **kwargs) -> CifarResNet: pass
def cifar100_usresnet32(*args, **kwargs) -> CifarResNet: pass
def cifar100_usresnet44(*args, **kwargs) -> CifarResNet: pass
def cifar100_usresnet56(*args, **kwargs) -> CifarResNet: pass


def tinyimagenet_usresnet20(*args, **kwargs) -> CifarResNet: pass
def tinyimagenet_usresnet32(*args, **kwargs) -> CifarResNet: pass
def tinyimagenet_usresnet44(*args, **kwargs) -> CifarResNet: pass
def tinyimagenet_usresnet56(*args, **kwargs) -> CifarResNet: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100", "tinyimagenet"]:
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["usresnet20", "usresnet32", "usresnet44", "usresnet56"]):
        method_name = f"{dataset}_{model_name}"
        # model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        if dataset == "cifar10":
            num_classes = 10 
            img_size = 32
        elif dataset == "cifar100":
            num_classes = 100
            img_size = 32
        elif dataset == "tinyimagenet":
            num_classes = 200
            img_size = 64
        setattr(
            thismodule,
            method_name,
            partial(_resnet,
                    arch=model_name,
                    layers=layers,
                    # model_urls=model_urls,
                    num_classes=num_classes,
                    img_size=img_size)
        )
        
        
def tinyimagenet_usresnet56_x25(*args, **kwargs) -> CifarResNet: 
    model = tinyimagenet_usresnet56(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.25))
    return model
    
def tinyimagenet_usresnet56_x50(*args, **kwargs) -> CifarResNet: 
    model = tinyimagenet_usresnet56(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.50))
    return model
    
def tinyimagenet_usresnet56_x75(*args, **kwargs) -> CifarResNet: 
    model = tinyimagenet_usresnet56(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.75))
    return model

def tinyimagenet_usresnet56_x100(*args, **kwargs) -> CifarResNet: 
    model = tinyimagenet_usresnet56(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 1.00))
    return model
