import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast
from slimmable_networks.models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear

# cifar10_pretrained_weight_urls = {
#     'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
#     'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
#     'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
#     'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
# }

# cifar100_pretrained_weight_urls = {
#     'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
#     'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
#     'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
#     'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
# }


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        img_size: int  = 32,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        if img_size == 64:
            self.features = nn.Sequential(
                *self.features,
                USConv2d(512, 512, kernel_size=2, padding=0),
                USBatchNorm2d(512), 
                nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            USLinear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            USLinear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            USLinear(512, num_classes, us=[True, False])
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, USBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    # First layer
    v = cfg[0]
    v = cast(int, v)
    conv2d = USConv2d(in_channels, v, kernel_size=3, padding=1, us=[False, True])
    if batch_norm:
        layers += [conv2d, USBatchNorm2d(v), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]
    in_channels = v

    # Rest of the layers
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = USConv2d(in_channels, v, kernel_size=3, padding=1)
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, USBatchNorm2d(v), nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool,
        #  model_urls: Dict[str, str],
         pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    # TODO: Add funcitonality for loading pretrained models
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def cifar10_usvgg11_bn(*args, **kwargs) -> VGG: pass
def cifar10_usvgg13_bn(*args, **kwargs) -> VGG: pass
def cifar10_usvgg16_bn(*args, **kwargs) -> VGG: pass
def cifar10_usvgg19_bn(*args, **kwargs) -> VGG: pass


def cifar100_usvgg11_bn(*args, **kwargs) -> VGG: pass
def cifar100_usvgg13_bn(*args, **kwargs) -> VGG: pass
def cifar100_usvgg16_bn(*args, **kwargs) -> VGG: pass
def cifar100_usvgg19_bn(*args, **kwargs) -> VGG: pass


def tinyimagenet_usvgg11_bn(*args, **kwargs) -> VGG: pass
def tinyimagenet_usvgg13_bn(*args, **kwargs) -> VGG: pass
def tinyimagenet_usvgg16_bn(*args, **kwargs) -> VGG: pass
def tinyimagenet_usvgg19_bn(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100", "tinyimagenet"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["usvgg11_bn", "usvgg13_bn", "usvgg16_bn", "usvgg19_bn"]):
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
            partial(_vgg,
                    arch=model_name,
                    cfg=cfg,
                    batch_norm=True,
                    # model_urls=model_urls,
                    num_classes=num_classes,
                    img_size=img_size)
        )

def tinyimagenet_usvgg16_bn_x25(*args, **kwargs) -> VGG: 
    model = tinyimagenet_usvgg16_bn(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.25))
    return model
    
def tinyimagenet_usvgg16_bn_x50(*args, **kwargs) -> VGG: 
    model = tinyimagenet_usvgg16_bn(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.50))
    return model
    
def tinyimagenet_usvgg16_bn_x75(*args, **kwargs) -> VGG: 
    model = tinyimagenet_usvgg16_bn(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 0.75))
    return model
    
def tinyimagenet_usvgg16_bn_x100(*args, **kwargs) -> VGG: 
    model = tinyimagenet_usvgg16_bn(*args, **kwargs)
    model.apply(lambda m: setattr(m, 'width_mult', 1.00))
    return model
