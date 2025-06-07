from .cifar import cifar10, cifar100
from .tinyimagenet import tinyimagenet_hf as tinyimagenet

n_cls = {
    'cifar10': 10,
    'cifar100': 100,
    'tinyimagenet': 200,
}