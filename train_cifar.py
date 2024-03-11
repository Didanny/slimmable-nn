import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

from pytorch_cifar_models import pytorch_cifar_models

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    return


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)