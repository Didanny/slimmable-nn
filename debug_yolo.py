import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

from models import USDetectionModel
from yolov5.models.yolo import Model

import torch
import torch.nn as nn
from utils import Flags

import os
os.chdir('..')

from slimmable_networks.models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear

from yolov5.models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
    autopad,
)
from yolov5.models.experimental import MixConv2d
from yolov5.utils.autoanchor import check_anchor_order
from yolov5.utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from yolov5.utils.plots import feature_visualization
from yolov5.utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="us_yolov5n.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    nano_cfg = check_yaml('yolov5n.yaml')
    print_args(vars(opt))
    device = select_device(opt.device)
    
    FLAGS = Flags()
    FLAGS.width_mult_list = [0.25, 1.0]

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = USDetectionModel(opt.cfg).to(device)
    model.apply(lambda m: setattr(m, 'width_mult', 0.8378489417760795))
    # nano_model = Model(nano_cfg).to(device)
    model(im)

    # # Options
    # if opt.line_profile:  # profile layer by layer
    #     model(im, profile=True)

    # elif opt.profile:  # profile forward-backward
    #     results = profile(input=im, ops=[model], n=3)

    # elif opt.test:  # test all models
    #     for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
    #         try:
    #             _ = DetectionModel(cfg)
    #         except Exception as e:
    #             print(f"Error in {cfg}: {e}")

    # else:  # report fused model summary
    #     model.fuse()