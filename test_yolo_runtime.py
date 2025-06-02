# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
print(os.getcwd())
import platform
import sys
from copy import deepcopy
from pathlib import Path
import itertools
import random
from tqdm import tqdm

import torch
import torch.nn as nn


from slimmable_networks.models.slimmable_ops import (
    USBatchNorm2d, 
    USConv2d, 
    USLinear, 
    USBatchNorm2dRuntime, 
    USConv2dRuntime, 
    USLinearRuntime,
)

class test: pass
FLAGS = test()
setattr(FLAGS, 'width_mult_list', [0.25, 1.0])
setattr(FLAGS, 'cumulative_bn_stats', True)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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

from utils import Profile

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

from utils import Flags    
from models import USDetectionModelRuntime

default_width_mult_list = [0.25, 1.0, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="us_yolov5s_runtime.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument('--iters', type=int)
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)
    
    FLAGS = Flags()
    FLAGS.width_mult_list = default_width_mult_list
    FLAGS.current_profiler = Profile()
    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = USDetectionModelRuntime(opt.cfg).to(device)
    model.apply(lambda m: setattr(m, 'width_mult', 1.0))
    model(im, 1.0)
    
    dummy_input = torch.rand((opt.batch_size, 3, 640, 640))
    
    testing_widths = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    results = {}
    
    # Add all pairs to the results dict
    for pair in list(itertools.combinations(testing_widths, 2)):
        # Add pair
        results[pair] = {
            'running_time': 0.0,
            'occurences': 0
        }
        
        # Add the inverse pair
        results[(pair[1], pair[0])] = {
            'running_time': 0.0,
            'occurences': 0
        }
    
    
    current_width = 1.0
    
    for i in tqdm(range(opt.iters)):
        next_width = random.choice(testing_widths)
        
        # If same width, re-try
        if next_width == current_width:
            continue
        
        # Resize model
        FLAGS.current_profiler = Profile()
        model(im, next_width)
        
        # Record results
        results[(current_width, next_width)]['running_time'] += FLAGS.current_profiler.t
        results[(current_width, next_width)]['occurences'] += 1
        
        current_width = next_width
        
    # Save the results
    torch.save(results, f'./results/yolos_resize_time_{opt.iters}.pt')
    
    