import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import random
from datetime import datetime
import socket

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
import torch.nn.utils.prune as prune
from tqdm import tqdm

from models import USDetectionModel
from yolov5.models.yolo import DetectionModel

import yolov5.val as validate  # for end-of-epoch mAP
from yolov5.models.experimental import attempt_load
from yolov5.utils.autoanchor import check_anchors
from yolov5.utils.autobatch import check_train_batch_size
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.downloads import attempt_download, is_url
from yolov5.utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from yolov5.utils.loggers import Loggers
from yolov5.utils.loggers.comet.comet_utils import check_comet_resume
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import fitness
from yolov5.utils.plots import plot_evolve
from yolov5.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

ROOT = Path('./yolov5')

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

from utils import Flags

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    
    # Slimmable training arguments
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--dp', action='store_true')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def get_meters(device: torch.device, phase: str):
    """util function for meters"""
    
    metrics = ['precision', 'recall', 'mAP@.5', 'mAP@.5-.95', 'loss1', 'loss2', 'loss3']
    
    def get_single_meter(phase, suffix=''):
        meters = {}
        for m in metrics:
            meters[m] = 0.0
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = {}
    for width_mult in ['0.25', '1.0']:
        meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    return meters

def log(writer: SummaryWriter, meters: dict, step: int):
    min_width = 0.25
    max_width = 1.0
    
    for width_mult in [min_width, max_width]:
        meter = meters[str(width_mult)]
        for k in meter:
            writer.add_scalar(f'val/{width_mult}_{k}', meter[k], step)

def main(rank: int, world_size: int, opt: argparse.Namespace):
    # Print args
    print_args(vars(opt))
    
    ddp_setup(rank, world_size)
    
    os.environ['TEST_VAR'] = str(rank)
    print(os.getenv('CUDA_VISIBLE_DEVICES', -1))
    
    # Set up FLAGS
    FLAGS = Flags()
    FLAGS.width_mult_range = [0.25, 1.0]
    FLAGS.width_mult_list = FLAGS.width_mult_range
    FLAGS.n = opt.n
    FLAGS.epochs = opt.epochs
    FLAGS.rank = rank
    FLAGS.world_size = world_size
    
    # Compatability with yolov5 base code
    RANK = FLAGS.rank
    LOCAL_RANK = FLAGS.rank
    WORLD_SIZE = FLAGS.world_size
    
    # Initialize Tensorboard writer
    writer = SummaryWriter(log_dir=f'runs_yolo/{datetime.strftime(datetime.now(), "%b_%d_%X")}_{socket.gethostname()}', comment=f'_{opt.weights}')
    opt.save_dir = writer.log_dir
    
    # Prepare options
    opt.hyp = ROOT / opt.hyp
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    # opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # Get current device
    device = torch.device('cuda', FLAGS.rank)
    
    # Begin Training
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Hyperparameters
    hyp = opt.hyp
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    
    # Save run settings
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Config
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + FLAGS.rank, deterministic=True)
    with torch_distributed_zero_first(FLAGS.rank):
        data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Check if model is slimmable or not
    if opt.cfg.split('/')[-1].startswith('us'):
        Model = USDetectionModel
    else:
        Model = DetectionModel
    
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        raise NotImplementedError
        # weights = attempt_download('./runs/Oct30_12-41-58_poison.ics.uci.edu_yolov5s.pt/weights/best_23.pt')  # download if not found locally
        # ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        # model = Model(cfg or ckpt['ema'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # for name, module in model.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         prune.identity(module, 'weight')
        #         if module.bias != None:
        #             prune.identity(module, 'bias')
        # exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # csd = ckpt['ema'].float().state_dict()  # checkpoint state_dict as FP32
        # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # model.load_state_dict(csd, strict=False)  # load
        # LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        
    # Warmup to apply the masks
    # model(torch.rand(1,3,640,640).to(device=device, non_blocking=True))
    
    amp = check_amp(model)  # check AMP
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Batch size
    # if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
    #     batch_size = check_train_batch_size(model, imgsz, amp)
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    
    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    # EMA
    ema = ModelEMA(model) if FLAGS.rank in {-1, 0} else None
    
    # Resume
    best_fitness, start_epoch = 0.0, 0
    
    # Train loader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // FLAGS.world_size,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=FLAGS.rank,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    
    # Validation loader
    if FLAGS.rank == 0:
        val_loader = create_dataloader(val_path,
                                        imgsz,
                                        batch_size // FLAGS.world_size * 2,
                                        gs,
                                        single_cls,
                                        hyp=hyp,
                                        cache=None if noval else opt.cache,
                                        rect=True,
                                        rank=-1,
                                        workers=workers * 2,
                                        pad=0.5,
                                        prefix=colorstr('val: '))[0]
    if not resume:
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
        model.half().float()  # pre-reduce anchor precision
        
    # DDP
    model = smart_DDP(model)
        
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    global_step = 0
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    compute_loss = ComputeLoss(model)  # init loss class
        
    # Initialize model checkpoints
    best_fitness = 0.0
    last, best = w / f'last.pt', w / f'best.pt'
    
    # Get the flags
    FLAGS = Flags()
    min_width = FLAGS.width_mult_range[0]
    max_width = FLAGS.width_mult_range[1]
    
    # Get the meters
    val_meters = get_meters(device, 'val')
        
    for epoch in range(epochs):
        model.train()
        
        mloss = torch.zeros(3, device=device)  # mean losses
        
        train_loader.sampler.set_epoch(epoch)
        
        # Progress bar
        pbar = enumerate(train_loader)
        if FLAGS.rank == 0:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        
        # Training batch
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Build the training widths list
            widths_train = []
            for _ in range(FLAGS.n - 2):
                widths_train.append(random.uniform(min_width, max_width))
            widths_train = [max_width, min_width] + widths_train
            # widths_train = [1.0]

            # Train at each width
            for width_mult in widths_train:
            # for width_mult in [1.0]:
                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    
                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    loss *= FLAGS.world_size
                
                # Backward
                scaler.scale(loss).backward()
                
            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # print(ni - last_opt_step >= accumulate, ni, last_opt_step, accumulate)
            # if ni - last_opt_step >= accumulate:
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            last_opt_step = ni
                    
            # Log
            if FLAGS.rank == 0:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                        (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
            
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
            
        # Validation
        if FLAGS.rank == 0:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs)
        
            for width_mult in sorted(FLAGS.width_mult_range, reverse=True):
                # Validate 
                ema.ema.apply(lambda m: setattr(m, 'width_mult', width_mult))
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // FLAGS.world_size * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                # model = model,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=Callbacks(),
                                                compute_loss=compute_loss)
                
                # Store full model performance
                if width_mult == 1.0:
                    results_full = results
                
                # Compile results into meters
                for k, r in zip(['precision', 'recall', 'mAP@.5', 'mAP@.5-.95', 'loss1', 'loss2', 'loss3'], results):
                    val_meters[str(width_mult)][k] = r
            
            # Update best mAP
            fi = fitness(np.array(results_full).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results_full)
            
            # Tensorboard
            log(writer, val_meters, global_step)
        global_step += 1
        
        # Save
        if FLAGS.rank == 0:
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)),
                'params': deepcopy(ema.ema.state_dict()),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'date': datetime.now().isoformat()}
            
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
            
        # End epoch
    # End training cycle
    
    dist.destroy_process_group()
    
    
if __name__ == '__main__':
    opt = parse_opt()
    # main(opt)
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, opt), nprocs=world_size)