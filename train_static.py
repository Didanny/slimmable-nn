import argparse
from pathlib import Path
from tqdm import tqdm
import socket
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric

import models
import data
from utils import Flags

from torch.utils.tensorboard import SummaryWriter

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cifar100_resnet20')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--vit-scheduler', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--width', type=float)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def make_log_dir(base="runs/static_runs", dataset="cifar100", model="usresnet32"):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")  
    hostname = socket.gethostname()                       
    log_dir = f"{base}/{dataset}/{timestamp}_{hostname}_{model}"
    return log_dir

def get_meters(device: torch.device, phase: str, num_classes: int):
    """util function for meters"""
    def get_single_meter(phase):
        meters = {}
        meters['loss'] = MeanMetric()
        meters['loss'] = meters['loss'].to(device = device)
        for k in [1, 5]:
            meters[f'top{k}_accuracy'] = Accuracy(task='multiclass', num_classes= num_classes, top_k=k)
            meters[f'top{k}_accuracy'] = meters[f'top{k}_accuracy'].to(device = device)
        return meters

    assert phase in ['train', 'val', 'test'], 'Invalid phase.'
    return get_single_meter(phase)
    
def log_meters(writer: SummaryWriter, meters: dict, prefix: str, step: int):    
    # Log meters to Tensorboard
    writer.add_scalar(f'{prefix}/loss', meters['loss'].compute(), step)
    for k in [1, 5]:
        writer.add_scalar(f'{prefix}/top{k}_accuracy', meters[f'top{k}_accuracy'].compute(), step)
            
    # Reset meters
    for m in meters.values():
        m.reset()
        
def log_sparsity(writer: SummaryWriter, sparsities: dict, prefix: str, step: int):
    # Log sparsity values to Tensorboard
    for key, val in sparsities.items():
        writer.add_scalar(f'{prefix}/{key}', val, step)

def train(model: nn.Module, criterion: nn.Module, optimizer: nn.Module, scheduler: object, train_loader: DataLoader, 
          device: torch.device, epoch: int, meters: dict) -> None:
    # Training mode
    model.train()
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}'), 0):
        # Load training data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Get loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        # Backprop
        loss.backward()
        
        # Update params
        optimizer.step()
        
        # Update the metrics
        if meters is not None:
            # Update accuracies
            for k in [1, 5]:
                meters[f'top{k}_accuracy'].update(outputs, labels)
            # Update running loss    
            meters['loss'].update(loss)
    
    # Update learning rate
    if scheduler:
        scheduler.step()
            
@torch.no_grad()    
def evaluate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, meters: dict):
    # Eval mode
    model.eval()
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
        # Load validation data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
            
        # Get loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Update the metrics
        if meters is not None:
            # Update accuracies
            for k in [1, 5]:
                meters[f'top{k}_accuracy'].update(outputs, labels)
            # Update running loss    
            meters['loss'].update(loss)

def main(opt: argparse.Namespace):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set up tensorboard summary writer
    # TODO: Create more comprehensive automated commenting
    writer = SummaryWriter(log_dir=make_log_dir(dataset=opt.dataset, model=opt.model))
    save_dir = Path(writer.log_dir)
    
    # Directories
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    
    # Set up FLAGS
    FLAGS = Flags()
    FLAGS.width_mult_range = [opt.width]
    FLAGS.width_mult_list = [opt.width]
    # FLAGS.inplace_distill = opt.inplace_distill
    # FLAGS.n = opt.n
    FLAGS.epochs = opt.epochs
    # FLAGS.ema = opt.ema
    
    # Dataset
    train_loader, val_loader = getattr(data, opt.dataset)()
    
    # Model
    model = getattr(models, opt.model)(pretrained = opt.pretrained, num_classes = data.n_cls[opt.dataset])
    model.to(device = device)
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    
    # Initialize optimizer
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(
            [v for n, v in model.named_parameters()],
            lr           = 5e-4,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
            weight_decay = 0.05
        )
    
    # Initialize scheduler
    if opt.vit_scheduler:
        iters_per_epoch = len(train_loader)
        warm_iters      = 10 * iters_per_epoch
        total_iters     = 100 * iters_per_epoch
        
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                # 1) linear warm‑up from 1e‑6 × lr -> 1 × lr
                LinearLR(
                    optimizer,
                    start_factor = 1e-6,
                    end_factor   = 1.0,
                    total_iters  = warm_iters
                ),
                # 2) cosine decay down to 0 over the rest of training
                CosineAnnealingLR(
                    optimizer,
                    T_max  = total_iters - warm_iters,
                    eta_min= 0.0
                )
            ],
            milestones=[warm_iters]   # switch point
        )
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    
    # Get the performance metrics
    train_meters = get_meters(device, 'train', data.n_cls[opt.dataset])
    val_meters = get_meters(device, 'val', data.n_cls[opt.dataset])
    
    # Initialize best and last model metrics
    best_dict, last_dict, best_fitness = None, None, 0.0
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Begin training
    for epoch in range(opt.epochs):
        # Train
        train(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, train_meters)
        log_meters(writer, train_meters, 'train', epoch)
        
        # Eval
        if (epoch + 1) % 5 == 0:
            evaluate(model, criterion, val_loader, device, epoch, val_meters)
        
            # Get fitness
            fitness = val_meters['top1_accuracy'].compute()
            top5_accuracy, top1_accuracy = val_meters['top5_accuracy'].compute(), val_meters['top1_accuracy'].compute() 

            # Update best model
            if best_fitness < fitness:
                best_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
                best_fitness = fitness
                torch.save(best_dict, best)
                
            # Log val meters
            log_meters(writer, val_meters, 'val', epoch) 
            
    # Update last model
    last_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
    torch.save(last_dict, last)
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    