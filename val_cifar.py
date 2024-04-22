import argparse

from pathlib import Path
from tqdm import tqdm
import random

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

from slimmable_networks.utils.meters import ScalarMeter, flush_scalar_meters

import models
import data

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cifar100_usvgg11_bn')
    parser.add_argument('--checkpoint', type=str, default='/home/dannya1/slimmable-nn/runs/Apr17_15-08-27_poison.ics.uci.edu_cifar100_usvgg11_bn/weights/last.pt')
    parser.add_argument('--dataset', type=str, default='cifar100')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def get_meters(device: torch.device, phase: str, dataset: str):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = 0
        for k in [1, 5]:
            meters[f'top{k}_accuracy'] = Accuracy(task='multiclass', num_classes=data.n_cls[dataset], top_k=k)
            if torch.cuda.is_available():
                meters[f'top{k}_accuracy'] = meters[f'top{k}_accuracy'].to(device = device)
            if phase == 'val':
                meters[f'best_top{k}_accuracy'] = 0
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = {}
    for width_mult in ['0.25', '1.0']:
        meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    return meters

def forward_loss(model: nn.Module, criterion: nn.Module, inputs: torch.tensor, labels: torch.tensor, meter: dict):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Track training/validation accuracy
    if meter is not None:
        # Update accuracies
        for k in [1, 5]:
            meter[f'top{k}_accuracy'].update(outputs, labels)
        # Update running loss    
        meter['loss'] += loss
        
        # Update best model metrics
        for k in [1, 5]:
            if f'best_top{k}_accuracy' in meter:
                meter[f'best_top{k}_accuracy'] = meter[f'top{k}_accuracy'].compute() if meter[f'top{k}_accuracy'].compute() > meter[f'best_top{k}_accuracy'] else meter[f'best_top{k}_accuracy']
    
    return loss        

def prepare_for_validation(device: torch.device, model_name: str, dataset: str, checkpoint_path: str) \
    -> tuple[nn.Module, nn.Module, nn.Module, object, DataLoader, DataLoader]:
    # Load the model
    model = getattr(models, model_name)(pretrained=False)
    
    # Get the data loaders
    _, val_loader = getattr(data, dataset)()
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['params'])
    
    # Move to cuda if available
    if torch.cuda.is_available():
        model.to(device = device)
    
    return model, val_loader

@torch.no_grad()    
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int):
    # Eval mode
    model.eval()
    
    # Define min and max width
    # TODO: Replace with user-defined variables
    min_width = 0.25
    max_width = 1.0
    
    # Define the quality metrics\
    metrics = {}
    for width_mult in [max_width, min_width]:
        metrics[f'{width_mult}'] = Accuracy(task='multiclass', num_classes=100, top_k=1)
    
    # Evaluate at smallest and largest width
    for width_mult in [max_width, min_width]:
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        
        # Run 1 epoch
        for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
            # Load validation data sample and label
            inputs, labels = data
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            
            # Get the predictions
            outputs = model(inputs)
            
            # Update the metrics
            metrics[f'{width_mult}'].update(outputs, labels)
            
    return metrics

def main(opt: argparse.Namespace):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # TODO: Make min and max widths user-defined
    min_width = 0.25
    max_width = 1.0
    
    # Set up training
    # TODO: Try to unify the meter and model initializations
    model, val_loader = prepare_for_validation(device, opt.model, opt.dataset, opt.checkpoint)

    # Eval
    result = evaluate(model, val_loader, device, 0)

    # Print results
    for width_mult in [max_width, min_width]:
        print(f'Width {width_mult}x : {result[str(width_mult)].compute()}')
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)