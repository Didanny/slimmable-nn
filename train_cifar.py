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
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--inplace-distill', action='store_true')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def get_meters(device: torch.device, phase: str):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = 0
        for k in [1, 5]:
            # TODO: Extract num_classes from the dataset
            meters[f'top{k}_accuracy'] = Accuracy(task='multiclass', num_classes=100, top_k=k)
            if torch.cuda.is_available():
                meters[f'top{k}_accuracy'] = meters[f'top{k}_accuracy'].to(device = device)
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = {}
    for width_mult in ['0.25', '1.0']:
        meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    # if phase == 'val':
    #     meters['best_val'] = 0
    return meters

def forward_loss(model: nn.Module, criterion: nn.Module, inputs: torch.tensor, labels: torch.tensor, meter: dict):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # TODO: Add tracking of training error/accuracy here
    if meter is not None:
        # Update accuracies
        for k in [1, 5]:
            meter[f'top{k}_accuracy'].update(outputs, labels)
        # Update running loss    
        meter['loss'] += loss        
    
    return loss

def log_meters(writer: SummaryWriter, meters: dict, prefix: str, step: int):
    min_width = 0.25
    max_width = 1.0
    
    # Log meters to Tensorboard
    for width_mult in [min_width, max_width]:
        meter = meters[f'{width_mult}']
        writer.add_scalar(f'{prefix}/loss_{width_mult}', meter['loss'], step)
        for k in [1, 5]:
            writer.add_scalar(f'{prefix}/top{k}_accuracy_{width_mult}', meter[f'top{k}_accuracy'].compute(), step)
            
    # Reset meters
    for width_mult in [min_width, max_width]:
        meter = meters[f'{width_mult}']
        meter['loss'] = 0
        for k in [1, 5]:
            meter[f'top{k}_accuracy'].reset()
        

def prepare_for_training(device: torch.device, model_name: str, dataset: str) \
    -> tuple[nn.Module, nn.Module, nn.Module, object, DataLoader, DataLoader]:
    # Load the model
    model = getattr(models, model_name)(pretrained=False)
    
    # Get the data loaders
    train_loader, val_loader = getattr(data, dataset)()
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Move to cuda if available
    if torch.cuda.is_available():
        model.to(device = device)
        criterion.to(device = device)
        
    # Initialize optimizer
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    
    # Initialize scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    
    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader

def train(model: nn.Module, criterion: nn.Module, optimizer: nn.Module, scheduler: object, train_loader: DataLoader, 
          device: torch.device, epoch: int, meters: dict, inplace_distill: bool) -> None:
    # Training mode
    model.train()
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}'), 0):
        # Load training data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        
        # Define min and max width
        # TODO: Replace with user-defined variables
        min_width = 0.25
        max_width = 1.0
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Load the widths for universally slimmable training
        widths_train = []
        # TODO: replace '4' here with user-defined variable
        for _ in range(4 - 2):
            widths_train.append(random.uniform(min_width, max_width))
        widths_train = [min_width, max_width] + widths_train
        
        # Train at each width
        for width_mult in widths_train:
            # The sandwich rule
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            
            # Track largest and smallest model
            if width_mult in [min_width, max_width]:
                meter = meters[f'{width_mult}']
            else:
                meter = None
                
            # In-place distillation
            if width_mult == max_width:
                loss = forward_loss(model, criterion, inputs, labels, meter)
            else:
                if inplace_distill:
                    # TODO: Implement inplace distillation
                    raise NotImplementedError
                else:
                    loss = forward_loss(model, criterion, inputs, labels, meter)
        
            loss.backward()
        
        optimizer.step()
        scheduler.step()
        
    # # Reset meters
    # for width_mult in [min_width, max_width]:
    #     meter = meters[f'{width_mult}']
    #     for k in [1, 5]:
    #         meter[f'top{k}_accuracy'].reset()
    #         meter['loss'] = 0

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
        
        # Define min and max width
        # TODO: Replace with user-defined variables
        min_width = 0.25
        max_width = 1.0
        
        # Evaluate at smallest and largest width
        for width_mult in [max_width, min_width]:
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            
            # Track largest and smallest model
            meter = meters[f'{width_mult}']
                
            # Get loss
            loss = forward_loss(model, criterion, inputs, labels, meter)

def main(opt: argparse.Namespace):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set up tensorboard summary writer
    # TODO: Create more comprhensive automated commenting
    writer = SummaryWriter(comment=f'_{opt.model}')
    save_dir = Path(writer.log_dir)
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Set up training
    # TODO: Try to unify the meter and model initializations
    model, criterion, optimizer, lr_scheduler, train_loader, val_loader = prepare_for_training(device, opt.model, opt.dataset) 
    train_meters = get_meters(device, 'train')
    val_meters = get_meters(device, 'val')
    
    # TODO: Log initial accuracy
    
    # Begin training
    # TODO: Make 'epochs' user-defined
    epochs = 300
    for epoch in range(epochs):
        # Train
        train(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, train_meters, False)
        
        # Eval
        evaluate(model, criterion, val_loader, device, epoch, val_meters)
        
        # Tensorboard
        log_meters(writer, train_meters, 'train', epoch)
        log_meters(writer, val_meters, 'val', epoch)        
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)