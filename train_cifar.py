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

from slimmable_networks.utils.loss_ops import CrossEntropyLossSoft

import models
import data
from utils import Flags

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cifar100_usvgg11_bn')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--inplace-distill', action='store_true')
    parser.add_argument('--width-mult-range', nargs=2, type=float, default=[0.25, 1.0])
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
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

def forward_loss(model: nn.Module, criterion: nn.Module, inputs: torch.tensor, labels: torch.tensor, meter: dict, 
                 soft_target=None, soft_criterion=None, return_soft_target=False):
    outputs = model(inputs)
    
    if soft_target is None:
        loss = criterion(outputs, labels)
    else:
        loss = soft_criterion(outputs, soft_target)
    
    # Track training/validation accuracy
    if meter is not None:
        # Update accuracies
        for k in [1, 5]:
            meter[f'top{k}_accuracy'].update(outputs, labels)
        # Update running loss    
        meter['loss'] += loss

    if return_soft_target:
        return loss, torch.nn.functional.softmax(outputs, dim=1)
    else:
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
            if prefix == 'val':
                writer.add_scalar(f'{prefix}/best_top{k}_accuracy_{width_mult}', meter[f'best_top{k}_accuracy'], step)
            
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
    
    # Initialize soft criterion
    # TODO: Implement soft verion
    soft_criterion = CrossEntropyLossSoft(reduction='none')
    # soft_criterion = nn.CrossEntropyLoss()
    
    # Move to cuda if available
    if torch.cuda.is_available():
        model.to(device = device)
        criterion.to(device = device)
        soft_criterion.to(device=device)
        
    # Initialize optimizer
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    
    # Initialize scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=0)
    
    return model, criterion, soft_criterion, optimizer, lr_scheduler, train_loader, val_loader

def train(model: nn.Module, criterion: nn.Module, soft_criterion: nn.Module, optimizer: nn.Module, scheduler: object, train_loader: DataLoader, 
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
        FLAGS = Flags()
        min_width = FLAGS.width_mult_range[0]
        max_width = FLAGS.width_mult_range[1]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Load the widths for universally slimmable training
        widths_train = []
        for _ in range(FLAGS.n - 2):
            widths_train.append(random.uniform(min_width, max_width))
        widths_train = [max_width, min_width] + widths_train
        
        # Train at each width
        for width_mult in widths_train:
            # The sandwich rule
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            
            # Track largest and smallest model
            if width_mult in [max_width, min_width]:
                meter = meters[f'{width_mult}']
            else:
                meter = None
                
            # In-place distillation
            if width_mult == max_width:
                loss, soft_target = forward_loss(model, criterion, inputs, labels, meter, return_soft_target=True)
            else:
                if inplace_distill:
                    loss = forward_loss(model, criterion, inputs, labels, meter, soft_target.detach(), soft_criterion)
                else:
                    loss = forward_loss(model, criterion, inputs, labels, meter)
        
            loss.backward()
        
        optimizer.step()
        scheduler.step()

@torch.no_grad()    
def evaluate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int, meters: dict):
    # Eval mode
    model.eval()
    
    # Define min and max width
    FLAGS = Flags()
    min_width = FLAGS.width_mult_range[0]
    max_width = FLAGS.width_mult_range[1]
    
    # Evaluate at smallest and largest width
    for width_mult in [max_width, min_width]:
        
        # Apply the 
        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        model.eval()
        
        # Track largest and smallest model
        meter = meters[f'{width_mult}']
    
        # Run 1 epoch
        for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
            # Load validation data sample and label
            inputs, labels = data
            inputs = inputs.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
                
            # Get loss
            loss = forward_loss(model, criterion, inputs, labels, meter)
            
        # Update best meter
        for k in [1, 5]:
            meter[f'best_top{k}_accuracy'] = meter[f'top{k}_accuracy'].compute() if meter[f'top{k}_accuracy'].compute() > meter[f'best_top{k}_accuracy'] else meter[f'best_top{k}_accuracy']
            

def main(opt: argparse.Namespace):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set up tensorboard summary writer
    # TODO: Create more comprehensive automated commenting
    writer = SummaryWriter(comment=f'_{opt.model}_{opt.dataset}')
    save_dir = Path(writer.log_dir)
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Set up FLAGS
    FLAGS = Flags()
    FLAGS.width_mult_range = opt.width_mult_range
    FLAGS.inplace_distill = opt.inplace_distill
    FLAGS.n = opt.n
    FLAGS.epochs = opt.epochs
    
    # Get max_width
    max_width = FLAGS.width_mult_range[1]
    
    # Set up training
    # TODO: Try to unify the meter and model initializations
    model, criterion, soft_criterion, optimizer, lr_scheduler, train_loader, val_loader = prepare_for_training(device, opt.model, opt.dataset) 
    train_meters = get_meters(device, 'train', opt.dataset)
    val_meters = get_meters(device, 'val', opt.dataset)
    
    # Initialize best and last model metrics
    best_dict = None
    last_dict = None
    best_fitness = 0.0
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Begin training
    for epoch in range(FLAGS.epochs):
        # Train
        train(model, criterion, soft_criterion, optimizer, lr_scheduler, train_loader, device, epoch, train_meters, False)
        
        # Eval
        evaluate(model, criterion, val_loader, device, epoch, val_meters)
        
        # Update best model
        top1_accuracy, top5_accuracy = val_meters[f'{max_width}']['top1_accuracy'].compute(), val_meters[f'{max_width}']['top5_accuracy'].compute()
        fitness = (0.9 * top1_accuracy) + (0.1 * top5_accuracy)
        if best_fitness < fitness:
            best_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
            best_fitness = fitness
            
        # Update last model
        if epoch == FLAGS.epochs - 1:
            last_dict = {'params': model.state_dict(), 'top5_accuracy': top5_accuracy, 'top1_accuracy': top1_accuracy, 'epoch': epoch}
        
        # Tensorboard
        log_meters(writer, train_meters, 'train', epoch)
        log_meters(writer, val_meters, 'val', epoch)      
        
    # Save the best and last
    torch.save(best_dict, best)
    torch.save(last_dict, last)
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)