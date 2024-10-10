import argparse

from pathlib import Path
from tqdm import tqdm
import random
import csv

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.classification import Accuracy

from slimmable_networks.utils.loss_ops import CrossEntropyLossSoft
from slimmable_networks.models.slimmable_ops import bn_calibration_init

import models
import data
from utils import Flags

default_width_mult_list = [0.25, 1.0, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
default_model_list = [
    'usresnet20',
    'usresnet32',
    'usresnet44',
    'usresnet56',
    'usvgg11_bn',
    'usvgg13_bn',
    'usvgg16_bn',
    'usvgg19_bn',
]

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='*', default=default_model_list)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--widths', nargs='*', type=float, default=default_width_mult_list)
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--no-cal', action='store_true')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def get_meters(device: torch.device, phase: str, dataset: str):
    """util function for meters"""
    FLAGS = Flags()
    
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = 0
        for k in [1, 5]:
            meters[f'top{k}_accuracy'] = Accuracy(task='multiclass', num_classes=data.n_cls[dataset], top_k=k)
            if torch.cuda.is_available():
                meters[f'top{k}_accuracy'] = meters[f'top{k}_accuracy'].to(device = device)
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = {}
    for width_mult in FLAGS.width_mult_list:
        meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    return meters

def flush_meters(meters):
    FLAGS = Flags()
    
    for width_mult in FLAGS.width_mult_list:
        for k in [1, 5]:
            meters[str(width_mult)][f'top{k}_accuracy'] = meters[str(width_mult)][f'top{k}_accuracy'].compute()
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
            meter[f'top{k}_accuracy'].update(outputs.detach(), labels.detach())
        # Update running loss    
        meter['loss'] += loss.detach()

    # if return_soft_target:
    #     return loss, torch.nn.functional.softmax(outputs, dim=1)
    # else:
    #     return loss
     
def evaluate(model: nn.Module, criterion: nn.Module, loader: DataLoader, device: torch.device, stage: str, meters: dict):
    # Define min and max width
    FLAGS = Flags()
    min_width = FLAGS.width_mult_range[0]
    max_width = FLAGS.width_mult_range[1]
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(loader, desc=f'{stage.upper()}'), 0):
        # Load validation data sample and label
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True) 
        
        # Evaluate at smallest and largest width
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
           
            # Apply the 
            model.apply(lambda m: setattr(m, 'width_mult', width_mult))
        
            # Track largest and smallest model
            meter = meters[f'{width_mult}']            
                
            # Get loss
            # loss = 
            forward_loss(model, criterion, inputs, labels, meter)
            # if stage == 'cal':
                # loss.backward()
            

def main(opt: argparse.Namespace):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set flags
    FLAGS = Flags()
    FLAGS.width_mult_range = [0.25, 1.0]
    FLAGS.width_mult_list = [0.25, 1.0]
    FLAGS.width_mult_list = opt.widths
    
    # Results
    results = [[model] for model in opt.models]
    # print(results)
    
    # Preprocess moel names
    opt.models = [f'{opt.dataset}_' + n for n in opt.models]
    
    # Get the data loaders
    train_loader, val_loader = getattr(data, opt.dataset)()
    
    # Get the criterion
    criterion = nn.CrossEntropyLoss()
    
    for i, model_name in enumerate(opt.models):
        # Load the model
        model = getattr(models, model_name)(pretrained=False)
        
        # Move to cuda if available
        if torch.cuda.is_available():
            model.to(device = device)
        
        # Load the checkpoint
        best_or_last = 'best' if opt.best else 'last'
        checkpoint = torch.load(f'./runs/{model_name}_{opt.dataset}/weights/{best_or_last}.pt', map_location=device)
        
        # Record all model accuracies 
        model.load_state_dict(checkpoint['params'], strict=False)
            
        # Get the calibration meters
        cal_meters = get_meters(device, 'cal', opt.dataset)
        val_meters = get_meters(device, 'val', opt.dataset)

        # Begin calibration
        if not opt.no_cal:
            model.eval()
            model.apply(bn_calibration_init)
        
            # Run calibration epoch
            evaluate(model, criterion, train_loader, device, 'cal', cal_meters)
        
        # Run validation epoch
        model.eval()
        with torch.no_grad():
            evaluate(model, criterion, val_loader, device, 'val', val_meters)
        
        flush_meters(val_meters)
        results[i] += [val_meters[str(width)]['top1_accuracy'].item() for width in sorted(FLAGS.width_mult_list, reverse=True)]    
        # print(results[i]) 
        
    # Save results
    with open('./results/us_results_cifar.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(['Model Name'] + [str(w) for w in FLAGS.width_mult_list])
        for row in results:
            writer.writerow(row)
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)