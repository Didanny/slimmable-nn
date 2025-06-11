import argparse
from pathlib import Path
from tqdm import tqdm
import socket
from datetime import datetime

import torch
from torch import nn, optim
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


def make_log_dir(base="runs/earlyexit_runs", dataset="cifar100", model="usresnet32"):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    hostname = socket.gethostname()
    return f"{base}/{dataset}/{timestamp}_{hostname}_{model}"


def create_meters(n_exits: int, device: torch.device, num_classes: int):
    meters = {}
    # loss shared across exits
    meters['loss'] = MeanMetric().to(device)
    # accuracy per exit and top-k
    for i in range(n_exits):
        for k in [1, 5]:
            key = f'exit{i+1}_top{k}'
            meters[key] = Accuracy(task='multiclass', num_classes=num_classes, top_k=k).to(device)
    return meters


def log_meters(writer: SummaryWriter, meters: dict, prefix: str, step: int):
    writer.add_scalar(f'{prefix}/loss', meters['loss'].compute(), step)
    for key, meter in meters.items():
        if key == 'loss':
            continue
        writer.add_scalar(f'{prefix}/{key}', meter.compute(), step)
    # reset
    for meter in meters.values():
        meter.reset()


def train(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
          scheduler: object, loader: DataLoader, device: torch.device,
          epoch: int, meters: dict) -> None:
    model.train()
    for inputs, labels in tqdm(loader, desc=f'Train Epoch {epoch}'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # list of logits
        # compute loss: sum over exits
        loss = sum(criterion(o, labels) for o in outputs)
        loss.backward()
        optimizer.step()
        if meters:
            meters['loss'].update(loss)
            for idx, out in enumerate(outputs):
                for k in [1, 5]:
                    meters[f'exit{idx+1}_top{k}'].update(out, labels)
    if scheduler:
        scheduler.step()


@torch.no_grad()
def evaluate(model: nn.Module, criterion: nn.Module, loader: DataLoader,
             device: torch.device, epoch: int, meters: dict):
    model.eval()
    for inputs, labels in tqdm(loader, desc=f'Val Epoch {epoch}'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = sum(criterion(o, labels) for o in outputs)
        if meters:
            meters['loss'].update(loss)
            for idx, out in enumerate(outputs):
                for k in [1, 5]:
                    meters[f'exit{idx+1}_top{k}'].update(out, labels)


def main(opt: argparse.Namespace):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    writer = SummaryWriter(log_dir=make_log_dir(dataset=opt.dataset, model=opt.model))
    save_dir = Path(writer.log_dir)
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)

    FLAGS = Flags()
    FLAGS.width_mult_range = [opt.width]
    FLAGS.width_mult_list = [opt.width]
    FLAGS.epochs = opt.epochs

    train_loader, val_loader = getattr(data, opt.dataset)()
    model = getattr(models, opt.model)(pretrained=opt.pretrained,
                                        num_classes=data.n_cls[opt.dataset])
    model.to(device)

    # determine number of exits
    sample_inputs, _ = next(iter(train_loader))
    with torch.no_grad():
        sample_outs = model(sample_inputs.to(device))
    n_exits = len(sample_outs)

    # meters per exit
    train_meters = create_meters(n_exits, device, data.n_cls[opt.dataset])
    val_meters = create_meters(n_exits, device, data.n_cls[opt.dataset])

    criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)

    # optimizer
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=5e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=5e-4,
                                betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.05)

    # scheduler
    if opt.vit_scheduler:
        iters = len(train_loader)
        warm = 10 * iters
        total = opt.epochs * iters
        scheduler = SequentialLR(
            optimizer,
            [LinearLR(optimizer, 1e-6, 1.0, warm),
             CosineAnnealingLR(optimizer, T_max=total-warm, eta_min=0.0)],
            milestones=[warm]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.0)

    best_fitness = 0.0
    last, best = save_dir/'weights'/'last.pt', save_dir/'weights'/'best.pt'

    for epoch in range(opt.epochs):
        train(model, criterion, optimizer, scheduler,
              train_loader, device, epoch, train_meters)
        log_meters(writer, train_meters, 'train', epoch)

        if (epoch + 1) % 5 == 0:
            evaluate(model, criterion, val_loader, device, epoch, val_meters)
            # fitness based on final exit top1
            final_top1 = val_meters[f'exit{n_exits}_top1'].compute()
            if final_top1 > best_fitness:
                best_fitness = final_top1
                torch.save(model.state_dict(), best)
            log_meters(writer, val_meters, 'val', epoch)

    # save last
    torch.save(model.state_dict(), last)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
