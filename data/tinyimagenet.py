from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from datasets import load_dataset
from PIL import Image

def get_tinyimagenet_train_transforms(
    mean: Sequence[float], std: Sequence[float]
):
    return T.Compose([
        T.RandomResizedCrop(64, scale=(0.5, 1.0)),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomHorizontalFlip(),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def get_tinyimagenet_val_transforms(
    mean: Sequence[float], std: Sequence[float]
):
    return T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

class HFDatasetWrapper(Dataset):
    """
    Wraps a HuggingFace Dataset so that __getitem__ returns
    (image_tensor, label) using torchvision transforms.
    """
    def __init__(self, hf_ds, transform: T.Compose):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        example = self.hf_ds[idx]
        img: Image.Image = example["image"]  # this is a PIL image
        img = img.convert("RGB")
        img = self.transform(img)
        label = example["label"]
        return img, label

def tinyimagenet_hf(
    batch_size: Optional[int] = 256,
    num_workers: int = 2,
    dataset_name: str = "zh-plus/tiny-imagenet",
) -> tuple[DataLoader, DataLoader]:
    """
    Downloads Tiny-ImageNet via HF `datasets`, wraps it in a
    torch Dataset, and returns train/val DataLoaders.
    """
    # 1) Download HF dataset
    ds = load_dataset(dataset_name)  # splits: "train", "valid"

    # 2) Build transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = get_tinyimagenet_train_transforms(mean, std)
    val_tf   = get_tinyimagenet_val_transforms(mean, std)

    # 3) Wrap in our torch Dataset
    train_ds = HFDatasetWrapper(ds["train"], transform=train_tf)
    val_ds   = HFDatasetWrapper(ds["valid"], transform=val_tf)

    # 4) DataLoaders (no custom collate_fn needed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader