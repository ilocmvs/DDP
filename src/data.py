from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    name: str
    data_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool


def build_transforms(train: bool):
    """Build image transforms for CIFAR training/evaluation.

    TODO(student): Read each transform and explain why train/eval differ.
    Later, this function will become important when we profile CPU-side data work.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])


def build_datasets(cfg: DataConfig):
    dataset_name = cfg.name.upper()

    if dataset_name == "CIFAR10":
        dataset_cls = datasets.CIFAR10
    elif dataset_name == "CIFAR100":
        dataset_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {cfg.name}. Use CIFAR10 or CIFAR100.")

    train_set = dataset_cls(
        root=cfg.data_dir,
        train=True,
        transform=build_transforms(train=True),
        download=True,
    )
    val_set = dataset_cls(
        root=cfg.data_dir,
        train=False,
        transform=build_transforms(train=False),
        download=True,
    )
    return train_set, val_set


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, int]:
    train_set, val_set = build_datasets(cfg)
    num_classes = 100 if cfg.name.upper() == "CIFAR100" else 10

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, num_classes
