from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def reset(self):
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy in percent."""
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return 100.0 * correct / targets.numel()


def save_checkpoint(state: Dict[str, Any], output_dir: str, filename: str = "checkpoint.pt"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    torch.save(state, path)
    return path
