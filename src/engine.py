from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .utils import AverageMeter, accuracy_top1


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """One training epoch.

    TODO(student): This is the most important file to study.
    Understand the order:
      1. model.train()
      2. move batch to device
      3. forward
      4. loss
      5. zero_grad
      6. backward
      7. optimizer.step
      8. metric/logging
    """
    model.train()

    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")
    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")

    end = time.perf_counter()
    num_images = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, (images, targets) in enumerate(pbar):
        data_time.update(time.perf_counter() - end)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # TODO(student): Why is zero_grad called before backward?
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # AMP path.
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 path.
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        acc = accuracy_top1(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        num_images += batch_size

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if step % log_interval == 0:
            images_per_sec = batch_size / max(batch_time.val, 1e-9)
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2f}",
                "img/s": f"{images_per_sec:.1f}",
            })

    epoch_time = batch_time.sum
    throughput = num_images / max(epoch_time, 1e-9)
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "throughput_img_s": throughput,
        "batch_time_avg": batch_time.avg,
        "data_time_avg": data_time.avg,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validation loop.

    TODO(student): Compare this with train_one_epoch.
    Why do we use model.eval() and torch.no_grad() here?
    """
    model.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")
    batch_time = AverageMeter("batch_time")

    end = time.perf_counter()
    num_images = 0

    pbar = tqdm(loader, desc=f"eval epoch {epoch}", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        acc = accuracy_top1(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        num_images += batch_size

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}",
        })

    epoch_time = batch_time.sum
    throughput = num_images / max(epoch_time, 1e-9)
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "throughput_img_s": throughput,
        "batch_time_avg": batch_time.avg,
    }
