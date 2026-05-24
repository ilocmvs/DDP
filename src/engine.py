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
    precise_timing: bool = False,
) -> Dict[str, float]:
    """One training epoch.

    Timing definitions:
      data_time_avg:
        Average wall-clock time spent waiting for the next batch from DataLoader.
        This is CPU/input-pipeline time before the batch is moved to GPU.

      batch_time_avg:
        Average full iteration wall-clock time, measured from the end of the
        previous iteration to the end of the current iteration. This includes
        data_time + H2D transfer + forward + backward + optimizer step + metrics.

      gpu_time_avg:
        Average CUDA event time for GPU work when precise_timing=True.
        We record the CUDA start event before moving tensors to device, so this
        includes H2D copy queued on the default stream plus forward/backward/update.
        When precise_timing=False, this is reported as 0.0.
    """
    model.train()

    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_acc")
    batch_time_meter = AverageMeter("batch_time")
    data_time_meter = AverageMeter("data_time")
    gpu_time_meter = AverageMeter("gpu_time")

    num_images = 0

    end = time.perf_counter()
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    use_cuda_timing = precise_timing and device.type == "cuda"

    for step, (images, targets) in enumerate(pbar):
        # At this point, DataLoader has already returned the batch.
        # Therefore, time since previous iteration ended is the input wait time.
        data_elapsed = time.perf_counter() - end
        data_time_meter.update(data_elapsed)

        # Full iteration time starts from the previous iteration end.
        # This intentionally includes DataLoader wait time.
        iter_start = end

        if use_cuda_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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

        if use_cuda_timing:
            end_event.record()
            torch.cuda.synchronize()
            gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            gpu_elapsed = 0.0

        batch_size = images.size(0)
        acc = accuracy_top1(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        num_images += batch_size

        batch_elapsed = time.perf_counter() - iter_start
        batch_time_meter.update(batch_elapsed)
        gpu_time_meter.update(gpu_elapsed)

        end = time.perf_counter()

        if log_interval > 0 and step % log_interval == 0:
            images_per_sec = batch_size / max(batch_time_meter.val, 1e-9)
            postfix = {
                "loss": f"{loss_meter.avg:.4f}",
                "acc": f"{acc_meter.avg:.2f}",
                "img/s": f"{images_per_sec:.1f}",
                "data": f"{data_time_meter.avg:.4f}s",
                "batch": f"{batch_time_meter.avg:.4f}s",
            }
            if use_cuda_timing:
                postfix["gpu"] = f"{gpu_time_meter.avg:.4f}s"
            pbar.set_postfix(postfix)

    epoch_time = batch_time_meter.sum
    throughput = num_images / max(epoch_time, 1e-9)

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "throughput_img_s": throughput,
        "batch_time_avg": batch_time_meter.avg,
        "data_time_avg": data_time_meter.avg,
        "gpu_time_avg": gpu_time_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validation loop."""
    model.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_acc")
    batch_time_meter = AverageMeter("batch_time")

    end = time.perf_counter()
    num_images = 0

    pbar = tqdm(loader, desc=f"eval epoch {epoch}", leave=False)
    for images, targets in pbar:
        iter_start = end

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        acc = accuracy_top1(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        num_images += batch_size

        batch_time_meter.update(time.perf_counter() - iter_start)
        end = time.perf_counter()

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "acc": f"{acc_meter.avg:.2f}",
        })

    epoch_time = batch_time_meter.sum
    throughput = num_images / max(epoch_time, 1e-9)
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "throughput_img_s": throughput,
        "batch_time_avg": batch_time_meter.avg,
    }