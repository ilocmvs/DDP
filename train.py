from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from src.data import DataConfig, build_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import build_model
from src.utils import load_config, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU CIFAR ResNet training baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()

import csv
import json
from pathlib import Path

METRIC_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "train_img_s",
    "val_img_s",
    "data_time",
    "batch_time",
    "gpu_mem_mb",
    "best_acc",
    "is_best",
]

def save_config_json(cfg, output_dir):
    """
    Save a frozen copy of the config used for this run.
    This is not meant to replace the editable YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def append_metrics_csv(output_dir, row):
    """
    Append one epoch of metrics to metrics.csv.
    Creates the file and header automatically if needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "metrics.csv"
    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This starter is intended for single-GPU CUDA training.")

    device = torch.device("cuda")
    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_config_json(cfg, output_dir)

    data_cfg = DataConfig(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
    )

    train_loader, val_loader, num_classes = build_dataloaders(data_cfg)

    model = build_model(
        name=cfg["model"]["name"],
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", False),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler_name = cfg.get("scheduler", {}).get("name", "none").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=cfg["scheduler"].get("min_lr", 0.0),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    use_amp = bool(cfg["training"].get("use_amp", False))
    scaler = GradScaler() if use_amp else None

    best_acc = 0.0
    epochs = int(cfg["training"]["epochs"])
    log_interval = int(cfg["logging"].get("log_interval", 100))

    print(f"Device: {device}")
    print(f"Dataset: {cfg['dataset']['name']} | classes={num_classes}")
    print(f"Model: {cfg['model']['name']}")
    print(f"AMP: {use_amp}")

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} lr={lr:.5f} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.2f} "
            f"train_img_s={train_metrics['throughput_img_s']:.1f} "
            f"val_img_s={val_metrics['throughput_img_s']:.1f} "
            f"data_time={train_metrics['data_time_avg']:.4f}s "
            f"batch_time={train_metrics['batch_time_avg']:.4f}s"
        )

        is_best = val_metrics["acc"] > best_acc
        best_acc = max(best_acc, val_metrics["acc"])

        if cfg["logging"].get("save_checkpoint", True):
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            }
            save_checkpoint(state, output_dir, "latest.pt")
            if is_best:
                save_checkpoint(state, output_dir, "best.pt")

        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_mem_mb = 0.0

        metrics_row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics['acc'],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "train_img_s": train_metrics["throughput_img_s"],
            "val_img_s": val_metrics["throughput_img_s"],
            "data_time": train_metrics["data_time_avg"],
            "batch_time": train_metrics["batch_time_avg"],
            "gpu_mem_mb": gpu_mem_mb,
            "best_acc": best_acc,
            "is_best": int(is_best),
        }

        append_metrics_csv(output_dir, metrics_row) 

    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
