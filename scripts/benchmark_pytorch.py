from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

# Allow running as: python scripts/benchmark_pytorch.py ...
# without installing the project as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DataConfig, build_dataloaders
from src.model import build_model
from src.utils import accuracy_top1, load_config, set_seed


CSV_FIELDS = [
    "backend",
    "checkpoint",
    "dataset",
    "model",
    "precision",
    "batch_size",
    "num_workers",
    "pin_memory",
    "warmup_batches",
    "actual_warmup_batches",
    "measured_batches",
    "num_images",
    "accuracy_top1",
    "latency_ms_avg",
    "latency_ms_p50",
    "latency_ms_p90",
    "latency_ms_p99",
    "throughput_img_s_gpu_timed",
    "throughput_img_s_host_timed",
    "throughput_img_s_end_to_end",
    "data_time_ms_avg",
    "host_batch_time_ms_avg",
    "end_to_end_batch_time_ms_avg",
    "gpu_mem_mb_peak",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch inference on CIFAR validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used for training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. Defaults to <config output_dir>/best.pt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <config output_dir>/inference_pytorch.csv",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to benchmark. Defaults to the training batch size from config.",
    )
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        choices=["fp32", "fp16"],
        default=["fp32", "fp16"],
        help="Inference precisions to benchmark.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help=(
            "Number of warmup batches to run before measurement. "
            "Warmup is done in a separate pass, then the validation loader is restarted "
            "so measured accuracy uses the full validation set."
        ),
    )
    parser.add_argument(
        "--max-measured-batches",
        type=int,
        default=0,
        help="Maximum measured batches after warmup. 0 means use the full validation set.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark for faster fixed-shape conv inference.",
    )
    return parser.parse_args()


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Good enough for benchmark reporting."""
    if not values:
        return 0.0
    values = sorted(values)
    idx = round((pct / 100.0) * (len(values) - 1))
    return float(values[idx])


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Allow loading future DDP checkpoints saved from model.module.state_dict()."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(cfg: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    dataset_name = cfg["dataset"]["name"].upper()
    num_classes = 100 if dataset_name == "CIFAR100" else 10

    model = build_model(
        name=cfg["model"]["name"],
        num_classes=num_classes,
        pretrained=cfg["model"].get("pretrained", False),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Also support a raw state_dict file.
        state_dict = checkpoint

    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


def build_val_loader(cfg: Dict[str, Any], batch_size: int):
    data_cfg = DataConfig(
        name=cfg["dataset"]["name"],
        data_dir=cfg["dataset"]["data_dir"],
        batch_size=batch_size,
        num_workers=int(cfg["dataset"].get("num_workers", 2)),
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
    )
    _, val_loader, _ = build_dataloaders(data_cfg)
    return val_loader


def make_autocast_context(precision: str):
    if precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported precision: {precision}")


def run_timed_inference_batch(
    model: torch.nn.Module,
    images_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    device: torch.device,
    precision: str,
) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
    """
    Run one batch and return:
      logits, targets, batch_size, gpu_elapsed_ms, host_elapsed_s

    gpu_elapsed_ms is measured with CUDA events on the default stream.
    Because the H2D copies and model call are between the CUDA events,
    this number includes async H2D transfer work plus model execution.

    host_elapsed_s is wall-clock time for the same region, including Python
    dispatch overhead, H2D transfer submission, model call, and final synchronize.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    host_start = time.perf_counter()
    start_event.record()

    images = images_cpu.to(device, non_blocking=True)
    targets = targets_cpu.to(device, non_blocking=True)

    with make_autocast_context(precision):
        logits = model(images)

    end_event.record()
    torch.cuda.synchronize()
    host_elapsed_s = time.perf_counter() - host_start
    gpu_elapsed_ms = start_event.elapsed_time(end_event)

    return logits, targets, images.size(0), gpu_elapsed_ms, host_elapsed_s


@torch.no_grad()
def run_warmup(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    precision: str,
    warmup_batches: int,
) -> int:
    """
    Run warmup in a separate pass.

    Important: the measured pass will iterate over loader again from the beginning,
    so warmup no longer removes validation images from the reported accuracy.
    """
    if warmup_batches <= 0:
        return 0

    actual_warmup_batches = 0
    for images_cpu, targets_cpu in loader:
        if actual_warmup_batches >= warmup_batches:
            break

        run_timed_inference_batch(
            model=model,
            images_cpu=images_cpu,
            targets_cpu=targets_cpu,
            device=device,
            precision=precision,
        )
        actual_warmup_batches += 1

    torch.cuda.synchronize()
    return actual_warmup_batches


@torch.no_grad()
def benchmark_one_setting(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    precision: str,
    warmup_batches: int,
    max_measured_batches: int,
) -> Dict[str, float]:
    assert precision in {"fp32", "fp16"}

    torch.cuda.reset_peak_memory_stats(device)

    actual_warmup_batches = run_warmup(
        model=model,
        loader=loader,
        device=device,
        precision=precision,
        warmup_batches=warmup_batches,
    )

    latency_ms: List[float] = []
    host_batch_times: List[float] = []
    data_times: List[float] = []
    end_to_end_batch_times: List[float] = []

    correct_weighted_acc_sum = 0.0
    num_images = 0
    measured_batches = 0

    torch.cuda.synchronize()

    # Restart the loader after warmup.
    # This is the key fix: accuracy and measured image count now use the full
    # validation set unless --max-measured-batches is explicitly set.
    previous_end = time.perf_counter()

    for images_cpu, targets_cpu in loader:
        if max_measured_batches > 0 and measured_batches >= max_measured_batches:
            break

        # Time spent waiting for DataLoader to produce this CPU batch.
        data_elapsed_s = time.perf_counter() - previous_end

        logits, targets, batch_size, gpu_elapsed_ms, host_elapsed_s = run_timed_inference_batch(
            model=model,
            images_cpu=images_cpu,
            targets_cpu=targets_cpu,
            device=device,
            precision=precision,
        )

        acc = accuracy_top1(logits, targets)

        latency_ms.append(gpu_elapsed_ms)
        host_batch_times.append(host_elapsed_s)
        data_times.append(data_elapsed_s)
        end_to_end_batch_times.append(data_elapsed_s + host_elapsed_s)

        correct_weighted_acc_sum += acc * batch_size
        num_images += batch_size
        measured_batches += 1

        previous_end = time.perf_counter()

    if measured_batches == 0 or num_images == 0:
        raise RuntimeError(
            "No batches were measured. Lower --warmup-batches, increase --max-measured-batches, "
            "or check that the validation set is not empty."
        )

    gpu_time_s_total = sum(latency_ms) / 1000.0
    host_time_s_total = sum(host_batch_times)
    end_to_end_time_s_total = sum(end_to_end_batch_times)

    return {
        "actual_warmup_batches": actual_warmup_batches,
        "measured_batches": measured_batches,
        "num_images": num_images,
        "accuracy_top1": correct_weighted_acc_sum / num_images,
        "latency_ms_avg": statistics.mean(latency_ms),
        "latency_ms_p50": statistics.median(latency_ms),
        "latency_ms_p90": percentile(latency_ms, 90),
        "latency_ms_p99": percentile(latency_ms, 99),
        "throughput_img_s_gpu_timed": num_images / max(gpu_time_s_total, 1e-9),
        "throughput_img_s_host_timed": num_images / max(host_time_s_total, 1e-9),
        "throughput_img_s_end_to_end": num_images / max(end_to_end_time_s_total, 1e-9),
        "data_time_ms_avg": 1000.0 * statistics.mean(data_times),
        "host_batch_time_ms_avg": 1000.0 * statistics.mean(host_batch_times),
        "end_to_end_batch_time_ms_avg": 1000.0 * statistics.mean(end_to_end_batch_times),
        "gpu_mem_mb_peak": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark is intended for GPU inference.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    output_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    out_path = Path(args.out) if args.out else output_dir / "inference_pytorch.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    batch_sizes = args.batch_sizes or [int(cfg["training"]["batch_size"])]

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {out_path}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    model = load_model_from_checkpoint(cfg, checkpoint_path, device)

    records: List[Dict[str, Any]] = []

    for batch_size in batch_sizes:
        loader = build_val_loader(cfg, batch_size=batch_size)

        for precision in args.precisions:
            print(f"\nBenchmarking PyTorch | precision={precision} | batch_size={batch_size}")
            metrics = benchmark_one_setting(
                model=model,
                loader=loader,
                device=device,
                precision=precision,
                warmup_batches=args.warmup_batches,
                max_measured_batches=args.max_measured_batches,
            )

            row = {
                "backend": "pytorch_eager",
                "checkpoint": str(checkpoint_path),
                "dataset": cfg["dataset"]["name"],
                "model": cfg["model"]["name"],
                "precision": precision,
                "batch_size": batch_size,
                "num_workers": cfg["dataset"].get("num_workers", 0),
                "pin_memory": cfg["dataset"].get("pin_memory", False),
                "warmup_batches": args.warmup_batches,
                **metrics,
            }
            records.append(row)

            print(
                f"acc={row['accuracy_top1']:.2f}% | "
                f"lat_avg={row['latency_ms_avg']:.3f} ms | "
                f"p50={row['latency_ms_p50']:.3f} ms | "
                f"p90={row['latency_ms_p90']:.3f} ms | "
                f"gpu_img/s={row['throughput_img_s_gpu_timed']:.1f} | "
                f"host_img/s={row['throughput_img_s_host_timed']:.1f} | "
                f"e2e_img/s={row['throughput_img_s_end_to_end']:.1f}"
            )

    write_csv(out_path, records)
    print(f"\nSaved PyTorch inference benchmark to: {out_path}")


if __name__ == "__main__":
    main()
