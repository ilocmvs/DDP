from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Allow running as: python scripts/export_onnx.py ...
# without installing the project as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained CIFAR model checkpoint from PyTorch to ONNX."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. Defaults to <config output_dir>/best.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path. Defaults to <config output_dir>/model.onnx.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size used during export. With dynamic batch enabled, this is only the example batch size.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Input image height/width. CIFAR uses 32.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Export device. Defaults to cuda if available, else cpu.",
    )

    # Dynamic batch is the default because TensorRT benchmarking will usually
    # compare multiple batch sizes from the same ONNX model.
    parser.set_defaults(dynamic_batch=True)
    parser.add_argument(
        "--dynamic-batch",
        dest="dynamic_batch",
        action="store_true",
        help="Export with dynamic batch dimension. This is the default.",
    )
    parser.add_argument(
        "--static-batch",
        dest="dynamic_batch",
        action="store_false",
        help="Export with fixed batch size equal to --batch-size.",
    )

    parser.add_argument(
        "--check-onnx",
        action="store_true",
        help="Run onnx.checker.check_model after export if the onnx package is installed.",
    )
    parser.add_argument(
        "--verify-onnxruntime",
        action="store_true",
        help="Compare PyTorch output against ONNX Runtime output on the dummy input if onnxruntime is installed.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for ONNX Runtime verification.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for ONNX Runtime verification.",
    )
    return parser.parse_args()


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Allow loading future DDP checkpoints saved from model.module.state_dict()."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def safe_torch_load(path: Path) -> Any:
    """Load checkpoints across older/newer PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_num_classes(cfg: Dict[str, Any]) -> int:
    dataset_name = cfg["dataset"]["name"].upper()
    if dataset_name == "CIFAR100":
        return 100
    if dataset_name == "CIFAR10":
        return 10
    raise ValueError(f"Unsupported dataset: {cfg['dataset']['name']}. Use CIFAR10 or CIFAR100.")


def load_model_from_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(
        name=cfg["model"]["name"],
        num_classes=infer_num_classes(cfg),
        pretrained=cfg["model"].get("pretrained", False),
    )

    checkpoint = safe_torch_load(checkpoint_path)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Also support a raw state_dict file.
        state_dict = checkpoint

    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


def check_onnx_model(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("ONNX checker skipped: package 'onnx' is not installed.")
        return

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("ONNX checker: passed.")


@torch.no_grad()
def verify_with_onnxruntime(
    model: torch.nn.Module,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime verification skipped: package 'onnxruntime' is not installed.")
        return

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(onnx_path), providers=providers)

    torch_out = model(dummy_input).detach().cpu().numpy()
    ort_out = session.run(["logits"], {"input": dummy_input.detach().cpu().numpy()})[0]

    abs_diff = np.abs(torch_out - ort_out)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    ok = bool(np.allclose(torch_out, ort_out, atol=atol, rtol=rtol))

    print(f"ONNX Runtime providers: {session.get_providers()}")
    print(f"ONNX Runtime verification: {'passed' if ok else 'FAILED'}")
    print(f"max_abs_diff={max_abs_diff:.6e} | mean_abs_diff={mean_abs_diff:.6e}")

    if not ok:
        raise RuntimeError(
            "ONNX Runtime output differs from PyTorch output beyond tolerance. "
            f"Try looser tolerance or inspect the exported graph. atol={atol}, rtol={rtol}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        device = torch.device(args.device)

    output_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best.pt"
    onnx_path = Path(args.output) if args.output else output_dir / "model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model_from_checkpoint(cfg, checkpoint_path, device)
    dummy_input = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
        dtype=torch.float32,
    )

    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output ONNX: {onnx_path}")
    print(f"Model: {cfg['model']['name']} | dataset={cfg['dataset']['name']} | classes={infer_num_classes(cfg)}")
    print(f"Dummy input shape: {tuple(dummy_input.shape)}")
    print(f"Opset: {args.opset}")
    print(f"Dynamic batch: {args.dynamic_batch}")

    with torch.no_grad():
        # Export FP32 ONNX. TensorRT can later build either FP32 or FP16 engines
        # from this same graph using builder flags such as --fp16.
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    print("Export complete.")

    if args.check_onnx:
        check_onnx_model(onnx_path)

    if args.verify_onnxruntime:
        verify_with_onnxruntime(
            model=model,
            onnx_path=onnx_path,
            dummy_input=dummy_input,
            atol=args.atol,
            rtol=args.rtol,
        )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"Saved: {onnx_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
