import argparse
import json
from pathlib import Path

import pandas as pd


def get_nested(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(run_dir: Path):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path, last_n: int, warmup_epochs: int):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    cfg = load_config(run_dir)

    # Sort just in case CSV rows are not perfectly ordered.
    if "epoch" in df.columns:
        df = df.sort_values("epoch")

    # Use later epochs for stable throughput.
    stable_df = df[df["epoch"] > warmup_epochs] if "epoch" in df.columns else df
    if stable_df.empty:
        stable_df = df

    last_df = df.tail(last_n)

    best_idx = df["val_acc"].idxmax()
    best_row = df.loc[best_idx]

    train_img_s_median = stable_df["train_img_s"].median()
    train_img_s_last = last_df["train_img_s"].mean()

    val_img_s_median = stable_df["val_img_s"].median() if "val_img_s" in df.columns else None

    data_time = last_df["data_time"].mean() if "data_time" in df.columns else None
    batch_time = last_df["batch_time"].mean() if "batch_time" in df.columns else None

    if data_time is not None and batch_time is not None and batch_time > 0:
        data_ratio = data_time / batch_time
    else:
        data_ratio = None

    gpu_mem_mb = df["gpu_mem_mb"].max() if "gpu_mem_mb" in df.columns else None

    return {
        "run": run_dir.name,
        "dataset": get_nested(cfg, ["dataset", "name"], "unknown"),
        "model": get_nested(cfg, ["model", "name"], "unknown"),
        "batch_size": get_nested(cfg, ["training", "batch_size"], None),
        "amp": get_nested(cfg, ["training", "use_amp"], None),
        "num_workers": get_nested(cfg, ["dataset", "num_workers"], None),
        "pin_memory": get_nested(cfg, ["dataset", "pin_memory"], None),
        "epochs": int(df["epoch"].max()) if "epoch" in df.columns else len(df),

        "best_val_acc": best_row["val_acc"],
        "best_epoch": int(best_row["epoch"]) if "epoch" in df.columns else None,
        "final_val_acc": df.iloc[-1]["val_acc"],
        "final_train_acc": df.iloc[-1]["train_acc"],

        "train_img_s_median_after_warmup": train_img_s_median,
        "train_img_s_last_n_avg": train_img_s_last,
        "val_img_s_median_after_warmup": val_img_s_median,

        "data_time_last_n_avg": data_time,
        "batch_time_last_n_avg": batch_time,
        "data_ratio_last_n": data_ratio,

        "gpu_mem_mb_peak": gpu_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--last-n", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--acc-tol", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="runs/summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    records = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        summary = summarize_run(
            run_dir=run_dir,
            last_n=args.last_n,
            warmup_epochs=args.warmup_epochs,
        )

        if summary is not None:
            records.append(summary)

    if not records:
        print(f"No valid runs found under {runs_dir}")
        return

    summary_df = pd.DataFrame(records)

    best_acc = summary_df["best_val_acc"].max()
    acc_floor = best_acc - args.acc_tol

    summary_df["acc_within_tolerance"] = summary_df["best_val_acc"] >= acc_floor

    # Main ranking:
    # 1. only runs close enough in accuracy
    # 2. higher throughput
    # 3. lower data ratio
    # 4. lower memory
    candidate_df = summary_df[summary_df["acc_within_tolerance"]].copy()

    candidate_df = candidate_df.sort_values(
        by=[
            "train_img_s_median_after_warmup",
            "data_ratio_last_n",
            "gpu_mem_mb_peak",
        ],
        ascending=[False, True, True],
    )

    summary_df = summary_df.sort_values(
        by=[
            "acc_within_tolerance",
            "train_img_s_median_after_warmup",
            "best_val_acc",
        ],
        ascending=[False, False, False],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_path, index=False)

    print()
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Accuracy tolerance: {args.acc_tol:.2f}%")
    print(f"Candidate accuracy floor: {acc_floor:.2f}%")
    print()
    print(f"Saved full summary to: {out_path}")
    print()

    print("Top candidate runs:")
    display_cols = [
        "run",
        "batch_size",
        "amp",
        "num_workers",
        "pin_memory",
        "best_val_acc",
        "final_val_acc",
        "train_img_s_median_after_warmup",
        "train_img_s_last_n_avg",
        "data_ratio_last_n",
        "gpu_mem_mb_peak",
    ]

    print(candidate_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()