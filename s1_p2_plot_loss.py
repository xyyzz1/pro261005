#!/usr/bin/env python3
"""
S2 阶段：从 results.csv 绘制 loss 曲线（独立脚本，仅依赖 pandas + matplotlib）

用法:
    python3 s2_plot_loss.py --csv s2_out/train/results.csv --out s2_out/train
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    x = df["epoch"] if "epoch" in df.columns else df.index

    train_cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    val_cols = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    metric_cols = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for c in train_cols:
        if c in df.columns:
            axes[0].plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for c in val_cols:
        if c in df.columns:
            axes[1].plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    axes[1].set_title("Val Loss"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for c in metric_cols:
        if c in df.columns:
            axes[2].plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    axes[2].set_title("Metrics"); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("score")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("YOLO11 S2 Training Curves (nc=1)")
    fig.tight_layout()
    png = out_dir / "loss_curves.png"
    fig.savefig(png, dpi=130)
    print(f"loss 曲线已保存: {png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="s2_out/train/results.csv")
    ap.add_argument("--out", default="s2_out/train")
    args = ap.parse_args()
    plot(Path(args.csv).resolve(), Path(args.out).resolve())
