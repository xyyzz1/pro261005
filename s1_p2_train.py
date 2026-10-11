#!/usr/bin/env python3
"""
S2 阶段：YOLO11 单类小规模训练 + 输出 loss 变化

- 使用 S0 生成的数据集（nc=1，单类 obj_000001）
- 用 yolo11n 预训练权重在【全部训练集】上小规模训练（默认 30 epochs）
- 训练后从 results.csv 读取并绘制 loss 曲线，打印 loss 汇总

用法:
    python3 s2_train.py [--epochs 30] [--model yolo11n.pt] [--imgsz 640] [--batch 16]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def plot_losses(csv_path: Path, out_dir: Path):
    """从 results.csv 绘制 loss 曲线并打印汇总。"""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    x = df["epoch"] if "epoch" in df.columns else df.index

    train_loss_cols = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    val_loss_cols = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    metric_cols = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 训练 loss
    ax = axes[0]
    for c in train_loss_cols:
        if c in df.columns:
            ax.plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    ax.set_title("Train Loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 验证 loss
    ax = axes[1]
    for c in val_loss_cols:
        if c in df.columns:
            ax.plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    ax.set_title("Val Loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 指标
    ax = axes[2]
    for c in metric_cols:
        if c in df.columns:
            ax.plot(x, df[c], label=c.split("/")[1], marker=".", markersize=3)
    ax.set_title("Metrics")
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("YOLO11 S2 Training Curves (nc=1)")
    fig.tight_layout()
    png = out_dir / "loss_curves.png"
    fig.savefig(png, dpi=130)
    print(f"loss 曲线已保存: {png}")

    # 打印汇总
    print("\n===== loss / 指标汇总 =====")
    first, last = df.iloc[0], df.iloc[-1]
    print(f"epoch 范围: {int(first['epoch'])} -> {int(last['epoch'])}")
    for c in train_loss_cols + val_loss_cols:
        if c in df.columns:
            print(f"  {c:<18} {first[c]:.4f} -> {last[c]:.4f}")
    for c in metric_cols:
        if c in df.columns:
            print(f"  {c:<18} {first[c]:.4f} -> {last[c]:.4f}  (best {df[c].max():.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="yolo11n.pt")
    ap.add_argument("--data", type=str,
                    default="my_dataset_corners/yolo_dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=30, help="训练轮数（小规模）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="s2_out")
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    model = YOLO(args.model)
    project = Path(args.project).resolve()

    # 小规模训练（使用全部训练集，单类 nc=1 由 data.yaml 指定）
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
    )

    csv_path = project / args.name / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"未找到 results.csv: {csv_path}")
    plot_losses(csv_path, project / args.name)


if __name__ == "__main__":
    main()
