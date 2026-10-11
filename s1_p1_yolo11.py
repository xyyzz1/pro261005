#!/usr/bin/env python3
"""
S1 阶段：YOLO11 网络架构 + 推理（使用 ultralytics 官方实现）

- 加载 YOLO11 预训练模型（默认 yolo11n.pt，首次运行自动下载）
- 打印网络架构（逐层结构）
- 对输入图像/目录执行推理，保存标注结果并打印检测框

用法:
    python3 s1_yolo11.py --source <图像或目录> [--model yolo11n.pt] [--conf 0.25] [--imgsz 640]
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# YOLO11 各尺度预训练模型（官方 COCO 权重）
MODEL_CHOICES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                 "yolo11l.pt", "yolo11x.pt"]


def print_architecture(model: YOLO) -> str:
    """打印模型逐层结构并返回文本（同时用于存档）。"""
    arch = str(model.model)
    print("=" * 70)
    print("YOLO11 网络架构 (model.model)")
    print("=" * 70)
    print(arch)
    print("=" * 70)
    return arch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=str, required=True,
                    help="输入图像或目录路径")
    ap.add_argument("--model", type=str, default="yolo11n.pt",
                    choices=MODEL_CHOICES, help="YOLO11 模型权重")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    ap.add_argument("--device", type=str, default=None,
                    help="设备，如 0(显卡)/cpu；默认自动")
    ap.add_argument("--save-dir", type=str, default="s1_out",
                    help="结果保存目录")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"输入不存在: {source}")

    # 1. 加载模型（架构 + 预训练权重，首次运行自动下载）
    print(f"[1/3] 加载模型: {args.model}")
    model = YOLO(args.model)

    # 2. 打印架构
    print(f"[2/3] 打印网络架构")
    arch_text = print_architecture(model)

    # 3. 推理
    print(f"[3/3] 推理: {source}")
    out_dir = Path(args.save_dir).resolve()
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,               # 保存标注图像
        project=str(out_dir),    # 绝对路径，避免被 ultralytics 加上 runs/detect 前缀
        name="exp",
        exist_ok=True,
        verbose=False,
    )

    # 打印检测结果
    n_det = 0
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        names = r.names
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            n_det += 1
            print(f"  {names[int(cls)]:<12} conf={conf:.3f}")
    print(f"共 {len(results)} 张图像，检测到 {n_det} 个目标")

    # 存档架构文本
    (out_dir / "architecture.txt").write_text(arch_text)
    print(f"架构已存档: {out_dir / 'architecture.txt'}")
    # 从结果回读实际保存目录（ultralytics 会给 name 加序号或处理重名）
    save_dir = results[0].save_dir if results else out_dir / "exp"
    print(f"标注结果保存在: {save_dir}")


if __name__ == "__main__":
    main()
