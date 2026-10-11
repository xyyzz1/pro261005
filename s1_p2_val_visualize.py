#!/usr/bin/env python3
"""
S2 阶段：验证集推理 + 可视化（把预测框画在图片上）

- 用训练好的 best.pt 对验证集跑推理
- 将预测框 + 标签画在图上并保存

用法:
    python3 s2_val_visualize.py [--weights s2_out/train/weights/best.pt] [--conf 0.25]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=str,
                    default="s2_out/train/weights/best.pt")
    ap.add_argument("--source", type=str,
                    default="my_dataset_corners/yolo_dataset/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--save-dir", type=str, default="s2_out/val_pred")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.save_dir).resolve()

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,                 # 画框并保存
        project=str(out_dir),      # 绝对路径，避免 runs/detect 前缀
        name="exp",
        exist_ok=True,
        verbose=False,
    )

    n_img = len(results)
    n_det = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
    print(f"验证集推理完成: {n_img} 张图像, 共 {n_det} 个预测框")
    print(f"标注结果保存在: {out_dir / 'exp'}")

    # 逐张统计
    for r in results:
        name = Path(r.path).name
        cnt = len(r.boxes) if r.boxes is not None else 0
        confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()] if cnt else []
        print(f"  {name}: {cnt} 框  conf={confs}")


if __name__ == "__main__":
    main()
