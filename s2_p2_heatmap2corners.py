#!/usr/bin/env python3
"""
s2_p2 阶段：把 s2_p1 网络输出的 8 张 heatmap 解码为 8 个角点

输入：
  - s2_p1 训练好的权重（默认 s2_p1_result/weights/best.pth）
  - 待推理的 patch（默认 s2_p0_data 的 val 集，也可 --split train）

流程：
  1. 加载 ResNet18Heatmap + 权重
  2. 对每张 patch 前向得到 8×64×64 heatmap（[0,1] 高斯响应）
  3. 每张 heatmap 解码为一个角点，三种解码方式可选：
       argmax      —— 峰值像素
       subpixel    —— argmax + 一维抛物线亚像素精化（默认，更准）
       softargmax  —— 归一化加权质心
  4. 角点坐标从 64×64 heatmap 空间映射回 patch 归一化坐标 [0,1]
  5. 与 annotations.json 的 GT 角点对比，输出平均误差
  6. 保存预测结果 + 可选可视化（角点散点 + 包围盒线框 + heatmap 叠加）

输出：s2_p2_result/predictions.json (+ visualize/*.png)

核心函数 `heatmap_to_coords` 可单独 import 复用：
    from s2_p2_heatmap2corners import heatmap_to_coords

用法:
    python3 s2_p2_heatmap2corners.py [--weights ...] [--split val] [--method subpixel] [--visualize]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from s2_p1_model import ResNet18Heatmap, CORNER_ORDER

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 8 个角点（CORNER_ORDER 顺序）构成的立方体 12 条边
CUBOID_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),   # 下底面 z=min
    (4, 5), (5, 7), (7, 6), (6, 4),   # 上顶面 z=max
    (0, 4), (1, 5), (2, 6), (3, 7),   # 竖直棱
]


def heatmap_to_coords(heatmaps: torch.Tensor, method: str = "subpixel") -> torch.Tensor:
    """把 heatmap 解码为角点坐标。

    Args:
        heatmaps: (..., C, H, W) 张量，C 为 heatmap/角点数（这里 8）
        method: "argmax" | "subpixel" | "softargmax"

    Returns:
        (..., C, 2) 归一化 [0,1] 角点，最后一维是 (u, v) = (x, y)
    """
    *lead, C, H, W = heatmaps.shape
    h = heatmaps.reshape(-1, C, H, W).contiguous()
    N = h.shape[0]
    dev, dt = h.device, h.dtype

    flat = h.reshape(N, C, -1)
    idx = flat.argmax(dim=-1)              # (N, C)
    y0 = (idx // W).long()
    x0 = (idx % W).long()

    if method == "softargmax":
        yy = torch.arange(H, device=dev, dtype=dt).reshape(1, 1, H, 1)
        xx = torch.arange(W, device=dev, dtype=dt).reshape(1, 1, 1, W)
        w = h.clamp(min=0.0)
        s = w.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        x = (xx * w).sum(dim=(-1, -2)) / s[..., 0, 0]
        y = (yy * w).sum(dim=(-1, -2)) / s[..., 0, 0]
    else:
        x = x0.float()
        y = y0.float()
        if method == "subpixel":
            Ni = torch.arange(N, device=dev).reshape(N, 1)
            Ci = torch.arange(C, device=dev).reshape(1, C)
            vc = h[Ni, Ci, y0, x0]
            xl = (x0 - 1).clamp(min=0)
            xr = (x0 + 1).clamp(max=W - 1)
            yl = (y0 - 1).clamp(min=0)
            yr = (y0 + 1).clamp(max=H - 1)
            vl = h[Ni, Ci, y0, xl]
            vr = h[Ni, Ci, y0, xr]
            vu = h[Ni, Ci, yl, x0]
            vd = h[Ni, Ci, yr, x0]

            denom_x = vl - 2 * vc + vr
            denom_y = vu - 2 * vc + vd
            dx = 0.5 * (vl - vr) / torch.where(denom_x.abs() < 1e-6,
                                               torch.ones_like(denom_x), denom_x)
            dy = 0.5 * (vu - vd) / torch.where(denom_y.abs() < 1e-6,
                                               torch.ones_like(denom_y), denom_y)
            dx = torch.where(denom_x.abs() < 1e-6, torch.zeros_like(dx), dx).clamp(-1.0, 1.0)
            dy = torch.where(denom_y.abs() < 1e-6, torch.zeros_like(dy), dy).clamp(-1.0, 1.0)
            x = x0.float() + dx
            y = y0.float() + dy

    coords = torch.stack([x, y], dim=-1) / torch.tensor(
        [W - 1, H - 1], device=dev, dtype=dt)
    return coords.clamp(0.0, 1.0).reshape(*lead, C, 2)


def load_model(weights_path: str, device: torch.device) -> ResNet18Heatmap:
    model = ResNet18Heatmap(num_heatmaps=8, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def visualize(pil_img: Image.Image, heatmaps: torch.Tensor,
              pred_corners: np.ndarray, gt_corners: np.ndarray, save_path: Path):
    """画：左图 = patch 上的 GT/预测角点 + 预测线框；右图 = 8 张 heatmap 平均叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = pil_img.size
    pred_px = pred_corners * (W - 1)
    gt_px = gt_corners * (W - 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pil_img)
    axes[0].scatter(gt_px[:, 0], gt_px[:, 1], c="lime", marker="o", s=45,
                    edgecolors="black", linewidths=0.6, label="GT")
    axes[0].scatter(pred_px[:, 0], pred_px[:, 1], c="red", marker="x", s=55,
                    linewidths=1.6, label="pred")
    for i, j in CUBOID_EDGES:
        axes[0].plot([pred_px[i, 0], pred_px[j, 0]],
                     [pred_px[i, 1], pred_px[j, 1]], color="red", lw=1.0, alpha=0.55)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("predicted corners (red x) vs GT (green o)")
    axes[0].axis("off")

    overlay = heatmaps.detach().cpu().numpy().mean(axis=0)  # (64,64)
    axes[1].imshow(overlay, cmap="hot", interpolation="nearest")
    axes[1].set_title("mean heatmap")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=str,
                    default="/home/xyyzz1/pro261005/s2_p1_result/weights/best.pth")
    ap.add_argument("--data-root", type=str,
                    default="/home/xyyzz1/pro261005/s2_p0_data")
    ap.add_argument("--out-root", type=str,
                    default="/home/xyyzz1/pro261005/s2_p2_result")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--method", type=str, default="subpixel",
                    choices=["argmax", "subpixel", "softargmax"])
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--visualize", action="store_true", help="保存可视化结果")
    ap.add_argument("--num-vis", type=int, default=8, help="可视化前 N 个实例")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available()
        else f"cuda:{args.device}")

    model = load_model(args.weights, device)
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    vis_dir = out_root / "visualize"
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    ann = json.loads((data_root / "annotations.json").read_text())
    instances = [a for a in ann["instances"] if a["split"] == args.split]

    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    preds = []
    errs = []
    for ins in instances:
        img = Image.open(data_root / ins["patch"]).convert("RGB")
        x = normalize(T.ToTensor()(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            hm = model(x)                              # (1,8,64,64)
        corners = heatmap_to_coords(hm, args.method)[0].cpu().numpy()  # (8,2)
        gt = np.array(ins["corners"], dtype=np.float32)
        err = float(np.linalg.norm(corners - gt, axis=1).mean())
        errs.append(err)

        preds.append({
            "id": ins["id"],
            "split": ins["split"],
            "patch": ins["patch"],
            "corners": [[round(float(c[0]), 6), round(float(c[1]), 6)] for c in corners],
            "gt_corners": ins["corners"],
            "error_norm": round(err, 6),
        })
        if args.visualize and len(preds) <= args.num_vis:
            visualize(img, hm[0], corners, gt, vis_dir / f"{ins['id']}.png")

    errs = np.array(errs)
    meta = {
        "description": "s2_p2: heatmap -> 8 角点（归一化 [0,1]）",
        "weights": str(Path(args.weights)),
        "split": args.split,
        "method": args.method,
        "num_instances": len(preds),
        "num_corners": 8,
        "corner_order": CORNER_ORDER,
        "mean_corner_error_norm": round(float(errs.mean()), 6),
        "mean_corner_error_px": round(float(errs.mean() * 256), 2),
    }
    (out_root / "predictions.json").write_text(
        json.dumps({"meta": meta, "predictions": preds}, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("s2_p2 heatmap -> 8 角点")
    print(f"  权重: {args.weights}")
    print(f"  数据集: {args.split} ({len(instances)} 实例), 解码: {args.method}")
    print(f"  平均角点误差: {errs.mean():.4f} (归一化) = {errs.mean() * 256:.1f} px")
    print(f"  输出: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
