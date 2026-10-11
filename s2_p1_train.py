#!/usr/bin/env python3
"""
s2_p1 阶段：训练 resnet18 + CNN decoder + heatmaps head 角点网络

数据：/home/xyyzz1/pro261005/s2_p0_data
  - images/train, images/val（256×256 patch）
  - annotations.json（每实例 8 个角点，归一化 [0,1]）

流程：
  1. 由角点生成 64×64 高斯 heatmap（sigma=2）作为 GT
  2. 模型输出 8×64×64 heatmap，加权 MSE 回归 GT（峰值像素强加权，强制抬峰）
  3. 训练/验证损失逐 epoch 记录，保存 best/last 权重 + 损失曲线
  4. 用 best 模型在验证集上按 argmax 解码角点，报告平均定位误差

结果输出：/home/xyyzz1/pro261005/s2_p1_result

用法:
    python3 s2_p1_train.py [--epochs 30] [--batch-size 16] [--lr-decoder 1e-3] [--lr-backbone 1e-4]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from s2_p1_model import ResNet18Heatmap, CORNER_ORDER

DATA_ROOT = "/home/xyyzz1/pro261005/s2_p0_data"
OUT_ROOT = "/home/xyyzz1/pro261005/s2_p1_result"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_gaussian_heatmaps(corners, size=64, sigma=2.0):
    """角点 (8,2) 归一化 [0,1] -> (8,size,size) 高斯 heatmap（峰值像素精确为 1）。"""
    n = corners.shape[0]
    hm = np.zeros((n, size, size), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(size, dtype=np.float32),
                         np.arange(size, dtype=np.float32))
    for i in range(n):
        cx = corners[i, 0] * (size - 1)
        cy = corners[i, 1] * (size - 1)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        px, py = int(round(cx)), int(round(cy))
        g = g / g[py, px]          # 峰值像素精确置 1
        hm[i] = g
    return hm


def weighted_mse_loss(pred, target, pos_weight=50.0):
    """加权 MSE：峰值像素（target 高）权重放大，强制把峰值抬起来。"""
    weight = 1.0 + pos_weight * target
    return (weight * (pred - target) ** 2).mean()


def decode_heatmaps(heatmaps):
    """(..., num, size, size) -> (..., num, 2) 归一化角点（argmax 解码）。"""
    size = heatmaps.shape[-1]
    flat = heatmaps.reshape(*heatmaps.shape[:-2], -1)   # (..., num, size*size)
    idx = flat.argmax(dim=-1)                            # (..., num)
    y = idx // size
    x = idx % size
    return torch.stack([x.float(), y.float()], dim=-1) / (size - 1)


class CornerHeatmapDataset(Dataset):
    def __init__(self, data_root, split, heatmap_size=64, sigma=2.0, augment=False):
        self.data_root = Path(data_root)
        ann = json.loads((self.data_root / "annotations.json").read_text())
        self.instances = [a for a in ann["instances"] if a["split"] == split]
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        ins = self.instances[idx]
        img = Image.open(self.data_root / ins["patch"]).convert("RGB")
        corners = np.array(ins["corners"], dtype=np.float32)  # (8,2) [0,1]
        if self.augment:
            img, corners = self._augment(img, corners)
        img = T.ToTensor()(img)              # (3,256,256) [0,1]
        img = self.normalize(img)            # ImageNet 归一化
        heatmaps = make_gaussian_heatmaps(corners, self.heatmap_size, self.sigma)
        return img, torch.from_numpy(heatmaps)

    def _augment(self, img, corners):
        """随机缩放 + 平移，角点做同样的仿射变换（train 专用）。"""
        scale = float(np.random.uniform(0.85, 1.15))
        tx = float(np.random.uniform(-0.1, 0.1))
        ty = float(np.random.uniform(-0.1, 0.1))
        img = T.functional.affine(img, angle=0, translate=(tx, ty),
                                  scale=scale, shear=0)
        corners = scale * (corners - 0.5) + 0.5 + np.array([tx, ty], dtype=np.float32)
        return img, np.clip(corners, 0.0, 1.0).astype(np.float32)


def evaluate_corner_error(model, loader, device):
    """验证集平均角点定位误差：预测 argmax 角点 vs GT 角点（归一化 L2 距离）。"""
    model.eval()
    errs = []
    with torch.no_grad():
        for img, hm in loader:
            img = img.to(device)
            pred = model(img)                                # (B,8,64,64) [0,1] heatmap
            gt_corners = decode_heatmaps(hm.to(device))            # (B,8,2)
            pred_corners = decode_heatmaps(pred)        # (B,8,2)
            errs.append(torch.norm(pred_corners - gt_corners, dim=2))  # (B,8)
    errs = torch.cat(errs, dim=0)                       # (N,8)
    return errs.mean().item(), errs.mean(dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, default=DATA_ROOT)
    ap.add_argument("--out-root", type=str, default=OUT_ROOT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-decoder", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone", action="store_true", default=True,
                    help="冻结 backbone，只训练 decoder+head（推荐，默认开启）")
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                    help="解冻 backbone 一起微调")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available()
        else f"cuda:{args.device}")

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    (out_root / "weights").mkdir(parents=True, exist_ok=True)

    train_ds = CornerHeatmapDataset(data_root, "train", sigma=args.sigma, augment=True)
    val_ds = CornerHeatmapDataset(data_root, "val", sigma=args.sigma, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    print(f"train: {len(train_ds)} 实例, val: {len(val_ds)} 实例")

    model = ResNet18Heatmap(num_heatmaps=8,
                            pretrained=not args.no_pretrained).to(device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr_decoder)
    else:
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder},
        ])

    print("=" * 70)
    print(f"开始训练: epochs={args.epochs}, batch={args.batch_size}, "
          f"freeze_backbone={args.freeze_backbone}, lr_decoder={args.lr_decoder}"
          + ("" if args.freeze_backbone else f", lr_backbone={args.lr_backbone}"))
    print("=" * 70)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        for img, hm in train_loader:
            img, hm = img.to(device), hm.to(device)
            pred = model(img)
            loss = weighted_mse_loss(pred, hm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, hm in val_loader:
                img, hm = img.to(device), hm.to(device)
                val_loss += weighted_mse_loss(model(img), hm).item() * img.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_root / "weights" / "best.pth")

        val_err = None
        if epoch % 5 == 0 or epoch == args.epochs:
            val_err, _ = evaluate_corner_error(model, val_loader, device)

        msg = (f"epoch {epoch:3d}/{args.epochs}  "
               f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if val_err is not None:
            msg += f"  val_err={val_err * 256:.1f}px"
        msg += "  *best*" if val_loss == best_val else ""
        print(msg)

    torch.save(model.state_dict(), out_root / "weights" / "last.pth")

    # 用 best 权重评估角点误差
    model.load_state_dict(torch.load(out_root / "weights" / "best.pth"))
    mean_err, per_inst_err = evaluate_corner_error(model, val_loader, device)

    # 保存 loss 曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("s2_p1 Corner Heatmap Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_root / "loss_curves.png", dpi=150)
    plt.close()

    # 保存训练汇总
    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_decoder": args.lr_decoder,
        "freeze_backbone": args.freeze_backbone,
        "sigma": args.sigma,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "train_loss_first": history["train_loss"][0],
        "train_loss_last": history["train_loss"][-1],
        "val_loss_first": history["val_loss"][0],
        "val_loss_last": history["val_loss"][-1],
        "best_val_loss": best_val,
        "mean_corner_error_norm": mean_err,
        "mean_corner_error_px": mean_err * 256.0,
        "corner_order": CORNER_ORDER,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("=" * 70)
    print("训练完成")
    print(f"  train_loss: {history['train_loss'][0]:.6f} -> {history['train_loss'][-1]:.6f}")
    print(f"  val_loss:   {history['val_loss'][0]:.6f} -> {history['val_loss'][-1]:.6f}")
    print(f"  best val_loss: {best_val:.6f}")
    print(f"  验证集平均角点定位误差: {mean_err:.5f} (归一化) = {mean_err * 256:.2f} px")
    print(f"  结果输出目录: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
