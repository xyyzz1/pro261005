#!/usr/bin/env python3
"""
s2_p0 阶段：BOP 数据预处理 —— 裁剪 gt_box + 整理为 heatmap 网络输入

- 从 bop_raw 读取：姿态 scene_gt.json、框 scene_gt_info.json、相机 camera.json、模型 models_info.json
- 投影 3D 包围盒 8 角点得到 corners_2d（与 annotations.json 一致，已验证）
- 按 bbox_obj 裁剪方形 patch（加 padding），resize 到固定尺寸 patch_size
- 角点变换到 patch 归一化坐标 [0,1]
- 划分 train/val，输出到 s2_p0_data/

输出结构:
    s2_p0_data/
        images/train/<id>.jpg
        images/val/<id>.jpg
        annotations.json    # 每个实例: patch 路径 + 8 角点(归一化) + bbox + 来源

用法:
    python3 s2_p0_preprocess.py [--patch-size 256] [--pad 1.3] [--val-ratio 0.2]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

CORNER_ORDER = [
    "(min_x,min_y,min_z)", "(max_x,min_y,min_z)",
    "(min_x,max_y,min_z)", "(max_x,max_y,min_z)",
    "(min_x,min_y,max_z)", "(max_x,min_y,max_z)",
    "(min_x,max_y,max_z)", "(max_x,max_y,max_z)",
]


def load_3d_corners(models_info: dict, obj_id: int) -> np.ndarray:
    """由 models_info 的 min/max 得到 8 个 3D 包围盒角点（mm）。"""
    mi = models_info[str(obj_id)]
    return np.array([
        [mi["min_x"], mi["min_y"], mi["min_z"]],
        [mi["max_x"], mi["min_y"], mi["min_z"]],
        [mi["min_x"], mi["max_y"], mi["min_z"]],
        [mi["max_x"], mi["max_y"], mi["min_z"]],
        [mi["min_x"], mi["min_y"], mi["max_z"]],
        [mi["max_x"], mi["min_y"], mi["max_z"]],
        [mi["min_x"], mi["max_y"], mi["max_z"]],
        [mi["max_x"], mi["max_y"], mi["max_z"]],
    ], dtype=np.float64)


def project(corners_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """3D 角点(mm) -> 2D 像素坐标。返回 (8, 2)。"""
    Pc = (R @ corners_3d.T).T + t.T          # (8,3) 相机系
    uv = (K @ Pc.T).T                        # (8,3) 齐次
    return np.stack([uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]], axis=1)


def crop_patch(img: Image.Image, bbox, pad: float, patch_size: int):
    """按 bbox 裁剪方形 patch 并 resize。返回 (patch, x0, y0, sx, sy)。"""
    W, H = img.size
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * pad
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + int(round(side))
    y1 = y0 + int(round(side))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    patch = img.crop((x0, y0, x1, y1)).resize((patch_size, patch_size), Image.BILINEAR)
    sx = patch_size / (x1 - x0)
    sy = patch_size / (y1 - y0)
    return patch, x0, y0, sx, sy


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bop-root", type=str,
                    default="/home/xyyzz1/pro261005/my_dataset_corners/bop_raw")
    ap.add_argument("--out-root", type=str,
                    default="/home/xyyzz1/pro261005/my_dataset_corners/s2_p0_data")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--pad", type=float, default=1.3, help="裁剪 padding（相对 max(w,h) 的倍数）")
    ap.add_argument("--bbox", type=str, choices=["bbox_obj", "bbox_visib"], default="bbox_obj")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bop_root = Path(args.bop_root).resolve()
    out_root = Path(args.out_root).resolve()
    patch_size = args.patch_size

    camera = json.loads((bop_root / "camera.json").read_text())
    models_info = json.loads((bop_root / "models" / "models_info.json").read_text())
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)

    # 收集所有实例
    instances = []  # dict
    for split_dir in sorted(bop_root.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "models":
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.isdigit():
                continue
            gt_path = scene_dir / "scene_gt.json"
            info_path = scene_dir / "scene_gt_info.json"
            rgb_dir = scene_dir / "rgb"
            if not (gt_path.exists() and info_path.exists() and rgb_dir.is_dir()):
                continue
            gt = json.loads(gt_path.read_text())
            info = json.loads(info_path.read_text())
            for im_id in sorted(gt.keys(), key=int):
                img_file = rgb_dir / f"{int(im_id):06d}.jpg"
                if not img_file.exists():
                    continue
                for inst_idx, (obj, ann) in enumerate(zip(gt[im_id], info[im_id])):
                    obj_id = int(obj["obj_id"])
                    R = np.array(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
                    t = np.array(obj["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
                    corners_2d = project(load_3d_corners(models_info, obj_id), R, t, K)
                    bbox = ann.get(args.bbox)
                    if bbox is None:
                        continue
                    instances.append({
                        "scene": scene_dir.name,
                        "split_name": split_dir.name,
                        "im_id": im_id,
                        "inst_idx": inst_idx,
                        "obj_id": obj_id,
                        "corners_2d": corners_2d,
                        "bbox": bbox,
                        "visib_fract": ann.get("visib_fract", 1.0),
                        "img_file": img_file,
                    })

    print(f"发现 {len(instances)} 个实例")

    # 划分 train/val（稳定排序 + 固定 seed）
    instances.sort(key=lambda d: (d["split_name"], d["scene"], int(d["im_id"]), d["inst_idx"]))
    rng = random.Random(args.seed)
    rng.shuffle(instances)
    n_val = int(len(instances) * args.val_ratio)

    out_annos = []
    n_outside = 0
    for i, ins in enumerate(instances):
        split = "val" if i < n_val else "train"
        img = Image.open(ins["img_file"]).convert("RGB")
        patch, x0, y0, sx, sy = crop_patch(img, ins["bbox"], args.pad, patch_size)

        # 角点 -> patch 归一化坐标
        corners_norm = (ins["corners_2d"] - np.array([x0, y0])) * np.array([sx, sy]) / patch_size
        if corners_norm.min() < -0.01 or corners_norm.max() > 1.01:
            n_outside += 1
        corners_norm = np.clip(corners_norm, 0.0, 1.0)

        patch_id = f"{ins['scene']}_{ins['im_id']}_{ins['inst_idx']}"
        patch_rel = f"images/{split}/{patch_id}.jpg"
        patch.save(out_root / patch_rel)

        out_annos.append({
            "id": patch_id,
            "split": split,
            "patch": patch_rel,
            "obj_id": ins["obj_id"],
            "source_image": str(ins["img_file"].relative_to(bop_root)),
            "bbox_obj": ins["bbox"],
            "visib_fract": round(ins["visib_fract"], 4),
            "corners": [[round(float(c[0]), 6), round(float(c[1]), 6)] for c in corners_norm],
        })

    # 写 annotations.json
    n_train = sum(1 for a in out_annos if a["split"] == "train")
    n_val2 = sum(1 for a in out_annos if a["split"] == "val")
    meta = {
        "description": "s2_p0: 裁剪 gt_box 得到的对象 patch + 8 角点（归一化）",
        "patch_size": patch_size,
        "pad": args.pad,
        "bbox": args.bbox,
        "num_corners": 8,
        "corner_order": CORNER_ORDER,
        "num_instances": len(out_annos),
        "num_train": n_train,
        "num_val": n_val2,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (out_root / "annotations.json").write_text(
        json.dumps({"meta": meta, "instances": out_annos}, indent=2))
    print(f"完成: 输出到 {out_root}")
    print(f"  train: {n_train} 个 patch, val: {n_val2} 个 patch, 共 {len(out_annos)}")
    print(f"  角点越界(pad 未覆盖)实例数: {n_outside}")


if __name__ == "__main__":
    main()
