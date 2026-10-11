#!/usr/bin/env python3
"""
将 BOP 数据集转换为 YOLO11 检测训练数据。

BOP 目录结构（输入）:
    bop_raw/
        camera.json
        models/models_info.json
        models/obj_000001.ply
        train_pbr/<scene_id>/
            scene_gt.json        # 位姿: {im_id: [{obj_id, ...}]}
            scene_gt_info.json   # 框:  {im_id: [{bbox_obj, bbox_visib, ...}]}
            rgb/<im_id>.jpg

输出 YOLO 目录结构:
    yolo_dataset/
        images/train, images/val
        labels/train, labels/val
        data.yaml

标签格式 (每行): class_id x_center y_center w h   （归一化到 [0,1]）
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def discover_splits(bop_root: Path) -> list[Path]:
    """返回所有含场景子目录的 split 目录（如 train_pbr, test 等）。"""
    splits = []
    for d in sorted(bop_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "models":
            continue
        # split 目录下应含有 <scene_id> 形式的场景目录
        if any((d / s).is_dir() and s.isdigit() for s in os.listdir(d)):
            splits.append(d)
    return splits


def read_image_size(camera_json: Path):
    """从 camera.json 读图像宽高，缺省退回默认值。"""
    if camera_json.exists():
        cam = json.loads(camera_json.read_text())
        w = cam.get("width")
        h = cam.get("height")
        if w and h:
            return int(w), int(h)
    return None, None


def build_class_map(models_info: Path) -> dict[int, int]:
    """obj_id -> class_id（0 起，按 obj_id 升序）。"""
    info = json.loads(models_info.read_text())
    obj_ids = sorted(int(k) for k in info.keys())
    return {obj_id: i for i, obj_id in enumerate(obj_ids)}


def class_names(models_info: Path) -> list[str]:
    info = json.loads(models_info.read_text())
    return [f"obj_{int(k):06d}" for k in sorted(info.keys(), key=int)]


def find_image_path(rgb_dir: Path, im_id: str) -> Path | None:
    """根据 im_id 找到对应图像文件（兼容不同扩展名/补零）。"""
    for ext in IMG_EXTENSIONS:
        cand = rgb_dir / f"{int(im_id):06d}{ext}"
        if cand.exists():
            return cand
    return None


def bbox_to_yolo(box, img_w: int, img_h: int) -> str:
    """BOP bbox [x, y, w, h]（左上角 + 宽高）-> YOLO 归一化字符串。"""
    x, y, w, h = box
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    # 裁剪到 [0,1] 防止越界
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w_n = min(max(w_n, 0.0), 1.0)
    h_n = min(max(h_n, 0.0), 1.0)
    return f"{x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bop-root", type=str,
                    default="/home/xyyzz1/pro261005/my_dataset_corners/bop_raw",
                    help="BOP 数据集根目录")
    ap.add_argument("--out-root", type=str,
                    default="/home/xyyzz1/pro261005/my_dataset_corners/yolo_dataset",
                    help="YOLO 数据集输出目录")
    ap.add_argument("--bbox", type=str, choices=["bbox_visib", "bbox_obj"],
                    default="bbox_visib",
                    help="使用可见框(bbox_visib)还是完整框(bbox_obj)")
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="验证集比例")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（划分用）")
    ap.add_argument("--min-visib", type=float, default=0.0,
                    help="丢弃可见比例低于该值的样本（默认不过滤）")
    args = ap.parse_args()

    bop_root = Path(args.bop_root).resolve()
    out_root = Path(args.out_root).resolve()

    camera_json = bop_root / "camera.json"
    models_info = bop_root / "models" / "models_info.json"
    if not models_info.exists():
        raise SystemExit(f"找不到 models_info.json: {models_info}")

    obj_to_cls = build_class_map(models_info)
    names = class_names(models_info)
    img_w, img_h = read_image_size(camera_json)

    # 收集所有 (split, scene, im_id) 样本
    samples = []  # (split, scene_id, im_id)
    for split_dir in discover_splits(bop_root):
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.isdigit():
                continue
            gt_info_path = scene_dir / "scene_gt_info.json"
            rgb_dir = scene_dir / "rgb"
            if not gt_info_path.exists() or not rgb_dir.is_dir():
                continue
            for im_id in json.loads(gt_info_path.read_text()).keys():
                samples.append((split_dir.name, scene_dir.name, im_id))

    print(f"发现 {len(samples)} 个样本，{len(obj_to_cls)} 个类别，"
          f"bbox 使用 {args.bbox}")

    # 按 (split, scene, im_id) 稳定排序后做 train/val 划分
    samples.sort()
    random.seed(args.seed)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_set = set(samples[:n_val])

    # 输出目录
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    for split_name, scene_id, im_id in samples:
        src_split = bop_root / split_name
        scene_dir = src_split / scene_id
        rgb_dir = scene_dir / "rgb"
        gt_info = json.loads((scene_dir / "scene_gt_info.json").read_text())
        gt = json.loads((scene_dir / "scene_gt.json").read_text())

        annos = gt_info[im_id]
        objs = gt[im_id]

        # 图像宽高优先取 camera.json，否则从图片实际尺寸读取
        w, h = img_w, img_h
        img_src = find_image_path(rgb_dir, im_id)
        if img_src is None:
            n_skipped += 1
            continue
        if w is None or h is None:
            try:
                from PIL import Image
                with Image.open(img_src) as im:
                    w, h = im.size
            except Exception:
                print(f"  无法读取图像尺寸: {img_src}，跳过")
                n_skipped += 1
                continue

        dest_split = "val" if (split_name, scene_id, im_id) in val_set else "train"

        # 生成唯一文件名，避免不同 split/scene 间 im_id 冲突
        base = f"{split_name}_{scene_id}_{int(im_id):06d}"
        img_dst = out_root / "images" / dest_split / f"{base}{img_src.suffix}"
        lbl_dst = out_root / "labels" / dest_split / f"{base}.txt"

        lines = []
        for anno, obj in zip(annos, objs):
            obj_id = int(obj["obj_id"])
            if obj_id not in obj_to_cls:
                continue
            if anno.get("visib_fract", 1.0) < args.min_visib:
                continue
            box = anno.get(args.bbox)
            if box is None:
                continue
            cls_id = obj_to_cls[obj_id]
            lines.append(f"{cls_id} " + bbox_to_yolo(box, w, h))

        # 无有效标注则跳过
        if not lines:
            n_skipped += 1
            continue

        shutil.copyfile(img_src, img_dst)
        lbl_dst.write_text("\n".join(lines) + "\n")
        n_written += 1

    # 写 data.yaml
    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    (out_root / "data.yaml").write_text(
        "path: {}\ntrain: {}\nval: {}\nnames:\n{}".format(
            data_yaml["path"], data_yaml["train"], data_yaml["val"],
            "\n".join(f"  {i}: {n}" for i, n in enumerate(names)),
        ) + "\n"
    )

    print(f"完成：写出 {n_written} 张图像 + 标签，跳过 {n_skipped} 张")
    print(f"输出目录：{out_root}")
    print(f"  images/train: {len(list((out_root/'images'/'train').iterdir()))} 张")
    print(f"  images/val:   {len(list((out_root/'images'/'val').iterdir()))} 张")
    print(f"  data.yaml 类别：{names}")


if __name__ == "__main__":
    main()
