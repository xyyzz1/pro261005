# 项目记录 (Record)

> 命名规范：`s<阶段>_p<子阶段>_<描述>.py`。s1 = YOLO11 检测（p0 数据 / p1 架构+推理 / p2 训练）；s2 = resnet18 + cnn decoder + heatmaps 关键点网络（p0 数据预处理 / …）。

## s1_p0 阶段：BOP 数据集 → YOLO11 检测训练数据

- 日期：2026-10-05
- 目标：将 BOP 数据集 `my_dataset_corners/bop_raw` 转换为 YOLO11 检测训练数据

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `my_dataset_corners/bop2yolo.py` | 新建 | BOP→YOLO 转换脚本（初始位置） |
| `./s1_p0_bop2yolo.py` | 移动 + 重命名 | 上述脚本移到项目根目录，按 `s1_p0` 命名 |

脚本功能（`s1_p0_bop2yolo.py`）：
- 读取 `scene_gt.json`（obj_id）与 `scene_gt_info.json`（bbox），将 BOP 框 `[x, y, w, h]` 转为 YOLO 归一化标签 `cls x_c y_c w h`
- `obj_id → class_id` 映射（0 起，按 obj_id 升序；本数据集 1 类：`obj_000001`）
- 按 split/scene 遍历所有样本，随机划分 train/val（默认 8:2，seed=42）
- 生成 `data.yaml`
- 可选参数：`--bbox bbox_visib|bbox_obj`、`--val-ratio`、`--min-visib`、`--bop-root`、`--out-root`

### 执行的命令

数据探索：
```bash
find /home/xyyzz1/pro261005/my_dataset_corners/bop_raw -maxdepth 3 -type d
ls -la /home/xyyzz1/pro261005/my_dataset_corners/bop_raw
cat /home/xyyzz1/pro261005/my_dataset_corners/bop_raw/camera.json
python3 -c "import json; d=json.load(open('train_pbr/000000/scene_gt.json')); print(len(d), d[list(d)[0]])"
python3 -c "import json; d=json.load(open('train_pbr/000000/scene_gt_info.json')); print(json.dumps(d[list(d)[0]], indent=2))"
python3 -c "import json; d=json.load(open('train_pbr/000000/scene_camera.json')); print(d[list(d)[0]])"
ls train_pbr/000000/rgb | wc -l
```

运行转换：
```bash
cd /home/xyyzz1/pro261005/my_dataset_corners
python3 bop2yolo.py
```

验证输出：
```bash
cat yolo_dataset/data.yaml
cat yolo_dataset/labels/train/<sample>.txt
find yolo_dataset -maxdepth 2 -type d
```

移动重命名：
```bash
mv /home/xyyzz1/pro261005/my_dataset_corners/bop2yolo.py /home/xyyzz1/pro261005/s1_p0_bop2yolo.py
```

### 结果

- 输入：`bop_raw`（1 个 split `train_pbr`、1 个 scene、100 张 RGB 图、1 个物体 obj_id=1）
- 输出：`my_dataset_corners/yolo_dataset/`
  - `images/train`: 80 张，`images/val`: 20 张
  - `labels/train`、`labels/val`（同名 `.txt`）
  - `data.yaml`：`names: {0: obj_000001}`
- 标签使用 `bbox_visib`（可见框）

### 后续训练

```bash
yolo detect train data=/home/xyyzz1/pro261005/my_dataset_corners/yolo_dataset/data.yaml model=yolo11n.pt epochs=300 imgsz=640
```

## s1_p1 阶段：YOLO11 网络架构 + 推理（ultralytics 官方实现）

- 日期：2026-10-05
- 目标：创建 conda 环境、使用 ultralytics 官方 YOLO11 实现编写网络架构并跑通推理

### 环境

- 新建 conda 环境 `pro261005`（Python 3.10）
- torch 2.11.0+cu128（CUDA 可用，NVIDIA GeForce RTX 5060 Laptop GPU）
- ultralytics 8.4.173

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `./s1_p1_yolo11.py` | 新建 | YOLO11 加载/架构打印/推理脚本 |
| `./s1_p1_yolo11.py` | 修改 | 修复保存路径（project 传绝对路径，避免被 ultralytics 加 `runs/detect` 前缀） |
| `./s1_data/bus.jpg`、`zidane.jpg` | 下载 | 推理测试图（ultralytics 官方示例图） |

脚本功能（`s1_p1_yolo11.py`）：
- 加载 YOLO11 预训练模型（`yolo11n/s/m/l/x.pt`，官方 COCO 权重）
- 打印网络架构逐层结构，并存档到 `architecture.txt`
- 对输入图像/目录执行推理，保存标注结果并打印检测框
- 参数：`--source`、`--model`、`--conf`、`--imgsz`、`--device`、`--save-dir`

### 执行的命令

创建环境：
```bash
conda create -n pro261005 python=3.10 -y
```

安装依赖：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
/home/xyyzz1/miniconda3/envs/pro261005/bin/pip install ultralytics
```

下载测试图：
```bash
mkdir -p /home/xyyzz1/pro261005/s1_data
curl -sL -o s1_data/bus.jpg https://ultralytics.com/images/bus.jpg
curl -sL -o s1_data/zidane.jpg https://ultralytics.com/images/zidane.jpg
```

验证环境：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> torch 2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU
```

跑通推理：
```bash
cd /home/xyyzz1/pro261005
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s1_p1_yolo11.py --source s1_data/bus.jpg --device 0
```

### 结果

- 推理成功：`bus.jpg` 检测到 5 个目标（bus ×1、person ×4）
- 标注结果：`s1_out/exp/bus.jpg`，架构文本：`s1_out/architecture.txt`
- 模型权重：`yolo11n.pt`（自动下载，约 5.4MB）
- GPU 推理速度：约 20 ms/图（RTX 5060）

## s1_p2 阶段：单类小规模训练 + loss 曲线 + 验证集可视化

- 日期：2026-10-05
- 目标：设置单类（nc=1）、用全部训练集小规模训练、输出 loss 变化、验证集跑框并画图

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `my_dataset_corners/yolo_dataset/data.yaml` | 修改 | 显式增加 `nc: 1`（单类） |
| `./s1_p2_train.py` | 新建 | 训练 + 从 results.csv 绘制 loss 曲线脚本 |
| `./s1_p2_plot_loss.py` | 新建 | 独立 loss 绘图脚本（仅 pandas+matplotlib，避免 ultralytics 导入冲突） |
| `./s1_p2_val_visualize.py` | 新建 | 验证集推理 + 把预测框画在图上保存 |

脚本功能：
- `s1_p2_train.py`：加载 `yolo11n.pt`，在全部训练集（80 图）上训练，训练后绘制 loss 曲线并打印 loss/mAP 汇总
- `s1_p2_plot_loss.py`：从 `results.csv` 画 train/val loss 与 mAP 曲线
- `s1_p2_val_visualize.py`：用 `best.pt` 对验证集（20 图）跑推理，`save=True` 画框保存

### 执行的命令

安装 pandas（绘图依赖）：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/pip install pandas
```

小规模训练（30 epochs，全部训练集）：
```bash
cd /home/xyyzz1/pro261005
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s1_p2_train.py --epochs 30
```

绘制 loss 曲线（重新生成无中文字体告警的图）：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s1_p2_plot_loss.py
```

验证集推理 + 框可视化：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s1_p2_val_visualize.py
```

### 结果

训练 loss 变化（30 epochs，单类）：
- `train/box_loss`: 0.8715 → 0.5357
- `train/cls_loss`: 2.9816 → 0.9424
- `train/dfl_loss`: 1.1897 → 0.9482
- `val/box_loss`: 0.8447 → 0.8897
- `val/cls_loss`: 2.8552 → 1.2714
- `val/dfl_loss`: 1.2109 → 1.2369

验证指标：
- `mAP50`: 0.0623 → 0.8670（best 0.8690）
- `mAP50-95`: 0.0338 → 0.6175（best 0.6195）

产物：
- `s2_out/train/results.csv`、`loss_curves.png`、`weights/best.pt`（5.5MB）、`weights/last.pt`
- `s2_out/val_pred/exp/*.jpg`：20 张验证图共 61 个预测框，已画在图上

## 命名重排（2026-10-05）

将原 S0/S1/S2 三个阶段的脚本重命名为 `s1_p0` / `s1_p1` / `s1_p2` 前缀：

```bash
cd /home/xyyzz1/pro261005
mv s0_bop2yolo.py s1_p0_bop2yolo.py
mv s1_yolo11.py s1_p1_yolo11.py
mv s2_train.py s1_p2_train.py
mv s2_plot_loss.py s1_p2_plot_loss.py
mv s2_val_visualize.py s1_p2_val_visualize.py
```

## s2_p0 阶段：BOP 数据预处理（裁剪 gt_box + 角点）

- 日期：2026-10-05
- 目标：从 bop_raw 裁剪 gt_box，整理为 resnet18 + cnn decoder + heatmaps head 网络的输入

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `./s2_p0_preprocess.py` | 新建 | 裁剪 gt_box + 投影 8 角点 + 划分 train/val |

脚本功能（`s2_p0_preprocess.py`）：
- 从 bop_raw 读 `scene_gt.json`（位姿）/ `scene_gt_info.json`（bbox）/ `camera.json` / `models_info.json`（3D bbox）
- 投影 3D 包围盒 8 角点 → `corners_2d`（已验证与 annotations.json 一致，误差 0.005px）
- 按 `bbox_obj` 裁剪方形 patch（pad=1.3），resize 到 256×256
- 角点变换到 patch 归一化坐标 [0,1]
- 划分 train/val（8:2，seed=42）

### 执行的命令

```bash
cd /home/xyyzz1/pro261005
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s2_p0_preprocess.py
```

### 结果

- 260 个实例 → train 208 / val 52 个 patch（256×256）
- 输出：`my_dataset_corners/s2_p0_data/`
  - `images/train`、`images/val`
  - `annotations.json`（每实例含 8 角点归一化坐标 + bbox + 来源图像）
- 角点越界实例数：0（pad=1.3 已覆盖全部角点）

## s2_p1 阶段：resnet18 + CNN decoder + heatmaps head 网络（搭建 + 训练）

- 日期：2026-10-05
- 目标：搭建 resnet18（torchvision）+ 手写 CNN decoder + heatmaps head，输出 8 张角点 heatmap；用 s2_p0_data 训练，结果输出到 s2_p1_result

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `./s2_p1_model.py` | 新建 | 网络定义 + 前向测试 |
| `./s2_p1_train.py` | 新建 | 训练脚本（数据加载 / 高斯 heatmap GT / 加权 MSE / 增强 / 评估） |
| `./s2_p1_train.py` | 修改 | 修正 DATA_ROOT；decode 批处理化；损失 MSE→focal→加权 MSE；冻结 backbone；数据增强；matplotlib 惰性导入 |

脚本功能：
- `s2_p1_model.py`：`ResNet18Heatmap`（torchvision resnet18 去 avgpool+fc，得 512×8×8）+ `CNNHeatmapDecoder`（3×ConvTranspose2d 256，逐层 8→16→32→64，各接 BN+ReLU）+ 1×1 Conv → 8×64×64；输出 raw（无激活），训练时用加权 MSE 回归 [0,1] 高斯 heatmap
- `s2_p1_train.py`：从 s2_p0_data 读 patch + 8 角点（归一化），生成 64×64 高斯 heatmap（σ=2、峰值=1），加权 MSE 损失（峰值像素×50），随机缩放/平移增强（角点同步仿射），训练并保存 best/last 权重 + loss 曲线 + summary.json

### 执行的命令

前向测试（验证输出 shape = (B,8,64,64)）：
```bash
cd /home/xyyzz1/pro261005
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s2_p1_model.py --device 0
```

训练（增强 + 微调 backbone，最终配置）：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s2_p1_train.py --epochs 100 --batch-size 16 --no-freeze-backbone --device 0
```

### 结果

- 网络：15,375,176 参数；输入 256×256 → 512×8×8 → 256×16×16 → 256×32×32 → 256×64×64 → 8×64×64
- 训练 loss：train 0.323 → 0.039；val 0.301 → 0.090（best 0.083）
- 验证集平均角点定位误差：112 px（归一化 0.44）
- 产物：`s2_p1_result/weights/best.pth`、`last.pth`、`loss_curves.png`、`summary.json`

### 说明（重要）

- 角点定位误差偏高（112px，差于"预测训练均值角点"基线 92px）：单物体仅 208 张训练 patch，任务本质是 6D 位姿 → 3D 包围盒角点投影，数据量不足以让网络泛化。模型在训练集上可近乎完美拟合（前向/过拟合测试 4 张图角点误差→0），但验证集不泛化。
- 训练中先后尝试：MSE / focal loss / 加权 MSE × 冻结 / 解冻 backbone × 有无增强。最优组合为「加权 MSE + 缩放平移增强 + 微调 backbone」。
- 数据实际路径为 `/home/xyyzz1/pro261005/s2_p0_data`（项目根目录，非 my_dataset_corners 内）。

## s2_p2 阶段：heatmap → 8 角点 解码程序

- 日期：2026-10-05
- 目标：把 s2_p1 网络输出的 8 张 heatmap 解码为 8 个角点（s2_p1 训练暂缓，后续回头再训）

### 编写操作

| 文件 | 操作 | 说明 |
|---|---|---|
| `./s2_p2_heatmap2corners.py` | 新建 | heatmap → 角点解码 + 推理 + 对比 GT + 可视化 |

脚本功能（`s2_p2_heatmap2corners.py`）：
- `heatmap_to_coords(heatmaps, method)`：核心解码函数，三种方式
  - `argmax`：峰值像素
  - `subpixel`：argmax + 一维抛物线亚像素精化（默认，更准）
  - `softargmax`：归一化加权质心
- `load_model`：加载 `ResNet18Heatmap` + `best.pth` 权重
- 主流程：对每个 patch 前向得 8×64×64 heatmap → 解码为 8 个归一化 [0,1] 角点 → 与 annotations.json 的 GT 对比算误差 → 存 predictions.json + 可视化（角点散点 + 立方体线框 + heatmap 叠加）

### 执行的命令

解码精度验证（合成高斯 heatmap，已知角点）：
```bash
/home/xyyzz1/miniconda3/envs/pro261005/bin/python -c "…heatmap_to_coords 三方法对比…"
```
结果：argmax 1.88px / subpixel 0.06px / softargmax 0.00px

完整运行（val 集 + 可视化）：
```bash
cd /home/xyyzz1/pro261005
/home/xyyzz1/miniconda3/envs/pro261005/bin/python s2_p2_heatmap2corners.py --split val --method subpixel --visualize --num-vis 6
```

### 结果

- 解码精度：subpixel 亚像素精化把 argmax 量化误差（1.88px）降到 0.06px
- val 集 52 实例：平均角点误差 112.0 px（与 s2_p1 模型当前泛化水平一致，待后续训练提升）
- 产物：`s2_p2_result/predictions.json`（每实例 8 角点 + GT + error_norm）、`s2_p2_result/visualize/*.png`（6 张）


