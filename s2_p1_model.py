#!/usr/bin/env python3
"""
s2_p1 阶段：resnet18 + CNN decoder + heatmaps head 网络

架构（输入 256×256 RGB，输出 8 张 64×64 角点 heatmap）：

    resnet18 backbone (torchvision 官方, 去掉 avgpool+fc)
        512 × 8 × 8
    Deconv (ConvTranspose2d 512→256, k4 s2 p1) -> 256 × 16 × 16  + BN + ReLU
    Deconv (ConvTranspose2d 256→256, k4 s2 p1) -> 256 × 32 × 32  + BN + ReLU
    Deconv (ConvTranspose2d 256→256, k4 s2 p1) -> 256 × 64 × 64  + BN + ReLU
    1×1 Conv (256→8)                            ->   8 × 64 × 64  (8 张角点 heatmap)

说明：
- backbone 用 torchvision 官方 resnet18（可选 ImageNet 预训练），只保留到 layer4，
  输出 512×8×8（输入 256×256 时）。
- decoder + head 手写。
- 最终 1×1 卷积不接激活函数，输出 raw logits；训练时用 sigmoid + 损失
  （见 s2_p2 阶段）。可视化时对输出做 torch.sigmoid 得到 [0,1] heatmap。
- 输入需要按 ImageNet 归一化（mean/std），因为 backbone 预训练。归一化在数据
  读取阶段做（s2_p2），不写进模型，保持模型通用。

用法:
    # 仅测试前向（打印结构 + 输出 shape + 参数量）
    python3 s2_p1_model.py [--device 0] [--no-pretrained]
"""

import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 与 s2_p0_preprocess.py 的 CORNER_ORDER 保持一致
CORNER_ORDER = [
    "(min_x,min_y,min_z)", "(max_x,min_y,min_z)",
    "(min_x,max_y,min_z)", "(max_x,max_y,min_z)",
    "(min_x,min_y,max_z)", "(max_x,min_y,max_z)",
    "(min_x,max_y,max_z)", "(max_x,max_y,max_z)",
]


class CNNHeatmapDecoder(nn.Module):
    """手写 CNN decoder + heatmaps head：8×8 -> 64×64，输出 8 张 heatmap。"""

    def __init__(self, in_channels: int = 512, mid_channels: int = 256,
                 num_heatmaps: int = 8):
        super().__init__()
        # 3 次上采样，每次 stride=2 尺寸翻倍：8->16->32->64
        self.deconv1 = nn.ConvTranspose2d(in_channels, mid_channels,
                                          kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.deconv2 = nn.ConvTranspose2d(mid_channels, mid_channels,
                                          kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.deconv3 = nn.ConvTranspose2d(mid_channels, mid_channels,
                                          kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.relu = nn.ReLU(inplace=True)

        # 1×1 conv: 256 -> 8（每张 heatmap 对应一个角点）
        self.head = nn.Conv2d(mid_channels, num_heatmaps, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.deconv1(x)))   # (N,256,16,16)
        x = self.relu(self.bn2(self.deconv2(x)))   # (N,256,32,32)
        x = self.relu(self.bn3(self.deconv3(x)))   # (N,256,64,64)
        x = self.head(x)                            # (N,8,64,64)
        return x


class ResNet18Heatmap(nn.Module):
    """resnet18 主干 + CNN decoder + heatmaps head，输出 8×64×64 角点 heatmap。"""

    def __init__(self, num_heatmaps: int = 8, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        # 去掉 avgpool + fc，只保留 conv1..layer4，输出 512×8×8（256×256 输入）
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = CNNHeatmapDecoder(in_channels=512, mid_channels=256,
                                         num_heatmaps=num_heatmaps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (N,512,8,8)
        x = self.decoder(x)    # (N,8,64,64)
        return x


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total:,}  trainable={trainable:,}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=str, default="0",
                    help="GPU 编号，或 'cpu'；默认 0")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不使用 ImageNet 预训练权重")
    ap.add_argument("--num-heatmaps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.device == "cpu" or not torch.cuda.is_available()
        else f"cuda:{args.device}")

    model = ResNet18Heatmap(num_heatmaps=args.num_heatmaps,
                            pretrained=not args.no_pretrained).to(device)

    print("=" * 70)
    print("resnet18 + CNN decoder + heatmaps head 网络")
    print("=" * 70)
    print(model)
    print("-" * 70)
    print(f"参数量: {count_params(model)}")
    print(f"backbone 输出: 512×8×8 (输入 {args.input_size}×{args.input_size})")
    print(f"最终输出: {args.num_heatmaps}×64×64 (8 个角点 heatmap)")
    print(f"角点顺序: {CORNER_ORDER}")
    print("-" * 70)

    # 前向测试
    x = torch.randn(args.batch, 3, args.input_size, args.input_size, device=device)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"输入 shape : {tuple(x.shape)}")
    print(f"输出 shape : {tuple(y.shape)}")
    assert y.shape == (args.batch, args.num_heatmaps, 64, 64), "输出 shape 不符合预期"
    print("前向通过 ✓ 输出 = (N, 8, 64, 64)")
    if torch.cuda.is_available() and args.device != "cpu":
        print(f"设备: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
