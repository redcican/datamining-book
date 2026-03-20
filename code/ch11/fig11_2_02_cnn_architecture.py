"""fig11_2_02_cnn_architecture.py
经典 CNN 架构可视化：(a) LeNet-5 via VisualTorch layered_view  (b) VGG 风格 via layered_view"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

# ── LeNet-5 模型定义 ──────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),   # 32×32 → 32×32
            nn.ReLU(),
            nn.AvgPool2d(2, 2),               # → 16×16
            nn.Conv2d(6, 16, 5),              # → 12×12
            nn.ReLU(),
            nn.AvgPool2d(2, 2),               # → 6×6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── VGG 风格模型定义 ──────────────────────────────────────────────
class VGGLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── 生成 VisualTorch 可视化 ──────────────────────────────────────
import visualtorch

# (a) LeNet-5 layered_view (same style as VGG for consistency)
lenet = LeNet5()
input_shape_lenet = (1, 1, 32, 32)
img_lenet = visualtorch.layered_view(lenet, input_shape=input_shape_lenet,
                                      legend=False)

# (b) VGG-like layered_view
vgg = VGGLike()
input_shape_vgg = (1, 3, 32, 32)
img_vgg = visualtorch.layered_view(vgg, input_shape=input_shape_vgg,
                                    legend=True)

# ── 组合绘图 ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("图 11.2.2　经典 CNN 架构可视化",
             fontsize=22, fontweight="bold", y=0.98)

# (a) LeNet-5
ax1.imshow(img_lenet)
ax1.set_axis_off()
ax1.set_title("(a) LeNet-5 层状结构", fontsize=17, fontweight="bold")

# (b) VGG-like
ax2.imshow(img_vgg)
ax2.set_axis_off()
ax2.set_title("(b) VGG 风格网络结构", fontsize=17, fontweight="bold")

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_2_02_cnn_architecture")
