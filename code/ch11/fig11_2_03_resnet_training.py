"""fig11_2_03_resnet_training.py
残差网络：(a) 残差块结构示意图  (b) 普通网络 vs 残差网络训练误差对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("图 11.2.3　残差网络结构与训练分析",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 残差块内部结构示意图
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-1, 11)
ax1.set_ylim(-0.5, 9)
ax1.set_aspect("equal")

# Colors
c_conv = COLORS["blue"]
c_bn = COLORS["teal"]
c_relu = COLORS["orange"]
c_skip = COLORS["red"]
c_sum = COLORS["green"]

box_w, box_h = 3.0, 0.8
center_x = 4.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=13):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black",
        linewidth=1.5, alpha=0.85,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")

def draw_arrow(ax, x1, y1, x2, y2, color="black", lw=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))

# Input label
y_input = 8.0
ax1.text(center_x, y_input + 0.3, "$\\mathbf{x}_l$（输入）",
         ha="center", va="bottom", fontsize=14, fontweight="bold")

# Main path boxes
y_conv1 = 7.0
draw_box(ax1, center_x, y_conv1, box_w, box_h, "Conv 3×3", c_conv)
draw_arrow(ax1, center_x, y_input - 0.1, center_x, y_conv1 + box_h/2 + 0.05)

y_bn1 = 5.8
draw_box(ax1, center_x, y_bn1, box_w, box_h, "Batch Norm", c_bn)
draw_arrow(ax1, center_x, y_conv1 - box_h/2 - 0.05, center_x, y_bn1 + box_h/2 + 0.05)

y_relu1 = 4.6
draw_box(ax1, center_x, y_relu1, box_w, box_h, "ReLU", c_relu)
draw_arrow(ax1, center_x, y_bn1 - box_h/2 - 0.05, center_x, y_relu1 + box_h/2 + 0.05)

y_conv2 = 3.4
draw_box(ax1, center_x, y_conv2, box_w, box_h, "Conv 3×3", c_conv)
draw_arrow(ax1, center_x, y_relu1 - box_h/2 - 0.05, center_x, y_conv2 + box_h/2 + 0.05)

y_bn2 = 2.2
draw_box(ax1, center_x, y_bn2, box_w, box_h, "Batch Norm", c_bn)
draw_arrow(ax1, center_x, y_conv2 - box_h/2 - 0.05, center_x, y_bn2 + box_h/2 + 0.05)

# Sum circle (⊕)
y_sum = 1.0
circle = plt.Circle((center_x, y_sum), 0.35, facecolor=c_sum,
                     edgecolor="black", linewidth=2, alpha=0.9, zorder=5)
ax1.add_patch(circle)
ax1.text(center_x, y_sum, "⊕", ha="center", va="center",
         fontsize=20, fontweight="bold", color="white", zorder=6)
draw_arrow(ax1, center_x, y_bn2 - box_h/2 - 0.05, center_x, y_sum + 0.4)

# ReLU after sum
y_relu_out = -0.1
draw_box(ax1, center_x, y_relu_out, box_w, box_h, "ReLU", c_relu)
draw_arrow(ax1, center_x, y_sum - 0.4, center_x, y_relu_out + box_h/2 + 0.05)

# Output label
ax1.text(center_x, y_relu_out - box_h/2 - 0.4, "$\\mathbf{x}_{l+1}$（输出）",
         ha="center", va="top", fontsize=14, fontweight="bold")

# Skip connection (curved arrow on the right side)
skip_x = center_x + box_w/2 + 0.8
ax1.annotate(
    "", xy=(center_x + 0.4, y_sum),
    xytext=(center_x + 0.4, y_input - 0.1),
    arrowprops=dict(
        arrowstyle="-|>", color=c_skip, lw=2.5,
        connectionstyle="arc3,rad=-0.4",
        mutation_scale=18,
    ),
)
# Skip label
ax1.text(skip_x + 0.6, (y_input + y_sum) / 2, "捷径连接\n(Identity)",
         ha="center", va="center", fontsize=12, fontweight="bold",
         color=c_skip,
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=c_skip, alpha=0.9))

# F(x) label on left
ax1.text(center_x - box_w/2 - 0.6, (y_conv1 + y_bn2) / 2,
         "$\\mathcal{F}(\\mathbf{x})$",
         ha="center", va="center", fontsize=16, fontweight="bold",
         color=COLORS["gray"], rotation=90)

ax1.set_title("(a) 残差块 (Basic Block)", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 普通网络 vs 残差网络训练误差
# ══════════════════════════════════════════════════════════════════
# Synthetic data matching He et al. 2015 CIFAR-10 trends
np.random.seed(42)
epochs = np.arange(1, 161)

def smooth(y, sigma=3):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma=sigma)

# Plain networks
base_plain20 = 9.5 * np.exp(-0.02 * epochs) + 8.0 + \
               0.3 * np.random.randn(160)
plain20_error = smooth(np.clip(base_plain20, 6.0, 20.0), sigma=4)

base_plain56 = 10.0 * np.exp(-0.015 * epochs) + 9.5 + \
               0.3 * np.random.randn(160)
plain56_error = smooth(np.clip(base_plain56, 8.0, 22.0), sigma=4)

# ResNets (better performance)
base_res20 = 9.0 * np.exp(-0.025 * epochs) + 6.8 + \
             0.25 * np.random.randn(160)
res20_error = smooth(np.clip(base_res20, 5.5, 18.0), sigma=4)

base_res56 = 8.0 * np.exp(-0.03 * epochs) + 5.5 + \
             0.25 * np.random.randn(160)
res56_error = smooth(np.clip(base_res56, 4.5, 17.0), sigma=4)

LW = 2.5

ax2.plot(epochs, plain20_error, color=COLORS["blue"], lw=LW,
         ls="--", label="Plain-20", alpha=0.85)
ax2.plot(epochs, plain56_error, color=COLORS["red"], lw=LW,
         ls="--", label="Plain-56", alpha=0.85)
ax2.plot(epochs, res20_error, color=COLORS["blue"], lw=LW,
         ls="-", label="ResNet-20", alpha=0.85)
ax2.plot(epochs, res56_error, color=COLORS["red"], lw=LW,
         ls="-", label="ResNet-56", alpha=0.85)

# Annotate the degradation problem
ax2.annotate(
    "退化问题:\nPlain-56 > Plain-20",
    xy=(140, plain56_error[-20]), xytext=(100, 16),
    fontsize=12, fontweight="bold", color=COLORS["red"],
    arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white",
              ec=COLORS["red"], alpha=0.9),
)

# Annotate ResNet improvement
ax2.annotate(
    "残差连接:\n更深 = 更好",
    xy=(140, res56_error[-20]), xytext=(95, 3.5),
    fontsize=12, fontweight="bold", color=COLORS["green"],
    arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="white",
              ec=COLORS["green"], alpha=0.9),
)

ax2.set_xlabel("训练轮数 (Epoch)", fontsize=16)
ax2.set_ylabel("训练误差 (%)", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=13, loc="upper right",
           ncol=2, framealpha=0.9)
ax2.set_xlim(0, 160)
ax2.set_ylim(2, 22)
ax2.grid(alpha=0.3)
ax2.set_title("(b) 普通网络 vs 残差网络", fontsize=17,
              fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_2_03_resnet_training")
