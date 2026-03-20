"""fig11_4_01_autoencoder_architecture.py
(a) 自编码器编码-解码框架  (b) 二维潜在空间聚类可视化"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.1, 1]})
fig.suptitle("图 11.4.1　自编码器架构与潜在空间",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 编码-解码框架
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-1, 21)
ax1.set_ylim(-1.5, 12)
ax1.set_aspect("equal")

c_encoder = COLORS["blue"]
c_decoder = COLORS["red"]
c_latent = COLORS["purple"]
c_arrow = COLORS["gray"]

LW = 2.0

def draw_layer(ax, cx, cy, w, h, color, alpha=0.85, lw=1.5):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=lw, alpha=alpha)
    ax.add_patch(rect)

def arr(ax, x1, y1, x2, y2, color=c_arrow, lw=LW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))

# 784 → 256 → 64 → 2 → 64 → 256 → 784
layer_heights = [8.0, 5.5, 3.5, 1.5, 3.5, 5.5, 8.0]
layer_widths  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
layer_x = [1.5, 4.5, 7.5, 10.0, 12.5, 15.5, 18.5]
layer_colors = [c_encoder, c_encoder, c_encoder, c_latent,
                c_decoder, c_decoder, c_decoder]
layer_labels = ["784", "256", "64", "2", "64", "256", "784"]

cy = 5.0

for i, (x, h, w, color, label) in enumerate(
        zip(layer_x, layer_heights, layer_widths, layer_colors, layer_labels)):
    alpha = 0.75 if i != 3 else 0.90
    draw_layer(ax1, x, cy, w, h, color, alpha=alpha,
               lw=2.5 if i == 3 else 1.5)
    ax1.text(x, cy - h/2 - 0.5, label, fontsize=13, fontweight="bold",
             color=color, ha="center", va="top")

for i in range(len(layer_x) - 1):
    x1 = layer_x[i] + layer_widths[i] / 2
    x2 = layer_x[i+1] - layer_widths[i+1] / 2
    h1 = layer_heights[i]
    h2 = layer_heights[i+1]
    for frac in [0.3, 0.5, 0.7]:
        y1 = cy - h1/2 + h1 * frac
        y2 = cy - h2/2 + h2 * frac
        ax1.plot([x1, x2], [y1, y2], color=c_arrow, lw=0.6, alpha=0.4)
    arr(ax1, x1 + 0.05, cy, x2 - 0.05, cy, color=c_arrow, lw=1.5)

# Labels
ax1.text(1.5, cy + 4.8, "$\\mathbf{x}$\n输入", fontsize=15,
         fontweight="bold", color=c_encoder, ha="center", va="bottom")
ax1.text(18.5, cy + 4.8, "$\\hat{\\mathbf{x}}$\n重建", fontsize=15,
         fontweight="bold", color=c_decoder, ha="center", va="bottom")
ax1.text(10.0, cy + 1.6, "$\\mathbf{z}$", fontsize=17,
         fontweight="bold", color=c_latent, ha="center", va="bottom")

# Brackets
ax1.annotate("", xy=(1.0, -0.8), xytext=(8.0, -0.8),
             arrowprops=dict(arrowstyle="-", color=c_encoder, lw=2))
ax1.plot([1.0, 1.0], [-0.6, -1.0], color=c_encoder, lw=2)
ax1.plot([8.0, 8.0], [-0.6, -1.0], color=c_encoder, lw=2)
ax1.text(4.5, -1.2, "编码器 $\\phi$", fontsize=15, fontweight="bold",
         color=c_encoder, ha="center", va="top")

ax1.plot([9.5, 9.5], [-0.6, -1.0], color=c_latent, lw=2)
ax1.plot([10.5, 10.5], [-0.6, -1.0], color=c_latent, lw=2)
ax1.annotate("", xy=(9.5, -0.8), xytext=(10.5, -0.8),
             arrowprops=dict(arrowstyle="-", color=c_latent, lw=2))
ax1.text(10.0, -1.2, "瓶颈", fontsize=14, fontweight="bold",
         color=c_latent, ha="center", va="top")

ax1.annotate("", xy=(12.0, -0.8), xytext=(19.0, -0.8),
             arrowprops=dict(arrowstyle="-", color=c_decoder, lw=2))
ax1.plot([12.0, 12.0], [-0.6, -1.0], color=c_decoder, lw=2)
ax1.plot([19.0, 19.0], [-0.6, -1.0], color=c_decoder, lw=2)
ax1.text(15.5, -1.2, "解码器 $\\psi$", fontsize=15, fontweight="bold",
         color=c_decoder, ha="center", va="top")

# Loss annotation
ax1.text(10.0, 10.5,
         "$\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2$",
         fontsize=16, fontweight="bold", color=COLORS["gray"],
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec=COLORS["gray"], alpha=0.9))
ax1.annotate("", xy=(5.5, 10.5), xytext=(8.5, 10.5),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"],
                             lw=1.5, ls="--", mutation_scale=12))
ax1.annotate("", xy=(14.5, 10.5), xytext=(11.8, 10.5),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"],
                             lw=1.5, ls="--", mutation_scale=12))

ax1.set_title("(a) 编码-解码框架", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 二维潜在空间可视化
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)

n_per_class = 80
class_names = ["0", "1", "2", "3", "4"]
class_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                COLORS["orange"], COLORS["purple"]]

centers = [(-2.0, 2.0), (2.0, 2.5), (0.0, -2.0), (-2.5, -1.0), (2.5, -1.5)]

for i, (cx, cy_c) in enumerate(centers):
    spread = 0.55 + 0.1 * np.random.rand()
    x = np.random.randn(n_per_class) * spread + cx
    y = np.random.randn(n_per_class) * spread + cy_c
    ax2.scatter(x, y, c=class_colors[i], alpha=0.5, s=25, edgecolors="none",
                label=f"数字 {class_names[i]}")
    ax2.text(cx, cy_c, class_names[i], fontsize=20, fontweight="bold",
             color=class_colors[i], ha="center", va="center", alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                       ec=class_colors[i], alpha=0.7))

ax2.set_xlabel("$z_1$", fontsize=17)
ax2.set_ylabel("$z_2$", fontsize=17)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=13, loc="upper left", ncol=1, framealpha=0.9)
ax2.set_xlim(-4.5, 4.5)
ax2.set_ylim(-4.5, 4.5)
ax2.grid(alpha=0.3)

ax2.text(3.5, -4.0,
         "不同类别自动\n形成聚类",
         fontsize=14, fontweight="bold", color=COLORS["gray"],
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0",
                   ec=COLORS["gray"], alpha=0.9))

ax2.set_title("(b) 二维潜在空间可视化", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_4_01_autoencoder_architecture")
