"""fig11_1_01_network_architecture.py
神经网络架构对比：感知机 + MLP（VisualTorch）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

# ── 画布 ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1, 1]})
fig.suptitle("图 11.1.1　神经网络架构对比",
             fontsize=22, fontweight="bold", y=0.98)

# =====================================================================
# Panel (a) — 感知机模型
# =====================================================================
ax1.set_axis_off()
ax1.set_xlim(-1.5, 8.5)
ax1.set_ylim(-2.5, 6.5)
ax1.set_aspect("equal")
ax1.set_title("(a) 感知机模型", fontsize=17, fontweight="bold", pad=12)

# Color helpers
blue_light = "#dbeafe"
orange_light = "#ffedd5"
neuron_ec = "#333333"
neuron_lw = 1.8
arrow_color = "#444444"

# ── Input neurons ─────────────────────────────────────────────────
input_x = 1.0
input_ys = [5.0, 3.0, 1.0]
input_labels = ["$x_1$", "$x_2$", "$x_3$"]
input_radius = 0.45

for y, label in zip(input_ys, input_labels):
    circ = Circle((input_x, y), input_radius,
                  fc=blue_light, ec=COLORS["blue"], lw=neuron_lw, zorder=5)
    ax1.add_patch(circ)
    ax1.text(input_x, y, label, ha="center", va="center",
             fontsize=15, fontweight="bold", color=COLORS["blue"], zorder=6)

# ── Bias node ─────────────────────────────────────────────────────
bias_x, bias_y = 1.0, -1.0
bias_radius = 0.45
circ_bias = Circle((bias_x, bias_y), bias_radius,
                   fc="#e2e8f0", ec=COLORS["gray"], lw=neuron_lw, zorder=5)
ax1.add_patch(circ_bias)
ax1.text(bias_x, bias_y, "$+1$", ha="center", va="center",
         fontsize=14, fontweight="bold", color=COLORS["gray"], zorder=6)

# ── Output neuron ─────────────────────────────────────────────────
output_x, output_y = 6.5, 2.5
output_radius = 0.7
circ_out = Circle((output_x, output_y), output_radius,
                  fc=orange_light, ec=COLORS["orange"], lw=2.2, zorder=5)
ax1.add_patch(circ_out)
ax1.text(output_x, output_y, r"$\Sigma \!\to\! \sigma$",
         ha="center", va="center",
         fontsize=15, fontweight="bold", color=COLORS["orange"], zorder=6)

# ── Output label ŷ ───────────────────────────────────────────────
ax1.annotate(
    r"$\hat{y}$", xy=(output_x + output_radius + 0.15, output_y),
    xytext=(output_x + output_radius + 0.9, output_y),
    fontsize=18, fontweight="bold", color=COLORS["orange"],
    ha="center", va="center",
    arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"],
                    lw=2.0, mutation_scale=18))

# ── Arrows: inputs → output with weight labels ───────────────────
weight_labels = ["$w_1$", "$w_2$", "$w_3$"]
# Offsets for label placement (perpendicular to arrow)
label_offsets = [0.50, 0.20, -0.30]

for y, w_label, offset in zip(input_ys, weight_labels, label_offsets):
    # Compute start/end on circle boundaries
    dx = output_x - input_x
    dy = output_y - y
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist

    start = (input_x + input_radius * ux, y + input_radius * uy)
    end = (output_x - output_radius * ux, output_y - output_radius * uy)

    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", color=arrow_color, lw=1.8,
        mutation_scale=16, zorder=3,
        connectionstyle="arc3,rad=0")
    ax1.add_patch(arrow)

    # Weight label (placed at midpoint with offset)
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    # Perpendicular direction for offset
    perp_x, perp_y = -uy, ux
    ax1.text(mid_x + perp_x * offset, mid_y + perp_y * offset,
             w_label, ha="center", va="center",
             fontsize=14, fontstyle="italic", color=COLORS["purple"],
             zorder=4)

# ── Arrow: bias → output ─────────────────────────────────────────
dx_b = output_x - bias_x
dy_b = output_y - bias_y
dist_b = np.sqrt(dx_b**2 + dy_b**2)
ux_b, uy_b = dx_b / dist_b, dy_b / dist_b

start_b = (bias_x + bias_radius * ux_b, bias_y + bias_radius * uy_b)
end_b = (output_x - output_radius * ux_b, output_y - output_radius * uy_b)

arrow_b = FancyArrowPatch(
    start_b, end_b,
    arrowstyle="-|>", color=COLORS["gray"], lw=1.8,
    mutation_scale=16, zorder=3, linestyle="--",
    connectionstyle="arc3,rad=0")
ax1.add_patch(arrow_b)

# Bias label
mid_bx = (start_b[0] + end_b[0]) / 2
mid_by = (start_b[1] + end_b[1]) / 2
perp_bx, perp_by = -uy_b, ux_b
ax1.text(mid_bx + perp_bx * (-0.45), mid_by + perp_by * (-0.45),
         "$b$", ha="center", va="center",
         fontsize=14, fontstyle="italic", color=COLORS["gray"], zorder=4)

# =====================================================================
# Panel (b) — 多层感知机 (MLP) via VisualTorch
# =====================================================================
ax2.set_axis_off()
ax2.set_title("(b) 多层感知机 (MLP)", fontsize=17, fontweight="bold", pad=12)

try:
    from torch import nn
    import visualtorch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(4, 8),
                nn.Linear(8, 6),
                nn.Linear(6, 3),
            )

        def forward(self, x):
            return self.layers(x)

    model = MLP()
    img = visualtorch.graph_view(model, input_shape=(1, 4))

    ax2.imshow(img)

    # Layer labels below the image
    layer_labels = ["输入层", "隐藏层1", "隐藏层2", "输出层"]
    img_w = img.size[0]
    n_layers = len(layer_labels)
    for i, label in enumerate(layer_labels):
        frac = (i + 0.5) / n_layers
        ax2.text(frac, -0.06, label,
                 transform=ax2.transAxes,
                 ha="center", va="top",
                 fontsize=13, fontweight="bold", color=COLORS["blue"])

except ImportError:
    # Fallback: draw a simple MLP diagram manually if visualtorch unavailable
    ax2.text(0.5, 0.5,
             "需要安装 torch 和 visualtorch\npip install torch visualtorch",
             ha="center", va="center", transform=ax2.transAxes,
             fontsize=14, color=COLORS["red"],
             bbox=dict(boxstyle="round,pad=0.5", fc="#fee2e2", ec=COLORS["red"]))

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, __file__, "fig11_1_01_network_architecture")
