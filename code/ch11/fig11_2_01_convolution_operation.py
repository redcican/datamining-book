"""fig11_2_01_convolution_operation.py
卷积运算示意图：(a) 二维卷积滑动窗口  (b) 多通道卷积"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
fig.suptitle("图 11.2.1　卷积运算示意图",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 二维卷积：5×5 输入, 3×3 卷积核, stride=1, padding=0 → 3×3 输出
# ══════════════════════════════════════════════════════════════════

# 输入矩阵
input_data = np.array([
    [2, 0, 1, 3, 1],
    [1, 3, 2, 0, 2],
    [0, 1, 3, 1, 0],
    [2, 0, 1, 2, 3],
    [1, 2, 0, 1, 1],
])
# 卷积核
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
])

cell = 0.7  # cell size
gap = 0.15  # gap between elements

# Helper: draw a grid with values
def draw_grid(ax, data, x0, y0, cell_size, title, color,
              highlight=None, fontsize=13, edgecolor="black"):
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            x = x0 + j * cell_size
            y = y0 + (rows - 1 - i) * cell_size
            fc = "white"
            if highlight is not None and highlight[i, j]:
                fc = color
            rect = FancyBboxPatch(
                (x, y), cell_size, cell_size,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor=edgecolor,
                linewidth=1.5, alpha=0.85 if fc != "white" else 1.0,
            )
            ax.add_patch(rect)
            txt_color = "white" if fc != "white" else "black"
            ax.text(x + cell_size / 2, y + cell_size / 2,
                    str(int(data[i, j])),
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold",
                    color=txt_color)
    # title
    ax.text(x0 + cols * cell_size / 2, y0 + rows * cell_size + 0.25,
            title, ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=color)

# Compute output
output = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        output[i, j] = np.sum(input_data[i:i+3, j:j+3] * kernel)

# Highlight: first output position (top-left 3×3 of input)
highlight_input = np.zeros((5, 5), dtype=bool)
highlight_input[0:3, 0:3] = True

highlight_output = np.zeros((3, 3), dtype=bool)
highlight_output[0, 0] = True

# Positions
inp_x0, inp_y0 = 0.3, 0.5
kern_x0, kern_y0 = inp_x0 + 5 * cell + 1.0, 1.2
out_x0, out_y0 = kern_x0 + 3 * cell + 1.0, 0.8

# Draw grids
draw_grid(ax1, input_data, inp_x0, inp_y0, cell, "输入 (5×5)",
          COLORS["blue"], highlight=highlight_input)
draw_grid(ax1, kernel, kern_x0, kern_y0, cell, "卷积核 (3×3)",
          COLORS["orange"])
draw_grid(ax1, output, out_x0, out_y0, cell, "输出 (3×3)",
          COLORS["green"], highlight=highlight_output)

# Operator symbols
mid1_x = inp_x0 + 5 * cell + 0.5
mid1_y = inp_y0 + 2.5 * cell
ax1.text(mid1_x, mid1_y, "∗", fontsize=28, fontweight="bold",
         ha="center", va="center", color=COLORS["gray"])

mid2_x = kern_x0 + 3 * cell + 0.5
mid2_y = kern_y0 + 1.5 * cell
ax1.text(mid2_x, mid2_y, "=", fontsize=28, fontweight="bold",
         ha="center", va="center", color=COLORS["gray"])

# Computation annotation
val = int(output[0, 0])
comp_text = (f"2×1 + 0×0 + 1×(−1)\n"
             f"+1×1 + 3×0 + 2×(−1)\n"
             f"+0×1 + 1×0 + 3×(−1)\n"
             f"= {val}")
ax1.annotate(
    comp_text,
    xy=(out_x0 + cell / 2, out_y0 + 2 * cell + cell / 2),
    xytext=(out_x0 + 2.0, out_y0 + 3.5 * cell + 0.6),
    fontsize=10, fontfamily="monospace",
    ha="center", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc="#fffbe6", ec=COLORS["orange"],
              alpha=0.95),
    arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5),
)

ax1.set_xlim(-0.3, out_x0 + 3 * cell + 2.5)
ax1.set_ylim(-0.2, 5.5)
ax1.set_aspect("equal")
ax1.set_axis_off()
ax1.set_title("(a) 二维卷积运算", fontsize=17, fontweight="bold", pad=20)

# ══════════════════════════════════════════════════════════════════
# (b) 多通道卷积示意图 (3D 块状图)
# ══════════════════════════════════════════════════════════════════
ax2.set_axis_off()
ax2.set_xlim(-1, 15)
ax2.set_ylim(-1, 8)
ax2.set_aspect("equal")

# Helper: draw a 3D block (parallelogram-style)
def draw_block(ax, x, y, w, h, d, color, label=None, label_pos="below",
               alpha=0.7, fontsize=12):
    dx, dy = d * 0.35, d * 0.25  # perspective offset
    # Front face
    front = plt.Polygon(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
        facecolor=color, edgecolor="black", linewidth=1.2, alpha=alpha)
    ax.add_patch(front)
    # Top face
    top = plt.Polygon(
        [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
        facecolor=color, edgecolor="black", linewidth=1.2, alpha=alpha * 0.8)
    ax.add_patch(top)
    # Right face
    right = plt.Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        facecolor=color, edgecolor="black", linewidth=1.2, alpha=alpha * 0.6)
    ax.add_patch(right)
    if label:
        if label_pos == "below":
            ax.text(x + w / 2, y - 0.35, label,
                    ha="center", va="top", fontsize=fontsize, fontweight="bold")
        elif label_pos == "above":
            ax.text(x + w / 2 + dx / 2, y + h + dy + 0.25, label,
                    ha="center", va="bottom", fontsize=fontsize, fontweight="bold")

# Input block (C_in=3, H=5, W=5 → draw as 3-layer slab)
channel_colors = [COLORS["red"], COLORS["green"], COLORS["blue"]]
for i, cc in enumerate(channel_colors):
    draw_block(ax2, 0.2 + i * 0.3, 1.5 + i * 0.2, 2.5, 3.0, 0.6, cc, alpha=0.5)
ax2.text(1.45 + 0.45, 0.85, "输入\n$C_{in}{=}3$\n$5{\\times}5$",
         ha="center", va="top", fontsize=11, fontweight="bold",
         color=COLORS["gray"])

# Arrow
ax2.annotate("", xy=(4.5, 3.3), xytext=(3.5, 3.3),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"],
                             lw=2, mutation_scale=18))

# Filter blocks (2 filters, each C_in=3, k=3×3)
for f_idx in range(2):
    y_off = 1.0 + f_idx * 3.2
    for i, cc in enumerate(channel_colors):
        draw_block(ax2, 5.0 + i * 0.2, y_off + i * 0.15, 1.2, 1.5, 0.4, cc, alpha=0.55)
    ax2.text(5.6 + 0.3, y_off - 0.35,
             f"卷积核 {f_idx+1}\n$3{{\\times}}3{{\\times}}3$",
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=COLORS["orange"])

# Arrow
ax2.annotate("", xy=(8.2, 3.3), xytext=(7.2, 3.3),
             arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"],
                             lw=2, mutation_scale=18))

# Output block (C_out=2, H_out=3, W_out=3)
out_colors = [COLORS["purple"], COLORS["teal"]]
for i, oc in enumerate(out_colors):
    draw_block(ax2, 8.8 + i * 0.3, 1.8 + i * 0.2, 2.0, 2.5, 0.5, oc, alpha=0.55)
ax2.text(9.8 + 0.45, 1.15, "输出\n$C_{out}{=}2$\n$3{\\times}3$",
         ha="center", va="top", fontsize=11, fontweight="bold",
         color=COLORS["gray"])

# "+" bias annotation
ax2.text(12.0, 3.3, "+$\\mathbf{b}$", fontsize=16, fontweight="bold",
         ha="center", va="center", color=COLORS["orange"])

ax2.set_title("(b) 多通道卷积", fontsize=17, fontweight="bold", pad=20)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_2_01_convolution_operation")
