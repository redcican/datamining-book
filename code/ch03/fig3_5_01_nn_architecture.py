"""
图 3.5.1  神经网络架构示意图：单神经元计算模型与多层感知机（MLP）结构
对应节次：3.5 神经网络分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_5_01_nn_architecture.py
输出路径：public/figures/ch03/fig3_5_01_nn_architecture.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

apply_style()

# ── 颜色方案 ──────────────────────────────────────────────────────────────────
C_INPUT  = COLORS["blue"]
C_HIDDEN = COLORS["teal"]
C_HIDDEN2 = COLORS["purple"]
C_OUTPUT = COLORS["red"]
C_CONN   = "#94a3b8"
C_BIAS   = COLORS["gray"]

fig = plt.figure(figsize=(18, 8))

# 两个子图：左(a)宽度约40%，右(b)宽度约60%
ax1 = fig.add_axes([0.02, 0.05, 0.40, 0.88])
ax2 = fig.add_axes([0.44, 0.05, 0.55, 0.88])
for ax in (ax1, ax2):
    ax.set_axis_off()

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): 单神经元计算模型
# ═══════════════════════════════════════════════════════════════════════════════
ax1.set_xlim(0, 10)
ax1.set_ylim(-0.8, 7.2)

def circle(ax, cx, cy, r, fc, ec="white", lw=1.5, zorder=4, alpha=0.92):
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
               linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(c)

def arrow(ax, x1, y1, x2, y2, color=C_CONN, lw=1.4, zorder=2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=zorder)

def roundbox(ax, cx, cy, w, h, fc, ec="white", lw=1.5, zorder=4, alpha=0.92):
    bbox = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.08", facecolor=fc,
                          edgecolor=ec, linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(bbox)

# --- 输入节点 (x=1.5) ---
INPUT_X = 1.5
SUM_X   = 5.0
ACT_X   = 7.0
OUT_X   = 9.0
NODE_R  = 0.38

input_ys = [5.8, 3.5, 1.2]
input_labels = ["$x_1$", "$x_2$", "$x_3$"]
weight_labels = ["$w_1$", "$w_2$", "$w_3$"]

for iy, lbl in zip(input_ys, input_labels):
    circle(ax1, INPUT_X, iy, NODE_R, fc=C_INPUT)
    ax1.text(INPUT_X, iy, lbl, ha="center", va="center",
             fontsize=13, color="white", fontweight="bold", zorder=5)

# --- 偏置节点 ---
BIAS_Y = -0.2
circle(ax1, INPUT_X, BIAS_Y, NODE_R, fc=C_BIAS)
ax1.text(INPUT_X, BIAS_Y, "+1", ha="center", va="center",
         fontsize=11, color="white", fontweight="bold", zorder=5)

# --- 求和节点 ---
SUM_Y = 3.5
circle(ax1, SUM_X, SUM_Y, NODE_R * 1.3, fc=C_HIDDEN)
ax1.text(SUM_X, SUM_Y, "$\\Sigma$", ha="center", va="center",
         fontsize=18, color="white", fontweight="bold", zorder=5)

# --- 连接：输入 → 求和 (带权重标签) ---
for iy, wlbl in zip(input_ys, weight_labels):
    ax1.annotate("", xy=(SUM_X - NODE_R * 1.3, SUM_Y),
                 xytext=(INPUT_X + NODE_R, iy),
                 arrowprops=dict(arrowstyle="-|>", color=C_CONN, lw=1.3),
                 zorder=2)
    mx = (INPUT_X + NODE_R + SUM_X - NODE_R * 1.3) / 2
    my = (iy + SUM_Y) / 2
    ax1.text(mx + 0.15, my, wlbl, ha="center", va="center",
             fontsize=12, color=COLORS["orange"],
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

# --- 偏置连接 (虚线) ---
ax1.annotate("", xy=(SUM_X - NODE_R * 1.3, SUM_Y - 0.4),
             xytext=(INPUT_X + NODE_R, BIAS_Y),
             arrowprops=dict(arrowstyle="-|>", color=C_BIAS, lw=1.2,
                             linestyle="dashed"),
             zorder=2)
ax1.text((INPUT_X + SUM_X) / 2 + 0.2, (BIAS_Y + SUM_Y) / 2 - 0.3,
         "$b$", ha="center", va="center", fontsize=13, color=C_BIAS,
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))

# --- 求和 → z 标注 ---
ax1.annotate("", xy=(ACT_X - NODE_R * 1.3, SUM_Y),
             xytext=(SUM_X + NODE_R * 1.3, SUM_Y),
             arrowprops=dict(arrowstyle="-|>", color=C_CONN, lw=1.5),
             zorder=2)
ax1.text((SUM_X + ACT_X) / 2, SUM_Y + 0.45, "$z = \\mathbf{w}^\\top\\mathbf{x}+b$",
         ha="center", va="bottom", fontsize=12, color=C_HIDDEN)

# --- 激活函数节点 ---
roundbox(ax1, ACT_X, SUM_Y, w=1.1, h=0.8, fc=C_HIDDEN2)
ax1.text(ACT_X, SUM_Y, "$\\sigma(\\cdot)$", ha="center", va="center",
         fontsize=14, color="white", fontweight="bold", zorder=5)

# --- 激活 → 输出 ---
ax1.annotate("", xy=(OUT_X - NODE_R, SUM_Y),
             xytext=(ACT_X + 0.55, SUM_Y),
             arrowprops=dict(arrowstyle="-|>", color=C_CONN, lw=1.5),
             zorder=2)
ax1.text((ACT_X + OUT_X) / 2, SUM_Y + 0.45,
         "$a = \\sigma(z)$", ha="center", va="bottom", fontsize=12, color=C_HIDDEN2)

# --- 输出节点 ---
circle(ax1, OUT_X, SUM_Y, NODE_R, fc=C_OUTPUT)
ax1.text(OUT_X, SUM_Y, "$a$", ha="center", va="center",
         fontsize=14, color="white", fontweight="bold", zorder=5)

# --- 公式注释 ---
ax1.text(5.0, 6.9, "$z = \\sum_j w_j x_j + b$", ha="center", fontsize=13,
         color=C_HIDDEN,
         bbox=dict(fc="#f0f9ff", ec=C_HIDDEN, lw=1, alpha=0.9, boxstyle="round,pad=0.4"))

ax1.set_title("(a) 单个人工神经元的计算模型\n"
              "输入加权求和 $z$，经激活函数 $\\sigma$ 变换后输出激活值 $a$",
              fontsize=13, pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): 多层感知机（MLP）架构
# ═══════════════════════════════════════════════════════════════════════════════
ax2.set_xlim(-0.3, 11.5)
ax2.set_ylim(-3.8, 4.5)

LAYER_SIZES  = [3, 5, 4, 3]
LAYER_X      = [1.0, 3.8, 6.6, 9.4]
LAYER_COLORS = [C_INPUT, C_HIDDEN, C_HIDDEN2, C_OUTPUT]
LAYER_NAMES  = ["输入层\n$d=3$", "隐藏层 1\n$n_1=5$", "隐藏层 2\n$n_2=4$", "输出层\n$K=3$"]
V_SPACE      = 1.35     # 纵向节点间距
MLP_R        = 0.30     # 节点半径

def get_ys(n, spacing=V_SPACE):
    ys = np.arange(n, dtype=float) * spacing
    return ys - ys.mean()

all_ys = [get_ys(n) for n in LAYER_SIZES]

# --- 连接（先画，在节点下方）---
for l in range(len(LAYER_SIZES) - 1):
    for i, y1 in enumerate(all_ys[l]):
        for j, y2 in enumerate(all_ys[l + 1]):
            ax2.plot([LAYER_X[l] + MLP_R, LAYER_X[l + 1] - MLP_R],
                     [y1, y2],
                     color=C_CONN, lw=0.5, alpha=0.60, zorder=1)

# --- 节点 ---
INPUT_LABELS  = ["$x_1$", "$x_2$", "$x_3$"]
OUTPUT_LABELS = ["$\\hat{p}_1$", "$\\hat{p}_2$", "$\\hat{p}_3$"]

for l, (n, lx, ys, lc) in enumerate(zip(LAYER_SIZES, LAYER_X, all_ys, LAYER_COLORS)):
    for i, y in enumerate(ys):
        circle(ax2, lx, y, MLP_R, fc=lc, lw=1.2)
        # 输入层：显示 x_i
        if l == 0:
            ax2.text(lx, y, INPUT_LABELS[i], ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold", zorder=5)
        # 输出层：显示 ŷ_i
        elif l == len(LAYER_SIZES) - 1:
            ax2.text(lx, y, OUTPUT_LABELS[i], ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold", zorder=5)
        # 隐藏层：显示 σ
        else:
            ax2.text(lx, y, "$\\sigma$", ha="center", va="center",
                     fontsize=9, color="white", zorder=5)

# --- 层标签（底部）---
for l, (lx, lname) in enumerate(zip(LAYER_X, LAYER_NAMES)):
    bot = all_ys[l].min() - MLP_R - 0.35
    ax2.text(lx, bot, lname, ha="center", va="top", fontsize=11,
             color=LAYER_COLORS[l], fontweight="bold")

# --- 权重矩阵标签（层间中间，顶部）---
WM_LABELS = ["$\\mathbf{W}^{(1)}$", "$\\mathbf{W}^{(2)}$", "$\\mathbf{W}^{(3)}$"]
ACT_LABELS = ["ReLU", "ReLU", "Softmax"]
for l in range(len(LAYER_SIZES) - 1):
    mid_x = (LAYER_X[l] + LAYER_X[l + 1]) / 2
    top_y = max(all_ys[l].max(), all_ys[l + 1].max()) + MLP_R + 0.55
    ax2.text(mid_x, top_y, WM_LABELS[l], ha="center", va="bottom",
             fontsize=13, color="#334155")
    ax2.text(mid_x, top_y + 0.50, ACT_LABELS[l], ha="center", va="bottom",
             fontsize=10, color="#64748b", style="italic")

# --- 输入箭头（左侧）---
for i, y in enumerate(all_ys[0]):
    ax2.annotate("", xy=(LAYER_X[0] - MLP_R, y),
                 xytext=(LAYER_X[0] - MLP_R - 0.55, y),
                 arrowprops=dict(arrowstyle="-|>", color=C_INPUT, lw=1.3), zorder=3)

# --- 输出箭头（右侧）---
for i, y in enumerate(all_ys[-1]):
    ax2.annotate("", xy=(LAYER_X[-1] + MLP_R + 0.55, y),
                 xytext=(LAYER_X[-1] + MLP_R, y),
                 arrowprops=dict(arrowstyle="-|>", color=C_OUTPUT, lw=1.3), zorder=3)

# --- 架构标注框 (右移避开 ReLU 标签) ---
ax2.text(7.5, 4.2,
         "$\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$,"
         "   $\\mathbf{a}^{(l)} = \\sigma(\\mathbf{z}^{(l)})$",
         ha="center", va="bottom", fontsize=12,
         color="#1e40af",
         bbox=dict(fc="#eff6ff", ec="#1e40af", lw=1.2, alpha=0.9,
                   boxstyle="round,pad=0.4"))

ax2.set_title(
    "(b) 多层感知机（MLP）：输入→隐藏层1→隐藏层2→输出\n"
    "每层节点与相邻层全连接；隐藏层用 ReLU 激活，输出层用 Softmax",
    fontsize=13, pad=8)

fig.suptitle("神经网络架构：从单个神经元到多层感知机",
             fontsize=14, y=1.025, fontweight="bold")

save_fig(fig, __file__, "fig3_5_01_nn_architecture")
