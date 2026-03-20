"""fig11_3_02_lstm_cell.py
LSTM 细胞结构详解：三门控机制 + 细胞状态信息高速公路"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle("图 11.3.2　LSTM 细胞结构详解",
             fontsize=22, fontweight="bold", y=0.97)
ax.set_axis_off()
ax.set_xlim(-1, 21)
ax.set_ylim(-2, 14)
ax.set_aspect("equal")

# ── 颜色定义 ─────────────────────────────────────────────────────
c_forget = COLORS["red"]
c_input = COLORS["green"]
c_output = COLORS["blue"]
c_cell = COLORS["orange"]
c_op = "#4a4a4a"
c_line = "#666666"

LW = 2.0  # line width

# ── 辅助函数 ─────────────────────────────────────────────────────
def draw_box(cx, cy, w, h, text, color, fontsize=14, text_color="white"):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def draw_circle(cx, cy, r, text, color, fontsize=16, text_color="white"):
    circ = Circle((cx, cy), r, facecolor=color, edgecolor="black",
                  linewidth=1.8, alpha=0.9, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=6)

def arr(x1, y1, x2, y2, color=c_line, lw=LW, head=True):
    style = "-|>" if head else "-"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=14))

def line(x1, y1, x2, y2, color=c_line, lw=LW, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=3)

# ══════════════════════════════════════════════════════════════════
# 布局参数
# ══════════════════════════════════════════════════════════════════
# Cell state highway: y = 11.5
y_cell = 11.5
# Hidden state output: y = 1.0
y_hout = 1.5
# Gate row: y = 5.0
y_gate = 5.5
# Input merge: y = 3.0
y_merge = 3.0

# x positions for gates
x_forget = 4.0
x_input_gate = 8.0
x_candidate = 10.5
x_output = 15.0

# ── 1. 细胞状态高速公路 (顶部水平线) ──────────────────────────────
arr(-0.5, y_cell, 20.5, y_cell, color=c_cell, lw=3.5)
ax.text(-0.8, y_cell, "$\\mathbf{c}_{t-1}$", fontsize=15,
        fontweight="bold", color=c_cell, ha="right", va="center")
ax.text(20.8, y_cell, "$\\mathbf{c}_t$", fontsize=15,
        fontweight="bold", color=c_cell, ha="left", va="center")
ax.text(10.0, y_cell + 0.8, "细胞状态 (信息高速公路)",
        fontsize=13, fontweight="bold", color=c_cell,
        ha="center", va="bottom", style="italic")

# ── 2. 输入: h_{t-1} 和 x_t ──────────────────────────────────────
ax.text(-0.8, y_merge, "$\\mathbf{h}_{t-1}$", fontsize=15,
        fontweight="bold", color=COLORS["blue"], ha="right", va="center")
line(-0.5, y_merge, 2.0, y_merge, color=COLORS["blue"], lw=2.5)

ax.text(1.0, 0.5, "$\\mathbf{x}_t$", fontsize=15,
        fontweight="bold", color=COLORS["green"], ha="center", va="center")
arr(1.0, 0.9, 1.0, y_merge - 0.3, color=COLORS["green"], lw=2.5)

# Concat point
draw_circle(2.0, y_merge, 0.3, "+", c_op, fontsize=14)

# Branch from concat point to all gates
line(2.3, y_merge, 18.0, y_merge, color=c_line, lw=1.5)

# ── 3. 遗忘门 (Forget Gate) ──────────────────────────────────────
# σ box
draw_box(x_forget, y_gate, 1.6, 1.0, "$\\sigma$", c_forget, fontsize=16)
ax.text(x_forget, y_gate + 0.8, "遗忘门", fontsize=12,
        fontweight="bold", color=c_forget, ha="center", va="bottom")

# Input → gate
arr(x_forget, y_merge + 0.3, x_forget, y_gate - 0.5, color=c_line)

# Gate → multiply with cell state
arr(x_forget, y_gate + 0.5, x_forget, y_cell - 0.5, color=c_forget)

# Multiply circle on cell state
draw_circle(x_forget, y_cell, 0.35, "⊗", c_forget, fontsize=14)

# Equation label
ax.text(x_forget, y_gate - 1.1, "$\\mathbf{f}_t$",
        fontsize=13, fontweight="bold", color=c_forget,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=c_forget, alpha=0.8))

# ── 4. 输入门 + 候选值 (Input Gate) ──────────────────────────────
# σ box (input gate)
draw_box(x_input_gate, y_gate, 1.6, 1.0, "$\\sigma$", c_input, fontsize=16)
ax.text(x_input_gate, y_gate + 0.8, "输入门", fontsize=12,
        fontweight="bold", color=c_input, ha="center", va="bottom")

# tanh box (candidate)
draw_box(x_candidate, y_gate, 1.6, 1.0, "tanh", c_input, fontsize=14)
ax.text(x_candidate, y_gate + 0.8, "候选值", fontsize=12,
        fontweight="bold", color=c_input, ha="center", va="bottom")

# Input arrows
arr(x_input_gate, y_merge + 0.3, x_input_gate, y_gate - 0.5, color=c_line)
arr(x_candidate, y_merge + 0.3, x_candidate, y_gate - 0.5, color=c_line)

# i_t label
ax.text(x_input_gate, y_gate - 1.1, "$\\mathbf{i}_t$",
        fontsize=13, fontweight="bold", color=c_input,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=c_input, alpha=0.8))

# c̃_t label
ax.text(x_candidate, y_gate - 1.1, "$\\tilde{\\mathbf{c}}_t$",
        fontsize=13, fontweight="bold", color=c_input,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=c_input, alpha=0.8))

# Multiply i_t and c̃_t
x_i_mul = (x_input_gate + x_candidate) / 2
y_i_mul = 8.5
draw_circle(x_i_mul, y_i_mul, 0.35, "⊗", c_input, fontsize=14)
arr(x_input_gate, y_gate + 0.5, x_i_mul - 0.25, y_i_mul - 0.25,
    color=c_input)
arr(x_candidate, y_gate + 0.5, x_i_mul + 0.25, y_i_mul - 0.25,
    color=c_input)

# i*c̃ → add to cell state
x_add = x_i_mul
draw_circle(x_add, y_cell, 0.35, "⊕", c_input, fontsize=14)
arr(x_i_mul, y_i_mul + 0.35, x_add, y_cell - 0.35, color=c_input)

# ── 5. 输出门 (Output Gate) ──────────────────────────────────────
# σ box
draw_box(x_output, y_gate, 1.6, 1.0, "$\\sigma$", c_output, fontsize=16)
ax.text(x_output, y_gate + 0.8, "输出门", fontsize=12,
        fontweight="bold", color=c_output, ha="center", va="bottom")

# Input → gate
arr(x_output, y_merge + 0.3, x_output, y_gate - 0.5, color=c_line)

# o_t label
ax.text(x_output, y_gate - 1.1, "$\\mathbf{o}_t$",
        fontsize=13, fontweight="bold", color=c_output,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec=c_output, alpha=0.8))

# tanh on cell state → multiply with o_t
x_tanh_out = 17.5
y_tanh_out = 8.5
draw_box(x_tanh_out, y_tanh_out, 1.4, 0.9, "tanh", c_output, fontsize=13)

# Cell → tanh
line(x_tanh_out, y_cell, x_tanh_out, y_tanh_out + 0.45, color=c_cell, lw=2)

# Multiply o_t and tanh(c_t)
x_out_mul = 17.5
y_out_mul = y_gate + 1.8
draw_circle(x_out_mul, y_out_mul, 0.35, "⊗", c_output, fontsize=14)

# σ → multiply
arr(x_output, y_gate + 0.5, x_out_mul - 0.3, y_out_mul - 0.15,
    color=c_output)
# tanh → multiply
arr(x_tanh_out, y_tanh_out - 0.45, x_out_mul, y_out_mul + 0.35,
    color=c_output)

# Output: h_t
arr(x_out_mul, y_out_mul - 0.35, x_out_mul, y_hout + 0.3, color=c_output)
ax.text(x_out_mul, y_hout - 0.3, "$\\mathbf{h}_t$", fontsize=15,
        fontweight="bold", color=c_output, ha="center", va="top")

# h_t continuation arrow (to next cell)
arr(x_out_mul + 0.4, y_hout, 20.5, y_hout, color=c_output, lw=2.5)
ax.text(20.8, y_hout, "$\\mathbf{h}_t$", fontsize=14,
        fontweight="bold", color=c_output, ha="left", va="center")

# ── 核心方程标注 ──────────────────────────────────────────────────
eqs = [
    ("$\\mathbf{f}_t = \\sigma(W_f[\\mathbf{h}_{t-1}, \\mathbf{x}_t]+\\mathbf{b}_f)$",
     c_forget),
    ("$\\mathbf{i}_t = \\sigma(W_i[\\mathbf{h}_{t-1}, \\mathbf{x}_t]+\\mathbf{b}_i)$",
     c_input),
    ("$\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t$",
     c_cell),
    ("$\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t)$",
     c_output),
]
for idx, (eq, color) in enumerate(eqs):
    ax.text(-0.5, -0.3 - idx * 0.75, eq, fontsize=12, fontweight="bold",
            color=color, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, alpha=0.85))

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_3_02_lstm_cell")
