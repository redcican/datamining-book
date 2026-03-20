"""fig11_3_01_rnn_unrolling.py
循环神经网络的结构与时间展开：(a) 紧凑形式  (b) 展开形式"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1, 2.2]})
fig.suptitle("图 11.3.1　循环神经网络的结构与时间展开",
             fontsize=22, fontweight="bold", y=0.98)

# ── 颜色 ─────────────────────────────────────────────────────────
c_input = COLORS["green"]
c_hidden = COLORS["blue"]
c_output = COLORS["orange"]
c_weight = COLORS["red"]
c_arrow = COLORS["gray"]

BOX_W, BOX_H = 1.8, 1.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=14, alpha=0.85):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=alpha)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")

def arrow(ax, x1, y1, x2, y2, color="black", lw=2, label=None,
          label_offset=(0, 0), label_fontsize=12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fontsize, fontweight="bold",
                color=c_weight, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=c_weight, alpha=0.9))

# ══════════════════════════════════════════════════════════════════
# (a) 紧凑形式 (Folded RNN)
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-2.5, 3.5)
ax1.set_ylim(-1.5, 8.5)
ax1.set_aspect("equal")

# Input
cx_a = 0.5
y_input = 0.5
draw_box(ax1, cx_a, y_input, BOX_W, BOX_H, "$\\mathbf{x}_t$", c_input)

# Hidden
y_hidden = 3.5
draw_box(ax1, cx_a, y_hidden, BOX_W, BOX_H, "$\\mathbf{h}_t$", c_hidden)

# Output
y_output = 6.5
draw_box(ax1, cx_a, y_output, BOX_W, BOX_H, "$\\mathbf{y}_t$", c_output)

# Arrows
arrow(ax1, cx_a, y_input + BOX_H/2 + 0.05, cx_a, y_hidden - BOX_H/2 - 0.05,
      color=c_arrow, label="$W_{xh}$", label_offset=(1.0, 0))
arrow(ax1, cx_a, y_hidden + BOX_H/2 + 0.05, cx_a, y_output - BOX_H/2 - 0.05,
      color=c_arrow, label="$W_{hy}$", label_offset=(1.0, 0))

# Self-loop for W_hh
loop_x = cx_a - BOX_W/2 - 0.4
ax1.annotate(
    "", xy=(cx_a - BOX_W/2, y_hidden + 0.15),
    xytext=(cx_a - BOX_W/2, y_hidden - 0.15),
    arrowprops=dict(arrowstyle="-|>", color=c_weight, lw=2.5,
                    connectionstyle="arc3,rad=-1.8",
                    mutation_scale=16))
ax1.text(loop_x - 0.6, y_hidden, "$W_{hh}$", fontsize=12, fontweight="bold",
         color=c_weight, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_weight, alpha=0.9))

ax1.set_title("(a) 紧凑形式", fontsize=17, fontweight="bold", pad=20)

# ══════════════════════════════════════════════════════════════════
# (b) 展开形式 (Unrolled RNN, T=4)
# ══════════════════════════════════════════════════════════════════
ax2.set_axis_off()
ax2.set_xlim(-2.5, 16)
ax2.set_ylim(-1.5, 8.5)
ax2.set_aspect("equal")

T = 4
gap_x = 3.8
bw, bh = 1.5, 0.85  # smaller boxes for unrolled

for t in range(T):
    cx = t * gap_x + 1.0

    # Input box
    y_in = 0.5
    draw_box(ax2, cx, y_in, bw, bh,
             f"$\\mathbf{{x}}_{{{t+1}}}$", c_input, fontsize=13)

    # Hidden box
    y_hid = 3.5
    draw_box(ax2, cx, y_hid, bw, bh,
             f"$\\mathbf{{h}}_{{{t+1}}}$", c_hidden, fontsize=13)

    # Output box
    y_out = 6.5
    draw_box(ax2, cx, y_out, bw, bh,
             f"$\\mathbf{{y}}_{{{t+1}}}$", c_output, fontsize=13)

    # Input → Hidden arrow
    arrow(ax2, cx, y_in + bh/2 + 0.05, cx, y_hid - bh/2 - 0.05,
          color=c_arrow)

    # Hidden → Output arrow
    arrow(ax2, cx, y_hid + bh/2 + 0.05, cx, y_out - bh/2 - 0.05,
          color=c_arrow)

    # Hidden → next Hidden arrow (horizontal)
    if t < T - 1:
        nx = (t + 1) * gap_x + 1.0
        arrow(ax2, cx + bw/2 + 0.05, y_hid, nx - bw/2 - 0.05, y_hid,
              color=c_weight, lw=2.5)

# h_0 arrow into first hidden
h0_x = 1.0 - gap_x * 0.4
ax2.text(h0_x - 0.5, 3.5, "$\\mathbf{h}_0$", fontsize=13, fontweight="bold",
         color=c_hidden, ha="center", va="center")
arrow(ax2, h0_x, 3.5, 1.0 - bw/2 - 0.05, 3.5, color=c_weight, lw=2.5)

# Weight labels (shared)
# W_xh label
ax2.text(1.0 + 0.55, 2.0, "$W_{xh}$", fontsize=12, fontweight="bold",
         color=c_weight, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_weight, alpha=0.9))

# W_hy label
ax2.text(1.0 + 0.55, 5.0, "$W_{hy}$", fontsize=12, fontweight="bold",
         color=c_weight, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_weight, alpha=0.9))

# W_hh label on horizontal arrows
mid_hh_x = (1.0 + 1.0 + gap_x) / 2
ax2.text(mid_hh_x, 3.5 + 0.6, "$W_{hh}$", fontsize=12, fontweight="bold",
         color=c_weight, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_weight, alpha=0.9))

# "参数共享" annotation bracket
ax2.annotate(
    "所有时间步共享 $W_{xh}, W_{hh}, W_{hy}$",
    xy=(7.5, -0.8), fontsize=13, fontweight="bold",
    color=COLORS["gray"], ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0",
              ec=COLORS["gray"], alpha=0.9))

# Unrolling arrow between (a) and (b)
ax2.text(-1.8, 3.5, "展开\n$\\Longrightarrow$", fontsize=16,
         fontweight="bold", color=COLORS["gray"],
         ha="center", va="center")

ax2.set_title("(b) 时间展开形式 ($T=4$)", fontsize=17,
              fontweight="bold", pad=20)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_3_01_rnn_unrolling")
