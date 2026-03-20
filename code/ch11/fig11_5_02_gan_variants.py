"""fig11_5_02_gan_variants.py
(a) DCGAN 生成器架构  (b) GAN vs WGAN 训练损失对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.5, 1]})
fig.suptitle("图 11.5.2　GAN 变体架构与训练对比",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) DCGAN 生成器架构
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
ax1.set_xlim(-1, 25)
ax1.set_ylim(-2, 12)
ax1.set_aspect("equal")

c_noise = COLORS["purple"]
c_conv = COLORS["blue"]
c_bn = COLORS["teal"]
c_out = COLORS["orange"]
c_line = COLORS["gray"]

def draw_box(cx, cy, w, h, text, color, fontsize=15, text_color="white"):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
    ax1.add_patch(rect)
    ax1.text(cx, cy, text, ha="center", va="center",
             fontsize=fontsize, fontweight="bold", color=text_color)

def draw_3d_block(cx, cy, w, h, depth, color, label, label_below):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85)
    ax1.add_patch(rect)
    d = depth * 0.3
    ax1.fill([cx - w/2, cx - w/2 + d, cx + w/2 + d, cx + w/2],
             [cy + h/2, cy + h/2 + d, cy + h/2 + d, cy + h/2],
             color=color, alpha=0.5, edgecolor="black", linewidth=0.8)
    ax1.fill([cx + w/2, cx + w/2 + d, cx + w/2 + d, cx + w/2],
             [cy - h/2, cy - h/2 + d, cy + h/2 + d, cy + h/2],
             color=color, alpha=0.65, edgecolor="black", linewidth=0.8)
    ax1.text(cx, cy, label, ha="center", va="center",
             fontsize=13, fontweight="bold", color="white")
    ax1.text(cx, cy - h/2 - 0.6, label_below, ha="center", va="top",
             fontsize=13, fontweight="bold", color=color)

def arr(x1, y1, x2, y2, color=c_line, lw=2):
    ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=color,
                                 lw=lw, mutation_scale=14))

# Noise input
draw_box(1.0, 5.5, 1.5, 1.5, "$\\mathbf{z}$", c_noise, fontsize=18)
ax1.text(1.0, 3.2, "$100$", fontsize=14, fontweight="bold",
         color=c_noise, ha="center")

arr(1.8, 5.5, 3.2, 5.5, color=c_line, lw=2)

# Project & Reshape
draw_box(4.5, 5.5, 2.2, 1.2, "投影+重塑", "#4a4a4a", fontsize=13)
arr(5.6, 5.5, 6.8, 5.5, color=c_line, lw=2)

# Feature map blocks
blocks = [
    (8.0,  5.5, 1.0, 2.0, 3, c_conv, "512", "$4{\\times}4$"),
    (12.0, 5.5, 1.5, 3.0, 2.5, c_conv, "256", "$8{\\times}8$"),
    (16.5, 5.5, 2.0, 4.0, 2, c_conv, "128", "$16{\\times}16$"),
    (21.0, 5.5, 2.5, 5.0, 1.5, c_out,  "1",  "$28{\\times}28$"),
]

for i, (cx, cy, w, h, d, color, ch, sz) in enumerate(blocks):
    draw_3d_block(cx, cy, w, h, d, color, ch, sz)
    if i < len(blocks) - 1:
        next_cx = blocks[i+1][0]
        next_w = blocks[i+1][2]
        arr(cx + w/2 + d*0.3 + 0.1, cy + 0.3,
            next_cx - next_w/2 - 0.1, cy + 0.3,
            color=c_line, lw=1.8)
        mid_x = (cx + w/2 + d*0.3 + next_cx - next_w/2) / 2
        label = "ConvT\nBN\nReLU" if i < len(blocks) - 2 else "ConvT\nTanh"
        color_label = c_bn if i < len(blocks) - 2 else c_out
        ax1.text(mid_x, cy + 2.5 + (0 if i > 0 else 0.5), label,
                 fontsize=11, fontweight="bold",
                 color=color_label, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec=color_label, alpha=0.8))

# Architecture flow label
ax1.text(12.0, -1.0,
         "$\\mathbf{z} \\to 4{\\times}4{\\times}512 \\to "
         "8{\\times}8{\\times}256 \\to "
         "16{\\times}16{\\times}128 \\to "
         "28{\\times}28{\\times}1$",
         fontsize=13, fontweight="bold", color=c_line, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0",
                   ec=c_line, alpha=0.9))

ax1.set_title("(a) DCGAN 生成器架构", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) GAN vs WGAN 训练损失对比
# ══════════════════════════════════════════════════════════════════
np.random.seed(123)
epochs = np.arange(1, 51)

fid_gan = 120 - 60 * (1 - np.exp(-epochs / 20)) + 15 * np.sin(epochs * 0.4) + \
          5 * np.random.randn(50)
fid_gan = np.clip(fid_gan, 30, 140)
fid_wgan = 120 * np.exp(-epochs / 10) + 15 + 3 * np.random.randn(50)
fid_wgan = np.clip(fid_wgan, 10, 130)

ax2.plot(epochs, fid_gan, color=COLORS["red"], lw=2.5, alpha=0.9,
         label="原始 GAN", ls="--")
ax2.plot(epochs, fid_wgan, color=COLORS["blue"], lw=2.5, alpha=0.9,
         label="WGAN-GP")

ax2.axhspan(0, 30, color=COLORS["green"], alpha=0.06)
ax2.text(45, 22, "高质量", fontsize=14, fontweight="bold",
         color=COLORS["green"], ha="center", alpha=0.8)

ax2.set_xlabel("训练轮次 (Epoch)", fontsize=16)
ax2.set_ylabel("FID 分数 (越低越好)", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, loc="upper right")
ax2.set_xlim(1, 50)
ax2.set_ylim(0, 150)
ax2.grid(alpha=0.3)

ax2.annotate("WGAN 稳定下降", xy=(30, fid_wgan[29]),
             xytext=(15, 100),
             fontsize=14, fontweight="bold", color=COLORS["blue"],
             arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=COLORS["blue"], alpha=0.9))

ax2.annotate("GAN 震荡不收敛", xy=(35, fid_gan[34]),
             xytext=(38, 120),
             fontsize=14, fontweight="bold", color=COLORS["red"],
             arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=COLORS["red"], alpha=0.9))

ax2.set_title("(b) 生成质量对比 (FID)", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_5_02_gan_variants")
