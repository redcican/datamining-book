"""fig11_7_02_domain_adaptation.py
(a) DANN 架构  (b) 特征空间可视化 (适应前/后)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig = plt.figure(figsize=(16, 8))
fig.suptitle("图 11.7.2　领域适应方法",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) DANN 架构
# ══════════════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.02, 0.06, 0.52, 0.84])
ax1.set_axis_off()
ax1.set_xlim(-1, 21)
ax1.set_ylim(-2, 14)
ax1.set_aspect("equal")

c_gray = COLORS["gray"]
c_blue = COLORS["blue"]
c_green = COLORS["green"]
c_orange = COLORS["orange"]
c_red = COLORS["red"]


def draw_box(ax, cx, cy, w, h, text, color, fontsize=15, text_color="white",
             lw=1.5):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=lw, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def arr(ax, x1, y1, x2, y2, color=c_gray, lw=2.0, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=16, linestyle=ls))


# --- Input box ---
draw_box(ax1, 1.0, 6.0, 2.2, 1.6, "输入 $x$", c_gray, fontsize=15)

# Arrow: input → feature extractor
arr(ax1, 2.1, 6.0, 4.3, 6.0, color=c_gray, lw=2)

# --- Feature extractor box (large) ---
draw_box(ax1, 6.0, 6.0, 3.2, 2.8, "特征提取器\n$\\phi$", c_blue,
         fontsize=15)

# === Top branch: task classifier ===
# Arrow: feature extractor → task classifier
arr(ax1, 7.6, 7.4, 10.8, 10.0, color=c_green, lw=2)

# Task classifier box
draw_box(ax1, 12.5, 10.5, 2.8, 1.6, "任务分类器", c_green, fontsize=14)

# Arrow: task classifier → class label
arr(ax1, 13.9, 10.5, 16.2, 10.5, color=c_green, lw=2)

# Class label box
draw_box(ax1, 17.8, 10.5, 2.4, 1.4, "类别 $y$", c_green, fontsize=14)

# Task loss label
ax1.text(12.5, 12.5,
         "$\\mathcal{L}_{task}$",
         fontsize=16, fontweight="bold", color=c_green, ha="center",
         va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=c_green, alpha=0.9))

# === Bottom branch: domain classifier ===
# Arrow: feature extractor → GRL
arr(ax1, 7.6, 4.6, 9.2, 2.5, color=c_orange, lw=2)

# GRL circle
grl_circle = Circle((10.2, 2.0), 0.9, facecolor=c_orange, edgecolor="black",
                     linewidth=1.5, alpha=0.9, zorder=5)
ax1.add_patch(grl_circle)
ax1.text(10.2, 2.0, "GRL", ha="center", va="center",
         fontsize=14, fontweight="bold", color="white", zorder=6)
ax1.text(10.2, 0.5, "梯度反转", ha="center", va="top",
         fontsize=13, fontweight="bold", color=c_orange)

# Arrow: GRL → domain classifier
arr(ax1, 11.1, 2.0, 12.6, 2.0, color=c_red, lw=2)

# Domain classifier box
draw_box(ax1, 14.5, 2.0, 2.8, 1.6, "域分类器", c_red, fontsize=14)

# Arrow: domain classifier → domain label
arr(ax1, 15.9, 2.0, 16.8, 2.0, color=c_red, lw=2)

# Domain label box
draw_box(ax1, 18.2, 2.0, 2.4, 1.4, "域标签 $d$", c_red, fontsize=13)

# Domain loss label
ax1.text(14.5, -0.2,
         "$\\mathcal{L}_d$",
         fontsize=16, fontweight="bold", color=c_red, ha="center",
         va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=c_red, alpha=0.9))

# === Gradient flow arrows ===
# Normal gradient (green): task loss → feature extractor
arr(ax1, 10.5, 12.5, 7.5, 8.5, color=c_green, lw=2, ls="--")
ax1.text(8.2, 11.2, "正常梯度", fontsize=13, fontweight="bold",
         color=c_green, ha="center", rotation=28)

# Reversed gradient (red dashed): domain loss → through GRL → feature extractor
arr(ax1, 9.3, 2.0, 7.6, 4.0, color=c_red, lw=2, ls="--")
ax1.text(7.4, 2.3, "$\\times(-1)$", fontsize=14, fontweight="bold",
         color=c_red, ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=c_red, alpha=0.8))
ax1.text(7.2, 1.2, "反转梯度", fontsize=13, fontweight="bold",
         color=c_red, ha="center")

ax1.set_title("(a) DANN 架构", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 特征空间可视化
# ══════════════════════════════════════════════════════════════════
fig.text(0.78, 0.94, "(b) 特征空间可视化",
         fontsize=17, fontweight="bold", ha="center")

ax2_top = fig.add_axes([0.60, 0.52, 0.36, 0.38])
ax2_bot = fig.add_axes([0.60, 0.06, 0.36, 0.38])

np.random.seed(42)
n = 40

# --- (b-top) Before adaptation ---
# Source domain (blue): 2 classes on the left side
src_c0_x = np.random.randn(n) * 0.5 - 2.0
src_c0_y = np.random.randn(n) * 0.5 + 2.0
src_c1_x = np.random.randn(n) * 0.5 - 2.0
src_c1_y = np.random.randn(n) * 0.5 - 2.0

# Target domain (red): 2 classes on the right side
tgt_c0_x = np.random.randn(n) * 0.5 + 2.0
tgt_c0_y = np.random.randn(n) * 0.5 + 2.0
tgt_c1_x = np.random.randn(n) * 0.5 + 2.0
tgt_c1_y = np.random.randn(n) * 0.5 - 2.0

ax2_top.scatter(src_c0_x, src_c0_y, c=c_blue, marker="o", s=30,
                alpha=0.7, edgecolors="none", label="源域")
ax2_top.scatter(src_c1_x, src_c1_y, c=c_blue, marker="^", s=30,
                alpha=0.7, edgecolors="none")
ax2_top.scatter(tgt_c0_x, tgt_c0_y, c=c_red, marker="o", s=30,
                alpha=0.7, edgecolors="none", label="目标域")
ax2_top.scatter(tgt_c1_x, tgt_c1_y, c=c_red, marker="^", s=30,
                alpha=0.7, edgecolors="none")

# Class annotations
ax2_top.text(-2.0, 3.2, "类别 0", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")
ax2_top.text(-2.0, -3.2, "类别 1", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")
ax2_top.text(2.0, 3.2, "类别 0", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")
ax2_top.text(2.0, -3.2, "类别 1", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")

# Domain separation line
ax2_top.axvline(x=0, color=c_gray, ls="--", lw=1.2, alpha=0.5)
ax2_top.text(0.1, 3.5, "域间隔", fontsize=11, color=c_gray, alpha=0.7)

ax2_top.set_xlim(-4, 4)
ax2_top.set_ylim(-4, 4)
ax2_top.set_xlabel("$z_1$", fontsize=14)
ax2_top.set_ylabel("$z_2$", fontsize=14)
ax2_top.legend(fontsize=13, loc="lower left", framealpha=0.9)
ax2_top.grid(alpha=0.3)
ax2_top.set_title("适应前", fontsize=15, fontweight="bold")
ax2_top.tick_params(labelsize=12)

# --- (b-bottom) After adaptation ---
np.random.seed(123)

# Class 0: both domains mixed, centered at (0, 2)
mix_c0_src_x = np.random.randn(n) * 0.6 + 0.0
mix_c0_src_y = np.random.randn(n) * 0.6 + 2.0
mix_c0_tgt_x = np.random.randn(n) * 0.6 + 0.0
mix_c0_tgt_y = np.random.randn(n) * 0.6 + 2.0

# Class 1: both domains mixed, centered at (0, -2)
mix_c1_src_x = np.random.randn(n) * 0.6 + 0.0
mix_c1_src_y = np.random.randn(n) * 0.6 - 2.0
mix_c1_tgt_x = np.random.randn(n) * 0.6 + 0.0
mix_c1_tgt_y = np.random.randn(n) * 0.6 - 2.0

ax2_bot.scatter(mix_c0_src_x, mix_c0_src_y, c=c_blue, marker="o", s=30,
                alpha=0.7, edgecolors="none", label="源域")
ax2_bot.scatter(mix_c1_src_x, mix_c1_src_y, c=c_blue, marker="^", s=30,
                alpha=0.7, edgecolors="none")
ax2_bot.scatter(mix_c0_tgt_x, mix_c0_tgt_y, c=c_red, marker="o", s=30,
                alpha=0.7, edgecolors="none", label="目标域")
ax2_bot.scatter(mix_c1_tgt_x, mix_c1_tgt_y, c=c_red, marker="^", s=30,
                alpha=0.7, edgecolors="none")

# Class annotations
ax2_bot.text(0.0, 3.5, "类别 0", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")
ax2_bot.text(0.0, -3.5, "类别 1", fontsize=12, fontweight="bold",
             color=c_gray, ha="center")

# Class separation line
ax2_bot.axhline(y=0, color=c_green, ls="--", lw=1.5, alpha=0.6)
ax2_bot.text(3.0, 0.3, "类别边界", fontsize=11, fontweight="bold",
             color=c_green, alpha=0.8)

# Annotation: domains aligned
ax2_bot.text(-3.0, 3.2,
             "域分布\n已对齐",
             fontsize=13, fontweight="bold", color=COLORS["purple"],
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=COLORS["purple"], alpha=0.9))

ax2_bot.set_xlim(-4, 4)
ax2_bot.set_ylim(-4, 4)
ax2_bot.set_xlabel("$z_1$", fontsize=14)
ax2_bot.set_ylabel("$z_2$", fontsize=14)
ax2_bot.legend(fontsize=13, loc="lower left", framealpha=0.9)
ax2_bot.grid(alpha=0.3)
ax2_bot.set_title("适应后", fontsize=15, fontweight="bold")
ax2_bot.tick_params(labelsize=12)

# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig11_7_02_domain_adaptation")
