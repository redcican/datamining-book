"""fig11_5_01_gan_training_process.py
(a) GAN 架构图  (b) 训练动态——分布逼近过程 (4 阶段)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig = plt.figure(figsize=(16, 8.5))
fig.suptitle("图 11.5.1　GAN 对抗训练过程",
             fontsize=22, fontweight="bold", y=0.98)

# (a) enlarged to ~52% width
ax_arch = fig.add_axes([0.02, 0.06, 0.52, 0.84])

# (b) 4 panels shifted right
axes_dyn = [
    fig.add_axes([0.58, 0.53, 0.19, 0.36]),
    fig.add_axes([0.79, 0.53, 0.19, 0.36]),
    fig.add_axes([0.58, 0.08, 0.19, 0.36]),
    fig.add_axes([0.79, 0.08, 0.19, 0.36]),
]

# ══════════════════════════════════════════════════════════════════
# (a) GAN 架构图
# ══════════════════════════════════════════════════════════════════
ax = ax_arch
ax.set_axis_off()
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 13)
ax.set_aspect("equal")

c_gen = COLORS["green"]
c_dis = COLORS["red"]
c_noise = COLORS["purple"]
c_data = COLORS["blue"]
c_line = COLORS["gray"]
LW = 2.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=16, text_color="white"):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=1.8, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def arr(ax, x1, y1, x2, y2, color=c_line, lw=LW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=16))

# Noise z
draw_box(ax, 2.0, 6.0, 2.0, 1.5, "$\\mathbf{z}$", c_noise, fontsize=20)
ax.text(2.0, 4.0, "随机噪声\n$\\sim\\mathcal{N}(0,I)$", fontsize=14,
        fontweight="bold", color=c_noise, ha="center")

# Generator
draw_box(ax, 7.0, 6.0, 3.0, 2.0, "生成器 $G$", c_gen, fontsize=17)

# Fake sample
draw_box(ax, 12.0, 6.0, 2.2, 1.5, "$G(\\mathbf{z})$", c_gen, fontsize=16)
ax.text(12.0, 4.0, "假样本", fontsize=15, fontweight="bold",
        color=c_gen, ha="center")

# Real data
draw_box(ax, 12.0, 10.5, 2.2, 1.5, "$\\mathbf{x}$", c_data, fontsize=18)
ax.text(12.0, 12.5, "真实数据", fontsize=15, fontweight="bold",
        color=c_data, ha="center")

# Discriminator
draw_box(ax, 17.0, 8.0, 3.0, 2.5, "判别器 $D$", c_dis, fontsize=17)

# Output
ax.text(17.0, 4.5, "真/假?", fontsize=17, fontweight="bold",
        color=c_dis, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_dis, alpha=0.9))

# Arrows
arr(ax, 3.0, 6.0, 5.5, 6.0, color=c_noise, lw=2.5)
arr(ax, 8.5, 6.0, 10.9, 6.0, color=c_gen, lw=2.5)
arr(ax, 13.1, 6.5, 15.5, 7.5, color=c_gen, lw=2.5)
arr(ax, 13.1, 10.0, 15.5, 8.5, color=c_data, lw=2.5)
arr(ax, 17.0, 6.75, 17.0, 5.3, color=c_dis, lw=2.5)

# Adversarial feedback
ax.annotate("", xy=(7.0, 2.0), xytext=(17.0, 2.0),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"],
                            lw=2, ls="--", mutation_scale=14))
ax.text(12.0, 1.3, "对抗反馈  $\\nabla_G$", fontsize=15, fontweight="bold",
        color=COLORS["orange"], ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                  ec=COLORS["orange"], alpha=0.9))

# Loss equation
ax.text(10.0, -0.3,
        "$V = \\mathbb{E}[\\log D(\\mathbf{x})] + \\mathbb{E}[\\log(1-D(G(\\mathbf{z})))]$",
        fontsize=14, fontweight="bold", color=c_line, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_line, alpha=0.9))

ax.set_title("(a) GAN 架构", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 训练动态 — 4 阶段
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)
x = np.linspace(-4, 4, 300)

p_data = 0.5 * np.exp(-(x - 1)**2 / 0.5) + 0.5 * np.exp(-(x + 1)**2 / 0.5)
p_data /= np.trapezoid(p_data, x)

stage_titles = ["初始阶段", "早期训练", "中期训练", "收敛"]
stage_labels = ["(i)", "(ii)", "(iii)", "(iv)"]

p_g_stages = []
pg1 = 0.15 * np.ones_like(x) + 0.1 * np.random.rand(len(x))
pg1 /= np.trapezoid(pg1, x)
p_g_stages.append(pg1)

pg2 = 0.8 * np.exp(-(x - 1.5)**2 / 1.0) + 0.2 * np.exp(-(x + 0.5)**2 / 2.0)
pg2 /= np.trapezoid(pg2, x)
p_g_stages.append(pg2)

pg3 = 0.45 * np.exp(-(x - 0.8)**2 / 0.6) + 0.55 * np.exp(-(x + 1.2)**2 / 0.5)
pg3 /= np.trapezoid(pg3, x)
p_g_stages.append(pg3)

pg4 = p_data + 0.005 * np.random.randn(len(x))
pg4 = np.clip(pg4, 0, None)
pg4 /= np.trapezoid(pg4, x)
p_g_stages.append(pg4)

D_stages = []
for pg in p_g_stages:
    D_stages.append(p_data / (p_data + pg + 1e-10))

for idx, (ax_d, pg, d_opt, title, label) in enumerate(
        zip(axes_dyn, p_g_stages, D_stages, stage_titles, stage_labels)):
    ax_d.fill_between(x, p_data, alpha=0.15, color=COLORS["blue"])
    ax_d.plot(x, p_data, color=COLORS["blue"], lw=2, label="$p_{\\text{data}}$")
    ax_d.fill_between(x, pg, alpha=0.15, color=COLORS["red"])
    ax_d.plot(x, pg, color=COLORS["red"], lw=2, ls="--", label="$p_g$")

    ax_d2 = ax_d.twinx()
    ax_d2.plot(x, d_opt, color=COLORS["green"], lw=1.8, ls=":", alpha=0.8)
    ax_d2.set_ylim(0, 1.1)
    ax_d2.set_yticks([0, 0.5, 1.0])
    ax_d2.tick_params(labelsize=11, colors=COLORS["green"])
    if idx in [1, 3]:
        ax_d2.set_ylabel("$D^*(x)$", fontsize=13, color=COLORS["green"])
    else:
        ax_d2.set_yticklabels([])

    ax_d.set_xlim(-4, 4)
    ax_d.set_ylim(0, max(p_data) * 1.3)
    ax_d.set_title(f"{label} {title}", fontsize=14, fontweight="bold", pad=5)
    ax_d.tick_params(labelsize=11)
    ax_d.grid(alpha=0.2)

    if idx == 0:
        ax_d.legend(fontsize=11, loc="upper right")
    if idx >= 2:
        ax_d.set_xlabel("$x$", fontsize=14)

fig.text(0.78, 0.94, "(b) 训练动态", fontsize=17, fontweight="bold",
         ha="center", va="center")

# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig11_5_01_gan_training_process")
