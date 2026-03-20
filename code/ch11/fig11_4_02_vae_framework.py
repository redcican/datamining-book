"""fig11_4_02_vae_framework.py
(a) VAE 架构——重参数化技巧  (b) AE vs VAE 潜在空间对比"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig = plt.figure(figsize=(16, 8))
fig.suptitle("图 11.4.2　变分自编码器框架与潜在空间对比",
             fontsize=22, fontweight="bold", y=0.98)

# Layout: (a) left half, (b) right half with label + two sub-panels
ax_arch = fig.add_axes([0.02, 0.06, 0.50, 0.82])
ax_ae   = fig.add_axes([0.56, 0.06, 0.19, 0.78])
ax_vae  = fig.add_axes([0.79, 0.06, 0.19, 0.78])
# (b) label horizontally centered above the two sub-panels
fig.text(0.755, 0.88, "(b) 潜在空间对比", fontsize=17, fontweight="bold",
         ha="center", va="center")

# ══════════════════════════════════════════════════════════════════
# (a) VAE 架构
# ══════════════════════════════════════════════════════════════════
ax = ax_arch
ax.set_axis_off()
ax.set_xlim(-1, 23)
ax.set_ylim(-2, 13)
ax.set_aspect("equal")

c_enc = COLORS["blue"]
c_dec = COLORS["red"]
c_mu = COLORS["green"]
c_sigma = COLORS["orange"]
c_z = COLORS["purple"]
c_eps = COLORS["teal"]
c_line = COLORS["gray"]

LW = 2.0

def draw_box(ax, cx, cy, w, h, text, color, fontsize=16, text_color="white"):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)

def draw_circle(ax, cx, cy, r, text, color, fontsize=16, text_color="white"):
    circ = Circle((cx, cy), r, facecolor=color, edgecolor="black",
                  linewidth=1.8, alpha=0.9, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=6)

def arr(ax, x1, y1, x2, y2, color=c_line, lw=LW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14))

# -- Input layer --
draw_box(ax, 1.5, 5.5, 1.2, 5.0, "$\\mathbf{x}$", c_enc, fontsize=18)
ax.text(1.5, 0.3, "输入", fontsize=14, fontweight="bold",
        color=c_enc, ha="center")

# -- Encoder --
draw_box(ax, 4.5, 5.5, 1.2, 4.0, "", c_enc, fontsize=14)
ax.text(4.5, 5.5, "编码器\n$q_\\phi$", fontsize=13, fontweight="bold",
        color="white", ha="center", va="center")
arr(ax, 2.2, 5.5, 3.9, 5.5, color=c_line, lw=2)

# -- μ branch --
draw_box(ax, 8.0, 8.0, 1.8, 1.2, "$\\boldsymbol{\\mu}$", c_mu, fontsize=18)
ax.text(8.0, 9.5, "均值向量", fontsize=13, fontweight="bold",
        color=c_mu, ha="center")

# -- σ² branch --
draw_box(ax, 8.0, 3.0, 1.8, 1.2, "$\\boldsymbol{\\sigma}^2$", c_sigma, fontsize=16)
ax.text(8.0, 1.5, "方差向量", fontsize=13, fontweight="bold",
        color=c_sigma, ha="center")

arr(ax, 5.1, 6.5, 7.1, 8.0, color=c_mu, lw=2)
arr(ax, 5.1, 4.5, 7.1, 3.0, color=c_sigma, lw=2)

# -- ε --
draw_circle(ax, 11.5, 0.5, 0.5, "$\\boldsymbol{\\epsilon}$", c_eps, fontsize=16)
ax.text(11.5, -0.5, "$\\sim \\mathcal{N}(0,I)$", fontsize=13,
        fontweight="bold", color=c_eps, ha="center")

# -- Reparameterization --
draw_circle(ax, 11.5, 3.0, 0.4, "⊙", "#4a4a4a", fontsize=18)
arr(ax, 8.9, 3.0, 11.1, 3.0, color=c_sigma, lw=2)
arr(ax, 11.5, 1.0, 11.5, 2.6, color=c_eps, lw=2)

draw_circle(ax, 11.5, 5.5, 0.4, "+", "#4a4a4a", fontsize=20)
arr(ax, 8.9, 8.0, 11.2, 5.9, color=c_mu, lw=2)
arr(ax, 11.5, 3.4, 11.5, 5.1, color=c_line, lw=2)

# -- z --
draw_box(ax, 14.0, 5.5, 1.2, 1.6, "$\\mathbf{z}$", c_z, fontsize=18)
arr(ax, 11.9, 5.5, 13.4, 5.5, color=c_z, lw=2.5)

ax.text(12.8, 7.5,
        "$\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\epsilon}$",
        fontsize=16, fontweight="bold", color=c_z, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c_z, alpha=0.9))
ax.annotate("", xy=(12.8, 6.4), xytext=(12.8, 7.2),
            arrowprops=dict(arrowstyle="-|>", color=c_z, lw=1.5,
                            mutation_scale=12))

# -- Decoder --
draw_box(ax, 17.0, 5.5, 1.2, 4.0, "", c_dec, fontsize=14)
ax.text(17.0, 5.5, "解码器\n$p_\\theta$", fontsize=13, fontweight="bold",
        color="white", ha="center", va="center")
arr(ax, 14.6, 5.5, 16.4, 5.5, color=c_line, lw=2)

# -- Output --
draw_box(ax, 20.0, 5.5, 1.2, 5.0, "$\\hat{\\mathbf{x}}$", c_dec, fontsize=18)
arr(ax, 17.6, 5.5, 19.4, 5.5, color=c_line, lw=2)
ax.text(20.0, 0.3, "重建", fontsize=14, fontweight="bold",
        color=c_dec, ha="center")

# -- Loss --
ax.text(10.7, 12.0,
        "$\\mathcal{L}_{\\mathrm{VAE}} = "
        "\\|\\mathbf{x}-\\hat{\\mathbf{x}}\\|^2"
        "+ D_{\\mathrm{KL}}(q_\\phi \\,\\|\\, p)$",
        fontsize=15, fontweight="bold", color=COLORS["gray"], ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=COLORS["gray"], alpha=0.9))
ax.text(8.5, 10.8, "重建损失", fontsize=12, fontweight="bold",
        color=c_dec, ha="center")
ax.text(13.0, 10.8, "KL 正则", fontsize=12, fontweight="bold",
        color=c_mu, ha="center")

ax.set_title("(a) VAE 架构与重参数化技巧", fontsize=17,
              fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 潜在空间对比: AE vs VAE
# ══════════════════════════════════════════════════════════════════
np.random.seed(123)

n_pts = 60
class_colors = [COLORS["blue"], COLORS["red"], COLORS["green"],
                COLORS["orange"], COLORS["purple"]]

# --- AE latent space ---
ae_centers = [(-2.5, 3.0), (2.8, 2.5), (-0.3, -2.5),
              (-3.0, -1.0), (3.0, -2.0)]
ae_spreads = [0.3, 0.25, 0.35, 0.28, 0.32]

for i, ((cx, cy_c), spread) in enumerate(zip(ae_centers, ae_spreads)):
    x = np.random.randn(n_pts) * spread + cx
    y = np.random.randn(n_pts) * spread + cy_c
    ax_ae.scatter(x, y, c=class_colors[i], alpha=0.5, s=15, edgecolors="none")

ax_ae.text(0.5, 0.5, "空洞", fontsize=13, fontweight="bold",
           color=COLORS["gray"], ha="center", va="center", alpha=0.7,
           style="italic")
ax_ae.text(-1.0, 1.5, "空洞", fontsize=13, fontweight="bold",
           color=COLORS["gray"], ha="center", va="center", alpha=0.7,
           style="italic")

ax_ae.set_xlabel("$z_1$", fontsize=15)
ax_ae.set_ylabel("$z_2$", fontsize=15)
ax_ae.tick_params(labelsize=12)
ax_ae.set_xlim(-4.5, 4.5)
ax_ae.set_ylim(-4.5, 4.5)
ax_ae.grid(alpha=0.3)
ax_ae.set_title("基本 AE", fontsize=16, fontweight="bold", pad=10)
ax_ae.text(0.0, -4.0, "不连续、有空洞", fontsize=12, fontweight="bold",
           color=COLORS["red"], ha="center",
           bbox=dict(boxstyle="round,pad=0.2", fc="white",
                     ec=COLORS["red"], alpha=0.8))

# --- VAE latent space ---
vae_centers = [(-1.3, 1.5), (1.5, 1.2), (0.0, -1.5),
               (-1.5, -0.5), (1.5, -1.0)]
vae_spreads = [0.7, 0.65, 0.7, 0.6, 0.65]

for i, ((cx, cy_c), spread) in enumerate(zip(vae_centers, vae_spreads)):
    x = np.random.randn(n_pts) * spread + cx
    y = np.random.randn(n_pts) * spread + cy_c
    ax_vae.scatter(x, y, c=class_colors[i], alpha=0.5, s=15, edgecolors="none")

theta = np.linspace(0, 2*np.pi, 100)
for r in [1.0, 2.0]:
    ax_vae.plot(r * np.cos(theta), r * np.sin(theta),
                color=COLORS["gray"], ls="--", lw=1.0, alpha=0.5)

ax_vae.set_xlabel("$z_1$", fontsize=15)
ax_vae.set_ylabel("$z_2$", fontsize=15)
ax_vae.tick_params(labelsize=12)
ax_vae.set_xlim(-4.5, 4.5)
ax_vae.set_ylim(-4.5, 4.5)
ax_vae.grid(alpha=0.3)
ax_vae.set_title("VAE", fontsize=16, fontweight="bold", pad=10)
ax_vae.text(0.0, -4.0, "光滑连续、可插值", fontsize=12, fontweight="bold",
            color=COLORS["green"], ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=COLORS["green"], alpha=0.8))

# (b) label already placed at top in layout section above

# ── 保存 ──────────────────────────────────────────────────────────
save_fig(fig, __file__, "fig11_4_02_vae_framework")
