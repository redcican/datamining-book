"""fig11_4_03_anomaly_detection.py
(a) 正常 vs 异常样本重建效果对比  (b) 重建误差分布"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5),
                                gridspec_kw={"width_ratios": [1.15, 1]})
fig.suptitle("图 11.4.3　自编码器异常检测示意",
             fontsize=22, fontweight="bold", y=0.98)

LW = 2.5

# ══════════════════════════════════════════════════════════════════
# (a) 重建效果对比
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)

def make_zero_like(size=14):
    img = np.zeros((size, size))
    cy, cx = size // 2, size // 2
    for i in range(size):
        for j in range(size):
            dist = ((i - cy) / (size * 0.35))**2 + ((j - cx) / (size * 0.25))**2
            if 0.5 < dist < 1.5:
                img[i, j] = 0.8 + 0.2 * np.random.rand()
    return img

def make_other_digit(size=14):
    img = np.zeros((size, size))
    img[2:4, 3:11] = 0.8 + 0.2 * np.random.rand(2, 8)
    for k in range(10):
        r = 3 + k
        c = 10 - k * 0.6
        if 0 <= r < size and 0 <= int(c) < size:
            img[int(r), int(c)] = 0.8 + 0.15 * np.random.rand()
            if int(c) + 1 < size:
                img[int(r), int(c) + 1] = 0.6 + 0.15 * np.random.rand()
    return img

def add_reconstruction_noise(img, noise_level):
    noisy = img + noise_level * np.random.randn(*img.shape)
    return np.clip(noisy, 0, 1)

ax1.set_axis_off()

sz = 14
normal_orig = make_zero_like(sz)
normal_recon = add_reconstruction_noise(normal_orig, 0.05)
anomaly_orig = make_other_digit(sz)
anomaly_recon = add_reconstruction_noise(anomaly_orig, 0.25)

# Larger images + layout that avoids textbox overlap
# 2 rows × 3 columns, images placed in left 60%, textboxes in right 40%
img_w, img_h = 0.17, 0.36
gap_x, gap_y = 0.02, 0.07
left_start = 0.01

col_labels = ["原始输入", "重建输出", "重建误差"]
row_labels = ["正常样本\n(数字 0)", "异常样本\n(数字 7)"]

for row in range(2):
    if row == 0:
        orig, recon = normal_orig, normal_recon
    else:
        orig, recon = anomaly_orig, anomaly_recon

    for img_type in range(3):
        if img_type == 0:
            data, cmap, vmin, vmax = orig, "gray", 0, 1
        elif img_type == 1:
            data, cmap, vmin, vmax = recon, "gray", 0, 1
        else:
            data = np.abs(orig - recon)
            cmap, vmin, vmax = "hot", 0, 0.5

        x_pos = left_start + img_type * (img_w + gap_x)
        y_pos = 0.82 - row * (img_h + gap_y) - img_h

        inax = ax1.inset_axes([x_pos, y_pos, img_w, img_h])
        inax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="nearest", aspect="equal")
        inax.set_xticks([])
        inax.set_yticks([])

        # Column labels (top row only)
        if row == 0:
            inax.set_title(col_labels[img_type], fontsize=14,
                           fontweight="bold", pad=6)

        # Row labels (first column only)
        if img_type == 0:
            color = COLORS["green"] if row == 0 else COLORS["red"]
            inax.set_ylabel(row_labels[row], fontsize=13, fontweight="bold",
                            color=color, labelpad=12)

        # Border color
        border_color = COLORS["green"] if row == 0 else COLORS["red"]
        for spine in inax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(2.5)

# Textboxes positioned to the right of images (no overlap)
tb_x = 0.78

ax1.text(tb_x, 0.72, "误差小\n高质量重建",
         fontsize=14, fontweight="bold", color=COLORS["green"],
         transform=ax1.transAxes, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=COLORS["green"], alpha=0.9))

ax1.text(tb_x, 0.28, "误差大\n重建失真严重",
         fontsize=14, fontweight="bold", color=COLORS["red"],
         transform=ax1.transAxes, ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                   ec=COLORS["red"], alpha=0.9))

ax1.annotate("", xy=(0.93, 0.72), xytext=(0.93, 0.28),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="<->", color=COLORS["gray"],
                             lw=2, mutation_scale=14))
ax1.text(0.96, 0.50, "阈值\n判定", fontsize=13, fontweight="bold",
         color=COLORS["gray"], transform=ax1.transAxes,
         ha="center", va="center")

ax1.set_title("(a) 重建效果对比", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 重建误差分布
# ══════════════════════════════════════════════════════════════════
np.random.seed(88)

n_samples = 2000
normal_errors = np.random.gamma(2.0, 0.008, n_samples)
normal_errors = np.clip(normal_errors, 0, 0.1)
anomaly_errors = np.random.gamma(3.0, 0.025, n_samples)
anomaly_errors = np.clip(anomaly_errors, 0.02, 0.3)

bins_all = np.linspace(0, 0.25, 60)

ax2.hist(normal_errors, bins=bins_all, density=True, alpha=0.6,
         color=COLORS["green"], edgecolor="white", linewidth=0.5,
         label="正常样本")
ax2.hist(anomaly_errors, bins=bins_all, density=True, alpha=0.6,
         color=COLORS["red"], edgecolor="white", linewidth=0.5,
         label="异常样本")

threshold = 0.065
ax2.axvline(x=threshold, color=COLORS["purple"], ls="--", lw=2.5,
            label=f"检测阈值 $\\tau$={threshold}")
ax2.fill_betweenx([0, 80], threshold, 0.25, color=COLORS["red"],
                  alpha=0.06)
ax2.fill_betweenx([0, 80], 0, threshold, color=COLORS["green"],
                  alpha=0.06)

ax2.text(threshold - 0.008, 65, "正常",
         fontsize=15, fontweight="bold", color=COLORS["green"],
         ha="right", va="center")
ax2.text(threshold + 0.008, 65, "异常",
         fontsize=15, fontweight="bold", color=COLORS["red"],
         ha="left", va="center")

ax2.text(0.18, 50,
         "判定规则：\n$\\|\\mathbf{x}-\\hat{\\mathbf{x}}\\|^2 > \\tau$\n→ 异常",
         fontsize=14, fontweight="bold", color=COLORS["purple"],
         ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                   ec=COLORS["purple"], alpha=0.9))

ax2.set_xlabel("重建误差 $\\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2$",
               fontsize=16)
ax2.set_ylabel("概率密度", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=13, loc="upper right")
ax2.set_xlim(0, 0.25)
ax2.set_ylim(0, 75)
ax2.grid(alpha=0.3)
ax2.set_title("(b) 重建误差分布", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_4_03_anomaly_detection")
