"""fig11_5_03_generation_results.py
(a) 生成样本网格 (合成数字图像)  (b) 训练损失曲线 (G/D loss)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                gridspec_kw={"width_ratios": [1, 1]})
fig.suptitle("图 11.5.3　DCGAN 生成结果与训练曲线",
             fontsize=22, fontweight="bold", y=0.98)

# ══════════════════════════════════════════════════════════════════
# (a) 生成样本网格
# ══════════════════════════════════════════════════════════════════
ax1.set_axis_off()
np.random.seed(42)

def make_digit_pattern(digit, size=16):
    img = np.zeros((size, size))
    cy, cx = size // 2, size // 2
    r = size * 0.35

    if digit == 0:
        for i in range(size):
            for j in range(size):
                dist = ((i - cy) / r)**2 + ((j - cx) / (r * 0.7))**2
                if 0.5 < dist < 1.4:
                    img[i, j] = 0.7 + 0.3 * np.random.rand()
    elif digit == 1:
        img[2:14, 7:9] = 0.7 + 0.3 * np.random.rand(12, 2)
        img[2:4, 5:9] = 0.6 + 0.3 * np.random.rand(2, 4)
    elif digit == 2:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:8, 10:12] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[7:9, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[9:14, 4:6] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[12:14, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
    elif digit == 3:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[7:9, 5:12] = 0.7 + 0.2 * np.random.rand(2, 7)
        img[12:14, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:14, 10:12] = 0.6 + 0.2 * np.random.rand(11, 2)
    elif digit == 4:
        img[3:9, 4:6] = 0.7 + 0.2 * np.random.rand(6, 2)
        img[8:10, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:14, 9:11] = 0.7 + 0.2 * np.random.rand(11, 2)
    elif digit == 5:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:8, 4:6] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[7:9, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[9:14, 10:12] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[12:14, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
    elif digit == 6:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:14, 4:6] = 0.6 + 0.2 * np.random.rand(11, 2)
        img[8:10, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[9:14, 10:12] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[12:14, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
    elif digit == 7:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        for k in range(12):
            r_idx = 4 + k
            c_idx = 11 - k * 0.5
            if 0 <= r_idx < size and 0 <= int(c_idx) < size:
                img[r_idx, int(c_idx)] = 0.7 + 0.2 * np.random.rand()
                if int(c_idx) + 1 < size:
                    img[r_idx, int(c_idx) + 1] = 0.5 + 0.2 * np.random.rand()
    elif digit == 8:
        for i in range(size):
            for j in range(size):
                d1 = ((i - 5) / 3)**2 + ((j - cx) / 3)**2
                if 0.4 < d1 < 1.3:
                    img[i, j] = max(img[i, j], 0.7 + 0.2 * np.random.rand())
                d2 = ((i - 11) / 3)**2 + ((j - cx) / 3)**2
                if 0.4 < d2 < 1.3:
                    img[i, j] = max(img[i, j], 0.7 + 0.2 * np.random.rand())
    elif digit == 9:
        img[3:5, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[3:9, 4:6] = 0.6 + 0.2 * np.random.rand(6, 2)
        img[3:9, 10:12] = 0.6 + 0.2 * np.random.rand(6, 2)
        img[7:9, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)
        img[9:14, 10:12] = 0.6 + 0.2 * np.random.rand(5, 2)
        img[12:14, 4:12] = 0.7 + 0.2 * np.random.rand(2, 8)

    img += 0.05 * np.random.randn(size, size)
    return np.clip(img, 0, 1)

grid_rows, grid_cols = 8, 8
grid_img = np.zeros((grid_rows * 16 + (grid_rows - 1),
                      grid_cols * 16 + (grid_cols - 1)))

for r in range(grid_rows):
    for c in range(grid_cols):
        digit = (r * grid_cols + c) % 10
        patch = make_digit_pattern(digit, 16)
        y0 = r * 17
        x0 = c * 17
        grid_img[y0:y0+16, x0:x0+16] = patch

inax = ax1.inset_axes([0.05, 0.05, 0.90, 0.85])
inax.imshow(grid_img, cmap="gray", vmin=0, vmax=1,
            interpolation="nearest", aspect="equal")
inax.set_xticks([])
inax.set_yticks([])
for spine in inax.spines.values():
    spine.set_visible(True)
    spine.set_color(COLORS["gray"])
    spine.set_linewidth(1.5)

ax1.text(0.50, 0.96, "$8 \\times 8$ 生成样本网格",
         fontsize=15, fontweight="bold", color=COLORS["gray"],
         transform=ax1.transAxes, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                   ec=COLORS["gray"], alpha=0.8))

ax1.set_title("(a) 生成样本", fontsize=17, fontweight="bold", pad=15)

# ══════════════════════════════════════════════════════════════════
# (b) 训练损失曲线
# ══════════════════════════════════════════════════════════════════
np.random.seed(88)
epochs = np.arange(1, 51)

d_loss = 1.5 * np.exp(-epochs / 5) + 0.5 + 0.08 * np.random.randn(50)
d_loss = np.clip(d_loss, 0.3, 2.0)
g_loss = 3.0 * np.exp(-epochs / 10) + 0.8 + 0.15 * np.random.randn(50)
g_loss = np.clip(g_loss, 0.4, 3.5)

ax2.plot(epochs, d_loss, color=COLORS["blue"], lw=2.5, alpha=0.9,
         label="判别器损失 $\\mathcal{L}_D$")
ax2.plot(epochs, g_loss, color=COLORS["red"], lw=2.5, alpha=0.9,
         label="生成器损失 $\\mathcal{L}_G$")

ax2.axhspan(0.5, 0.8, color=COLORS["green"], alpha=0.06)
ax2.text(42, 0.65, "均衡区间", fontsize=14, fontweight="bold",
         color=COLORS["green"], ha="center", alpha=0.8)

ax2.axvline(x=10, color=COLORS["gray"], ls=":", lw=1, alpha=0.5)
ax2.axvline(x=30, color=COLORS["gray"], ls=":", lw=1, alpha=0.5)
ax2.text(5, 3.2, "早期\nD 占优", fontsize=13, fontweight="bold",
         color=COLORS["gray"], ha="center", alpha=0.7)
ax2.text(20, 3.2, "中期\n对抗加剧", fontsize=13, fontweight="bold",
         color=COLORS["gray"], ha="center", alpha=0.7)
ax2.text(40, 3.2, "后期\n趋于均衡", fontsize=13, fontweight="bold",
         color=COLORS["gray"], ha="center", alpha=0.7)

ax2.set_xlabel("训练轮次 (Epoch)", fontsize=16)
ax2.set_ylabel("损失值", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=14, loc="upper right")
ax2.set_xlim(1, 50)
ax2.set_ylim(0, 3.8)
ax2.grid(alpha=0.3)
ax2.set_title("(b) 训练损失曲线", fontsize=17, fontweight="bold", pad=15)

# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig11_5_03_generation_results")
