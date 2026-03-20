"""
fig8_3_03_dtw_matrix.py
DTW 距离矩阵与最优规整路径（热力图 + Sakoe-Chiba 带约束）
左：两条时间偏移的序列  右：累积代价矩阵 + 最优路径
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成两条时间偏移序列 ──────────────────────────────────────────
T = 60
t = np.arange(T)
x = np.sin(2 * np.pi * t / 30) + 0.3 * np.cos(2 * np.pi * t / 15)
# y 有非线性时间偏移
warp = t.astype(float).copy()
warp[:20] = t[:20] * 1.3
warp[20:40] = warp[19] + (t[20:40] - 19) * 0.6
warp[40:] = warp[39] + (t[40:] - 39) * 1.1
warp = np.clip(warp, 0, T - 1)
y = np.interp(warp, t, x) + np.random.normal(0, 0.05, T)
# ── DTW 计算（带 Sakoe-Chiba 约束） ──────────────────────────────
n, m = len(x), len(y)
band_width = 10  # Sakoe-Chiba 带宽
D = np.full((n + 1, m + 1), np.inf)
D[0, 0] = 0.0
cost_matrix = np.zeros((n, m))
for i in range(1, n + 1):
    for j in range(1, m + 1):
        cost = (x[i - 1] - y[j - 1]) ** 2
        cost_matrix[i - 1, j - 1] = cost
        D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
# 回溯最优路径
path = []
i, j = n, m
while i > 0 or j > 0:
    path.append((i - 1, j - 1))
    if i == 0:
        j -= 1
    elif j == 0:
        i -= 1
    else:
        argmin = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if argmin == 0:
            i, j = i - 1, j - 1
        elif argmin == 1:
            i -= 1
        else:
            j -= 1
path.reverse()
path_i = [p[0] for p in path]
path_j = [p[1] for p in path]
# ── 绘图 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.suptitle("图 8.3.3　DTW 距离矩阵与最优规整路径",
             fontsize=20, fontweight="bold", y=1.02)
# 使用 gridspec 创建不对称布局
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.3)
# ── 左面板：两条序列 ──────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
offset = 3.0
ax0.plot(t, x + offset, color=COLORS["blue"], lw=2.0, label="序列 X")
ax0.plot(t, y - offset, color=COLORS["orange"], lw=2.0, label="序列 Y")
# DTW 对齐连线（每隔几个点）
step = max(1, len(path) // 15)
for k in range(0, len(path), step):
    pi, pj = path[k]
    ax0.plot([pi, pj], [x[pi] + offset, y[pj] - offset],
             color=COLORS["green"], lw=0.7, alpha=0.5)
ax0.set_xlabel("时间步", fontsize=14)
ax0.set_title("(a) DTW 对齐", fontsize=15)
ax0.legend(fontsize=13, loc="upper right")
ax0.tick_params(labelsize=13)
ax0.set_yticks([])
# ── 右面板：累积代价矩阵热力图 ────────────────────────────────────
ax1 = fig.add_subplot(gs[1])
# 用累积代价矩阵（去掉 inf 边界行列）
D_plot = D[1:, 1:].copy()
im = ax1.imshow(D_plot, origin="lower", aspect="auto",
                cmap="YlOrRd", interpolation="nearest")
# 最优路径（白色曲线）
ax1.plot(path_j, path_i, color="white", lw=2.5, label="最优规整路径")
# Sakoe-Chiba 带约束（虚线）
diag_i = np.arange(n)
diag_j = diag_i * (m - 1) / (n - 1)
ax1.plot(diag_j + band_width, diag_i, color="cyan", ls="--", lw=1.5,
         alpha=0.8, label=f"Sakoe-Chiba 带 (r={band_width})")
ax1.plot(diag_j - band_width, diag_i, color="cyan", ls="--", lw=1.5,
         alpha=0.8)
ax1.set_xlim(0, m - 1)
ax1.set_ylim(0, n - 1)
ax1.set_xlabel("序列 Y 的时间步", fontsize=14)
ax1.set_ylabel("序列 X 的时间步", fontsize=14)
ax1.set_title("(b) 累积代价矩阵", fontsize=15)
ax1.legend(fontsize=12, loc="upper left",
           facecolor="black", edgecolor="white",
           labelcolor="white", framealpha=0.7)
ax1.tick_params(labelsize=13)
# 颜色条
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("累积代价", fontsize=13)
cbar.ax.tick_params(labelsize=12)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_3_03_dtw_matrix")
