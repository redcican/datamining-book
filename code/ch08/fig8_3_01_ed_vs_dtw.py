"""
fig8_3_01_ed_vs_dtw.py
欧氏距离与 DTW 的对齐方式对比
左：锁步对齐（红色连线）  右：DTW 弹性对齐（绿色连线）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成两条形状相似但时间偏移的序列 ──────────────────────────────
T = 50
t = np.arange(T)
x = np.sin(2 * np.pi * t / 25) + 0.3 * np.sin(2 * np.pi * t / 12)
# y 是 x 的非线性时间变形版本
warp = t.astype(float).copy()
warp[10:30] = 10 + (t[10:30] - 10) * 0.6   # 压缩中段
warp[30:] = warp[29] + (t[30:] - 29) * 1.4  # 拉伸后段
warp = np.clip(warp, 0, T - 1)
y = np.interp(warp, t, x) + np.random.normal(0, 0.05, T)
# ── DTW 计算 ──────────────────────────────────────────────────────
n, m = len(x), len(y)
D = np.full((n + 1, m + 1), np.inf)
D[0, 0] = 0.0
for i in range(1, n + 1):
    for j in range(1, m + 1):
        cost = (x[i - 1] - y[j - 1]) ** 2
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
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 8.3.1　欧氏距离与 DTW 的对齐方式对比",
             fontsize=20, fontweight="bold", y=1.02)
offset = 3.0  # 上下偏移量
# ── (a) 欧氏距离 — 锁步对齐 ──────────────────────────────────────
ax = axes[0]
ax.plot(t, x + offset, color=COLORS["blue"], lw=2.0, label="序列 X")
ax.plot(t, y - offset, color=COLORS["orange"], lw=2.0, label="序列 Y")
# 每隔 3 个点画一根红色连线
for k in range(0, T, 3):
    ax.plot([k, k], [x[k] + offset, y[k] - offset],
            color=COLORS["red"], lw=0.8, alpha=0.6)
ax.set_xlabel("时间步", fontsize=14)
ax.set_title("(a) 欧氏距离 — 锁步对齐", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)
ax.set_yticks([])
# ── (b) DTW — 弹性对齐 ───────────────────────────────────────────
ax = axes[1]
ax.plot(t, x + offset, color=COLORS["blue"], lw=2.0, label="序列 X")
ax.plot(t, y - offset, color=COLORS["orange"], lw=2.0, label="序列 Y")
# 按 DTW 路径画绿色连线（每隔几个点）
step = max(1, len(path) // 20)
for k in range(0, len(path), step):
    i_idx, j_idx = path[k]
    ax.plot([i_idx, j_idx], [x[i_idx] + offset, y[j_idx] - offset],
            color=COLORS["green"], lw=0.8, alpha=0.6)
ax.set_xlabel("时间步", fontsize=14)
ax.set_title("(b) DTW — 弹性对齐", fontsize=15)
ax.legend(fontsize=13, loc="upper right")
ax.tick_params(labelsize=13)
ax.set_yticks([])
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig8_3_01_ed_vs_dtw")
