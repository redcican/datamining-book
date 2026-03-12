"""
图 1.1.2  数据矩阵与欧氏距离矩阵可视化（鸢尾花数据集）
对应节次：1.1 数据挖掘的定义与发展历程
运行方式：python code/ch01/fig1_1_03_matrix.py
输出路径：public/figures/ch01/fig1_1_03_matrix.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris

apply_style()

# ── 数据准备 ───────────────────────────────────────────────────────────
iris = load_iris()
# 每类各取前 7 个样本，共 21 个
idx = list(range(0, 7)) + list(range(50, 57)) + list(range(100, 107))
X = iris.data[idx]
y = iris.target[idx]
feature_names = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
n = X.shape[0]

# 欧氏距离矩阵
diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
dist = np.sqrt((diff ** 2).sum(axis=-1))

# ── 物种配色 ──────────────────────────────────────────────────────────
CLASS_COLORS = ["#2563eb", "#16a34a", "#dc2626"]
CLASS_NAMES  = ["山鸢尾 (Setosa)", "杂色鸢尾 (Versicolor)", "维吉尼亚 (Virginica)"]
sample_colors = [CLASS_COLORS[c] for c in y]

# ── 画布与子图布局 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.18,
                       left=0.07, right=0.96, top=0.88, bottom=0.09)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ── 左图：特征矩阵热力图 ───────────────────────────────────────────────
im1 = ax1.imshow(X, aspect="auto", cmap="Blues", interpolation="nearest",
                 vmin=X.min() - 0.3)

ax1.set_xticks(range(4))
ax1.set_xticklabels(feature_names, fontsize=12, rotation=18, ha="right")
ax1.set_yticks(range(n))
ax1.set_yticklabels([f"$\\mathbf{{x}}_{{{i+1}}}$" for i in range(n)], fontsize=12.5)
ax1.set_xlabel("特征维度（$d = 4$）", fontsize=12)
ax1.set_ylabel("样本编号（$n = 21$）", fontsize=12)

# 子图标题放在 axes 外部（上方）
ax1.set_title("(a)  特征矩阵 $\\mathbf{X} \\in \\mathbb{R}^{21\\times4}$",
              loc="left", fontsize=12, fontweight="bold", color="#1e293b", pad=8)

# 热力图数值
for i in range(n):
    for j in range(4):
        val = X[i, j]
        tc = "white" if val > X[:, j].max() * 0.70 else "#1e293b"
        ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                 fontsize=12, color=tc, fontweight="bold")

# 左侧物种色条
for i, c in enumerate(sample_colors):
    ax1.add_patch(plt.Rectangle((-0.75, i - 0.5), 0.4, 1.0,
                                 color=c, transform=ax1.transData, clip_on=False))

cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.032, pad=0.03)
cbar1.set_label("特征值（cm）", fontsize=12)
cbar1.ax.tick_params(labelsize=10)

# ── 右图：欧氏距离矩阵热力图 ───────────────────────────────────────────
im2 = ax2.imshow(dist, cmap="YlOrRd", interpolation="nearest")

ax2.set_xlabel("样本编号", fontsize=12)
ax2.set_ylabel("样本编号", fontsize=12)
ax2.set_xticks(range(n))
ax2.set_xticklabels(range(1, n + 1), fontsize=12.5)
ax2.set_yticks(range(n))
ax2.set_yticklabels(range(1, n + 1), fontsize=12.5)

# 子图标题放在 axes 外部（上方）
ax2.set_title("(b)  欧氏距离矩阵 $D \\in \\mathbb{R}^{21\\times21}$",
              loc="left", fontsize=12, fontweight="bold", color="#1e293b", pad=8)

# 类别分隔虚线
for bd in [6.5, 13.5]:
    ax2.axhline(bd, color="#1e293b", lw=1.8, linestyle="--", alpha=0.65)
    ax2.axvline(bd, color="#1e293b", lw=1.8, linestyle="--", alpha=0.65)

cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.032, pad=0.03)
cbar2.set_label("欧氏距离（cm）", fontsize=12)
cbar2.ax.tick_params(labelsize=10)

save_fig(fig, __file__, "fig1_1_03_matrix")
