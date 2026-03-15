"""
图 5.7.1　高维距离集中现象
在 d=2/10/100/1000 维空间中，计算查询点到 500 个均匀分布点的距离分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成不同维度的距离分布 ─────────────────────────────
dims = [2, 10, 100, 1000]
n_points = 500
query = None  # 使用原点

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for ax, d in zip(axes, dims):
    X = np.random.uniform(0, 1, size=(n_points, d))
    q = np.zeros(d)
    dists = np.linalg.norm(X - q, axis=1)

    # 归一化距离（除以 sqrt(d)）
    dists_norm = dists / np.sqrt(d)

    ax.hist(dists_norm, bins=30, color=COLORS['blue'], alpha=0.7,
            edgecolor='white', density=True)

    d_min, d_max = dists.min(), dists.max()
    rho = (d_max - d_min) / d_min
    cv = np.std(dists) / np.mean(dists)

    ax.set_title(f'd = {d}\n$\\rho$ = {rho:.3f}，CV = {cv:.3f}', fontsize=12)
    ax.set_xlabel('归一化距离 $\\|\\mathbf{x}\\| / \\sqrt{d}$')
    if ax == axes[0]:
        ax.set_ylabel('密度')

plt.suptitle('高维距离集中现象：随维度增加，距离分布愈发集中', fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig5_7_01_curse_dimensionality')
