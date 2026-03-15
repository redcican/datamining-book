"""
图 6.5.2　量化属性离散化策略对比
以"年龄"属性为例，展示等宽、等频和基于聚类三种离散化方法
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成年龄数据（模拟购物者分布：双峰 + 偏态）───────────────
ages = np.concatenate([
    np.random.normal(28, 5, 300),   # 年轻群体
    np.random.normal(52, 8, 200),   # 中年群体
    np.random.normal(38, 3, 100),   # 小峰
]).clip(18, 75)

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

methods = [
    ('等宽离散化', 'equal_width'),
    ('等频离散化', 'equal_freq'),
    ('基于聚类离散化', 'cluster'),
]
method_colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]

for idx, (ax, (title, method), color) in enumerate(zip(axes, methods, method_colors)):
    # 计算分箱边界
    n_bins = 4
    if method == 'equal_width':
        bins = np.linspace(ages.min(), ages.max(), n_bins + 1)
    elif method == 'equal_freq':
        bins = np.percentile(ages, np.linspace(0, 100, n_bins + 1))
    else:  # cluster-based
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        km.fit(ages.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.flatten())
        boundaries = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
        bins = np.array([ages.min()] + boundaries + [ages.max()])

    # 绘制直方图
    n, _, patches = ax.hist(ages, bins=30, alpha=0.3, color=color, edgecolor='white')

    # 绘制分箱边界
    for i, b in enumerate(bins):
        ax.axvline(x=b, color=color, ls='--', lw=2, alpha=0.8)

    # 填充分箱区域 + 标注
    alphas = [0.15, 0.25, 0.15, 0.25]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        ax.axvspan(lo, hi, alpha=alphas[i], color=color)
        count = np.sum((ages >= lo) & (ages < hi)) if i < n_bins-1 else np.sum((ages >= lo) & (ages <= hi))
        mid = (lo + hi) / 2
        ax.text(mid, max(n) * 0.92, f'B{i+1}\nn={count}',
                ha='center', va='top', fontsize=13, fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9))
        # 区间标注
        lo_str = f'{lo:.0f}'
        hi_str = f'{hi:.0f}'
        ax.text(mid, max(n) * 0.05, f'[{lo_str},{hi_str})',
                ha='center', va='bottom', fontsize=11, color=color)

    ax.set_xlabel('年龄', fontsize=15)
    ax.set_title(title, fontsize=17, color=color)
    ax.set_xlim(15, 78)

axes[0].set_ylabel('频次', fontsize=15)

# 底部标注
fig.text(0.5, -0.02,
         '等宽：区间等长但样本不均 | 等频：样本均匀但区间不等 | 聚类：自适应数据分布',
         ha='center', fontsize=14, color=COLORS['gray'],
         bbox=dict(boxstyle='round,pad=0.4', fc='#f8fafc', ec=COLORS['gray']))

plt.suptitle('量化属性离散化策略对比（以年龄为例）', fontsize=19, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_5_02_discretization')
