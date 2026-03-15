"""
fig5_2_01_dbscan_concepts.py
DBSCAN 核心概念图：
左：ε-邻域 + 核心点/边界点/噪声点 四类点示意
右：密度可达/密度连接关系链示意
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()
rng = np.random.default_rng(7)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ══════════════════════════════════════════════
# Panel 1: 四类点示意
# ══════════════════════════════════════════════
ax = axes[0]
EPS = 1.0
MINPTS = 4

# 手工放置点
core1   = np.array([3.0, 3.0])   # 核心点1（大邻域）
core2   = np.array([4.5, 3.2])   # 核心点2
border1 = np.array([1.8, 3.6])   # 边界点（在core1邻域内，但自身邻域<MinPts）
noise1  = np.array([6.5, 1.2])   # 噪声点
noise2  = np.array([0.5, 1.0])   # 噪声点2

# core1 的邻域内的点
neighbors_c1 = np.array([
    [2.4, 2.5], [3.5, 2.6], [3.2, 3.7], [2.6, 3.4], border1
])
# core2 的邻域内的点
neighbors_c2 = np.array([
    [4.0, 2.5], [5.1, 2.8], [4.8, 3.9], core1
])

# 画 ε 圆
for center, alpha in [(core1, 0.12), (core2, 0.10)]:
    circ = Circle(center, EPS, fill=True, facecolor=PALETTE[0],
                  edgecolor=PALETTE[0], linewidth=1.5, alpha=alpha, zorder=1)
    ax.add_patch(circ)
    circ2 = Circle(center, EPS, fill=False,
                   edgecolor=PALETTE[0], linewidth=1.5, linestyle='--', zorder=2)
    ax.add_patch(circ2)

# 画边界点的小邻域（点稀疏）
circ_b = Circle(border1, EPS, fill=True, facecolor='gray',
                edgecolor='gray', linewidth=1, alpha=0.06, zorder=1)
ax.add_patch(circ_b)
circ_b2 = Circle(border1, EPS, fill=False,
                 edgecolor='gray', linewidth=1, linestyle=':', zorder=2)
ax.add_patch(circ_b2)

# 画点
ax.scatter(*core1,   c=PALETTE[0], s=120, zorder=5, edgecolors='k', linewidths=0.8)
ax.scatter(*core2,   c=PALETTE[0], s=120, zorder=5, edgecolors='k', linewidths=0.8)
ax.scatter(neighbors_c1[:,0], neighbors_c1[:,1],
           c=PALETTE[1], s=60, zorder=4, edgecolors='k', linewidths=0.6)
ax.scatter(neighbors_c2[:,0], neighbors_c2[:,1],
           c=PALETTE[1], s=60, zorder=4, edgecolors='k', linewidths=0.6)
ax.scatter(*border1, c=PALETTE[2], s=80, zorder=6,
           edgecolors='k', linewidths=0.8, marker='s')
ax.scatter(*noise1,  c=PALETTE[3], s=80, zorder=6,
           edgecolors='k', linewidths=0.8, marker='x')
ax.scatter(*noise2,  c=PALETTE[3], s=80, zorder=6,
           edgecolors='k', linewidths=0.8, marker='x')

# 标注
ax.annotate('核心点 $p$\n$(|N_\\varepsilon(p)| \\geq \\mathrm{MinPts})$',
            xy=core1, xytext=(1.0, 4.6),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=8.5, ha='center')
ax.annotate('核心点 $q$', xy=core2, xytext=(5.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=8.5, ha='center')
ax.annotate('边界点\n$(|N_\\varepsilon| < \\mathrm{MinPts})$',
            xy=border1, xytext=(0.2, 2.0),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.0),
            fontsize=8.5, ha='center')
ax.annotate('噪声点', xy=noise1, xytext=(6.0, 0.5),
            arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0),
            fontsize=8.5)
ax.text(3.5, 3.0, f'$\\varepsilon$', fontsize=11, color=PALETTE[0])
ax.text(3.0, 0.3, f'$\\varepsilon={EPS}$，MinPts$={MINPTS}$',
        fontsize=9, ha='center', style='italic')

legend_handles = [
    mpatches.Patch(color=PALETTE[0], label='核心点'),
    mpatches.Patch(color=PALETTE[1], label='（核心点邻域内的点）'),
    mpatches.Patch(color=PALETTE[2], label='边界点'),
    mpatches.Patch(color=PALETTE[3], label='噪声点'),
]
ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
ax.set_xlim(-0.2, 7.5); ax.set_ylim(-0.2, 5.8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('（a）四类点：核心点、边界点、噪声点', fontsize=10)

# ══════════════════════════════════════════════
# Panel 2: 密度可达 & 密度连接关系链
# ══════════════════════════════════════════════
ax = axes[1]

# 构造一条密度可达链: p1 -> p2 -> p3 -> q
# p1, p2, p3 都是核心点；q 是边界点
pts = {
    'p_1': np.array([1.0, 3.0]),
    'p_2': np.array([2.5, 3.0]),
    'p_3': np.array([4.0, 3.0]),
    'q':   np.array([5.3, 3.0]),   # 边界点（密度可达自 p_3）
    'r':   np.array([0.0, 1.0]),   # 另一个边界点（密度可达自 p_1）
    'o':   np.array([2.5, 1.5]),   # 公共核心点，使 r 和 q 密度连接
}
# o 是核心点，既到 r 密度可达，又到 q 密度可达
# 为简化：让 p_1=o 的角色，直接画链

core_pts = ['p_1', 'p_2', 'p_3']
border_pts = ['q', 'r']

# ε 圆（只画核心点）
for name in core_pts:
    c = pts[name]
    circ = Circle(c, 0.8, fill=True, facecolor=PALETTE[0],
                  alpha=0.10, zorder=1)
    ax.add_patch(circ)
    circ2 = Circle(c, 0.8, fill=False,
                   edgecolor=PALETTE[0], linewidth=1.2,
                   linestyle='--', zorder=2)
    ax.add_patch(circ2)

# 画箭头：直接密度可达
for a_name, b_name in [('p_1','p_2'), ('p_2','p_3'), ('p_3','q'),
                        ('p_1','r')]:
    a, b = pts[a_name], pts[b_name]
    ax.annotate('', xy=b, xytext=a,
                arrowprops=dict(arrowstyle='->', color=PALETTE[0],
                                lw=1.8, connectionstyle='arc3,rad=0.0'))

# 密度连接弧（r 与 q 通过 p_1→p_2→p_3 密度连接）
ax.annotate('', xy=pts['q'], xytext=pts['r'],
            arrowprops=dict(arrowstyle='<->', color=PALETTE[2],
                            lw=2.0, linestyle='dashed',
                            connectionstyle='arc3,rad=-0.35'))

# 画点
for name in core_pts:
    ax.scatter(*pts[name], c=PALETTE[0], s=130, zorder=5,
               edgecolors='k', linewidths=0.8)
for name in border_pts:
    ax.scatter(*pts[name], c=PALETTE[2], s=90, zorder=5,
               edgecolors='k', linewidths=0.8, marker='s')

# 标签
offsets = {'p_1': (-0.1, 0.25), 'p_2': (0, 0.25),
           'p_3': (0, 0.25), 'q': (0.15, 0.25), 'r': (-0.1, -0.35)}
for name, pt in pts.items():
    if name == 'o': continue
    dx, dy = offsets.get(name, (0, 0.25))
    ax.text(pt[0]+dx, pt[1]+dy, f'${name}$', fontsize=10, ha='center')

# 标注
ax.text(1.75, 3.35, '直接密度可达', fontsize=8, color=PALETTE[0], ha='center')
ax.text(1.75, 2.65, '直接密度可达', fontsize=8, color=PALETTE[0], ha='center')
ax.text(2.5, 1.0, '密度连接（经 $p_1 \\to p_2 \\to p_3$）',
        fontsize=8.5, color=PALETTE[2], ha='center')

ax.set_xlim(-0.8, 6.5); ax.set_ylim(-0.2, 4.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('（b）密度可达链与密度连接关系', fontsize=10)

fig.suptitle('DBSCAN 核心概念：点类型与密度关系', fontsize=11, y=1.01)
fig.tight_layout()
save_fig(fig, __file__, 'fig5_2_01_dbscan_concepts')
