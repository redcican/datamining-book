"""
图 6.3.1　FP-Tree 构建过程示意
左：最终 FP-Tree 结构（含项头表和 node-link）
右：事务插入顺序
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                gridspec_kw={'width_ratios': [1.4, 1]})

IC = {'A': COLORS['blue'], 'B': COLORS['green'],
      'C': COLORS['orange'], 'D': COLORS['purple'],
      'E': COLORS['red'], 'null': COLORS['gray']}

# ── 左图：FP-Tree ─────────────────────────────────────────────
ax = ax1
nodes = [
    ('null', '',   4.5, 7.2, -1),
    ('A', ':4',    2.8, 6.0, 0),
    ('B', ':4',    1.6, 4.8, 1),
    ('C', ':2',    0.5, 3.6, 2),
    ('E', ':2',    0.5, 2.4, 3),
    ('E', ':1',    2.8, 3.6, 2),
    ('D', ':1',    2.8, 2.4, 5),
    ('C', ':1',    4.5, 4.8, 1),
    ('D', ':1',    4.5, 3.6, 7),
    ('B', ':1',    6.8, 6.0, 0),
    ('C', ':1',    6.8, 4.8, 9),
    ('E', ':1',    6.8, 3.6, 10),
]
R = 0.32

for i, (l, c, x, y, p) in enumerate(nodes):
    if p >= 0:
        px, py = nodes[p][2], nodes[p][3]
        ax.plot([px, x], [py, y], '-', color='#94a3b8', lw=2, alpha=0.5, zorder=1)

for l, c, x, y, p in nodes:
    col = IC[l]
    if l == 'null':
        ax.add_patch(plt.Circle((x, y), R, fc='#e2e8f0', ec='#94a3b8', lw=2, zorder=5))
        ax.text(x, y, 'null', ha='center', va='center', fontsize=15,
                color='#64748b', fontweight='bold', zorder=10)
    else:
        ax.add_patch(plt.Circle((x, y), R, fc=col, ec='white', lw=2, zorder=5))
        ax.text(x, y, f'{l}{c}', ha='center', va='center', fontsize=15,
                fontweight='bold', color='white', zorder=10)

for item, pairs in {'B': [(2,9)], 'C': [(3,7),(7,10)],
                     'E': [(4,5),(5,11)], 'D': [(6,8)]}.items():
    for s, d in pairs:
        sx, sy = nodes[s][2], nodes[s][3]
        dx, dy = nodes[d][2], nodes[d][3]
        ax.annotate('', xy=(dx, dy+R*0.9), xytext=(sx+R*0.3, sy-R*0.3),
                    arrowprops=dict(arrowstyle='->', color=IC[item],
                                   ls='dashed', lw=1.5, alpha=0.5))

ht_x = -1.0
for j, (item, count) in enumerate([('A',4),('B',4),('C',4),('E',4),('D',2)]):
    ht_y = 6.0 - j * 0.85
    col = IC[item]
    rect = mpatches.FancyBboxPatch(
        (ht_x-0.55, ht_y-0.25), 1.1, 0.5,
        boxstyle='round,pad=0.06', fc=col, alpha=0.15, ec=col, lw=1.5)
    ax.add_patch(rect)
    ax.text(ht_x, ht_y, f'{item}:{count}', ha='center', va='center',
            fontsize=15, fontweight='bold', color=col)
ax.text(ht_x, 6.7, '项头表', ha='center', fontsize=16,
        fontweight='bold', color='#64748b')

ax.set_xlim(-2.2, 8.2)
ax.set_ylim(1.5, 8.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('FP-Tree 最终结构', fontsize=18, pad=8)

# ── 右图：事务插入顺序 ───────────────────────────────────────
ax = ax2
trans = [
    ('T1', 'A→B→C→E',   '新建完整路径'),
    ('T2', 'A→B→E→D',   '共享A→B, 分支E→D'),
    ('T3', 'B→C→E',     '新路径'),
    ('T4', 'A→B→C→E',   '与T1共享, 计数+1'),
    ('T5', 'A→C→D',     '共享A, 分支C→D'),
]
tc = [COLORS['blue'], COLORS['green'], COLORS['orange'],
      COLORS['purple'], COLORS['red']]

ax.text(0.05, 0.97, '$f$-list: A(4) B(4) C(4) E(4) D(2)',
        transform=ax.transAxes, fontsize=16, color=COLORS['blue'],
        fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='#eff6ff', ec=COLORS['blue']))

for i, (tid, path, desc) in enumerate(trans):
    y = 0.84 - i * 0.175
    ax.text(0.02, y, tid, transform=ax.transAxes, fontsize=18,
            fontweight='bold', color=tc[i], va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc=tc[i], alpha=0.12, ec=tc[i]))
    ax.text(0.18, y, path, transform=ax.transAxes, fontsize=18,
            fontfamily='monospace', color='black', va='center')
    ax.text(0.18, y - 0.05, desc, transform=ax.transAxes, fontsize=14,
            color='#64748b', va='center')

ax.axis('off')
ax.set_title('事务插入过程', fontsize=18, pad=8)

plt.suptitle('FP-Tree 构建过程示意（定义 6.8, 算法 6.2）', fontsize=20, y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, __file__, 'fig6_3_01_fptree_construction')
