"""
图 6.1.3　项集格与 Apriori 剪枝
左：完整的 4-项集格
右：当 {D} 非频繁时，剪枝后的格
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()

items = ['A', 'B', 'C', 'D']

def generate_lattice_positions(items):
    """生成项集格的节点位置"""
    nodes = {}
    # Level 0: empty set
    nodes[frozenset()] = (0, 0)
    # Level 1
    n1 = len(items)
    for i, item in enumerate(items):
        x = (i - (n1 - 1) / 2) * 2
        nodes[frozenset({item})] = (x, 1.5)
    # Level 2
    pairs = list(combinations(items, 2))
    n2 = len(pairs)
    for i, pair in enumerate(pairs):
        x = (i - (n2 - 1) / 2) * 1.5
        nodes[frozenset(pair)] = (x, 3.0)
    # Level 3
    triples = list(combinations(items, 3))
    n3 = len(triples)
    for i, triple in enumerate(triples):
        x = (i - (n3 - 1) / 2) * 2
        nodes[frozenset(triple)] = (x, 4.5)
    # Level 4
    nodes[frozenset(items)] = (0, 6.0)
    return nodes

def draw_lattice(ax, nodes, pruned=None, title=''):
    """Draw the lattice with optional pruned nodes."""
    if pruned is None:
        pruned = set()

    # Draw edges first
    all_sets = list(nodes.keys())
    for s1 in all_sets:
        for s2 in all_sets:
            if len(s2) == len(s1) + 1 and s1.issubset(s2):
                x1, y1 = nodes[s1]
                x2, y2 = nodes[s2]
                is_pruned = s1 in pruned or s2 in pruned
                color = '#d1d5db' if is_pruned else COLORS['blue']
                alpha = 0.3 if is_pruned else 0.5
                ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=1.5,
                        alpha=alpha, zorder=1)

    # Draw nodes
    for itemset, (x, y) in nodes.items():
        is_pruned = itemset in pruned
        if len(itemset) == 0:
            label = '∅'
        else:
            label = '{' + ','.join(sorted(itemset)) + '}'

        if is_pruned:
            color = '#e5e7eb'
            edge_color = '#9ca3af'
            text_color = '#9ca3af'
        else:
            color = COLORS['blue'] if len(itemset) <= 1 else PALETTE[min(len(itemset), 5)]
            edge_color = 'white'
            text_color = 'white'

        circle = plt.Circle((x, y), 0.4, color=color, ec=edge_color,
                             linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color=text_color, zorder=10)

    # Cross out pruned nodes
    for itemset in pruned:
        x, y = nodes[itemset]
        ax.plot([x-0.3, x+0.3], [y-0.3, y+0.3], '-', color=COLORS['red'],
                linewidth=2, zorder=15)
        ax.plot([x-0.3, x+0.3], [y+0.3, y-0.3], '-', color=COLORS['red'],
                linewidth=2, zorder=15)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.8, 7)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13)
    ax.axis('off')

    # Level labels
    for level, y_pos in [(0, 0), (1, 1.5), (2, 3.0), (3, 4.5), (4, 6.0)]:
        ax.text(-4.5, y_pos, f'{level}-项集', fontsize=9, ha='center',
                color=COLORS['gray'])

nodes = generate_lattice_positions(items)

# Determine pruned nodes (all supersets of {D})
pruned = set()
for itemset in nodes:
    if 'D' in itemset and len(itemset) >= 1:
        pruned.add(itemset)

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

draw_lattice(ax1, nodes, pruned=None,
             title='完整项集格（4 项，$2^4=16$ 个节点）')

draw_lattice(ax2, nodes, pruned=pruned,
             title='Apriori 剪枝后（{D} 非频繁，8 个节点被剪除）')

# Add annotation
ax2.annotate('{D} 非频繁\n→ 所有包含 D 的\n项集全部剪枝',
             xy=(nodes[frozenset({'D'})][0], nodes[frozenset({'D'})][1]),
             xytext=(3.5, 0.5), fontsize=10, color=COLORS['red'],
             arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef2f2',
                       edgecolor=COLORS['red'], alpha=0.9))

n_total = len(nodes)
n_pruned = len(pruned)
ax2.text(0, -0.5, f'搜索空间：{n_total} → {n_total - n_pruned}（减少 {n_pruned/n_total:.0%}）',
         ha='center', fontsize=11, color=COLORS['red'], fontweight='bold')

plt.suptitle('项集格与 Apriori 性质的剪枝效果（定理 6.1）', fontsize=15, y=0.98)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_1_03_apriori_lattice')
