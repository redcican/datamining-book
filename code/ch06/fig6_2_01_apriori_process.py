"""
图 6.2.1　Apriori 算法逐层搜索过程示意
展示从 1-项集到 3-项集的候选生成与剪枝过程
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()

fig, ax = plt.subplots(figsize=(16, 10))

# ── 数据：模拟 Apriori 过程 ──────────────────────────────────
# Level 1: 扫描后的频繁 1-项集
level1_freq = ['{A}', '{B}', '{C}', '{D}', '{E}']
level1_pruned = ['{F}']  # 非频繁

# Level 2: 候选 + 频繁
level2_cand = ['{A,B}', '{A,C}', '{A,D}', '{A,E}', '{B,C}',
               '{B,D}', '{B,E}', '{C,D}', '{C,E}', '{D,E}']
level2_freq = ['{A,B}', '{A,C}', '{B,C}', '{B,D}', '{C,D}']
level2_pruned = ['{A,D}', '{A,E}', '{B,E}', '{C,E}', '{D,E}']

# Level 3: 候选 + 频繁
level3_cand = ['{A,B,C}', '{B,C,D}']
level3_pruned = ['{A,B,D}', '{A,C,D}']  # 被剪枝（子集非频繁）
level3_freq = ['{A,B,C}', '{B,C,D}']

# ── 绘图参数 ──────────────────────────────────────────────────
y_positions = {1: 8, 2: 5, 3: 2}
box_width = 1.3
box_height = 0.7
gap = 0.15

def draw_itemset_box(ax, x, y, label, is_freq=True, is_pruned=False):
    """绘制一个项集方块"""
    if is_pruned:
        fc = '#fee2e2'
        ec = COLORS['red']
        tc = COLORS['red']
        alpha = 0.7
    elif is_freq:
        fc = '#dbeafe'
        ec = COLORS['blue']
        tc = COLORS['blue']
        alpha = 0.9
    else:
        fc = '#f1f5f9'
        ec = COLORS['gray']
        tc = COLORS['gray']
        alpha = 0.6

    rect = mpatches.FancyBboxPatch(
        (x - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle='round,pad=0.1',
        facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=tc)
    if is_pruned:
        # 划掉
        ax.plot([x - box_width/2 + 0.1, x + box_width/2 - 0.1],
                [y - 0.05, y - 0.05], '-', color=COLORS['red'],
                linewidth=2, alpha=0.6)

# ── Level 1 ──────────────────────────────────────────────────
y1 = y_positions[1]
all_l1 = level1_freq + level1_pruned
total_w1 = len(all_l1) * (box_width + gap) - gap
start_x1 = -total_w1 / 2 + box_width / 2

for i, item in enumerate(all_l1):
    x = start_x1 + i * (box_width + gap)
    is_pruned = item in level1_pruned
    draw_itemset_box(ax, x, y1, item, is_freq=not is_pruned, is_pruned=is_pruned)

# ── Level 2 ──────────────────────────────────────────────────
y2 = y_positions[2]
# 频繁的在左，非频繁的在右
all_l2 = level2_freq + level2_pruned
total_w2 = len(all_l2) * (box_width + gap) - gap
start_x2 = -total_w2 / 2 + box_width / 2

for i, item in enumerate(all_l2):
    x = start_x2 + i * (box_width + gap)
    is_pruned = item in level2_pruned
    draw_itemset_box(ax, x, y2, item, is_freq=not is_pruned, is_pruned=is_pruned)

# ── Level 3 ──────────────────────────────────────────────────
y3 = y_positions[3]
all_l3 = level3_freq + level3_pruned
total_w3 = len(all_l3) * (box_width + gap) - gap
start_x3 = -total_w3 / 2 + box_width / 2

for i, item in enumerate(all_l3):
    x = start_x3 + i * (box_width + gap)
    is_pruned = item in level3_pruned
    draw_itemset_box(ax, x, y3, item, is_freq=not is_pruned, is_pruned=is_pruned)

# ── 箭头连接各层 ─────────────────────────────────────────────
arrow_props = dict(arrowstyle='->', color=COLORS['blue'], lw=2, alpha=0.6)

# Level 1 → Level 2
ax.annotate('', xy=(0, y2 + box_height/2 + 0.3),
            xytext=(0, y1 - box_height/2 - 0.1),
            arrowprops=arrow_props)

# Level 2 → Level 3
ax.annotate('', xy=(0, y3 + box_height/2 + 0.3),
            xytext=(0, y2 - box_height/2 - 0.1),
            arrowprops=arrow_props)

# ── 标注 ──────────────────────────────────────────────────────
# 层标签
ax.text(-8.5, y1, '第 1 层\n$L_1$', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLORS['blue'])
ax.text(-8.5, y2, '第 2 层\n$L_2$', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLORS['blue'])
ax.text(-8.5, y3, '第 3 层\n$L_3$', ha='center', va='center',
        fontsize=12, fontweight='bold', color=COLORS['blue'])

# 操作标注
ax.text(8, (y1 + y2) / 2, '自连接生成 $C_2$\n+ 扫描数据库',
        ha='center', va='center', fontsize=11, color=COLORS['gray'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc',
                  edgecolor=COLORS['gray'], alpha=0.8))

ax.text(8, (y2 + y3) / 2, '自连接 + Apriori 剪枝\n生成 $C_3$ + 扫描',
        ha='center', va='center', fontsize=11, color=COLORS['gray'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc',
                  edgecolor=COLORS['gray'], alpha=0.8))

# 图例
legend_elements = [
    mpatches.Patch(facecolor='#dbeafe', edgecolor=COLORS['blue'],
                   label='频繁项集'),
    mpatches.Patch(facecolor='#fee2e2', edgecolor=COLORS['red'],
                   label='非频繁（被剪枝）'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

# 统计标注
ax.text(0, 0.5,
        '搜索空间压缩：$C_2$ = 10 个候选 → $L_2$ = 5 个频繁；'
        '$C_3$ = 4 个候选（剪枝 2 个）→ $L_3$ = 2 个频繁',
        ha='center', va='center', fontsize=11, color=COLORS['gray'],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbeb',
                  edgecolor=COLORS['orange'], alpha=0.9))

ax.set_xlim(-10, 10)
ax.set_ylim(-0.3, 9.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Apriori 算法逐层搜索过程', fontsize=15)

plt.tight_layout()
save_fig(fig, __file__, 'fig6_2_01_apriori_process')
