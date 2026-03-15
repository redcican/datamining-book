"""
图 6.5.1　概念层次与多层次挖掘示意
上方：三层商品概念层次树  下方：各层挖掘结果对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [1.2, 1]})

# ── 上半：概念层次树 ──────────────────────────────────────────
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 4)
ax1.axis('off')

# 层次节点定义
root = {'pos': (5, 3.5), 'label': '所有商品', 'color': COLORS['gray']}
level1 = [
    {'pos': (1.8, 2.3), 'label': '乳制品', 'color': COLORS['blue']},
    {'pos': (5.0, 2.3), 'label': '面包类', 'color': COLORS['green']},
    {'pos': (8.2, 2.3), 'label': '饮料', 'color': COLORS['orange']},
]
level2 = [
    # 乳制品
    {'pos': (0.6, 1.0), 'label': '2%牛奶', 'color': COLORS['blue'], 'parent': 0},
    {'pos': (1.8, 1.0), 'label': '全脂牛奶', 'color': COLORS['blue'], 'parent': 0},
    {'pos': (3.0, 1.0), 'label': '酸奶', 'color': COLORS['blue'], 'parent': 0},
    # 面包类
    {'pos': (4.2, 1.0), 'label': '全麦面包', 'color': COLORS['green'], 'parent': 1},
    {'pos': (5.6, 1.0), 'label': '白面包', 'color': COLORS['green'], 'parent': 1},
    # 饮料
    {'pos': (7.2, 1.0), 'label': '可乐', 'color': COLORS['orange'], 'parent': 2},
    {'pos': (8.2, 1.0), 'label': '橙汁', 'color': COLORS['orange'], 'parent': 2},
    {'pos': (9.2, 1.0), 'label': '矿泉水', 'color': COLORS['orange'], 'parent': 2},
]

bbox_kw = dict(boxstyle='round,pad=0.3', lw=1.5, alpha=0.9)
fs_node = 14
fs_layer = 15

# 层标签
ax1.text(-0.2, 3.5, '第0层\n(根)', fontsize=fs_layer, fontweight='bold',
         color=COLORS['gray'], ha='center', va='center')
ax1.text(-0.2, 2.3, '第1层\n(类别)', fontsize=fs_layer, fontweight='bold',
         color=COLORS['purple'], ha='center', va='center')
ax1.text(-0.2, 1.0, '第2层\n(商品)', fontsize=fs_layer, fontweight='bold',
         color=COLORS['teal'], ha='center', va='center')

# 绘制根
ax1.text(*root['pos'], root['label'], fontsize=fs_node+2, fontweight='bold',
         ha='center', va='center',
         bbox=dict(**bbox_kw, fc='#f1f5f9', ec=COLORS['gray']))

# 绘制第1层 + 连线到根
for node in level1:
    ax1.annotate('', xy=(root['pos'][0], root['pos'][1]-0.3),
                 xytext=(node['pos'][0], node['pos'][1]+0.3),
                 arrowprops=dict(arrowstyle='-', color='#94a3b8', lw=2))
    ax1.text(*node['pos'], node['label'], fontsize=fs_node, fontweight='bold',
             ha='center', va='center',
             bbox=dict(**bbox_kw, fc='white', ec=node['color']))

# 绘制第2层 + 连线到第1层
for node in level2:
    parent = level1[node['parent']]
    ax1.annotate('', xy=(parent['pos'][0], parent['pos'][1]-0.3),
                 xytext=(node['pos'][0], node['pos'][1]+0.3),
                 arrowprops=dict(arrowstyle='-', color='#94a3b8', lw=1.5))
    ax1.text(*node['pos'], node['label'], fontsize=fs_node-1,
             ha='center', va='center',
             bbox=dict(**bbox_kw, fc='white', ec=node['color']))

ax1.set_title('商品概念层次树（三层结构）', fontsize=18, pad=10)

# ── 下半：各层挖掘结果对比 ────────────────────────────────────
levels = ['第2层\n(具体商品)', '第1层\n(类别)', '第0层\n(根)']
freq_counts = [8, 18, 5]
avg_support = [0.03, 0.12, 0.45]
rules_examples = [
    '2%牛奶→全麦面包\n(supp=0.02, conf=0.35)',
    '乳制品→面包类\n(supp=0.15, conf=0.52)',
    '所有商品→所有商品\n(trivial)'
]
bar_colors = [COLORS['teal'], COLORS['purple'], COLORS['gray']]

x = np.arange(len(levels))
width = 0.35

bars1 = ax2.bar(x - width/2, freq_counts, width, color=bar_colors,
                alpha=0.85, edgecolor='white', linewidth=2, label='频繁项集数')
ax2r = ax2.twinx()
bars2 = ax2r.bar(x + width/2, avg_support, width, color=bar_colors,
                 alpha=0.4, edgecolor=bar_colors, linewidth=2,
                 hatch='///', label='平均支持度')

ax2.set_xticks(x)
ax2.set_xticklabels(levels, fontsize=14)
ax2.set_ylabel('频繁项集数量', fontsize=15)
ax2r.set_ylabel('平均支持度', fontsize=15)
ax2.set_title('不同层次的挖掘结果对比', fontsize=18, pad=10)

# 标注典型规则
for i, (cnt, rule) in enumerate(zip(freq_counts, rules_examples)):
    ax2.annotate(rule, xy=(i - width/2, cnt),
                 xytext=(i - 0.1, cnt + 3),
                 fontsize=12, color=bar_colors[i],
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec=bar_colors[i], alpha=0.9))

# 合并图例
handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2, fontsize=13, loc='upper left')

ax2.set_ylim(0, max(freq_counts) * 1.6)
ax2r.set_ylim(0, 0.6)

plt.tight_layout()
save_fig(fig, __file__, 'fig6_5_01_concept_hierarchy')
