"""
图 6.7.3　第六章知识体系总结
以关联规则挖掘管线为主线的知识地图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# ── 配色与样式 ────────────────────────────────────────────
section_colors = {
    '6.1': COLORS['blue'],
    '6.2': COLORS['teal'],
    '6.3': COLORS['green'],
    '6.4': COLORS['orange'],
    '6.5': COLORS['purple'],
    '6.6': COLORS['red'],
    '6.7': '#475569',
}

bbox_kw = dict(boxstyle='round,pad=0.4', lw=2, alpha=0.95)
fs_title = 18
fs_body = 15
fs_arrow = 16

# ── 顶部：管线流程 ────────────────────────────────────────
pipeline_y = 9.0
pipeline_stages = [
    ('数据', 1.5, COLORS['gray']),
    ('频繁项集', 4.0, section_colors['6.2']),
    ('关联规则', 6.8, section_colors['6.1']),
    ('过滤评估', 9.5, section_colors['6.4']),
    ('可视化', 12.0, section_colors['6.6']),
    ('决策', 14.5, section_colors['6.7']),
]

for name, x, color in pipeline_stages:
    ax.text(x, pipeline_y, name, fontsize=fs_title, fontweight='bold',
            ha='center', va='center', color='white',
            bbox=dict(**bbox_kw, fc=color, ec=color))

# 箭头连接
arrow_kw = dict(arrowstyle='->', color='#94a3b8', lw=2.5)
for i in range(len(pipeline_stages) - 1):
    x1 = pipeline_stages[i][1] + 0.8
    x2 = pipeline_stages[i+1][1] - 0.8
    ax.annotate('', xy=(x2, pipeline_y), xytext=(x1, pipeline_y),
                arrowprops=arrow_kw)

# ── 中间：各节内容 ────────────────────────────────────────
sections = [
    {
        'label': '§6.1 基本概念',
        'x': 1.5, 'y': 6.8,
        'items': ['事务数据库', '支持度/置信度/提升度', '反单调性定理'],
        'color': section_colors['6.1'],
    },
    {
        'label': '§6.2 Apriori',
        'x': 4.0, 'y': 6.8,
        'items': ['逐层搜索', '候选生成与剪枝', '连接+验证步骤'],
        'color': section_colors['6.2'],
    },
    {
        'label': '§6.3 FP-Growth',
        'x': 6.8, 'y': 6.8,
        'items': ['FP-Tree 压缩存储', '条件模式基', '无候选生成'],
        'color': section_colors['6.3'],
    },
    {
        'label': '§6.4 兴趣度量',
        'x': 9.5, 'y': 6.8,
        'items': ['2×2 列联表', 'Kulczynski + IR', '零不变性'],
        'color': section_colors['6.4'],
    },
    {
        'label': '§6.5 多层多维',
        'x': 12.0, 'y': 6.8,
        'items': ['概念层次', '递减支持度', '量化属性离散化'],
        'color': section_colors['6.5'],
    },
    {
        'label': '§6.6 可视化',
        'x': 14.5, 'y': 6.8,
        'items': ['规则网络图', '矩阵热力图', '冗余消除'],
        'color': section_colors['6.6'],
    },
]

for sec in sections:
    # 节标题
    ax.text(sec['x'], sec['y'], sec['label'], fontsize=fs_title,
            fontweight='bold', ha='center', va='center',
            color=sec['color'],
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=sec['color'], lw=1.5))
    # 内容项
    for k, item in enumerate(sec['items']):
        ax.text(sec['x'], sec['y'] - 0.7 - k * 0.55, f'• {item}',
                fontsize=fs_body, ha='center', va='center',
                color='#334155')

    # 连接到管线
    ax.annotate('', xy=(sec['x'], pipeline_y - 0.5),
                xytext=(sec['x'], sec['y'] + 0.4),
                arrowprops=dict(arrowstyle='->', color=sec['color'],
                                lw=1.5, alpha=0.5, ls='--'))

# ── 底部：定理列表 ────────────────────────────────────────
theorems_y = 3.0
theorem_items = [
    ('定理 6.1: 反单调性', section_colors['6.1']),
    ('定理 6.6: FP-Tree 完备性', section_colors['6.3']),
    ('定理 6.9: χ²与提升度', section_colors['6.4']),
    ('定理 6.10: 零不变性', section_colors['6.4']),
    ('定理 6.11: 层次单调性', section_colors['6.5']),
    ('定理 6.12: 冗余检测', section_colors['6.6']),
]

x_positions = np.linspace(1.5, 14.5, len(theorem_items))
for (thm, color), x in zip(theorem_items, x_positions):
    ax.text(x, theorems_y, thm, fontsize=fs_body-2, ha='center',
            va='center', color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      ec=color, lw=1, alpha=0.9))

# ── 最底部：章节总结 ──────────────────────────────────────
summary_y = 1.5
ax.text(8.0, summary_y,
        '§6.7 应用案例：零售购物篮 | 医疗诊断 | Web推荐 — 从理论到实践的完整闭环',
        fontsize=fs_body+2, ha='center', va='center',
        color=section_colors['6.7'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', fc='#f1f5f9',
                  ec=section_colors['6.7'], lw=2))

ax.set_title('第六章 关联规则挖掘 — 知识体系总结', fontsize=22, pad=15)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_7_03_chapter_summary')
