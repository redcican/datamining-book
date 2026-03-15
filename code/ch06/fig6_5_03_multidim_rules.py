"""
图 6.5.3　多维关联规则发现
散点图展示各规则的支持度-提升度分布，标注含量化维的高提升度规则
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 模拟多维关联规则挖掘结果 ──────────────────────────────────
# 规则格式：(规则描述, 支持度, 置信度, 提升度, 类型)
# 类型: 'boolean' = 纯布尔维, 'quantitative' = 含量化维, 'mixed' = 混合
rules = [
    # 纯布尔维规则
    ('面包→牛奶', 0.15, 0.52, 1.18, 'boolean'),
    ('啤酒→薯片', 0.08, 0.61, 2.15, 'boolean'),
    ('鸡蛋→面包', 0.12, 0.45, 1.05, 'boolean'),
    ('牛奶→酸奶', 0.06, 0.38, 1.42, 'boolean'),
    ('面包,牛奶→鸡蛋', 0.07, 0.48, 1.35, 'boolean'),
    ('水果→酸奶', 0.05, 0.33, 1.25, 'boolean'),
    ('蔬菜→水果', 0.09, 0.41, 0.95, 'boolean'),
    ('可乐→薯片', 0.04, 0.55, 1.95, 'boolean'),
    # 含量化维规则（年龄、收入等）
    ('年龄[25-35]→有机食品', 0.06, 0.72, 3.20, 'quantitative'),
    ('收入[高]→进口红酒', 0.03, 0.68, 4.15, 'quantitative'),
    ('年龄[45-60],收入[中]→保健品', 0.04, 0.58, 3.50, 'quantitative'),
    ('年龄[18-25]→速食面', 0.07, 0.65, 2.80, 'quantitative'),
    ('收入[低]→特价商品', 0.11, 0.48, 1.85, 'quantitative'),
    ('年龄[30-45]→儿童用品', 0.05, 0.55, 2.45, 'quantitative'),
    # 混合规则
    ('年龄[25-35],面包→牛奶', 0.04, 0.62, 2.10, 'mixed'),
    ('收入[高],红酒→奶酪', 0.02, 0.75, 4.80, 'mixed'),
    ('年龄[45-60],啤酒→花生', 0.03, 0.58, 3.10, 'mixed'),
    ('收入[中],牛奶→面包', 0.06, 0.50, 1.65, 'mixed'),
]

fig, ax = plt.subplots(figsize=(14, 9))

# 按类型分组绘制
type_config = {
    'boolean':      {'color': COLORS['blue'],   'marker': 'o', 'label': '纯布尔维规则'},
    'quantitative': {'color': COLORS['red'],    'marker': 's', 'label': '含量化维规则'},
    'mixed':        {'color': COLORS['purple'], 'marker': 'D', 'label': '混合维规则'},
}

for rtype, cfg in type_config.items():
    subset = [(r[0], r[1], r[2], r[3]) for r in rules if r[4] == rtype]
    names = [s[0] for s in subset]
    supps = [s[1] for s in subset]
    confs = [s[2] for s in subset]
    lifts = [s[3] for s in subset]

    sizes = [c * 500 + 50 for c in confs]
    ax.scatter(supps, lifts, c=cfg['color'], s=sizes, marker=cfg['marker'],
               alpha=0.75, edgecolors='white', linewidths=1.5,
               label=cfg['label'], zorder=3)

    # 标注高提升度规则
    for name, sup, conf, lift in subset:
        if lift > 2.5:
            ax.annotate(name, xy=(sup, lift),
                        xytext=(sup + 0.008, lift + 0.15),
                        fontsize=12, fontweight='bold', color=cfg['color'],
                        arrowprops=dict(arrowstyle='->', color=cfg['color'],
                                        lw=1.2, alpha=0.7),
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec=cfg['color'], alpha=0.9))

# 阈值线
ax.axhline(y=1.0, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)
ax.text(0.155, 1.05, '提升度=1（独立基线）', fontsize=13, color=COLORS['gray'])

ax.axhline(y=2.5, color=COLORS['green'], ls=':', lw=1.5, alpha=0.5)
ax.text(0.155, 2.55, '高提升度阈值', fontsize=13, color=COLORS['green'])

# 区域标注
ax.text(0.13, 0.7, '弱关联区',
        fontsize=14, color=COLORS['gray'], style='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f8fafc', ec=COLORS['gray']))
ax.text(0.13, 4.5, '强关联区',
        fontsize=14, color=COLORS['green'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f0fdf4', ec=COLORS['green']))

ax.set_xlabel('支持度（Support）', fontsize=15)
ax.set_ylabel('提升度（Lift）', fontsize=15)
ax.set_title('多维关联规则发现：支持度-提升度分布', fontsize=18)
ax.legend(fontsize=14, loc='upper right',
          title='规则类型', title_fontsize=14)

# 右侧文字注释
ax.text(0.98, 0.02,
        '气泡大小 ∝ 置信度',
        transform=ax.transAxes, fontsize=13,
        ha='right', va='bottom', color=COLORS['gray'],
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['gray']))

ax.set_xlim(0.01, 0.18)
ax.set_ylim(0.5, 5.5)

plt.tight_layout()
save_fig(fig, __file__, 'fig6_5_03_multidim_rules')
