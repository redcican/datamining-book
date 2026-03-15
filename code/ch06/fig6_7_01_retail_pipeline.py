"""
图 6.7.1　零售购物篮分析完整管线
左：管线各阶段数据规模变化  右：高价值规则支持度-提升度分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ── 左图：管线漏斗图 ─────────────────────────────────────
stages = ['原始事务\n(50万条)', '频繁项集\n(3000个)', '候选规则\n(800条)',
          '过滤后规则\n(120条)', '可操作建议\n(20条)']
values = [500000, 3000, 800, 120, 20]
colors = [COLORS['blue'], COLORS['teal'], COLORS['green'],
          COLORS['orange'], COLORS['red']]

# 绘制水平条形图（漏斗效果）
y_pos = np.arange(len(stages))[::-1]
# 归一化宽度（对数尺度映射到可视化宽度）
log_vals = np.log10(np.array(values))
widths = log_vals / max(log_vals)

for i, (y, w, v, c, s) in enumerate(zip(y_pos, widths, values, colors, stages)):
    bar = ax1.barh(y, w, height=0.7, color=c, alpha=0.85,
                   edgecolor='white', linewidth=2)
    # 标注数量
    ax1.text(w + 0.02, y, f'{v:,}', fontsize=15, fontweight='bold',
             va='center', color=c)
    # 缩减率
    if i > 0:
        reduction = (1 - values[i] / values[i-1]) * 100
        ax1.annotate(f'↓ {reduction:.0f}%',
                     xy=(0, y + 0.4), fontsize=12, color=COLORS['gray'],
                     fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(stages, fontsize=14)
ax1.set_xlabel('相对规模（对数）', fontsize=14)
ax1.set_title('管线各阶段数据规模', fontsize=17)
ax1.set_xlim(0, 1.25)
ax1.set_xticks([])
ax1.spines['bottom'].set_visible(False)

# 阶段标签
stage_names = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5-6']
for y, name in zip(y_pos, stage_names):
    ax1.text(-0.02, y, name, fontsize=11, color=COLORS['gray'],
             ha='right', va='center', style='italic')

# ── 右图：高价值规则散点图 ────────────────────────────────
# 模拟最终 20 条可操作规则
rule_names = [
    '啤酒→薯片', '面包→牛奶', '牛奶→鸡蛋', '酸奶→水果',
    '薯片→可乐', '面包→鸡蛋', '肉类→蔬菜', '蔬菜→调料',
    '啤酒→可乐', '牛奶→面包', '水果→酸奶', '鸡蛋→面包',
    '肉类→调料', '可乐→薯片', '面包→酸奶', '牛奶→酸奶',
    '啤酒→花生', '蔬菜→肉类', '水果→蔬菜', '调料→肉类'
]
supps = np.array([0.08, 0.25, 0.12, 0.06, 0.05, 0.15, 0.07, 0.04,
                  0.04, 0.22, 0.05, 0.13, 0.03, 0.04, 0.03, 0.04,
                  0.02, 0.06, 0.08, 0.03])
lifts = np.array([2.5, 1.4, 1.8, 2.2, 2.1, 1.5, 1.9, 2.4,
                  1.7, 1.3, 2.1, 1.4, 2.6, 2.0, 1.6, 1.5,
                  3.0, 1.8, 1.1, 2.3])
confs = np.array([0.72, 0.62, 0.55, 0.68, 0.65, 0.48, 0.58, 0.70,
                  0.52, 0.58, 0.66, 0.45, 0.73, 0.63, 0.42, 0.46,
                  0.78, 0.55, 0.38, 0.68])

# 分类：捆绑促销 vs 货架摆放 vs 交叉推荐
categories = []
cat_colors = []
for s, l in zip(supps, lifts):
    if l > 2.0 and s > 0.05:
        categories.append('捆绑促销')
        cat_colors.append(COLORS['red'])
    elif s > 0.10:
        categories.append('货架摆放')
        cat_colors.append(COLORS['blue'])
    elif l > 2.0:
        categories.append('交叉推荐')
        cat_colors.append(COLORS['orange'])
    else:
        categories.append('监控关注')
        cat_colors.append(COLORS['gray'])

sizes = confs * 400 + 30
ax2.scatter(supps, lifts, c=cat_colors, s=sizes, alpha=0.8,
            edgecolors='white', linewidths=1.5, zorder=3)

# 阈值线
ax2.axhline(y=2.0, color=COLORS['green'], ls=':', lw=1.5, alpha=0.5)
ax2.axvline(x=0.10, color=COLORS['green'], ls=':', lw=1.5, alpha=0.5)

# 标注高价值规则
for i, (s, l, name) in enumerate(zip(supps, lifts, rule_names)):
    if l > 2.3 or (s > 0.15 and l > 1.3):
        ax2.annotate(name, xy=(s, l), xytext=(s+0.01, l+0.1),
                     fontsize=12, fontweight='bold', color=cat_colors[i],
                     arrowprops=dict(arrowstyle='->', color=cat_colors[i],
                                     lw=1, alpha=0.6))

# 区域标注
ax2.text(0.15, 2.8, '捆绑促销区', fontsize=14, color=COLORS['red'],
         fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
         fc='#fef2f2', ec=COLORS['red']))
ax2.text(0.17, 1.15, '货架摆放区', fontsize=14, color=COLORS['blue'],
         fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
         fc='#eff6ff', ec=COLORS['blue']))
ax2.text(0.01, 2.8, '交叉推荐区', fontsize=14, color=COLORS['orange'],
         fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
         fc='#fff7ed', ec=COLORS['orange']))

ax2.set_xlabel('支持度', fontsize=15)
ax2.set_ylabel('提升度', fontsize=15)
ax2.set_title('高价值规则分布', fontsize=17)
ax2.text(0.98, 0.02, '气泡大小 ∝ 置信度', transform=ax2.transAxes,
         fontsize=12, ha='right', color=COLORS['gray'],
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=COLORS['gray']))

plt.suptitle('零售购物篮分析管线（定义 6.27）', fontsize=19, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_7_01_retail_pipeline')
