"""
图 6.6.3　规则多维度量可视化
左：支持度-置信度散点图  右：平行坐标图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import combinations
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 生成数据 ──────────────────────────────────────────────
items = ['面包', '牛奶', '鸡蛋', '啤酒', '薯片', '可乐',
         '酸奶', '水果', '蔬菜', '肉类']
n_trans = 2000
transactions = []
for _ in range(n_trans):
    basket = set()
    if np.random.random() < 0.35: basket.update(['面包', '牛奶'])
    if np.random.random() < 0.25: basket.update(['啤酒', '薯片'])
    if np.random.random() < 0.20: basket.update(['面包', '牛奶', '鸡蛋'])
    if np.random.random() < 0.18: basket.update(['酸奶', '水果'])
    if np.random.random() < 0.12: basket.update(['肉类', '蔬菜'])
    for item, prob in [('水果',0.45),('蔬菜',0.40),('肉类',0.30),
                       ('鸡蛋',0.25),('可乐',0.20)]:
        if np.random.random() < prob: basket.add(item)
    for item in items:
        if item not in basket and np.random.random() < 0.04:
            basket.add(item)
    if basket: transactions.append(basket)
n = len(transactions)

# 计算规则
item_supp = {item: sum(1 for t in transactions if item in t)/n for item in items}
rules = []
for a, b in combinations(items, 2):
    pAB = sum(1 for t in transactions if a in t and b in t) / n
    pA, pB = item_supp[a], item_supp[b]
    if pAB < 0.015 or pA == 0 or pB == 0: continue
    lift = pAB / (pA * pB)
    kulc = 0.5 * (pAB/pA + pAB/pB)
    all_conf = pAB / max(pA, pB)
    for ant, con, conf in [(a, b, pAB/pA), (b, a, pAB/pB)]:
        if conf >= 0.2 and lift > 0.8:
            rules.append({
                'name': f'{ant}→{con}', 'ant': ant, 'con': con,
                'supp': pAB, 'conf': conf, 'lift': lift,
                'kulc': kulc, 'all_conf': all_conf
            })

# ── 绘图 ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ── 左图：支持度-置信度散点图 ─────────────────────────────
supps = np.array([r['supp'] for r in rules])
confs = np.array([r['conf'] for r in rules])
lifts = np.array([r['lift'] for r in rules])
n_items_rule = np.array([2 for _ in rules])  # 都是2-项规则

sc = ax1.scatter(supps, confs, c=lifts, s=lifts*80+30,
                 cmap='RdYlBu_r', alpha=0.75, edgecolors='white',
                 linewidths=1, vmin=0.6, vmax=max(lifts))
cbar = plt.colorbar(sc, ax=ax1, shrink=0.85)
cbar.set_label('提升度', fontsize=14)

# 阈值线
ax1.axhline(y=0.3, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)
ax1.axvline(x=0.05, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)
ax1.text(0.06, 0.31, 'minconf=0.3', fontsize=12, color=COLORS['gray'])
ax1.text(0.051, 0.15, 'minsup=0.05', fontsize=12, color=COLORS['gray'],
         rotation=90)

# 高价值区域
ax1.fill_between([0.05, supps.max()+0.02], 0.3, confs.max()+0.05,
                 alpha=0.05, color=COLORS['green'])
ax1.text(0.15, 0.75, '高价值区域', fontsize=14, color=COLORS['green'],
         fontweight='bold')

# 标注 Top 规则
top_idx = np.argsort(-lifts)[:5]
for idx in top_idx:
    r = rules[idx]
    ax1.annotate(r['name'], xy=(r['supp'], r['conf']),
                 xytext=(r['supp']+0.01, r['conf']+0.03),
                 fontsize=12, fontweight='bold', color=COLORS['red'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'],
                                 lw=1, alpha=0.6))

ax1.set_xlabel('支持度', fontsize=15)
ax1.set_ylabel('置信度', fontsize=15)
ax1.set_title('支持度-置信度散点图', fontsize=17)
ax1.set_xlim(-0.01, supps.max()+0.03)
ax1.set_ylim(0.1, confs.max()+0.08)

# ── 右图：平行坐标图 ─────────────────────────────────────
metrics = ['supp', 'conf', 'lift', 'kulc', 'all_conf']
metric_labels = ['支持度', '置信度', '提升度', 'Kulc', '全置信度']
n_metrics = len(metrics)

# 归一化到 [0, 1]
data_matrix = np.array([[r[m] for m in metrics] for r in rules])
data_min = data_matrix.min(axis=0)
data_max = data_matrix.max(axis=0)
data_norm = (data_matrix - data_min) / (data_max - data_min + 1e-10)

# 按提升度分为高/低两组
lift_threshold = np.median(lifts)
high_lift_mask = lifts >= lift_threshold

# 绘制平行坐标
xs = np.arange(n_metrics)
for i in range(len(rules)):
    color = COLORS['red'] if high_lift_mask[i] else COLORS['blue']
    alpha = 0.6 if high_lift_mask[i] else 0.15
    lw = 2.0 if high_lift_mask[i] else 0.8
    ax2.plot(xs, data_norm[i], color=color, alpha=alpha, lw=lw)

# 轴标签和刻度
for j, (label, mn, mx) in enumerate(zip(metric_labels, data_min, data_max)):
    ax2.axvline(x=j, color='#cbd5e1', lw=1, zorder=0)
    ax2.text(j, -0.08, label, ha='center', fontsize=14,
             transform=ax2.get_xaxis_transform())
    ax2.text(j, 1.03, f'{mx:.2f}', ha='center', fontsize=11,
             color=COLORS['gray'], transform=ax2.get_xaxis_transform())
    ax2.text(j, -0.03, f'{mn:.2f}', ha='center', fontsize=11,
             color=COLORS['gray'], transform=ax2.get_xaxis_transform())

ax2.set_xlim(-0.3, n_metrics - 0.7)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS['red'], lw=2.5, label=f'高提升度 (≥{lift_threshold:.2f})'),
    Line2D([0], [0], color=COLORS['blue'], lw=1.5, alpha=0.4, label=f'低提升度 (<{lift_threshold:.2f})')
]
ax2.legend(handles=legend_elements, fontsize=13, loc='upper right')
ax2.set_title('平行坐标图', fontsize=17)

plt.suptitle('规则多维度量可视化', fontsize=19, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_6_03_rule_scatter')
