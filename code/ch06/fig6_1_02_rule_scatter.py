"""
图 6.1.2　关联规则的支持度-置信度-提升度散点图
点的颜色和大小映射提升度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import combinations
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成合成购物篮数据 ─────────────────────────────────
items_list = ['面包', '牛奶', '鸡蛋', '啤酒', '薯片', '可乐',
              '酸奶', '水果', '蔬菜', '肉类']
n_trans = 800
patterns = {
    '早餐': (['面包', '牛奶', '鸡蛋'], 0.45),
    '零食': (['啤酒', '薯片', '可乐'], 0.30),
    '健康': (['水果', '蔬菜', '酸奶'], 0.25),
}

transactions = []
for _ in range(n_trans):
    basket = set()
    for name, (group, prob) in patterns.items():
        if np.random.random() < prob:
            for item in group:
                if np.random.random() < 0.75:
                    basket.add(item)
    n_rand = np.random.poisson(1)
    if n_rand > 0:
        basket.update(np.random.choice(items_list, size=min(n_rand, 3), replace=False))
    if len(basket) > 0:
        transactions.append(basket)

n = len(transactions)

# ── 2. 计算支持度 ────────────────────────────────────────
item_sup = {}
for item in items_list:
    item_sup[item] = sum(1 for t in transactions if item in t) / n

pair_sup = {}
for i1, i2 in combinations(items_list, 2):
    count = sum(1 for t in transactions if {i1, i2}.issubset(t))
    pair_sup[(i1, i2)] = count / n

# ── 3. 生成规则 ──────────────────────────────────────────
rules = []
for (i1, i2), sup in pair_sup.items():
    if sup > 0.01:
        # i1 → i2
        conf1 = sup / item_sup[i1] if item_sup[i1] > 0 else 0
        lift1 = conf1 / item_sup[i2] if item_sup[i2] > 0 else 0
        rules.append((i1, i2, sup, conf1, lift1))
        # i2 → i1
        conf2 = sup / item_sup[i2] if item_sup[i2] > 0 else 0
        lift2 = conf2 / item_sup[i1] if item_sup[i1] > 0 else 0
        rules.append((i2, i1, sup, conf2, lift2))

# ── 4. 绘图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

sups = [r[2] for r in rules]
confs = [r[3] for r in rules]
lifts = [r[4] for r in rules]

norm = Normalize(vmin=min(lifts), vmax=max(lifts))
sizes = [max(20, l * 80) for l in lifts]

sc = ax.scatter(sups, confs, c=lifts, s=sizes, cmap='RdYlBu_r',
                norm=norm, alpha=0.7, edgecolors='white', linewidths=0.5)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('提升度 (Lift)', fontsize=12)

# 阈值线
minsup_th = 0.05
minconf_th = 0.4
ax.axvline(x=minsup_th, color=COLORS['red'], linestyle='--', linewidth=1.5,
           alpha=0.6, label=f'minsup = {minsup_th}')
ax.axhline(y=minconf_th, color=COLORS['red'], linestyle=':', linewidth=1.5,
           alpha=0.6, label=f'minconf = {minconf_th}')
ax.axhline(y=1.0, color=COLORS['gray'], linestyle='-', linewidth=0.5, alpha=0.3)

# 标注 top 5 规则（高提升度）
top_rules = sorted(rules, key=lambda r: -r[4])[:5]
for ante, cons, sup, conf, lift in top_rules:
    ax.annotate(f'{ante}→{cons}\nlift={lift:.2f}',
                xy=(sup, conf), fontsize=8,
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

ax.set_xlabel('支持度（Support）', fontsize=12)
ax.set_ylabel('置信度（Confidence）', fontsize=12)
ax.set_title('关联规则的支持度-置信度-提升度分布', fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(-0.01, max(sups) * 1.1)
ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
save_fig(fig, __file__, 'fig6_1_02_rule_scatter')
