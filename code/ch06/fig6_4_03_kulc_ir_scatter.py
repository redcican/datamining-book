"""
图 6.4.3　Kulczynski-IR 散点图
x 轴 Kulczynski（关联强度），y 轴 IR（不平衡比）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成合成数据 ──────────────────────────────────────────
items = ['面包', '牛奶', '鸡蛋', '啤酒', '薯片', '可乐',
         '酸奶', '水果', '蔬菜', '肉类', '调料', '饮料']
n_trans = 1500
transactions = []
for _ in range(n_trans):
    basket = set()
    if np.random.random() < 0.28:
        basket.update(['啤酒', '薯片'])
    if np.random.random() < 0.42:
        basket.update(['面包', '牛奶'])
        if np.random.random() < 0.5:
            basket.add('鸡蛋')
    if np.random.random() < 0.22:
        basket.update(['酸奶', '水果'])
    for item, prob in [('水果', 0.50), ('蔬菜', 0.45),
                       ('鸡蛋', 0.28), ('可乐', 0.22),
                       ('肉类', 0.35)]:
        if np.random.random() < prob:
            basket.add(item)
    for item in items:
        if item not in basket and np.random.random() < 0.05:
            basket.add(item)
    if basket:
        transactions.append(basket)
n = len(transactions)

# ── 2. 计算所有项对的 Kulc 和 IR ──────────────────────────────
pairs, kulcs, irs, lifts, supps = [], [], [], [], []
for a, b in combinations(items, 2):
    f11 = sum(1 for t in transactions if a in t and b in t)
    f10 = sum(1 for t in transactions if a in t and b not in t)
    f01 = sum(1 for t in transactions if a not in t and b in t)
    pA, pB, pAB = (f11+f10)/n, (f11+f01)/n, f11/n
    if pA == 0 or pB == 0 or pAB == 0:
        continue
    kulc = 0.5 * (pAB/pA + pAB/pB)
    denom = pA + pB - pAB
    ir = abs(pA - pB) / denom if denom > 0 else 0
    lift = pAB / (pA * pB)
    pairs.append(f'{a}-{b}')
    kulcs.append(kulc)
    irs.append(ir)
    lifts.append(lift)
    supps.append(pAB)

kulcs = np.array(kulcs)
irs = np.array(irs)
lifts = np.array(lifts)
supps = np.array(supps)

# ── 3. 绘图 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

sizes = supps * 3000 + 30
sc = ax.scatter(kulcs, irs, c=lifts, s=sizes, cmap='RdYlBu_r',
                alpha=0.7, edgecolors='white', linewidths=0.8,
                vmin=0.5, vmax=max(lifts))
cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
cbar.set_label('提升度（Lift）', fontsize=14)

# 阈值线
ax.axvline(x=0.5, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)
ax.axhline(y=0.3, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)

# 区域标注
ax.text(0.78, 0.05, '对称强关联\n(最有价值)', fontsize=14,
        color=COLORS['green'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f0fdf4', ec=COLORS['green']))
ax.text(0.08, 0.75, '不对称弱关联', fontsize=14,
        color=COLORS['red'],
        bbox=dict(boxstyle='round,pad=0.3', fc='#fef2f2', ec=COLORS['red']))
ax.text(0.08, 0.05, '弱关联', fontsize=14,
        color=COLORS['gray'],
        bbox=dict(boxstyle='round,pad=0.3', fc='#f8fafc', ec=COLORS['gray']))

# 标注 Top 项对
top_kulc_idx = np.argsort(-kulcs)[:6]
offsets_y = [0.04, -0.05, 0.04, -0.05, 0.04, -0.05]
for rank, idx in enumerate(top_kulc_idx):
    ax.annotate(pairs[idx], xy=(kulcs[idx], irs[idx]),
                xytext=(kulcs[idx]+0.02, irs[idx]+offsets_y[rank]),
                fontsize=12, fontweight='bold', color=COLORS['blue'],
                arrowprops=dict(arrowstyle='->', color=COLORS['blue'],
                                lw=1, alpha=0.6))

ax.set_xlabel('Kulczynski 度量（关联强度）', fontsize=14)
ax.set_ylabel('不平衡比 IR（对称性）', fontsize=14)
ax.set_title('Kulczynski-IR 散点图（定义 6.16–6.17）', fontsize=18)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
save_fig(fig, __file__, 'fig6_4_03_kulc_ir_scatter')
