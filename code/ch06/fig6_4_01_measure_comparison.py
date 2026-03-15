"""
图 6.4.1　多种兴趣度量对 Top-20 项对的排序对比热力图
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
n_trans = 1000
transactions = []
for _ in range(n_trans):
    basket = set()
    if np.random.random() < 0.25:
        basket.update(['啤酒', '薯片'])
    if np.random.random() < 0.40:
        basket.update(['面包', '牛奶'])
        if np.random.random() < 0.5:
            basket.add('鸡蛋')
    if np.random.random() < 0.20:
        basket.update(['酸奶', '水果'])
    for item, prob in [('水果', 0.5), ('蔬菜', 0.45),
                       ('鸡蛋', 0.25), ('可乐', 0.20)]:
        if np.random.random() < prob:
            basket.add(item)
    for item in items:
        if item not in basket and np.random.random() < 0.06:
            basket.add(item)
    if basket:
        transactions.append(basket)
n = len(transactions)

# ── 2. 计算兴趣度量 ──────────────────────────────────────────
records = []
for a, b in combinations(items, 2):
    f11 = sum(1 for t in transactions if a in t and b in t)
    f10 = sum(1 for t in transactions if a in t and b not in t)
    f01 = sum(1 for t in transactions if a not in t and b in t)
    pA, pB, pAB = (f11+f10)/n, (f11+f01)/n, f11/n
    if pA == 0 or pB == 0 or pAB == 0:
        continue
    records.append({
        'pair': f'{a}-{b}',
        'supp': pAB,
        'conf': pAB / pA,
        'lift': pAB / (pA * pB),
        'cosine': pAB / np.sqrt(pA * pB),
        'kulc': 0.5 * (pAB/pA + pAB/pB),
        'all_conf': pAB / max(pA, pB),
    })

# ── 3. 取 Top-20 (按 lift 取并集) ─────────────────────────────
import pandas as pd
df = pd.DataFrame(records)
metrics = ['lift', 'cosine', 'kulc', 'all_conf', 'conf', 'supp']
top_indices = set()
for m in metrics:
    top_indices.update(df.nlargest(8, m).index.tolist())
top_indices = sorted(top_indices, key=lambda i: -df.loc[i, 'lift'])[:20]
sub = df.loc[top_indices].reset_index(drop=True)

# ── 4. 绘图 ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# 归一化到 [0, 1]
matrix = np.zeros((len(sub), len(metrics)))
for j, m in enumerate(metrics):
    vals = sub[m].values
    vmin, vmax = vals.min(), vals.max()
    matrix[:, j] = (vals - vmin) / (vmax - vmin + 1e-10)

im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(['提升度', '余弦', 'Kulc', '全置信度', '置信度', '支持度'],
                    fontsize=14)
ax.set_yticks(range(len(sub)))
ax.set_yticklabels(sub['pair'].values, fontsize=12)
ax.set_xlabel('兴趣度量', fontsize=14)
ax.set_ylabel('项对', fontsize=14)

# 标注数值
for i in range(len(sub)):
    for j, m in enumerate(metrics):
        val = sub.iloc[i][m]
        color = 'white' if matrix[i, j] > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=12, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('归一化得分', fontsize=13)

ax.set_title('多种兴趣度量对项对的排序对比', fontsize=18)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_4_01_measure_comparison')
