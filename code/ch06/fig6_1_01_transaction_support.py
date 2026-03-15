"""
图 6.1.1　事务数据库的矩阵表示与支持度分布
左：事务-商品热力图
右：各项支持度条形图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 示例事务数据 ──────────────────────────────────────
items = ['面包', '牛奶', '鸡蛋', '啤酒', '尿布', '可乐', '薯片', '酸奶']
transactions = [
    {'面包', '牛奶', '鸡蛋'},
    {'面包', '牛奶', '啤酒', '尿布'},
    {'牛奶', '鸡蛋', '啤酒'},
    {'面包', '牛奶', '鸡蛋', '啤酒'},
    {'面包', '鸡蛋'},
    {'可乐', '薯片', '啤酒'},
    {'面包', '牛奶', '酸奶'},
    {'牛奶', '鸡蛋', '酸奶', '面包'},
    {'可乐', '薯片'},
    {'面包', '牛奶', '鸡蛋', '可乐'},
]

n = len(transactions)
m = len(items)

# ── 2. 构建事务矩阵 ──────────────────────────────────────
matrix = np.zeros((n, m), dtype=int)
for i, t in enumerate(transactions):
    for j, item in enumerate(items):
        if item in t:
            matrix[i, j] = 1

# 计算支持度
support = matrix.sum(axis=0) / n
minsup = 0.3

# ── 3. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                gridspec_kw={'width_ratios': [1.3, 1]})

# 左：热力图
im = ax1.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(m))
ax1.set_xticklabels(items, fontsize=11, rotation=30, ha='right')
ax1.set_yticks(range(n))
ax1.set_yticklabels([f'T{i+1}' for i in range(n)], fontsize=10)
ax1.set_title('事务-商品矩阵', fontsize=13)
ax1.set_xlabel('商品')
ax1.set_ylabel('事务 ID')
# 在每个格子中标注 0/1
for i in range(n):
    for j in range(m):
        ax1.text(j, i, str(matrix[i, j]), ha='center', va='center',
                 fontsize=9, color='white' if matrix[i, j] else 'gray')

# 右：支持度条形图
colors = [COLORS['blue'] if s >= minsup else COLORS['gray'] for s in support]
bars = ax2.barh(range(m), support, color=colors, edgecolor='white', height=0.6)
ax2.axvline(x=minsup, color=COLORS['red'], linestyle='--', linewidth=2,
            label=f'minsup = {minsup}')
ax2.set_yticks(range(m))
ax2.set_yticklabels(items, fontsize=11)
ax2.set_xlabel('支持度')
ax2.set_title('各商品（1-项集）支持度', fontsize=13)
ax2.legend(fontsize=10)
ax2.invert_yaxis()
# 标注支持度数值
for i, s in enumerate(support):
    ax2.text(s + 0.01, i, f'{s:.2f}', va='center', fontsize=10,
             color=COLORS['red'] if s < minsup else 'black')

plt.tight_layout()
save_fig(fig, __file__, 'fig6_1_01_transaction_support')
