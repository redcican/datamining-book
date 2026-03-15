"""
图 6.6.2　关联强度矩阵热力图
左：原始顺序  右：层次聚类重排
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
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

# ── 计算提升度矩阵 ────────────────────────────────────────
item_supp = {item: sum(1 for t in transactions if item in t)/n for item in items}
n_items = len(items)
M = np.ones((n_items, n_items))

for i, j in combinations(range(n_items), 2):
    a, b = items[i], items[j]
    pAB = sum(1 for t in transactions if a in t and b in t) / n
    pA, pB = item_supp[a], item_supp[b]
    if pA > 0 and pB > 0 and pAB > 0:
        lift = pAB / (pA * pB)
        M[i, j] = lift
        M[j, i] = lift

# ── 层次聚类重排 ──────────────────────────────────────────
# 将提升度矩阵转为距离矩阵
D = 1.0 / (M + 0.01)  # 高提升度 → 低距离
np.fill_diagonal(D, 0)
# 提取上三角为压缩距离向量
from scipy.spatial.distance import squareform
condensed = squareform(D)
Z = linkage(condensed, method='ward')
order = leaves_list(Z)
M_reordered = M[np.ix_(order, order)]
items_reordered = [items[i] for i in order]

# ── 绘图 ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

vmin, vmax = 0.5, min(M.max(), 3.0)
cmap = 'RdYlBu_r'

# 左图：原始顺序
im1 = ax1.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax1.set_xticks(range(n_items))
ax1.set_xticklabels(items, fontsize=13, rotation=45, ha='right')
ax1.set_yticks(range(n_items))
ax1.set_yticklabels(items, fontsize=13)
ax1.set_title('原始顺序', fontsize=17)

# 标注高提升度
for i in range(n_items):
    for j in range(n_items):
        if i != j and M[i, j] > 1.3:
            ax1.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if M[i,j] > 1.8 else 'black')

# 右图：聚类重排
im2 = ax2.imshow(M_reordered, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
ax2.set_xticks(range(n_items))
ax2.set_xticklabels(items_reordered, fontsize=13, rotation=45, ha='right')
ax2.set_yticks(range(n_items))
ax2.set_yticklabels(items_reordered, fontsize=13)
ax2.set_title('层次聚类重排', fontsize=17)

for i in range(n_items):
    for j in range(n_items):
        if i != j and M_reordered[i, j] > 1.3:
            ax2.text(j, i, f'{M_reordered[i,j]:.1f}', ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if M_reordered[i,j] > 1.8 else 'black')

# 标注聚类块
# 找聚类边界
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=3, criterion='maxclust')
cluster_order = [clusters[i] for i in order]
boundaries = []
for k in range(1, len(cluster_order)):
    if cluster_order[k] != cluster_order[k-1]:
        boundaries.append(k - 0.5)

for b in boundaries:
    ax2.axhline(y=b, color=COLORS['green'], lw=2.5, alpha=0.8)
    ax2.axvline(x=b, color=COLORS['green'], lw=2.5, alpha=0.8)

# colorbar 放在最右侧
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label('提升度（Lift）', fontsize=14)

plt.suptitle('关联强度矩阵热力图（定义 6.25）', fontsize=19, y=1.01)
save_fig(fig, __file__, 'fig6_6_02_matrix_heatmap')
