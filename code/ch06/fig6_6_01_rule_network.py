"""
图 6.6.1　关联规则网络可视化
左：关联规则有向图  右：项共现网络（含社区检测）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()
np.random.seed(42)

# ── 生成合成数据 ───────────────────────────────────────────
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

# 计算度量
item_supp = {item: sum(1 for t in transactions if item in t)/n for item in items}
pair_data = {}
for a, b in combinations(items, 2):
    pAB = sum(1 for t in transactions if a in t and b in t) / n
    pA, pB = item_supp[a], item_supp[b]
    if pAB < 0.02 or pA == 0 or pB == 0: continue
    lift = pAB / (pA * pB)
    kulc = 0.5 * (pAB/pA + pAB/pB)
    pair_data[(a, b)] = {'pAB': pAB, 'lift': lift, 'kulc': kulc,
                          'conf_ab': pAB/pA, 'conf_ba': pAB/pB}

# ── 绘图 ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ── 左图：关联规则有向图 ──────────────────────────────────
G_dir = nx.DiGraph()
for item in items:
    G_dir.add_node(item)

rules_for_graph = []
for (a, b), data in pair_data.items():
    if data['conf_ab'] >= 0.3 and data['lift'] > 1.1:
        rules_for_graph.append((a, b, data['lift'], data['conf_ab']))
    if data['conf_ba'] >= 0.3 and data['lift'] > 1.1:
        rules_for_graph.append((b, a, data['lift'], data['conf_ba']))

for ant, con, lift, conf in rules_for_graph:
    if not G_dir.has_edge(ant, con) or G_dir[ant][con]['weight'] < lift:
        G_dir.add_edge(ant, con, weight=lift, conf=conf)

# 只保留有边的节点
isolated = [n for n in G_dir.nodes() if G_dir.degree(n) == 0]
G_dir.remove_nodes_from(isolated)

pos1 = nx.spring_layout(G_dir, k=3.0, seed=42)
node_sizes = [item_supp.get(n, 0.1) * 4000 + 200 for n in G_dir.nodes()]
edge_weights = [G_dir[u][v]['weight'] for u, v in G_dir.edges()]

nx.draw_networkx_nodes(G_dir, pos1, node_size=node_sizes,
                       node_color=COLORS['blue'], alpha=0.8, ax=ax1)
nx.draw_networkx_labels(G_dir, pos1, font_size=14, font_family='SimHei', ax=ax1)
edges = nx.draw_networkx_edges(G_dir, pos1, width=[w*1.2 for w in edge_weights],
                               edge_color=edge_weights, edge_cmap=plt.cm.Reds,
                               alpha=0.7, arrows=True, arrowsize=18,
                               arrowstyle='-|>', ax=ax1,
                               connectionstyle='arc3,rad=0.1')

ax1.set_title('关联规则有向图', fontsize=17)
ax1.text(0.02, 0.02, '节点大小 ∝ 支持度\n边粗细 ∝ 提升度',
         transform=ax1.transAxes, fontsize=13, color=COLORS['gray'],
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['gray']))
ax1.axis('off')

# ── 右图：项共现网络（含社区检测）─────────────────────────
G_co = nx.Graph()
for item in items:
    G_co.add_node(item)

for (a, b), data in pair_data.items():
    if data['kulc'] > 0.2:
        G_co.add_edge(a, b, weight=data['kulc'])

# 社区检测
communities = nx.community.greedy_modularity_communities(G_co)
community_colors = {}
comm_palette = [COLORS['blue'], COLORS['red'], COLORS['green'],
                COLORS['orange'], COLORS['purple']]
for i, comm in enumerate(communities):
    for node in comm:
        community_colors[node] = comm_palette[i % len(comm_palette)]

pos2 = nx.spring_layout(G_co, k=2.5, seed=42)
node_colors = [community_colors.get(n, COLORS['gray']) for n in G_co.nodes()]
node_sizes = [item_supp.get(n, 0.1) * 4000 + 200 for n in G_co.nodes()]
edge_weights_co = [G_co[u][v]['weight'] * 4 for u, v in G_co.edges()]

nx.draw_networkx_nodes(G_co, pos2, node_size=node_sizes,
                       node_color=node_colors, alpha=0.85, ax=ax2,
                       edgecolors='white', linewidths=2)
nx.draw_networkx_labels(G_co, pos2, font_size=14, font_family='SimHei', ax=ax2)
nx.draw_networkx_edges(G_co, pos2, width=edge_weights_co,
                       alpha=0.4, edge_color='#94a3b8', ax=ax2)

ax2.set_title('项共现网络（社区检测）', fontsize=17)
# 社区图例
for i, comm in enumerate(communities):
    comm_items = sorted(comm)[:3]
    label = ', '.join(comm_items) + ('...' if len(comm) > 3 else '')
    ax2.scatter([], [], c=comm_palette[i], s=120, label=f'社区{i+1}: {label}')
ax2.legend(fontsize=12, loc='lower left', framealpha=0.9)
ax2.axis('off')

plt.suptitle('关联规则网络可视化（定义 6.23–6.24）', fontsize=19, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_6_01_rule_network')
