"""
图 6.7.2　医疗症状-疾病关联分析
左：症状共现网络  右：按疾病分组的规则置信度箱线图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()
np.random.seed(42)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ── 左图：症状共现网络 ────────────────────────────────────
G = nx.Graph()

# 定义症状节点和疾病归属
symptoms = {
    # 呼吸系统
    '发热': {'freq': 0.35, 'disease': '呼吸系统'},
    '咳嗽': {'freq': 0.30, 'disease': '呼吸系统'},
    '胸痛': {'freq': 0.12, 'disease': '呼吸系统'},
    '呼吸困难': {'freq': 0.08, 'disease': '呼吸系统'},
    # 消化系统
    '恶心': {'freq': 0.20, 'disease': '消化系统'},
    '腹痛': {'freq': 0.18, 'disease': '消化系统'},
    '腹泻': {'freq': 0.15, 'disease': '消化系统'},
    # 神经系统
    '头痛': {'freq': 0.25, 'disease': '神经系统'},
    '眩晕': {'freq': 0.10, 'disease': '神经系统'},
    '视觉模糊': {'freq': 0.06, 'disease': '神经系统'},
    # 骨骼肌肉
    '关节痛': {'freq': 0.15, 'disease': '骨骼肌肉'},
    '疲劳': {'freq': 0.28, 'disease': '骨骼肌肉'},
}

disease_colors = {
    '呼吸系统': COLORS['red'],
    '消化系统': COLORS['green'],
    '神经系统': COLORS['purple'],
    '骨骼肌肉': COLORS['orange'],
}

for symptom, info in symptoms.items():
    G.add_node(symptom, freq=info['freq'], disease=info['disease'])

# 定义共现关系 (症状对, 提升度)
edges = [
    ('发热', '咳嗽', 3.2), ('发热', '胸痛', 2.8), ('咳嗽', '胸痛', 2.5),
    ('咳嗽', '呼吸困难', 2.1), ('发热', '呼吸困难', 1.8),
    ('恶心', '腹痛', 3.5), ('腹痛', '腹泻', 2.9), ('恶心', '腹泻', 2.3),
    ('头痛', '眩晕', 3.0), ('头痛', '视觉模糊', 2.6), ('眩晕', '视觉模糊', 2.2),
    ('关节痛', '疲劳', 2.4),
    # 跨系统弱关联
    ('发热', '疲劳', 1.3), ('头痛', '恶心', 1.5), ('发热', '头痛', 1.2),
    ('疲劳', '头痛', 1.1),
]

for s1, s2, lift in edges:
    G.add_edge(s1, s2, weight=lift)

pos = nx.spring_layout(G, k=2.8, seed=42)
node_colors = [disease_colors[symptoms[n]['disease']] for n in G.nodes()]
node_sizes = [symptoms[n]['freq'] * 3500 + 300 for n in G.nodes()]
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       alpha=0.85, edgecolors='white', linewidths=2, ax=ax1)
nx.draw_networkx_labels(G, pos, font_size=16, font_family='SimHei', ax=ax1)
nx.draw_networkx_edges(G, pos, width=[w*0.8 for w in edge_weights],
                       alpha=0.4, edge_color='#94a3b8', ax=ax1)

# 图例
for disease, color in disease_colors.items():
    ax1.scatter([], [], c=color, s=200, label=disease, edgecolors='white')
# 扩展左侧留白，避免图例遮挡节点
xlim = ax1.get_xlim()
ax1.set_xlim(xlim[0] - 0.35 * (xlim[1] - xlim[0]), xlim[1])
ylim = ax1.get_ylim()
ax1.set_ylim(ylim[0] - 0.15 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))
ax1.legend(fontsize=15, loc='lower left', title='疾病类别', title_fontsize=15,
           framealpha=0.9)
ax1.set_title('症状共现网络', fontsize=19)
ax1.text(0.02, 0.98, '节点大小 ∝ 症状频率\n边粗细 ∝ 提升度',
         transform=ax1.transAxes, fontsize=15, va='top', color=COLORS['gray'],
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['gray']))
ax1.axis('off')

# ── 右图：按疾病分组的规则置信度箱线图 ─────────────────────
diseases = ['肺炎', '支气管炎', '胃肠炎', '偏头痛', '类风湿\n关节炎', '流感']
disease_cat = ['呼吸系统', '呼吸系统', '消化系统', '神经系统', '骨骼肌肉', '呼吸系统']

# 模拟每种疾病的关联规则置信度分布
conf_data = [
    np.random.beta(8, 3, 15),    # 肺炎 (高置信度)
    np.random.beta(6, 4, 12),    # 支气管炎
    np.random.beta(7, 3, 10),    # 胃肠炎
    np.random.beta(9, 2, 8),     # 偏头痛 (很高)
    np.random.beta(5, 4, 6),     # 类风湿关节炎
    np.random.beta(4, 3, 20),    # 流感 (较低，因为症状不特异)
]

bp = ax2.boxplot(conf_data, tick_labels=diseases, patch_artist=True,
                 widths=0.6, medianprops=dict(color='white', lw=2))

box_colors = [disease_colors[cat] for cat in disease_cat]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 阈值线
ax2.axhline(y=0.7, color=COLORS['gray'], ls='--', lw=1.5, alpha=0.5)
ax2.text(6.6, 0.71, 'minconf=0.7', fontsize=15, color=COLORS['gray'])

# 标注中位数
for i, data in enumerate(conf_data, 1):
    median = np.median(data)
    ax2.text(i, median + 0.03, f'{median:.2f}', ha='center',
             fontsize=15, fontweight='bold', color=box_colors[i-1])

ax2.set_ylabel('规则置信度', fontsize=17)
ax2.set_title('按疾病分组的规则置信度', fontsize=19)
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=14)

plt.suptitle('医疗症状-疾病关联分析（定义 6.28）', fontsize=20, y=1.01)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_7_02_medical_association')
