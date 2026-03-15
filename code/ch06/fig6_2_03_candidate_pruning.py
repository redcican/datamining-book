"""
图 6.2.3　Apriori 候选剪枝效果
左：各层候选数 vs 频繁项集数对比
右：剪枝比例随层级的变化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import comb
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成合成数据 ──────────────────────────────────────────
items_all = [f'I{i}' for i in range(1, 16)]  # 15 items
n_trans = 600

patterns = {
    'p1': (['I1', 'I2', 'I3', 'I4', 'I5'], 0.40),
    'p2': (['I3', 'I6', 'I7', 'I8'], 0.35),
    'p3': (['I9', 'I10', 'I11'], 0.25),
    'p4': (['I1', 'I6', 'I12'], 0.20),
}

transactions = []
for _ in range(n_trans):
    basket = set()
    for name, (group, prob) in patterns.items():
        if np.random.random() < prob:
            for item in group:
                if np.random.random() < 0.7:
                    basket.add(item)
    n_rand = np.random.poisson(1.5)
    if n_rand > 0:
        basket.update(np.random.choice(items_all, size=min(n_rand, 3), replace=False))
    if len(basket) > 0:
        transactions.append(frozenset(basket))

# ── 2. 运行 Apriori 并记录各层信息 ─────────────────────────────
minsup_ratio = 0.08
minsup_count = minsup_ratio * len(transactions)

# Level 1
item_counts = {}
for t in transactions:
    for item in t:
        item_counts[item] = item_counts.get(item, 0) + 1

L1 = {frozenset({item}): count for item, count in item_counts.items()
       if count >= minsup_count}

# 记录各层数据
levels = []
n_items = len(items_all)

# Level 1
levels.append({
    'k': 1,
    'n_possible': n_items,
    'n_candidates': n_items,  # 所有单项
    'n_frequent': len(L1),
})

Lk = L1
k = 2

while len(Lk) > 1 and k <= 6:
    # 不剪枝的候选数（所有可能的 k-组合）
    n_possible = comb(n_items, k)

    # Generate candidates with Apriori pruning
    items_in_Lk = list(Lk.keys())
    candidates = set()
    candidates_before_prune = set()

    for i in range(len(items_in_Lk)):
        for j in range(i + 1, len(items_in_Lk)):
            union = items_in_Lk[i] | items_in_Lk[j]
            if len(union) == k:
                candidates_before_prune.add(union)
                # Prune
                all_sub_freq = True
                for item in union:
                    subset = union - frozenset({item})
                    if subset not in Lk:
                        all_sub_freq = False
                        break
                if all_sub_freq:
                    candidates.add(union)

    if not candidates:
        levels.append({
            'k': k,
            'n_possible': n_possible,
            'n_candidates': len(candidates_before_prune),
            'n_frequent': 0,
        })
        break

    # Count
    cand_counts = {c: 0 for c in candidates}
    for t in transactions:
        for c in candidates:
            if c.issubset(t):
                cand_counts[c] += 1

    Lk = {c: count for c, count in cand_counts.items()
           if count >= minsup_count}

    levels.append({
        'k': k,
        'n_possible': n_possible,
        'n_candidates': len(candidates),
        'n_frequent': len(Lk),
    })

    k += 1

# ── 3. 绘图 ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ks = [l['k'] for l in levels]
n_possible = [l['n_possible'] for l in levels]
n_candidates = [l['n_candidates'] for l in levels]
n_frequent = [l['n_frequent'] for l in levels]

x = np.arange(len(ks))
width = 0.25

# 左：三组条形图
bars1 = ax1.bar(x - width, n_possible, width, label='理论搜索空间 $\\binom{n}{k}$',
                color=COLORS['gray'], alpha=0.5, edgecolor='white')
bars2 = ax1.bar(x, n_candidates, width, label='Apriori 候选数 $|C_k|$',
                color=COLORS['orange'], alpha=0.85, edgecolor='white')
bars3 = ax1.bar(x + width, n_frequent, width, label='频繁项集数 $|L_k|$',
                color=COLORS['blue'], alpha=0.85, edgecolor='white')

ax1.set_xlabel('项集层级 $k$', fontsize=12)
ax1.set_ylabel('数量', fontsize=12)
ax1.set_title('各层搜索空间压缩效果', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{k}-项集' for k in ks], fontsize=11)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_yscale('log')

# 在条形上方标注数值
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=8)

# 右：剪枝比率
pruning_ratio_vs_possible = []
pruning_ratio_vs_candidates = []
for l in levels:
    if l['n_possible'] > 0:
        ratio1 = 1 - l['n_candidates'] / l['n_possible']
        pruning_ratio_vs_possible.append(ratio1 * 100)
    else:
        pruning_ratio_vs_possible.append(0)
    if l['n_candidates'] > 0:
        ratio2 = 1 - l['n_frequent'] / l['n_candidates']
        pruning_ratio_vs_candidates.append(ratio2 * 100)
    else:
        pruning_ratio_vs_candidates.append(0)

ax2.plot(x, pruning_ratio_vs_possible, 'o-', color=COLORS['red'],
         markersize=10, linewidth=2.5,
         label='Apriori 剪枝率\n（相对理论空间）')
ax2.plot(x, pruning_ratio_vs_candidates, 's--', color=COLORS['blue'],
         markersize=10, linewidth=2.5,
         label='支持度过滤率\n（候选 → 频繁）')

ax2.fill_between(x, pruning_ratio_vs_possible, alpha=0.1, color=COLORS['red'])
ax2.fill_between(x, pruning_ratio_vs_candidates, alpha=0.1, color=COLORS['blue'])

ax2.set_xlabel('项集层级 $k$', fontsize=12)
ax2.set_ylabel('压缩比率（%）', fontsize=12)
ax2.set_title('搜索空间压缩比率', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{k}-项集' for k in ks], fontsize=11)
ax2.set_ylim(-5, 105)
ax2.legend(fontsize=10, loc='center right')

# 标注数值
for i, (r1, r2) in enumerate(zip(pruning_ratio_vs_possible,
                                   pruning_ratio_vs_candidates)):
    if r1 > 0:
        ax2.text(i, r1 + 3, f'{r1:.0f}%', ha='center', fontsize=9,
                 color=COLORS['red'], fontweight='bold')
    ax2.text(i, r2 - 5, f'{r2:.0f}%', ha='center', fontsize=9,
             color=COLORS['blue'], fontweight='bold')

plt.suptitle(f'Apriori 候选剪枝效果（minsup = {minsup_ratio}，'
             f'{len(items_all)} 项，{len(transactions)} 条事务）',
             fontsize=15, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_2_03_candidate_pruning')
