"""
图 6.2.2　最小支持度阈值对 Apriori 算法的影响
左：频繁项集数量随 minsup 变化
右：算法运行时间随 minsup 变化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── 1. 生成合成购物篮数据 ─────────────────────────────────────
items_all = [f'I{i}' for i in range(1, 21)]  # 20 items
n_trans = 500

patterns = {
    'p1': (['I1', 'I2', 'I3', 'I4'], 0.40),
    'p2': (['I5', 'I6', 'I7'], 0.35),
    'p3': (['I8', 'I9', 'I10'], 0.25),
    'p4': (['I3', 'I5', 'I11'], 0.20),
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
        basket.update(np.random.choice(items_all, size=min(n_rand, 4), replace=False))
    if len(basket) > 0:
        transactions.append(frozenset(basket))

# ── 2. Apriori 实现 ──────────────────────────────────────────
def apriori(transactions, minsup_ratio):
    n = len(transactions)
    minsup_count = minsup_ratio * n

    # 1-itemsets
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    L1 = {frozenset({item}): count for item, count in item_counts.items()
           if count >= minsup_count}

    all_freq = dict(L1)
    Lk = L1
    k = 2

    while len(Lk) > 1:
        # Generate candidates
        items_in_Lk = list(Lk.keys())
        candidates = set()
        for i in range(len(items_in_Lk)):
            for j in range(i + 1, len(items_in_Lk)):
                union = items_in_Lk[i] | items_in_Lk[j]
                if len(union) == k:
                    # Prune: check all (k-1)-subsets are frequent
                    all_sub_freq = True
                    for item in union:
                        subset = union - frozenset({item})
                        if subset not in Lk:
                            all_sub_freq = False
                            break
                    if all_sub_freq:
                        candidates.add(union)

        if not candidates:
            break

        # Count support
        cand_counts = {c: 0 for c in candidates}
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    cand_counts[c] += 1

        Lk = {c: count for c, count in cand_counts.items()
               if count >= minsup_count}
        all_freq.update(Lk)
        k += 1

    return all_freq

# ── 3. 不同 minsup 下的实验 ──────────────────────────────────
minsup_values = np.arange(0.02, 0.42, 0.02)
freq_counts = []
freq_by_level = {1: [], 2: [], 3: [], 4: []}
run_times = []

for ms in minsup_values:
    t0 = time.time()
    freq = apriori(transactions, ms)
    t1 = time.time()

    freq_counts.append(len(freq))
    run_times.append((t1 - t0) * 1000)  # ms

    # 按层统计
    level_counts = {}
    for itemset in freq:
        k = len(itemset)
        level_counts[k] = level_counts.get(k, 0) + 1
    for lv in freq_by_level:
        freq_by_level[lv].append(level_counts.get(lv, 0))

# ── 4. 绘图 ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左：频繁项集数量（堆叠面积图）
colors_stack = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
labels_stack = ['1-项集', '2-项集', '3-项集', '4-项集']

bottoms = np.zeros(len(minsup_values))
for lv in [1, 2, 3, 4]:
    vals = np.array(freq_by_level[lv])
    ax1.bar(minsup_values, vals, bottom=bottoms, width=0.016,
            color=colors_stack[lv-1], label=labels_stack[lv-1],
            edgecolor='white', linewidth=0.5, alpha=0.85)
    bottoms += vals

ax1.set_xlabel('最小支持度阈值（minsup）', fontsize=12)
ax1.set_ylabel('频繁项集数量', fontsize=12)
ax1.set_title('频繁项集数量随 minsup 变化', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')

# 标注关键点
peak_idx = np.argmax(freq_counts)
ax1.annotate(f'minsup={minsup_values[1]:.2f}\n共 {freq_counts[1]} 个',
             xy=(minsup_values[1], freq_counts[1]),
             xytext=(minsup_values[1] + 0.08, freq_counts[1] * 0.9),
             fontsize=10, color=COLORS['red'],
             arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

# 右：运行时间
ax2.plot(minsup_values, run_times, 'o-', color=COLORS['red'],
         markersize=6, linewidth=2, label='运行时间')
ax2.fill_between(minsup_values, run_times, alpha=0.15, color=COLORS['red'])
ax2.set_xlabel('最小支持度阈值（minsup）', fontsize=12)
ax2.set_ylabel('运行时间（毫秒）', fontsize=12)
ax2.set_title('算法运行时间随 minsup 变化', fontsize=14)

# 标注趋势
ax2.annotate('minsup 越低\n搜索空间越大\n耗时越长',
             xy=(minsup_values[1], run_times[1]),
             xytext=(0.15, max(run_times) * 0.8),
             fontsize=11, color=COLORS['red'],
             arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef2f2',
                       edgecolor=COLORS['red'], alpha=0.9))

plt.suptitle('最小支持度阈值对 Apriori 算法性能的影响', fontsize=15, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_2_02_minsup_impact')
