"""
图 6.3.3　FP-Growth vs Apriori 性能对比
左：不同 minsup 下运行时间对比
右：不同数据规模下运行时间对比
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import time
from shared.plot_config import apply_style, save_fig, PALETTE, COLORS

apply_style()
np.random.seed(42)

# ── Apriori 实现 ──────────────────────────────────────────────
def apriori_simple(transactions, minsup_count):
    n = len(transactions)
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1
    L1 = {frozenset({item}): c for item, c in item_counts.items()
           if c >= minsup_count}
    all_freq = dict(L1)
    Lk = L1
    k = 2
    while len(Lk) > 1:
        items_list = list(Lk.keys())
        candidates = set()
        for i in range(len(items_list)):
            for j in range(i+1, len(items_list)):
                union = items_list[i] | items_list[j]
                if len(union) == k:
                    all_sub = True
                    for item in union:
                        if union - frozenset({item}) not in Lk:
                            all_sub = False
                            break
                    if all_sub:
                        candidates.add(union)
        if not candidates:
            break
        cand_counts = {c: 0 for c in candidates}
        for t in transactions:
            t_set = frozenset(t)
            for c in candidates:
                if c.issubset(t_set):
                    cand_counts[c] += 1
        Lk = {c: cnt for c, cnt in cand_counts.items() if cnt >= minsup_count}
        all_freq.update(Lk)
        k += 1
    return all_freq

# ── FP-Growth 实现 ────────────────────────────────────────────
class FPNode:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

def build_fptree(transactions, minsup_count):
    freq = {}
    for t in transactions:
        for item in t:
            freq[item] = freq.get(item, 0) + 1
    header = {item: [cnt, None] for item, cnt in freq.items()
              if cnt >= minsup_count}
    if not header:
        return None, header
    root = FPNode()
    for t in transactions:
        filtered = sorted(
            [item for item in t if item in header],
            key=lambda x: (-header[x][0], x)
        )
        node = root
        for item in filtered:
            if item in node.children:
                node.children[item].count += 1
            else:
                new_node = FPNode(item, 1, node)
                node.children[item] = new_node
                if header[item][1] is None:
                    header[item][1] = new_node
                else:
                    curr = header[item][1]
                    while curr.next:
                        curr = curr.next
                    curr.next = new_node
            node = node.children[item]
    return root, header

def mine_tree(header, minsup_count, suffix, results):
    sorted_items = sorted(header.items(), key=lambda x: x[1][0])
    for item, (count, node) in sorted_items:
        pattern = suffix + [item]
        results.append(frozenset(pattern))
        paths = []
        n = node
        while n:
            prefix = []
            p = n.parent
            while p and p.item is not None:
                prefix.append(p.item)
                p = p.parent
            if prefix:
                paths.append((prefix, n.count))
            n = n.next
        if not paths:
            continue
        cond_freq = {}
        for path, cnt in paths:
            for it in path:
                cond_freq[it] = cond_freq.get(it, 0) + cnt
        cond_header = {it: [cnt, None] for it, cnt in cond_freq.items()
                       if cnt >= minsup_count}
        if not cond_header:
            continue
        cond_root = FPNode()
        for path, cnt in paths:
            filtered = sorted(
                [it for it in path if it in cond_header],
                key=lambda x: (-cond_header[x][0], x)
            )
            nd = cond_root
            for it in filtered:
                if it in nd.children:
                    nd.children[it].count += cnt
                else:
                    new_nd = FPNode(it, cnt, nd)
                    nd.children[it] = new_nd
                    if cond_header[it][1] is None:
                        cond_header[it][1] = new_nd
                    else:
                        cur = cond_header[it][1]
                        while cur.next:
                            cur = cur.next
                        cur.next = new_nd
                nd = nd.children[it]
        mine_tree(cond_header, minsup_count, pattern, results)

def fpgrowth_simple(transactions, minsup_count):
    root, header = build_fptree(transactions, minsup_count)
    if root is None:
        return []
    results = []
    mine_tree(header, minsup_count, [], results)
    return results

# ── 数据生成 ──────────────────────────────────────────────────
def gen_data(n_trans, n_items=20):
    items = list(range(n_items))
    patterns = [
        (items[:5], 0.40),
        (items[3:8], 0.35),
        (items[7:11], 0.25),
        (items[0:3] + items[10:13], 0.20),
    ]
    transactions = []
    for _ in range(n_trans):
        basket = set()
        for group, prob in patterns:
            if np.random.random() < prob:
                for item in group:
                    if np.random.random() < 0.7:
                        basket.add(item)
        nr = np.random.poisson(1.5)
        if nr > 0:
            basket.update(np.random.choice(items, size=min(nr, 4), replace=False))
        if basket:
            transactions.append(frozenset(basket))
    return transactions

# ── 实验 1：不同 minsup ──────────────────────────────────────
trans_fixed = gen_data(1500)
minsup_values = [0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03]
times_ap_ms = []
times_fp_ms = []

for ms in minsup_values:
    msc = int(ms * len(trans_fixed))
    t0 = time.time()
    apriori_simple(trans_fixed, msc)
    times_ap_ms.append((time.time() - t0) * 1000)
    t0 = time.time()
    fpgrowth_simple(trans_fixed, msc)
    times_fp_ms.append((time.time() - t0) * 1000)

# ── 实验 2：不同数据规模 ─────────────────────────────────────
n_values = [200, 500, 1000, 1500, 2000, 3000]
times_ap_n = []
times_fp_n = []
ms_fixed = 0.08

for nv in n_values:
    trans_n = gen_data(nv)
    msc = int(ms_fixed * len(trans_n))
    t0 = time.time()
    apriori_simple(trans_n, msc)
    times_ap_n.append((time.time() - t0) * 1000)
    t0 = time.time()
    fpgrowth_simple(trans_n, msc)
    times_fp_n.append((time.time() - t0) * 1000)

# ── 绘图 ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左：不同 minsup
ax1.plot(minsup_values, times_ap_ms, 'o-', color=COLORS['red'],
         markersize=8, linewidth=2.5, label='Apriori')
ax1.plot(minsup_values, times_fp_ms, 's-', color=COLORS['blue'],
         markersize=8, linewidth=2.5, label='FP-Growth')
ax1.fill_between(minsup_values, times_ap_ms, times_fp_ms,
                 alpha=0.1, color=COLORS['green'])
ax1.set_xlabel('最小支持度（minsup）', fontsize=12)
ax1.set_ylabel('运行时间（毫秒）', fontsize=12)
ax1.set_title(f'不同 minsup 下的运行时间（n={len(trans_fixed)}）', fontsize=14)
ax1.legend(fontsize=12)
ax1.invert_xaxis()

# 标注加速比
for i, ms in enumerate(minsup_values):
    if times_fp_ms[i] > 0:
        speedup = times_ap_ms[i] / times_fp_ms[i]
        if speedup > 1.5 and i % 2 == 0:
            ax1.annotate(f'{speedup:.1f}x',
                         xy=(ms, (times_ap_ms[i] + times_fp_ms[i]) / 2),
                         fontsize=9, ha='center', color=COLORS['green'],
                         fontweight='bold')

# 右：不同数据规模
ax2.plot(n_values, times_ap_n, 'o-', color=COLORS['red'],
         markersize=8, linewidth=2.5, label='Apriori')
ax2.plot(n_values, times_fp_n, 's-', color=COLORS['blue'],
         markersize=8, linewidth=2.5, label='FP-Growth')
ax2.fill_between(n_values, times_ap_n, times_fp_n,
                 alpha=0.1, color=COLORS['green'])
ax2.set_xlabel('事务数量 $n$', fontsize=12)
ax2.set_ylabel('运行时间（毫秒）', fontsize=12)
ax2.set_title(f'不同数据规模下的运行时间（minsup={ms_fixed}）', fontsize=14)
ax2.legend(fontsize=12)

plt.suptitle('FP-Growth vs Apriori 性能对比（定理 6.8）', fontsize=15, y=1.02)
plt.tight_layout()
save_fig(fig, __file__, 'fig6_3_03_fpgrowth_vs_apriori')
