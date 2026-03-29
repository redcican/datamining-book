"""
图 13.1.2　关联规则支持度-置信度散点图（提升度着色）
基于 UCI Online Retail 真实 Apriori 结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_retail import load_clean, build_basket

# ── 1. 加载清洗后数据并挖掘关联规则 ───────────────────────
df, stats = load_clean()
basket, freq_items, rules, transactions = build_basket(df, min_support=0.02)

print(f"清洗后: {stats['clean_rows']:,} 行, {stats['clean_invoices']:,} 笔交易")
print(f"事务数: {len(transactions):,}, "
      f"平均商品数: {np.mean([len(t) for t in transactions]):.1f}")
print(f"事务矩阵: {basket.shape}")
print(f"\n频繁项集数: {len(freq_items)}")
print(f"  1-项集: {(freq_items['length']==1).sum()}")
print(f"  2-项集: {(freq_items['length']==2).sum()}")
print(f"  3-项集: {(freq_items['length']==3).sum()}")

rules_filtered = rules[rules["lift"] >= 1.5].copy()
print(f"\n关联规则数 (lift≥1.5): {len(rules_filtered)}")
print(f"\nTop-10 规则 (按提升度):")
for _, r in rules_filtered.head(10).iterrows():
    ant = ", ".join(sorted(r["antecedents"]))
    con = ", ".join(sorted(r["consequents"]))
    print(f"  {ant:45s} → {con:45s}  "
          f"sup={r['support']:.3f} conf={r['confidence']:.2f} "
          f"lift={r['lift']:.2f}")

high_conf = rules[(rules["confidence"] >= 0.6) & (rules["lift"] >= 2.0)]
print(f"\n高置信度规则 (conf≥0.6, lift≥2.0): {len(high_conf)} 条")
for _, r in high_conf.head(5).iterrows():
    ant = ", ".join(sorted(r["antecedents"]))
    con = ", ".join(sorted(r["consequents"]))
    print(f"  {ant} → {con}")
    print(f"    置信度={r['confidence']:.2f}, 提升度={r['lift']:.2f}")

# ── 2. 绘制散点图 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

sups = rules["support"].values
confs = rules["confidence"].values
lifts = rules["lift"].values

norm = Normalize(vmin=1.0, vmax=min(lifts.max(), 8.0))
sizes = np.clip(lifts * 30, 20, 300)

sc = ax.scatter(sups, confs, c=lifts, s=sizes, cmap="RdYlBu_r",
                norm=norm, alpha=0.7, edgecolors="white", linewidths=0.5)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("提升度 (Lift)", fontsize=12)

# 阈值线
ax.axvline(x=0.03, color=COLORS["blue"], linestyle="--", linewidth=1.5,
           alpha=0.6, label="min_support = 3%")
ax.axhline(y=0.6, color=COLORS["green"], linestyle="--", linewidth=1.5,
           alpha=0.6, label="min_confidence = 60%")

# 黄金区域标注
golden = rules[(rules["support"] >= 0.03) &
               (rules["confidence"] >= 0.6) &
               (rules["lift"] >= 2.0)]
if len(golden) > 0:
    ax.fill_between([0.03, sups.max() * 1.05], 0.6, 1.05,
                    alpha=0.08, color=COLORS["orange"],
                    label=f"黄金区域 ({len(golden)} 条规则)")

# 标注 Top-5 高提升度规则
top5 = rules.nlargest(5, "lift")
for idx, (_, r) in enumerate(top5.iterrows()):
    ant = ", ".join(sorted(r["antecedents"]))[:25]
    con = ", ".join(sorted(r["consequents"]))[:25]
    offset_y = 12 + (idx % 3) * 8
    offset_x = 10 if idx % 2 == 0 else -80
    ax.annotate(
        f"{ant}→\n{con}\nlift={r['lift']:.1f}",
        xy=(r["support"], r["confidence"]),
        fontsize=7.5,
        xytext=(offset_x, offset_y), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )

ax.set_xlabel("支持度 (Support)", fontsize=12)
ax.set_ylabel("置信度 (Confidence)", fontsize=12)
ax.set_title("关联规则的支持度-置信度-提升度分布（UCI Online Retail）",
             fontsize=14)
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim(-0.005, sups.max() * 1.1)
ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
save_fig(fig, __file__, "fig13_1_02_rule_scatter")
