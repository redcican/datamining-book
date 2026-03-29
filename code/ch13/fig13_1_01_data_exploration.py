"""
图 13.1.1　UCI Online Retail 数据探索
(a) 数据清洗各步骤保留行数  (b) Top-15 高频商品
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE

apply_style()

from _load_retail import load_raw

# ── 1. 加载原始数据并逐步清洗，记录每步行数 ────────────────
df = load_raw()
steps = [("原始数据", len(df))]

df = df.dropna(subset=["InvoiceNo", "Description"])
steps.append(("移除缺失值", len(df)))

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
steps.append(("移除退货 (C-)", len(df)))

df = df[df["Quantity"] > 0]
steps.append(("移除异常数量", len(df)))

df = df[df["Country"] == "United Kingdom"]
steps.append(("筛选英国市场", len(df)))

print("=== 数据清洗统计 ===")
for name, count in steps:
    print(f"  {name}: {count:,} 行")
print(f"  保留比例: {steps[-1][1]/steps[0][1]:.1%}")

# ── 2. Top-15 高频商品 ──────────────────────────────────────
top15 = (df["Description"].value_counts().head(15)
         .sort_values(ascending=True))

print("\n=== Top-15 高频商品 ===")
for desc, cnt in top15.items():
    print(f"  {cnt:5d}  {desc}")

# ── 3. 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                gridspec_kw={"width_ratios": [1, 1.3]})

# --- Panel (a): 清洗漏斗 ---
labels = [s[0] for s in steps]
counts = [s[1] for s in steps]
colors_bar = [COLORS["gray"]] + [COLORS["blue"]] * 3 + [COLORS["green"]]
bars = ax1.barh(labels, counts, color=colors_bar, edgecolor="white",
                linewidth=0.5, height=0.6)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_width() + 3000, bar.get_y() + bar.get_height() / 2,
             f"{count:,}", va="center", fontsize=10, color=COLORS["gray"])
ax1.set_xlabel("记录数")
ax1.set_title("(a) 数据清洗流程", fontsize=13, fontweight="bold")
ax1.set_xlim(0, max(counts) * 1.25)
ax1.invert_yaxis()

# --- Panel (b): Top-15 高频商品 ---
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(top15))]
bars2 = ax2.barh(range(len(top15)), top15.values, color=bar_colors,
                 edgecolor="white", linewidth=0.5, height=0.65)
ax2.set_yticks(range(len(top15)))
# 截断过长的商品名
short_labels = [d[:30] + "…" if len(d) > 30 else d for d in top15.index]
ax2.set_yticklabels(short_labels, fontsize=9)
for bar, count in zip(bars2, top15.values):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
             f"{count:,}", va="center", fontsize=9, color=COLORS["gray"])
ax2.set_xlabel("出现次数")
ax2.set_title("(b) Top-15 高频商品（英国市场）", fontsize=13, fontweight="bold")
ax2.set_xlim(0, top15.max() * 1.2)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_1_01_data_exploration")
