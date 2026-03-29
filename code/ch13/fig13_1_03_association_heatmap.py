"""
图 13.1.3　高频商品关联提升度热力图
取 Top-12 高频商品，展示两两之间的提升度
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_retail import load_clean

# ── 1. 加载清洗后数据 ──────────────────────────────────────
df, stats = load_clean()

# ── 2. 构建共现矩阵（Top-12 商品） ────────────────────────
top_n = 12
top_items = df["Description"].value_counts().head(top_n).index.tolist()
df_top = df[df["Description"].isin(top_items)]

# 事务级别：每笔交易包含哪些 top 商品
invoice_items = (df_top.groupby("InvoiceNo")["Description"]
                 .apply(set).reset_index())

n_trans = df["InvoiceNo"].nunique()

# 单品支持度
item_support = {}
for item in top_items:
    count = invoice_items["Description"].apply(lambda s: item in s).sum()
    item_support[item] = count / n_trans

# 两两提升度矩阵
lift_matrix = pd.DataFrame(np.ones((top_n, top_n)),
                           index=top_items, columns=top_items)

for i, item_a in enumerate(top_items):
    for j, item_b in enumerate(top_items):
        if i == j:
            lift_matrix.iloc[i, j] = np.nan
            continue
        co_count = invoice_items["Description"].apply(
            lambda s: item_a in s and item_b in s
        ).sum()
        co_support = co_count / n_trans
        expected = item_support[item_a] * item_support[item_b]
        lift_matrix.iloc[i, j] = co_support / expected if expected > 0 else 0

print("=== Top-12 商品提升度矩阵 ===")
print(f"提升度范围: {np.nanmin(lift_matrix.values):.2f} – "
      f"{np.nanmax(lift_matrix.values):.2f}")
print(f"lift > 2.0 的商品对: "
      f"{(lift_matrix.values[~np.isnan(lift_matrix.values)] > 2.0).sum() // 2}")

# 打印 Top-10 高提升度商品对
pairs = []
for i in range(top_n):
    for j in range(i + 1, top_n):
        val = lift_matrix.iloc[i, j]
        if not np.isnan(val):
            pairs.append((top_items[i], top_items[j], val))
pairs.sort(key=lambda x: -x[2])
print("\nTop-10 高提升度商品对:")
for a, b, lift in pairs[:10]:
    print(f"  {a[:35]:35s} × {b[:35]:35s}  lift={lift:.2f}")

# ── 3. 绘制热力图 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

# 截断商品名用于显示
short_names = [d[:28] + "…" if len(d) > 28 else d for d in top_items]

# 使用 TwoSlopeNorm 让 lift=1 居中
vmin = max(0, np.nanmin(lift_matrix.values) - 0.1)
vmax = np.nanmax(lift_matrix.values)
norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=min(vmax, 6.0))

im = ax.imshow(lift_matrix.values, cmap="RdYlBu_r", norm=norm, aspect="auto")
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("提升度 (Lift)", fontsize=12)

# 在每个格子中标注数值
for i in range(top_n):
    for j in range(top_n):
        val = lift_matrix.iloc[i, j]
        if np.isnan(val):
            continue
        color = "white" if val > 3.0 or val < 0.5 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold" if val > 2.0 else "normal")

ax.set_xticks(range(top_n))
ax.set_yticks(range(top_n))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
ax.set_title("Top-12 高频商品关联提升度热力图", fontsize=14, pad=15)

# 恢复网格（imshow 默认无网格）
ax.set_xticks(np.arange(-0.5, top_n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, top_n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", bottom=False, left=False)

# 添加 spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)

plt.tight_layout()
save_fig(fig, __file__, "fig13_1_03_association_heatmap")
