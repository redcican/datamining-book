"""
图 13.5.2　评分矩阵稀疏性可视化
展示 Top-100 活跃用户 × Top-100 热门电影的评分子矩阵
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_movielens import load_movielens

ratings, movies = load_movielens()

# ── 构建评分矩阵 ─────────────────────────────────────────
n_users = ratings["user_id"].nunique()
n_items = ratings["item_id"].nunique()
n_ratings = len(ratings)

# 选取 Top-100 活跃用户和 Top-100 热门电影
top_users = (ratings.groupby("user_id").size()
             .sort_values(ascending=False).head(100).index)
top_items = (ratings.groupby("item_id").size()
             .sort_values(ascending=False).head(100).index)

sub = ratings[ratings["user_id"].isin(top_users) &
              ratings["item_id"].isin(top_items)]

# 构建子矩阵
user_map = {u: i for i, u in enumerate(sorted(top_users))}
item_map = {m: j for j, m in enumerate(sorted(top_items))}
matrix = np.zeros((100, 100))
for _, row in sub.iterrows():
    i = user_map[row["user_id"]]
    j = item_map[row["item_id"]]
    matrix[i, j] = row["rating"]

fill_rate_sub = (matrix > 0).sum() / matrix.size
fill_rate_full = n_ratings / (n_users * n_items)

print("=== 评分矩阵稀疏性 ===")
print(f"  完整矩阵: {n_users} × {n_items} = {n_users*n_items:,} 单元格")
print(f"  已评分:   {n_ratings:,} ({fill_rate_full:.1%})")
print(f"  缺失:     {n_users*n_items - n_ratings:,} ({1-fill_rate_full:.1%})")
print(f"\n  Top-100 子矩阵填充率: {fill_rate_sub:.1%}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                gridspec_kw={"width_ratios": [1.3, 1]})

# (a) 评分矩阵热图（子矩阵）
masked = np.ma.masked_where(matrix == 0, matrix)
im = ax1.imshow(masked, cmap="YlOrRd", aspect="auto",
                interpolation="nearest", vmin=1, vmax=5)
ax1.set_xlabel("电影编号 (Top-100 热门)")
ax1.set_ylabel("用户编号 (Top-100 活跃)")
ax1.set_title(f"(a) 评分矩阵 (Top-100 × Top-100, 填充率 {fill_rate_sub:.0%})",
              fontweight="bold")
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label("评分", fontsize=10)
cbar.set_ticks([1, 2, 3, 4, 5])
ax1.spines["top"].set_visible(True)
ax1.spines["right"].set_visible(True)

# (b) 全矩阵 spy plot（二值化：有/无评分）
full_matrix = np.zeros((n_users, n_items), dtype=np.int8)
for _, row in ratings.iterrows():
    full_matrix[row["user_id"] - 1, row["item_id"] - 1] = 1

# 下采样显示（取每 N 行/列的切片）
step_u, step_i = max(1, n_users // 200), max(1, n_items // 200)
spy_matrix = full_matrix[::step_u, ::step_i]
ax2.imshow(spy_matrix, cmap="Blues", aspect="auto",
           interpolation="nearest", vmin=0, vmax=1)
ax2.set_xlabel(f"电影 (每 {step_i} 列采样)")
ax2.set_ylabel(f"用户 (每 {step_u} 行采样)")
ax2.set_title(f"(b) 完整矩阵稀疏模式 ({n_users}×{n_items}, "
              f"稀疏度 {1-fill_rate_full:.1%})",
              fontweight="bold")
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(True)

plt.tight_layout(w_pad=2)
save_fig(fig, __file__, "fig13_5_02_sparsity")
