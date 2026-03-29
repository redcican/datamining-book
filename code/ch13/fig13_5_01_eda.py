"""
图 13.5.1　MovieLens 100K 数据探索
(a) 评分分布  (b) 用户活跃度分布
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

# ── 基本统计 ──────────────────────────────────────────────
n_users = ratings["user_id"].nunique()
n_items = ratings["item_id"].nunique()
n_ratings = len(ratings)
sparsity = 1 - n_ratings / (n_users * n_items)

print("=== MovieLens 100K 基本统计 ===")
print(f"  评分数: {n_ratings:,}")
print(f"  用户数: {n_users}")
print(f"  电影数: {n_items}")
print(f"  稀疏度: {sparsity:.1%}")
print(f"  评分范围: {ratings['rating'].min()}-{ratings['rating'].max()}")
print(f"  平均评分: {ratings['rating'].mean():.2f}")
print(f"  评分中位数: {ratings['rating'].median():.1f}")

# 用户和电影的统计
user_counts = ratings.groupby("user_id").size()
item_counts = ratings.groupby("item_id").size()
print(f"\n  每用户平均评分数: {user_counts.mean():.1f}")
print(f"  每用户评分数范围: {user_counts.min()}-{user_counts.max()}")
print(f"  每电影平均评分数: {item_counts.mean():.1f}")
print(f"  每电影评分数范围: {item_counts.min()}-{item_counts.max()}")

# Top-5 电影
top_movies = ratings.groupby("item_id").agg(
    count=("rating", "size"),
    mean_rating=("rating", "mean")
).sort_values("count", ascending=False).head(5)
print("\n  最多评分的 5 部电影:")
for item_id, row in top_movies.iterrows():
    title = movies[movies["item_id"] == item_id]["title"].values[0]
    print(f"    {title[:35]:<36s} "
          f"评分数={int(row['count']):>3d}  "
          f"均分={row['mean_rating']:.2f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) 评分分布
rating_counts = ratings["rating"].value_counts().sort_index()
bars = ax1.bar(rating_counts.index, rating_counts.values,
               color=COLORS["blue"], edgecolor="white", width=0.6)
for bar, cnt in zip(bars, rating_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
             f"{cnt:,}\n({cnt/n_ratings*100:.1f}%)",
             ha="center", fontsize=9, color=COLORS["gray"])
ax1.set_xlabel("评分 (1-5 星)")
ax1.set_ylabel("评分次数")
ax1.set_title("(a) 评分分布", fontweight="bold")
ax1.set_xticks([1, 2, 3, 4, 5])

# (b) 用户活跃度
ax2.hist(user_counts, bins=50, color=COLORS["orange"], edgecolor="white",
         alpha=0.85)
ax2.axvline(user_counts.mean(), color=COLORS["red"], linestyle="--",
            linewidth=1.5,
            label=f"平均 = {user_counts.mean():.0f} 条/用户")
ax2.axvline(user_counts.median(), color=COLORS["blue"], linestyle="--",
            linewidth=1.5,
            label=f"中位数 = {user_counts.median():.0f} 条/用户")
ax2.set_xlabel("每用户评分数")
ax2.set_ylabel("用户数量")
ax2.set_title("(b) 用户活跃度分布", fontweight="bold")
ax2.legend(fontsize=10)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_5_01_eda")
