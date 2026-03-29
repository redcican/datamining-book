"""
图 13.5.4　NMF 隐因子分析
(a) RMSE vs 隐因子维度 k  (b) 示例用户推荐结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_movielens import prepare_data

train_df, test_df, n_users, n_items, movies = prepare_data()

# ── 构建评分矩阵 ─────────────────────────────────────────
R_train = np.zeros((n_users, n_items))
for _, row in train_df.iterrows():
    R_train[row["user_id"], row["item_id"]] = row["rating"]

user_means = np.zeros(n_users)
for u in range(n_users):
    rated = R_train[u][R_train[u] > 0]
    user_means[u] = rated.mean() if len(rated) > 0 else 3.0

# 填充缺失值
R_filled = R_train.copy()
for u in range(n_users):
    missing = R_train[u] == 0
    R_filled[u][missing] = user_means[u]

actuals = test_df["rating"].values


# ── (a) RMSE vs k ────────────────────────────────────────
ks = [5, 10, 15, 20, 30, 50, 75, 100]
rmse_list, mae_list = [], []

print("=== NMF 隐因子维度调优 ===")
print(f"{'k':>5s} {'RMSE':>8s} {'MAE':>8s}")

for k in ks:
    model = NMF(n_components=k, init="nndsvda", random_state=42,
                max_iter=500)
    W = model.fit_transform(R_filled)
    H = model.components_
    R_pred = np.clip(W @ H, 1, 5)

    preds = np.array([R_pred[row["user_id"], row["item_id"]]
                      for _, row in test_df.iterrows()])
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    rmse_list.append(rmse)
    mae_list.append(mae)
    print(f"  {k:>3d}   {rmse:>8.4f} {mae:>8.4f}")

best_k_idx = np.argmin(rmse_list)
best_k = ks[best_k_idx]
print(f"\n最佳 k = {best_k}, RMSE = {rmse_list[best_k_idx]:.4f}")


# ── (b) 示例用户推荐 ─────────────────────────────────────
# 使用最佳 k 重新训练
model = NMF(n_components=best_k, init="nndsvda", random_state=42,
            max_iter=500)
W = model.fit_transform(R_filled)
H = model.components_
R_pred_full = np.clip(W @ H, 1, 5)

# 选一个活跃用户（评分数 > 50）
user_rating_counts = train_df.groupby("user_id").size()
active_users = user_rating_counts[user_rating_counts > 50].index.tolist()
demo_user = active_users[0]

# 该用户训练集中评分最高的电影
user_train = train_df[train_df["user_id"] == demo_user].sort_values(
    "rating", ascending=False)
print(f"\n=== 用户 {demo_user} 的推荐示例 ===")
print(f"  训练集评分数: {len(user_train)}")
print(f"  已评分最高的 5 部:")
for _, row in user_train.head(5).iterrows():
    item_id_1based = row["item_id"] + 1
    title = movies[movies["item_id"] == item_id_1based]["title"].values
    title = title[0] if len(title) > 0 else f"Movie {item_id_1based}"
    print(f"    {title[:40]:<42s} 评分={row['rating']:.0f}")

# 推荐：预测未评分电影中得分最高的
rated_items = set(train_df[train_df["user_id"] == demo_user]["item_id"])
pred_scores = R_pred_full[demo_user].copy()
for i in rated_items:
    pred_scores[i] = -1  # 排除已评分

top10_items = np.argsort(pred_scores)[-10:][::-1]
print(f"\n  NMF 推荐 Top-10:")
rec_titles, rec_scores = [], []
for rank, item_id in enumerate(top10_items, 1):
    item_id_1based = item_id + 1
    title = movies[movies["item_id"] == item_id_1based]["title"].values
    title = title[0] if len(title) > 0 else f"Movie {item_id_1based}"
    score = pred_scores[item_id]
    rec_titles.append(title[:30])
    rec_scores.append(score)
    print(f"    {rank:>2d}. {title[:40]:<42s} 预测={score:.2f}")


# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                gridspec_kw={"width_ratios": [1, 1.2]})

# (a) RMSE vs k
ax1.plot(ks, rmse_list, "o-", color=COLORS["blue"], linewidth=2,
         markersize=8, label="RMSE", zorder=3)
ax1.plot(ks, mae_list, "s--", color=COLORS["orange"], linewidth=2,
         markersize=7, label="MAE", zorder=3)
ax1.axvline(best_k, color=COLORS["red"], linestyle=":", linewidth=1.5,
            alpha=0.7, label=f"最佳 k = {best_k}")
ax1.set_xlabel("隐因子维度 k")
ax1.set_ylabel("误差")
ax1.set_title("(a) NMF 隐因子维度 vs 预测误差", fontweight="bold")
ax1.legend(fontsize=10)
ax1.set_xticks(ks)

# (b) Top-10 推荐
y_pos = np.arange(len(rec_titles))[::-1]
bars = ax2.barh(y_pos, rec_scores, color=COLORS["green"],
                edgecolor="white", height=0.65)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(rec_titles, fontsize=9)
ax2.set_xlabel("NMF 预测评分")
ax2.set_title(f"(b) 用户 {demo_user} 的 Top-10 推荐电影",
              fontweight="bold")
for bar, score in zip(bars, rec_scores):
    ax2.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
             f"{score:.2f}", va="center", fontsize=9, color=COLORS["gray"])
ax2.set_xlim(0, max(rec_scores) * 1.15)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_5_04_nmf_analysis")
