"""
图 13.5.3　四种推荐方法性能对比
User Mean Baseline / User-based CF / Item-based CF / NMF
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

mask_train = (R_train > 0).astype(float)

# 用户均值
user_means = np.zeros(n_users)
for u in range(n_users):
    rated = R_train[u][R_train[u] > 0]
    user_means[u] = rated.mean() if len(rated) > 0 else 3.0

print(f"训练集: {len(train_df):,} 条, 测试集: {len(test_df):,} 条")


# ── 1. User Mean Baseline ─────────────────────────────────
def predict_user_mean(test_df, user_means):
    preds = []
    for _, row in test_df.iterrows():
        preds.append(user_means[row["user_id"]])
    return np.array(preds)


# ── 2. User-based CF ──────────────────────────────────────
def predict_user_cf(test_df, R_train, mask_train, user_means, k=30):
    # 中心化评分矩阵
    R_centered = R_train.copy()
    for u in range(R_train.shape[0]):
        rated_mask = R_train[u] > 0
        R_centered[u][rated_mask] -= user_means[u]
        R_centered[u][~rated_mask] = 0

    sim = cosine_similarity(R_centered)
    np.fill_diagonal(sim, 0)

    preds = []
    for _, row in test_df.iterrows():
        u, i = row["user_id"], row["item_id"]
        # 找到评价过 item i 的用户
        rated_users = np.where(R_train[:, i] > 0)[0]
        if len(rated_users) == 0:
            preds.append(user_means[u])
            continue

        sims = sim[u, rated_users]
        top_k_idx = np.argsort(np.abs(sims))[-k:]
        top_users = rated_users[top_k_idx]
        top_sims = sims[top_k_idx]

        denom = np.abs(top_sims).sum()
        if denom == 0:
            preds.append(user_means[u])
        else:
            weighted = (top_sims *
                        (R_train[top_users, i] -
                         user_means[top_users])).sum()
            preds.append(user_means[u] + weighted / denom)
    return np.clip(preds, 1, 5)


# ── 3. Item-based CF ──────────────────────────────────────
def predict_item_cf(test_df, R_train, mask_train, user_means, k=30):
    # Adjusted cosine similarity (center by user mean)
    R_adj = R_train.copy()
    for u in range(R_train.shape[0]):
        rated_mask = R_train[u] > 0
        R_adj[u][rated_mask] -= user_means[u]
        R_adj[u][~rated_mask] = 0

    sim = cosine_similarity(R_adj.T)  # item × item
    np.fill_diagonal(sim, 0)

    preds = []
    for _, row in test_df.iterrows():
        u, i = row["user_id"], row["item_id"]
        # 用户 u 评过的其他电影
        rated_items = np.where(R_train[u] > 0)[0]
        if len(rated_items) == 0:
            preds.append(user_means[u])
            continue

        sims = sim[i, rated_items]
        # 仅使用正相似度的物品
        pos_mask = sims > 0
        if pos_mask.sum() == 0:
            preds.append(user_means[u])
            continue

        pos_items = rated_items[pos_mask]
        pos_sims = sims[pos_mask]

        # 选 top-k
        if len(pos_sims) > k:
            top_k_idx = np.argsort(pos_sims)[-k:]
            pos_items = pos_items[top_k_idx]
            pos_sims = pos_sims[top_k_idx]

        weighted = (pos_sims * R_train[u, pos_items]).sum()
        preds.append(weighted / pos_sims.sum())
    return np.clip(preds, 1, 5)


# ── 4. NMF ────────────────────────────────────────────────
def predict_nmf(test_df, R_train, user_means, k=20):
    # 填充缺失值为用户均值
    R_filled = R_train.copy()
    for u in range(R_train.shape[0]):
        missing = R_train[u] == 0
        R_filled[u][missing] = user_means[u]

    model = NMF(n_components=k, init="nndsvda", random_state=42,
                max_iter=500)
    W = model.fit_transform(R_filled)
    H = model.components_
    R_pred = W @ H

    preds = []
    for _, row in test_df.iterrows():
        preds.append(R_pred[row["user_id"], row["item_id"]])
    return np.clip(preds, 1, 5)


# ── 评估 ─────────────────────────────────────────────────
actuals = test_df["rating"].values

print("\n计算中...")
methods = {}

preds = predict_user_mean(test_df, user_means)
methods["User Mean\nBaseline"] = preds
print("  User Mean done")

preds = predict_user_cf(test_df, R_train, mask_train, user_means, k=30)
methods["User-based\nCF (k=30)"] = preds
print("  User-based CF done")

preds = predict_item_cf(test_df, R_train, mask_train, user_means, k=30)
methods["Item-based\nCF (k=30)"] = preds
print("  Item-based CF done")

preds = predict_nmf(test_df, R_train, user_means, k=20)
methods["NMF\n(k=20)"] = preds
print("  NMF done")

print(f"\n{'方法':<22s} {'RMSE':>8s} {'MAE':>8s}")
print("-" * 40)
results = {}
for name, preds in methods.items():
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    results[name] = {"RMSE": rmse, "MAE": mae}
    name_flat = name.replace("\n", " ")
    print(f"  {name_flat:<20s} {rmse:>8.4f} {mae:>8.4f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

method_names = list(results.keys())
x = np.arange(len(method_names))
width = 0.3
colors = [COLORS["blue"], COLORS["orange"]]
metric_names = ["RMSE", "MAE"]

for i, metric in enumerate(metric_names):
    vals = [results[m][metric] for m in method_names]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=metric,
                  color=colors[i], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(method_names, fontsize=11)
ax.set_ylabel("误差")
ax.set_title("四种推荐方法性能对比（MovieLens 100K）", fontweight="bold")
ax.set_ylim(0, max(r["RMSE"] for r in results.values()) * 1.25)
ax.legend(fontsize=11)

plt.tight_layout()
save_fig(fig, __file__, "fig13_5_03_model_comparison")
