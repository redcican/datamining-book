"""
fig7_7_02_model_fusion.py
多模型融合效果对比
(a) PR 曲线   (b) PR-AUC 条形图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
apply_style()
np.random.seed(42)
# ── 合成数据：5000 正常 + 100 欺诈，10 维 ────────────────────────
n_normal, n_fraud, n_feat = 5000, 100, 10
X_normal = np.random.randn(n_normal, n_feat)
X_fraud = np.random.randn(n_fraud, n_feat) * 1.8 + np.random.choice([-3, 3], size=(n_fraud, n_feat))
X = np.vstack([X_normal, X_fraud])
y_true = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
# ── 各方法异常分数 ────────────────────────────────────────────────
# 1) Z-score (max abs)
scores_zscore = np.max(np.abs((X - X.mean(0)) / (X.std(0) + 1e-8)), axis=1)
# 2) KNN 距离 (k=10)
knn = NearestNeighbors(n_neighbors=10)
knn.fit(X)
dists_knn, _ = knn.kneighbors(X)
scores_knn = dists_knn[:, -1]
# 3) LOF (取负，使越大越异常)
lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
lof.fit_predict(X)
scores_lof = -lof.negative_outlier_factor_
# 4) K-means 距离 (k=5)
km = KMeans(n_clusters=5, random_state=42, n_init=10)
km.fit(X)
scores_kmeans = np.min(
    np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2),
    axis=1)
# 5) Isolation Forest (取负)
iso = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
iso.fit(X)
scores_iforest = -iso.decision_function(X)
# ── 排名融合 ──────────────────────────────────────────────────────
all_scores = [scores_zscore, scores_knn, scores_lof, scores_kmeans, scores_iforest]
ranks = np.column_stack([rankdata(s) for s in all_scores])
scores_fusion = ranks.mean(axis=1)
# ── PR 曲线和 AUC ────────────────────────────────────────────────
method_names = ["Z-score", "KNN", "LOF", "K-means", "iForest"]
method_scores = all_scores + [scores_fusion]
method_labels = method_names + ["融合"]
pr_data = {}
for name, sc in zip(method_labels, method_scores):
    prec, rec, _ = precision_recall_curve(y_true, sc)
    pr_auc = auc(rec, prec)
    pr_data[name] = (prec, rec, pr_auc)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.7.2　多模型融合效果对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── (a) PR 曲线 ──────────────────────────────────────────────────
ax = axes[0]
line_colors = PALETTE[:5]
for i, name in enumerate(method_names):
    prec, rec, pr_auc = pr_data[name]
    ax.plot(rec, prec, color=line_colors[i], lw=1.5, alpha=0.7,
            label=f"{name} ({pr_auc:.3f})")
prec_f, rec_f, auc_f = pr_data["融合"]
ax.plot(rec_f, prec_f, color="black", lw=3.0,
        label=f"融合 ({auc_f:.3f})")
ax.set_xlabel("召回率 (Recall)", fontsize=14)
ax.set_ylabel("精确率 (Precision)", fontsize=14)
ax.set_title("(a) PR 曲线", fontsize=17)
ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=13)
# ── (b) PR-AUC 条形图 ────────────────────────────────────────────
ax = axes[1]
bar_names = method_names + ["融合"]
bar_aucs = [pr_data[n][2] for n in bar_names]
bar_colors = line_colors + ["#1e293b"]
y_pos = np.arange(len(bar_names))
bars = ax.barh(y_pos, bar_aucs, color=bar_colors, edgecolor="white", height=0.6)
# 标注数值
for i, (val, bar) in enumerate(zip(bar_aucs, bars)):
    ax.text(val + 0.008, y_pos[i], f"{val:.3f}",
            va="center", ha="left", fontsize=13, fontweight="bold")
ax.set_yticks(y_pos)
ax.set_yticklabels(bar_names, fontsize=14)
ax.set_xlabel("PR-AUC", fontsize=14)
ax.set_title("(b) PR-AUC 对比", fontsize=17)
ax.set_xlim(0, max(bar_aucs) * 1.15)
ax.tick_params(labelsize=13)
ax.invert_yaxis()
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_7_02_model_fusion")
