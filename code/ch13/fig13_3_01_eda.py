"""
图 13.3.1　乳腺癌数据集探索性分析
(a) 类别分布  (b) PCA 二维投影
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_medical import load_cancer

X, y, feature_names = load_cancer()

print(f"数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
print(f"  恶性 (malignant, 0): {(y==0).sum()} ({(y==0).mean():.1%})")
print(f"  良性 (benign, 1):    {(y==1).sum()} ({(y==1).mean():.1%})")
print(f"\n关键特征统计 (恶性 vs 良性):")
for col in ["mean radius", "mean texture", "mean concavity",
            "worst radius", "worst concave points"]:
    m = X.loc[y == 0, col]
    b = X.loc[y == 1, col]
    print(f"  {col:<25s} mal={m.mean():.2f}±{m.std():.2f}  "
          f"ben={b.mean():.2f}±{b.std():.2f}")

# ── PCA ──────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA 前 2 主成分解释方差: "
      f"{pca.explained_variance_ratio_[0]:.1%}, "
      f"{pca.explained_variance_ratio_[1]:.1%} "
      f"(合计 {pca.explained_variance_ratio_.sum():.1%})")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [0.7, 1.3]})

# (a) 类别分布
n_mal, n_ben = (y == 0).sum(), (y == 1).sum()
bars = ax1.bar(["恶性\n(Malignant)", "良性\n(Benign)"],
               [n_mal, n_ben],
               color=[COLORS["red"], COLORS["blue"]], width=0.5)
for bar, cnt in zip(bars, [n_mal, n_ben]):
    pct = cnt / len(y) * 100
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 8,
             f"{cnt} ({pct:.1f}%)", ha="center", fontsize=12,
             fontweight="bold")
ax1.set_ylabel("样本数")
ax1.set_title("(a) 诊断类别分布", fontweight="bold")
ax1.set_ylim(0, n_ben * 1.2)

# (b) PCA 散点
mask_m = y == 0
mask_b = y == 1
ax2.scatter(X_pca[mask_m, 0], X_pca[mask_m, 1],
            c=COLORS["red"], alpha=0.6, s=30, label="恶性 (Malignant)",
            edgecolors="white", linewidths=0.3)
ax2.scatter(X_pca[mask_b, 0], X_pca[mask_b, 1],
            c=COLORS["blue"], alpha=0.6, s=30, label="良性 (Benign)",
            edgecolors="white", linewidths=0.3)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)")
ax2.set_title("(b) PCA 二维投影", fontweight="bold")
ax2.legend(fontsize=10)

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_3_01_eda")
