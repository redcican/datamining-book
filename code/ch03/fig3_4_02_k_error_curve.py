"""
图 3.4.2  KNN 超参数敏感性：k 值对训练/测试准确率的影响
对应节次：3.4 K 近邻（KNN）算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_4_02_k_error_curve.py
输出路径：public/figures/ch03/fig3_4_02_k_error_curve.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

apply_style()

# ── 数据准备（乳腺癌数据集） ────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# ── k 从 1 到 60 ────────────────────────────────────────────────────────────
k_range = np.arange(1, 61)
train_accs, test_accs, cv_accs = [], [], []

for k in k_range:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)
    train_accs.append(clf.score(X_tr, y_tr))
    test_accs.append(clf.score(X_te, y_te))
    cv_accs.append(cross_val_score(
        KNeighborsClassifier(n_neighbors=k),
        X_tr, y_tr, cv=5, scoring="accuracy").mean())

train_accs = np.array(train_accs)
test_accs  = np.array(test_accs)
cv_accs    = np.array(cv_accs)

best_k_cv  = k_range[np.argmax(cv_accs)]
best_k_te  = k_range[np.argmax(test_accs)]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.38)

# ── 面板 (a): 准确率曲线 ──────────────────────────────────────────────────────
ax = axes[0]
ax.plot(k_range, 1 - train_accs, color=COLORS["blue"],
        lw=2.0, label="训练误差率")
ax.plot(k_range, 1 - test_accs, color=COLORS["red"],
        lw=2.0, label="测试误差率")
ax.plot(k_range, 1 - cv_accs, color=COLORS["teal"],
        lw=1.8, ls="--", label="5折CV误差率")
ax.axvline(best_k_cv, color=COLORS["teal"], lw=1.2, ls=":",
           alpha=0.8, label=f"最优 $k={best_k_cv}$（CV）")
ax.set_xlabel("邻居数 $k$", fontsize=13)
ax.set_ylabel("误差率（$1-$准确率）", fontsize=13)
ax.set_xlim(1, 60)
ax.set_title("(a) $k$ 值对误差率的影响（威斯康辛乳腺癌数据集）\n"
             "小 $k$：高方差（训练误差低，测试误差高）；大 $k$：高偏差",
             fontsize=12, pad=6)
ax.legend(fontsize=11)

# 标注偏差–方差区域
ax.annotate("高方差区\n（$k$ 过小）",
            xy=(3, (1 - test_accs)[2]),
            xytext=(10, (1 - test_accs)[2] + 0.04),
            arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2),
            fontsize=11, color=COLORS["red"])
ax.annotate("高偏差区\n（$k$ 过大）",
            xy=(55, (1 - test_accs)[54]),
            xytext=(38, 0.005),
            arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.2),
            fontsize=11, color=COLORS["blue"])

# ── 面板 (b): 不同 k 值最优/次优准确率 ────────────────────────────────────────
ax = axes[1]
# 绘制 CV 准确率热度图（每个 k 一个横条）
im = ax.imshow(cv_accs.reshape(1, -1), aspect="auto",
               cmap="RdYlGn", vmin=0.92, vmax=1.0,
               extent=[0.5, 60.5, 0, 1])
plt.colorbar(im, ax=ax, label="5折CV 准确率", shrink=0.60)
ax.set_yticks([])
ax.set_xlabel("邻居数 $k$", fontsize=13)
ax.set_title(f"(b) 各 $k$ 值的交叉验证准确率热力条\n"
             f"最优 $k={best_k_cv}$，CV 准确率={cv_accs[best_k_cv-1]:.3f}",
             fontsize=12, pad=6)
# 标注最优 k
ax.axvline(best_k_cv, color="white", lw=2.5, ls="--", zorder=5)
ax.text(best_k_cv + 1, 0.5, f"$k^*={best_k_cv}$",
        fontsize=12, color="white", va="center", fontweight="bold")

fig.suptitle(
    "KNN 超参数敏感性分析：$k$ 的选择决定偏差–方差权衡\n"
    "5折交叉验证是选择最优 $k$ 的标准工程手段",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_4_02_k_error_curve")
