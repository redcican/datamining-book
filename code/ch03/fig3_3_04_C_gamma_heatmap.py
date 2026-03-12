"""
图 3.3.4  RBF-SVM 超参数敏感性：C 与 gamma 的网格搜索热力图
对应节次：3.3 支持向量机（SVM）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_3_04_C_gamma_heatmap.py
输出路径：public/figures/ch03/fig3_3_04_C_gamma_heatmap.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

apply_style()

# ── 数据准备 ──────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# ── 网格搜索 ──────────────────────────────────────────────────────────────────
C_vals    = np.logspace(-2, 4, 13)
gam_vals  = np.logspace(-4, 2, 13)
param_grid = {"C": C_vals, "gamma": gam_vals}

grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1, refit=True)
grid.fit(X_tr, y_tr)

scores = grid.cv_results_["mean_test_score"].reshape(len(C_vals), len(gam_vals))
best_C   = grid.best_params_["C"]
best_gam = grid.best_params_["gamma"]
best_est = grid.best_estimator_

best_acc = accuracy_score(y_te, best_est.predict(X_te))
best_auc = roc_auc_score(y_te, best_est.predict_proba(X_te)[:, 1])

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.40)

# ── Panel (a): 热力图 ──────────────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(scores, interpolation="nearest", cmap="YlOrRd",
               aspect="auto", vmin=scores.min(), vmax=1.0)
plt.colorbar(im, ax=ax, label="5折CV AUC", shrink=0.88)

tick_labels_gam = [f"$10^{{{v:.0f}}}$" for v in np.linspace(-4, 2, 13)]
tick_labels_C   = [f"$10^{{{v:.0f}}}$" for v in np.linspace(-2, 4, 13)]
ax.set_xticks(range(13))
ax.set_yticks(range(13))
ax.set_xticklabels(tick_labels_gam, fontsize=10)
ax.set_yticklabels(tick_labels_C,   fontsize=10)
ax.set_xlabel(r"$\gamma$（RBF 核宽度）", fontsize=13)
ax.set_ylabel(r"$C$（正则化强度倒数）", fontsize=13)
ax.set_title("(a) RBF-SVM 超参数热力图（5折 CV AUC）\n"
             "横轴大→过局部化（高方差）；纵轴小→强正则（高偏差）",
             fontsize=12, pad=6)

# 标注最优超参数
best_gam_idx = np.argmin(np.abs(gam_vals - best_gam))
best_C_idx   = np.argmin(np.abs(C_vals   - best_C))
ax.scatter(best_gam_idx, best_C_idx, s=200, c="white",
           marker="*", zorder=6, label=f"最优：C={best_C:.2f}, $\\gamma$={best_gam:.4f}")
ax.legend(fontsize=11, loc="upper left")

# 标注偏差–方差分区
ax.text(1.5, 1.0, "高偏差区\n（间隔过宽）", fontsize=10,
        color="white", ha="center", va="center",
        bbox=dict(fc="#374151", alpha=0.65, boxstyle="round,pad=0.3"))
ax.text(10.5, 10.5, "高方差区\n（过拟合）", fontsize=10,
        color="white", ha="center", va="center",
        bbox=dict(fc="#374151", alpha=0.65, boxstyle="round,pad=0.3"))

# ── Panel (b): 最优模型混淆矩阵 ───────────────────────────────────────────────
ax = axes[1]
cm = confusion_matrix(y_te, best_est.predict(X_te))
disp = ConfusionMatrixDisplay(cm, display_labels=["恶性", "良性"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    f"(b) 最优 RBF-SVM 混淆矩阵（测试集）\n"
    f"$C={best_C:.2f}$，$\\gamma={best_gam:.4f}$\n"
    f"准确率={best_acc:.1%}，AUC={best_auc:.4f}",
    fontsize=12, pad=6)
ax.set_xlabel("预测类别", fontsize=13)
ax.set_ylabel("真实类别", fontsize=13)

fig.suptitle(
    "威斯康辛乳腺癌数据集：RBF-SVM 超参数选择与最优模型评估",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_3_04_C_gamma_heatmap")
