"""
fig7_6_03_iforest_vs_ocsvm.py
隔离森林与 One-Class SVM 的决策边界对比
(a) 隔离森林  (b) One-Class SVM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成数据 ──────────────────────────────────────────────────────
# 椭圆簇 — 用协方差矩阵控制形状
n_normal = 200
cov = [[1.5, 0.8], [0.8, 0.6]]
normal = np.random.multivariate_normal([0, 0], cov, n_normal)
# 5 个异常点
outliers = np.array([[4, 3], [-4, 2.5], [3, -3], [-3, -3.5], [5, 0]])
X_all = np.vstack([normal, outliers])
y_labels = np.concatenate([np.zeros(n_normal), np.ones(len(outliers))])
# ── 网格 ──────────────────────────────────────────────────────────
margin = 1.5
x_min, x_max = X_all[:, 0].min() - margin, X_all[:, 0].max() + margin
y_min, y_max = X_all[:, 1].min() - margin, X_all[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))
grid = np.column_stack([xx.ravel(), yy.ravel()])
# ── 训练模型 ──────────────────────────────────────────────────────
iforest = IsolationForest(n_estimators=200, max_samples=256,
                          contamination=0.05, random_state=42)
iforest.fit(normal)
Z_if = -iforest.decision_function(grid).reshape(xx.shape)
# ──
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(normal)
Z_svm = -ocsvm.decision_function(grid).reshape(xx.shape)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.6.3　隔离森林与 One-Class SVM 的决策边界对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 绘制辅助函数 ──────────────────────────────────────────────────
def plot_panel(ax, Z, title):
    """在 ax 上绘制等高线 + 散点。"""
    cf = ax.contourf(xx, yy, Z, levels=30, cmap="RdYlBu_r", alpha=0.75, zorder=1)
    ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=1.5,
               linestyles="--", zorder=2)
    ax.scatter(normal[:, 0], normal[:, 1], c=COLORS["blue"], s=25,
               alpha=0.7, edgecolors="k", linewidths=0.3, zorder=3,
               label="正常点")
    ax.scatter(outliers[:, 0], outliers[:, 1], marker="*", s=300,
               c=COLORS["red"], edgecolors="k", linewidths=0.8, zorder=5,
               label="异常点")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("异常分数", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("$x_1$", fontsize=15)
    ax.set_ylabel("$x_2$", fontsize=15)
    ax.legend(fontsize=13, loc="upper left")
    ax.tick_params(labelsize=13)
# ══════════════════════════════════════════════════════════════════
# (a) 隔离森林
# ══════════════════════════════════════════════════════════════════
plot_panel(axes[0], Z_if, "(a) 隔离森林")
# ══════════════════════════════════════════════════════════════════
# (b) One-Class SVM
# ══════════════════════════════════════════════════════════════════
plot_panel(axes[1], Z_svm, "(b) One-Class SVM")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_6_03_iforest_vs_ocsvm")
