"""
图 3.7.3  交叉验证示意图与学习曲线
对应节次：3.7 分类算法评估与比较
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_7_03_cv_learning_curve.py
输出路径：public/figures/ch03/fig3_7_03_cv_learning_curve.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

apply_style()

C_TRAIN = COLORS["blue"]
C_VAL   = COLORS["orange"]
C_TEST  = COLORS["red"]
C_GRAY  = COLORS["gray"]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# ── 面板(a): 5 折 CV 示意图 ───────────────────────────────────────────────────
ax = axes[0]
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.5)

K = 5
fold_h = 0.68
fold_w = 7.5
x0 = 1.2

# 总数据集
p = FancyBboxPatch((x0, 7.2), fold_w, 0.65, boxstyle="round,pad=0.05",
                   facecolor=COLORS["light"], edgecolor=C_GRAY, linewidth=1.5)
ax.add_patch(p)
ax.text(x0 + fold_w/2, 7.52, "完整训练集（$n$ 个样本）",
        ha="center", va="center", fontsize=13, color=C_GRAY, fontweight="bold")

# 折叠示意
fold_xs = np.linspace(x0, x0 + fold_w, K + 1)
fold_w_each = fold_xs[1] - fold_xs[0]

for k in range(K):
    y_center = 6.3 - k * 1.12
    for j in range(K):
        fc = C_VAL if j == k else C_TRAIN
        label = "验证折" if j == k else "训练折"
        p2 = FancyBboxPatch((fold_xs[j], y_center - fold_h/2),
                            fold_w_each - 0.04, fold_h,
                            boxstyle="round,pad=0.03",
                            facecolor=fc, edgecolor="white",
                            linewidth=1.0, alpha=0.88)
        ax.add_patch(p2)
        if j == k:
            ax.text(fold_xs[j] + fold_w_each/2, y_center,
                    f"验证\n折 {j+1}", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        else:
            ax.text(fold_xs[j] + fold_w_each/2, y_center,
                    "训", ha="center", va="center",
                    fontsize=11, color="white")
    ax.text(x0 + fold_w + 0.3, y_center,
            f"第 {k+1} 折", ha="left", va="center",
            fontsize=12, color=C_GRAY)

legend_patches = [
    mpatches.Patch(facecolor=C_TRAIN, label="训练数据"),
    mpatches.Patch(facecolor=C_VAL,   label="验证数据"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=12,
          bbox_to_anchor=(0.99, 0.99))
ax.text(x0 + fold_w/2, 0.55,
        "5 折 CV：每折轮流作为验证集，共训练 5 个模型\n"
        "最终性能 = 5 折验证分数的均值 ± 标准差",
        ha="center", va="center", fontsize=12, color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f9ff",
                  edgecolor=C_TRAIN, alpha=0.9))
ax.set_title("(a) $k$ 折交叉验证示意（$k=5$）", fontsize=14)

# ── 面板(b): 学习曲线（决策树 vs 随机森林） ───────────────────────────────────
ax = axes[1]
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
train_sizes = np.linspace(0.1, 1.0, 10)
for clf, name, color, ls in [
    (DecisionTreeClassifier(max_depth=None, random_state=42),
     "决策树（无限深度）", COLORS["orange"], "--"),
    (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
     "随机森林（$T=100$）", COLORS["green"], "-"),
]:
    t_sizes, t_scores, v_scores = learning_curve(
        clf, X_s, y, cv=5, scoring="accuracy",
        train_sizes=train_sizes, n_jobs=-1)
    t_mean, t_std = t_scores.mean(1), t_scores.std(1)
    v_mean, v_std = v_scores.mean(1), v_scores.std(1)
    ax.plot(t_sizes, t_mean, color=color, ls=ls, lw=2.2,
            label=f"{name}（训练）")
    ax.plot(t_sizes, v_mean, color=color, ls=ls, lw=2.2,
            alpha=0.55, marker="o", markersize=5,
            label=f"{name}（CV 验证）")
    ax.fill_between(t_sizes, v_mean - v_std, v_mean + v_std,
                    alpha=0.12, color=color)
ax.set_xlabel("训练样本数", fontsize=13)
ax.set_ylabel("准确率", fontsize=13)
ax.set_ylim(0.82, 1.01)
ax.legend(fontsize=11, ncol=1, labelspacing=0.3)
ax.set_title("(b) 学习曲线：决策树 vs 随机森林\n（Breast Cancer，5 折 CV）",
             fontsize=13)
# 标注高方差区域
ax.annotate("决策树训练准确率≈1\n（过拟合，高方差）",
            xy=(t_sizes[-1], 1.0), xytext=(300, 0.87),
            arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.5),
            fontsize=11, color=COLORS["orange"],
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", alpha=0.9))

# ── 面板(c): 验证曲线（决策树最大深度 vs 准确率） ────────────────────────────
ax = axes[2]
from sklearn.model_selection import validation_curve
depths = np.arange(1, 16)
t_scores_vc, v_scores_vc = validation_curve(
    DecisionTreeClassifier(random_state=42), X_s, y,
    param_name="max_depth", param_range=depths,
    cv=5, scoring="accuracy", n_jobs=-1)
t_mean_vc = t_scores_vc.mean(1)
v_mean_vc = v_scores_vc.mean(1)
v_std_vc  = v_scores_vc.std(1)
ax.plot(depths, t_mean_vc, color=C_TRAIN, lw=2.2, label="训练准确率")
ax.plot(depths, v_mean_vc, color=C_VAL, lw=2.2, marker="o",
        markersize=5, label="CV 验证准确率")
ax.fill_between(depths, v_mean_vc - v_std_vc, v_mean_vc + v_std_vc,
                alpha=0.15, color=C_VAL)
best_d = depths[np.argmax(v_mean_vc)]
ax.axvline(best_d, color=COLORS["green"], ls="--", lw=1.8,
           label=f"最优深度 = {best_d}")
ax.set_xlabel("决策树最大深度 (max_depth)", fontsize=13)
ax.set_ylabel("准确率", fontsize=13)
ax.set_ylim(0.84, 1.01)
ax.set_xticks(depths)
ax.legend(fontsize=12)
ax.set_title("(c) 验证曲线：决策树深度 vs 准确率\n（欠拟合↔过拟合 U 形曲线）",
             fontsize=13)
# 区域标注
ax.axvspan(0.5, best_d - 0.5, alpha=0.06, color=C_TRAIN, label="_欠拟合区")
ax.axvspan(best_d + 0.5, 15.5, alpha=0.06, color=C_VAL, label="_过拟合区")
ax.text(best_d/2, 0.855, "欠拟合\n（高偏差）",
        ha="center", fontsize=11, color=C_TRAIN)
ax.text((best_d + 15)/2, 0.855, "过拟合\n（高方差）",
        ha="center", fontsize=11, color=C_VAL)

fig.suptitle("交叉验证、学习曲线与验证曲线",
             fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig3_7_03_cv_learning_curve")
