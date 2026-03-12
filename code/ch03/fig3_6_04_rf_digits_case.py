"""
图 3.6.4  随机森林手写数字识别：特征重要性、OOB 误差与全章算法对比
对应节次：3.6 集成学习算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_6_04_rf_digits_case.py
输出路径：public/figures/ch03/fig3_6_digits_case.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

apply_style()

# ── 数据 ──────────────────────────────────────────────────────────────────────
data = load_digits()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 最优 RF 模型（用于特征重要性） ────────────────────────────────────────────
rf_best = RandomForestClassifier(n_estimators=200, oob_score=True,
                                  random_state=42, n_jobs=-1)
rf_best.fit(X_tr_s, y_tr)
importances = rf_best.feature_importances_.reshape(8, 8)

# ── OOB 误差随 n_estimators ───────────────────────────────────────────────────
ns = np.arange(1, 201, 4)
oob_errors = []
for n in ns:
    rf = RandomForestClassifier(n_estimators=int(n), oob_score=True,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    oob_errors.append(1 - rf.oob_score_)
oob_errors = np.array(oob_errors)

# ── 全章算法对比（5折CV，load_digits） ───────────────────────────────────────
all_methods = [
    ("决策树",         DecisionTreeClassifier(max_depth=None, random_state=42)),
    ("KNN ($k=3$)",    KNeighborsClassifier(n_neighbors=3)),
    ("朴素贝叶斯",     None),   # 用 GNB
    ("SVM (RBF)",      SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("MLP",            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ("Bagging",        BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("随机森林",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("AdaBoost",       AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)),
    ("GBDT",           GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                   learning_rate=0.1, random_state=42)),
]

# 替换 GNB
from sklearn.naive_bayes import GaussianNB
all_methods[2] = ("朴素贝叶斯", GaussianNB())

method_names = [m[0] for m in all_methods]
method_accs  = []
for name, clf in all_methods:
    cv = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="accuracy")
    method_accs.append(cv.mean())
method_accs = np.array(method_accs)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 7))
gs  = fig.add_gridspec(1, 3, wspace=0.42)
ax_imp  = fig.add_subplot(gs[0])
ax_oob  = fig.add_subplot(gs[1])
ax_comp = fig.add_subplot(gs[2])

# ── 面板 (a): 特征重要性热力图（8×8 像素） ──────────────────────────────────
im = ax_imp.imshow(importances, cmap="YlOrRd", interpolation="nearest")
plt.colorbar(im, ax=ax_imp, shrink=0.85, label="相对重要性")
ax_imp.set_xticks(np.arange(8))
ax_imp.set_yticks(np.arange(8))
ax_imp.set_xticklabels([f"列{i}" for i in range(8)], fontsize=10)
ax_imp.set_yticklabels([f"行{i}" for i in range(8)], fontsize=10)
ax_imp.set_title(
    f"(a) 随机森林像素特征重要性（$T=200$）\n"
    f"中央像素对数字分类贡献最大，边角像素贡献极低",
    fontsize=12, pad=6)
# 标注最重要像素
max_idx = np.unravel_index(np.argmax(importances), importances.shape)
ax_imp.add_patch(plt.Rectangle((max_idx[1]-0.5, max_idx[0]-0.5), 1, 1,
                                fill=False, edgecolor=COLORS["blue"], lw=2.5))

# ── 面板 (b): OOB 误差随 n_estimators ───────────────────────────────────────
ax_oob.plot(ns, oob_errors * 100, color=COLORS["teal"], lw=2.0,
            label="OOB 误差率")
ax_oob.axhline(oob_errors[-30:].mean() * 100,
               color=COLORS["gray"], lw=1.2, ls="--", alpha=0.8, label="收敛均值")
ax_oob.set_xlabel("决策树数量 $T$", fontsize=13)
ax_oob.set_ylabel("OOB 误差率（%）", fontsize=13)
ax_oob.set_xlim(1, 200)
ax_oob.set_title(
    f"(b) 随机森林 OOB 误差随 $T$ 变化\n"
    f"OOB 误差是泛化误差的无偏估计，$T\\approx100$ 后基本收敛",
    fontsize=12, pad=6)
ax_oob.legend(fontsize=11)

# ── 面板 (c): 全章算法对比柱状图 ────────────────────────────────────────────
bar_colors = (
    [COLORS["gray"]] +           # 决策树（对照）
    [COLORS["blue"]] * 3 +       # KNN, NB, SVM
    [COLORS["purple"]] +         # MLP
    [COLORS["teal"]] * 2 +       # Bagging, RF
    [COLORS["orange"]] * 2       # AdaBoost, GBDT
)
bars = ax_comp.bar(range(len(method_names)), method_accs * 100,
                   color=bar_colors, edgecolor="white", lw=1.0)
ax_comp.set_xticks(range(len(method_names)))
ax_comp.set_xticklabels(method_names, rotation=38, ha="right", fontsize=11)
ax_comp.set_ylim(70, 101)
ax_comp.set_ylabel("5折CV 准确率（%）", fontsize=13)
ax_comp.set_title("(c) 全章算法横向对比（load_digits）\n"
                  "集成方法（绿/橙）整体优于单个分类器（灰/蓝）",
                  fontsize=12, pad=6)
for i, (bar, acc) in enumerate(zip(bars, method_accs)):
    ax_comp.text(bar.get_x() + bar.get_width()/2, acc*100 + 0.1,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

# 图例
from matplotlib.patches import Patch
legend_patches = [
    Patch(color=COLORS["gray"],   label="决策树（基线）"),
    Patch(color=COLORS["blue"],   label="§3.1–3.4 单分类器"),
    Patch(color=COLORS["purple"], label="§3.5 神经网络"),
    Patch(color=COLORS["teal"],   label="§3.6 Bagging 系列"),
    Patch(color=COLORS["orange"], label="§3.6 Boosting 系列"),
]
ax_comp.legend(handles=legend_patches, fontsize=10, loc="lower right")

fig.suptitle(
    f"随机森林手写数字识别与全章算法对比（load_digits, $n=1797$, $d=64$, $K=10$）",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_6_04_rf_digits_case")
