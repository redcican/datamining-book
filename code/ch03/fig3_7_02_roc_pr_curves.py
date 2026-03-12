"""
图 3.7.2  ROC 曲线与 Precision-Recall 曲线（全章分类器对比）
对应节次：3.7 分类算法评估与比较
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_7_02_roc_pr_curves.py
输出路径：public/figures/ch03/fig3_7_02_roc_pr_curves.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

apply_style()

# ── 数据 ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 分类器 ────────────────────────────────────────────────────────────────────
classifiers = [
    ("决策树",     COLORS["gray"],   DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("KNN",        COLORS["teal"],   KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ("SVM",        COLORS["blue"],   SVC(kernel="rbf", C=10, gamma="scale",
                                         probability=True, random_state=42)),
    ("MLP",        COLORS["purple"], MLPClassifier(hidden_layer_sizes=(128, 64),
                                                    max_iter=500, random_state=42)),
    ("随机森林",   COLORS["green"],  RandomForestClassifier(n_estimators=200,
                                                              random_state=42, n_jobs=-1)),
    ("AdaBoost",   COLORS["orange"], AdaBoostClassifier(n_estimators=100,
                                                         algorithm="SAMME", random_state=42)),
    ("GBDT",       COLORS["red"],    GradientBoostingClassifier(n_estimators=100,
                                                                  max_depth=3, random_state=42)),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# ── 面板(a): ROC 曲线 ─────────────────────────────────────────────────────────
ax = axes[0]
for name, color, clf in classifiers:
    clf.fit(X_tr_s, y_tr)
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_te_s)[:, 1]
    else:
        scores = clf.decision_function(X_te_s)
    fpr, tpr, _ = roc_curve(y_te, scores)
    roc_auc = auc(fpr, tpr)
    lw = 2.5 if name in ("随机森林", "GBDT", "SVM") else 1.8
    ax.plot(fpr, tpr, color=color, lw=lw,
            label=f"{name}  (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="随机猜测 (AUC=0.500)")
ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
ax.set_xlabel("假阳性率 (FPR = FP / (FP+TN))", fontsize=13)
ax.set_ylabel("真阳性率 (TPR = TP / (TP+FN))", fontsize=13)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=11,
          labelspacing=0.3, handlelength=1.5)
ax.set_title("(a) ROC 曲线对比（Breast Cancer）", fontsize=14)

# ── 面板(b): Precision-Recall 曲线 ───────────────────────────────────────────
ax = axes[1]
baseline_pr = y_te.sum() / len(y_te)
for name, color, clf in classifiers:
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_te_s)[:, 1]
    else:
        scores = clf.decision_function(X_te_s)
    precision, recall, _ = precision_recall_curve(y_te, scores)
    ap = average_precision_score(y_te, scores)
    lw = 2.5 if name in ("随机森林", "GBDT", "SVM") else 1.8
    ax.plot(recall, precision, color=color, lw=lw,
            label=f"{name}  (AP={ap:.3f})")
ax.axhline(baseline_pr, color="black", linestyle="--", lw=1.2,
           label=f"随机基线 (P={baseline_pr:.3f})")
ax.set_xlabel("召回率 (Recall)", fontsize=13)
ax.set_ylabel("精确率 (Precision)", fontsize=13)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.0, 1.05)
ax.legend(loc="lower left", fontsize=11,
          labelspacing=0.3, handlelength=1.5)
ax.set_title("(b) Precision-Recall 曲线对比（Breast Cancer）", fontsize=14)

# ── 面板(c): AUC vs AP 散点对比 + F1 阈值曲线（RF示例） ──────────────────────
ax = axes[2]
names_short, aucs, aps = [], [], []
for name, color, clf in classifiers:
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_te_s)[:, 1]
    else:
        scores = clf.decision_function(X_te_s)
    fpr2, tpr2, _ = roc_curve(y_te, scores)
    names_short.append(name)
    aucs.append(auc(fpr2, tpr2))
    aps.append(average_precision_score(y_te, scores))

x_pos = np.arange(len(names_short))
bar_w = 0.38
bars1 = ax.bar(x_pos - bar_w/2, aucs, bar_w,
               color=COLORS["blue"], alpha=0.82, label="AUC-ROC")
bars2 = ax.bar(x_pos + bar_w/2, aps,  bar_w,
               color=COLORS["orange"], alpha=0.82, label="AP (PR 曲线)")
for bar, val in zip(list(bars1) + list(bars2), aucs + aps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=12)
ax.set_ylabel("指标值", fontsize=13)
ax.set_ylim(0.7, 1.03)
ax.legend(fontsize=12)
ax.set_title("(c) AUC-ROC 与平均精确率（AP）对比", fontsize=14)

fig.suptitle("ROC 曲线、Precision-Recall 曲线与 AUC 指标对比",
             fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig3_7_02_roc_pr_curves")
