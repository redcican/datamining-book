"""
图 13.2.4　决策阈值调整对混淆矩阵的影响
(a) 默认阈值 0.5  (b) 降低阈值 0.3
展示精确率-召回率权衡
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_credit import prepare_data

X_train, X_test, y_train, y_test, preprocessor, *_ = prepare_data()

# ── 1. 训练模型，用不同阈值预测 ──────────────────────────
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
])
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]

# 默认阈值 0.5
y_pred_05 = (y_prob >= 0.5).astype(int)
cm_05 = confusion_matrix(y_test, y_pred_05)

# 降低阈值 0.3（更积极地预测"坏客户"）
y_pred_03 = (y_prob >= 0.3).astype(int)
cm_03 = confusion_matrix(y_test, y_pred_03)

print("=== 阈值 = 0.5（默认） ===")
print(classification_report(y_test, y_pred_05,
                            target_names=["Good", "Bad"]))
print("=== 阈值 = 0.3（降低） ===")
print(classification_report(y_test, y_pred_03,
                            target_names=["Good", "Bad"]))

# ── 2. 绘图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
labels = ["Good\n(好客户)", "Bad\n(坏客户)"]


def plot_cm(ax, cm, title, color_map):
    im = ax.imshow(cm, cmap=color_map, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            total = cm[i].sum()
            pct = val / total * 100
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


plot_cm(ax1, cm_05, "(a) 默认阈值 (threshold=0.5)", "Blues")
plot_cm(ax2, cm_03, "(b) 降低阈值 (threshold=0.3)", "Oranges")

# Highlight key difference
recall_05 = cm_05[1, 1] / cm_05[1].sum()
recall_03 = cm_03[1, 1] / cm_03[1].sum()
prec_05 = cm_05[1, 1] / cm_05[:, 1].sum() if cm_05[:, 1].sum() > 0 else 0
prec_03 = cm_03[1, 1] / cm_03[:, 1].sum() if cm_03[:, 1].sum() > 0 else 0
fig.text(0.5, 0.02,
         f"坏客户召回率：{recall_05:.0%} → {recall_03:.0%} (↑)　"
         f"精确率：{prec_05:.0%} → {prec_03:.0%} (↓)　"
         f"—— 精确率-召回率权衡",
         ha="center", fontsize=11, fontweight="bold",
         color=COLORS["red"],
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                   edgecolor=COLORS["orange"], alpha=0.9))

plt.tight_layout(rect=[0, 0.08, 1, 1])
save_fig(fig, __file__, "fig13_2_04_confusion_matrices")
