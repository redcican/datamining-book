"""
图 3.7.1  混淆矩阵与分类指标体系
对应节次：3.7 分类算法评估与比较
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_7_01_confusion_metrics.py
输出路径：public/figures/ch03/fig3_7_01_confusion_metrics.png
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

apply_style()

C_TP = COLORS["green"]
C_TN = COLORS["blue"]
C_FP = COLORS["orange"]
C_FN = COLORS["red"]
C_GRAY = COLORS["gray"]

# ── 数据与模型 ────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr)
y_pred = rf.predict(X_te_s)
cm = confusion_matrix(y_te, y_pred)   # [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
plt.subplots_adjust(wspace=0.45)

# ── 面板(a): 混淆矩阵四格示意（概念图） ──────────────────────────────────────
ax = axes[0]
ax.set_axis_off()
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)

def rbox(ax, cx, cy, w, h, fc, ec="white", lw=1.5, alpha=0.88, zorder=3):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)

# 四个格子
cells = [
    (2.0, 5.2, C_TN, "TN\n真阴性", "预测阴性\n实际阴性 ✓"),
    (6.0, 5.2, C_FP, "FP\n假阳性", "预测阳性\n实际阴性 ✗"),
    (2.0, 2.4, C_FN, "FN\n假阴性", "预测阴性\n实际阳性 ✗"),
    (6.0, 2.4, C_TP, "TP\n真阳性", "预测阳性\n实际阳性 ✓"),
]
for cx, cy, fc, label, desc in cells:
    rbox(ax, cx, cy, 3.4, 2.2, fc=fc)
    ax.text(cx, cy + 0.42, label, ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(cx, cy - 0.35, desc, ha="center", va="center",
            fontsize=12, color="white")

C_LABEL = "#374151"
# 轴标签
ax.text(4.0, 7.5, "预测标签", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_LABEL)
ax.text(2.0, 7.0, "阴性（−）", ha="center", fontsize=12, color=C_LABEL)
ax.text(6.0, 7.0, "阳性（+）", ha="center", fontsize=12, color=C_LABEL)
ax.text(0.35, 3.8, "实\n际\n标\n签", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_LABEL)
ax.text(0.85, 5.2, "阴\n性", ha="center", va="center", fontsize=12, color=C_LABEL)
ax.text(0.85, 2.4, "阳\n性", ha="center", va="center", fontsize=12, color=C_LABEL)
ax.set_title("(a) 混淆矩阵四格结构", fontsize=14, pad=10)

# ── 面板(b): 真实混淆矩阵热图（Breast Cancer RF） ─────────────────────────────
ax = axes[1]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = cm_norm[i, j]
        color = "white" if pct > 0.5 else "black"
        ax.text(j, i, f"{val}\n({pct:.1%})",
                ha="center", va="center", fontsize=14,
                color=color, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["预测阴性（良性）", "预测阳性（恶性）"], fontsize=12)
ax.set_yticklabels(["实际阴性（良性）", "实际阳性（恶性）"], fontsize=12)
ax.set_xlabel("预测标签", fontsize=13)
ax.set_ylabel("真实标签", fontsize=13)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
acc  = (TP + TN) / (TP + TN + FP + FN)
prec = TP / (TP + FP) if (TP + FP) > 0 else 0
rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
ax.set_title(f"(b) 随机森林混淆矩阵（Breast Cancer，$n_{{te}}={len(y_te)}$）\n"
             f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}",
             fontsize=13, pad=8)

# ── 面板(c): 指标公式与含义 ───────────────────────────────────────────────────
ax = axes[2]
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0.2, 10.4)

# 列边界与中心
X_DIV1, X_DIV2 = 2.5, 4.7   # 两条分隔线
X_NAME = 1.25                 # 名称列中心
X_FORM = 3.6                  # 公式列中心
X_DESC = 7.15                 # 描述列中心

BOX_W = 9.6
BOX_H = 1.55
BOX_CX = 4.8
ys = [8.9, 7.1, 5.3, 3.5, 1.7]

metrics = [
    ("准确率\n(Accuracy)",
     r"$\dfrac{TP+TN}{N}$",
     "整体预测正确的比例\n类别平衡时首选（N=全部样本数）"),
    ("精确率\n(Precision)",
     r"$\dfrac{TP}{TP+FP}$",
     "预测为正中真正为正的比例\n↑ 降低误报，宁缺毋滥场景"),
    ("召回率\n(Recall)",
     r"$\dfrac{TP}{TP+FN}$",
     "所有正样本中被检出的比例\n↑ 降低漏报，医疗/欺诈检测"),
    ("F1 分数\n(F1-Score)",
     r"$\dfrac{2\,P\!\cdot\!R}{P+R}$",
     "精确率与召回率的调和均值\n类别不平衡时优于准确率"),
    ("特异性\n(Specificity)",
     r"$\dfrac{TN}{TN+FP}$",
     "所有负样本中正确识别的比例\n即负类召回率（TNR）"),
]
colors_m = [C_TN, C_TP, C_FN, COLORS["purple"], C_GRAY]

for (name, formula, desc), fc, yc in zip(metrics, colors_m, ys):
    # 背景框
    rbox(ax, BOX_CX, yc, BOX_W, BOX_H, fc=fc, alpha=0.15, ec=fc, lw=1.5)
    # 分隔线（垂直居中于框内）
    pad = 0.12
    for xd in (X_DIV1, X_DIV2):
        ax.plot([xd, xd], [yc - BOX_H/2 + pad, yc + BOX_H/2 - pad],
                color=fc, lw=0.9, alpha=0.45, zorder=4)
    # 名称：水平居中于名称列，垂直居中于框
    ax.text(X_NAME, yc, name,
            ha="center", va="center",
            fontsize=12, color=fc, fontweight="bold", linespacing=1.25,
            zorder=5)
    # 公式：水平居中于公式列，垂直居中于框
    ax.text(X_FORM, yc, formula,
            ha="center", va="center",
            fontsize=13, zorder=5)
    # 描述：水平居中于描述列，垂直居中于框
    ax.text(X_DESC, yc, desc,
            ha="center", va="center",
            fontsize=10.5, color="#374151", linespacing=1.45, zorder=5)

# 列标题行
HDR_Y = 9.85
ax.text(X_NAME, HDR_Y, "指标名称",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#374151")
ax.text(X_FORM, HDR_Y, "计算公式",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#374151")
ax.text(7.15, HDR_Y, "适用场景说明",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#374151")
ax.plot([BOX_CX - BOX_W/2, BOX_CX + BOX_W/2], [9.55, 9.55],
        color="#9ca3af", lw=1.0)
ax.set_title("(c) 常用分类评估指标汇总", fontsize=14, pad=10)

fig.suptitle("混淆矩阵与分类指标体系", fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, __file__, "fig3_7_01_confusion_metrics")
