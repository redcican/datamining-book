"""
图 3.5.4  MLP 手写数字识别：超参数搜索、混淆矩阵与算法对比
对应节次：3.5 神经网络分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_5_04_mlp_digits_case.py
输出路径：public/figures/ch03/fig3_5_04_mlp_digits_case.png
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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

apply_style()

# ── 数据准备 ──────────────────────────────────────────────────────────────────
data = load_digits()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 不同架构 MLP 的 CV 准确率 ──────────────────────────────────────────────────
configs = [
    (16,),
    (32,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),
]
config_labels = [
    "16",
    "32",
    "64",
    "128",
    "64-32",
    "128-64",
    "128-64-32",
    "256-128-64",
]
cv_accs = []
for cfg in configs:
    clf = MLPClassifier(hidden_layer_sizes=cfg, activation="relu",
                        solver="adam", max_iter=500, random_state=42)
    cv = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="accuracy")
    cv_accs.append(cv.mean())
cv_accs = np.array(cv_accs)
best_idx = np.argmax(cv_accs)
best_cfg = configs[best_idx]

# ── 最优 MLP 训练 ─────────────────────────────────────────────────────────────
best_clf = MLPClassifier(hidden_layer_sizes=best_cfg, activation="relu",
                         solver="adam", max_iter=500, random_state=42)
best_clf.fit(X_tr_s, y_tr)
y_pred = best_clf.predict(X_te_s)
best_acc = accuracy_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred)

# ── 算法对比 ──────────────────────────────────────────────────────────────────
comparators = [
    ("KNN ($k=3$)",  KNeighborsClassifier(n_neighbors=3)),
    ("决策树",       DecisionTreeClassifier(max_depth=None, random_state=42)),
    ("SVM (RBF)",    SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ("MLP",          MLPClassifier(hidden_layer_sizes=best_cfg, activation="relu",
                                   solver="adam", max_iter=500, random_state=42)),
]
comp_names = [n for n, _ in comparators]
comp_accs  = []
for name, clf in comparators:
    cv = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="accuracy")
    comp_accs.append(cv.mean())

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7))
gs  = fig.add_gridspec(1, 3, wspace=0.40)
ax_arch = fig.add_subplot(gs[0])
ax_cm   = fig.add_subplot(gs[1])
ax_comp = fig.add_subplot(gs[2])

# ── 面板 (a): 架构 vs CV 准确率 ───────────────────────────────────────────────
bar_colors = [COLORS["teal"] if i == best_idx else "#cbd5e1" for i in range(len(configs))]
bars = ax_arch.bar(config_labels, cv_accs * 100, color=bar_colors, edgecolor="white", lw=1.2)
ax_arch.set_ylim(90, 101)
ax_arch.set_xlabel("隐藏层架构（神经元数）", fontsize=13)
ax_arch.set_ylabel("5折CV 准确率（%）", fontsize=13)
ax_arch.set_title(f"(a) 网络架构超参数搜索\n最优架构: {config_labels[best_idx]}，CV准确率={cv_accs[best_idx]:.3f}",
                  fontsize=12, pad=6)
ax_arch.tick_params(axis="x", rotation=35)
# 标注最优
ax_arch.bar(config_labels[best_idx], cv_accs[best_idx] * 100,
            color=COLORS["teal"], edgecolor=COLORS["blue"], lw=2)
ax_arch.text(best_idx, cv_accs[best_idx] * 100 + 0.1,
             f"{cv_accs[best_idx]:.3f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=COLORS["blue"])
ax_arch.axhline(cv_accs[best_idx] * 100, color=COLORS["blue"],
                lw=1.2, ls="--", alpha=0.6)

# ── 面板 (b): 混淆矩阵 ───────────────────────────────────────────────────────
disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
disp.plot(ax=ax_cm, colorbar=False, cmap="Blues", values_format="d")
ax_cm.set_aspect("auto")
ax_cm.set_title(
    f"(b) 最优 MLP 混淆矩阵（测试集）\n"
    f"架构 {config_labels[best_idx]}，测试准确率={best_acc:.1%}，样本数={len(y_te)}",
    fontsize=12, pad=6)
ax_cm.set_xlabel("预测标签", fontsize=13)
ax_cm.set_ylabel("真实标签", fontsize=13)

# ── 面板 (c): 算法对比 ────────────────────────────────────────────────────────
bar_colors2 = [COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["teal"]]
ax_comp.bar(comp_names, [a * 100 for a in comp_accs],
            color=bar_colors2, edgecolor="white", lw=1.2)
ax_comp.set_ylim(90, 101)
ax_comp.set_ylabel("5折CV 准确率（%）", fontsize=13)
ax_comp.set_title("(c) 算法横向对比（相同数据集）\nMLP 与 SVM 性能接近，均优于 KNN 和决策树",
                  fontsize=12, pad=6)
for i, (name, acc) in enumerate(zip(comp_names, comp_accs)):
    ax_comp.text(i, acc * 100 + 0.1, f"{acc:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

fig.suptitle(
    f"MLP 手写数字识别（load_digits, $n=1797$, $d=64$, $K=10$）："
    f"架构搜索、混淆矩阵与跨算法对比",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_5_04_mlp_digits_case")
