"""
图 3.4.4  KNN 手写数字识别：样本展示、k 值选择与最优模型混淆矩阵
对应节次：3.4 K 近邻（KNN）算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_4_04_digits_case.py
输出路径：public/figures/ch03/fig3_4_04_digits_case.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

apply_style()

# ── 数据准备 ──────────────────────────────────────────────────────────────────
data = load_digits()
X, y = data.data, data.target       # (1797, 64), labels 0~9
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── k 扫描 ───────────────────────────────────────────────────────────────────
k_range  = np.arange(1, 31)
cv_accs  = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                            X_tr_s, y_tr, cv=5, scoring="accuracy").mean()
            for k in k_range]
te_accs  = [KNeighborsClassifier(n_neighbors=k).fit(X_tr_s, y_tr).score(X_te_s, y_te)
            for k in k_range]
cv_accs  = np.array(cv_accs)
te_accs  = np.array(te_accs)
best_k   = k_range[np.argmax(cv_accs)]

# ── 最优模型 ──────────────────────────────────────────────────────────────────
best_clf = KNeighborsClassifier(n_neighbors=best_k)
best_clf.fit(X_tr_s, y_tr)
y_pred   = best_clf.predict(X_te_s)
best_acc = accuracy_score(y_te, y_pred)
cm       = confusion_matrix(y_te, y_pred)

# ── Figure: 3 panels ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38,
                       height_ratios=[1.15, 1.0])
ax_imgs = fig.add_subplot(gs[0, :])     # top: digit images
ax_k    = fig.add_subplot(gs[1, 0])    # bottom-left: k curve
ax_cm   = fig.add_subplot(gs[1, 1])    # bottom-right: confusion matrix

# ── Panel (a): 样本图像展示 ────────────────────────────────────────────────────
# 每个数字取 4 个代表样本，共 10×4 = 40 张图
n_cols, n_per_class = 10, 4
sample_imgs  = []
sample_labels = []
for digit in range(10):
    idx = np.where(y_tr == digit)[0][:n_per_class]
    sample_imgs.extend(X_tr[idx])
    sample_labels.extend(y_tr[idx])

ax_imgs.set_xlim(0, n_cols * 2.4)
ax_imgs.set_ylim(0, n_per_class * 2.6)
ax_imgs.axis("off")
ax_imgs.set_title("(a) 手写数字数据集（sklearn.datasets.load_digits）：各数字代表样本（8×8像素）",
                  fontsize=12, pad=6)

for i, (img, lbl) in enumerate(zip(sample_imgs, sample_labels)):
    row_i = i // n_cols
    col_i = i % n_cols
    x0    = col_i * 2.4 + 0.1
    y0    = (n_per_class - 1 - row_i) * 2.6 + 0.3
    # 在图上放置 8×8 图像
    newax = fig.add_axes(
        [ax_imgs.get_position().x0 + (x0 / (n_cols * 2.4)) * ax_imgs.get_position().width,
         ax_imgs.get_position().y0 + (y0 / (n_per_class * 2.6)) * ax_imgs.get_position().height,
         ax_imgs.get_position().width / (n_cols * 2.4) * 1.9,
         ax_imgs.get_position().height / (n_per_class * 2.6) * 2.2],
        facecolor="none"
    )
    newax.imshow(img.reshape(8, 8), cmap="gray_r", interpolation="nearest")
    newax.set_xticks([]); newax.set_yticks([])
    for spine in newax.spines.values():
        spine.set_visible(False)

# ── Panel (b): k vs 准确率曲线 ──────────────────────────────────────────────────
ax = ax_k
ax.plot(k_range, cv_accs, color=COLORS["teal"], lw=2.0,
        marker="o", markersize=4, label="5折CV准确率")
ax.plot(k_range, te_accs, color=COLORS["red"], lw=1.8,
        ls="--", marker="s", markersize=4, label="测试集准确率")
ax.axvline(best_k, color=COLORS["orange"], lw=1.5, ls=":",
           label=f"最优 $k^*={best_k}$")
ax.scatter([best_k], [cv_accs[best_k - 1]], s=120,
           color=COLORS["orange"], zorder=5)
ax.set_xlabel("邻居数 $k$", fontsize=13)
ax.set_ylabel("准确率", fontsize=13)
ax.set_xlim(1, 30)
ax.set_ylim(0.93, 1.005)
ax.set_title(f"(b) $k$ 值选择（5折CV）\n"
             f"最优 $k^*={best_k}$，测试准确率={te_accs[best_k-1]:.3f}",
             fontsize=12, pad=6)
ax.legend(fontsize=11)

# ── Panel (c): 混淆矩阵 ────────────────────────────────────────────────────────
ax = ax_cm
disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
ax.set_title(
    f"(c) 最优 KNN（$k={best_k}$）的混淆矩阵\n"
    f"测试准确率={best_acc:.1%}，样本数={len(y_te)}",
    fontsize=12, pad=6)
ax.set_xlabel("预测标签", fontsize=13)
ax.set_ylabel("真实标签", fontsize=13)

fig.suptitle(
    f"手写数字识别（load_digits, $n=1797$, $d=64$, $K=10$类）：KNN 从 1-NN 到 30-NN 的性能演变",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_4_04_digits_case")
