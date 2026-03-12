"""
图 3.5.3  MLP 训练过程：损失曲线、梯度与正则化效果
对应节次：3.5 神经网络分类方法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_5_03_training_curves.py
输出路径：public/figures/ch03/fig3_5_03_training_curves.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

apply_style()

rng = np.random.default_rng(42)

# ── 数据准备 ──────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=600, noise=0.28, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 训练三种配置 ─────────────────────────────────────────────────────────────
def train_record(hidden, alpha, max_iter=800):
    """训练并记录逐 epoch 的训练/测试损失与准确率"""
    from sklearn.neural_network import MLPClassifier
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        solver="sgd", learning_rate_init=0.02,
                        alpha=alpha, max_iter=1, warm_start=True,
                        random_state=42)
    for _ in range(max_iter):
        clf.fit(X_tr_s, y_tr)
        train_loss.append(clf.loss_)
        proba_tr = clf.predict_proba(X_tr_s)
        proba_te = clf.predict_proba(X_te_s)
        eps = 1e-12
        tl = -np.mean(np.log(proba_te[np.arange(len(y_te)), y_te] + eps))
        test_loss.append(tl)
        train_acc.append(clf.score(X_tr_s, y_tr))
        test_acc.append(clf.score(X_te_s, y_te))
    return clf, np.array(train_loss), np.array(test_loss), \
           np.array(train_acc), np.array(test_acc)

clf_ov,  tr_l_ov,  te_l_ov,  tr_a_ov,  te_a_ov  = train_record((128, 64, 32), alpha=0.0)
clf_reg, tr_l_reg, te_l_reg, tr_a_reg, te_a_reg  = train_record((128, 64, 32), alpha=0.01)
clf_sm,  tr_l_sm,  te_l_sm,  tr_a_sm,  te_a_sm   = train_record((16, 8),      alpha=0.0)

epochs = np.arange(1, 801)

# ── 绘图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.subplots_adjust(wspace=0.38)

# ── 面板 (a): 过拟合 vs 正则化 损失曲线 ──────────────────────────────────────
ax = axes[0]
ax.plot(epochs, tr_l_ov,  color=COLORS["blue"],   lw=1.8, label="过拟合模型 — 训练损失")
ax.plot(epochs, te_l_ov,  color=COLORS["blue"],   lw=1.8, ls="--", label="过拟合模型 — 测试损失")
ax.plot(epochs, tr_l_reg, color=COLORS["teal"],   lw=1.8, label="L2 正则 — 训练损失")
ax.plot(epochs, te_l_reg, color=COLORS["teal"],   lw=1.8, ls="--", label="L2 正则 — 测试损失")
ax.plot(epochs, tr_l_sm,  color=COLORS["orange"], lw=1.8, label="小模型 — 训练损失")
ax.plot(epochs, te_l_sm,  color=COLORS["orange"], lw=1.8, ls="--", label="小模型 — 测试损失")
ax.set_xlabel("训练轮次（Epoch）", fontsize=13)
ax.set_ylabel("交叉熵损失", fontsize=13)
ax.set_xlim(1, 800)
ax.set_ylim(0, 1.0)
ax.set_title("(a) 训练/测试损失曲线对比\n过拟合模型测试损失上升，正则化与小模型保持稳定", fontsize=12, pad=6)
ax.legend(fontsize=10, ncol=1)

# 过拟合区域标注
ax.axvspan(300, 800, alpha=0.06, color=COLORS["red"])
ax.text(550, 0.75, "过拟合区域", fontsize=11, color=COLORS["red"],
        ha="center", va="center",
        bbox=dict(fc="white", ec=COLORS["red"], alpha=0.85, pad=2, boxstyle="round"))

# ── 面板 (b): 准确率曲线 ──────────────────────────────────────────────────────
ax = axes[1]
ax.plot(epochs, tr_a_ov * 100,  color=COLORS["blue"],   lw=1.8, label="过拟合模型 — 训练")
ax.plot(epochs, te_a_ov * 100,  color=COLORS["blue"],   lw=1.8, ls="--", label="过拟合模型 — 测试")
ax.plot(epochs, tr_a_reg * 100, color=COLORS["teal"],   lw=1.8, label="L2 正则 — 训练")
ax.plot(epochs, te_a_reg * 100, color=COLORS["teal"],   lw=1.8, ls="--", label="L2 正则 — 测试")
ax.plot(epochs, tr_a_sm * 100,  color=COLORS["orange"], lw=1.8, label="小模型 — 训练")
ax.plot(epochs, te_a_sm * 100,  color=COLORS["orange"], lw=1.8, ls="--", label="小模型 — 测试")
ax.set_xlabel("训练轮次（Epoch）", fontsize=13)
ax.set_ylabel("准确率（%）", fontsize=13)
ax.set_xlim(1, 800)
ax.set_ylim(60, 102)
ax.set_title("(b) 训练/测试准确率曲线\n训练准确率↑而测试准确率停滞 → 过拟合信号", fontsize=12, pad=6)
ax.legend(fontsize=10, ncol=1)

# ── 面板 (c): 最终决策边界对比 ──────────────────────────────────────────────────
ax = axes[2]
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                     np.linspace(x2_min, x2_max, 300))
grid_s = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

for clf_plot, color, label in [
    (clf_ov,  COLORS["blue"],   "过拟合"),
    (clf_reg, COLORS["teal"],   "L2 正则"),
]:
    Z = clf_plot.predict_proba(grid_s)[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0.5], colors=[color], linewidths=2.0,
               linestyles=["-" if label=="过拟合" else "--"])

ax.scatter(X_te[y_te == 0, 0], X_te[y_te == 0, 1],
           c=COLORS["blue"], s=20, alpha=0.6, edgecolors="none")
ax.scatter(X_te[y_te == 1, 0], X_te[y_te == 1, 1],
           c=COLORS["red"],  s=20, alpha=0.6, edgecolors="none")

# 手动图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS["blue"], lw=2,   label=f"过拟合边界（测试 {te_a_ov[-1]:.1%}）"),
    Line2D([0], [0], color=COLORS["teal"], lw=2, ls="--", label=f"L2正则边界（测试 {te_a_reg[-1]:.1%}）"),
]
ax.legend(handles=legend_elements, fontsize=11)
ax.set_xlabel("$x_1$", fontsize=13)
ax.set_ylabel("$x_2$", fontsize=13)
ax.set_title("(c) 决策边界对比（测试集散点）\n过拟合边界扭曲，正则化边界更平滑泛化更佳", fontsize=12, pad=6)

fig.suptitle(
    "MLP 训练动态：损失曲线揭示过拟合，L2 正则化抑制模型复杂度\n"
    "数据集：make_moons（$n=600$，noise=0.28），模型大（128-64-32）vs 小（16-8）",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_5_03_training_curves")
