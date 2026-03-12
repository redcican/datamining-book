"""
图 3.6.2  集成规模对偏差-方差的影响：OOB 误差曲线与分解对比
对应节次：3.6 集成学习算法
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch03/fig3_6_02_bias_variance_ensemble.py
输出路径：public/figures/ch03/fig3_6_02_bias_variance_ensemble.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)

apply_style()

# ── 数据 ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                            random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 面板(a): Bagging/RF 随 n_estimators 的测试误差曲线 ─────────────────────────
ns = np.arange(1, 201, 4)
bag_err, rf_err = [], []
for n in ns:
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None),
                            n_estimators=int(n), random_state=42, n_jobs=-1)
    bag.fit(X_tr_s, y_tr)
    bag_err.append(1 - bag.score(X_te_s, y_te))

    rf = RandomForestClassifier(n_estimators=int(n), random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    rf_err.append(1 - rf.score(X_te_s, y_te))

bag_err = np.array(bag_err)
rf_err  = np.array(rf_err)

# ── 面板(b): Boosting 随轮次的训练/测试误差曲线 ────────────────────────────────
n_boost = np.arange(1, 201, 4)
ada_tr, ada_te = [], []
gbm_tr, gbm_te = [], []
for n in n_boost:
    ada = AdaBoostClassifier(n_estimators=int(n), learning_rate=0.5,
                             algorithm="SAMME", random_state=42)
    ada.fit(X_tr_s, y_tr)
    ada_tr.append(1 - ada.score(X_tr_s, y_tr))
    ada_te.append(1 - ada.score(X_te_s, y_te))

    gbm = GradientBoostingClassifier(n_estimators=int(n), learning_rate=0.1,
                                     max_depth=3, random_state=42)
    gbm.fit(X_tr_s, y_tr)
    gbm_tr.append(1 - gbm.score(X_tr_s, y_tr))
    gbm_te.append(1 - gbm.score(X_te_s, y_te))

ada_tr, ada_te = np.array(ada_tr), np.array(ada_te)
gbm_tr, gbm_te = np.array(gbm_tr), np.array(gbm_te)

# ── 面板(c): 偏差-方差分解——通过 bootstrap 重采样估计 ─────────────────────────
def bias_var_estimate(clf_fn, X_tr, y_tr, X_te, y_te, n_boot=100, seed=42):
    """用 100 个 Bootstrap 样本估计偏差^2 和方差"""
    rng = np.random.default_rng(seed)
    preds = np.zeros((n_boot, len(y_te)), dtype=int)
    for b in range(n_boot):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        clf = clf_fn()
        clf.fit(X_tr[idx], y_tr[idx])
        preds[b] = clf.predict(X_te)
    # 主预测（众数）
    from scipy import stats
    main_pred = stats.mode(preds, axis=0, keepdims=True).mode[0]
    bias2 = np.mean(main_pred != y_te)
    variance = np.mean(preds != main_pred)
    noise = 0.0
    return bias2, variance

methods_bv = [
    ("单棵决策树",      lambda: DecisionTreeClassifier(max_depth=None, random_state=42)),
    ("Bagging\n(T=50)",  lambda: BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ("随机森林\n(T=50)", lambda: RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ("AdaBoost\n(T=50)", lambda: AdaBoostClassifier(n_estimators=50, algorithm="SAMME", random_state=42)),
    ("GBDT\n(T=50)",     lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
]
bv_names  = [m[0] for m in methods_bv]
bv_bias   = []
bv_var    = []
for name, fn in methods_bv:
    b2, var = bias_var_estimate(fn, X_tr_s, y_tr, X_te_s, y_te, n_boot=80)
    bv_bias.append(b2)
    bv_var.append(var)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.subplots_adjust(wspace=0.40)

# ── 面板 (a): Bagging 与 RF 误差随 n_estimators ───────────────────────────────
ax = axes[0]
ax.plot(ns, bag_err * 100, color=COLORS["blue"],   lw=2.0, label="Bagging（决策树）")
ax.plot(ns, rf_err  * 100, color=COLORS["teal"],   lw=2.0, label="随机森林")
ax.axhline(1 - DecisionTreeClassifier(random_state=42).fit(X_tr_s, y_tr).score(X_te_s, y_te),
           color=COLORS["gray"], lw=1.2, ls="--", label="单棵决策树")
ax.set_xlabel("基学习器数量 $T$", fontsize=13)
ax.set_ylabel("测试误差率（%）", fontsize=13)
ax.set_xlim(1, 200)
ax.set_title("(a) Bagging / 随机森林：测试误差随 $T$ 变化\n"
             "误差随 $T$ 增大快速下降后收敛（方差减小，偏差不变）",
             fontsize=12, pad=6)
ax.legend(fontsize=11)
ax.annotate("收敛区域\n（方差饱和）",
            xy=(150, float(rf_err[-20:].mean() * 100)),
            xytext=(100, float(rf_err[-20:].mean() * 100) + 3),
            arrowprops=dict(arrowstyle="->", color=COLORS["teal"], lw=1.2),
            fontsize=11, color=COLORS["teal"])

# ── 面板 (b): Boosting 训练/测试误差 ─────────────────────────────────────────
ax = axes[1]
ax.plot(n_boost, ada_tr * 100, color=COLORS["orange"], lw=1.8, label="AdaBoost 训练误差")
ax.plot(n_boost, ada_te * 100, color=COLORS["orange"], lw=1.8, ls="--", label="AdaBoost 测试误差")
ax.plot(n_boost, gbm_tr * 100, color=COLORS["red"],   lw=1.8, label="GBDT 训练误差")
ax.plot(n_boost, gbm_te * 100, color=COLORS["red"],   lw=1.8, ls="--", label="GBDT 测试误差")
ax.set_xlabel("基学习器数量 $T$（轮次）", fontsize=13)
ax.set_ylabel("误差率（%）", fontsize=13)
ax.set_xlim(1, 200)
ax.set_title("(b) Boosting：训练/测试误差随轮次变化\n"
             "GBDT 训练误差单调下降；过多轮次可能导致测试误差回升（过拟合）",
             fontsize=12, pad=6)
ax.legend(fontsize=11)

# ── 面板 (c): 偏差-方差分解柱状图 ──────────────────────────────────────────────
ax = axes[2]
x_pos = np.arange(len(bv_names))
w = 0.38
bars_b = ax.bar(x_pos - w/2, [v * 100 for v in bv_bias], w,
                color=COLORS["blue"], label="偏差（$\\mathrm{Bias}^2$）", alpha=0.9)
bars_v = ax.bar(x_pos + w/2, [v * 100 for v in bv_var], w,
                color=COLORS["red"],  label="方差（$\\mathrm{Var}$）", alpha=0.9)
ax.set_xticks(x_pos)
ax.set_xticklabels(bv_names, fontsize=11)
ax.set_ylabel("误差分量（%）", fontsize=13)
ax.set_title("(c) 偏差-方差分解（Bootstrap 估计，乳腺癌数据集）\n"
             "Bagging/RF 主要降方差；Boosting 主要降偏差",
             fontsize=12, pad=6)
ax.legend(fontsize=11)
for bar in bars_b:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9)
for bar in bars_v:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}",
            ha="center", va="bottom", fontsize=9)

fig.suptitle(
    "集成规模与偏差-方差权衡：Bagging 主降方差，Boosting 主降偏差\n"
    "数据集：威斯康辛乳腺癌（$n=426$ 训练，$d=30$，标准化）",
    fontsize=13, y=1.01)

save_fig(fig, __file__, "fig3_6_02_bias_variance_ensemble")
