"""
fig7_6_02_iforest_parameter.py
隔离森林参数对检测性能的影响
(a) 树数量 t 的影响  (b) 子采样大小 ψ 的影响
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成数据 ──────────────────────────────────────────────────────
n1, n2, n_out = 300, 200, 30
cluster1 = np.random.multivariate_normal([2, 2], [[0.8, 0.2], [0.2, 0.6]], n1)
cluster2 = np.random.multivariate_normal([-3, -2], [[0.7, -0.1], [-0.1, 0.9]], n2)
outliers = np.column_stack([
    np.random.uniform(-8, 8, n_out),
    np.random.uniform(-8, 8, n_out),
])
X = np.vstack([cluster1, cluster2, outliers])
y_true = np.concatenate([np.zeros(n1 + n2), np.ones(n_out)])
# ── 辅助函数 ──────────────────────────────────────────────────────
def evaluate_iforest(X, y_true, n_estimators, max_samples, n_repeats=5):
    """多次运行取 AUC 均值和标准差。"""
    aucs = []
    for i in range(n_repeats):
        clf = IsolationForest(n_estimators=n_estimators,
                              max_samples=max_samples,
                              random_state=42 + i, contamination="auto")
        clf.fit(X)
        scores = -clf.decision_function(X)  # 越高越异常
        aucs.append(roc_auc_score(y_true, scores))
    return np.mean(aucs), np.std(aucs)
# ── 实验 (a): 树数量 ─────────────────────────────────────────────
t_values = [10, 25, 50, 100, 200, 300, 500]
auc_t_mean, auc_t_std = [], []
for t in t_values:
    m, s = evaluate_iforest(X, y_true, n_estimators=t, max_samples=256)
    auc_t_mean.append(m)
    auc_t_std.append(s)
auc_t_mean = np.array(auc_t_mean)
auc_t_std = np.array(auc_t_std)
# ── 实验 (b): 子采样大小 ─────────────────────────────────────────
psi_values = [32, 64, 128, 256, 512, 1024]
auc_psi_mean, auc_psi_std = [], []
for psi in psi_values:
    m, s = evaluate_iforest(X, y_true, n_estimators=100, max_samples=min(psi, len(X)))
    auc_psi_mean.append(m)
    auc_psi_std.append(s)
auc_psi_mean = np.array(auc_psi_mean)
auc_psi_std = np.array(auc_psi_std)
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("图 7.6.2　隔离森林参数对检测性能的影响",
             fontsize=20, fontweight="bold", y=1.02)
# ══════════════════════════════════════════════════════════════════
# (a) 树数量 t
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.errorbar(t_values, auc_t_mean, yerr=auc_t_std, fmt="o-",
            color=COLORS["blue"], capsize=4, capthick=1.5, lw=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label="AUC (均值 ± 标准差)")
ax.axvline(x=100, color=COLORS["gray"], ls="--", lw=1.5, alpha=0.7)
ax.annotate("t ≈ 100 后趋于收敛",
            xy=(100, auc_t_mean[t_values.index(100)]),
            xytext=(280, auc_t_mean.max() + 0.002),
            fontsize=13, color=COLORS["gray"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["gray"], alpha=0.9))
ax.set_xlabel("树数量 $t$", fontsize=15)
ax.set_ylabel("AUC", fontsize=15)
ax.set_title("(a) 树数量 $t$ 的影响", fontsize=17)
ax.tick_params(labelsize=13)
ax.legend(fontsize=13, loc="lower right")
# ══════════════════════════════════════════════════════════════════
# (b) 子采样大小 ψ
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.errorbar(psi_values, auc_psi_mean, yerr=auc_psi_std, fmt="s-",
            color=COLORS["red"], capsize=4, capthick=1.5, lw=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label="AUC (均值 ± 标准差)")
ax.axvline(x=256, color=COLORS["gray"], ls="--", lw=1.5, alpha=0.7)
ax.annotate("默认值",
            xy=(256, auc_psi_mean[psi_values.index(256)]),
            xytext=(600, auc_psi_mean.max() + 0.002),
            fontsize=14, fontweight="bold", color=COLORS["gray"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS["gray"], alpha=0.9))
ax.set_xlabel("子采样大小 $\\psi$", fontsize=15)
ax.set_ylabel("AUC", fontsize=15)
ax.set_title("(b) 子采样大小 $\\psi$ 的影响", fontsize=17)
ax.set_xscale("log", base=2)
ax.set_xticks(psi_values)
ax.set_xticklabels([str(v) for v in psi_values])
ax.tick_params(labelsize=13)
ax.legend(fontsize=13, loc="lower right")
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout()
save_fig(fig, __file__, "fig7_6_02_iforest_parameter")
