"""
图 4.5.1  回归树 CART：递归分割、深度控制与代价复杂度剪枝
对应节次：4.5 树模型回归
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_5_01_cart_regression.py
输出路径：public/figures/ch04/fig4_5_01_cart_regression.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
apply_style()
# --- 1. 生成数据 ---
np.random.seed(42)
x = np.linspace(0, 10, 80)
y = np.sin(x) + 0.5 * np.cos(2 * x) + np.random.normal(0, 0.3, len(x))
X = x.reshape(-1, 1)
x_plot = np.linspace(0, 10, 500)
X_plot = x_plot.reshape(-1, 1)
y_true = np.sin(x_plot) + 0.5 * np.cos(2 * x_plot)
# --- 2. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
ax_a, ax_b, ax_c = axes
# --- 3. 面板 (a): CART 递归二叉分裂 ---
clf_a = DecisionTreeRegressor(max_depth=3, random_state=42)
clf_a.fit(X, y)
y_pred_a = clf_a.predict(X_plot)
ax_a.scatter(x, y, color=COLORS['gray'], s=25, alpha=0.7, zorder=3, label="数据点")
ax_a.plot(x_plot, y_pred_a, color=COLORS['blue'], lw=2, label="CART 拟合（depth=3）", zorder=4)
# Draw split boundaries
thresholds = np.unique(clf_a.tree_.threshold[clf_a.tree_.threshold != -2])
for thr in thresholds:
    ax_a.axvline(thr, color=COLORS['gray'], lw=1.2, ls='--', alpha=0.6)
# Label leaf regions with mean values
leaf_ids = np.where(clf_a.tree_.children_left == -1)[0]
region_means = clf_a.tree_.value[leaf_ids, 0, 0]
# Find region boundaries from prediction step function
prev_x = 0.0
prev_val = y_pred_a[0]
region_starts = [0.0]
for i in range(1, len(x_plot)):
    if abs(y_pred_a[i] - prev_val) > 1e-6:
        region_starts.append(x_plot[i])
        prev_val = y_pred_a[i]
region_starts.append(10.0)
for k in range(len(region_starts) - 1):
    xmid = (region_starts[k] + region_starts[k + 1]) / 2
    idx_region = (x_plot >= region_starts[k]) & (x_plot < region_starts[k + 1])
    if idx_region.any():
        ymean = y_pred_a[idx_region][0]
        ax_a.text(xmid, ymean + 0.25, f"$R_{{{k+1}}}$\n{ymean:.2f}",
                  ha='center', va='bottom', fontsize=9, color=COLORS['blue'])
ax_a.set_xlabel("$x$", fontsize=13)
ax_a.set_ylabel("$y$", fontsize=13)
ax_a.set_title("(a) CART 递归二叉分裂\n最小化加权 MSE 确定分裂点 $(x_j, s)$", fontsize=12)
ax_a.legend(fontsize=11)
# --- 4. 面板 (b): 树深度与过拟合 ---
depth_cfg = [
    (1, COLORS['gray'],   "depth=1（欠拟合）"),
    (3, COLORS['blue'],   "depth=3（适中）"),
    (7, COLORS['red'],    "depth=7（过拟合）"),
]
ax_b.scatter(x, y, color=COLORS['gray'], s=25, alpha=0.5, zorder=2)
ax_b.plot(x_plot, y_true, color=COLORS['green'], lw=2, ls='--', label="真实函数", zorder=5)
for depth, color, label in depth_cfg:
    clf_b = DecisionTreeRegressor(max_depth=depth, random_state=42)
    clf_b.fit(X, y)
    y_pred_b = clf_b.predict(X_plot)
    ax_b.plot(x_plot, y_pred_b, color=color, lw=2, label=label, zorder=4)
ax_b.set_xlabel("$x$", fontsize=13)
ax_b.set_ylabel("$y$", fontsize=13)
ax_b.set_title("(b) 树深度与过拟合\ndepth=1: 欠拟合；depth=7: 过拟合", fontsize=12)
ax_b.legend(fontsize=11)
# --- 5. 面板 (c): 代价复杂度剪枝路径 ---
clf_c = DecisionTreeRegressor(random_state=42)
clf_c.fit(X, y)
path = clf_c.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas
impurities = path.impurities
# Compute number of leaves and training MSE for each alpha
n_leaves_list = []
train_mse_list = []
for alpha in ccp_alphas:
    clf_tmp = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    clf_tmp.fit(X, y)
    n_leaves_list.append(clf_tmp.get_n_leaves())
    y_hat = clf_tmp.predict(X)
    train_mse_list.append(np.mean((y - y_hat) ** 2))
n_leaves_arr = np.array(n_leaves_list)
train_mse_arr = np.array(train_mse_list)
# Filter to meaningful range (exclude trivial single-leaf and very large trees)
valid = (n_leaves_arr > 1) & (ccp_alphas > 0)
alpha_valid = ccp_alphas[valid]
leaves_valid = n_leaves_arr[valid]
mse_valid = train_mse_arr[valid]
# Find optimal alpha (elbow in MSE): use minimal alpha where MSE is within 5% of max
mse_range = mse_valid.max() - mse_valid.min()
opt_mask = mse_valid < mse_valid.min() + 0.1 * mse_range
opt_alpha = alpha_valid[opt_mask][-1] if opt_mask.any() else alpha_valid[len(alpha_valid) // 3]
color_leaves = COLORS['blue']
color_mse = COLORS['red']
ax_c.semilogx(alpha_valid, leaves_valid, color=color_leaves, lw=2, marker='o', ms=4, label="|T| 叶子数")
ax_c.set_xlabel(r"$\alpha$（代价复杂度参数）", fontsize=13)
ax_c.set_ylabel("|T| 叶子数", fontsize=13, color=color_leaves)
ax_c.tick_params(axis='y', labelcolor=color_leaves)
ax2 = ax_c.twinx()
ax2.semilogx(alpha_valid, mse_valid, color=color_mse, lw=2, ls='--', marker='s', ms=4, label="训练 MSE")
ax2.set_ylabel("训练 MSE", fontsize=13, color=color_mse)
ax2.tick_params(axis='y', labelcolor=color_mse)
ax_c.axvline(opt_alpha, color=COLORS['orange'], lw=1.5, ls='--', label=f"最优 α≈{opt_alpha:.4f}")
lines1, labels1 = ax_c.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_c.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')
ax_c.set_title(r"(c) 代价复杂度剪枝路径" + "\n" + r"$R_\alpha(T) = R(T) + \alpha|T|$", fontsize=12)
# --- 6. 总标题与保存 ---
fig.suptitle("回归树 CART：递归分割、深度控制与代价复杂度剪枝",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig(fig, __file__, "fig4_5_01_cart_regression")
