"""
图 4.2.3  案例 4.2：California Housing 嵌套模型比较、VIF 与偏回归图
对应节次：4.2 统计推断与模型诊断（t/F 检验、残差分析）
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_2_03_california_case.py
输出路径：public/figures/ch04/fig4_2_03_california_case.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
apply_style()
# --- 1. 数据准备 ---
housing = fetch_california_housing()
X, y = housing.data, housing.target
feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
              "Population", "AveOccup", "Latitude", "Longitude"]
feat_names_cn = ["收入中位数", "房龄", "平均房间数", "平均卧室数",
                 "人口数量", "平均入住人数", "纬度", "经度"]
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
n_tr = len(y_tr)
p_total = X_tr_s.shape[1]
# --- 2. 辅助函数：OLS 拟合返回 R² ---
def ols_r2(X_feat, y_vec):
    n = len(y_vec)
    if X_feat.shape[1] == 0:
        return 0.0, np.sum((y_vec - y_vec.mean()) ** 2)
    X_mat = np.column_stack([np.ones(n), X_feat])
    beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    yhat = X_mat @ beta
    rss = np.sum((y_vec - yhat) ** 2)
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    r2 = 1 - rss / ss_tot
    return r2, rss
# --- 3. 嵌套模型定义与 R²、偏 F 计算 ---
model_defs = [
    ([], "截距"),
    ([0, 6, 7], "MedInc+Lat+Lon"),
    ([0, 1, 5, 6, 7], "+HouseAge+AveOccup"),
    (list(range(8)), "全部 8 个特征"),
]
r2_vals = []
rss_vals = []
for feat_idx, _ in model_defs:
    if len(feat_idx) == 0:
        r2_v, rss_v = 0.0, np.sum((y_tr - y_tr.mean()) ** 2)
    else:
        r2_v, rss_v = ols_r2(X_tr_s[:, feat_idx], y_tr)
    r2_vals.append(r2_v)
    rss_vals.append(rss_v)
# 偏 F：(RSS_prev - RSS_curr) / delta_p / MSE_full
_, rss_full = ols_r2(X_tr_s, y_tr)
mse_full = rss_full / (n_tr - p_total - 1)
partial_f = []
partial_f.append(None)
for i in range(1, len(model_defs)):
    delta_rss = rss_vals[i - 1] - rss_vals[i]
    delta_p = len(model_defs[i][0]) - len(model_defs[i - 1][0])
    f_val = (delta_rss / delta_p) / mse_full if delta_p > 0 else 0.0
    partial_f.append(f_val)
# --- 4. VIF 计算 ---
vif_vals = []
for j in range(p_total):
    others = [k for k in range(p_total) if k != j]
    r2_j, _ = ols_r2(X_tr_s[:, others], X_tr_s[:, j])
    vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf
    vif_vals.append(vif_j)
vif_vals = np.array(vif_vals)
# --- 5. 偏回归图：MedInc (index 0) ---
medinc_idx = 0
others_idx = [k for k in range(p_total) if k != medinc_idx]
def get_residuals(X_feat, y_vec):
    n = len(y_vec)
    if X_feat.shape[1] == 0:
        return y_vec - y_vec.mean()
    X_mat = np.column_stack([np.ones(n), X_feat])
    beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    return y_vec - X_mat @ beta
e_y = get_residuals(X_tr_s[:, others_idx], y_tr)
e_x = get_residuals(X_tr_s[:, others_idx], X_tr_s[:, medinc_idx])
rng = np.random.default_rng(7)
sub_idx = rng.choice(n_tr, 1500, replace=False)
e_x_sub = e_x[sub_idx]
e_y_sub = e_y[sub_idx]
slope_avp = np.dot(e_x_sub, e_y_sub) / np.dot(e_x_sub, e_x_sub)
# --- 6. 创建图形 ---
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
bar_colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]]
model_labels = ["截距模型\n(p=0)", "3 特征\n(MedInc+Lat+Lon)", "5 特征\n(+HouseAge+AveOccup)", "全部 8 特征"]
# --- 7. 面板(a)：嵌套模型 R² 与偏 F ---
ax = axes[0]
x_pos = np.arange(len(model_defs))
bars = ax.bar(x_pos, r2_vals, color=bar_colors, alpha=0.85, width=0.55, zorder=3)
for i in range(1, len(model_defs)):
    f_v = partial_f[i]
    top = r2_vals[i] + 0.025
    ax.annotate(
        f"偏$F={f_v:.0f}$\n$p<0.001$",
        xy=(x_pos[i], r2_vals[i]),
        xytext=(x_pos[i], top),
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["gray"], alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=1.0)
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylabel("$R^2$", fontsize=13)
ax.set_ylim(0, 1.1)
ax.set_title("(a) 嵌套模型 $R^2$ 与偏 $F$ 检验\n逐步增加特征，检验每组新特征的显著增量",
            fontsize=13, pad=8)
# --- 8. 面板(b)：VIF 水平条形图 ---
ax = axes[1]
vif_sort_idx = np.argsort(vif_vals)
vif_sorted = vif_vals[vif_sort_idx]
labels_sorted = [feat_names_cn[i] for i in vif_sort_idx]
vif_colors = []
for v in vif_sorted:
    if v >= 10:
        vif_colors.append(COLORS["red"])
    elif v >= 5:
        vif_colors.append(COLORS["orange"])
    else:
        vif_colors.append(COLORS["blue"])
ax.barh(np.arange(p_total), vif_sorted, color=vif_colors, alpha=0.85, height=0.6)
ax.axvline(5, color=COLORS["orange"], lw=1.8, ls="--", label="VIF=5")
ax.axvline(10, color=COLORS["red"], lw=1.8, ls="--", label="VIF=10")
for i, v in enumerate(vif_sorted):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=11)
ax.set_yticks(np.arange(p_total))
ax.set_yticklabels(labels_sorted, fontsize=12)
ax.set_xlabel("VIF", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(b) 方差膨胀因子（VIF）诊断多重共线性\n$\\mathrm{VIF}_j = 1/(1-R_j^2)$；VIF>10 须警惕系数不稳定",
            fontsize=13, pad=8)
# --- 9. 面板(c)：偏回归图 ---
ax = axes[2]
ax.scatter(e_x_sub, e_y_sub, color=COLORS["blue"], alpha=0.25, s=15, zorder=3)
x_line = np.array([e_x_sub.min(), e_x_sub.max()])
ax.plot(x_line, slope_avp * x_line, color=COLORS["red"], lw=2.2,
        label=f"斜率=$\\hat{{\\beta}}_{{\\mathrm{{MedInc}}}}={slope_avp:.3f}$")
ax.axhline(0, color=COLORS["gray"], lw=1.0, ls="--", alpha=0.6)
ax.axvline(0, color=COLORS["gray"], lw=1.0, ls="--", alpha=0.6)
ax.set_xlabel("$e($MedInc $|$ 其余变量$)$", fontsize=13)
ax.set_ylabel("$e(y|$ 其余变量$)$", fontsize=13)
ax.legend(fontsize=11)
ax.set_title("(c) 偏回归图（Added Variable Plot）：收入中位数\n斜率=$\\hat{\\beta}_{\\text{MedInc}}$（排除其余变量后的纯线性效应）",
            fontsize=13, pad=8)
# --- 10. 总标题与保存 ---
fig.suptitle("案例 4.2：California Housing 嵌套模型比较、VIF 诊断与偏回归分析",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, __file__, "fig4_2_03_california_case")
