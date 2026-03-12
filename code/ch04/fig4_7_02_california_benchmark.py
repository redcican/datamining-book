"""
图 4.7.2  California Housing 全方法基准测试：精度–速度–可解释性三角权衡
对应节次：4.7 回归算法系统比较与案例分析
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_7_02_california_benchmark.py
输出路径：public/figures/ch04/fig4_7_02_california_benchmark.png
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

apply_style()

# ── 数据加载 ──────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── 模型定义 ──────────────────────────────────────────────────────────────────
models_scaled = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=5000),
}
models_raw = {
    "CART depth=5": DecisionTreeRegressor(max_depth=5, random_state=42),
    "CART depth=10": DecisionTreeRegressor(max_depth=10, random_state=42),
    "RF B=200": RandomForestRegressor(n_estimators=200, max_features=0.33, random_state=42, n_jobs=-1),
    "GBDT M=300": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                             learning_rate=0.1, subsample=0.8, random_state=42),
}

# ── 训练与评估 ────────────────────────────────────────────────────────────────
results = {}
for name, mdl in models_scaled.items():
    t0 = time.time()
    mdl.fit(X_tr_s, y_tr)
    train_t = time.time() - t0
    rmse_tr = np.sqrt(mean_squared_error(y_tr, mdl.predict(X_tr_s)))
    rmse_te = np.sqrt(mean_squared_error(y_te, mdl.predict(X_te_s)))
    r2_te   = r2_score(y_te, mdl.predict(X_te_s))
    results[name] = {'rmse_tr': rmse_tr, 'rmse_te': rmse_te, 'r2': r2_te, 'time': train_t}

for name, mdl in models_raw.items():
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    train_t = time.time() - t0
    rmse_tr = np.sqrt(mean_squared_error(y_tr, mdl.predict(X_tr)))
    rmse_te = np.sqrt(mean_squared_error(y_te, mdl.predict(X_te)))
    r2_te   = r2_score(y_te, mdl.predict(X_te))
    results[name] = {'rmse_tr': rmse_tr, 'rmse_te': rmse_te, 'r2': r2_te, 'time': train_t}

# 排序（按测试RMSE升序）
sorted_names = sorted(results, key=lambda k: results[k]['rmse_te'])

# ── 可解释性打分（主观，1–5） ──────────────────────────────────────────────────
interp_score = {
    "OLS": 5, "Ridge": 5, "Lasso": 5,
    "CART depth=5": 4, "CART depth=10": 2,
    "RF B=200": 2, "GBDT M=300": 1,
}
# 精度分（1–5，基于RMSE倒数归一化）
max_rmse = max(r['rmse_te'] for r in results.values())
min_rmse = min(r['rmse_te'] for r in results.values())

def perf_score(rmse):
    return 1 + 4 * (max_rmse - rmse) / (max_rmse - min_rmse)

# ── 颜色映射 ──────────────────────────────────────────────────────────────────
bar_colors = {
    "OLS":          COLORS['gray'],
    "Ridge":        COLORS['blue'],
    "Lasso":        COLORS['teal'],
    "CART depth=5": COLORS['orange'],
    "CART depth=10":COLORS['red'],
    "RF B=200":     COLORS['green'],
    "GBDT M=300":   COLORS['purple'],
}

# ── 作图 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
ax_a, ax_b, ax_c = axes

# ── 面板 (a): RMSE 对比（训练 vs 测试） ─────────────────────────────────────
x_idx = np.arange(len(sorted_names))
bar_w = 0.38

rmse_tr_vals = [results[n]['rmse_tr'] for n in sorted_names]
rmse_te_vals = [results[n]['rmse_te'] for n in sorted_names]
clrs         = [bar_colors[n]         for n in sorted_names]

bars_tr = ax_a.bar(x_idx - bar_w/2, rmse_tr_vals, bar_w, label='训练 RMSE',
                   color=clrs, alpha=0.4, edgecolor='gray', linewidth=0.5)
bars_te = ax_a.bar(x_idx + bar_w/2, rmse_te_vals, bar_w, label='测试 RMSE',
                   color=clrs, alpha=0.9, edgecolor='gray', linewidth=0.5)

for i, (tr, te) in enumerate(zip(rmse_tr_vals, rmse_te_vals)):
    ax_a.text(i - bar_w/2, tr + 0.005, f'{tr:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax_a.text(i + bar_w/2, te + 0.005, f'{te:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

short_names = [n.replace(' ', '\n') for n in sorted_names]
ax_a.set_xticks(x_idx)
ax_a.set_xticklabels(short_names, fontsize=9)
ax_a.set_ylabel('RMSE（$10^4$ 美元）', fontsize=12)
ax_a.set_title('(a) 各方法训练/测试 RMSE\n（California Housing，80/20 分割）',
               fontsize=13, fontweight='bold')
ax_a.legend(fontsize=10)

# 添加方差差值（过拟合指示箭头）
for i, (tr, te) in enumerate(zip(rmse_tr_vals, rmse_te_vals)):
    gap = te - tr
    if gap > 0.03:
        ax_a.annotate('', xy=(i + bar_w/2, te), xytext=(i - bar_w/2, tr),
                      arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

# ── 面板 (b): R² vs 训练时间散点图（气泡=可解释性） ─────────────────────────
for name in results:
    r2  = results[name]['r2']
    t   = max(results[name]['time'], 0.01)   # 避免log(0)
    ps  = interp_score[name]
    col = bar_colors[name]
    ax_b.scatter(np.log10(t), r2, s=ps * 60 + 40, color=col, alpha=0.85, zorder=4, edgecolors='k', linewidths=0.5)
    ax_b.annotate(name.replace(' ', '\n'), xy=(np.log10(t), r2),
                  xytext=(5, -12), textcoords='offset points', fontsize=8.5, color='black')

# 气泡图图例
for score, label in [(1, '可解释性低(1)'), (3, '可解释性中(3)'), (5, '可解释性高(5)')]:
    ax_b.scatter([], [], s=score * 60 + 40, color='gray', alpha=0.6,
                 label=label, edgecolors='k', linewidths=0.5)

ax_b.set_xlabel('训练时间 $\\log_{10}$(s)', fontsize=12)
ax_b.set_ylabel('测试 $R^2$', fontsize=12)
ax_b.set_title('(b) 精度–速度–可解释性三角权衡\n（气泡大小=可解释性，越大越可解释）',
               fontsize=13, fontweight='bold')
ax_b.legend(fontsize=9, loc='lower right')

# ── 面板 (c): 学习曲线（RF 与 Ridge 对比） ────────────────────────────────────
train_sizes_abs = np.array([500, 1000, 2000, 4000, 8000, 12000, 16000])
train_sizes_abs = train_sizes_abs[train_sizes_abs < len(X_tr)]
train_sizes_frac = train_sizes_abs / len(X_tr)

lc_models = {
    "Ridge":    (Ridge(alpha=1.0), X_tr_s, X_te_s),
    "RF B=100": (RandomForestRegressor(n_estimators=100, max_features=0.33, random_state=42, n_jobs=-1), X_tr, X_te),
    "GBDT M=200": (GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                              learning_rate=0.1, random_state=42), X_tr, X_te),
}
lc_colors = [COLORS['blue'], COLORS['green'], COLORS['purple']]

for (name, (mdl, Xtr_lc, Xte_lc)), col in zip(lc_models.items(), lc_colors):
    tr_scores, val_scores = [], []
    for n_samp in train_sizes_abs:
        idx = np.random.choice(len(Xtr_lc), n_samp, replace=False)
        mdl.fit(Xtr_lc[idx], y_tr[idx])
        tr_scores.append(np.sqrt(mean_squared_error(y_tr[idx], mdl.predict(Xtr_lc[idx]))))
        val_scores.append(np.sqrt(mean_squared_error(y_te, mdl.predict(Xte_lc))))
    ax_c.plot(train_sizes_abs, tr_scores, '--', color=col, lw=1.5, alpha=0.7)
    ax_c.plot(train_sizes_abs, val_scores, '-', color=col, lw=2.0, label=name)
    ax_c.fill_between(train_sizes_abs, tr_scores, val_scores, color=col, alpha=0.08)

ax_c.set_xlabel('训练样本数 $n$', fontsize=12)
ax_c.set_ylabel('RMSE', fontsize=12)
ax_c.set_title('(c) 学习曲线对比（实线=测试，虚线=训练）\n随样本量增加，复杂模型相对优势扩大',
               fontsize=13, fontweight='bold')
ax_c.legend(fontsize=10)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_7_02_california_benchmark')
