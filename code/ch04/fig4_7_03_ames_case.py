"""
图 4.7.3  Ames Housing 综合建模案例：全流程比较、学习曲线与特征重要性
对应节次：4.7 回归算法系统比较与案例分析
运行方式：MPLBACKEND=Agg PYTHONPATH=code python3 code/ch04/fig4_7_03_ames_case.py
输出路径：public/figures/ch04/fig4_7_03_ames_case.png
备注：使用 sklearn 的 fetch_openml 获取 Ames Housing（需要联网首次下载）
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig, COLORS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

warnings.filterwarnings('ignore')
apply_style()

# ── 1. 数据加载 ────────────────────────────────────────────────────────────────
print("正在加载 Ames Housing 数据集...")
try:
    house = fetch_openml(name='house_prices', version=1, as_frame=True, parser='auto')
    df = house.frame
    y_col = 'SalePrice'
    X_raw = df.drop(columns=[y_col])
    y_raw = np.log1p(df[y_col].astype(float).values)   # log1p 变换使目标更接近正态
except Exception as e:
    print(f"OpenML 加载失败，回退到 California Housing: {e}")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df   = data.frame
    X_raw = df.drop(columns=['MedHouseVal'])
    y_raw = df['MedHouseVal'].values
    # 模拟 Ames 列类型
    X_raw['DummyCat'] = (X_raw['AveRooms'] > 5).map({True: 'High', False: 'Low'})

# ── 2. 特征预处理：数值 + 类别分离 ────────────────────────────────────────────
if hasattr(X_raw, 'dtypes'):
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
else:
    num_cols = list(X_raw.columns)
    cat_cols = []

num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
)

if cat_cols:
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])
else:
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols)])

X_proc = preprocessor.fit_transform(X_raw)
feature_names_out = (
    num_cols + [f'cat_{c}' for c in cat_cols]
)

X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y_raw, test_size=0.2, random_state=42)
print(f"数据集: {X_tr.shape[0]} 训练 / {X_te.shape[0]} 测试，特征数={X_proc.shape[1]}")

# ── 3. 模型训练与评估 ──────────────────────────────────────────────────────────
models = {
    "Ridge":    Ridge(alpha=10.0),
    "Lasso":    Lasso(alpha=0.001, max_iter=10000),
    "RF B=200": RandomForestRegressor(n_estimators=200, max_features=0.33, random_state=42, n_jobs=-1),
    "GBDT M=300": GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                             learning_rate=0.05, subsample=0.8, random_state=42),
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    rmse_te = np.sqrt(mean_squared_error(y_te, mdl.predict(X_te)))
    r2_te   = r2_score(y_te, mdl.predict(X_te))
    rmse_tr = np.sqrt(mean_squared_error(y_tr, mdl.predict(X_tr)))
    results[name] = {'rmse_te': rmse_te, 'r2_te': r2_te, 'rmse_tr': rmse_tr}
    print(f"{name:18s}  测试 RMSE={rmse_te:.4f}  R²={r2_te:.4f}")

# ── 4. RF 特征重要性（前15） ───────────────────────────────────────────────────
rf_model = models["RF B=200"]
importances = rf_model.feature_importances_
feat_names  = feature_names_out[:len(importances)]

top_k = min(15, len(feat_names))
top_idx  = np.argsort(importances)[-top_k:][::-1]
top_imp  = importances[top_idx]
top_names = [feat_names[i] for i in top_idx]

# ── 5. 作图 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
ax_a, ax_b, ax_c = axes

# ── 面板 (a): 模型精度对比 ────────────────────────────────────────────────────
sorted_names = sorted(results, key=lambda k: results[k]['rmse_te'])
rmse_tr_vals = [results[n]['rmse_tr'] for n in sorted_names]
rmse_te_vals = [results[n]['rmse_te'] for n in sorted_names]
bar_colors   = [COLORS['blue'], COLORS['teal'], COLORS['green'], COLORS['purple']]
bar_colors_s = bar_colors[:len(sorted_names)]

x_idx = np.arange(len(sorted_names))
bar_w = 0.38
ax_a.bar(x_idx - bar_w/2, rmse_tr_vals, bar_w, label='训练 RMSE', color=bar_colors_s, alpha=0.35, edgecolor='gray')
ax_a.bar(x_idx + bar_w/2, rmse_te_vals, bar_w, label='测试 RMSE', color=bar_colors_s, alpha=0.9,  edgecolor='gray')

for i, (tr, te) in enumerate(zip(rmse_tr_vals, rmse_te_vals)):
    ax_a.text(i + bar_w/2, te + 0.002, f'{te:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_a.set_xticks(x_idx)
ax_a.set_xticklabels(sorted_names, fontsize=11)
ax_a.set_ylabel('RMSE（log 房价）', fontsize=12)
ax_a.set_title('(a) Ames Housing 各方法精度对比\n（训练/测试 RMSE）', fontsize=13, fontweight='bold')
ax_a.legend(fontsize=10)

# ── 面板 (b): 预测值 vs 真实值（GBDT最优模型） ─────────────────────────────────
gbdt = models["GBDT M=300"]
y_pred_gbdt = gbdt.predict(X_te)
ax_b.scatter(y_te, y_pred_gbdt, color=COLORS['purple'], s=15, alpha=0.4, zorder=3)
lims = [min(y_te.min(), y_pred_gbdt.min()), max(y_te.max(), y_pred_gbdt.max())]
ax_b.plot(lims, lims, 'r--', lw=1.8, label='理想预测线 $y=\hat{y}$')

# 标注误差带 ±0.1
ax_b.fill_between(lims, [l - 0.1 for l in lims], [l + 0.1 for l in lims],
                   color='red', alpha=0.07, label='误差带 ±0.1')

r2_str = f"$R^2={results['GBDT M=300']['r2_te']:.4f}$"
rmse_str = f"RMSE$={results['GBDT M=300']['rmse_te']:.4f}$"
ax_b.text(0.05, 0.92, f"{r2_str}\n{rmse_str}", transform=ax_b.transAxes,
          fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.8))

ax_b.set_xlabel('真实值（log 房价）', fontsize=12)
ax_b.set_ylabel('预测值（log 房价）', fontsize=12)
ax_b.set_title('(b) GBDT 最优模型：预测值 vs 真实值\n（测试集，偏离程度反映残差分布）',
               fontsize=13, fontweight='bold')
ax_b.legend(fontsize=10)
ax_b.set_aspect('equal', adjustable='box')

# ── 面板 (c): RF 特征重要性（前15）水平条形图 ─────────────────────────────────
colors_imp = [COLORS['green'] if imp > top_imp.mean() else COLORS['teal'] for imp in top_imp]
bars = ax_c.barh(range(top_k), top_imp[::-1], color=colors_imp[::-1], edgecolor='gray', linewidth=0.4)
ax_c.set_yticks(range(top_k))
ax_c.set_yticklabels(top_names[::-1], fontsize=9.5)
ax_c.set_xlabel('特征重要性（MDI，均值不纯度下降）', fontsize=12)
ax_c.set_title(f'(c) 随机森林特征重要性排名\n（前 {top_k} 个特征，Ames Housing）',
               fontsize=13, fontweight='bold')

# 标注数值
for bar, imp in zip(bars, top_imp[::-1]):
    ax_c.text(imp + 0.0005, bar.get_y() + bar.get_height()/2,
              f'{imp:.4f}', va='center', fontsize=8)

ax_c.axvline(x=top_imp.mean(), color='red', ls='--', lw=1.2, alpha=0.7, label=f'均值={top_imp.mean():.4f}')
ax_c.legend(fontsize=9)

plt.tight_layout()
save_fig(fig, __file__, 'fig4_7_03_ames_case')
