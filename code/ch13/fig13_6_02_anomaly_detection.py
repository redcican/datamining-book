"""
图 13.6.2　Isolation Forest 异常检测
扭矩 vs 转速 二维散点，颜色=异常分数，标记实际故障
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_maintenance import load_maintenance

df = load_maintenance()

num_features = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]",
                "Tool wear [min]"]

X = df[num_features].values
y_true = df["Machine failure"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Isolation Forest ──────────────────────────────────────
contamination = y_true.mean()  # 使用真实故障率
iforest = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42)
y_pred_if = iforest.fit_predict(X_scaled)  # -1=异常, 1=正常
anomaly_scores = -iforest.score_samples(X_scaled)  # 越高越异常

# 转换为 0/1（1=异常）
y_pred_binary = (y_pred_if == -1).astype(int)

prec = precision_score(y_true, y_pred_binary)
rec = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)

print("=== Isolation Forest 异常检测 ===")
print(f"  contamination = {contamination:.4f}")
print(f"  检出异常数: {y_pred_binary.sum()}")
print(f"  真实故障数: {y_true.sum()}")
print(f"  精确率: {prec:.3f}")
print(f"  召回率: {rec:.3f}")
print(f"  F1:     {f1:.3f}")

# 交叉统计
tp = ((y_pred_binary == 1) & (y_true == 1)).sum()
fp = ((y_pred_binary == 1) & (y_true == 0)).sum()
fn = ((y_pred_binary == 0) & (y_true == 1)).sum()
tn = ((y_pred_binary == 0) & (y_true == 0)).sum()
print(f"\n  混淆矩阵: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# 异常分数在故障/正常样本上的差异
print(f"\n  异常分数（故障样本）: "
      f"均值={anomaly_scores[y_true==1].mean():.4f}, "
      f"中位数={np.median(anomaly_scores[y_true==1]):.4f}")
print(f"  异常分数（正常样本）: "
      f"均值={anomaly_scores[y_true==0].mean():.4f}, "
      f"中位数={np.median(anomaly_scores[y_true==0]):.4f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# (a) 扭矩 vs 转速，异常分数着色
torque = df["Torque [Nm]"].values
speed = df["Rotational speed [rpm]"].values

# 正常样本
normal_mask = y_true == 0
sc = ax1.scatter(torque[normal_mask], speed[normal_mask],
                 c=anomaly_scores[normal_mask], cmap="YlOrRd",
                 s=8, alpha=0.4, edgecolors="none")

# 故障样本（用黑色圈标记）
fail_mask = y_true == 1
ax1.scatter(torque[fail_mask], speed[fail_mask],
            c=anomaly_scores[fail_mask], cmap="YlOrRd",
            s=40, alpha=0.9, edgecolors="black", linewidths=1.2)

cbar = plt.colorbar(sc, ax=ax1, shrink=0.8)
cbar.set_label("异常分数", fontsize=10)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=5, label='正常样本'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='black', markeredgewidth=1.2,
           markersize=8, label='实际故障'),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=10)
ax1.set_xlabel("扭矩 Torque [Nm]")
ax1.set_ylabel("转速 Rotational speed [rpm]")
ax1.set_title("(a) Isolation Forest 异常检测（扭矩 vs 转速）",
              fontweight="bold")

# (b) 异常分数分布（正常 vs 故障）
bins = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 50)
ax2.hist(anomaly_scores[y_true == 0], bins=bins, alpha=0.6,
         color=COLORS["blue"], label="正常", density=True)
ax2.hist(anomaly_scores[y_true == 1], bins=bins, alpha=0.7,
         color=COLORS["red"], label="故障", density=True)

# 判定阈值
threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
            label=f"阈值 = {threshold:.3f}")

ax2.set_xlabel("异常分数")
ax2.set_ylabel("密度")
ax2.set_title("(b) 异常分数分布：正常 vs 故障", fontweight="bold")
ax2.legend(fontsize=10)

plt.tight_layout(w_pad=2)
save_fig(fig, __file__, "fig13_6_02_anomaly_detection")
