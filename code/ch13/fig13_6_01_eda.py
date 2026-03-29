"""
图 13.6.1　AI4I 预测性维护数据探索
(a) 故障 vs 正常样本分布  (b) 关键特征按故障状态的箱线图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_maintenance import load_maintenance

df = load_maintenance()

# ── 基本统计 ──────────────────────────────────────────────
n = len(df)
n_fail = df["Machine failure"].sum()
n_normal = n - n_fail

print("=== AI4I 2020 预测性维护数据集 ===")
print(f"  样本数: {n:,}")
print(f"  正常: {n_normal:,} ({n_normal/n:.1%})")
print(f"  故障: {n_fail:,} ({n_fail/n:.1%})")
print(f"  不平衡比: {n_normal/n_fail:.1f}:1")

print("\n  产品类型分布:")
for t in ["L", "M", "H"]:
    cnt = (df["Type"] == t).sum()
    print(f"    {t}: {cnt} ({cnt/n:.1%})")

num_features = ["Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]",
                "Tool wear [min]"]
print("\n  特征统计:")
for feat in num_features:
    print(f"    {feat:<28s}  均值={df[feat].mean():.1f}  "
          f"标准差={df[feat].std():.1f}")

failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
print("\n  故障模式分布:")
for mode in failure_modes:
    cnt = df[mode].sum()
    print(f"    {mode}: {cnt} ({cnt/n:.2%})")

# ── 绘图 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                          gridspec_kw={"width_ratios": [0.6, 1.4]})

# (a) 故障分布
ax = axes[0]
bars = ax.bar(["正常\n(Normal)", "故障\n(Failure)"],
              [n_normal, n_fail],
              color=[COLORS["blue"], COLORS["red"]], width=0.5)
for bar, cnt in zip(bars, [n_normal, n_fail]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{cnt:,}\n({cnt/n:.1%})",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("样本数")
ax.set_title("(a) 故障分布", fontweight="bold")
ax.set_ylim(0, n_normal * 1.15)

# (b) 特征箱线图
ax = axes[1]
feat_short = ["Air Temp", "Process Temp", "Speed", "Torque", "Wear"]
normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

positions = np.arange(len(num_features))
width = 0.35

bp1 = ax.boxplot(
    [normal[f].values for f in num_features],
    positions=positions - width/2,
    widths=width * 0.8,
    patch_artist=True,
    boxprops=dict(facecolor=COLORS["blue"], alpha=0.6),
    medianprops=dict(color="black"),
    flierprops=dict(markersize=2),
    manage_ticks=False)

bp2 = ax.boxplot(
    [failure[f].values for f in num_features],
    positions=positions + width/2,
    widths=width * 0.8,
    patch_artist=True,
    boxprops=dict(facecolor=COLORS["red"], alpha=0.6),
    medianprops=dict(color="black"),
    flierprops=dict(markersize=2),
    manage_ticks=False)

ax.set_xticks(positions)
ax.set_xticklabels(feat_short, fontsize=10)
ax.set_title("(b) 特征分布：正常 vs 故障", fontweight="bold")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
          ["正常", "故障"], loc="upper right", fontsize=10)

plt.tight_layout(w_pad=2)
save_fig(fig, __file__, "fig13_6_01_eda")
