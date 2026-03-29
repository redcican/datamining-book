"""
图 13.2.1　German Credit 数据探索
(a) 信用等级分布  (b) 关键数值特征按类别分布
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS

apply_style()

from _load_credit import load_credit

df = load_credit()

# ── 统计信息 ──────────────────────────────────────────────
print(f"数据集: {df.shape[0]} 样本, {df.shape[1]} 列")
print(f"  数值特征: {df.select_dtypes(include='number').shape[1]}")
print(f"  分类特征: {df.select_dtypes(include='category').shape[1]}")
print(f"\n信用等级分布:")
for cls, cnt in df["class"].value_counts().items():
    print(f"  {cls}: {cnt} ({cnt/len(df):.1%})")
print(f"\n关键数值特征统计:")
for col in ["duration", "credit_amount", "age"]:
    g = df.loc[df["class"] == "good", col]
    b = df.loc[df["class"] == "bad", col]
    print(f"  {col}: good={g.mean():.1f}±{g.std():.1f}, "
          f"bad={b.mean():.1f}±{b.std():.1f}")

# ── 绘图 ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [0.8, 1.2]})

# --- (a) 类别分布 ---
counts = df["class"].value_counts()
good_n, bad_n = counts.get("good", 0), counts.get("bad", 0)
bars = ax1.bar(["好客户\n(Good)", "坏客户\n(Bad)"],
               [good_n, bad_n],
               color=[COLORS["blue"], COLORS["red"]],
               width=0.5, edgecolor="white")
for bar, cnt in zip(bars, [good_n, bad_n]):
    pct = cnt / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 12,
             f"{cnt} ({pct:.0f}%)", ha="center", fontsize=12,
             fontweight="bold")
ax1.set_ylabel("样本数")
ax1.set_title("(a) 信用等级分布", fontweight="bold")
ax1.set_ylim(0, good_n * 1.18)

# --- (b) 关键特征 box plots ---
features = ["duration", "credit_amount", "age", "installment_commitment"]
labels_zh = ["贷款期限\n(月)", "贷款金额\n(马克)", "年龄\n(岁)", "分期比例\n(%)"]

positions_g = [1, 4, 7, 10]
positions_b = [2, 5, 8, 11]
data_g = [df.loc[df["class"] == "good", f].values for f in features]
data_b = [df.loc[df["class"] == "bad", f].values for f in features]

# Normalize each feature for visual comparison
for i in range(len(features)):
    mu = df[features[i]].mean()
    sigma = df[features[i]].std()
    data_g[i] = (data_g[i] - mu) / sigma
    data_b[i] = (data_b[i] - mu) / sigma

bp1 = ax2.boxplot(data_g, positions=positions_g, widths=0.65,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=COLORS["blue"], alpha=0.55),
                   medianprops=dict(color="white", linewidth=2))
bp2 = ax2.boxplot(data_b, positions=positions_b, widths=0.65,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=COLORS["red"], alpha=0.55),
                   medianprops=dict(color="white", linewidth=2))
ax2.set_xticks([1.5, 4.5, 7.5, 10.5])
ax2.set_xticklabels(labels_zh, fontsize=10)
ax2.set_ylabel("标准化值")
ax2.set_title("(b) 关键数值特征按信用等级分布", fontweight="bold")
ax2.legend([bp1["boxes"][0], bp2["boxes"][0]],
           ["Good (好客户)", "Bad (坏客户)"], loc="upper right")

plt.tight_layout(w_pad=3)
save_fig(fig, __file__, "fig13_2_01_eda")
