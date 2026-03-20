"""
fig8_7_03_comparison.py
异常检测方法综合对比：四种方法在合成数据上的检测结果
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# ── 导入 ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from shared.plot_config import apply_style, save_fig, COLORS, PALETTE
# ── 初始化 ────────────────────────────────────────────────────────
apply_style()
np.random.seed(42)
# ── 生成数据 ──────────────────────────────────────────────────────
T = 400
t = np.arange(T)
seasonal = 15 * np.sin(2 * np.pi * t / 40)
signal = 50 + seasonal + np.random.normal(0, 2.5, T)
# 注入异常并记录真实标签
true_anomalies = np.zeros(T, dtype=bool)
# 点异常
for loc in [60, 200, 320]:
    signal[loc] += 28
    true_anomalies[loc] = True
# 上下文异常
signal[130:142] += 16
true_anomalies[130:142] = True
# 集体异常
signal[250:275] = 50 + np.random.normal(0, 8, 25)
true_anomalies[250:275] = True
# ── 四种检测方法 ──────────────────────────────────────────────────
# 方法 1：Shewhart
mu0, s0 = signal[:40].mean(), signal[:40].std()
det_shewhart = np.abs(signal - mu0) > 3 * s0
# 方法 2：CUSUM
k = 0.5 * s0
h = 5 * s0
C_plus = np.zeros(T)
C_minus = np.zeros(T)
det_cusum = np.zeros(T, dtype=bool)
for i in range(1, T):
    C_plus[i] = max(0, C_plus[i-1] + (signal[i] - mu0) - k)
    C_minus[i] = max(0, C_minus[i-1] - (signal[i] - mu0) - k)
    if C_plus[i] > h or C_minus[i] > h:
        det_cusum[i] = True
        C_plus[i] = 0
        C_minus[i] = 0
# 方法 3：预测残差
W = 20
pred = np.convolve(signal, np.ones(W)/W, mode='same')
res = np.abs(signal - pred)
thr = np.percentile(res[:60], 95) * 2
det_pred = res > thr
# 方法 4：简化 Matrix Profile Discord 近似
m = 25
n_subs = T - m + 1
mp_approx = np.zeros(n_subs)
for i in range(0, n_subs, 3):
    sub_i = signal[i:i+m]
    sub_i_n = (sub_i - sub_i.mean()) / (sub_i.std() + 1e-8)
    min_dist = np.inf
    for j in range(0, n_subs, 3):
        if abs(i - j) < m // 4:
            continue
        sub_j = signal[j:j+m]
        sub_j_n = (sub_j - sub_j.mean()) / (sub_j.std() + 1e-8)
        d = np.sqrt(np.sum((sub_i_n - sub_j_n)**2))
        if d < min_dist:
            min_dist = d
    mp_approx[i] = min_dist
# 插值
for i in range(n_subs):
    if mp_approx[i] == 0 and i > 0:
        mp_approx[i] = mp_approx[i-1]
mp_threshold = np.percentile(mp_approx[mp_approx > 0], 90)
det_mp = np.zeros(T, dtype=bool)
for i in range(n_subs):
    if mp_approx[i] > mp_threshold:
        det_mp[i:i+m] = True
# ── 计算性能 ──────────────────────────────────────────────────────
def calc_metrics(det, truth):
    tp = np.sum(det & truth)
    fp = np.sum(det & ~truth)
    fn = np.sum(~det & truth)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1
# ── 绘图 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [1.5, 1]})
fig.suptitle("图 8.7.3　异常检测方法综合对比",
             fontsize=20, fontweight="bold", y=1.02)
# ── 上面板：原始序列 + 真实异常标注 ────────────────────────────────
ax = axes[0]
ax.plot(t, signal, color=COLORS["blue"], lw=0.8, alpha=0.7)
# 标注真实异常区间
anom_ranges = []
in_anom = False
for i in range(T):
    if true_anomalies[i] and not in_anom:
        start = i
        in_anom = True
    elif not true_anomalies[i] and in_anom:
        anom_ranges.append((start, i))
        in_anom = False
if in_anom:
    anom_ranges.append((start, T))
for s, e in anom_ranges:
    ax.axvspan(s, e, alpha=0.2, color=COLORS["red"])
ax.set_ylabel("值", fontsize=14)
ax.set_title("原始序列（红色区域 = 真实异常）", fontsize=15)
ax.tick_params(labelsize=13)
# ── 下面板：各方法 F1 对比 ────────────────────────────────────────
ax = axes[1]
methods = ["Shewhart", "CUSUM", "预测残差", "MP Discord"]
detections = [det_shewhart, det_cusum, det_pred, det_mp]
f1_scores = []
prec_scores = []
rec_scores = []
for det in detections:
    p, r, f = calc_metrics(det, true_anomalies)
    prec_scores.append(p)
    rec_scores.append(r)
    f1_scores.append(f)
x_pos = np.arange(len(methods))
width = 0.25
bars1 = ax.bar(x_pos - width, prec_scores, width, color=COLORS["blue"],
               alpha=0.7, label="精确率")
bars2 = ax.bar(x_pos, rec_scores, width, color=COLORS["green"],
               alpha=0.7, label="召回率")
bars3 = ax.bar(x_pos + width, f1_scores, width, color=COLORS["red"],
               alpha=0.7, label="F1")
# 标注 F1 值
for bar, f1 in zip(bars3, f1_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{f1:.2f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel("分数", fontsize=14)
ax.set_title("检测性能对比", fontsize=15)
ax.legend(fontsize=12, loc="upper right")
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=12)
# ── 保存 ──────────────────────────────────────────────────────────
fig.tight_layout(h_pad=1.5)
save_fig(fig, __file__, "fig8_7_03_comparison")
