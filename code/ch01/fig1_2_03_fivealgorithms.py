"""
图 1.2.3  机器学习五种常见任务算法可视化（分类 / 回归 / 聚类 / 关联规则 / 异常检测）
对应节次：1.2 数据挖掘的基本任务与应用领域
运行方式：python code/ch01/fig1_2_03_fivealgorithms.py
输出路径：public/figures/ch01/fig1_2_03_fivealgorithms.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.plot_config import apply_style, save_fig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons, make_blobs
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

apply_style()

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 6, hspace=0.30, wspace=0.45)

# ==========================================
# 1. 预测性：分类 (Classification) - 使用 SVM
# ==========================================
ax = fig.add_subplot(gs[0, 0:2])
X_cls, y_cls = make_moons(n_samples=200, noise=0.15, random_state=42)
clf = SVC(kernel='rbf', C=1, gamma=1)
clf.fit(X_cls, y_cls)

xx, yy = np.meshgrid(np.linspace(X_cls[:, 0].min()-0.5, X_cls[:, 0].max()+0.5, 100),
                     np.linspace(X_cls[:, 1].min()-0.5, X_cls[:, 1].max()+0.5, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.5)
ax.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.5)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

ax.scatter(X_cls[y_cls==0, 0], X_cls[y_cls==0, 1], c='blue', edgecolors='k', label='类别 C1')
ax.scatter(X_cls[y_cls==1, 0], X_cls[y_cls==1, 1], c='red', edgecolors='k', label='类别 C2')
ax.set_title('1. 分类 (Classification)\n预测离散标签 $y \\in \\{C_1, C_2\\}$', fontsize=14)
ax.legend()

# ==========================================
# 2. 预测性：回归 (Regression) - 使用 SVR
# ==========================================
ax = fig.add_subplot(gs[0, 2:4])
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.randn(80) * 0.2
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_pred = svr.fit(X_reg, y_reg).predict(X_reg)

ax.scatter(X_reg, y_reg, color='darkorange', label='真实数据点', edgecolors='k')
ax.plot(X_reg, y_pred, color='navy', lw=3, label='回归拟合曲线')
ax.set_title('2. 回归 (Regression)\n预测连续值 $y \\in \\mathbb{R}$', fontsize=14)
ax.legend()

# ==========================================
# 3. 描述性：聚类 (Clustering) - 使用 K-Means
# ==========================================
ax = fig.add_subplot(gs[0, 4:6])
X_clu, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X_clu)

ax.scatter(X_clu[:, 0], X_clu[:, 1], c=y_kmeans, s=40, cmap='viridis', edgecolors='k')
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='*', edgecolors='k', label='聚类中心')
ax.set_title('3. 聚类 (Clustering)\n无标签，将样本划分为自然类别', fontsize=14)
ax.legend()

# ==========================================
# 4. 描述性：关联规则挖掘 (Association Rule Mining)
# ==========================================
ax = fig.add_subplot(gs[1, 0:3])
np.random.seed(42)
support = np.random.uniform(0.05, 0.4, 100)
confidence = support + np.random.uniform(0.1, 0.6, 100)
confidence = np.clip(confidence, 0, 1)
lift = np.random.uniform(1, 5, 100)

sc = ax.scatter(support, confidence, c=lift, s=lift*40, cmap='autumn', alpha=0.7, edgecolors='k')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('提升度 (Lift)')
ax.set_xlabel('支持度 (Support)')
ax.set_ylabel('置信度 (Confidence)')
ax.set_title('4. 关联规则挖掘 (Association Rules)\n发现变量共现模式 (支持度 vs 置信度)', fontsize=14)

# ==========================================
# 5. 描述性：异常检测 (Anomaly Detection) - 孤立森林
# ==========================================
ax = fig.add_subplot(gs[1, 3:6])
X_normal = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_ano = np.r_[X_normal, X_outliers]

clf_ano = IsolationForest(contamination=0.15, random_state=42)
y_ano_pred = clf_ano.fit_predict(X_ano)

xx_ano, yy_ano = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
Z_ano = clf_ano.decision_function(np.c_[xx_ano.ravel(), yy_ano.ravel()])
Z_ano = Z_ano.reshape(xx_ano.shape)
ax.contourf(xx_ano, yy_ano, Z_ano, cmap=plt.cm.Blues_r, alpha=0.4)

ax.scatter(X_ano[y_ano_pred == 1, 0], X_ano[y_ano_pred == 1, 1], c='white', edgecolors='k', s=40, label='正常样本')
ax.scatter(X_ano[y_ano_pred == -1, 0], X_ano[y_ano_pred == -1, 1], c='red', edgecolors='k', s=50, marker='s', label='异常样本')
ax.set_title('5. 异常检测 (Anomaly Detection)\n识别偏离正常分布的样本', fontsize=14)
ax.legend()

save_fig(fig, __file__, "fig1_2_03_fivealgorithms")
