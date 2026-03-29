---
title: "附录 C 编程实践指南"
description: "Python 数据挖掘环境搭建、核心库速查（NumPy/Pandas/sklearn/Matplotlib/SciPy）、Jupyter 最佳实践、sklearn 高级模式、性能优化和常见问题排查，为全书代码实践提供完整技术支持。"
status: complete
---

# 附录 C 编程实践指南

本附录为全书的编程实践提供技术指南，包括环境搭建、核心库速查、高级模式和常见问题排查。

---

## C.1 环境搭建

### 推荐配置

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| Python | 3.9–3.11 | 3.12+ 部分库兼容性可能有问题 |
| 包管理器 | Anaconda / Miniconda | 科学计算首选，自带 NumPy/Pandas/Jupyter |
| IDE | JupyterLab / VS Code | 探索式分析用 Jupyter，工程化开发用 VS Code |
| 版本控制 | Git | 代码和实验追踪的基础 |
| GPU（可选） | CUDA 11.8+ | 深度学习加速（§11） |

### Anaconda 安装与环境管理

```bash
# 创建独立环境（推荐为每个项目创建独立环境）
conda create -n datamining python=3.10

# 激活环境
conda activate datamining

# 安装核心依赖
conda install numpy pandas scikit-learn matplotlib seaborn jupyter scipy

# 安装额外依赖
pip install mlxtend networkx jieba wordcloud category_encoders

# 深度学习（按需安装）
pip install torch torchvision  # PyTorch（推荐）
# 或
pip install tensorflow         # TensorFlow

# 梯度提升库（竞赛常用，附录 D）
pip install lightgbm xgboost catboost

# 导出环境（用于复现，§14.4）
conda env export > environment.yml

# 从文件重建环境
conda env create -f environment.yml
```

### pip + virtualenv 替代方案

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip freeze > requirements.txt      # 锁定版本
```

### GPU 环境配置（PyTorch）

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 安装带 CUDA 支持的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

{% hint style="tip" %}
💡 **锁定依赖版本**

`pip freeze > requirements.txt` 或 `conda env export > environment.yml` 可以锁定当前环境的精确版本号，确保他人能完全复现你的结果。这是 §14.4 可重复性保证的技术基础。
{% endhint %}

---

## C.2 核心库速查

### NumPy — 数值计算

```python
import numpy as np

# ─── 创建数组 ───
a = np.array([1, 2, 3])                    # 一维
X = np.random.randn(100, 5)                # 100×5 标准正态矩阵
zeros = np.zeros((3, 4))                   # 全零矩阵
eye = np.eye(3)                            # 单位矩阵

# ─── 线性代数 ───
np.dot(a, b)                                # 内积（一维）
A @ B                                       # 矩阵乘法（推荐写法）
np.linalg.norm(a)                           # L2 范数
np.linalg.norm(a, ord=1)                    # L1 范数
np.linalg.inv(A)                            # 矩阵求逆
np.linalg.det(A)                            # 行列式
eigenvalues, eigenvectors = np.linalg.eig(A)   # 特征分解
U, S, Vt = np.linalg.svd(X, full_matrices=False)  # SVD

# ─── 统计 ───
X.mean(axis=0)                              # 列均值
X.std(axis=0)                               # 列标准差
np.corrcoef(X.T)                            # 相关系数矩阵
np.percentile(X, [25, 50, 75], axis=0)      # 分位数

# ─── 广播机制 ───
# (100, 5) - (5,) = (100, 5)  自动按行广播
X_centered = X - X.mean(axis=0)

# ─── 随机种子（可重复性） ───
np.random.seed(42)                          # 旧式（全局）
rng = np.random.default_rng(42)             # 新式（推荐）
X = rng.standard_normal((100, 5))
```

### Pandas — 数据处理

```python
import pandas as pd

# ─── 数据读写 ───
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", encoding="gbk")      # 中文编码
df = pd.read_parquet("data.parquet")                # 大文件推荐
df.to_csv("output.csv", index=False)

# ─── 数据探索（EDA 第一步）───
df.shape                                     # (行数, 列数)
df.info()                                    # 数据类型与缺失
df.describe()                                # 数值列统计摘要
df.describe(include="object")                # 类别列统计
df.isnull().sum()                            # 各列缺失数
df.nunique()                                 # 各列唯一值数
df["target"].value_counts(normalize=True)    # 类别分布

# ─── 数据清洗 ───
df.dropna(subset=["target"])                 # 删除目标列缺失的行
df["col"].fillna(df["col"].median(), inplace=True)
df = df.drop_duplicates()

# ─── 特征工程 ───
df["log_amount"] = np.log1p(df["amount"])
df["age_bin"] = pd.cut(df["age"], bins=[0, 25, 45, 65, 100],
                       labels=["青年", "中年", "中老年", "老年"])
df = pd.get_dummies(df, columns=["category"], drop_first=True)

# ─── 时间特征（§8, §13.7）───
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["lag_1"] = df["value"].shift(1)
df["rolling_mean_24"] = df["value"].rolling(24).mean()

# ─── 分组统计 ───
df.groupby("label").agg(
    mean_val=("feature", "mean"),
    std_val=("feature", "std"),
    count=("feature", "count")
)

# ─── 内存优化 ───
def reduce_mem_usage(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].max() < 2**31:
            df[col] = df[col].astype("int32")
    return df
```

### Scikit-learn — 机器学习

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# ─── 数据划分 ───
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Pipeline（推荐写法，避免数据泄漏）───
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# ─── 训练与预测 ───
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# ─── 评估 ───
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ─── 交叉验证 ───
scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### SciPy — 科学计算与统计检验

```python
from scipy import stats

# ─── 假设检验（§14.4）───
t_stat, p_value = stats.ttest_rel(scores_A, scores_B)     # 配对 t 检验
stat, p_value = stats.wilcoxon(scores_A, scores_B)         # Wilcoxon 符号秩
stat, p_value = stats.friedmanchisquare(*all_scores)        # Friedman 检验

# ─── 正态性检验 ───
stat, p_value = stats.shapiro(data)        # Shapiro-Wilk（n < 5000）
stat, p_value = stats.kstest(data, "norm") # KS 检验

# ─── 分布拟合 ───
mu, sigma = stats.norm.fit(data)           # 正态分布参数估计
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, mu, sigma)         # 概率密度

# ─── 距离计算 ───
from scipy.spatial.distance import cdist, pdist, squareform
D = cdist(X, Y, metric="euclidean")         # 两组点的距离矩阵
D = squareform(pdist(X, metric="cosine"))   # 一组点的两两距离
```

### Matplotlib — 可视化

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── 全局设置（中文支持）───
mpl.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ─── 基本图表 ───
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, label="模型 A", color="#2563eb", linewidth=2)
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color="#2563eb")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=10)
ax.set_title("训练损失曲线", fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")

# ─── 多子图 ───
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].bar(categories, values, color="#2563eb")
axes[1].scatter(x, y, c=labels, cmap="coolwarm", s=20, alpha=0.7)
axes[2].hist(data, bins=30, edgecolor="white", color="#16a34a")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()

# ─── 热力图（相关系数矩阵）───
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
```

### 本书常用的 sklearn 模型速查

| 任务 | 模型 | 导入路径 | 关键参数 | 适用场景 |
|------|------|---------|---------|---------|
| 二分类 | 逻辑回归 | `linear_model.LogisticRegression` | `C`, `class_weight`, `max_iter` | 基线、可解释 |
| 分类 | 随机森林 | `ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `class_weight` | 通用、特征重要性 |
| 分类 | 梯度提升 | `ensemble.GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth` | 高精度 |
| 分类 | SVM | `svm.SVC` | `C`, `kernel`, `gamma`, `probability` | 中小数据 |
| 分类 | KNN | `neighbors.KNeighborsClassifier` | `n_neighbors`, `weights`, `metric` | 简单、非参数 |
| 分类 | 朴素贝叶斯 | `naive_bayes.GaussianNB / MultinomialNB` | `var_smoothing` / `alpha` | 文本、快速 |
| 分类 | 决策树 | `tree.DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `criterion` | 可解释 |
| 回归 | 线性回归 | `linear_model.LinearRegression` | — | 基线 |
| 回归 | Ridge | `linear_model.Ridge` | `alpha` | 防过拟合 |
| 回归 | Lasso | `linear_model.Lasso` | `alpha` | 特征选择 |
| 回归 | 随机森林 | `ensemble.RandomForestRegressor` | `n_estimators`, `max_depth` | 通用 |
| 聚类 | K-means | `cluster.KMeans` | `n_clusters`, `init`, `n_init` | 球形簇 |
| 聚类 | DBSCAN | `cluster.DBSCAN` | `eps`, `min_samples` | 任意形状簇 |
| 聚类 | GMM | `mixture.GaussianMixture` | `n_components`, `covariance_type` | 软聚类 |
| 异常检测 | Isolation Forest | `ensemble.IsolationForest` | `contamination`, `n_estimators` | 通用异常检测 |
| 异常检测 | LOF | `neighbors.LocalOutlierFactor` | `n_neighbors`, `contamination` | 基于密度 |
| 降维 | PCA | `decomposition.PCA` | `n_components` | 线性降维 |
| 降维 | t-SNE | `manifold.TSNE` | `n_components`, `perplexity` | 可视化 |

---

## C.3 Sklearn 高级模式

### ColumnTransformer（混合类型特征处理）

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ["age", "income", "balance"]
cat_features = ["gender", "region"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
])

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 网格搜索（小空间）
param_grid = {"model__C": [0.01, 0.1, 1, 10], "model__max_iter": [500, 1000]}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"最优参数: {grid.best_params_}, AUC: {grid.best_score_:.4f}")

# 随机搜索（大空间，更高效）
from scipy.stats import uniform, randint
param_dist = {"model__n_estimators": randint(50, 500),
              "model__max_depth": randint(3, 15),
              "model__learning_rate": uniform(0.01, 0.3)}
random_search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                                   scoring="roc_auc", random_state=42)
```

### 自定义评估指标

```python
from sklearn.metrics import make_scorer, fbeta_score

# F2 分数（加倍重视召回率，适合欺诈检测）
f2_scorer = make_scorer(fbeta_score, beta=2)
scores = cross_val_score(pipe, X, y, cv=5, scoring=f2_scorer)
```

### 时间序列交叉验证

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # 保证训练集时间早于测试集，避免数据泄漏
```

---

## C.4 Jupyter 最佳实践

### 推荐工作流

| 阶段 | 工具 | 说明 |
|------|------|------|
| 探索与原型 | Jupyter Notebook | 快速迭代、可视化、交互式分析 |
| 工程化重构 | .py 脚本 | 将稳定的函数提取为模块 |
| 生产部署 | Python 包 / API | 不要在生产环境中运行 Notebook |

### Notebook 组织规范

```
notebook_template.ipynb
├── [Markdown] 1. 问题定义与数据描述
├── [Code]     1.1 导入库 + 设置随机种子
├── [Code]     1.2 加载数据
├── [Markdown] 2. 数据探索 (EDA)
├── [Code]     2.1 数据概览 (shape, info, describe)
├── [Code]     2.2 分布可视化
├── [Code]     2.3 相关性分析
├── [Markdown] 3. 数据预处理
├── [Code]     3.1 缺失值处理
├── [Code]     3.2 特征工程
├── [Code]     3.3 数据划分
├── [Markdown] 4. 建模与评估
├── [Code]     4.1 基线模型
├── [Code]     4.2 改进模型
├── [Code]     4.3 模型比较
├── [Code]     4.4 误差分析
└── [Markdown] 5. 结论与下一步
```

### 实用技巧

```python
# ─── 在 Notebook 顶部设置 ───
%matplotlib inline
%load_ext autoreload
%autoreload 2                    # 自动重载外部模块修改

# ─── Pandas 显示选项 ───
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.4f}".format)

# ─── 计时 ───
%%time                            # 单个 cell 总耗时
%%timeit                          # 多次运行取平均

# ─── 抑制警告 ───
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── 全局随机种子 ───
SEED = 42
np.random.seed(SEED)
# 如果使用 PyTorch:
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
```

---

## C.5 性能优化技巧

| 场景 | 问题 | 解决方案 | 预期提升 |
|------|------|---------|---------|
| 大 CSV 加载慢 | 解析耗时 | `pd.read_parquet()` 替代 CSV | 5–10x |
| 内存不足 | DataFrame 过大 | `reduce_mem_usage()` 降低精度 | 2–4x |
| 模型训练慢 | 数据量大 | 先在 10% 子样本调参，确认后用全量 | — |
| 交叉验证慢 | 网格搜索 | `RandomizedSearchCV` 替代 `GridSearchCV` | 5–50x |
| 文本向量化慢 | 词汇表过大 | `max_features=10000` 限制 TF-IDF 维度 | 2–5x |
| K-means 慢 | 大数据集 | `MiniBatchKMeans` 替代 `KMeans` | 3–10x |
| 距离矩阵大 | O(n²) 内存 | 使用 `BallTree` 或 `KDTree` 近邻搜索 | — |
| 深度学习慢 | 无 GPU | 使用 Google Colab 免费 GPU | 10–100x |

---

## C.6 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `ModuleNotFoundError` | 库未安装或环境未激活 | `pip install <库名>` 或 `conda activate <env>` |
| `ConvergenceWarning` | 逻辑回归/SVM 未收敛 | 增大 `max_iter`（1000→5000）或先标准化数据 |
| 训练集高、测试集低 | 过拟合 | 增加正则化、减少特征、增大训练集、Early Stopping |
| `ValueError: shape mismatch` | 训练/测试特征数不一致 | 使用 Pipeline 确保预处理一致 |
| AUC 异常高（>0.99） | 数据泄漏 | 检查特征中是否含目标衍生信息（§14.4） |
| `MemoryError` | 数据超内存 | 分块读取 `pd.read_csv(chunksize=10000)` |
| 中文乱码 | 编码不匹配 | `encoding="utf-8"` 或 `"gbk"` |
| 随机种子无效 | 未全局设置 | 同时设 `np.random.seed()` + 模型 `random_state` |
| `CUDA out of memory` | GPU 内存不足 | 减小 batch_size 或使用 `torch.cuda.empty_cache()` |
| sklearn Pipeline 中泄漏 | fit 时用了测试集 | StandardScaler 等必须放在 Pipeline 内部 |
| 类别编码不一致 | 测试集出现训练集未见类别 | `OneHotEncoder(handle_unknown="ignore")` |

---

## C.7 本书代码结构

```
code/
├── shared/
│   └── plot_config.py         # 全书统一的绘图样式配置
├── ch02/                       # 第2章：数据预处理
│   ├── fig02_*.py             # 各图生成脚本
│   └── _load_*.py             # 数据加载工具
├── ch03/ ... ch12/             # 第3–12章（同上结构）
├── ch13/                       # 第13章：综合案例
│   ├── _load_fraud.py         # §13.2 信用卡欺诈数据
│   ├── _load_maintenance.py   # §13.6 预测性维护数据
│   ├── _load_pm25.py          # §13.7 PM2.5 数据
│   └── fig13_*.py             # 案例图表脚本
└── ch14/                       # 第14章：项目管理
    └── fig14_*.md             # 图片生成提示词（AI 生成）
```

### 运行方式

```bash
# 激活环境
conda activate datamining

# 运行单个图表脚本
python code/ch13/fig13_7_01_eda.py

# 运行某章全部图表
for f in code/ch05/fig05_*.py; do python "$f"; done

# 图表输出位置：public/figures/chXX/
```

{% hint style="tip" %}
💡 `_load_*.py` 文件会自动下载数据集并缓存到 `~/.datamining_data/`，首次运行需网络，后续使用本地缓存。如遇下载问题，可手动下载数据文件放入缓存目录。
{% endhint %}
