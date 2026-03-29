---
title: "附录 B 数学与统计学基础"
description: "本书涉及的线性代数、矩阵微积分、概率论、统计学、信息论和优化方法的核心知识，包含完整公式推导和直觉解释，标注各知识点在正文中的使用位置。"
status: complete
---

# 附录 B 数学与统计学基础

本附录系统整理全书涉及的数学与统计学基础知识。每个主题标注其在正文中的主要应用章节，兼顾**公式速查**和**直觉理解**。对于仅需快速查阅的读者，可直接跳至各节的公式汇总表；对于希望深入理解的读者，每个小节包含推导思路和几何直觉。

---

## B.1 符号约定

全书使用的数学符号遵循以下约定：

| 类别 | 符号 | 含义 | 示例 |
|------|------|------|------|
| 标量 | $$x, y, z, \alpha, \lambda$$ | 小写斜体（拉丁字母或希腊字母） | 特征值、学习率、正则化系数 |
| 向量 | $$\mathbf{x}, \mathbf{w}, \boldsymbol{\mu}$$ | 小写粗体，默认为列向量 | 特征向量、权重向量、均值向量 |
| 矩阵 | $$\mathbf{X}, \mathbf{A}, \mathbf{\Sigma}$$ | 大写粗体 | 数据矩阵、邻接矩阵、协方差矩阵 |
| 集合 | $$\mathcal{D}, \mathcal{S}, \mathcal{X}$$ | 花体大写 | 数据集、样本空间、特征空间 |
| 随机变量 | $$X, Y, Z$$ | 大写斜体 | 与其实现值 $$x, y, z$$ 区分 |

### 常用运算符

| 符号 | 含义 | 说明 |
|------|------|------|
| $$\mathbf{x}^\top$$ | 转置 | 列向量变行向量，$$(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$$ |
| $$\|\mathbf{x}\|_p$$ | $$L_p$$ 范数 | $$p=2$$ 时省略下标；$$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$$ |
| $$\nabla f$$ | 梯度 | $$(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n})^\top$$ |
| $$\nabla^2 f$$ 或 $$\mathbf{H}$$ | Hessian 矩阵 | 二阶偏导矩阵，$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$ |
| $$\arg\min_x f(x)$$ | 使 $$f(x)$$ 最小化的 $$x$$ | — |
| $$\mathbb{1}[\cdot]$$ | 指示函数 | 条件为真时取 1，否则取 0 |
| $$\odot$$ | Hadamard 积（逐元素乘积） | LSTM 门控运算（§8, §11） |
| $$\otimes$$ | 外积（张量积） | $$\mathbf{x}\otimes\mathbf{y} = \mathbf{x}\mathbf{y}^\top$$（矩阵分解，§10） |
| $$\text{tr}(\mathbf{A})$$ | 矩阵的迹 | $$\sum_i A_{ii}$$，即对角线元素之和 |
| $$\text{det}(\mathbf{A})$$ 或 $$|\mathbf{A}|$$ | 行列式 | 衡量线性变换的"体积缩放因子" |
| $$\text{diag}(\mathbf{x})$$ | 以 $$\mathbf{x}$$ 为对角线元素的对角矩阵 | — |
| $$\propto$$ | 正比于 | $$P(A \mid B) \propto P(B \mid A) P(A)$$（忽略归一化常数） |

---

## B.2 线性代数

> **主要应用**：§2（PCA 降维）、§3（SVM、逻辑回归）、§4（线性回归）、§9（LSA、词嵌入）、§10（图谱分析）、§11（神经网络）

### B.2.1 向量与向量空间

| 运算 | 定义 | 几何意义 | 本书用途 |
|------|------|---------|---------|
| 内积（点积） | $$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i$$ | $$\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$$ | 相似度、投影、神经元计算 |
| L2 范数 | $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$ | 向量的"长度" | 距离度量、Ridge 正则化 |
| L1 范数 | $$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$ | 各坐标绝对值之和 | Lasso 正则化（§4） |
| L∞ 范数 | $$\|\mathbf{x}\|_\infty = \max_i |x_i|$$ | 最大坐标绝对值 | 对抗样本（§11） |
| 余弦相似度 | $$\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}$$ | 向量夹角的余弦 | 文本相似度（§9）、推荐（§10） |
| 投影 | $$\text{proj}_{\mathbf{u}}\mathbf{x} = \frac{\mathbf{x}^\top \mathbf{u}}{\mathbf{u}^\top \mathbf{u}} \mathbf{u}$$ | $$\mathbf{x}$$ 在 $$\mathbf{u}$$ 方向上的"影子" | PCA 降维（§2） |

**向量空间的关键性质**：

- **线性无关**：一组向量中没有任何一个能被其余向量的线性组合表示。$$n$$ 个线性无关向量构成 $$\mathbb{R}^n$$ 的一组**基**。
- **子空间**：$$\mathbb{R}^n$$ 中对加法和数乘封闭的子集。PCA 找到的是数据方差最大的 $$k$$ 维子空间。
- **正交性**：$$\mathbf{x}^\top\mathbf{y} = 0$$ 表示两向量垂直。正交基使得坐标分解最简——PCA 的主成分就是正交的。

### B.2.2 矩阵运算与性质

| 概念 | 定义/性质 | 用途 |
|------|---------|------|
| 矩阵乘法 | $$(\mathbf{AB})_{ij} = \sum_k A_{ik} B_{kj}$$ | 线性变换、神经网络前向传播 |
| 转置 | $$(\mathbf{A}^\top)_{ij} = A_{ji}$$ | $$(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$$ |
| 逆矩阵 | $$\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$ | 正规方程（§4） |
| 迹 | $$\text{tr}(\mathbf{A}) = \sum_i A_{ii} = \sum_i \lambda_i$$ | 等于特征值之和；循环性质：$$\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{CAB})$$ |
| 行列式 | $$\text{det}(\mathbf{A}) = \prod_i \lambda_i$$ | 高斯分布归一化项、矩阵可逆条件 |
| 秩 | $$\text{rank}(\mathbf{A})$$ = 非零特征值个数 | 数据的"有效维度"；欠秩→共线性 |

**特殊矩阵**：

| 矩阵类型 | 定义 | 性质 | 本书应用 |
|---------|------|------|---------|
| 对称矩阵 | $$\mathbf{A} = \mathbf{A}^\top$$ | 特征值为实数，特征向量正交 | 协方差矩阵、图的拉普拉斯矩阵 |
| 正定矩阵 | $$\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0,\; \forall \mathbf{x} \neq \mathbf{0}$$ | 所有特征值 > 0；保证凸性 | 高斯分布的协方差矩阵 |
| 半正定矩阵 | $$\mathbf{x}^\top\mathbf{A}\mathbf{x} \geq 0$$ | 特征值 ≥ 0 | 核矩阵（§3 SVM 核方法） |
| 正交矩阵 | $$\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$$ | 保持向量长度和角度 | PCA 投影矩阵 |
| 对角矩阵 | 仅对角线非零 | 各维独立缩放 | SVD 的奇异值矩阵 |
| 稀疏矩阵 | 大部分元素为零 | 节省存储和计算 | 文本 TF-IDF 矩阵、图的邻接矩阵 |

### B.2.3 特征分解与 SVD

**特征分解**（仅限方阵）：

$$\mathbf{A}\mathbf{v}_i = \lambda_i\mathbf{v}_i \quad \Longleftrightarrow \quad \mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$$

其中 $$\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_n]$$ 为特征向量矩阵，$$\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$$。

- 对称矩阵的特征分解：$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$$（$$\mathbf{Q}$$ 为正交矩阵）
- **直觉**：特征向量是矩阵"不改变方向"的方向，特征值是对应的缩放因子

**奇异值分解**（适用于任意 $$m \times n$$ 矩阵）：

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

| 成分 | 维度 | 含义 |
|------|------|------|
| $$\mathbf{U}$$ | $$m \times m$$ | 左奇异向量（行空间的正交基） |
| $$\mathbf{\Sigma}$$ | $$m \times n$$ | 奇异值对角矩阵（$$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$$） |
| $$\mathbf{V}$$ | $$n \times n$$ | 右奇异向量（列空间的正交基） |

- **截断 SVD**：只保留前 $$k$$ 个奇异值，$$\mathbf{X}_k = \mathbf{U}_k\mathbf{\Sigma}_k\mathbf{V}_k^\top$$ 是 $$\mathbf{X}$$ 的最优秩-$$k$$ 近似（Eckart-Young 定理）
- **应用**：LSA（§9）对 TF-IDF 矩阵做 SVD；矩阵分解推荐（§10）；PCA 等价于对中心化数据矩阵做 SVD

### B.2.4 PCA 详解

主成分分析将 $$d$$ 维数据投影到 $$k$$ 维（$$k < d$$），使得投影后的方差最大化：

**目标函数**：

$$\max_{\mathbf{w}: \|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}$$

其中 $$\mathbf{\Sigma} = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$$ 为协方差矩阵。

**求解步骤**：

1. 中心化：$$\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{x}}$$
2. 计算协方差矩阵：$$\mathbf{\Sigma} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$$
3. 特征分解：$$\mathbf{\Sigma}\mathbf{v}_i = \lambda_i\mathbf{v}_i$$，按特征值降序排列
4. 取前 $$k$$ 个特征向量组成投影矩阵 $$\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$$
5. 投影：$$\mathbf{Z} = \mathbf{X}\mathbf{W}$$

**方差保留比**（选择 $$k$$ 的依据）：

$$\text{VR}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

通常选 $$k$$ 使得 $$\text{VR}(k) \geq 95\%$$。

**PCA 与 SVD 的关系**：对中心化数据矩阵 $$\mathbf{X}$$ 做 SVD：$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$，则 PCA 的投影矩阵就是 $$\mathbf{V}$$ 的前 $$k$$ 列，主成分就是 $$\mathbf{U}_k\mathbf{\Sigma}_k$$。实际计算中通常直接用 SVD 而非显式构造协方差矩阵（数值稳定性更好）。

### B.2.5 图的矩阵表示

> **主要应用**：§10（PageRank、社区检测、谱聚类）

| 矩阵 | 定义 | 性质 |
|------|------|------|
| 邻接矩阵 $$\mathbf{A}$$ | $$A_{ij} = 1$$ 若节点 $$i, j$$ 相连 | 无向图中 $$\mathbf{A}$$ 对称 |
| 度矩阵 $$\mathbf{D}$$ | $$D_{ii} = \sum_j A_{ij}$$ | 对角矩阵，记录每个节点的度 |
| 拉普拉斯矩阵 $$\mathbf{L}$$ | $$\mathbf{L} = \mathbf{D} - \mathbf{A}$$ | 半正定；最小特征值 = 0 |
| 归一化拉普拉斯 $$\mathbf{L}_{\text{sym}}$$ | $$\mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$$ | 谱聚类使用的标准形式 |
| 转移矩阵 $$\mathbf{P}$$ | $$\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$$ | 随机游走的转移概率；PageRank 的基础 |

**谱聚类的核心思想**：拉普拉斯矩阵 $$\mathbf{L}$$ 的最小非零特征值对应的特征向量，编码了图的"最佳切割"方向。对这些特征向量做 K-means 聚类，就得到了图的社区划分。

---

## B.3 矩阵微积分

> **主要应用**：§3（逻辑回归梯度推导）、§4（线性回归正规方程）、§11（反向传播）

矩阵微积分是理解机器学习优化的基础。以下是最常用的求导法则：

### 标量对向量的导数

| 函数 | 导数 $$\frac{\partial f}{\partial \mathbf{x}}$$ | 应用 |
|------|-------|------|
| $$f = \mathbf{a}^\top\mathbf{x}$$ | $$\mathbf{a}$$ | 线性模型 |
| $$f = \mathbf{x}^\top\mathbf{A}\mathbf{x}$$ | $$(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$（若 $$\mathbf{A}$$ 对称则为 $$2\mathbf{A}\mathbf{x}$$） | 二次型（Ridge 正则项） |
| $$f = \|\mathbf{x} - \mathbf{a}\|_2^2$$ | $$2(\mathbf{x} - \mathbf{a})$$ | 均方误差 |
| $$f = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2$$ | $$2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b})$$ | OLS 正规方程推导（§4） |
| $$f = \log(\mathbf{a}^\top\mathbf{x})$$ | $$\frac{\mathbf{a}}{\mathbf{a}^\top\mathbf{x}}$$ | 逻辑回归（§3） |

### 线性回归正规方程的推导

目标函数 $$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 = (\mathbf{y} - \mathbf{X}\mathbf{w})^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

展开并求导：

$$\frac{\partial L}{\partial \mathbf{w}} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{0}$$

解得正规方程：$$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

加入 L2 正则化（Ridge）后：$$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

{% hint style="tip" %}
💡 **正则化的几何意义**：$$\lambda\mathbf{I}$$ 使 $$\mathbf{X}^\top\mathbf{X}$$ 的所有特征值增加 $$\lambda$$，保证矩阵可逆。当特征之间存在共线性时，$$\mathbf{X}^\top\mathbf{X}$$ 接近奇异，正则化起到"数值稳定器"的作用。
{% endhint %}

### 链式法则与反向传播

神经网络训练（§11）的核心是链式法则。对于复合函数 $$L = f(g(h(\mathbf{x})))$$：

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial \mathbf{x}}$$

**反向传播**就是从输出层到输入层逐层应用链式法则，将梯度"反向传播"回每一层的参数。对于 $$L$$ 层网络的第 $$l$$ 层：

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{z}^{(l)}} \cdot (\mathbf{a}^{(l-1)})^\top$$

其中 $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$ 为线性输出，$$\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$ 为激活输出。

---

## B.4 概率论

> **主要应用**：§3（朴素贝叶斯、逻辑回归）、§6（EM 算法、GMM）、§7（统计异常检测）、§9（LDA 主题模型）

### B.4.1 基本概念

| 概念 | 公式 | 说明 |
|------|------|------|
| 条件概率 | $$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$ | — |
| 联合概率 | $$P(A, B) = P(A \mid B) P(B)$$ | 链式法则 |
| 全概率公式 | $$P(A) = \sum_i P(A \mid B_i) P(B_i)$$ | 边际化——"对你不关心的变量求和" |
| 贝叶斯定理 | $$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$ | 朴素贝叶斯分类器的核心（§3） |
| 条件独立 | $$P(A, B \mid C) = P(A \mid C) P(B \mid C)$$ | 朴素贝叶斯的"朴素"假设 |
| 期望 | $$\mathbb{E}[X] = \sum_x x \cdot P(X = x)$$ | 连续情形：$$\int x f(x) dx$$ |
| 方差 | $$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$ | — |
| 协方差 | $$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$ | 衡量两变量的线性关联度 |
| 相关系数 | $$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$$ | $$\rho \in [-1, 1]$$；$$|\rho| = 1$$ 为完全线性相关 |

**贝叶斯定理的直觉**：

$$\underbrace{P(\text{类别} \mid \text{数据})}_{\text{后验}} = \frac{\overbrace{P(\text{数据} \mid \text{类别})}^{\text{似然}} \cdot \overbrace{P(\text{类别})}^{\text{先验}}}{\underbrace{P(\text{数据})}_{\text{证据}}}$$

- **先验**：在看到数据之前对类别的信念
- **似然**：假设属于某类别时，观察到该数据的概率
- **后验**：在看到数据之后更新的信念
- **证据**：归一化常数，使后验概率之和为 1

### B.4.2 常用分布

| 分布 | 概率密度/质量函数 | 参数 | 期望 | 方差 | 本书用途 |
|------|-----------------|------|------|------|---------|
| 伯努利 | $$P(x) = p^x(1-p)^{1-x}$$ | $$p$$ | $$p$$ | $$p(1-p)$$ | 二分类输出 |
| 二项分布 | $$\binom{n}{k} p^k (1-p)^{n-k}$$ | $$n, p$$ | $$np$$ | $$np(1-p)$$ | 成功次数统计 |
| 多项分布 | $$\frac{n!}{\prod x_i!} \prod p_i^{x_i}$$ | $$\mathbf{p}$$ | $$np_i$$ | — | 文本词频（§9） |
| 泊松分布 | $$\frac{\lambda^k e^{-\lambda}}{k!}$$ | $$\lambda$$ | $$\lambda$$ | $$\lambda$$ | 稀有事件计数 |
| 均匀分布 | $$f(x) = \frac{1}{b-a}$$ | $$a, b$$ | $$\frac{a+b}{2}$$ | $$\frac{(b-a)^2}{12}$$ | 随机初始化、分箱 |
| 正态分布 | $$f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$ | $$\mu, \sigma^2$$ | $$\mu$$ | $$\sigma^2$$ | GMM（§6）、异常检测（§7） |
| 多元正态 | $$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$ | $$\boldsymbol{\mu}, \mathbf{\Sigma}$$ | $$\boldsymbol{\mu}$$ | $$\mathbf{\Sigma}$$ | GMM（§6）、马氏距离 |
| 指数分布 | $$f(x) = \lambda e^{-\lambda x}$$ | $$\lambda$$ | $$1/\lambda$$ | $$1/\lambda^2$$ | 事件间隔时间 |
| 狄利克雷分布 | $$f(\boldsymbol{\theta}) \propto \prod_i \theta_i^{\alpha_i - 1}$$ | $$\boldsymbol{\alpha}$$ | $$\frac{\alpha_i}{\sum \alpha_j}$$ | — | LDA 先验（§9） |

### B.4.3 中心极限定理与大数定律

| 定理 | 内容 | 意义 |
|------|------|------|
| **大数定律** | 样本均值以概率收敛于总体期望：$$\bar{X}_n \xrightarrow{P} \mu$$ | 样本量越大，样本统计量越准确 |
| **中心极限定理** | 无论总体分布如何，样本均值的分布近似正态：$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$ | 假设检验和置信区间的理论基础 |

### B.4.4 极大似然估计（MLE）与最大后验估计（MAP）

给定独立同分布数据 $$\mathcal{D} = \{x_1, \ldots, x_n\}$$：

**MLE**（频率学派）：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n P(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^n \log P(x_i \mid \theta)$$

**MAP**（贝叶斯学派）：

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta \left[\sum_{i=1}^n \log P(x_i \mid \theta) + \log P(\theta)\right]$$

**MLE 与 MAP 的关系**：

- 当先验 $$P(\theta)$$ 为均匀分布时，MAP = MLE
- 当先验为高斯分布 $$P(\theta) = \mathcal{N}(0, \sigma^2)$$ 时，MAP 等价于带 L2 正则化的 MLE（Ridge 回归）
- 当先验为拉普拉斯分布时，MAP 等价于带 L1 正则化的 MLE（Lasso 回归）

{% hint style="tip" %}
💡 **正则化 = 先验**：从贝叶斯视角看，正则化不是"人为添加的惩罚"，而是对参数的先验信念。L2 正则化（Ridge）相当于相信"参数应该接近零但不必精确为零"（高斯先验）；L1 正则化（Lasso）相当于相信"大部分参数应该恰好为零"（拉普拉斯先验，即稀疏先验）。
{% endhint %}

### B.4.5 共轭先验

如果后验与先验属于同一分布族，则称该先验为似然的**共轭先验**：

| 似然分布 | 共轭先验 | 后验 | 应用 |
|---------|---------|------|------|
| 二项分布 | Beta 分布 | Beta 分布 | 贝叶斯分类器的参数估计 |
| 多项分布 | Dirichlet 分布 | Dirichlet 分布 | LDA 主题模型（§9） |
| 正态（已知方差） | 正态分布 | 正态分布 | 贝叶斯线性回归 |
| 正态（已知均值） | 逆 Gamma 分布 | 逆 Gamma 分布 | 方差估计 |

---

## B.5 统计学

> **主要应用**：§3.7（模型评估）、§4.6（偏差-方差分解）、§7（统计异常检测）、§14.4（统计检验）

### B.5.1 描述统计

| 统计量 | 公式 | 稳健性 | 用途 |
|--------|------|--------|------|
| 均值 | $$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$ | 对离群值敏感 | 中心趋势 |
| 中位数 | 排序后的中间值 | 稳健 | 偏态分布的中心趋势 |
| 标准差 | $$s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$$ | 对离群值敏感 | 离散程度 |
| IQR | $$Q_3 - Q_1$$ | 稳健 | 异常检测（§7）：$$x < Q_1 - 1.5 \cdot \text{IQR}$$ 为离群 |
| 偏度 | $$\frac{1}{n}\sum\left(\frac{x_i - \bar{x}}{s}\right)^3$$ | — | 分布对称性；> 0 右偏 |
| 峰度 | $$\frac{1}{n}\sum\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$ | — | 尾部厚度；> 0 厚尾 |

### B.5.2 假设检验

| 步骤 | 内容 | 说明 |
|------|------|------|
| 1 | 建立 $$H_0$$（无效果）和 $$H_1$$（有效果） | $$H_0$$ 是"默认立场"，需要证据来推翻 |
| 2 | 选择显著性水平 $$\alpha$$（通常 0.05） | 愿意承受的 I 类错误概率上限 |
| 3 | 计算检验统计量 | 将数据转化为一个标准化的量 |
| 4 | 计算 p 值 | 在 $$H_0$$ 为真时，观察到当前或更极端结果的概率 |
| 5 | 若 $$p < \alpha$$，拒绝 $$H_0$$ | 注意：不拒绝 $$\neq$$ 接受 $$H_0$$ |

**两类错误**：

| | $$H_0$$ 为真 | $$H_0$$ 为假 |
|---|---|---|
| **拒绝 $$H_0$$** | I 类错误（假阳性），概率 = $$\alpha$$ | 正确（检出力 = $$1 - \beta$$） |
| **不拒绝 $$H_0$$** | 正确 | II 类错误（假阴性），概率 = $$\beta$$ |

### B.5.3 模型比较的统计检验

| 检验 | 场景 | 检验统计量 | $$H_0$$ | 本书用途 |
|------|------|-----------|---------|---------|
| DeLong 检验 | 两个 AUC 比较 | 基于 Mann-Whitney U 统计量的方差估计 | 两个 AUC 相等 | §14.4 |
| McNemar 检验 | 两分类器错误率 | $$\chi^2 = \frac{(b - c)^2}{b + c}$$ | 两分类器错误率相同 | §14.4 |
| 配对 t 检验 | K 折 CV 评分 | $$t = \frac{\bar{d}}{s_d / \sqrt{K}}$$ | K 折评分均值相等 | §14.4 |
| Wilcoxon 符号秩 | 非正态分布 | 基于正负差值的秩和 | 两组中位数相等 | 非参数替代 |
| Friedman 检验 | 多模型排名 | $$\chi_F^2 = \frac{12N}{k(k+1)}\left[\sum_j R_j^2 - \frac{k(k+1)^2}{4}\right]$$ | 所有模型排名无差异 | §14.4 多模型 |

**McNemar 检验的列联表**：

| | 模型 B 正确 | 模型 B 错误 |
|---|---|---|
| **模型 A 正确** | $$a$$（都对） | $$b$$（A 对 B 错） |
| **模型 A 错误** | $$c$$（A 错 B 对） | $$d$$（都错） |

只有 $$b$$ 和 $$c$$ 提供了区分两模型的信息。若 $$b + c \geq 25$$，可用 $$\chi^2 = \frac{(b-c)^2}{b+c}$$；否则用精确二项检验。

### B.5.4 多重比较修正

同时进行 $$m$$ 次检验时，至少出现一次 I 类错误的概率为 $$1 - (1-\alpha)^m$$。当 $$m = 20, \alpha = 0.05$$ 时，此概率高达 64%。

| 方法 | 修正方式 | 保守程度 | 控制目标 |
|------|---------|---------|---------|
| Bonferroni | 每次检验使用 $$\alpha / m$$ | 最保守 | 族错误率 (FWER) |
| Holm-Bonferroni | 将 p 值排序后逐步检验 | 中等 | FWER |
| Benjamini-Hochberg | 将 p 值排序，与 $$\frac{i}{m}\alpha$$ 比较 | 最宽松 | 错误发现率 (FDR) |

### B.5.5 置信区间

| 场景 | 公式 | 条件 |
|------|------|------|
| 均值（大样本） | $$\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$ | $$n \geq 30$$ 或正态总体 |
| 均值（小样本） | $$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$ | 正态总体 |
| 比例 | $$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$ | $$n\hat{p} > 5$$ |
| AUC | 基于 DeLong 方法的标准误 | — |

### B.5.6 偏差-方差分解

模型泛化误差的分解（§4.6）：

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{(\mathbb{E}[\hat{f}(\mathbf{x})] - f(\mathbf{x}))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Noise}}$$

| 模型 | 偏差 | 方差 | 策略 |
|------|------|------|------|
| 线性回归 | 高（如果真实关系非线性） | 低 | — |
| 高次多项式 | 低 | 高 | 正则化 |
| 决策树（深） | 低 | 高 | 剪枝、集成（随机森林） |
| 随机森林 | 低 | 中（通过平均降低方差） | — |
| K-NN（K 小） | 低 | 高 | 增大 K |
| K-NN（K 大） | 高 | 低 | — |

### B.5.7 Bootstrap 方法

Bootstrap 是一种基于重采样的非参数统计方法，不需要对数据分布做假设：

1. 从 $$n$$ 个样本中**有放回**地抽取 $$n$$ 个样本，构成一个 Bootstrap 样本
2. 在 Bootstrap 样本上计算感兴趣的统计量（如均值、AUC）
3. 重复 $$B$$ 次（通常 $$B = 1000$$–$$10000$$）
4. 用 $$B$$ 个统计量的分布来估计标准误和置信区间

```python
from sklearn.utils import resample

scores = []
for _ in range(1000):
    X_boot, y_boot = resample(X_test, y_test, random_state=None)
    score = roc_auc_score(y_boot, model.predict_proba(X_boot)[:, 1])
    scores.append(score)

ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
```

---

## B.6 信息论

> **主要应用**：§3（决策树）、§9（文本特征选择）、§11（交叉熵损失）、§14.5（PSI 漂移检测）

### 基本概念

| 概念 | 公式 | 直觉 | 本书用途 |
|------|------|------|---------|
| 信息量 | $$I(x) = -\log_2 P(x)$$ | 越不可能的事件包含越多信息 | — |
| 信息熵 | $$H(X) = -\sum_{i} p_i \log_2 p_i$$ | 随机变量的平均不确定性 | 决策树分裂（§3） |
| 条件熵 | $$H(X \mid Y) = \sum_y P(y) H(X \mid Y = y)$$ | 在已知 Y 后 X 的剩余不确定性 | 信息增益计算 |
| 联合熵 | $$H(X, Y) = -\sum_{x,y} P(x,y) \log P(x,y)$$ | $$= H(X) + H(Y \mid X)$$ | — |
| 信息增益 | $$IG(X, A) = H(X) - H(X \mid A)$$ | 知道 A 后 X 的不确定性减少量 | ID3 决策树（§3） |
| 增益率 | $$GR = \frac{IG(X, A)}{H(A)}$$ | 对多值属性的修正 | C4.5 决策树（§3） |
| 基尼系数 | $$Gini = 1 - \sum_i p_i^2$$ | "随机抽两个样本类别不同"的概率 | CART 决策树（§3） |

### 分布间距离

| 概念 | 公式 | 性质 | 本书用途 |
|------|------|------|---------|
| 交叉熵 | $$H(p, q) = -\sum_i p_i \log q_i$$ | $$= H(p) + D_{KL}(p \| q)$$ | 分类损失函数（§3, §11） |
| KL 散度 | $$D_{KL}(P \| Q) = \sum_i p_i \log \frac{p_i}{q_i}$$ | 非对称、非负、$$D_{KL} = 0 \Leftrightarrow P = Q$$ | 分布差异度量 |
| JS 散度 | $$D_{JS} = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$$, $$M = \frac{P+Q}{2}$$ | 对称、有界 $$\in [0, 1]$$ | GAN 训练目标 |
| 互信息 | $$I(X; Y) = H(X) - H(X \mid Y) = D_{KL}(P_{XY} \| P_X P_Y)$$ | 衡量两变量的非线性关联 | 特征选择（§2） |

### PSI（群体稳定性指数）

$$\text{PSI} = \sum_{i=1}^{k} (p_i - q_i) \ln \frac{p_i}{q_i}$$

- 本质：$$\text{PSI} = D_{KL}(P \| Q) + D_{KL}(Q \| P)$$，即 KL 散度的对称化版本
- 当 $$p_i$$ 或 $$q_i$$ 为 0 时，需加平滑（如 $$p_i \leftarrow \max(p_i, 0.0001)$$）
- **阈值**：PSI < 0.10 正常，0.10–0.25 关注，≥ 0.25 显著漂移

### 交叉熵损失的推导

对于二分类问题，真实标签 $$y \in \{0, 1\}$$，模型预测概率 $$\hat{y} = \sigma(\mathbf{w}^\top\mathbf{x})$$：

$$L = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

这就是对伯努利分布取负对数似然——**最小化交叉熵等价于最大化似然**。

对于多分类（$$K$$ 类），真实标签为 one-hot 向量 $$\mathbf{y}$$，模型输出为 softmax 概率 $$\hat{\mathbf{y}}$$：

$$L = -\sum_{k=1}^K y_k \log \hat{y}_k$$

---

## B.7 优化方法

> **主要应用**：§3（SVM、逻辑回归）、§4（线性回归、正则化）、§6（K-means、EM）、§11（神经网络训练）

### B.7.1 凸优化基础

| 概念 | 定义 | 意义 |
|------|------|------|
| 凸集 | 集合中任意两点的连线仍在集合内 | 可行域为凸集是凸优化的前提 |
| 凸函数 | $$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$ | 局部最优 = 全局最优 |
| 严格凸 | 上述不等式严格成立 | 全局最优唯一 |
| 判定方法 | Hessian 矩阵 $$\nabla^2 f$$ 半正定 $$\Rightarrow$$ $$f$$ 凸 | — |

**重要结论**：线性回归 (OLS)、Ridge 回归、逻辑回归的损失函数都是凸函数，因此梯度下降一定能收敛到全局最优。而神经网络的损失函数是**非凸的**，梯度下降只能找到局部最优（但实践中效果通常足够好）。

### B.7.2 梯度下降及其变体

给定目标函数 $$L(\mathbf{w})$$，梯度下降的更新规则：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

| 变体 | 梯度计算方式 | 每步计算量 | 收敛行为 | 适用场景 |
|------|------------|----------|---------|---------|
| 批量 GD (BGD) | 使用全部 $$n$$ 个样本 | $$O(n)$$ | 平滑但慢 | 小数据集、凸问题 |
| 随机 GD (SGD) | 使用 1 个样本 | $$O(1)$$ | 震荡但有逃离局部最优的可能 | 大数据集 |
| 小批量 SGD | 使用 $$m$$ 个样本（如 $$m = 32, 64$$） | $$O(m)$$ | 兼顾速度和稳定性 | **实际最常用** |

### B.7.3 高级优化器

| 优化器 | 更新规则 | 特点 | 默认超参数 |
|--------|---------|------|-----------|
| **Momentum** | $$\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla L$$; $$\mathbf{w} \leftarrow \mathbf{w} - \eta\mathbf{v}_t$$ | 加速收敛、减少震荡 | $$\beta = 0.9$$ |
| **RMSProp** | $$s_t = \gamma s_{t-1} + (1-\gamma)(\nabla L)^2$$; $$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{\sqrt{s_t + \epsilon}}\nabla L$$ | 自适应学习率 | $$\gamma = 0.9$$ |
| **Adam** | 结合 Momentum 和 RMSProp | 自适应 + 动量 | $$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$$ |

**Adam 完整公式**（§11 深度学习训练默认使用）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L \quad \text{（一阶矩估计——动量）}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2 \quad \text{（二阶矩估计——自适应学习率）}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{（偏差修正）}$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### B.7.4 学习率调度

| 策略 | 公式 | 适用场景 |
|------|------|---------|
| 常数学习率 | $$\eta_t = \eta_0$$ | 凸问题、调试 |
| 线性衰减 | $$\eta_t = \eta_0 (1 - t/T)$$ | Transformer 预训练 |
| 指数衰减 | $$\eta_t = \eta_0 \cdot \gamma^t$$ | 经典深度学习训练 |
| 余弦退火 | $$\eta_t = \eta_{\min} + \frac{\eta_0 - \eta_{\min}}{2}(1 + \cos\frac{\pi t}{T})$$ | 现代深度学习 |
| Warmup + 衰减 | 前 $$T_w$$ 步线性增长，之后衰减 | Transformer（§9, §11） |

### B.7.5 正则化

| 方法 | 损失函数 | 参数效果 | 贝叶斯解释 |
|------|---------|---------|-----------|
| L2（Ridge） | $$L + \lambda\|\mathbf{w}\|_2^2$$ | 权重收缩（趋近零但不为零） | 高斯先验 |
| L1（Lasso） | $$L + \lambda\|\mathbf{w}\|_1$$ | 权重稀疏化（部分精确为零） | 拉普拉斯先验 |
| Elastic Net | $$L + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2$$ | 兼顾稀疏和稳定 | — |
| Dropout | 训练时以概率 $$p$$ 丢弃神经元 | 近似集成多个子网络 | — |
| Batch Norm | 对每层输入标准化 | 平滑损失面、加速收敛 | — |
| Early Stopping | 验证集指标不再提升时停止训练 | 隐式正则化 | — |

### B.7.6 EM 算法

期望最大化算法用于含隐变量模型的参数估计（§6 GMM）：

**问题**：直接最大化含隐变量 $$\mathbf{z}$$ 的对数似然 $$\log P(\mathbf{x} \mid \theta) = \log \sum_\mathbf{z} P(\mathbf{x}, \mathbf{z} \mid \theta)$$ 困难（求和在 $$\log$$ 内部）。

**EM 的策略**——交替执行两步：

1. **E 步**（Expectation）：在当前参数 $$\theta^{(t)}$$ 下，计算隐变量的后验概率：

$$Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{\mathbf{z} \mid \mathbf{x}, \theta^{(t)}} [\log P(\mathbf{x}, \mathbf{z} \mid \theta)]$$

2. **M 步**（Maximization）：最大化 $$Q$$ 函数更新参数：

$$\theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)})$$

**GMM 的 EM**：

- E 步：对每个数据点 $$\mathbf{x}_i$$，计算其属于第 $$k$$ 个高斯成分的后验概率（"责任"）：

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)}$$

- M 步：更新每个成分的参数：

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}, \quad \mathbf{\Sigma}_k^{\text{new}} = \frac{\sum_i \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top}{\sum_i \gamma_{ik}}, \quad \pi_k^{\text{new}} = \frac{\sum_i \gamma_{ik}}{n}$$

**性质**：EM 保证每次迭代不降低似然函数值（单调性），但可能收敛到局部最优。实践中通常用多组随机初始化运行多次，取似然最高的结果。

### B.7.7 拉格朗日对偶与 SVM

用于求解带约束优化问题（§3 SVM）：

**原始问题**：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \geq 1, \; i = 1, \ldots, n$$

**拉格朗日函数**：

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1]$$

**KKT 条件**：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \;\Rightarrow\; \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i \qquad \frac{\partial \mathcal{L}}{\partial b} = 0 \;\Rightarrow\; \sum_i \alpha_i y_i = 0$$

$$\alpha_i \geq 0, \quad \alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1] = 0 \quad \text{（互补松弛性）}$$

互补松弛性意味着：只有在决策边界上的样本（$$y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$$）才有 $$\alpha_i > 0$$，这些就是**支持向量**。

**对偶问题**（代入 KKT 条件后）：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top\mathbf{x}_j \quad \text{s.t.} \quad \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0$$

**核技巧**：对偶问题中数据只以内积 $$\mathbf{x}_i^\top\mathbf{x}_j$$ 形式出现，可替换为核函数 $$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$$，无需显式计算高维映射 $$\phi$$。

| 核函数 | $$K(\mathbf{x}, \mathbf{y})$$ | 适用场景 |
|--------|------|---------|
| 线性核 | $$\mathbf{x}^\top\mathbf{y}$$ | 线性可分数据 |
| 多项式核 | $$(\gamma\mathbf{x}^\top\mathbf{y} + r)^d$$ | 多项式决策边界 |
| RBF（高斯核） | $$\exp(-\gamma\|\mathbf{x} - \mathbf{y}\|^2)$$ | 通用非线性分类（默认选择） |

---

## B.8 距离与相似度度量

> **主要应用**：§3（KNN）、§5（K-means）、§6（关联规则）、§7（异常检测）、§8（时间序列）、§9（文本）

### 数值数据的距离

| 度量 | 公式 | 特点 | 本书应用 |
|------|------|------|---------|
| 欧氏距离 | $$d = \sqrt{\sum_i (x_i - y_i)^2}$$ | 最常用；受量纲影响→需标准化 | §3 KNN, §5 K-means |
| 曼哈顿距离 | $$d = \sum_i |x_i - y_i|$$ | 高维时比欧氏更稳定 | §3 |
| 切比雪夫距离 | $$d = \max_i |x_i - y_i|$$ | 只关注最大差异维度 | — |
| 闵可夫斯基距离 | $$d = (\sum_i |x_i - y_i|^p)^{1/p}$$ | $$p=1$$ 曼哈顿，$$p=2$$ 欧氏，$$p=\infty$$ 切比雪夫 | — |
| 马氏距离 | $$d = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$ | 考虑特征相关性和尺度 | §7 异常检测 |

### 集合与二值数据的相似度

| 度量 | 公式 | 值域 | 本书应用 |
|------|------|------|---------|
| Jaccard 系数 | $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$ | $$[0, 1]$$ | §6 关联规则 |
| Dice 系数 | $$D(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$ | $$[0, 1]$$ | 文本相似度 |
| 汉明距离 | 对应位不同的个数 | $$[0, n]$$ | 二值特征比较 |

### 文本与序列的距离

| 度量 | 定义 | 本书应用 |
|------|------|---------|
| 余弦相似度 | $$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}$$ | §9 文本相似度, §10 推荐 |
| 编辑距离 | 最少增/删/改操作数 | §9 字符串匹配 |
| DTW 距离 | 动态规划对齐后的累积距离 | §8 时间序列 |

**DTW 递推公式**：

$$D(i, j) = d(x_i, y_j) + \min\begin{cases} D(i-1, j) \\ D(i, j-1) \\ D(i-1, j-1) \end{cases}$$

其中 $$d(x_i, y_j) = |x_i - y_j|$$ 为逐点距离。DTW 允许两条时间序列在时间轴上"弹性对齐"，从而比较不同速度的模式。

{% hint style="tip" %}
💡 **距离度量的选择影响算法结果**：KNN、K-means、DBSCAN 等算法的行为强烈依赖距离度量的选择。使用欧氏距离前**必须标准化**，否则量纲大的特征会主导距离计算。对于文本数据，余弦相似度（忽略向量长度，只关注方向）通常优于欧氏距离。
{% endhint %}

---

## B.9 常用激活函数

> **主要应用**：§3（逻辑回归）、§8（LSTM）、§11（神经网络）

| 函数 | 公式 | 值域 | 导数 | 用途 |
|------|------|------|------|------|
| Sigmoid | $$\sigma(x) = \frac{1}{1+e^{-x}}$$ | $$(0, 1)$$ | $$\sigma(1-\sigma)$$ | 二分类输出、LSTM 门控 |
| Tanh | $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$ | $$(-1, 1)$$ | $$1 - \tanh^2(x)$$ | RNN 隐藏层 |
| ReLU | $$\max(0, x)$$ | $$[0, +\infty)$$ | $$\mathbb{1}[x > 0]$$ | 深度网络默认 |
| Leaky ReLU | $$\max(\alpha x, x)$$，$$\alpha = 0.01$$ | $$(-\infty, +\infty)$$ | $$\alpha$$ if $$x < 0$$, else 1 | 避免"神经元死亡" |
| GELU | $$x \cdot \Phi(x)$$（$$\Phi$$ 为标准正态 CDF） | ≈ $$(-0.17, +\infty)$$ | — | Transformer（BERT, GPT） |
| Softmax | $$\frac{e^{x_i}}{\sum_j e^{x_j}}$$ | $$(0, 1)$$，和 = 1 | — | 多分类输出 |

**梯度消失问题**：Sigmoid 和 Tanh 在输入绝对值大时，导数趋近于零，导致深层网络的梯度指数衰减。ReLU 在正区间导数恒为 1，有效缓解了这一问题——这是它成为深度学习默认激活函数的根本原因。

---

## B.10 模型评估指标速查

> **主要应用**：§3.7, §4, §13, §14.4

### 分类指标

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| 准确率 | $$\frac{TP + TN}{TP + TN + FP + FN}$$ | 类别平衡时 |
| 精确率 | $$\frac{TP}{TP + FP}$$ | 关注误报时 |
| 召回率 | $$\frac{TP}{TP + FN}$$ | 关注漏报时（欺诈检测、故障预测） |
| F1 | $$\frac{2 \cdot P \cdot R}{P + R}$$ | 精确率和召回率的调和均值 |
| AUC | ROC 曲线下面积 | 综合评估，阈值无关 |
| AUPRC | PR 曲线下面积 | 极不平衡数据（§13.2） |
| Log Loss | $$-\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$ | 评估预测概率的校准度 |

### 回归指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | $$\frac{1}{n}\sum(y_i - \hat{y}_i)^2$$ | 对大误差敏感 |
| RMSE | $$\sqrt{\text{MSE}}$$ | 与目标变量同单位 |
| MAE | $$\frac{1}{n}\sum|y_i - \hat{y}_i|$$ | 对离群值稳健 |
| R² | $$1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$ | 可解释为"模型解释的方差比例" |
| MAPE | $$\frac{1}{n}\sum\frac{|y_i - \hat{y}_i|}{|y_i|}$$ | 相对误差，可比较不同量纲 |

### 聚类指标

| 指标 | 需要真实标签 | 说明 |
|------|-----------|------|
| 轮廓系数 | 否 | 衡量类内紧密度 vs 类间分离度；$$\in [-1, 1]$$ |
| 调整兰德指数 (ARI) | 是 | 聚类结果与真实标签的一致性；$$\in [-1, 1]$$ |
| 归一化互信息 (NMI) | 是 | 基于信息论的一致性度量；$$\in [0, 1]$$ |
| 肘部法则 | 否 | 通过 SSE vs K 的拐点确定最优聚类数 |
