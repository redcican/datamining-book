---
title: "附录 E 进阶学习资源"
description: "数据挖掘与机器学习领域的推荐教材、在线课程、重要期刊与会议、开源项目、社区资源和学习路线图索引，为读者的持续学习提供系统化指南。"
status: complete
---

# 附录 E 进阶学习资源

本附录为希望深入学习的读者提供经过筛选的资源推荐，按主题和难度组织。

---

## E.1 经典教材

### 数据挖掘与机器学习

| 书名 | 作者 | 难度 | 特色 | 对应本书内容 |
|------|------|------|------|------------|
| *Introduction to Data Mining* | Tan, Steinbach, Kumar | ★★☆ | 本领域最经典教材，覆盖全面 | 全书 |
| *Data Mining: Concepts and Techniques* | Han, Kamber, Pei | ★★★ | 偏数据库视角，工业界广泛使用 | 全书 |
| *Pattern Recognition and Machine Learning* | Bishop | ★★★★ | 贝叶斯视角，数学严谨 | §3, §6, §11 |
| *The Elements of Statistical Learning* | Hastie, Tibshirani, Friedman | ★★★★ | 统计学习理论经典，免费 PDF | §3, §4 |
| *An Introduction to Statistical Learning* | James, Witten, Hastie, Tibshirani | ★★☆ | ESL 的入门版，有 R 和 Python 版 | §3, §4 |
| *Machine Learning* | 周志华 | ★★★ | 中文"西瓜书"，理论与实践并重 | 全书 |
| *统计学习方法* | 李航 | ★★★★ | 中文经典，公式推导严谨完整 | §3, §4 |
| *Hands-On Machine Learning* | Géron | ★★☆ | 实践导向，代码丰富（Python） | §3, §4, §11, §12 |
| *Machine Learning: A Probabilistic Perspective* | Murphy | ★★★★★ | 概率视角的百科全书式教材 | §3, §6, §9 |

### 深度学习

| 书名 | 作者 | 难度 | 特色 | 免费获取 |
|------|------|------|------|---------|
| *Deep Learning* | Goodfellow, Bengio, Courville | ★★★★ | "花书"，理论基石 | deeplearningbook.org |
| *动手学深度学习 (d2l)* | 李沐等 | ★★★ | 中文，交互式代码，持续更新 | d2l.ai |
| *Neural Networks and Deep Learning* | Nielsen | ★★☆ | 直觉解释优秀 | neuralnetworksanddeeplearning.com |
| *Dive into Deep Learning* | 英文版 d2l | ★★★ | PyTorch/TF 双版本 | d2l.ai |

### 专题教材

| 主题 | 推荐书目 | 作者 | 对应章节 | 难度 |
|------|---------|------|---------|------|
| 自然语言处理 | *Speech and Language Processing* | Jurafsky & Martin | §9 | ★★★ |
| 推荐系统 | *Recommender Systems Handbook* | Ricci et al. | §10 | ★★★ |
| 图挖掘/网络科学 | *Networks, Crowds, and Markets* | Easley & Kleinberg | §10 | ★★☆ |
| 图神经网络 | *Graph Representation Learning* | Hamilton | §10 | ★★★★ |
| 时间序列 | *Forecasting: Principles and Practice* | Hyndman & Athanasopoulos | §8 | ★★☆ |
| 因果推断 | *The Book of Why* | Pearl | 超出本书范围 | ★★☆ |
| 因果推断（技术） | *Causal Inference in Statistics* | Pearl, Glymour, Jewell | 超出本书范围 | ★★★ |
| MLOps | *Designing Machine Learning Systems* | Huyen | §14 | ★★★ |
| 特征工程 | *Feature Engineering and Selection* | Kuhn & Johnson | §2 | ★★☆ |
| 概率图模型 | *Probabilistic Graphical Models* | Koller & Friedman | §6, §9 | ★★★★★ |

---

## E.2 在线课程

### 入门级（对应本书前半部分）

| 课程 | 平台 | 讲师 | 时长 | 特色 | 免费 |
|------|------|------|------|------|------|
| Machine Learning Specialization | Coursera | Andrew Ng | ~90h | 2022 更新版，使用 Python | 旁听 |
| Introduction to ML with Python | edX | — | ~40h | 侧重 scikit-learn | 旁听 |
| Google ML Crash Course | Google | — | ~15h | 配合 TensorFlow | ✅ |
| Kaggle Micro-Courses | Kaggle | 社区 | 5–10h/门 | 交互式，覆盖多主题 | ✅ |

### 进阶级（对应本书后半部分）

| 课程 | 平台 | 讲师 | 时长 | 特色 | 免费 |
|------|------|------|------|------|------|
| Deep Learning Specialization | Coursera | Andrew Ng | ~120h | 5 门课覆盖 DL 核心 | 旁听 |
| CS229: Machine Learning | Stanford/YouTube | Andrew Ng | ~30h | 数学深度版本 | ✅ |
| Fast.ai Practical Deep Learning | fast.ai | Jeremy Howard | ~60h | 自顶向下，快速上手 | ✅ |
| CS224n: NLP with Deep Learning | Stanford/YouTube | Manning | ~30h | NLP 顶级课程 | ✅ |
| CS224w: Machine Learning with Graphs | Stanford/YouTube | Leskovec | ~30h | 图学习 | ✅ |
| CS231n: CNNs for Visual Recognition | Stanford/YouTube | Karpathy et al. | ~25h | 计算机视觉 | ✅ |
| 动手学深度学习 | Bilibili/d2l.ai | 李沐 | ~100h | 中文，代码即教材 | ✅ |
| 机器学习白板推导 | Bilibili | shuhuai008 | ~50h | 数学推导极为清晰 | ✅ |

### 实践导向

| 课程 | 平台 | 特色 | 费用 |
|------|------|------|------|
| DataCamp | DataCamp | 交互式编程练习 | 订阅制 |
| 365 Data Science | 365datascience | 系统化数据科学路径 | 订阅制 |
| Coursera Applied DS Specialization | Coursera (Michigan) | 项目导向 | 旁听免费 |

---

## E.3 重要期刊与会议

### 顶级会议（CCF A 类）

| 会议 | 全称 | 领域 | 录用率 | 频率 | 投稿周期 |
|------|------|------|--------|------|---------|
| **NeurIPS** | Neural Information Processing Systems | ML 综合 | ~25% | 年度（12月） | 5月截稿 |
| **ICML** | Intl. Conf. on Machine Learning | ML 理论 | ~25% | 年度（7月） | 1月截稿 |
| **ICLR** | Intl. Conf. on Learning Representations | 表示学习 | ~30% | 年度（5月） | 10月截稿 |
| **KDD** | Knowledge Discovery and Data Mining | 数据挖掘 | ~15% | 年度（8月） | 2月截稿 |
| **AAAI** | Assoc. for the Advancement of AI | AI 综合 | ~15% | 年度（2月） | 8月截稿 |
| **IJCAI** | Intl. Joint Conf. on AI | AI 综合 | ~15% | 年度 | 1月截稿 |
| **ACL** | Assoc. for Computational Linguistics | NLP | ~25% | 年度 | 1月截稿 |
| **WWW** | The Web Conference | Web 挖掘 | ~20% | 年度（4月） | 10月截稿 |

### 其他重要会议

| 会议 | 领域 | 特色 |
|------|------|------|
| **ICDM** | 数据挖掘 | IEEE 旗下，工业应用丰富 |
| **SDM** | 数据挖掘 | SIAM 旗下，算法理论强 |
| **RecSys** | 推荐系统 | 唯一推荐系统专门会议 |
| **CIKM** | 信息与知识管理 | 信息检索+数据挖掘 |
| **EMNLP** | NLP | ACL 系列会议 |
| **CVPR/ICCV/ECCV** | 计算机视觉 | 视觉领域三大顶会 |

### 重要期刊

| 期刊 | 缩写 | 领域 | 特色 |
|------|------|------|------|
| *Journal of Machine Learning Research* | JMLR | ML | 开放获取，高质量 |
| *IEEE Trans. on Knowledge and Data Eng.* | TKDE | 数据挖掘 | IEEE 旗舰 |
| *Data Mining and Knowledge Discovery* | DMKD | 数据挖掘 | Springer |
| *Machine Learning* | ML | ML 理论 | 历史悠久 |
| *Pattern Recognition* | PR | 模式识别 | 应用广泛 |
| *Artificial Intelligence* | AI | AI 综合 | 历史最悠久的 AI 期刊 |

### 如何追踪最新进展

| 渠道 | 用途 | 推荐方式 |
|------|------|---------|
| **arXiv** (arxiv.org) | 预印本 | 关注 cs.LG, stat.ML 分类 |
| **Papers with Code** | 论文+代码+排行榜 | 快速了解 SOTA |
| **Google Scholar Alerts** | 论文推送 | 设置关键词和作者订阅 |
| **Semantic Scholar** | AI 论文搜索 | 推荐相关论文 |
| **Connected Papers** | 论文关系图 | 可视化论文引用网络 |

### 如何阅读学术论文

| 步骤 | 方法 | 用时 |
|------|------|------|
| **第一遍：扫读** | 标题 → 摘要 → 图表 → 结论 | 5–10 min |
| **第二遍：精读** | 引言 → 方法 → 实验设计 | 30–60 min |
| **第三遍：复现** | 推导公式、运行代码、对比结果 | 数小时–数天 |

{% hint style="tip" %}
💡 **不要试图从头到尾"读完"一篇论文**。先判断这篇论文是否值得你花时间——第一遍扫读就能做出 80% 的判断。只有与你的研究/项目高度相关的论文才值得第三遍。
{% endhint %}

---

## E.4 开源项目与工具

### 核心 ML 框架

| 项目 | 用途 | 语言 | 本书章节 | Stars |
|------|------|------|---------|-------|
| **scikit-learn** | 传统 ML 全栈 | Python | §2–§7, §12, §13 | 59K+ |
| **PyTorch** | 深度学习（研究首选） | Python/C++ | §11 | 83K+ |
| **TensorFlow** | 深度学习（部署成熟） | Python/C++ | §11 | 185K+ |
| **JAX** | 高性能数值计算 | Python | — | 30K+ |

### 梯度提升

| 项目 | 特色 | 本书章节 |
|------|------|---------|
| **XGBoost** | 最成熟，GPU 支持 | §4, §13 |
| **LightGBM** | 最快，内存最低 | §4, §13, 附录 D |
| **CatBoost** | 类别特征最优 | §4, 附录 D |

### 数据处理

| 项目 | 用途 | 特色 |
|------|------|------|
| **Pandas** | 表格数据处理 | 生态最完善 |
| **Polars** | 高性能 DataFrame | Rust 实现，速度 5–10x |
| **Dask** | 分布式 Pandas | 大数据集 |
| **Feature-engine** | 特征工程自动化 | sklearn 兼容 |
| **Category Encoders** | 多种类别编码 | 目标编码等 |

### NLP 与文本

| 项目 | 用途 | 本书章节 |
|------|------|---------|
| **Hugging Face Transformers** | 预训练模型库 | §9 |
| **spaCy** | 工业级 NLP 流水线 | §9 |
| **jieba** | 中文分词 | §9 |
| **Gensim** | 主题模型、词嵌入 | §9 |

### 图与网络

| 项目 | 用途 | 本书章节 |
|------|------|---------|
| **NetworkX** | 图算法（小规模） | §10 |
| **PyG (PyTorch Geometric)** | 图神经网络 | §10 |
| **DGL** | 图深度学习 | §10 |

### MLOps 与实验管理

| 项目 | 用途 | 本书章节 |
|------|------|---------|
| **MLflow** | 实验追踪、模型注册 | §14.4 |
| **Weights & Biases** | 实验追踪与可视化 | §14.4 |
| **DVC** | 数据版本控制 | §14.3 |
| **Airflow** | 工作流编排 | §14.5 |
| **BentoML** | 模型服务化部署 | §14.5 |
| **Evidently AI** | 模型监控、数据漂移 | §14.5 |

### 可解释性

| 项目 | 用途 | 本书章节 |
|------|------|---------|
| **SHAP** | Shapley 值解释 | §4 |
| **LIME** | 局部可解释 | §4 |
| **ELI5** | 简单模型解释 | — |
| **InterpretML** | 微软可解释 ML 框架 | — |

### 可视化

| 项目 | 特色 |
|------|------|
| **Matplotlib** | 科研出版级（本书全部使用） |
| **Seaborn** | 统计可视化，更美观 |
| **Plotly** | 交互式图表 |
| **Streamlit** | 快速构建数据应用原型 |
| **Gradio** | ML 模型演示界面 |

---

## E.5 社区与交流

### 中文社区

| 平台 | 说明 | 适合 |
|------|------|------|
| **知乎** | ML/DM 话题，高质量回答 | 理论理解、方法讨论 |
| **Bilibili** | 技术教程视频 | 视频学习 |
| **天池/和鲸** | 竞赛讨论和 Notebook | 实践交流 |
| **机器之心** (jiqizhixin.com) | AI 新闻与技术文章 | 行业动态 |
| **PaperWeekly** | 论文解读 | 论文学习 |

### 英文社区

| 平台 | 说明 | 适合 |
|------|------|------|
| **Kaggle Discussion** | 竞赛讨论、Notebook 分享 | 实战技巧 |
| **Reddit** (r/MachineLearning) | 论文讨论、行业动态 | 前沿追踪 |
| **Stack Overflow** | 编程问题解答 | 技术问题 |
| **Twitter/X** | 研究者和机构动态 | 论文速递 |
| **Hugging Face Community** | NLP 和多模态讨论 | 模型使用 |
| **Discord** (ML 相关服务器) | 实时讨论 | 即时交流 |

---

## E.6 学习路线建议

### 按本书章节的深入路径

| 本书内容 | 想要深入时 | 推荐资源 | 用时估计 |
|---------|-----------|---------|---------|
| §2 数据预处理 | 特征工程系统化 | Kaggle Feature Engineering 微课 + Kuhn 教材 | 1–2 周 |
| §3 分类算法 | 数学证明和理论 | 李航《统计学习方法》ch2–7 | 3–4 周 |
| §4 高级分类 | 集成学习理论 | ESL ch10, 15–16 | 2–3 周 |
| §5 关联规则 | 频繁模式挖掘进阶 | Han et al. ch6–7 | 1–2 周 |
| §6 聚类分析 | 高维/大规模聚类 | Bishop PRML ch9 | 2–3 周 |
| §7 异常检测 | 深度异常检测 | Pang et al. (2021) ACM Survey | 2 周 |
| §8 时间序列 | 深度时序预测 | Hyndman 教材 + d2l 时序章节 | 3–4 周 |
| §9 文本挖掘 | 大语言模型 | CS224n + Hugging Face NLP Course | 4–6 周 |
| §10 图挖掘 | 图神经网络 | CS224w + PyG 教程 | 3–4 周 |
| §11 深度学习 | 系统深入 | 花书 + 李沐 d2l | 8–12 周 |
| §14 项目管理 | MLOps 实践 | Huyen *Designing ML Systems* | 2–3 周 |

### 能力发展阶段

| 阶段 | 特征 | 建议行动 | 预期时间 |
|------|------|---------|---------|
| **L1 模仿** | 能跟着教程运行代码 | 完成本书所有 D 级习题 | 2–3 月 |
| **L2 理解** | 能解释算法原理和参数含义 | 完成 B 级习题 + 阅读推荐教材 | 3–6 月 |
| **L3 应用** | 能独立完成数据挖掘项目 | 参加 Kaggle 获得铜牌 | 6–12 月 |
| **L4 创新** | 能针对特定问题设计方案 | 阅读论文 + 参与开源项目 | 1–2 年 |
| **L5 领导** | 能带领团队完成复杂项目 | 实践 §14 的方法论 | 2+ 年 |

### 推荐学习顺序

对于零基础读者，建议按以下顺序学习本书：

```
§1 引论 → §2 预处理 → §3 分类基础 → §4 高级分类
    ↓
§5 关联规则 → §6 聚类 → §7 异常检测
    ↓
§8 时间序列 → §9 文本挖掘 → §10 图挖掘
    ↓
§11 深度学习 → §12 工具生态 → §13 综合案例 → §14 项目管理
```

对于有基础的读者，可直接跳至感兴趣的章节——每章开头的"前置知识"会告诉你需要先阅读哪些内容。

{% hint style="tip" %}
💡 **学习的最佳方式是教**

如果你觉得理解了一个概念，试着向别人解释它（或写一篇博客）。能清楚地解释，才是真正理解了。费曼学习法在数据挖掘领域同样有效——如果你无法向一个非技术人员解释"什么是过拟合"，那你可能还没有真正理解它。
{% endhint %}
