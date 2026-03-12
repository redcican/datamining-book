"""
Generate all placeholder markdown files for the textbook.
Run once: python _scaffold.py
"""
from pathlib import Path

ROOT = Path(__file__).parent
CONTENT = ROOT / "content"

# ── Chapter structure ──────────────────────────────────────────────────────
CHAPTERS = {
    "ch01": {
        "title": "第一章 数据挖掘概述",
        "sections": {
            "1.1": "数据挖掘的定义与发展历程",
            "1.2": "数据挖掘的基本任务与应用领域",
            "1.3": "数据挖掘的流程与方法论",
        },
    },
    "ch02": {
        "title": "第二章 数据预处理",
        "sections": {
            "2.1": "数据清洗技术",
            "2.2": "数据集成与转换",
            "2.3": "数据规约方法",
            "2.4": "特征选择与特征工程",
            "2.5": "数据标准化与归一化",
            "2.6": "缺失值处理策略",
            "2.7": "异常值检测与处理",
        },
    },
    "ch03": {
        "title": "第三章 分类算法",
        "sections": {
            "3.1": "决策树算法（ID3, C4.5, CART）",
            "3.2": "贝叶斯分类方法",
            "3.3": "支持向量机（SVM）",
            "3.4": "K 近邻（KNN）算法",
            "3.5": "神经网络分类方法",
            "3.6": "集成学习算法",
            "3.7": "分类算法评估与比较",
        },
    },
    "ch04": {
        "title": "第四章 回归算法",
        "sections": {
            "4.1": "线性回归基础（OLS、正规方程、Gauss-Markov 定理）",
            "4.2": "统计推断与模型诊断（t/F 检验、残差分析）",
            "4.3": "正则化回归（Ridge、Lasso、弹性网络）",
            "4.4": "非线性回归（多项式、样条、广义加法模型）",
            "4.5": "树模型回归（回归树 CART、随机森林、XGBoost）",
            "4.6": "偏差–方差分解与超参数选择",
            "4.7": "回归算法系统比较与案例分析",
        },
    },
    "ch05": {
        "title": "第五章 聚类分析",
        "sections": {
            "5.1": "基于划分的聚类算法（K-means, K-medoids）",
            "5.2": "基于密度的聚类算法（DBSCAN, OPTICS）",
            "5.3": "基于层次的聚类算法",
            "5.4": "基于网格的聚类算法",
            "5.5": "基于模型的聚类方法",
            "5.6": "聚类有效性评估",
            "5.7": "高维数据聚类的挑战与方法",
        },
    },
    "ch06": {
        "title": "第六章 关联规则挖掘",
        "sections": {
            "6.1": "频繁项集与关联规则基本概念",
            "6.2": "Apriori 算法",
            "6.3": "FP-Growth 算法",
            "6.4": "关联规则的兴趣度量",
            "6.5": "多层次与多维关联规则挖掘",
            "6.6": "关联规则可视化与解释",
            "6.7": "应用案例分析",
        },
    },
    "ch07": {
        "title": "第七章 异常检测",
        "sections": {
            "7.1": "异常检测的基本概念",
            "7.2": "统计学方法",
            "7.3": "基于距离的方法",
            "7.4": "基于密度的方法（LOF）",
            "7.5": "基于聚类的方法",
            "7.6": "基于分类的方法",
            "7.7": "异常检测在欺诈识别中的应用",
        },
    },
    "ch08": {
        "title": "第八章 时序数据挖掘",
        "sections": {
            "8.1": "时间序列数据特征",
            "8.2": "时间序列预处理方法",
            "8.3": "时间序列相似性度量",
            "8.4": "时间序列模式发现",
            "8.5": "时间序列分类与聚类",
            "8.6": "时间序列预测模型",
            "8.7": "时间序列异常检测",
        },
    },
    "ch09": {
        "title": "第九章 文本挖掘",
        "sections": {
            "9.1": "文本预处理技术",
            "9.2": "文本表示模型",
            "9.3": "文本分类方法",
            "9.4": "文本聚类与主题模型",
            "9.5": "情感分析与意见挖掘",
            "9.6": "文本摘要与信息抽取",
            "9.7": "跨语言文本挖掘",
        },
    },
    "ch10": {
        "title": "第十章 图数据挖掘",
        "sections": {
            "10.1": "图数据表示与存储",
            "10.2": "图特征与相似度计算",
            "10.3": "社区发现算法",
            "10.4": "图分类与聚类方法",
            "10.5": "图中频繁模式挖掘",
            "10.6": "链接预测方法",
            "10.7": "图神经网络基础",
        },
    },
    "ch11": {
        "title": "第十一章 深度学习在数据挖掘中的应用",
        "sections": {
            "11.1": "深度学习基础",
            "11.2": "卷积神经网络及其应用",
            "11.3": "循环神经网络与序列数据挖掘",
            "11.4": "自编码器与表示学习",
            "11.5": "生成对抗网络",
            "11.6": "深度强化学习基础",
            "11.7": "迁移学习与领域适应",
        },
    },
    "ch12": {
        "title": "第十二章 数据挖掘系统与工具",
        "sections": {
            "12.1": "数据挖掘平台概述",
            "12.2": "Python 数据挖掘生态系统",
            "12.3": "R 语言数据挖掘工具包",
            "12.4": "大数据挖掘平台（Hadoop, Spark）",
            "12.5": "可视化数据挖掘工具",
            "12.6": "云端数据挖掘服务",
            "12.7": "开源数据挖掘工具比较",
        },
    },
    "ch13": {
        "title": "第十三章 数据挖掘实践案例",
        "sections": {
            "13.1": "零售领域购物篮分析",
            "13.2": "金融风险评估与欺诈检测",
            "13.3": "医疗健康数据挖掘",
            "13.4": "社交网络数据分析",
            "13.5": "推荐系统实现",
            "13.6": "工业大数据分析",
            "13.7": "智慧城市数据挖掘应用",
        },
    },
    "ch14": {
        "title": "第十四章 数据隐私与伦理",
        "sections": {
            "14.1": "数据挖掘中的隐私保护技术",
            "14.2": "差分隐私基础",
            "14.3": "联邦学习与隐私计算",
            "14.4": "数据匿名化技术",
            "14.5": "数据安全与隐私法规",
            "14.6": "数据挖掘的伦理考量",
            "14.7": "负责任的数据挖掘实践",
        },
    },
    "ch15": {
        "title": "第十五章 数据挖掘项目管理",
        "sections": {
            "15.1": "数据挖掘项目生命周期",
            "15.2": "问题定义与目标设定",
            "15.3": "数据需求分析与获取",
            "15.4": "模型开发与评估流程",
            "15.5": "模型部署与监控",
            "15.6": "团队协作与沟通",
            "15.7": "数据挖掘项目成功案例分析",
        },
    },
}

APPENDICES = {
    "A": "常用数据集资源",
    "B": "数学与统计学基础",
    "C": "编程实践指南",
    "D": "数据挖掘竞赛攻略",
    "E": "进阶学习资源",
}

# ── Templates ──────────────────────────────────────────────────────────────

PREFACE_TEMPLATE = """\
---
title: "前言"
description: "本书的写作背景、目标读者与使用说明"
status: draft
---

# 前言

> 🚧 本节内容正在编写中。

## 写作背景

## 目标读者

## 本书结构

## 使用建议

## 致谢
"""

CHAPTER_INDEX_TEMPLATE = """\
---
title: "{title}"
description: "本章介绍{title}的核心概念、主要算法与实践应用。"
status: draft
section: "{ch}"
---

# {title}

> 🚧 本章内容正在编写中。

## 本章概要

本章学习目标：

1. ...
2. ...
3. ...

## 本章知识导图

<!-- FIGURE PLACEHOLDER -->
**图 {ch_num}.0**：[外部图，需手工制作]
**Caption**：{title}的知识结构思维导图，展示各节之间的逻辑关系。
**来源**：原创绘制
<!-- END PLACEHOLDER -->

## 各节简介

| 节次 | 内容 |
|------|------|
"""

SECTION_TEMPLATE = """\
---
title: "{section_num} {section_title}"
description: "本节介绍{section_title}的基本概念、数学原理与实践应用。"
status: draft
---

# {section_num} {section_title}

> 🚧 本节内容正在编写中。请参考 `writing_plan.md` 中对本节的详细规划。

## 本节学习目标

完成本节学习后，读者应能够：

1. ...
2. ...
3. ...

---

## {section_num}.1 概念引入

*（在此描述本节要解决的问题，用真实场景引出核心概念。）*

## {section_num}.2 数学基础

*（在此进行完整的数学推导，包含定义框、定理框与证明。）*

## {section_num}.3 算法描述

*（伪代码框 + 逐步说明。）*

## {section_num}.4 实验与代码

> 📁 对应代码：`code/{ch}/fig{section_num_flat}_*.py`

```python
# TODO: 参照 code/shared/plot_config.py 编写本节代码
```

<!-- FIGURE PLACEHOLDER -->
**图 {section_num}.1**：[Python 生成，见 code/{ch}/fig{section_num_flat}_main.py]
**Caption**：此处应为本节核心算法的可视化图，展示……
**来源**：Python 生成（matplotlib）
<!-- END PLACEHOLDER -->

## {section_num}.5 案例分析

*（在此插入与本节对应的真实数据案例。）*

---

## 本节小结

| 关键词 | 定义 |
|--------|------|
| ... | ... |

---

## 习题

### A 级（概念理解）

**A1.** ...

**A2.** ...

### B 级（数学推导）

**B1.** ...

### C 级（数值计算）

**C1.** ...

### D 级（编程实践）

**D1.** ...

### E 级（案例分析）

**E1.** ...

### F 级（开放探讨）

**F1.** ...
"""

APPENDIX_TEMPLATE = """\
---
title: "附录 {letter} {title}"
description: "{title}"
status: draft
---

# 附录 {letter} {title}

> 🚧 本附录内容正在完善中。

"""


def write_if_missing(path: Path, content: str):
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"  created: {path.relative_to(ROOT)}")
    else:
        print(f"  exists:  {path.relative_to(ROOT)}")


def main():
    # ── preface ──────────────────────────────────────────────────────────
    write_if_missing(CONTENT / "preface.md", PREFACE_TEMPLATE)

    # ── chapters ─────────────────────────────────────────────────────────
    for ch, info in CHAPTERS.items():
        ch_dir = CONTENT / ch
        ch_dir.mkdir(parents=True, exist_ok=True)
        ch_num = ch[2:]  # "01", "02", …

        # Chapter index
        rows = "\n".join(
            f"| {sec} | {sec_title} |"
            for sec, sec_title in info["sections"].items()
        )
        idx_content = (
            CHAPTER_INDEX_TEMPLATE.format(
                title=info["title"], ch=ch, ch_num=ch_num
            )
            + rows
            + "\n"
        )
        write_if_missing(ch_dir / "index.md", idx_content)

        # Section files
        for sec_num, sec_title in info["sections"].items():
            sec_flat = sec_num.replace(".", "_")
            content = SECTION_TEMPLATE.format(
                section_num=sec_num,
                section_title=sec_title,
                ch=ch,
                section_num_flat=sec_flat,
            )
            write_if_missing(ch_dir / f"{sec_num}.md", content)

    # ── appendices ───────────────────────────────────────────────────────
    app_dir = CONTENT / "appendix"
    app_dir.mkdir(parents=True, exist_ok=True)
    for letter, title in APPENDICES.items():
        content = APPENDIX_TEMPLATE.format(letter=letter, title=title)
        write_if_missing(app_dir / f"{letter}.md", content)

    print("\n✓ Scaffold complete.")


if __name__ == "__main__":
    main()
