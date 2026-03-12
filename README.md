# 数据挖掘方法与实践

> 大学本科 / 研究生数据挖掘课程教材

## 快速开始

```bash
# 安装依赖（仅需一次）
bun install

# 启动本地开发服务器（热重载）
bun dev
# → 在浏览器中打开 http://localhost:3000
```

修改 `content/` 目录下的任何 `.md` 文件后，页面将自动刷新。

---

## 项目结构

```
datamining-book/
├── SUMMARY.md              ← 全书目录（同时作为侧边栏导航）
├── content/                ← 教材 Markdown 内容（按章节组织）
│   ├── preface.md
│   ├── ch01/               ← 第一章：数据挖掘概述
│   │   ├── index.md        ← 章节概述页
│   │   ├── 1.1.md
│   │   ├── 1.2.md
│   │   └── 1.3.md
│   ├── ch02/ … ch14/       ← 各章节（结构同上）
│   └── appendix/           ← 附录 A–E
│       ├── A.md … E.md
├── code/                   ← Python 图表生成脚本（按章节组织）
│   ├── shared/
│   │   ├── plot_config.py  ← 全书统一 matplotlib 配置
│   │   └── datasets.py     ← 公共数据集加载工具
│   ├── ch01/
│   │   ├── fig1_1_kdd_process.py
│   │   └── output/         ← 生成图片输出目录（.gitignore 忽略）
│   └── ch02/ … ch14/
├── assets/                 ← 外部图片（手工制作 / 引用图）
│   └── ch01/ … ch14/
└── src/                    ← Next.js 渲染应用（无需修改）
    ├── app/
    ├── components/
    └── lib/
```

## 内容编写规范

### 1. Markdown 格式

使用标准 GFM Markdown，另支持以下 GitBook 扩展语法：

```markdown
{% hint style="info" %}
这是一个信息提示框（支持 info / success / warning / danger）。
{% endhint %}

数学公式：行内 $E = mc^2$，块级：
$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}
$$
```

### 2. 图表规范

**能用 Python 生成的图** → 写在 `code/chXX/` 目录下：

```python
# 文件命名规范：fig{章节号}_{描述}.py
# 例：code/ch03/fig3_4_svm_hyperplane.py

import sys
sys.path.insert(0, str(__file__ + '/../../'))
from shared.plot_config import apply_style, save_fig

apply_style()
# ... 绘图代码 ...
save_fig(fig, __file__, "fig3_4_svm_hyperplane")
```

在 Markdown 中引用生成的图：

```markdown
> 📁 对应代码：`code/ch03/fig3_4_svm_hyperplane.py`

![图 3.4 SVM 最优超平面](../../code/ch03/output/fig3_4_svm_hyperplane.png)
**图 3.4**：SVM 最优超平面示意图。实线为决策边界，虚线为间隔边界，圆圈标注的点为支持向量。
```

**无法用 Python 生成的图** → 留占位符：

```markdown
<!-- FIGURE PLACEHOLDER -->
**图 X.Y**：[外部图，需手工制作]
**Caption**：此处应为……的示意图，展示……（详细描述）。
**来源**：原创绘制 / 引用自 [论文, 年份]
<!-- END PLACEHOLDER -->
```

### 3. 文件状态

每个 `.md` 文件的 frontmatter 中设置 `status` 字段：

```yaml
---
title: "3.3 支持向量机（SVM）"
status: draft        # draft | in-progress | done
---
```

`draft` 状态的页面会在页面顶部显示黄色草稿横幅。

## 与 GitBook.com 同步

本项目包含 `.gitbook.yaml`，可通过 Git Sync 直接连接到 GitBook.com：

1. 在 GitBook.com 创建新 Space
2. 在 Space 设置中启用 "Git Sync"
3. 关联本项目的 Git 仓库（GitHub / GitLab）
4. 选择 `main` 分支，GitBook 将自动读取 `SUMMARY.md` 和 `content/`

## 生成图表

```bash
# 生成单个图表
python code/ch01/fig1_1_kdd_process.py

# 生成某章所有图表
for f in code/ch03/fig*.py; do python "$f"; done
```

---

> 教材编写遵循 `writing_plan.md` 中的规范。每次提交前请更新章节的 `status` 字段。
