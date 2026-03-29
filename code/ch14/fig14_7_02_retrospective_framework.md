# Figure 14.7.2 — Project Retrospective and Knowledge Management Framework

## Prompt

Create a **professional, publication-quality infographic** depicting the 5L retrospective framework, the three-level knowledge evolution ladder, and the project knowledge base structure. This figure serves as the capstone visual for the entire chapter.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three horizontal bands**:
  - **Top band** (~35% height): 5L retrospective flower diagram.
  - **Middle band** (~30% height): Three-level knowledge evolution ladder.
  - **Bottom band** (~35% height): Knowledge base module cards.

### Top Band — 5L Retrospective Flower Diagram

A **five-petal flower** arrangement, centered, with each petal representing one L dimension:

| Petal # | Position | Label | Color | Core Question | Output |
|---------|----------|-------|-------|---------------|--------|
| 1 | Top | Liked | Green `#16a34a` | 哪些值得保留？ | 最佳实践 |
| 2 | Top-right | Learned | Blue `#2563eb` | 学到了什么？ | 知识点 |
| 3 | Bottom-right | Lacked | Orange `#ea580c` | 缺少了什么？ | 缺口清单 |
| 4 | Bottom-left | Longed for | Purple `#9333ea` | 希望改变什么？ | 改进建议 |
| 5 | Top-left | Loathed | Red `#dc2626` | 应避免什么？ | 反模式 |

Each petal:
- **Shape**: Rounded teardrop/petal, wider at the outer end, narrow at center.
- **Fill**: Respective color at 15% opacity, with solid border (2pt) in full color.
- **Content**: Label (English, bold 10pt) + Chinese core question (8pt) + Output type (8pt, italic).
- **Center circle**: White with gray border, containing "5L 复盘" in bold 11pt.

- **Title**: "5L 项目复盘框架" in bold 12pt, centered above.
- **Subtitle**: "团队成员独立填写 → 集体讨论 → 提炼行动项" in 8pt gray, below the flower.

### Middle Band — Three-Level Knowledge Evolution Ladder

A **staircase/ladder** diagram, ascending from left to right:

| Level | Step Height | Color | Label | Carrier | Example |
|-------|-----------|-------|-------|---------|---------|
| L1 个人 | Low | Gray `#94a3b8` | 个人经验 | 笔记、实验日志 | "XGBoost 比 RF 快 3x" |
| L2 团队 | Medium | Blue `#2563eb` | 团队知识 | 复盘报告、知识库 | "时序项目必须时间分割" |
| L3 组织 | High | Green `#16a34a` | 组织规范 | 流程规范、自动化检查 | "CI 自动检测数据泄漏" |

Each step:
- **Shape**: A rectangular step, each higher and wider than the previous.
- **Content**: Level label (bold), carrier name, and a speech bubble with the example text.
- **Connecting arrows**: Upward arrows between steps, labeled:
  - L1→L2: "复盘沉淀" (blue text)
  - L2→L3: "编码为规范" (green text)

- **Title**: "从经验到规范的三级进化" in bold 11pt, centered.

### Bottom Band — Knowledge Base Module Cards

Five **cards** arranged horizontally, each representing a knowledge base module:

| # | Module | Icon | Color | Update Frequency | Key Content |
|---|--------|------|-------|-----------------|-------------|
| 1 | 模式库 | 🏆 | Green `#16a34a` | 每次复盘 | S1–S6 成功模式 |
| 2 | 决策日志 | 📝 | Blue `#2563eb` | 实时 | 关键决策 + 理由 |
| 3 | 检查清单 | ✅ | Orange `#ea580c` | 定期修订 | 预防性检查项 |
| 4 | Model Card | 📄 | Purple `#9333ea` | 每次更新 | 模型说明书 |
| 5 | 数据源目录 | 🗃️ | Gray `#64748b` | 持续 | 源 + 质量状态 |

Each card (~280 × 200 px):
- **Top border**: 4px in module color.
- **Icon**: Large icon centered (or stylized).
- **Module name**: Bold 10pt.
- **Update frequency**: Small pill badge in module color.
- **Content description**: 8pt gray text, 1–2 lines.

- **Title**: "项目知识库结构" in bold 11pt, centered above.

### Connecting Elements

- A **thin dashed arrow** from the 5L flower's center down to the L1 step of the ladder, labeled "个人产出".
- A **thin dashed arrow** from the L2 step down to the knowledge base cards, labeled "团队沉淀".
- These arrows create a visual flow: **复盘 → 进化 → 知识库**.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Primary color**: Purple `#9333ea` as the thematic accent (matching the chapter's "复盘与沉淀" theme, and consistent with the purple used for Stage 4 documentation in §14.3).
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc`.
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 20 px between elements, ≥ 40 px outer margins.
- **Overall aesthetic**: The visual "capstone" of Chapter 14 — comprehensive yet clean, showing the full knowledge management cycle.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_7_02_retrospective_framework.png`
- Place in `public/figures/ch14/`
