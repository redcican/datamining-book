# Figure 14.6.2 — Data Mining Project Communication Framework

## Prompt

Create a **professional, publication-quality infographic** depicting the communication framework for data mining projects, including a three-layer documentation funnel, meeting cadence timeline, and technical-to-business translation bridge.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three columns**:
  - **Left column** (~30%): Three-layer documentation funnel.
  - **Center column** (~35%): Translation bridge (tech ↔ business).
  - **Right column** (~35%): Meeting cadence timeline.

### Left Column — Three-Layer Documentation Funnel

A **vertical funnel** (widest at bottom, narrowest at top), divided into three layers:

| Layer | Width | Color | Label | Audience | Content |
|-------|-------|-------|-------|----------|---------|
| 摘要层 | Narrow (top) | Purple `#9333ea` | 1 页 PPT | 高管、业务方 | 做什么、效果、需求 |
| 报告层 | Medium | Orange `#ea580c` | 实验报告 | 产品经理、分析师 | 方法、对比、建议 |
| 技术层 | Wide (bottom) | Blue `#2563eb` | Model Card + 代码 | 工程师、科学家 | 完整技术细节 |

- Each layer is a **trapezoid** section of the funnel, colored with the respective color at 20% opacity, with a solid left border (3px) in the full color.
- Inside each layer:
  - Layer name (bold, 11pt, colored).
  - Audience (9pt, gray).
  - Format badge (8pt, white text on colored background).
  - Content keywords (8pt, dark gray).
- An **upward arrow** along the right side of the funnel, labeled "抽象度 ↑" in gray.
- A **downward arrow** along the left side, labeled "细节度 ↓" in gray.
- Title: "三层漏斗文档体系" in bold 11pt, centered above.

### Center Column — Translation Bridge

A **bridge diagram** connecting the technical side (left) with the business side (right):

- **Left bank**: "技术语言" header in blue `#2563eb`.
  - Stack of 5 technical terms in blue rounded rectangles:
    - AUC = 0.92
    - FPR = 5%
    - PSI = 0.12
    - RMSE = 21.5
    - Feature Importance

- **Right bank**: "业务语言" header in green `#16a34a`.
  - Stack of 5 business translations in green rounded rectangles:
    - "78% 欺诈被拦截"
    - "每 100 笔误报 5 笔"
    - "数据轻微变化"
    - "预测误差 ±21.5"
    - "关键影响因素"

- **Bridge**: Connecting arrows (curved, bidirectional) between each pair, crossing over a central "翻译表" label in a hexagonal badge.

- **Below the bridge**: STAR structure in a compact card:
  - Four horizontal segments: S (gray) → T (blue) → A (orange) → R (green)
  - Labels: "背景 → 目标 → 方案 → 结果"
  - Title: "STAR 汇报结构" in bold 9pt.

### Right Column — Meeting Cadence Timeline

A **vertical timeline** showing a 2-week Sprint, with meeting types positioned along the timeline:

- **Left side**: Day markers (Day 1 through Day 10, weekdays only).
- **Right side**: Meeting cards at their positions.

| Day(s) | Meeting | Duration | Participants | Color |
|--------|---------|----------|-------------|-------|
| Every day | 数据晨会 | 15 min | DS + Eng | Blue `#2563eb` (small dots) |
| Day 3 | 业务同步 | 30 min | All + Biz | Orange `#ea580c` |
| Day 5 | 技术评审 | 60 min | DS team | Green `#16a34a` |
| Day 8 | 业务同步 | 30 min | All + Biz | Orange `#ea580c` |
| Day 10 | 回顾会 | 45 min | All | Purple `#9333ea` |
| Day 10 | 里程碑汇报 | 60 min | All + Mgmt | Red `#dc2626` (if at gate) |

- Daily standups shown as small blue dots on each day.
- Other meetings shown as small cards (colored left border, white background):
  - Meeting name (bold, 9pt).
  - Duration and participants (7pt, gray).

- Title: "Sprint 会议节奏" in bold 11pt, centered above.
- A note at the bottom: "实验笔记本替代传统日报" in italic 8pt gray.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 30 px between columns, ≥ 40 px outer margins.
- **Overall aesthetic**: Clean infographic style with balanced visual weight across three columns.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_6_02_communication_framework.png`
- Place in `public/figures/ch14/`
