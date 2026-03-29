# Figure 14.2.2 — Goodhart's Law in Data Mining Projects

## Prompt

Create a **professional, publication-quality infographic** illustrating Goodhart's Law in the context of data mining projects. The figure is split into two halves: an ideal alignment path (top) and three failure modes (bottom).

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two horizontal bands**:
  - **Top band** (~35% height): "理想路径" (Ideal Path) — clean, green-toned.
  - **Bottom band** (~65% height): "三种偏差" (Three Failure Modes) — structured as three side-by-side panels with red/orange accents.
- A horizontal **dividing line** (dashed, gray `#94a3b8`, 1pt) separates the two bands, with a label on the left: "理想 vs 现实" in italic gray.

### Top Band — Ideal Path

- A **horizontal flow diagram** with four nodes connected by right-pointing arrows:
  1. **业务目标** (Business Objective) — blue `#2563eb` rounded rectangle.
  2. **挖掘目标** (Mining Objective) — green `#16a34a` rounded rectangle.
  3. **技术指标** (Technical Metric) — orange `#ea580c` rounded rectangle.
  4. **模型优化** (Model Optimization) — purple `#9333ea` rounded rectangle.
- Arrows: solid, 2pt, dark gray `#475569`, with arrowheads.
- Below each arrow, a small green checkmark (✓) icon indicating alignment.
- Above the entire flow: **"各层一致对齐"** in bold green `#16a34a`, 12pt.
- Below the flow: A thin green bar spanning the full width, labeled "指标忠实反映业务价值" in 9pt.

### Bottom Band — Three Failure Panels

Three equal-width panels arranged horizontally, each with a distinct failure mode:

#### Panel (a): 指标错选 (Metric Misselection)

- **Background**: Light red `#fef2f2` with thin red `#dc2626` left border (3px).
- **Header**: "(a) 指标错选" in bold red, 11pt.
- **Illustration**: A small 2-column comparison:
  - Left column: "选择: Accuracy = 99.5%" with a green checkmark (looks good).
  - Right column: "实际: 欺诈召回 = 12%" with a red cross (actual problem).
  - A red arrow labeled "代理偏差" connecting the two.
- **Root cause**: "准确率无法反映少数类的预测质量" in 8pt gray italic.
- **Prevention**: "✅ 选择与业务损失直接相关的指标 (如 Recall@FPR)" in 8pt dark text.

#### Panel (b): 过度优化 (Over-optimization)

- **Background**: Light orange `#fff7ed` with thin orange `#ea580c` left border (3px).
- **Header**: "(b) 过度优化" in bold orange, 11pt.
- **Illustration**: A small line chart sketch showing:
  - X-axis: "迭代轮次" (Iterations).
  - Blue line: "验证集 AUC" — rising steadily to 0.98.
  - Red dashed line: "上线后 AUC" — dropping to 0.82.
  - Gap between them labeled "过拟合" with a red bracket.
- **Root cause**: "模型记忆了验证集的分布特征" in 8pt gray italic.
- **Prevention**: "✅ 时间分割 + Hold-out 测试集 + 线上 A/B 验证" in 8pt dark text.

#### Panel (c): 目标漂移 (Goal Drift)

- **Background**: Light yellow `#fefce8` with thin yellow `#eab308` left border (3px).
- **Header**: "(c) 目标漂移" in bold dark yellow `#a16207`, 11pt.
- **Illustration**: A horizontal timeline with two markers:
  - T0 (project start): "业务目标: 预测流失" (blue flag).
  - T1 (6 months later): "业务需求: 优化挽留策略" (orange flag, new direction).
  - A gray dotted line between them labeled "业务环境变化".
  - Below: "技术团队仍在优化流失预测模型" in red text with ⚠️ icon.
- **Root cause**: "业务目标已变，技术指标未同步更新" in 8pt gray italic.
- **Prevention**: "✅ 每月业务回顾，重新校准指标与目标的对齐" in 8pt dark text.

### Bottom Annotation Bar

- A thin bar below the three panels with a **Goodhart's Law quote** centered:
  - "When a measure becomes a target, it ceases to be a good measure. — C. Goodhart, 1975"
  - In 9pt italic, gray `#64748b`, with a subtle left-aligned Chinese translation: "当度量指标变成目标时，它就不再是好的度量指标。"

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`, `yellow #eab308`.
- **Background**: Pure white `#ffffff`.
- **Panel borders**: Thin colored left borders (3px) for each failure panel.
- **No decorative elements** — clean, structured, textbook aesthetic.
- **Margins**: ≥ 40 px on all sides.
- **Shadows**: Subtle `2px 2px 6px rgba(0,0,0,0.05)` on panels.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_2_02_goodhart_law.png`
- Place in `public/figures/ch14/`
