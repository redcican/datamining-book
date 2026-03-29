# Figure 14.6.1 — RACI Collaboration Matrix and Team Topology

## Prompt

Create a **professional, publication-quality infographic** depicting a RACI responsibility matrix for data mining teams and three organizational topology models. This figure helps readers understand role assignments and team structure options.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two rows**:
  - **Top row** (~55% height): RACI heatmap matrix.
  - **Bottom row** (~45% height): Three team topology diagrams side by side.

### Top Row — RACI Heatmap Matrix

A **color-coded matrix** with the following structure:

- **Columns** (x-axis, 5 roles):
  1. 数据科学家 (with a brain/chart icon)
  2. 数据工程师 (with a database/pipe icon)
  3. 业务分析师 (with a briefcase icon)
  4. ML 工程师 (with a server/deploy icon)
  5. 项目经理 (with a clipboard icon)

- **Rows** (y-axis, 8 activities):
  1. 目标定义 (§14.2)
  2. 数据获取 (§14.3)
  3. 数据质量验证
  4. 特征工程
  5. 模型训练与评估 (§14.4)
  6. 模型部署 (§14.5)
  7. 模型监控 (§14.5)
  8. 结果汇报

- **Cell colors** — four RACI categories:
  - **R** (Responsible): Blue `#2563eb` — bold letter "R" in white.
  - **A** (Accountable): Green `#16a34a` — bold letter "A" in white, slightly larger font.
  - **C** (Consulted): Orange `#ea580c` at 40% opacity — letter "C" in dark text.
  - **I** (Informed): Gray `#e2e8f0` — letter "I" in `#64748b`.

- **Cell values** matching Definition 14.15:
  - Row 1 (目标定义): C, I, A/R, I, C
  - Row 2 (数据获取): C, A/R, C, I, I
  - Row 3 (数据质量验证): R, C, I, I, A
  - Row 4 (特征工程): A/R, C, C, I, I
  - Row 5 (模型训练与评估): A/R, I, C, C, I
  - Row 6 (模型部署): C, C, I, A/R, I
  - Row 7 (模型监控): C, R, I, A, I
  - Row 8 (结果汇报): R, I, A, I, C

- For cells with "A/R", use a **split diagonal**: top-left half in green (A), bottom-right in blue (R).

- **Legend** at the top-right corner: Four colored squares with labels:
  - 🔵 R = 执行者 (Responsible)
  - 🟢 A = 负责人 (Accountable)
  - 🟠 C = 咨询者 (Consulted)
  - ⬜ I = 知会者 (Informed)

- **Title**: "数据挖掘项目 RACI 矩阵" in bold 12pt.

### Bottom Row — Three Team Topology Diagrams

Three equal-width cards (~480 × 300 px each), each showing an organizational diagram:

#### Card 1: 嵌入式 (Embedded) — Blue `#2563eb` border

- **Diagram**: Three business unit boxes (e.g., "营销部", "风控部", "运营部"), each containing a small data scientist icon inside.
- **Pros badge**: "✅ 深度业务理解" in green.
- **Cons badge**: "❌ 技术孤立" in red.
- **Label**: "嵌入式" in bold blue.

#### Card 2: 中央式 (Centralized) — Orange `#ea580c` border

- **Diagram**: One central "数据科学中心" box containing multiple data scientist icons, with dotted arrows going out to three business unit boxes.
- **Pros badge**: "✅ 标准统一" in green.
- **Cons badge**: "❌ 离业务远" in red.
- **Label**: "中央式" in bold orange.

#### Card 3: 混合式 (Hybrid) — Green `#16a34a` border

- **Diagram**: A central hub with three spokes going to business units. Data scientist icons appear both in the hub and at the business units. Dashed lines connect the embedded scientists back to the hub (dual reporting).
- **Pros badge**: "✅ 兼顾深度与广度" in green.
- **Cons badge**: "⚠️ 管理复杂" in orange.
- **Label**: "混合式" in bold green.
- A small "推荐" ribbon in the top-right corner of this card.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Light gray `#f8fafc`.
- **Card backgrounds**: Pure white `#ffffff`.
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 30 px between elements, ≥ 40 px outer margins.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_6_01_raci_collaboration.png`
- Place in `public/figures/ch14/`
