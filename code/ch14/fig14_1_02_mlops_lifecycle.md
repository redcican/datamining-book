# Figure 14.1.2 — From CRISP-DM to MLOps: Lifecycle Evolution

## Prompt

Create a **professional, publication-quality infographic** showing the evolution from the classic CRISP-DM lifecycle to the modern MLOps-extended lifecycle, with a maturity model comparison at the bottom.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two-part layout**:
  - **Top section** (~60% height): Side-by-side comparison of two lifecycle diagrams connected by an evolution arrow.
  - **Bottom section** (~40% height): MLOps maturity model (Level 0 → Level 1 → Level 2) as a horizontal progression.

### Top Left — Classic CRISP-DM Cycle

- A clean **hexagonal cycle** (6 nodes connected by clockwise arrows) in **blue tones** (`#2563eb` at 20% opacity background, `#2563eb` solid borders).
- Six phase nodes as rounded rectangles:
  1. 业务理解 (Business Understanding)
  2. 数据理解 (Data Understanding)
  3. 数据准备 (Data Preparation)
  4. 建模 (Modeling)
  5. 评估 (Evaluation)
  6. 部署 (Deployment)
- Center label: "CRISP-DM\n(1999)" in bold navy text.
- Connecting arrows in solid blue, 1.5pt.
- A small "终点?" (End?) label with a question mark after the Deployment node, in gray italic.

### Center — Evolution Arrow

- A large **gradient arrow** pointing from left to right, filled with a blue-to-orange gradient (`#2563eb` → `#ea580c`).
- Label above: "方法论演进" (Methodology Evolution).
- Label below: "+ 持续训练 + 持续监控 + 模型治理" in 9pt text.

### Top Right — MLOps Extended Lifecycle

- A larger **circular cycle** with the original 6 CRISP-DM nodes (same style but in **orange tones** `#ea580c`) **plus 3 additional nodes** on the outer ring:
  - **CT** (持续训练, Continuous Training) — positioned between Evaluation and Deployment, green `#16a34a`.
  - **CM** (持续监控, Continuous Monitoring) — positioned after Deployment, looping back, red `#dc2626`.
  - **MG** (模型治理, Model Governance) — positioned as a central overlay element, purple `#9333ea`.
- The three new nodes should be visually distinct from the original six (e.g., slightly larger, different shape — hexagonal or diamond — with bolder borders).
- Arrow from CM back to Data Preparation (labeled "漂移触发重训练").
- Arrow from CM to MG (labeled "版本记录").
- Center label: "MLOps\n(2019–)" in bold dark orange text.
- PSI formula annotation near CM node: "PSI = Σ(pᵢ - qᵢ)ln(pᵢ/qᵢ)" in 8pt monospace, with a small note "> 0.25 → 触发更新".

### Bottom Section — MLOps Maturity Model

- Three **horizontal cards** arranged left-to-right, connected by gradient arrows:

| Level | Label | Color | Key Characteristics | Tools |
|---|---|---|---|---|
| Level 0 | 手动全流程 | Gray `#64748b` at 10% opacity | Notebook 手动执行; 手动部署; 无监控 | Jupyter, Excel |
| Level 1 | ML 管道自动化 | Blue `#2563eb` at 10% opacity | 自动化训练管道; 手动触发部署; 基础监控 | MLflow, Airflow |
| Level 2 | CI/CD for ML | Orange `#ea580c` at 10% opacity | 全自动训练+测试+部署; 漂移检测+自动回滚 | Kubeflow, Seldon, Evidently |

- Each card: rounded rectangle, 280 × 180 px, with the level number as a large bold numeral in the top-left corner.
- Inside each card: 3–4 bullet points listing characteristics in 8pt text.
- Below each card: tool names in small badges (gray background, tool name in dark text).
- Connecting arrows between cards: gradient-filled, with labels "投入产出比最高" between Level 0 and Level 1.

### Annotations

- Top-right corner: A small callout box with light yellow `#fefce8` background:
  - "CRISP-DM 回答: 如何构建模型?"
  - "MLOps 回答: 如何持续运维模型?"
  - In 9pt italic text with a thin `#eab308` border.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Lines**: 1–1.5 pt, with subtle shadows on nodes.
- **No decorative elements** — clean, structured, textbook aesthetic.
- **Margins**: ≥ 40 px on all sides.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_1_02_mlops_lifecycle.png`
- Place in `public/figures/ch14/`
