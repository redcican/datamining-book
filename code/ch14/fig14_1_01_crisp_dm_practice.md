# Figure 14.1.1 — CRISP-DM Project Management Practice View

## Prompt

Create a **professional, publication-quality circular infographic** depicting the CRISP-DM six-phase cycle from a project management perspective, emphasizing deliverables, role mapping, and quality gates.

### Layout & Composition

- **Orientation**: Square (1:1 aspect ratio, ~1400 × 1400 px at 150 DPI).
- **Central element**: A circular/hexagonal diagram with six **wedge-shaped sectors**, one per CRISP-DM phase. Sector **areas are weighted by typical time allocation** (Data Preparation is the largest sector at ~40%, Business Understanding the smallest at ~8%).
- **Three concentric rings** surround the center:
  - **Inner ring** (closest to center): Quality gates G1–G6, shown as red `#dc2626` dashed gate icons at each phase boundary.
  - **Middle ring** (the sectors): Phase names, key deliverables, and time percentages.
  - **Outer ring**: Dominant role badges for each phase.
- **Center hub**: A small circle labeled "数据\nData" in gray `#64748b`, with thin gray spokes radiating to each sector boundary.

### Six Sectors (Clockwise from Top)

| # | Phase (Chinese) | Phase (English) | Color Fill | Time % | Key Deliverable | Dominant Role |
|---|---|---|---|---|---|---|
| 1 | 业务理解 | Business Understanding | Blue `#2563eb` at 15% opacity | 8% | 项目章程 | 业务分析师 (橙) |
| 2 | 数据理解 | Data Understanding | Green `#16a34a` at 15% opacity | 12% | 数据质量报告 | 数据科学家 (蓝) |
| 3 | 数据准备 | Data Preparation | Green `#16a34a` at 25% opacity | 40% | 清洗数据集 + 特征文档 | 数据工程师 (绿) |
| 4 | 建模 | Modeling | Orange `#ea580c` at 15% opacity | 15% | 实验报告 | 数据科学家 (蓝) |
| 5 | 评估 | Evaluation | Purple `#9333ea` at 15% opacity | 10% | 商业评估报告 | 业务分析师 (橙) |
| 6 | 部署 | Deployment | Red `#dc2626` at 15% opacity | 15% | 部署方案 + 监控面板 | ML 工程师 (紫) |

### Inner Ring — Quality Gates

- At each phase boundary (between sectors), place a small **gate icon** (stylized barrier/checkpoint, red `#dc2626`).
- Label each gate "G1" through "G6" in 8pt bold red text.
- Add a subtle annotation line from G3 (between Data Preparation and Modeling) with text: "关口未过\n不推进" in 7pt red italic.

### Outer Ring — Role Badges

- Place small **rounded-rectangle badges** on the outer edge of each sector with the dominant role:
  - Blue badge `#2563eb`: 数据科学家
  - Green badge `#16a34a`: 数据工程师
  - Orange badge `#ea580c`: 业务分析师
  - Purple badge `#9333ea`: ML 工程师
- Each badge has white text, 8pt bold.

### Iteration Arrows

- Between sectors, draw **curved arrows** (thin, gray `#94a3b8`, 1pt) showing the primary iteration paths:
  - A prominent arrow from Modeling back to Data Preparation (the most frequent iteration).
  - A lighter arrow from Evaluation back to Business Understanding (goal redefinition).
  - The canonical outer clockwise arrows connecting all phases.

### Bottom Annotation Bar

- Below the circular diagram, add a horizontal bar (~200px height) with three key takeaways:
  - Left: "数据工作 = 60-80% 项目时间" with a proportional bar chart (blue/green fill).
  - Center: "6 个质量关口 = 项目风险防火墙" with gate icons.
  - Right: "5 种角色 × 6 个阶段 = 协作矩阵" with a small role-color legend.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Borders**: 0.5 pt, `#e2e8f0` gray for sector boundaries.
- **Shadows**: Subtle `2px 2px 6px rgba(0,0,0,0.06)` on badges and gate icons.
- **Overall aesthetic**: Clean, structured, textbook-quality — data-driven and informative, not decorative.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_1_01_crisp_dm_practice.png`
- Place in `public/figures/ch14/`
