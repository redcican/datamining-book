# Figure 14.2.1 — Goal Alignment Matrix for Data Mining Projects

## Prompt

Create a **professional, publication-quality matrix infographic** depicting a Goal Alignment Matrix for a customer churn prediction project, showing how four stakeholder groups align (or conflict) across business objectives, mining goals, technical metrics, and acceptance criteria.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Main element**: A **4 × 5 grid/matrix** occupying ~75% of the width, with a summary column on the right.
- **Left header column**: Four stakeholder roles (rows).
- **Top header row**: Four goal layers + alignment status column (columns).

### Matrix Structure

**Column Headers** (top row, bold, 11pt):

| Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|
| 利益相关方 | 业务目标 | 挖掘目标 | 技术指标 | 对齐状态 |

**Row Data** (each row = one stakeholder):

| Stakeholder | Color | Business Objective | Mining Objective | Technical Metric | Alignment |
|---|---|---|---|---|---|
| 业务方 (Business) | Orange `#ea580c` | 流失率从 8% 降到 5% | 预测 30 天内流失 | Recall@Top-10% ≥ 60% | 🟢 已对齐 |
| 数据团队 (Data) | Blue `#2563eb` | 建立可迭代的预测能力 | 二分类 + 特征工程 | AUC-ROC ≥ 0.85 | 🟢 已对齐 |
| 工程团队 (Engineering) | Green `#16a34a` | 系统稳定可维护 | 在线推理服务 | 延迟 ≤ 100ms, 可用性 99.9% | 🟡 需讨论 |
| 合规团队 (Compliance) | Purple `#9333ea` | 符合个人信息保护法 | 模型可解释 + 特征审计 | 无敏感特征, 提供拒绝原因 | 🟡 需讨论 |

- Each cell is a **rounded rectangle** with the stakeholder's color at 10% opacity fill and 1px border.
- Text inside cells: 9pt, left-aligned, dark gray `#1e293b`.

### Alignment Status Column (Col 5)

- **Green circle** (🟢, `#16a34a`): "已对齐" — objectives are consistent.
- **Yellow circle** (🟡, `#eab308`): "需讨论" — potential conflict identified.
- **Red circle** (🔴, `#dc2626`): "冲突" — (not present in this example but shown in legend).

### Connecting Arrows

- Draw **blue directional arrows** (`#2563eb`, 1.5pt) connecting Col 4 → Col 3 → Col 2 horizontally across the matrix, showing the causal chain: Technical Metrics → Mining Goals → Business Objectives.
- Label above the arrows: "因果链: 技术指标 → 挖掘目标 → 业务价值" in 8pt italic.

### Conflict Highlight

- Between Row 1 (Business) and Row 3 (Engineering), draw a **red dashed connector** (`#dc2626`, 1pt dashed) between their Technical Metric cells.
- Annotation callout: "潜在冲突: 高召回率可能增加推理延迟" in 8pt red text, with a small warning icon (⚠️).

### Right Sidebar — Legend & Key Takeaway

- A narrow panel (~20% width) on the right with:
  - **Legend**: Green/Yellow/Red circles with labels "已对齐 / 需讨论 / 冲突".
  - **Key Insight box** (light blue `#eff6ff` background, thin `#2563eb` border):
    - "目标对齐是反复协商的过程"
    - "一个项目只能有一个北极星指标"
    - 9pt text, bold key phrases.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Grid lines**: 0.5 pt, `#e2e8f0`.
- **Cell padding**: 12px.
- **Shadows**: Subtle `2px 2px 6px rgba(0,0,0,0.05)` on the matrix container.
- **Overall aesthetic**: Clean, structured, corporate — like a well-designed project management slide, not a technical diagram.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_2_01_goal_alignment.png`
- Place in `public/figures/ch14/`
