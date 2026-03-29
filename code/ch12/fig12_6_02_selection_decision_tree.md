# Figure 12.6.2 — Data Mining Tool Selection Decision Tree

## Prompt

Create a **professional, publication-quality decision tree diagram** for selecting data mining tools, with three decision layers and tool recommendations at the leaf nodes.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Main area (~80%)**: Decision tree flowing top-to-bottom.
- **Right sidebar (~20%)**: Simplified "3 Questions" quick-reference card.

### Decision Tree Structure

**Root Node** (top center):
- Diamond shape, gray `#64748b` fill
- Text: "Start: Tool Selection"
- Subtitle: "开始选型"

**Layer 1 — Data Scale** (first branch):
Root splits into 3 branches:

| Branch | Condition | Color |
|--------|-----------|-------|
| Left | "< 10 GB" | Blue `#2563eb` |
| Center | "10 GB – TB" | Orange `#ea580c` |
| Right | "> TB" | Red `#dc2626` |

Each condition label sits on the branch line in a small rounded badge.

**Layer 2 — Capability / Infrastructure**:

Left branch ("< 10 GB") splits into:
- "Python proficient?" → Yes / No
  - Yes → Layer 3A
  - No → Layer 3B

Center branch ("10 GB – TB") splits into:
- "Existing cluster?" → Yes / No
  - Yes → "PySpark + Spark MLlib" (leaf, orange)
  - No → Layer 3C (cloud)

Right branch ("> TB"):
- Direct to "Spark on YARN / Cloud Platform" (leaf, red)

**Layer 3 — Specific Recommendations** (leaf nodes):

Layer 3A (Python, < 10 GB):
- "Tabular ML" → **scikit-learn + XGBoost** (blue leaf)
- "Deep Learning" → **PyTorch / TensorFlow** (blue leaf)
- "AutoML" → **auto-sklearn / TPOT** (blue leaf)

Layer 3B (No coding, < 10 GB):
- "Teaching" → **Orange** (purple leaf)
- "Enterprise" → **KNIME** (green leaf)
- "Rapid Prototype" → **RapidMiner** (green leaf)

Layer 3C (No cluster, 10 GB – TB):
- "AWS" → **SageMaker** (orange leaf)
- "Google" → **Vertex AI** (blue leaf)
- "Spark focus" → **Databricks** (red leaf)

### Leaf Node Design

Each leaf node is a **rounded rectangle** with:
- Tool name in bold (12 pt)
- Category color fill (matching the tool's section in the book)
- A small icon representing the tool type (code icon, visual icon, cloud icon)
- Bottom line: one-word positioning (e.g., "经典 ML", "教学", "全栈云端")

### Decision Node Design

Each decision node is a **diamond** shape:
- Light gray fill `#f1f5f9`
- Border color matches the layer's theme color
- Question text in bold 10 pt
- Branch labels ("Yes"/"No" or specific conditions) on the edges

### Right Sidebar — Quick Reference

A card with rounded corners and light blue `#eff6ff` background:

**Title**: "3 Questions to Choose" (3 个问题锁定工具)

Three numbered items with icons:
1. 📊 "How big is your data?" (数据多大？)
   - < 10 GB → Single machine
   - > 10 GB → Distributed / Cloud
2. 💻 "Can your team code?" (团队会编程吗？)
   - Yes → Python ecosystem
   - No → Visual tools
3. 🚀 "Need production deployment?" (需要生产部署吗？)
   - Yes → Cloud platform / MLOps
   - No → Local tools sufficient

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: Layer 1 uses blue/orange/red for scale severity. Leaf nodes match their section colors (§12.2 blue, §12.3 orange, §12.4 green/purple, §12.5 cloud colors).
- **Decision diamonds**: 1.5 pt border, light gray fill, ~80 × 50 px.
- **Leaf rectangles**: 1.5 pt border, colored fill at 20% opacity, ~120 × 45 px.
- **Edges**: 1.5 pt solid, gray `#475569`, with arrowheads.
- **Edge labels**: 8 pt, colored badges on edges.
- **Background**: Pure white `#ffffff`.
- **Shadows**: Subtle on leaf nodes.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_6_02_selection_decision_tree.png`
- Place in `public/figures/ch12/`
