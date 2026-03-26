# Figure 12.4.3 — Hybrid Modeling Workflow: Visual Nodes + Python Script

## Prompt

Create a **professional, publication-quality workflow diagram** showing a hybrid modeling workflow that combines standard visual nodes with an embedded Python script node, demonstrating how visual and code-based approaches integrate seamlessly.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Main workflow** (top ~65%): A horizontal node chain on a light gray canvas.
- **Comparison strip** (bottom ~35%): Side-by-side "Visual Nodes" vs "Code" annotation with mapping lines.

### Main Workflow — Node Chain

Eight nodes arranged left-to-right on a canvas with light gray grid background (`#f8fafc`):

| Position | Node Label | Type | Color | Special |
|----------|-----------|------|-------|---------|
| 1 | CSV Reader | Source | Blue `#dbeafe`, border `#2563eb` | File icon, "data.csv" subtitle |
| 2 | Missing Value | Process | Blue `#dbeafe`, border `#2563eb` | Wrench icon |
| 3 | Column Filter | Process | Green `#dcfce7`, border `#16a34a` | Filter icon |
| 4 | **Python Script** | Script | **Yellow `#fef9c3`, border `#ca8a04`** | **Python logo icon, highlighted with glow effect** |
| 5 | Normalizer | Process | Green `#dcfce7`, border `#16a34a` | Scale icon |
| 6 | Random Forest | Model | Orange `#ffedd5`, border `#ea580c` | Tree icon |
| 7 | Cross Validation | Evaluate | Purple `#f3e8ff`, border `#9333ea` | Grid icon, "k=10" badge |
| 8 | Scorer | Evaluate | Purple `#f3e8ff`, border `#9333ea` | Chart icon |

Node 4 (Python Script) is **visually emphasized**:
- 1.5× the width of other nodes
- Bright yellow background with a subtle glow/halo effect (2 px golden border)
- Inside the node, show 3 lines of visible Python code (very small, ~6 pt, but legible):
  ```
  # Custom feature engineering
  df["ratio"] = df["A"] / df["B"]
  df["log_C"] = np.log1p(df["C"])
  ```
- A small badge below: "Custom Feature Engineering" in italic

All nodes connected by curved edges with arrowheads. Standard edges are gray `#475569`; edges entering and leaving the Python node are dashed orange `#ca8a04`.

### Annotation Banners

Two annotation banners below the workflow (floating, with arrow pointers):

**Banner 1** (pointing at nodes 1-3 and 5-8):
- Light blue background `#eff6ff`
- Text: "Built-in Nodes: Low Barrier + Quick Configuration"
- Icon: drag-and-drop cursor

**Banner 2** (pointing at node 4):
- Light yellow background `#fefce8`
- Text: "Python Script: Full Flexibility + Complete Ecosystem"
- Icon: code brackets `</>`

### Comparison Strip — Visual ↔ Code Mapping

At the bottom, a thin horizontal strip divided into two halves:

**Left half — "Visual Workflow Strengths"**:
- 4 small icon+text items in a row: "Drag & Drop" | "No Coding" | "Self-Documenting" | "Quick Iteration"
- Icons in blue/green tones

**Right half — "Code Integration Strengths"**:
- 4 small icon+text items: "Custom Logic" | "Full Libraries" | "Git-Friendly" | "Automation"
- Icons in yellow/orange tones

**Center divider**: A "+" symbol in a circle, indicating "Best of Both Worlds" in bold text below.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese text; JetBrains Mono for code.
- **Color palette**:
  - Data nodes: blue (`#2563eb`, `#dbeafe`)
  - Process nodes: green (`#16a34a`, `#dcfce7`)
  - Model nodes: orange (`#ea580c`, `#ffedd5`)
  - Evaluate nodes: purple (`#9333ea`, `#f3e8ff`)
  - Python script: yellow/gold (`#ca8a04`, `#fef9c3`)
  - Neutral: gray `#64748b`
- **Node size**: Standard nodes ~130 × 48 px; Python node ~200 × 60 px.
- **Node borders**: 1.5 pt solid (2 pt for Python node), corner radius 10 px.
- **Edges**: 1.5 pt curved, gray; dashed orange for Python node connections.
- **Shadows**: Subtle on all nodes; stronger golden glow on Python node.
- **Background**: Canvas `#f8fafc`, overall `#ffffff`, bottom strip `#f1f5f9`.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_4_03_hybrid_workflow.png`
- Place in `public/figures/ch12/`
