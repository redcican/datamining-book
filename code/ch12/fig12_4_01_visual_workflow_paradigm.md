# Figure 12.4.1 — Visual Workflow Paradigm: Canvas vs Code

## Prompt

Create a **professional, publication-quality split diagram** comparing a visual drag-and-drop data mining workflow (left) with its equivalent Python code (right), emphasizing the one-to-one correspondence between visual nodes and code statements.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two-panel layout** with a thin vertical dashed divider:
  - **Left panel (~55%)**: Visual workflow canvas, labeled "Visual Workflow (KNIME-style)" at top.
  - **Right panel (~45%)**: Equivalent Python code, labeled "Python Code (scikit-learn)" at top.
- **Bottom strip** (~12% height): Mapping arrows connecting nodes to code lines.

### Left Panel — Visual Workflow Canvas

A light gray canvas (`#f8fafc`) with a subtle grid pattern. Six rounded-rectangle nodes arranged in a top-to-bottom zigzag layout (natural workflow reading order), connected by curved edges with arrowheads:

| Node | Label | Icon | Color Fill | Status Indicator |
|------|-------|------|-----------|-----------------|
| 1 | CSV Reader | file icon | Blue `#dbeafe`, border `#2563eb` | Green dot ● (completed) |
| 2 | Missing Value | wrench icon | Blue `#dbeafe`, border `#2563eb` | Green dot ● |
| 3 | Column Filter | filter icon | Green `#dcfce7`, border `#16a34a` | Green dot ● |
| 4 | Normalizer | scale icon | Green `#dcfce7`, border `#16a34a` | Green dot ● |
| 5 | Random Forest Learner | tree icon | Orange `#ffedd5`, border `#ea580c` | Yellow dot ● (pending) |
| 6 | Scorer | chart icon | Purple `#f3e8ff`, border `#9333ea` | Yellow dot ● |

- Each node: 140 × 50 px, corner radius 10 px, with a small icon on the left and label text on the right.
- Status indicator: small filled circle (8 px) in the top-right corner of each node.
- Edges: 1.5 pt curved lines, dark gray `#475569`, with small filled arrowheads.
- Between nodes 4 and 5, show a **branch point**: an additional edge going from node 4 to a grayed-out "SVM Learner" node (dashed border, very light fill), illustrating the branching capability. This branch also connects to node 6.

### Right Panel — Python Code

A code editor-style panel with dark background (`#1e293b`) and syntax-highlighted Python code:

```python
# 1. Load data
df = pd.read_csv("data.csv")
# 2. Handle missing values
df = df.fillna(df.median())
# 3. Select features
X = df[selected_cols]
# 4. Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 5. Train model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# 6. Evaluate
accuracy = rf.score(X_test, y_test)
```

- Syntax highlighting: keywords in purple, strings in green, comments in gray, functions in blue.
- Line numbers on the left margin.
- Each comment (`# 1.` through `# 6.`) aligns horizontally with the corresponding node in the left panel.

### Bottom Strip — Mapping

Six thin dashed arrows connecting each node (left) to its corresponding code block (right), with a centered label: **"Visual Workflow ↔ Code Pipeline"** in bold gray text.

Color-coded mapping badges at each arrow:
- Nodes 1–2: "Data" (blue badge)
- Nodes 3–4: "Processing" (green badge)
- Node 5: "Modeling" (orange badge)
- Node 6: "Evaluation" (purple badge)

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese text. Code in JetBrains Mono or Fira Code.
- **Color palette**: Blue `#2563eb` (data), Green `#16a34a` (processing), Orange `#ea580c` (modeling), Purple `#9333ea` (evaluation), Gray `#64748b`.
- **Node borders**: 1.5 pt solid, corner radius 10 px.
- **Shadows**: Subtle on nodes (`2px 2px 6px rgba(0,0,0,0.06)`).
- **Background**: Left panel light gray `#f8fafc` (canvas), right panel dark `#1e293b` (code editor).
- **Overall feel**: Clean side-by-side comparison, immediately communicating that visual and code approaches are equivalent representations.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_4_01_visual_workflow_paradigm.png`
- Place in `public/figures/ch12/`
