# Figure 12.3.2 — Spark Execution Model: RDD Lineage, DAG, and Memory Advantage

## Prompt

Create a **professional, publication-quality two-panel diagram** illustrating (a) how Spark compiles RDD transformations into a DAG execution plan, and (b) why in-memory caching makes Spark dramatically faster than MapReduce for iterative algorithms.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two panels**, separated by a thin vertical dashed line:
  - **Panel (a)** (~55% width): "RDD Lineage & DAG Execution", labeled at top.
  - **Panel (b)** (~45% width): "Iterative Algorithm: Memory vs Disk", labeled at top.
- Panel labels: Bold, 11 pt, with "(a)" and "(b)" prefixes in gray circles.

### Panel (a) — RDD Lineage & DAG

**Top half — RDD Transformation Chain**:

Show a left-to-right chain of 5 RDD nodes (rounded rectangles, orange `#ffedd5` fill, `#ea580c` border):

| RDD | Operation (arrow label) | Partitions | Type |
|-----|------------------------|------------|------|
| RDD₀ | `textFile("hdfs://...")` | 4 (shown as 4 small horizontal bars inside) | — |
| RDD₁ | `flatMap(split)` | 4 | Narrow |
| RDD₂ | `map(word → (word, 1))` | 4 | Narrow |
| RDD₃ | `reduceByKey(+)` | 3 | **Wide** (Shuffle) |
| Result | `collect()` | 1 | Action |

- **Narrow dependency** arrows: solid orange lines connecting partitions 1-to-1.
- **Wide dependency** arrow (RDD₂ → RDD₃): Multiple dashed lines fanning from each partition of RDD₂ to multiple partitions of RDD₃, with a "Shuffle" label badge (red `#dc2626` background, white text).
- The final `collect()` arrow is thicker, with an "Action triggers execution" annotation in italic gray.

**Bottom half — DAG Stage Breakdown**:

Below the RDD chain, draw a **DAG representation** showing how the above is compiled into 2 Stages:

- **Stage 1** (green dashed border `#16a34a`): encompasses RDD₀ → RDD₁ → RDD₂. Label: "Stage 1 — Pipelined (narrow deps)". Inside, show the 3 RDDs compressed into a single pipeline block with "4 parallel tasks" badge.
- **Stage 2** (purple dashed border `#9333ea`): encompasses RDD₃ → Result. Label: "Stage 2 — Post-Shuffle". Badge: "3 parallel tasks".
- A thick gray arrow from Stage 1 to Stage 2, labeled "Shuffle Barrier".
- Small annotation box: "Stages execute sequentially; tasks within a stage run in parallel across executors."

### Panel (b) — Memory vs Disk: Iterative Comparison

Show a **vertical timeline comparison** for an iterative algorithm (e.g., K-Means with 5 iterations):

**Left column — MapReduce** (gray/blue theme):
- A vertical sequence of 5 iteration blocks, each containing:
  - "Map" box → "HDFS Write" (disk icon, red warning) → "HDFS Read" (disk icon) → "Reduce" box
- Between iterations: a "Write to HDFS" step (highlighted with a red circle-cross icon).
- Total time bar on the left: long, labeled "~50 min".
- Color: gray `#94a3b8` boxes, red `#dc2626` disk I/O highlights.

**Right column — Spark** (orange theme):
- A vertical sequence showing:
  - "Iteration 1": "Read" → "Transform" → **"Cache in Memory"** (bright orange highlight with star icon)
  - "Iterations 2–5": "Transform" (reuses cached data, shown with a curved arrow back to cache). No disk I/O.
- Total time bar on the right: short (~1/10 of MapReduce), labeled "~5 min".
- Color: orange `#ea580c` boxes, bright fill for cache.

**Bottom comparison banner**:
- A centered badge: "10× faster for iterative ML algorithms" in bold, orange text on white background with orange border.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: Spark orange (`#ea580c`, `#ffedd5`), Stage 1 green (`#16a34a`), Stage 2 purple (`#9333ea`), Shuffle red (`#dc2626`), MapReduce gray (`#94a3b8`, `#64748b`).
- **RDD nodes**: 1.5 pt border, corner radius 10 px, partition bars 3 px height.
- **Arrows**: 1 pt, with arrowheads. Narrow deps solid, wide deps dashed.
- **Background**: Pure white `#ffffff` with ≥ 40 px margins.
- **Shadows**: Subtle on RDD nodes and stage borders.
- Aesthetic: the left panel resembles a compiler dataflow diagram; the right panel uses a Gantt-chart-like visual for time comparison.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_3_02_spark_execution_model.png`
- Place in `public/figures/ch12/`
