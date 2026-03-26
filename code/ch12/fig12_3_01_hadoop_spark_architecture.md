# Figure 12.3.1 — Hadoop vs Spark Architecture Comparison

## Prompt

Create a **professional, publication-quality split architecture diagram** comparing the Hadoop classic stack (left) and the Apache Spark stack (right), highlighting their structural differences and the shift from disk-based to memory-based computation.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two-column split** with a vertical dashed divider in the center:
  - **Left half (~48%)**: Hadoop classic architecture, labeled "Hadoop 经典架构" at the top in bold.
  - **Right half (~48%)**: Spark architecture, labeled "Spark 架构" at the top in bold.
- **Center divider** (~4%): A vertical dashed line with **three comparison callout badges** placed on top of it (see below).

### Left Half — Hadoop Architecture (Blue Palette)

Three horizontal layers stacked bottom-to-top, each a rounded rectangle with blue-tinted fill:

| Layer | Label | Color | Content |
|-------|-------|-------|---------|
| Bottom | HDFS (Distributed Storage) | `#dbeafe` fill, `#2563eb` border | Show 4 small server icons arranged horizontally, each containing 3 small block rectangles labeled "Block". Add "×3 replicas" annotation with a small copy icon. A "NameNode" box sits above center with arrows down to each server. |
| Middle | YARN (Resource Manager) | `#e0f2fe` fill, `#0ea5e9` border | A central "ResourceManager" box with arrows to 4 "NodeManager" boxes aligned with the servers below. |
| Top | MapReduce Job | `#eff6ff` fill, `#2563eb` border | A left-to-right flow: `Input Splits` → `Map Tasks` (4 parallel boxes) → `Shuffle & Sort` (with a **disk icon** and red warning badge "Disk I/O") → `Reduce Tasks` (2 boxes) → `Output (HDFS)`. |

Between Middle and Top layers, draw a thin arrow labeled "Submit Job".

### Right Half — Spark Architecture (Orange Palette)

Three horizontal layers stacked bottom-to-top, each a rounded rectangle with orange-tinted fill:

| Layer | Label | Color | Content |
|-------|-------|-------|---------|
| Bottom | Storage (Pluggable) | `#fff7ed` fill, `#ea580c` border | Show 3 storage source icons side by side: "HDFS", "S3", "Local FS", connected by a horizontal bar. |
| Middle | Spark Core (Driver + Executors) | `#ffedd5` fill, `#ea580c` border | A "Driver" box (left, larger) with "SparkContext" label inside, arrows fanning out to 4 "Executor" boxes (right). Each Executor contains a small "Cache" block highlighted in bright orange. |
| Top | Spark Modules | `#fff7ed` fill, `#ea580c` border | Four equal-width module boxes arranged horizontally: **Spark SQL**, **MLlib**, **Streaming**, **GraphX**. Each has a small icon (table, brain, stream, graph). |

Between Middle and Top layers, draw a thin arrow labeled "DataFrame / RDD API".

### Center Comparison Badges

Three rounded-rectangle badges centered on the vertical divider, positioned at heights corresponding to the three layers:

| Badge | Left Text (Hadoop) | Arrow | Right Text (Spark) |
|-------|-------------------|-------|-------------------|
| Top | "Two-stage (Map → Reduce)" | ⟷ | "DAG Execution Engine" |
| Middle | "Disk I/O between stages" | ⟷ | "In-Memory Caching" |
| Bottom | "Batch Only" | ⟷ | "Batch + Streaming" |

Each badge: white fill, gray `#64748b` border, bold text, 9 pt font. The left text in blue `#2563eb`, right text in orange `#ea580c`, arrow in gray.

### Bottom Annotation Bar

A thin horizontal bar across the full width at the very bottom:
- Left side: "2006 — Hadoop 开源发布" in blue.
- Right side: "2014 — Spark 成为 Apache 顶级项目" in orange.
- Center: "Performance: Spark achieves 10–100× speedup for iterative ML algorithms" in dark gray italic.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for Chinese labels.
- **Color palette**: Hadoop = blue family (`#2563eb`, `#dbeafe`, `#e0f2fe`); Spark = orange family (`#ea580c`, `#ffedd5`, `#fff7ed`); neutral = gray `#64748b`.
- **Layer borders**: 1.5 pt solid, corner radius 12 px.
- **Icons**: Simple, monochrome line icons — no clip art.
- **Shadows**: Subtle (`2px 2px 6px rgba(0,0,0,0.06)`) on layer boxes.
- **Background**: Pure white `#ffffff` with ≥ 40 px margins.
- Aesthetic: clean, symmetric split-view diagram. The visual weight should feel balanced between the two halves.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_3_01_hadoop_spark_architecture.png`
- Place in `public/figures/ch12/`
