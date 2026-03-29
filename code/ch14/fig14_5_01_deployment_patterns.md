# Figure 14.5.1 — Model Deployment Architecture Patterns

## Prompt

Create a **professional, publication-quality infographic** comparing three model deployment architectures (batch inference, online serving, streaming) with a deployment decision tree at the bottom.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Two rows**:
  - **Top row** (~65% height): Three architecture diagrams side by side.
  - **Bottom row** (~35% height): Deployment decision tree + grayscale release timeline.

### Top Row — Three Architecture Diagrams

Three equal-width **cards** (each ~480 × 450 px), separated by 30px gaps:

#### Card 1: 批量推理 (Batch Inference) — Blue Theme `#2563eb`

- **Header**: "批量推理" in blue bold 12pt, with a clock icon.
- **Architecture flow** (vertical, top-to-bottom):
  1. `Airflow 调度器` (rounded rect, light blue fill)
  2. ↓ arrow
  3. `Spark 计算引擎` (rounded rect, medium blue fill) — with model icon inside
  4. ↓ arrow
  5. `结果数据库` (cylinder shape, dark blue fill)
- **Metadata badges** (right side of card):
  - 延迟: "小时级" (blue pill)
  - 复杂度: "★☆☆" (1/3 stars)
  - 场景: "营销名单、月度评分"
- **Bottom bar**: "最简单，优先考虑" in blue text on light blue background.

#### Card 2: 在线服务 (Online Serving) — Orange Theme `#ea580c`

- **Header**: "在线服务" in orange bold 12pt, with a lightning bolt icon.
- **Architecture flow** (horizontal, left-to-right):
  1. `客户端请求` (rounded rect)
  2. → arrow
  3. `API 网关` (rounded rect, light orange)
  4. → arrow
  5. `模型容器` (rounded rect with Docker logo hint, medium orange) — with "< 100ms" label
  6. → arrow
  7. `实时响应` (rounded rect)
- **Load balancer** icon between API gateway and model container.
- **Metadata badges**:
  - 延迟: "50–200ms" (orange pill)
  - 复杂度: "★★☆" (2/3 stars)
  - 场景: "欺诈检测、推荐、搜索"
- **Bottom bar**: "逐条实时推理" in orange text.

#### Card 3: 流式处理 (Stream Processing) — Green Theme `#16a34a`

- **Header**: "流式处理" in green bold 12pt, with a wave/stream icon.
- **Architecture flow** (horizontal with feedback loop):
  1. `事件源 (IoT/日志)` (rounded rect)
  2. → arrow
  3. `Kafka 消息队列` (parallelogram, light green)
  4. → arrow
  5. `Flink 处理引擎` (rounded rect, medium green) — with embedded model icon
  6. → arrow
  7. `实时告警/写入` (rounded rect)
- A **feedback arrow** from output back to Kafka for downstream consumers.
- **Metadata badges**:
  - 延迟: "< 1s" (green pill)
  - 复杂度: "★★★" (3/3 stars)
  - 场景: "IoT 异常检测、实时风控"
- **Bottom bar**: "事件驱动，连续处理" in green text.

### Bottom Row Left (~60%) — Deployment Decision Tree

A **horizontal decision tree** flowing left to right:

```
[开始] → "业务能否接受小时级延迟？"
  → 是 → [批量推理] (blue box)
  → 否 → "需要逐条实时响应？"
    → 是 → [在线服务] (orange box)
    → 否 → "需要处理连续数据流？"
      → 是 → [流式处理] (green box)
```

- Decision diamonds in gray `#64748b`.
- "是" arrows in green, "否" arrows in red.
- Terminal boxes colored to match their respective architecture cards.
- Each terminal box has a small icon matching the card header.

### Bottom Row Right (~40%) — Grayscale Release Timeline

A **horizontal timeline** showing four stages of gradual release:

| Stage | Traffic | Duration | Color |
|-------|---------|----------|-------|
| 影子模式 | 0% | 1–2 周 | Light gray |
| 小流量测试 | 1–5% | 1 周 | Light blue |
| 扩大灰度 | 10–50% | 1–2 周 | Medium blue |
| 全量上线 | 100% | — | Dark blue |

- Each stage as a progressively wider/taller block on the timeline.
- Labels above each block: traffic percentage.
- Labels below: duration.
- Title: "灰度发布策略" in bold 10pt.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels in Noto Sans SC.
- **Color palette**: Book standard — `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`, `red #dc2626`, `gray #64748b`.
- **Background**: Pure white `#ffffff`.
- **Card backgrounds**: White with thin colored top border (4px).
- **Card shadows**: `2px 2px 8px rgba(0,0,0,0.06)`.
- **Margins**: ≥ 30 px between cards, ≥ 40 px outer margins.

### Output

- Export as **PNG** at 150 DPI, filename: `fig14_5_01_deployment_patterns.png`
- Place in `public/figures/ch14/`
