# Figure 12.1.1 — Data Mining Platform Evolution Timeline

## Prompt

Create a **professional, publication-quality infographic** depicting the three-generation evolution of data mining platforms from 1990 to 2025.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Horizontal axis**: A clean, continuous timeline running left-to-right from **1990** to **2025**, with decade tick marks (1990, 2000, 2010, 2020) and a subtle grid.
- **Three horizontal era bands** span the full width, stacked vertically behind the timeline:
  - **Bottom band — Generation 1** "Single-Machine Statistical Tools" (1990s–2000s): soft blue background (`#dbeafe`), left-aligned label in bold navy text.
  - **Middle band — Generation 2** "Integrated Mining Platforms" (2000s–2010s): soft green background (`#dcfce7`), label in dark green.
  - **Top band — Generation 3** "Cloud-Native AI Platforms" (2010s–present): soft orange background (`#ffedd5`), label in dark orange.
- Each band's opacity fades in at the start year and remains solid through the end, suggesting gradual emergence rather than hard cutoffs.

### Platform Milestones

Place **rounded-rectangle badges** along the timeline at their first-release year. Each badge has:
- An **icon or small logo silhouette** (monochrome, stylized — not the actual logo for copyright safety).
- The **platform name** in bold and the **year** below in a smaller font.
- A subtle **drop shadow** to lift it above the era band.

Milestones to include (approximate positions):
| Platform | Year | Band |
|----------|------|------|
| SAS | 1976 (show at left edge ≈1990 with "est. 1976" note) | Gen 1 |
| SPSS | 1968 (same treatment) | Gen 1 |
| R | 1993 | Gen 1 |
| WEKA | 1999 | Gen 2 |
| RapidMiner | 2001 | Gen 2 |
| KNIME | 2004 | Gen 2 |
| Hadoop | 2006 | Gen 2 |
| scikit-learn | 2010 | Gen 3 |
| Apache Spark | 2014 | Gen 3 |
| TensorFlow | 2015 | Gen 3 |
| PyTorch | 2016 | Gen 3 |
| AWS SageMaker | 2017 | Gen 3 |
| Google Vertex AI | 2021 | Gen 3 |

### Trend Arrows (Bottom)

Below the timeline, draw **three gradient arrows** running left-to-right, each with a label pair at the two ends:
1. `Command-Line` → `Workflow` → `Notebook + AutoML` (interaction paradigm)
2. `MB–GB` → `TB` → `PB` (data scale)
3. `Statistical Methods` → `Machine Learning` → `Deep Learning` (algorithm complexity)

Use a smooth gradient fill (gray-to-accent-color) for each arrow, with labels in a clean sans-serif font.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Chinese labels (if any) in Noto Sans SC.
- **Color palette**: Accent colors `blue #2563eb`, `green #16a34a`, `orange #ea580c` — matching the book's standard `COLORS` dict.
- **Lines and dividers**: 0.5 pt, `#e2e8f0` gray.
- **No decorative clip art** — clean, data-driven, textbook aesthetic.
- **White background** with ample padding (≥ 40 px margins).
- Overall feel: the clarity of an Edward Tufte timeline crossed with a modern SaaS landing-page graphic.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_1_01_platform_evolution_timeline.png`
- Place in `public/figures/ch12/`
