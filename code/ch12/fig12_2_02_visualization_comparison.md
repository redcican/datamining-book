# Figure 12.2.2 — Python Visualization Library Comparison

## Prompt

Create a **professional, publication-quality triptych** showing the same dataset visualized with three different Python plotting libraries, highlighting their distinct strengths and aesthetics.

### Layout & Composition

- **Orientation**: Landscape (16:9 aspect ratio, ~1600 × 900 px at 150 DPI).
- **Three equal-width panels** arranged side by side with 20 px gaps.
- Each panel has:
  - A **header bar** (8 px tall, colored) at the top with the library name in bold white text.
  - The **chart area** (main content).
  - A **footer label** (1 line, 9 pt italic gray) describing the best use case.

### Dataset

All three panels visualize the **same synthetic scatter dataset** (e.g., Iris-like: two continuous features, one categorical variable with 3 classes). The data points are identical across panels — only the visual treatment differs.

- ~150 data points, 3 groups (clusters) with distinct colors.
- X-axis: "Feature 1 (Sepal Length)" — range roughly 4–8.
- Y-axis: "Feature 2 (Petal Length)" — range roughly 1–7.
- Groups: "Setosa", "Versicolor", "Virginica".

### Panel (a) — matplotlib

- **Header color**: Blue `#2563eb`
- **Header text**: "matplotlib"
- **Chart style**: Basic scatter plot with manual styling.
  - Three marker groups with `COLORS` blue, orange, green.
  - Simple legend in upper-left corner.
  - Grid lines visible (light gray).
  - Axis labels in standard font.
  - Title: "Basic Scatter Plot" in 11 pt.
  - Overall feel: functional, clean, no frills — showing that matplotlib requires manual configuration but gives full control.
- **Footer**: "Publication-quality · Full control · Manual styling"

### Panel (b) — seaborn

- **Header color**: Green `#16a34a`
- **Header text**: "seaborn"
- **Chart style**: `lmplot`-style scatter with regression lines.
  - Same data, but with per-group linear regression lines and shaded 95% confidence intervals.
  - Automatic color palette (muted tones).
  - Rug plots along axes (small tick marks showing marginal distributions).
  - Seaborn's default white-grid style.
  - Title: "Statistical Scatter + Regression" in 11 pt.
  - Overall feel: polished statistical visualization with minimal code — showing seaborn's strength in one-line statistical plots.
- **Footer**: "Statistical plots · One-line API · Auto-aesthetics"

### Panel (c) — plotly (static representation)

- **Header color**: Purple `#9333ea`
- **Header text**: "plotly"
- **Chart style**: Interactive-look scatter (rendered as static image but mimicking plotly's UI).
  - Same data with slightly larger markers and subtle drop shadows.
  - A **hover tooltip box** drawn near one data point, showing:
    ```
    Species: Virginica
    Sepal: 6.7
    Petal: 5.2
    ```
  - A **toolbar mockup** in the top-right corner (zoom, pan, home icons as small gray rectangles).
  - Plotly's characteristic white background with minimal gridlines.
  - Title: "Interactive Scatter + Hover" in 11 pt.
  - Overall feel: modern, interactive-ready — showing plotly's strength in exploration and dashboards.
- **Footer**: "Interactive · Hover tooltips · Dashboard-ready"

### Connecting Element

Below the three panels, a **horizontal arrow bar** spans the full width:
- Left end labeled: "More Control" (blue `#2563eb`)
- Right end labeled: "More Convenience" (purple `#9333ea`)
- Gradient fill from blue to purple.
- Midpoint labeled: "Statistical Focus" (green `#16a34a`)

This arrow communicates the trade-off spectrum across the three libraries.

### Style & Typography

- **Font**: Inter or Helvetica Neue; Noto Sans SC for CJK.
- **Color palette**: `blue #2563eb`, `green #16a34a`, `orange #ea580c`, `purple #9333ea`.
- **Panel borders**: 1 pt `#e2e8f0`, corner radius 8 px.
- **Background**: Pure white `#ffffff` (overall), each panel also white.
- **Shadows**: Subtle (`2px 2px 8px rgba(0,0,0,0.06)`) on each panel.
- Aesthetic: clean comparison layout, like a tool review in a tech blog, adapted for textbook quality.

### Output

- Export as **PNG** at 150 DPI, filename: `fig12_2_02_visualization_comparison.png`
- Place in `public/figures/ch12/`
