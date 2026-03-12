"""
Shared matplotlib configuration for all textbook figures.

Usage:
    from shared.plot_config import apply_style, save_fig, COLORS

Every figure script should call apply_style() at the top before any plotting.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from pathlib import Path

# Register Windows CJK fonts if available (WSL2 / Windows)
for _font_path in [
    '/mnt/c/Windows/Fonts/simhei.ttf',
    '/mnt/c/Windows/Fonts/STXIHEI.TTF',
]:
    if Path(_font_path).exists():
        _fm.fontManager.addfont(_font_path)

# ── Color palette (accessible, print-friendly) ────────────────────────────
COLORS = {
    "blue":   "#2563eb",
    "red":    "#dc2626",
    "green":  "#16a34a",
    "orange": "#ea580c",
    "purple": "#9333ea",
    "teal":   "#0d9488",
    "gray":   "#64748b",
    "light":  "#e2e8f0",
}
PALETTE = list(COLORS.values())


def apply_style():
    """Apply the textbook-wide matplotlib style."""
    mpl.rcParams.update({
        # Font (CJK-safe fallback chain)
        "font.family":        ["SimHei", "STXiHei", "DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,

        # Figure
        "figure.dpi":         150,
        "figure.figsize":     (12, 8),
        "figure.facecolor":   "white",

        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.35,
        "axes.prop_cycle":    mpl.cycler(color=PALETTE),
        "axes.labelsize":     12,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",

        # Lines
        "lines.linewidth":    2.0,
        "lines.markersize":   7,

        # Legend
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.fontsize":    12,
        "legend.title_fontsize": 13,

        # Ticks
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,

        # Save
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
    })


def save_fig(fig: plt.Figure, script_path: str, name: str) -> Path:
    """
    Save figure directly to public/figures/<chXX>/ for web serving.

    All markdown files can then reference the image via the absolute
    path /figures/<chXX>/<name>.png with no relative path gymnastics.

    Args:
        fig:         The matplotlib Figure object.
        script_path: Pass __file__ from the calling script.
        name:        Output filename without extension (e.g. "fig1_1_01_timeline").
    """
    script_path = Path(script_path)
    ch_dir = script_path.parent.name           # e.g. "ch01"
    project_root = script_path.parent.parent.parent  # code/chXX/../.. = project root
    pub_dir = project_root / "public" / "figures" / ch_dir
    pub_dir.mkdir(parents=True, exist_ok=True)
    out_path = pub_dir / f"{name}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close(fig)
    return out_path


def add_panel_label(ax: plt.Axes, label: str, fontsize: int = 12):
    """Add a panel label (a), (b), (c) … to the top-left of an axes."""
    ax.text(
        -0.08, 1.02, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
    )
