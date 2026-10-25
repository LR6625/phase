from pathlib import Path

from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
import numpy as np


MACHINES = [
    "Super Sorter",
    "PhyzBatch 9000",
    "ROCA Sorter",
    "ROCA Sorter Max",
    "CardBot",
    "Magic Sorter",
    "P.H.A.S.E.",
]
PRICE = np.array([40945, 45621, 38018, 51718, 13410, 4280, 410], dtype=float)
CARDS_PER_HOUR = np.array([1200, 3000, 700, 700, 1000, 300, 225], dtype=float)
CPH_PER_DOLLAR = CARDS_PER_HOUR / PRICE
PHASE_IDX = MACHINES.index("P.H.A.S.E.")
OTHERS_IDX = [i for i in range(len(MACHINES)) if i != PHASE_IDX]


THEMES = [
    {
        "name": "colorway_01_sunrise",
        "fig_bg": "#FFFFFF",
        "panel_bg": "#FFF7EF",
        "base": "#F38A2A",
        "accent": "#2E78C8",
        "phase_blue": "#2E78C8",
        "grid": "#A7C9EA",
        "ink": "#1E2A35",
    },
    {
        "name": "colorway_02_harbor",
        "fig_bg": "#FFFFFF",
        "panel_bg": "#F8FBFF",
        "base": "#F08B3B",
        "accent": "#2668B8",
        "phase_blue": "#2668B8",
        "grid": "#9DC0E4",
        "ink": "#172532",
    },
    {
        "name": "colorway_03_cream",
        "fig_bg": "#FFFFFF",
        "panel_bg": "#FFFDF8",
        "base": "#EC7B24",
        "accent": "#3A83D0",
        "phase_blue": "#3A83D0",
        "grid": "#B4D0EB",
        "ink": "#1F2E3A",
    },
]


def _money_tick(value, _pos):
    return f"${int(value):,}"


def build_exact_graph_copies_and_correlation(output_dir: Path, theme: dict) -> tuple[Path, Path, Path, float]:
    """
    Build exact standalone copies of the two existing poster graphs plus a
    dedicated correlation visualization for price vs speed.
    """
    phase_blue = theme["phase_blue"]
    edge = theme["accent"]

    # ------------------------------------------------------------------
    # Graph copy 1: Card Output per Dollar Spent (same values/style)
    # ------------------------------------------------------------------
    order = np.argsort(CPH_PER_DOLLAR)
    names_sorted = [MACHINES[i] for i in order]
    val_sorted = CPH_PER_DOLLAR[order]
    y = np.arange(len(names_sorted))

    fig_value, ax_value = plt.subplots(figsize=(12.8, 8.0), dpi=180, facecolor=theme["fig_bg"])
    ax_value.set_facecolor(theme["panel_bg"])

    colors = [phase_blue if n == "P.H.A.S.E." else theme["base"] for n in names_sorted]
    bars = ax_value.barh(y, val_sorted, color=colors, edgecolor=edge, linewidth=2.0)
    ax_value.set_yticks(y)
    ax_value.set_yticklabels(names_sorted, fontsize=15, color=theme["ink"], fontweight="bold")
    ax_value.tick_params(axis="x", colors=theme["ink"], labelsize=12)
    ax_value.tick_params(axis="y", pad=8)
    ax_value.invert_yaxis()
    ax_value.set_xlim(0.0, max(0.62, float(np.max(val_sorted)) * 1.16))
    ax_value.grid(axis="x", alpha=0.26, color=theme["grid"], linestyle="--")

    for spine in ax_value.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)

    ax_value.set_xlabel("Cards per hour per dollar", fontsize=14, color=theme["ink"], labelpad=10, fontweight="bold")
    ax_value.set_title("Card Output per Dollar Spent*", fontsize=22, color=theme["ink"], pad=14, fontweight="bold")

    for label in ax_value.get_yticklabels():
        if label.get_text() == "P.H.A.S.E.":
            label.set_color(phase_blue)
            label.set_fontweight("heavy")

    for bar, value, name in zip(bars, val_sorted, names_sorted):
        x = bar.get_width()
        text_color = phase_blue if name == "P.H.A.S.E." else theme["ink"]
        ax_value.text(
            x + (0.004 if name != "P.H.A.S.E." else 0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=14,
            color=text_color,
            fontweight="bold",
        )

    phase_val = CPH_PER_DOLLAR[PHASE_IDX]
    best_other_val = float(np.max(CPH_PER_DOLLAR[OTHERS_IDX]))
    multiplier = phase_val / best_other_val
    ax_value.text(
        0.98,
        0.98,
        f"P.H.A.S.E. value lead: {multiplier:.1f}x",
        transform=ax_value.transAxes,
        fontsize=15,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7A25C", edgecolor="#FFD0A5", linewidth=2.0),
    )
    ax_value.text(
        0.98,
        0.87,
        "* [Output rate (Cards/hr) / Price in CAD]",
        transform=ax_value.transAxes,
        fontsize=12,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.46", facecolor="#FFF2E5", edgecolor="#FFD0A5", linewidth=2.2),
    )
    fig_value.tight_layout()

    # ------------------------------------------------------------------
    # Graph copy 2: Speed vs Price (same values/style)
    # ------------------------------------------------------------------
    fig_speed, ax_speed = plt.subplots(figsize=(12.8, 8.0), dpi=180, facecolor=theme["fig_bg"])
    ax_speed.set_facecolor(theme["panel_bg"])

    ax_speed.scatter(
        PRICE[OTHERS_IDX],
        CARDS_PER_HOUR[OTHERS_IDX],
        s=290,
        c=theme["base"],
        edgecolors=theme["ink"],
        linewidth=1.8,
        alpha=0.95,
        zorder=3,
    )
    ax_speed.scatter(
        PRICE[PHASE_IDX],
        CARDS_PER_HOUR[PHASE_IDX],
        s=360,
        c=phase_blue,
        edgecolors=phase_blue,
        linewidth=2.8,
        alpha=1.0,
        zorder=5,
    )

    for i in OTHERS_IDX:
        offset = (8, 7)
        if MACHINES[i] == "Magic Sorter":
            offset = (10, 14)
        ax_speed.annotate(
            MACHINES[i],
            (PRICE[i], CARDS_PER_HOUR[i]),
            textcoords="offset points",
            xytext=offset,
            fontsize=11,
            color=theme["ink"],
            fontweight="bold",
        )

    ax_speed.annotate(
        "P.H.A.S.E.",
        (PRICE[PHASE_IDX], CARDS_PER_HOUR[PHASE_IDX]),
        textcoords="offset points",
        xytext=(14, -14),
        fontsize=16,
        color=phase_blue,
        fontweight="bold",
        zorder=6,
    )

    ax_speed.vlines(
        PRICE[PHASE_IDX],
        0,
        CARDS_PER_HOUR[PHASE_IDX],
        colors=phase_blue,
        linestyles="--",
        linewidth=1.8,
        alpha=0.8,
        zorder=2,
    )
    ax_speed.annotate(
        f"P.H.A.S.E. price: ${int(PRICE[PHASE_IDX]):,} CAD",
        xy=(PRICE[PHASE_IDX], 0),
        xytext=(0.05, -0.08),
        textcoords="axes fraction",
        fontsize=12,
        color=phase_blue,
        fontweight="bold",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=phase_blue, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#EAF2FC", edgecolor=phase_blue, linewidth=1.5),
        annotation_clip=False,
        zorder=7,
    )

    ax_speed.set_xlim(0, 56000)
    ax_speed.set_ylim(0, 3300)
    ax_speed.xaxis.set_major_locator(MultipleLocator(10000))
    ax_speed.xaxis.set_minor_locator(MultipleLocator(5000))
    ax_speed.xaxis.set_major_formatter(FuncFormatter(_money_tick))
    ax_speed.set_xlabel("Price (CAD)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_speed.set_ylabel("Sorting speed (cards/hour)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_speed.set_title("Speed vs Price", fontsize=24, color=theme["ink"], pad=16, fontweight="bold")
    ax_speed.grid(which="major", alpha=0.24, color=theme["grid"], linestyle="--")
    ax_speed.grid(which="minor", alpha=0.10, color=theme["grid"], linestyle=":")
    ax_speed.tick_params(axis="x", colors=theme["ink"], labelsize=11, rotation=20)
    ax_speed.tick_params(axis="y", colors=theme["ink"], labelsize=12)
    for spine in ax_speed.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)
    fig_speed.tight_layout()

    # ------------------------------------------------------------------
    # Correlation: Pearson r + regression visualization
    # ------------------------------------------------------------------
    corr = float(np.corrcoef(PRICE, CARDS_PER_HOUR)[0, 1])
    slope, intercept = np.polyfit(PRICE, CARDS_PER_HOUR, 1)
    x_line = np.linspace(float(np.min(PRICE)), float(np.max(PRICE)), 200)
    y_line = slope * x_line + intercept

    fig_corr, ax_corr = plt.subplots(figsize=(12.8, 8.0), dpi=180, facecolor=theme["fig_bg"])
    ax_corr.set_facecolor(theme["panel_bg"])

    ax_corr.scatter(
        PRICE[OTHERS_IDX],
        CARDS_PER_HOUR[OTHERS_IDX],
        s=220,
        c=theme["base"],
        edgecolors=theme["ink"],
        linewidth=1.4,
        alpha=0.95,
        zorder=3,
        label="Other machines",
    )
    ax_corr.scatter(
        PRICE[PHASE_IDX],
        CARDS_PER_HOUR[PHASE_IDX],
        s=300,
        c=phase_blue,
        edgecolors=phase_blue,
        linewidth=2.2,
        alpha=1.0,
        zorder=5,
        label="P.H.A.S.E.",
    )
    ax_corr.plot(
        x_line,
        y_line,
        color=theme["accent"],
        linewidth=2.4,
        linestyle="-",
        zorder=2,
        label="Linear fit",
    )

    for i, name in enumerate(MACHINES):
        ax_corr.annotate(
            name,
            (PRICE[i], CARDS_PER_HOUR[i]),
            textcoords="offset points",
            xytext=(8, 7),
            fontsize=10,
            color=theme["ink"],
            fontweight="bold",
        )

    ax_corr.set_xlim(0, 56000)
    ax_corr.set_ylim(0, 3300)
    ax_corr.xaxis.set_major_locator(MultipleLocator(10000))
    ax_corr.xaxis.set_minor_locator(MultipleLocator(5000))
    ax_corr.xaxis.set_major_formatter(FuncFormatter(_money_tick))
    ax_corr.set_xlabel("Price (CAD)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_corr.set_ylabel("Sorting speed (cards/hour)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_corr.set_title("Price vs Speed Correlation", fontsize=24, color=theme["ink"], pad=16, fontweight="bold")
    ax_corr.grid(which="major", alpha=0.24, color=theme["grid"], linestyle="--")
    ax_corr.grid(which="minor", alpha=0.10, color=theme["grid"], linestyle=":")
    ax_corr.tick_params(axis="x", colors=theme["ink"], labelsize=11, rotation=20)
    ax_corr.tick_params(axis="y", colors=theme["ink"], labelsize=12)
    for spine in ax_corr.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)

    ax_corr.text(
        0.98,
        0.98,
        f"Pearson r = {corr:.3f}",
        transform=ax_corr.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        color=theme["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF2E5", edgecolor="#FFD0A5", linewidth=2.0),
    )
    ax_corr.legend(loc="lower right", fontsize=11, frameon=True)
    fig_corr.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    value_copy_path = output_dir / f"phase_value_graph_copy_{theme['name']}.png"
    speed_copy_path = output_dir / f"phase_speed_vs_price_graph_copy_{theme['name']}.png"
    corr_path = output_dir / f"phase_price_speed_correlation_{theme['name']}.png"
    fig_value.savefig(value_copy_path, facecolor=fig_value.get_facecolor())
    fig_speed.savefig(speed_copy_path, facecolor=fig_speed.get_facecolor())
    fig_corr.savefig(corr_path, facecolor=fig_corr.get_facecolor())
    plt.close(fig_value)
    plt.close(fig_speed)
    plt.close(fig_corr)
    return value_copy_path, speed_copy_path, corr_path, corr


def build_visual_variant(output_dir: Path, theme: dict) -> tuple[Path, Path]:
    order = np.argsort(CPH_PER_DOLLAR)
    names_sorted = [MACHINES[i] for i in order]
    val_sorted = CPH_PER_DOLLAR[order]
    y = np.arange(len(names_sorted))

    fig = plt.figure(figsize=(22, 12.8), dpi=180, facecolor=theme["fig_bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.21], wspace=0.16)
    ax_bar = fig.add_subplot(gs[0, 0], facecolor=theme["panel_bg"])
    ax_scatter = fig.add_subplot(gs[0, 1], facecolor=theme["panel_bg"])
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.14, right=0.94)

    phase_blue = theme["phase_blue"]
    colors = [phase_blue if n == "P.H.A.S.E." else theme["base"] for n in names_sorted]
    edge = theme["accent"]

    bars = ax_bar.barh(y, val_sorted, color=colors, edgecolor=edge, linewidth=2.0)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names_sorted, fontsize=17, color=theme["ink"], fontweight="bold")
    ax_bar.tick_params(axis="x", colors=theme["ink"], labelsize=13)
    ax_bar.tick_params(axis="y", pad=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, max(0.62, float(np.max(val_sorted)) * 1.16))
    ax_bar.grid(axis="x", alpha=0.26, color=theme["grid"], linestyle="--")

    for spine in ax_bar.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)

    ax_bar.set_xlabel("Cards per hour per dollar", fontsize=16, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_bar.set_title("Card Output per Dollar Spent*", fontsize=25, color=theme["ink"], pad=16, fontweight="bold")

    for label in ax_bar.get_yticklabels():
        if label.get_text() == "P.H.A.S.E.":
            label.set_color(phase_blue)
            label.set_fontweight("heavy")

    for bar, value, name in zip(bars, val_sorted, names_sorted):
        x = bar.get_width()
        text_color = phase_blue if name == "P.H.A.S.E." else theme["ink"]
        ax_bar.text(
            x + (0.004 if name != "P.H.A.S.E." else 0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=16,
            color=text_color,
            fontweight="bold",
        )

    phase_val = CPH_PER_DOLLAR[PHASE_IDX]
    best_other_val = float(np.max(CPH_PER_DOLLAR[OTHERS_IDX]))
    multiplier = phase_val / best_other_val
    ax_bar.text(
        0.98,
        0.98,
        f"P.H.A.S.E. value lead: {multiplier:.1f}x",
        transform=ax_bar.transAxes,
        fontsize=18,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7A25C", edgecolor="#FFD0A5", linewidth=2.0),
    )
    ax_bar.text(
        0.98,
        0.87,
        "* [Output rate (Cards/hr) / Price in CAD]",
        transform=ax_bar.transAxes,
        fontsize=15,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.46", facecolor="#FFF2E5", edgecolor="#FFD0A5", linewidth=2.2),
    )

    # Honest linear scale with clear price divisions.
    ax_scatter.scatter(
        PRICE[OTHERS_IDX],
        CARDS_PER_HOUR[OTHERS_IDX],
        s=290,
        c=theme["base"],
        edgecolors=theme["ink"],
        linewidth=1.8,
        alpha=0.95,
        zorder=3,
    )
    ax_scatter.scatter(
        PRICE[PHASE_IDX],
        CARDS_PER_HOUR[PHASE_IDX],
        s=360,
        c=phase_blue,
        edgecolors=phase_blue,
        linewidth=2.8,
        alpha=1.0,
        zorder=5,
    )

    for i in OTHERS_IDX:
        offset = (8, 7)
        if MACHINES[i] == "Magic Sorter":
            offset = (10, 14)
        ax_scatter.annotate(
            MACHINES[i],
            (PRICE[i], CARDS_PER_HOUR[i]),
            textcoords="offset points",
            xytext=offset,
            fontsize=11,
            color=theme["ink"],
            fontweight="bold",
        )

    ax_scatter.annotate(
        "P.H.A.S.E.",
        (PRICE[PHASE_IDX], CARDS_PER_HOUR[PHASE_IDX]),
        textcoords="offset points",
        xytext=(14, -14),
        fontsize=16,
        color=phase_blue,
        fontweight="bold",
        zorder=6,
    )

    # Add linear regression trendline
    slope, intercept = np.polyfit(PRICE, CARDS_PER_HOUR, 1)
    x_line = np.linspace(float(np.min(PRICE)), float(np.max(PRICE)), 200)
    y_line = slope * x_line + intercept
    ax_scatter.plot(
        x_line,
        y_line,
        color=theme["accent"],
        linewidth=2.4,
        linestyle="-",
        zorder=2,
        alpha=0.8,
    )

    # Draw a direct price guide so the low-cost P.H.A.S.E. point reads clearly on the x-axis.
    ax_scatter.vlines(
        PRICE[PHASE_IDX],
        0,
        CARDS_PER_HOUR[PHASE_IDX],
        colors=phase_blue,
        linestyles="--",
        linewidth=1.8,
        alpha=0.8,
        zorder=2,
    )
    ax_scatter.annotate(
        f"P.H.A.S.E. price: ${int(PRICE[PHASE_IDX]):,} CAD",
        xy=(PRICE[PHASE_IDX], 0),
        xytext=(0.05, -0.08),
        textcoords="axes fraction",
        fontsize=12,
        color=phase_blue,
        fontweight="bold",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=phase_blue, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#EAF2FC", edgecolor=phase_blue, linewidth=1.5),
        annotation_clip=False,
        zorder=7,
    )

    ax_scatter.set_xlim(0, 56000)
    ax_scatter.set_ylim(0, 3300)
    ax_scatter.xaxis.set_major_locator(MultipleLocator(10000))
    ax_scatter.xaxis.set_minor_locator(MultipleLocator(5000))
    ax_scatter.xaxis.set_major_formatter(FuncFormatter(_money_tick))
    ax_scatter.set_xlabel("Price (CAD)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_scatter.set_ylabel("Sorting speed (cards/hour)", fontsize=15, color=theme["ink"], labelpad=12, fontweight="bold")
    ax_scatter.set_title("Speed vs Price", fontsize=24, color=theme["ink"], pad=16, fontweight="bold")
    ax_scatter.grid(which="major", alpha=0.24, color=theme["grid"], linestyle="--")
    ax_scatter.grid(which="minor", alpha=0.10, color=theme["grid"], linestyle=":")
    ax_scatter.tick_params(axis="x", colors=theme["ink"], labelsize=11, rotation=20)
    ax_scatter.tick_params(axis="y", colors=theme["ink"], labelsize=12)
    for spine in ax_scatter.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)

    corr = float(np.corrcoef(PRICE, CARDS_PER_HOUR)[0, 1])
    ax_scatter.text(
        0.98,
        0.98,
        f"Pearson r = {corr:.3f}",
        transform=ax_scatter.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        color=theme["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF2E5", edgecolor="#FFD0A5", linewidth=2.0),
    )

    fig.suptitle(
        "P.H.A.S.E. Compared by the Numbers",
        fontsize=42,
        color=theme["ink"],
        fontweight="bold",
        y=0.975,
    )

    summary = (
        "Using worst-case sorting estimates, P.H.A.S.E. still delivers dramatically better value efficiency than every listed competitor. "
        "Even under conservative assumptions, its cards-per-dollar output is the strongest in this comparison."
    )
    fig.text(
        0.5,
        0.065,
        summary,
        ha="center",
        va="bottom",
        color=theme["ink"],
        fontsize=15,
        wrap=True,
    )

    disclaimer = "Graphic generated with Python and Matplotlib (https://matplotlib.org/) using the provided comparison table values."
    fig.text(
        0.5,
        0.028,
        disclaimer,
        ha="center",
        va="bottom",
        color="#5E6E7D",
        fontsize=10,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"phase_compared_by_numbers_{theme['name']}.png"
    svg_path = output_dir / f"phase_compared_by_numbers_{theme['name']}.svg"
    fig.savefig(png_path, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, svg_path


def build_left_value_graph_only(output_dir: Path, theme: dict) -> tuple[Path, Path]:
    """Build only the left value graph from the poster layout."""
    order = np.argsort(CPH_PER_DOLLAR)
    names_sorted = [MACHINES[i] for i in order]
    val_sorted = CPH_PER_DOLLAR[order]
    y = np.arange(len(names_sorted))

    fig, ax_bar = plt.subplots(figsize=(12.8, 8.0), dpi=180, facecolor=theme["fig_bg"])
    ax_bar.set_facecolor(theme["panel_bg"])

    phase_blue = theme["phase_blue"]
    colors = [phase_blue if n == "P.H.A.S.E." else theme["base"] for n in names_sorted]
    edge = theme["accent"]

    bars = ax_bar.barh(y, val_sorted, color=colors, edgecolor=edge, linewidth=2.0)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(names_sorted, fontsize=15, color=theme["ink"], fontweight="bold")
    ax_bar.tick_params(axis="x", colors=theme["ink"], labelsize=12)
    ax_bar.tick_params(axis="y", pad=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, max(0.62, float(np.max(val_sorted)) * 1.16))
    ax_bar.grid(axis="x", alpha=0.26, color=theme["grid"], linestyle="--")

    for spine in ax_bar.spines.values():
        spine.set_color(theme["accent"])
        spine.set_linewidth(1.4)

    ax_bar.set_xlabel("Cards per hour per dollar", fontsize=14, color=theme["ink"], labelpad=10, fontweight="bold")
    ax_bar.set_title("Card Output per Dollar Spent*", fontsize=22, color=theme["ink"], pad=14, fontweight="bold")

    for label in ax_bar.get_yticklabels():
        if label.get_text() == "P.H.A.S.E.":
            label.set_color(phase_blue)
            label.set_fontweight("heavy")

    for bar, value, name in zip(bars, val_sorted, names_sorted):
        x = bar.get_width()
        text_color = phase_blue if name == "P.H.A.S.E." else theme["ink"]
        ax_bar.text(
            x + (0.004 if name != "P.H.A.S.E." else 0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=14,
            color=text_color,
            fontweight="bold",
        )

    phase_val = CPH_PER_DOLLAR[PHASE_IDX]
    best_other_val = float(np.max(CPH_PER_DOLLAR[OTHERS_IDX]))
    multiplier = phase_val / best_other_val
    ax_bar.text(
        0.98,
        0.98,
        f"P.H.A.S.E. value lead: {multiplier:.1f}x",
        transform=ax_bar.transAxes,
        fontsize=15,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7A25C", edgecolor="#FFD0A5", linewidth=2.0),
    )
    ax_bar.text(
        0.98,
        0.87,
        "* [Output rate (Cards/hr) / Price in CAD]",
        transform=ax_bar.transAxes,
        fontsize=12,
        color="#102335",
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.46", facecolor="#FFF2E5", edgecolor="#FFD0A5", linewidth=2.2),
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"phase_left_value_graph_only_{theme['name']}.png"
    svg_path = output_dir / f"phase_left_value_graph_only_{theme['name']}.svg"
    fig.savefig(png_path, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, svg_path


def build_visual_set(output_dir: Path) -> list[tuple[Path, Path]]:
    outputs = []
    for theme in THEMES:
        outputs.append(build_visual_variant(output_dir, theme))
    return outputs


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    target_dir = here / "poster_iterations_3"
    generate_left_graph_only = False

    # Standalone left graph export using the most yellow existing colorway.
    yellow_theme = THEMES[0]
    left_png, left_svg = build_left_value_graph_only(target_dir, yellow_theme)
    print(f"Saved left graph visual: {left_png}")
    print(f"Saved left graph visual: {left_svg}")

    if generate_left_graph_only:
        raise SystemExit(0)

    generated = build_visual_set(target_dir)
    for out_png, out_svg in generated:
        print(f"Saved poster visual: {out_png}")
        print(f"Saved poster visual: {out_svg}")

    for theme in THEMES:
        value_copy, speed_copy, corr_plot, corr = build_exact_graph_copies_and_correlation(target_dir, theme)
        print(f"Saved exact graph copy: {value_copy}")
        print(f"Saved exact graph copy: {speed_copy}")
        print(f"Saved correlation visual: {corr_plot}")
        print(f"Price-speed Pearson correlation (r): {corr:.6f}")
