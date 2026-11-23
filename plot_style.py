"""
Shared figure design for general-properties plots.

Visual constants and drawing helpers only — no Excel I/O, no KW/Dunn stats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
DAY_X = {1: 1.0, 2: 1.55, 3: 2.1}
POINT_SIZE = 36
POINT_ALPHA = 0.55
MT_AGE_OFFSET = 50
MT_DISPLAY_LABEL = "Dams"  # Excel MT; postnatal age not recorded
STRETCH_AGE_HI = 50
STRETCH_FRAC = 0.75
MAJOR_AGE_TARGETS = (20, 30, 40, 50)
AGE_CMAP = LinearSegmentedColormap.from_list(
    "age_orange_purple",
    ["#FB8C00", "#EF6C00", "#D84315", "#8E24AA", "#4A148C", "#12001F"],
)


# ---------------------------------------------------------------------------
# Axis / save helpers
# ---------------------------------------------------------------------------
def style_pub_ax(ax):
    """Publication-like axis: no top/right frame, light spines, no grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("0.2")
    ax.spines["bottom"].set_color("0.2")
    ax.tick_params(
        axis="both", which="major", length=3.5, width=0.8, colors="0.2", labelsize=9
    )
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.grid(False)


def style_day_ax(ax):
    """Day-figure axis: publication style + soft vertical day guides."""
    style_pub_ax(ax)
    for x in (DAY_X[1], DAY_X[2], DAY_X[3]):
        ax.axvline(x, color="0.85", lw=0.7, zorder=0)


def savefig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    existed = path.is_file()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{'Overwrote' if existed else 'Saved'}: {path}")


def sig_bracket(
    ax,
    x1: float,
    x2: float,
    y: float,
    text: str,
    *,
    h: float | None = None,
    fs: float = 9,
    color: str = "0.2",
):
    """Draw a significance bracket between x1 and x2 at height y."""
    if h is None:
        ymin, ymax = ax.get_ylim()
        h = 0.03 * (ymax - ymin)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c=color, clip_on=False)
    ax.text(
        0.5 * (x1 + x2),
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fs,
        color=color,
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Warped age colour scale
# ---------------------------------------------------------------------------
def age_to_unit(
    age: float | None,
    is_mt: bool,
    vmin: float,
    max_age: float,
    mt_val: float,
) -> float:
    """
    Piecewise map age -> [0, 1] for colormap:
    - [vmin, STRETCH_AGE_HI] occupies STRETCH_FRAC of the bar (expanded)
    - (STRETCH_AGE_HI, max_age] is compressed
    - MT at mt_val = darkest (1.0)
    """
    if is_mt or age is None or (isinstance(age, float) and np.isnan(age)):
        return 1.0
    a = float(age)
    stretch_hi = min(float(STRETCH_AGE_HI), max_age)
    if a <= stretch_hi:
        if stretch_hi <= vmin:
            return 0.0
        return STRETCH_FRAC * (a - vmin) / (stretch_hi - vmin)
    if a <= max_age:
        if max_age <= stretch_hi:
            return STRETCH_FRAC
        return STRETCH_FRAC + (0.92 - STRETCH_FRAC) * (a - stretch_hi) / (
            max_age - stretch_hi
        )
    return 0.92 + 0.08 * (a - max_age) / max(mt_val - max_age, 1e-9)


def age_colors(
    age_num: np.ndarray, is_mt: np.ndarray, vmin: float, max_age: float, mt_val: float
):
    colors = []
    for a, mt in zip(age_num, is_mt):
        u = age_to_unit(a, bool(mt), vmin, max_age, mt_val)
        colors.append(AGE_CMAP(u))
    norm = Normalize(vmin=0.0, vmax=1.0)
    return colors, norm, AGE_CMAP


def fmt_age_tick(t: float) -> str:
    return str(int(t)) if abs(t - round(t)) < 1e-9 else f"{t:g}"


def major_labeled_ages(ages: list[float], min_gap: float = 8.0) -> list[float]:
    """Major age labels for colourbars (min, ~20/30/40/50, sparse older)."""
    if not ages:
        return []
    ages = sorted(float(a) for a in ages)
    max_age = ages[-1]
    out: list[float] = [ages[0]]
    for target in MAJOR_AGE_TARGETS:
        nearest = min(ages, key=lambda a: abs(a - target))
        if nearest not in out:
            out.append(nearest)
    nearest_p50 = min(ages, key=lambda a: abs(a - float(STRETCH_AGE_HI)))
    if nearest_p50 not in out:
        out.append(nearest_p50)
    for a in ages:
        if a <= float(STRETCH_AGE_HI) or a >= max_age:
            continue
        if all(abs(a - o) >= min_gap for o in out):
            out.append(a)
    return sorted(out)


def warped_age_x(age_num, is_mt, vmin, max_age, mt_val) -> float:
    return age_to_unit(age_num, bool(is_mt), vmin, max_age, mt_val)


def set_warped_age_ticks(
    ax, vmin, max_age, mt_val, age_ticks: list[float], has_mt: bool
):
    majors = major_labeled_ages(age_ticks)
    tick_u = [warped_age_x(a, False, vmin, max_age, mt_val) for a in majors]
    tick_lab = [fmt_age_tick(a) for a in majors]
    if has_mt:
        tick_u.append(warped_age_x(mt_val, True, vmin, max_age, mt_val))
        tick_lab.append(MT_DISPLAY_LABEL)
    if max_age > STRETCH_AGE_HI:
        u50 = warped_age_x(float(STRETCH_AGE_HI), False, vmin, max_age, mt_val)
        ax.axvline(u50, color="0.8", ls="--", lw=0.7, zorder=0)
    ax.set_xticks(tick_u)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_xlim(-0.02, 1.02)


def add_age_colorbar(
    fig,
    axs,
    vmin: float,
    max_age: float,
    mt_val: float,
    age_ticks: list[float] | None = None,
    has_mt: bool = False,
):
    """Vertical age colourbar (stretched to P50, compressed older tail)."""
    sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap=AGE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.035, pad=0.02)
    cbar.set_label("Age [postnatal days]")

    ages = list(age_ticks or [])
    major_ages = major_labeled_ages(ages)

    def _unit(t: float, is_mt: bool = False) -> float:
        return age_to_unit(t, is_mt, vmin, max_age, mt_val)

    minor_ages = list(ages)
    minor_units = [_unit(a) for a in minor_ages]
    if has_mt:
        minor_units.append(_unit(mt_val, is_mt=True))
    if minor_units:
        cbar.ax.yaxis.set_ticks(sorted(set(minor_units)), minor=True)
        cbar.ax.tick_params(which="minor", length=3, width=0.8)

    major_units = [_unit(a) for a in major_ages]
    major_labels = [fmt_age_tick(a) for a in major_ages]
    if has_mt:
        major_units.append(_unit(mt_val, is_mt=True))
        major_labels.append(MT_DISPLAY_LABEL)
    if major_units:
        cbar.set_ticks(major_units)
        cbar.set_ticklabels(major_labels)
        cbar.ax.tick_params(which="major", length=6, width=1.0)

    if max_age > STRETCH_AGE_HI:
        u_div = _unit(float(STRETCH_AGE_HI))
        cbar.ax.axhline(u_div, color="0.75", lw=0.8, ls="--")
    return cbar


def add_horizontal_age_colorbar_axis(
    fig,
    ax,
    *,
    vmin: float,
    max_age: float,
    mt_val: float,
    age_ticks: list[float],
    has_mt: bool,
):
    """Horizontal age colourbar under a warped-age scatter (figs 39–40)."""
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_xlim(-0.02, 1.02)
    if max_age > STRETCH_AGE_HI:
        u50 = warped_age_x(float(STRETCH_AGE_HI), False, vmin, max_age, mt_val)
        ax.axvline(u50, color="0.8", ls="--", lw=0.7, zorder=0)

    sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap=AGE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.085,
        pad=0.10,
        aspect=42,
    )
    ages = list(age_ticks or [])
    major_ages = major_labeled_ages(ages)

    def _unit(t: float, is_mt: bool = False) -> float:
        return age_to_unit(t, is_mt, vmin, max_age, mt_val)

    if ages:
        cbar.ax.set_xticks([_unit(a) for a in ages], minor=True)
        cbar.ax.tick_params(which="minor", length=3, width=0.6)
    major_units = [_unit(a) for a in major_ages]
    major_labels = [fmt_age_tick(a) for a in major_ages]
    if has_mt:
        major_units.append(_unit(mt_val, is_mt=True))
        major_labels.append(MT_DISPLAY_LABEL)
    if major_units:
        cbar.ax.set_xticks(major_units)
        cbar.ax.set_xticklabels(major_labels)
        cbar.ax.tick_params(which="major", length=6, width=1.0, labelsize=8)
    if max_age > STRETCH_AGE_HI:
        cbar.ax.axvline(_unit(float(STRETCH_AGE_HI)), color="0.75", lw=0.8, ls="--")
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Age [postnatal days]", fontsize=9)
    return cbar


def age_group_background_spans(
    bins: list,
    vmin: float,
    max_age: float,
    mt_val: float,
) -> list[tuple[str, float, float, str]]:
    """
    Warped-x spans for age-group bins (label, x0, x1, color).
    bins: list of (label, lo, hi, include_mt, color) like AGE_ASYM_BINS.
    """
    spans: list[tuple[str, float, float, str]] = []
    numeric_bins = [
        (lab, lo, hi, include_mt, color)
        for lab, lo, hi, include_mt, color in bins
        if not (lo is None and hi is None)
    ]
    for i, (lab, lo, hi, include_mt, color) in enumerate(numeric_bins):
        if hi is None:
            a0 = float(lo)
            x0 = warped_age_x(a0, False, vmin, max_age, mt_val)
            x1 = (
                1.0
                if include_mt
                else warped_age_x(max_age, False, vmin, max_age, mt_val)
            )
        else:
            a0 = float(lo) if i == 0 else float(lo) - 0.5
            a1 = float(hi) + 0.5
            a0 = max(a0, float(vmin))
            a1 = min(a1, float(max_age)) if max_age >= float(lo) else float(hi)
            if a1 < a0:
                continue
            x0 = warped_age_x(a0, False, vmin, max_age, mt_val)
            x1 = warped_age_x(a1, False, vmin, max_age, mt_val)
            if float(lo) >= float(STRETCH_AGE_HI):
                x0 = warped_age_x(
                    max(float(lo), float(STRETCH_AGE_HI) + 0.01),
                    False,
                    vmin,
                    max_age,
                    mt_val,
                )
                x1 = warped_age_x(
                    min(float(hi), float(max_age)),
                    False,
                    vmin,
                    max_age,
                    mt_val,
                )
        if x1 <= x0:
            x1 = min(x0 + 0.02, 1.0)
        spans.append((lab, float(x0), float(x1), color))

    for lab, lo, hi, include_mt, color in bins:
        if lo is None and hi is None and include_mt:
            x_mt = warped_age_x(None, True, vmin, max_age, mt_val)
            spans.append(
                (lab, max(0.0, x_mt - 0.045), min(1.0, x_mt + 0.01), color)
            )
    for i in range(len(spans) - 1):
        lab0, x0, x1, c0 = spans[i]
        lab1, y0, y1, c1 = spans[i + 1]
        if x1 < y0:
            mid = 0.5 * (x1 + y0)
            spans[i] = (lab0, x0, mid, c0)
            spans[i + 1] = (lab1, mid, y1, c1)
    return spans


def draw_age_group_background(
    ax,
    bins: list,
    vmin: float,
    max_age: float,
    mt_val: float,
    *,
    alpha: float = 0.11,
    label: bool = True,
):
    """Paint age-group colour bands behind a warped-age scatter (e.g. fig 40)."""
    spans = age_group_background_spans(bins, vmin, max_age, mt_val)
    for lab, x0, x1, color in spans:
        ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0, lw=0)
        if label:
            ax.text(
                0.5 * (x0 + x1),
                0.98,
                lab,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color=color,
                alpha=0.95,
                clip_on=False,
                rotation=0,
            )
