"""
General properties plotting for mPFC electrical synapse pairs.

PyCharm: Run this file. It will ask for the data folder path.
Expected Excel name (in that folder):
  General_Properties_FOR_PLOTTING_ONLY.xlsx

Figures are saved to:
  <chosen_folder>/general properties figures/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXCEL_NAME = "General_Properties_FOR_PLOTTING_ONLY.xlsx"
EXCEL_NAME_ALIASES = (
    "General Properties FOR PLOTTING ONLY.xlsx",
    "Generaral Properties FOR PLOTTING ONLY.xlsx",
)
FIGURES_DIR_NAME = "general properties figures"
POOL_DAYS = {1, 2}
MT_AGE_PLACEHOLDER = 400  # x-position for MT on age plots (days)
P50_ZOOM_MAX = 50
DAY_PAIR_OFFSET = 0.14  # left/right shift within a day (wider for visibility)
DAY_X = {1: 1.0, 2: 1.55, 3: 2.1}  # days closer together
POINT_SIZE = 36
POINT_ALPHA = 0.55
MT_AGE_OFFSET = 50  # MT darker than max age by this many "days" on color scale
STRETCH_AGE_HI = 50  # ages up to this use most of the colorbar (linear ruler)
STRETCH_FRAC = 0.75  # fraction of colorbar reserved for <= STRETCH_AGE_HI
MAJOR_AGE_TARGETS = (20, 30, 40, 50)
# saturated orange (young) -> almost-black purple (old / MT)
AGE_CMAP = LinearSegmentedColormap.from_list(
    "age_orange_purple",
    ["#FB8C00", "#EF6C00", "#D84315", "#8E24AA", "#4A148C", "#12001F"],
)
SHAPIRO_ALPHA = 0.05
DAY_TICK_LABELS = {1: "Day 1", 2: "Day 2", 3: "Day 3"}

PARAM_DIRS = {
    "Gj": ("Gj_12", "Gj_21"),
    "CC": ("CC_12", "CC_21"),
    "Rin": ("Rin_1", "Rin_2"),
    "Cm": ("Cm_1", "Cm_2"),
    "Vm": ("V_rest_mV_1", "V_rest_mV_2"),
}

PARAM_MATH = {
    "Gj": r"$G_{j}$",
    "CC": r"$CC$",
    "Rin": r"$R_{in}$",
    "Cm": r"$C_{m}$",
    "Vm": r"$V_{m}$",
}

PARAM_YLABELS = {
    "Gj": r"$G_{j}$ [nS]",
    "CC": r"$CC$",
    "Rin": r"$R_{in}$ [M$\Omega$]",
    "Cm": r"$C_{m}$ [pF]",
    "Vm": r"$V_{m}$ [mV]",
}

# Day 1–2–3 figure panels (identity left/right, not sorted)
DAY_FIG_COUPLING = [
    ("Gj", "Gj_12", "Gj_21"),
    ("CC", "CC_12", "CC_21"),
]
DAY_FIG_CELL = [
    ("Rin", "Rin_1", "Rin_2"),
    ("Cm", "Cm_1", "Cm_2"),
    ("Vm", "V_rest_mV_1", "V_rest_mV_2"),
]
STAT_PARAMS = [
    ("Gj", "Gj_12", "Gj_21", "Gj_pair_mean"),
    ("CC", "CC_12", "CC_21", "CC_pair_mean"),
    ("Rin", "Rin_1", "Rin_2", "Rin_pair_mean"),
    ("Cm", "Cm_1", "Cm_2", "Cm_pair_mean"),
    ("Vm", "V_rest_mV_1", "V_rest_mV_2", "Vm_pair_mean"),
]

AI_FORMULA = r"$AI=(max-min)/(max+min)$"
PAIR_MEAN_FORMULA = r"$x_{pair}=(x_1+x_2)/2$"
POOL_FORMULA = r"pool $=\{x_1,x_2\}$ for each pair"
TEST_RULE = r"Shapiro ($\alpha=0.05$) $\rightarrow$ Welch if both normal, else Mann–Whitney"


def ask_data_folder(cli_folder: str | None = None) -> Path:
    if cli_folder:
        raw = cli_folder
    else:
        print("Enter the folder that contains the Excel file:")
        print(f"  (expected file name: {EXCEL_NAME})")
        raw = input("> ").strip().strip('"').strip("'")
    if not raw:
        print("No folder given. Exit.")
        sys.exit(1)
    folder = Path(raw).expanduser()
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    return folder


def find_excel(folder: Path) -> Path:
    preferred = folder / EXCEL_NAME
    if preferred.is_file():
        return preferred
    for alias in EXCEL_NAME_ALIASES:
        alt = folder / alias
        if alt.is_file():
            print(f"Using Excel: {alt.name}")
            return alt
    candidates = sorted(p for p in folder.glob("*.xlsx") if not p.name.startswith("~$"))
    plotting = [p for p in candidates if "plotting" in p.name.lower().replace(" ", "_")]
    if len(plotting) == 1:
        print(f"Using Excel: {plotting[0].name}")
        return plotting[0]
    fuzzy = [
        p for p in candidates if "general" in p.name.lower() and "plotting" in p.name.lower()
    ]
    if len(fuzzy) == 1:
        print(f"Using Excel: {fuzzy[0].name}")
        return fuzzy[0]
    print(f"Excel not found: {preferred}")
    if candidates:
        print("xlsx files in folder:")
        for p in candidates:
            print(f"  - {p.name}")
    sys.exit(1)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_age(value) -> tuple[float | None, bool]:
    """Return (numeric_age_or_None, is_mt)."""
    if pd.isna(value):
        return None, False
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None, False
        return float(value), False
    text = str(value).strip()
    if not text:
        return None, False
    if text.upper() == "MT":
        return None, True
    m = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ".")), False
    return None, False


def load_and_prepare(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["pair_id", "age", "day", "Gj_12", "Gj_21"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df = df.copy()
    df["Gj_12"] = _to_float(df["Gj_12"])
    df["Gj_21"] = _to_float(df["Gj_21"])
    df = df[df["Gj_12"].notna() & df["Gj_21"].notna()].copy()

    df["day"] = _to_float(df["day"])
    for col_a, col_b in PARAM_DIRS.values():
        if col_a in df.columns:
            df[col_a] = _to_float(df[col_a])
        if col_b in df.columns:
            df[col_b] = _to_float(df[col_b])

    ages = df["age"].map(parse_age)
    df["age_num"] = [a for a, _ in ages]
    df["is_mt"] = [m for _, m in ages]
    df["age_plot"] = [
        MT_AGE_PLACEHOLDER if is_mt else age_num
        for age_num, is_mt in zip(df["age_num"], df["is_mt"])
    ]

    for name, (c1, c2) in PARAM_DIRS.items():
        if c1 not in df.columns or c2 not in df.columns:
            continue
        a = df[c1]
        b = df[c2]
        both = a.notna() & b.notna()
        mx = pd.Series(np.nan, index=df.index, dtype=float)
        mn = pd.Series(np.nan, index=df.index, dtype=float)
        ai = pd.Series(np.nan, index=df.index, dtype=float)
        ratio = pd.Series(np.nan, index=df.index, dtype=float)
        pair_mean = pd.Series(np.nan, index=df.index, dtype=float)
        mx[both] = np.maximum(a[both], b[both])
        mn[both] = np.minimum(a[both], b[both])
        denom = mx + mn
        ai[both] = np.where(denom[both] != 0, (mx[both] - mn[both]) / denom[both], np.nan)
        ratio[both] = np.where(mn[both] != 0, mx[both] / mn[both], np.nan)
        pair_mean[both] = (a[both] + b[both]) / 2.0
        df[f"{name}_max"] = mx
        df[f"{name}_min"] = mn
        df[f"AI_{name}"] = ai
        df[f"{name}_ratio"] = ratio
        df[f"{name}_pair_mean"] = pair_mean

    df["in_pool"] = df["day"].isin(POOL_DAYS)
    return df.reset_index(drop=True)


def age_limits(df: pd.DataFrame) -> tuple[float, float, float, bool]:
    """Return vmin, max_numeric_age, mt_scale_value, has_mt."""
    numeric = df.loc[~df["is_mt"], "age_num"].dropna()
    vmin = float(numeric.min()) if len(numeric) else 10.0
    max_age = float(numeric.max()) if len(numeric) else float(STRETCH_AGE_HI)
    if vmin == max_age:
        max_age = vmin + 1.0
    has_mt = bool(df["is_mt"].any())
    mt_val = max_age + MT_AGE_OFFSET
    return vmin, max_age, mt_val, has_mt


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
        return STRETCH_FRAC + (0.92 - STRETCH_FRAC) * (a - stretch_hi) / (max_age - stretch_hi)
    # between max_age and MT
    return 0.92 + 0.08 * (a - max_age) / max(mt_val - max_age, 1e-9)


def age_colors(age_num: np.ndarray, is_mt: np.ndarray, vmin: float, max_age: float, mt_val: float):
    colors = []
    for a, mt in zip(age_num, is_mt):
        u = age_to_unit(a, bool(mt), vmin, max_age, mt_val)
        colors.append(AGE_CMAP(u))
    # dummy Normalize only for APIs that expect one; colorbar uses custom mapping
    norm = Normalize(vmin=0.0, vmax=1.0)
    return colors, norm, AGE_CMAP


def unique_ages_for_colorbar(df: pd.DataFrame) -> list[float]:
    ages = sorted({float(a) for a in df.loc[~df["is_mt"], "age_num"].dropna().unique()})
    return ages


def _fmt_age_tick(t: float) -> str:
    return str(int(t)) if abs(t - round(t)) < 1e-9 else f"{t:g}"


def major_labeled_ages(ages: list[float]) -> list[float]:
    """
    Labels: minimum age, nearest available to 20/30/40/50,
    then ages >50 except the numeric maximum (max keeps a tick only;
    MT is labeled separately).
    """
    if not ages:
        return []
    ages = sorted(float(a) for a in ages)
    max_age = ages[-1]
    out: list[float] = [ages[0]]
    for target in MAJOR_AGE_TARGETS:
        nearest = min(ages, key=lambda a: abs(a - target))
        if nearest not in out:
            out.append(nearest)
    for a in ages:
        # keep tick for max age, but do not label it (e.g. 166)
        if a > STRETCH_AGE_HI and a < max_age and a not in out:
            out.append(a)
    return sorted(out)


def add_age_colorbar(
    fig,
    axs,
    vmin: float,
    max_age: float,
    mt_val: float,
    age_ticks: list[float] | None = None,
    has_mt: bool = False,
):
    """
    Colorbar: linear-stretched ruler to P50, compressed older tail.
    Minor ticks = all ages present; major labels = min, ~20/30/40/50, >50, MT.
    """
    sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap=AGE_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.035, pad=0.02)
    cbar.set_label("Age [postnatal days]")

    ages = list(age_ticks or [])
    major_ages = major_labeled_ages(ages)

    def _unit(t: float, is_mt: bool = False) -> float:
        return age_to_unit(t, is_mt, vmin, max_age, mt_val)

    # minor ticks: every observed age (+ MT mark without label duplication)
    minor_ages = list(ages)
    minor_units = [_unit(a) for a in minor_ages]
    if has_mt:
        minor_units.append(_unit(mt_val, is_mt=True))
    if minor_units:
        cbar.ax.yaxis.set_ticks(sorted(set(minor_units)), minor=True)
        cbar.ax.tick_params(which="minor", length=3, width=0.8)

    major_units = [_unit(a) for a in major_ages]
    major_labels = [_fmt_age_tick(a) for a in major_ages]
    if has_mt:
        major_units.append(_unit(mt_val, is_mt=True))
        major_labels.append("MT")
    if major_units:
        cbar.set_ticks(major_units)
        cbar.set_ticklabels(major_labels)
        cbar.ax.tick_params(which="major", length=6, width=1.0)

    if max_age > STRETCH_AGE_HI:
        u_div = _unit(float(STRETCH_AGE_HI))
        cbar.ax.axhline(u_div, color="0.75", lw=0.8, ls="--")
    return cbar


def savefig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def shapiro_ok(values: np.ndarray) -> tuple[bool, float | None]:
    """Return (is_normal, p_value). Requires n>=3."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return False, None
    # Shapiro is limited for very large n; subsample note not needed for this dataset size
    if len(v) > 5000:
        rng = np.random.default_rng(0)
        v = rng.choice(v, size=5000, replace=False)
    try:
        _, p = stats.shapiro(v)
    except Exception:
        return False, None
    return bool(p >= SHAPIRO_ALPHA), float(p)


def _mean_sem_median(v: np.ndarray) -> tuple[float, float, float]:
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(v))
    sem = float(stats.sem(v, ddof=1)) if len(v) > 1 else 0.0
    median = float(np.median(v))
    return mean, sem, median


def compare_two_groups(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    normal_a, p_sha = shapiro_ok(a)
    normal_b, p_shb = shapiro_ok(b)
    both_normal = normal_a and normal_b
    if both_normal and len(a) >= 2 and len(b) >= 2:
        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        test_name = "Welch t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann–Whitney U"
    mean1, sem1, med1 = _mean_sem_median(a)
    mean2, sem2, med2 = _mean_sem_median(b)
    return {
        "n1": len(a),
        "n2": len(b),
        "mean1": mean1,
        "mean2": mean2,
        "sem1": sem1,
        "sem2": sem2,
        "median1": med1,
        "median2": med2,
        "shapiro_p1": p_sha,
        "shapiro_p2": p_shb,
        "normal1": normal_a,
        "normal2": normal_b,
        "test": test_name,
        "stat": float(stat),
        "p": float(p),
    }


def format_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def pool_both_sides(df: pd.DataFrame, day: int, c1: str, c2: str) -> np.ndarray:
    sub = df[df["day"] == day]
    vals = pd.concat([sub[c1], sub[c2]], ignore_index=True)
    return vals.dropna().to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Figure 1: day comparison (X=day; pair left/right = channel identity)
# ---------------------------------------------------------------------------
def _style_pub_ax(ax):
    """Publication-like axis: no top/right frame, light spines, no grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("0.2")
    ax.spines["bottom"].set_color("0.2")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, colors="0.2", labelsize=9)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.grid(False)


def _style_day_ax(ax):
    """Day-figure axis: publication style + soft vertical day guides."""
    _style_pub_ax(ax)
    for x in (DAY_X[1], DAY_X[2], DAY_X[3]):
        ax.axvline(x, color="0.85", lw=0.7, zorder=0)


def plot_day_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    params: list[tuple[str, str, str]],
    outfile: str,
    subtitle: str = "",
):
    """Left/right = column identity (side 1 / side 2), not sorted by magnitude."""
    vmin, max_age, mt_val, has_mt = age_limits(df)
    age_ticks = unique_ages_for_colorbar(df)

    n_parts = []
    for day in (1, 2, 3):
        n = int((df["day"] == day).sum())
        n_parts.append(f"Day {day}: $n_{{ec}}$={n}")
    n_line = "   |   ".join(n_parts)

    n_panels = len(params)
    fig_w = 4.0 * n_panels + 1.5
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(
            1, n_panels, figsize=(fig_w, 4.4), sharex=True, constrained_layout=True
        )
        if n_panels == 1:
            axes = [axes]
        for ax, (name, c1, c2) in zip(axes, params):
            if c1 not in df.columns or c2 not in df.columns:
                ax.set_visible(False)
                continue
            sub = df[df[c1].notna() & df[c2].notna() & df["day"].notna()].copy()
            colors, _, _ = age_colors(
                sub["age_num"].to_numpy(),
                sub["is_mt"].to_numpy(),
                vmin,
                max_age,
                mt_val,
            )
            for (_, row), color in zip(sub.iterrows(), colors):
                day = int(row["day"])
                if day not in DAY_X:
                    continue
                x0 = DAY_X[day]
                x1 = x0 - DAY_PAIR_OFFSET
                x2 = x0 + DAY_PAIR_OFFSET
                y1, y2 = float(row[c1]), float(row[c2])
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.22, lw=0.7, zorder=1)
                ax.scatter(
                    [x1, x2],
                    [y1, y2],
                    c=[color, color],
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA,
                    zorder=2,
                    edgecolors="white",
                    linewidths=0.35,
                )
            ax.set_title("")
            ax.set_ylabel(PARAM_YLABELS.get(name, name))
            ax.set_xticks([DAY_X[1], DAY_X[2], DAY_X[3]])
            ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2], DAY_TICK_LABELS[3]])
            ax.set_xlabel("")
            ax.set_xlim(DAY_X[1] - 0.35, DAY_X[3] + 0.35)
            _style_day_ax(ax)

        cbar = add_age_colorbar(
            fig,
            list(axes),
            vmin=vmin,
            max_age=max_age,
            mt_val=mt_val,
            age_ticks=age_ticks,
            has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)

        title = "Electrical couples by recording day"
        if subtitle:
            title = f"{title} — {subtitle}"
        fig.suptitle(f"{title}\n{n_line}", fontsize=11, color="0.15")
        savefig(fig, out_dir, outfile)


# ---------------------------------------------------------------------------
# Figure 1b: day1 vs day2 statistics
# ---------------------------------------------------------------------------
def _plot_group_summary(ax, res: dict):
    """Mean±SEM (crimson) and median (blue diamonds)."""
    xs = np.array([1.0, 2.0])
    means = np.array([res["mean1"], res["mean2"]])
    sems = np.array([res["sem1"], res["sem2"]])
    meds = np.array([res["median1"], res["median2"]])
    ax.errorbar(
        xs,
        means,
        yerr=sems,
        fmt="o-",
        color="crimson",
        lw=1.8,
        markersize=7,
        capsize=4,
        zorder=3,
        label="mean ± SEM",
    )
    ax.plot(
        xs,
        meds,
        linestyle="none",
        marker="D",
        markersize=6,
        color="royalblue",
        zorder=4,
        label="median",
    )


def plot_day12_statistics(df: pd.DataFrame, out_dir: Path):
    """
    Two rows:
      1) pooled both sides {x1, x2} for day1 vs day2
      2) pair means (x1+x2)/2 for day1 vs day2
    Shapiro -> Welch or Mann-Whitney. Show mean±SEM and median.
    Includes Gj, CC, Rin, Cm, Vm on one figure.
    """
    params = list(STAT_PARAMS)
    n = len(params)

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n + 1.5, 7.5), constrained_layout=True)
    fig.suptitle(
        "Day 1 vs Day 2\n"
        + POOL_FORMULA
        + r"  |  "
        + PAIR_MEAN_FORMULA
        + "\n"
        + TEST_RULE
        + r"  |  mean $\pm$ SEM, median"
    )

    # Row 0: pooled both sides
    for ax, (name, c1, c2, _) in zip(axes[0], params):
        if c1 not in df.columns or c2 not in df.columns:
            ax.set_visible(False)
            continue
        d1 = pool_both_sides(df, 1, c1, c2)
        d2 = pool_both_sides(df, 2, c1, c2)
        res = compare_two_groups(d1, d2)
        ax.scatter(
            np.full(len(d1), 1) + np.random.default_rng(0).normal(0, 0.03, len(d1)),
            d1,
            s=14,
            alpha=0.45,
            color="0.45",
        )
        ax.scatter(
            np.full(len(d2), 2) + np.random.default_rng(1).normal(0, 0.03, len(d2)),
            d2,
            s=14,
            alpha=0.45,
            color="0.45",
        )
        _plot_group_summary(ax, res)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2]])
        ax.set_xlabel("")
        ax.set_ylabel(PARAM_YLABELS[name])
        sh1 = format_p(res["shapiro_p1"])
        sh2 = format_p(res["shapiro_p2"])
        ax.set_title(
            f"pooled sides\n"
            f"Shapiro D1 {sh1}, D2 {sh2}\n"
            f"{res['test']}: {format_p(res['p'])}\n"
            f"n={res['n1']}/{res['n2']}; "
            f"mean {res['mean1']:.3g}±{res['sem1']:.3g}, {res['mean2']:.3g}±{res['sem2']:.3g}\n"
            f"median {res['median1']:.3g}, {res['median2']:.3g}"
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
        ax.legend(frameon=False, fontsize=7, loc="best")

    # Row 1: pair means
    for ax, (name, c1, c2, mean_col) in zip(axes[1], params):
        if mean_col not in df.columns:
            ax.set_visible(False)
            continue
        s1 = df.loc[df["day"] == 1, mean_col].dropna().to_numpy(dtype=float)
        s2 = df.loc[df["day"] == 2, mean_col].dropna().to_numpy(dtype=float)
        res = compare_two_groups(s1, s2)
        ax.scatter(
            np.full(len(s1), 1) + np.random.default_rng(2).normal(0, 0.03, len(s1)),
            s1,
            s=18,
            alpha=0.55,
            color="0.45",
        )
        ax.scatter(
            np.full(len(s2), 2) + np.random.default_rng(3).normal(0, 0.03, len(s2)),
            s2,
            s=18,
            alpha=0.55,
            color="0.45",
        )
        _plot_group_summary(ax, res)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2]])
        ax.set_xlabel("")
        ax.set_ylabel(PARAM_YLABELS[name])
        sh1 = format_p(res["shapiro_p1"])
        sh2 = format_p(res["shapiro_p2"])
        ax.set_title(
            f"pair mean\n"
            f"Shapiro D1 {sh1}, D2 {sh2}\n"
            f"{res['test']}: {format_p(res['p'])}\n"
            f"n={res['n1']}/{res['n2']}; "
            f"mean {res['mean1']:.3g}±{res['sem1']:.3g}, {res['mean2']:.3g}±{res['sem2']:.3g}\n"
            f"median {res['median1']:.3g}, {res['median2']:.3g}"
        )
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(False, axis="y")
        ax.legend(frameon=False, fontsize=7, loc="best")

    savefig(fig, out_dir, "01b_day1_vs_day2_stats.png")

    # Print summary to console
    print("\nDay 1 vs Day 2 statistics:")
    for name, c1, c2, mean_col in params:
        d1 = pool_both_sides(df, 1, c1, c2)
        d2 = pool_both_sides(df, 2, c1, c2)
        rp = compare_two_groups(d1, d2)
        s1 = df.loc[df["day"] == 1, mean_col].dropna().to_numpy(dtype=float)
        s2 = df.loc[df["day"] == 2, mean_col].dropna().to_numpy(dtype=float)
        rm = compare_two_groups(s1, s2)
        print(
            f"  {name} pooled: {rp['test']} {format_p(rp['p'])} "
            f"(Shapiro {format_p(rp['shapiro_p1'])}/{format_p(rp['shapiro_p2'])})"
        )
        print(
            f"  {name} pair-mean: {rm['test']} {format_p(rm['p'])} "
            f"(Shapiro {format_p(rm['shapiro_p1'])}/{format_p(rm['shapiro_p2'])})"
        )


# ---------------------------------------------------------------------------
# Figure 2: AI vs age (+ P50 zoom)
# ---------------------------------------------------------------------------
def plot_ai_vs_age(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    metrics = [
        ("AI_Gj", r"AI $G_{j}$"),
        ("AI_CC", r"AI $CC$"),
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_{m}$"),
        ("AI_Vm", r"AI $V_{m}$"),
    ]
    vmin, max_age, mt_val, has_mt = age_limits(sub)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna()]
        d_num = d[~d["is_mt"]]
        colors_num, _, _ = age_colors(
            d_num["age_num"].to_numpy(),
            d_num["is_mt"].to_numpy(),
            vmin,
            max_age,
            mt_val,
        )
        ax.scatter(d_num["age_num"], d_num[col], c=colors_num, s=22, alpha=0.85, edgecolors="none")
        d_mt = d[d["is_mt"]]
        if len(d_mt):
            ax.axvline(MT_AGE_PLACEHOLDER - 40, color="0.5", ls="--", lw=1)
            mt_color = AGE_CMAP(1.0)
            ax.scatter(
                np.full(len(d_mt), MT_AGE_PLACEHOLDER),
                d_mt[col],
                color=mt_color,
                s=28,
                alpha=0.9,
                label="MT",
                zorder=3,
                edgecolors="none",
            )
            ax.legend(frameon=False, loc="upper right")
        ax.set_ylabel(title)
        ax.set_xlabel("Age [postnatal days]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)

    fig.suptitle(f"Asymmetry index vs age (days 1–2); MT at right of dashed line\n{AI_FORMULA}")
    savefig(fig, out_dir, "02_AI_vs_age.png")

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes2 = axes2.ravel()
    for ax, (col, title) in zip(axes2, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna() & (~sub["is_mt"]) & (sub["age_num"] <= P50_ZOOM_MAX)]
        colors_num, _, _ = age_colors(
            d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
        )
        ax.scatter(d["age_num"], d[col], c=colors_num, s=22, alpha=0.85, edgecolors="none")
        ax.set_ylabel(title)
        ax.set_xlabel("Age [postnatal days]")
        ax.set_xlim(0, P50_ZOOM_MAX)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    for ax in axes2[len(metrics) :]:
        ax.set_visible(False)
    fig2.suptitle(f"AI vs age zoom (≤ P{P50_ZOOM_MAX}, days 1–2, MT excluded)\n{AI_FORMULA}")
    savefig(fig2, out_dir, "02b_AI_vs_age_P50_zoom.png")


# ---------------------------------------------------------------------------
# Figure 3–4: asymmetry clouds max vs min
# ---------------------------------------------------------------------------
def _fit_slope_through_origin(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.allclose(x, 0):
        return None
    return float(np.dot(x, y) / np.dot(x, x))


def plot_asymmetry_clouds(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    units = {"Gj": "nS", "CC": "", "Rin": r"M$\Omega$", "Cm": "pF", "Vm": "mV"}

    def _panel(ax, name, with_fit: bool, unit_note: str = ""):
        xmin_c, xmax_c = f"{name}_min", f"{name}_max"
        if xmin_c not in sub.columns:
            ax.set_visible(False)
            return False
        d = sub[sub[xmin_c].notna() & sub[xmax_c].notna()].copy()
        colors, _, _ = age_colors(
            d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
        )
        ax.scatter(
            d[xmin_c],
            d[xmax_c],
            c=colors,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        lim_max = float(np.nanmax([d[xmin_c].max(), d[xmax_c].max()]))
        lim_min = 0.0
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            color="0.35",
            ls="--",
            lw=0.9,
            zorder=1,
            label="y = x",
        )
        if with_fit:
            k = _fit_slope_through_origin(d[xmin_c].to_numpy(), d[xmax_c].to_numpy())
            if k is not None:
                ax.plot(
                    [lim_min, lim_max],
                    [lim_min, k * lim_max],
                    color="crimson",
                    lw=1.2,
                    zorder=1,
                    label=f"k = {k:.2f}",
                )
        sym = PARAM_MATH.get(name, name)
        unit = f" [{unit_note}]" if unit_note else ""
        ax.set_xlabel(rf"{sym} min{unit}")
        ax.set_ylabel(rf"{sym} max{unit}")
        ax.set_title("")
        ax.set_aspect("equal", adjustable="box")
        _style_pub_ax(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        return True

    def _finish(fig, axes, outfile: str):
        cbar = add_age_colorbar(
            fig,
            list(axes),
            vmin=vmin,
            max_age=max_age,
            mt_val=mt_val,
            age_ticks=age_ticks,
            has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle("Asymmetry", fontsize=11, color="0.15")
        savefig(fig, out_dir, outfile)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        for ax, name in zip(axes, ["Gj", "CC"]):
            _panel(ax, name, with_fit=True, unit_note=units[name])
        _finish(fig, axes, "03_asymmetry_Gj_CC.png")

        fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
        for ax, name in zip(axes2, ["Rin", "Cm", "Vm"]):
            _panel(ax, name, with_fit=False, unit_note=units[name])
        _finish(fig2, axes2, "04_asymmetry_Rin_Cm_Vm.png")


# ---------------------------------------------------------------------------
# Figure 5: AI histograms
# ---------------------------------------------------------------------------
def plot_ai_histograms(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    metrics = [
        ("AI_Gj", r"AI $G_{j}$"),
        ("AI_CC", r"AI $CC$"),
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_{m}$"),
        ("AI_Vm", r"AI $V_{m}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        vals = sub[col].dropna().to_numpy()
        ax.hist(vals, bins=np.linspace(0, 1, 16), color="steelblue", edgecolor="white", alpha=0.9)
        if len(vals):
            ax.axvline(np.mean(vals), color="crimson", lw=1.5, label=f"mean={np.mean(vals):.2f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(f"AI distributions (days 1–2)\n{AI_FORMULA}")
    savefig(fig, out_dir, "05_AI_histograms.png")


def parse_args():
    p = argparse.ArgumentParser(description="Plot general properties of ES pairs")
    p.add_argument(
        "--folder",
        default=None,
        help="Folder with the Excel file. If omitted, the script asks interactively.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    folder = ask_data_folder(args.folder)
    excel_path = find_excel(folder)
    out_dir = folder / FIGURES_DIR_NAME
    print(f"Reading: {excel_path}")
    df = load_and_prepare(excel_path)
    print(f"Rows after Gj filter: {len(df)}")
    print(f"Days present: {sorted(df['day'].dropna().unique().tolist())}")
    print(f"MT pairs: {int(df['is_mt'].sum())}")
    print(f"Pool days 1–2: {int(df['in_pool'].sum())}")

    # Day 1–2–3 (identity left/right only)
    plot_day_comparison(
        df,
        out_dir,
        params=DAY_FIG_COUPLING,
        outfile="01_day_comparison_Gj_CC.png",
        subtitle=r"$G_{j}$ and $CC$",
    )
    plot_day_comparison(
        df,
        out_dir,
        params=DAY_FIG_CELL,
        outfile="01_day_comparison_Rin_Cm_Vm.png",
        subtitle=r"$R_{in}$, $C_m$, $V_m$",
    )

    plot_day12_statistics(df, out_dir)
    plot_ai_vs_age(df, out_dir)
    plot_asymmetry_clouds(df, out_dir)
    plot_ai_histograms(df, out_dir)
    print("Done. Figures are in:", out_dir)


if __name__ == "__main__":
    main()
