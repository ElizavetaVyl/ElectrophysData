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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
AI_FORMULA_VM_NOTE = r"for $V_m$: $AI$ from $|V_m|$"
# AI=0 equal; AI≈0.33 ≈ 2×; 0–0.33 = up to ~2-fold
PAIR_MEAN_FORMULA = r"$x_{pair}=(x_1+x_2)/2$"
POOL_FORMULA = r"pool $=\{x_1,x_2\}$ for each pair"
TEST_RULE = r"Shapiro ($\alpha=0.05$) $\rightarrow$ Welch if both normal, else Mann–Whitney"

# Age bins for asymmetry / AI group analyses (MT → P50+)
AGE_ASYM_BINS = [
    ("P10–32", 10, 32, False, "#FB8C00"),  # orange
    ("P33–50", 33, 50, False, "#8E24AA"),  # purple
    ("P50+", 50, None, True, "#2E0854"),   # dark violet; includes MT
]

AI_METRICS = [
    ("AI_Gj", r"AI $G_{j}$"),
    ("AI_CC", r"AI $CC$"),
    ("AI_Rin", r"AI $R_{in}$"),
    ("AI_Cm", r"AI $C_{m}$"),
    ("AI_Vm", r"AI $|V_{m}|$"),
]


def _age_bin_mask(df: pd.DataFrame, lo: float, hi: float | None, include_mt: bool) -> pd.Series:
    if hi is None:
        mask = df["age_num"].notna() & (df["age_num"] > lo)
    else:
        mask = df["age_num"].notna() & (df["age_num"] >= lo) & (df["age_num"] <= hi)
    if include_mt:
        mask = mask | df["is_mt"]
    else:
        mask = mask & (~df["is_mt"])
    return mask


def _age_group_label_color(age_num, is_mt) -> tuple[str | None, str | None]:
    for label, lo, hi, include_mt, color in AGE_ASYM_BINS:
        if is_mt:
            if include_mt:
                return label, color
            continue
        if age_num is None or (isinstance(age_num, float) and np.isnan(age_num)):
            return None, None
        if hi is None:
            if age_num > lo:
                return label, color
        elif lo <= age_num <= hi:
            return label, color
    return None, None


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
        # algebraic min/max (keep signed Vm for scatter plots)
        mx[both] = np.maximum(a[both], b[both])
        mn[both] = np.minimum(a[both], b[both])
        # AI: for Vm use absolute values so index stays in [0, 1]
        if name == "Vm":
            a_ai = a.abs()
            b_ai = b.abs()
            mx_ai = pd.Series(np.nan, index=df.index, dtype=float)
            mn_ai = pd.Series(np.nan, index=df.index, dtype=float)
            mx_ai[both] = np.maximum(a_ai[both], b_ai[both])
            mn_ai[both] = np.minimum(a_ai[both], b_ai[both])
        else:
            mx_ai, mn_ai = mx, mn
        denom = mx_ai + mn_ai
        ai[both] = np.where(
            denom[both] != 0, (mx_ai[both] - mn_ai[both]) / denom[both], np.nan
        )
        ratio[both] = np.where(mn_ai[both] != 0, mx_ai[both] / mn_ai[both], np.nan)
        pair_mean[both] = (a[both] + b[both]) / 2.0
        df[f"{name}_max"] = mx
        df[f"{name}_min"] = mn
        df[f"AI_{name}"] = ai
        df[f"{name}_ratio"] = ratio
        df[f"{name}_pair_mean"] = pair_mean

    # Distance (µm): accept a few header spellings from Excel
    dist_aliases = ("Distance, um", "Distance_um", "Distance", "distance", "Distance (um)")
    dist_col = next((c for c in dist_aliases if c in df.columns), None)
    if dist_col is not None:
        df["Distance_um"] = _to_float(df[dist_col])

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
# Figure 2: AI vs age (warped X, continuous age Greys) + separate group stats
# ---------------------------------------------------------------------------
def _warped_age_x(age_num, is_mt, vmin, max_age, mt_val) -> float:
    return age_to_unit(age_num, bool(is_mt), vmin, max_age, mt_val)


def _set_warped_age_ticks(ax, vmin, max_age, mt_val, age_ticks: list[float], has_mt: bool):
    majors = major_labeled_ages(age_ticks)
    tick_u = [_warped_age_x(a, False, vmin, max_age, mt_val) for a in majors]
    tick_lab = [_fmt_age_tick(a) for a in majors]
    if has_mt:
        tick_u.append(_warped_age_x(mt_val, True, vmin, max_age, mt_val))
        tick_lab.append("MT")
    # divider after P50
    if max_age > STRETCH_AGE_HI:
        u50 = _warped_age_x(float(STRETCH_AGE_HI), False, vmin, max_age, mt_val)
        ax.axvline(u50, color="0.8", ls="--", lw=0.7, zorder=0)
    ax.set_xticks(tick_u)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_xlim(-0.02, 1.02)


def plot_ai_vs_age(df: pd.DataFrame, out_dir: Path):
    """Scatter AI vs warped age; continuous age Greys colormap (same as day/asymmetry figs)."""
    sub = df[df["in_pool"]].copy()
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    sub = sub.copy()
    sub["x_warp"] = [
        _warped_age_x(a, bool(mt), vmin, max_age, mt_val)
        for a, mt in zip(sub["age_num"], sub["is_mt"])
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, constrained_layout=True)
        row1 = [("AI_Gj", r"AI $G_{j}$"), ("AI_CC", r"AI $CC$")]
        row2 = [
            ("AI_Rin", r"AI $R_{in}$"),
            ("AI_Cm", r"AI $C_{m}$"),
            ("AI_Vm", r"AI $V_{m}$"),
        ]

        def _ai_age_panel(ax, col: str, ylabel: str):
            if col not in sub.columns:
                ax.set_visible(False)
                return
            d = sub[sub[col].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            ax.scatter(
                d["x_warp"],
                d[col],
                c=colors,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            ax.axhline(1 / 3, color="0.6", ls=":", lw=0.8, zorder=0)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Age [postnatal days]")
            ax.set_ylim(0, 1.05)
            _set_warped_age_ticks(ax, vmin, max_age, mt_val, age_ticks, has_mt)
            _style_pub_ax(ax)

        # top: Gj, CC; bottom: Rin, Cm, Vm
        _ai_age_panel(axes[0, 0], *row1[0])
        _ai_age_panel(axes[0, 1], *row1[1])
        axes[0, 2].set_visible(False)
        for ax, (col, ylabel) in zip(axes[1], row2):
            _ai_age_panel(ax, col, ylabel)

        visible_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
        cbar = add_age_colorbar(
            fig,
            visible_axes,
            vmin=vmin,
            max_age=max_age,
            mt_val=mt_val,
            age_ticks=age_ticks,
            has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            "AI (asymmetry index) vs age\n"
            f"{AI_FORMULA}; X stretched to P50; dotted: AI≈0.33 (~2×)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "02_AI_vs_age.png")


def _dunn_posthoc(
    samples: list[np.ndarray], labels: list[str]
) -> list[tuple[str, str, float]]:
    """
    Dunn's test (Bonferroni-adjusted two-sided p) after Kruskal–Wallis.
    samples/labels aligned; empty groups are skipped.
    """
    from itertools import combinations

    groups = [
        (lab, np.asarray(s, dtype=float))
        for lab, s in zip(labels, samples)
        if len(s) > 0
    ]
    if len(groups) < 2:
        return []
    data = np.concatenate([s for _, s in groups])
    ranks = stats.rankdata(data)
    n_total = len(data)
    mean_ranks: dict[str, float] = {}
    ns: dict[str, int] = {}
    idx = 0
    for lab, s in groups:
        n = len(s)
        mean_ranks[lab] = float(np.mean(ranks[idx : idx + n]))
        ns[lab] = n
        idx += n
    # tie correction
    _, counts = np.unique(ranks, return_counts=True)
    tie_sum = float(np.sum(counts**3 - counts))
    tie_factor = 1.0 - tie_sum / (n_total**3 - n_total) if n_total > 1 else 1.0

    pairs = []
    raw_p = []
    for (lab_i, _), (lab_j, _) in combinations(groups, 2):
        se = np.sqrt(
            (n_total * (n_total + 1) / 12.0)
            * (1.0 / ns[lab_i] + 1.0 / ns[lab_j])
            * tie_factor
        )
        if se == 0:
            p = 1.0
        else:
            z = abs(mean_ranks[lab_i] - mean_ranks[lab_j]) / se
            p = float(2 * (1 - stats.norm.cdf(z)))
        pairs.append((lab_i, lab_j))
        raw_p.append(p)
    raw_p = np.asarray(raw_p, dtype=float)
    adj = np.minimum(raw_p * len(raw_p), 1.0)  # Bonferroni
    return [(a, b, float(p)) for (a, b), p in zip(pairs, adj)]


def _short_age_pair(a: str, b: str) -> str:
    def _short(lab: str) -> str:
        return lab.replace("P", "").replace("–", "-")

    return f"{_short(a)} vs {_short(b)}"


def plot_ai_by_age_groups(df: pd.DataFrame, out_dir: Path):
    """AI by age group: Kruskal–Wallis + Dunn post-hoc (Bonferroni)."""
    sub = df[df["in_pool"]].copy()
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    x_pos = {lab: i + 1 for i, lab in enumerate(group_labels)}
    row1 = [("AI_Gj", r"AI $G_{j}$"), ("AI_CC", r"AI $CC$")]
    row2 = [
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_{m}$"),
        ("AI_Vm", r"AI $|V_{m}|$"),
    ]

    def _panel(ax, col: str, ylabel: str):
        if col not in sub.columns:
            ax.set_visible(False)
            return
        samples = []
        for label, lo, hi, include_mt, color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            vals = sub.loc[mask, col].dropna().to_numpy(dtype=float)
            samples.append(vals)
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(abs(hash(col + label)) % (2**32)).normal(
                0, 0.06, len(vals)
            )
            ax.scatter(
                np.full(len(vals), x_pos[label]) + jitter,
                vals,
                c=color,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            ax.errorbar(
                [x_pos[label]],
                [np.mean(vals)],
                yerr=[stats.sem(vals) if len(vals) > 1 else 0.0],
                fmt="o",
                color="crimson",
                markersize=6,
                capsize=3,
                zorder=3,
            )
            ax.plot(
                [x_pos[label]],
                [np.median(vals)],
                marker="D",
                color="royalblue",
                markersize=5,
                zorder=4,
            )

        nonempty = [s for s in samples if len(s) > 0]
        if len(nonempty) >= 2:
            try:
                H, p = stats.kruskal(*nonempty)
                dunn = _dunn_posthoc(samples, group_labels)
                dunn_bits = [
                    f"{_short_age_pair(a, b)} {format_p(pv)}" for a, b, pv in dunn
                ]
                stat_txt = f"KW {format_p(p)}"
                if dunn_bits:
                    stat_txt += "\nDunn: " + "; ".join(dunn_bits)
                print(f"  {col}: KW H={H:.3f}, {format_p(p)}; Dunn: {dunn_bits}")
            except Exception as exc:
                stat_txt = "KW n/a"
                print(f"  {col}: KW n/a ({exc})")
        else:
            stat_txt = "KW n/a"
            print(f"  {col}: KW n/a")

        ax.set_xticks([x_pos[l] for l in group_labels])
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.axhline(1 / 3, color="0.6", ls=":", lw=0.8, zorder=0)
        ax.text(
            0.5,
            1.02,
            stat_txt,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
        _style_pub_ax(ax)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8), constrained_layout=True)
        print("\nAI by age group — Kruskal–Wallis + Dunn (Bonferroni):")

        # top: Gj, CC; bottom: Rin, Cm, Vm
        _panel(axes[0, 0], *row1[0])
        _panel(axes[0, 1], *row1[1])
        axes[0, 2].set_visible(False)
        for ax, (col, ylabel) in zip(axes[1], row2):
            _panel(ax, col, ylabel)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="crimson",
                linestyle="None",
                markersize=6,
                label="mean ± SEM",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="royalblue",
                linestyle="None",
                markersize=5,
                label="median",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            "AI (asymmetry index) by age group\nKruskal–Wallis; Dunn post-hoc (Bonferroni)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "02b_AI_by_age_group.png")


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


def _asym_axis_limits(x: np.ndarray, y: np.ndarray, name: str) -> tuple[float, float]:
    """Axis limits for max-vs-min panels; Vm is negative and must not start at 0."""
    vals = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-80.0, -40.0) if name == "Vm" else (0.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if name == "Vm":
        span = max(hi - lo, 1.0)
        pad = max(0.1 * span, 2.0)
        lim_min = lo - pad
        lim_max = hi + pad
        # Vm is more negative than -48: keep a little room toward zero, but not to 0
        lim_max = min(lim_max, -45.0)
        if lim_max <= lim_min:
            lim_max = lim_min + 5.0
        return lim_min, lim_max
    # positive parameters
    lim_min = 0.0
    lim_max = hi * 1.08 if hi > 0 else 1.0
    return lim_min, lim_max


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
        lim_min, lim_max = _asym_axis_limits(
            d[xmin_c].to_numpy(), d[xmax_c].to_numpy(), name=name
        )
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            color="0.35",
            ls="--",
            lw=0.9,
            zorder=1,
        )
        if with_fit:
            k = _fit_slope_through_origin(d[xmin_c].to_numpy(), d[xmax_c].to_numpy())
            if k is not None:
                ax.plot(
                    [lim_min, lim_max],
                    [k * lim_min, k * lim_max],
                    color="crimson",
                    lw=1.2,
                    zorder=1,
                )
                ax.text(
                    0.04,
                    0.96,
                    f"k = {k:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="0.25",
                )
        sym = PARAM_MATH.get(name, name)
        unit = f" [{unit_note}]" if unit_note else ""
        ax.set_xlabel(rf"{sym} min{unit}")
        ax.set_ylabel(rf"{sym} max{unit}")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal", adjustable="box")
        _style_pub_ax(ax)
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
        fig.suptitle("Coupling asymmetry", fontsize=11, color="0.15")
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
            _panel(ax, name, with_fit=True, unit_note=units[name])
        _finish(fig2, axes2, "04_asymmetry_Rin_Cm_Vm.png")


# ---------------------------------------------------------------------------
# Figure 3b: asymmetry by age bins (CC top, Gj bottom; columns = age groups)
# ---------------------------------------------------------------------------
def plot_asymmetry_by_age_bins(
    df: pd.DataFrame,
    out_dir: Path,
    rows: list[tuple[str, str]],
    outfile: str,
    title: str,
):
    """
    rows x 3 age bins. Fixed colors per column, no age colorbar.
    rows: list of (param_name, unit_note)
    """
    sub = df[df["in_pool"]].copy()
    n_rows = len(rows)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig_h = 3.5 * n_rows + 0.8
        fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, fig_h), constrained_layout=True)
        if n_rows == 1:
            axes = np.array([axes])

        n_by_col = []
        for _label, lo, hi, include_mt, _color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            n_by_col.append(int(mask.sum()))

        for r, (name, unit_note) in enumerate(rows):
            xmin_c, xmax_c = f"{name}_min", f"{name}_max"
            if xmin_c not in sub.columns:
                for c in range(3):
                    axes[r, c].set_visible(False)
                continue

            sym = PARAM_MATH.get(name, name)
            unit = f" [{unit_note}]" if unit_note else ""

            # Shared X/Y range across age-group columns (same row)
            all_x, all_y = [], []
            for _label, lo, hi, include_mt, _color in AGE_ASYM_BINS:
                mask = _age_bin_mask(sub, lo, hi, include_mt)
                d_all = sub.loc[mask & sub[xmin_c].notna() & sub[xmax_c].notna()]
                if len(d_all):
                    all_x.append(d_all[xmin_c].to_numpy(dtype=float))
                    all_y.append(d_all[xmax_c].to_numpy(dtype=float))
            if all_x:
                lim_min, lim_max = _asym_axis_limits(
                    np.concatenate(all_x), np.concatenate(all_y), name=name
                )
            else:
                lim_min, lim_max = _asym_axis_limits(
                    np.array([]), np.array([]), name=name
                )

            for c, (label, lo, hi, include_mt, color) in enumerate(AGE_ASYM_BINS):
                ax = axes[r, c]
                mask = _age_bin_mask(sub, lo, hi, include_mt)
                d = sub.loc[mask & sub[xmin_c].notna() & sub[xmax_c].notna()]

                ax.plot(
                    [lim_min, lim_max],
                    [lim_min, lim_max],
                    color="0.35",
                    ls="--",
                    lw=0.9,
                    zorder=1,
                )
                if len(d):
                    ax.scatter(
                        d[xmin_c],
                        d[xmax_c],
                        c=color,
                        s=POINT_SIZE,
                        alpha=POINT_ALPHA,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=2,
                        clip_on=False,
                    )
                    k = _fit_slope_through_origin(
                        d[xmin_c].to_numpy(), d[xmax_c].to_numpy()
                    )
                    if k is not None:
                        ax.plot(
                            [lim_min, lim_max],
                            [k * lim_min, k * lim_max],
                            color="crimson",
                            lw=1.2,
                            zorder=1,
                        )
                        ax.text(
                            0.04,
                            0.96,
                            f"k = {k:.2f}",
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=8,
                            color="0.25",
                        )

                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
                ax.set_aspect("equal", adjustable="box")
                _style_pub_ax(ax)
                if r == 0:
                    ax.set_title(
                        f"{label}\n$n_{{ec}}$={n_by_col[c]}",
                        fontsize=10,
                        color="0.2",
                    )
                else:
                    ax.set_title("")
                if c == 0:
                    ax.set_ylabel(rf"{sym} max{unit}")
                else:
                    ax.set_ylabel("")
                ax.set_xlabel(rf"{sym} min{unit}")

        fig.suptitle(title, fontsize=11, color="0.15")
        savefig(fig, out_dir, outfile)


# ---------------------------------------------------------------------------
# Coupling vs inter-soma Distance
# ---------------------------------------------------------------------------
def plot_coupling_vs_distance(df: pd.DataFrame, out_dir: Path):
    """
    Exploratory: pair-mean Gj, pair-mean CC, and AI_Gj vs Distance (µm).
    Continuous age colormap; Spearman ρ annotated.
    """
    if "Distance_um" not in df.columns:
        print("Skipping distance plots: no Distance column in Excel.")
        return

    sub = df[df["in_pool"] & df["Distance_um"].notna()].copy()
    if len(sub) == 0:
        print("Skipping distance plots: no Distance values in pool days 1–2.")
        return

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        ("Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]"),
        ("CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean"),
        ("AI_Gj", r"AI $G_{j}$"),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
        print("\nCoupling vs Distance (Spearman):")
        for ax, (col, ylabel) in zip(axes, panels):
            if col not in sub.columns:
                ax.set_visible(False)
                continue
            d = sub[sub[col].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            ax.scatter(
                d["Distance_um"],
                d[col],
                c=colors,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            if len(d) >= 3:
                rho, p = stats.spearmanr(d["Distance_um"], d[col])
                ax.text(
                    0.04,
                    0.96,
                    f"Spearman ρ={rho:.2f}\n{format_p(p)}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="0.25",
                )
                print(f"  {col}: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
            else:
                print(f"  {col}: n={len(d)} (too few for Spearman)")
            if col == "AI_Gj":
                ax.axhline(1 / 3, color="0.6", ls=":", lw=0.8, zorder=0)
                ax.set_ylim(0, 1.05)
            ax.set_xlabel(r"Distance [$\mu$m]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)

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
        fig.suptitle(
            r"Coupling vs inter-soma distance"
            "\n"
            r"pair mean $=(x_1+x_2)/2$; pool days 1–2",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "06_coupling_vs_distance.png")


# ---------------------------------------------------------------------------
# Pair-mean Gj / CC by age group (KW + Dunn)
# ---------------------------------------------------------------------------
def plot_coupling_by_age_groups(df: pd.DataFrame, out_dir: Path):
    """Pair-mean Gj and CC across P10–32 / P33–50 / P50+; Kruskal–Wallis + Dunn."""
    sub = df[df["in_pool"]].copy()
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    x_pos = {lab: i + 1 for i, lab in enumerate(group_labels)}
    panels = [
        ("Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]"),
        ("CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean"),
    ]

    def _panel(ax, col: str, ylabel: str):
        if col not in sub.columns:
            ax.set_visible(False)
            return
        samples = []
        for label, lo, hi, include_mt, color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            vals = sub.loc[mask, col].dropna().to_numpy(dtype=float)
            samples.append(vals)
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(abs(hash(col + label)) % (2**32)).normal(
                0, 0.06, len(vals)
            )
            ax.scatter(
                np.full(len(vals), x_pos[label]) + jitter,
                vals,
                c=color,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            ax.errorbar(
                [x_pos[label]],
                [np.mean(vals)],
                yerr=[stats.sem(vals) if len(vals) > 1 else 0.0],
                fmt="o",
                color="crimson",
                markersize=6,
                capsize=3,
                zorder=3,
            )
            ax.plot(
                [x_pos[label]],
                [np.median(vals)],
                marker="D",
                color="royalblue",
                markersize=5,
                zorder=4,
            )

        nonempty = [s for s in samples if len(s) > 0]
        if len(nonempty) >= 2:
            try:
                H, p = stats.kruskal(*nonempty)
                dunn = _dunn_posthoc(samples, group_labels)
                dunn_bits = [
                    f"{_short_age_pair(a, b)} {format_p(pv)}" for a, b, pv in dunn
                ]
                stat_txt = f"KW {format_p(p)}"
                if dunn_bits:
                    stat_txt += "\nDunn: " + "; ".join(dunn_bits)
                print(f"  {col}: KW H={H:.3f}, {format_p(p)}; Dunn: {dunn_bits}")
            except Exception as exc:
                stat_txt = "KW n/a"
                print(f"  {col}: KW n/a ({exc})")
        else:
            stat_txt = "KW n/a"
            print(f"  {col}: KW n/a")

        ax.set_xticks([x_pos[l] for l in group_labels])
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.text(
            0.5,
            1.02,
            stat_txt,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )
        _style_pub_ax(ax)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), constrained_layout=True)
        print("\nPair-mean Gj / CC by age group — Kruskal–Wallis + Dunn (Bonferroni):")
        for ax, (col, ylabel) in zip(axes, panels):
            _panel(ax, col, ylabel)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="crimson",
                linestyle="None",
                markersize=6,
                label="mean ± SEM",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="royalblue",
                linestyle="None",
                markersize=5,
                label="median",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            r"Pair-mean $G_{j}$ and $CC$ by age group"
            "\n"
            r"pair mean $=(x_1+x_2)/2$; Kruskal–Wallis; Dunn post-hoc (Bonferroni)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "07_Gj_CC_by_age_group.png")


# ---------------------------------------------------------------------------
# Figure 5: AI histograms
# ---------------------------------------------------------------------------
def plot_ai_histograms(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    # row1: coupling; row2: cell properties (Rin, Cm, Vm together)
    row1 = [("AI_Gj", r"AI $G_{j}$"), ("AI_CC", r"AI $CC$")]
    row2 = [
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_{m}$"),
        ("AI_Vm", r"AI $|V_{m}|$"),
    ]
    hist_color = "0.55"  # gray (no age encoding on these plots)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)

        def _hist_panel(ax, col: str, title: str):
            if col not in sub.columns:
                ax.set_visible(False)
                return
            vals = sub[col].dropna().to_numpy()
            ax.hist(
                vals,
                bins=np.linspace(0, 1, 16),
                color=hist_color,
                edgecolor="white",
                alpha=0.9,
            )
            if len(vals):
                m = float(np.mean(vals))
                med = float(np.median(vals))
                ax.axvline(m, color="crimson", lw=1.3, label="mean")
                ax.axvline(med, color="darkorange", ls="--", lw=1.2, label="median")
                ax.axvline(1 / 3, color="0.45", ls=":", lw=0.9, label=r"AI$\approx$0.33 (~2×)")
                ax.text(
                    0.98,
                    0.96,
                    f"mean = {m:.2f}\nmedian = {med:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="0.25",
                )
            ax.set_xlim(0, 1)
            ax.set_xlabel(title)
            ax.set_ylabel("Count")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            _style_pub_ax(ax)

        # top row: Gj, CC, empty
        _hist_panel(axes[0, 0], *row1[0])
        _hist_panel(axes[0, 1], *row1[1])
        axes[0, 2].set_visible(False)

        # bottom row: Rin, Cm, Vm
        for ax, (col, title) in zip(axes[1], row2):
            _hist_panel(ax, col, title)

        handles = [
            Line2D([0], [0], color="crimson", lw=1.3, label="mean"),
            Line2D([0], [0], color="darkorange", ls="--", lw=1.2, label="median"),
            Line2D([0], [0], color="0.45", ls=":", lw=0.9, label=r"AI$\approx$0.33 (~2×)"),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(
            f"AI (asymmetry index)\n{AI_FORMULA}; dotted reference: AI≈0.33 (~2×)",
            fontsize=11,
            color="0.15",
        )
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
    plot_ai_by_age_groups(df, out_dir)
    plot_asymmetry_clouds(df, out_dir)
    plot_asymmetry_by_age_bins(
        df,
        out_dir,
        rows=[("CC", ""), ("Gj", "nS")],
        outfile="03b_asymmetry_CC_Gj_by_age.png",
        title="Coupling asymmetry",
    )
    plot_asymmetry_by_age_bins(
        df,
        out_dir,
        rows=[("Rin", r"M$\Omega$"), ("Cm", "pF"), ("Vm", "mV")],
        outfile="04b_asymmetry_Rin_Cm_Vm_by_age.png",
        title="Within-pair asymmetry",
    )
    plot_ai_histograms(df, out_dir)
    plot_coupling_vs_distance(df, out_dir)
    plot_coupling_by_age_groups(df, out_dir)
    print("Done. Figures are in:", out_dir)


if __name__ == "__main__":
    main()
