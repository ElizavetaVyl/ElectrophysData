"""
General properties plotting for mPFC electrical synapse pairs.

PyCharm: Run this file. It asks for the data folder, then shows a window
to choose which figures to (re)generate. Existing files are overwritten.

Expected Excel name (in that folder):
  General_Properties_FOR_PLOTTING_ONLY.xlsx

Figures are saved to:
  <chosen_folder>/general properties figures/

CLI tips:
  python plot_general_properties.py --folder PATH --all-figures
  python plot_general_properties.py --folder PATH --figures 01,07,13
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
MT_AGE_PLACEHOLDER = 400  # x-position for MT/Dam on age plots (days)
P50_ZOOM_MAX = 50
DAY_PAIR_OFFSET = 0.14  # left/right shift within a day (wider for visibility)
DAY_X = {1: 1.0, 2: 1.55, 3: 2.1}  # days closer together
POINT_SIZE = 36
POINT_ALPHA = 0.55
MT_AGE_OFFSET = 50  # MT darker than max age by this many "days" on color scale
# Excel age column uses German Muttertier token; figures show English label
MT_EXCEL_TOKEN = "MT"
MT_DISPLAY_LABEL = "Dam"  # breeding female, postnatal age not recorded
STRETCH_AGE_HI = 50  # ages up to this use most of the colorbar (linear ruler)
STRETCH_FRAC = 0.75  # fraction of colorbar reserved for <= STRETCH_AGE_HI
MAJOR_AGE_TARGETS = (20, 30, 40, 50)
# saturated orange (young) -> almost-black purple (old / Dam)
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
    # Excel: MT (Muttertier); optionally Dam if already relabelled
    if text.upper() in {MT_EXCEL_TOKEN.upper(), MT_DISPLAY_LABEL.upper()}:
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


def major_labeled_ages(ages: list[float], min_gap: float = 8.0) -> list[float]:
    """
    Labels: minimum age, nearest available to 20/30/40/50 (always keep
    the age nearest to P50, e.g. 49), then sparsely ages >50.
    Avoid crowding (49+51+53) that makes the ~P50 mark unreadable.
    Numeric maximum is not labeled (MT is labeled separately on the colorbar).
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
    # Always keep the observed age closest to the P50 stretch boundary
    nearest_p50 = min(ages, key=lambda a: abs(a - float(STRETCH_AGE_HI)))
    if nearest_p50 not in out:
        out.append(nearest_p50)
    # Older ages: only if far enough from already chosen majors
    for a in ages:
        if a <= float(STRETCH_AGE_HI) or a >= max_age:
            continue
        if all(abs(a - o) >= min_gap for o in out):
            out.append(a)
    return sorted(out)


def binom_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided binomial p; works on SciPy <1.7 (binom_test) and >=1.7 (binomtest)."""
    if n <= 0:
        return float("nan")
    if hasattr(stats, "binomtest"):
        return float(stats.binomtest(k, n, p).pvalue)
    # SciPy < 1.7
    return float(stats.binom_test(k, n, p))


def lowess_smooth(
    x: np.ndarray, y: np.ndarray, frac: float = 0.45, n_grid: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight LOWESS (local weighted linear) without statsmodels.
    Returns (x_grid, y_hat) sorted along x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        order = np.argsort(x)
        return x[order], y[order]
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    k = max(int(np.ceil(frac * n)), 5)
    k = min(k, n)
    x_grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    y_hat = np.empty_like(x_grid)
    for i, x0 in enumerate(x_grid):
        dist = np.abs(x - x0)
        idx = np.argpartition(dist, k - 1)[:k]
        d = dist[idx]
        dmax = float(np.max(d)) if len(d) else 0.0
        if dmax <= 0:
            y_hat[i] = float(np.mean(y[idx]))
            continue
        w = (1.0 - (d / dmax) ** 3) ** 3  # tricube
        X = np.column_stack([np.ones(k), x[idx] - x0])
        WX = X * w[:, None]
        beta, _, _, _ = np.linalg.lstsq(WX, y[idx] * w, rcond=None)
        y_hat[i] = float(beta[0])
    return x_grid, y_hat


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
        major_labels.append(MT_DISPLAY_LABEL)
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
    existed = path.is_file()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{'Overwrote' if existed else 'Saved'}: {path}")


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


def p_to_stars(p: float | None) -> str:
    """Significance stars; 'ns' if not significant / missing."""
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def bonferroni_adjust(p: float | None, n_tests: int) -> float | None:
    if p is None or not np.isfinite(p) or n_tests < 1:
        return None
    return float(min(1.0, p * n_tests))


def _sig_bracket(
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
        tick_lab.append(MT_DISPLAY_LABEL)
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
    Exploratory: pair-mean Gj/CC and AI_Gj/AI_CC vs Distance (µm).
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
    # top: means; bottom: asymmetry indices
    row1 = [
        ("Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]"),
        ("CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean"),
    ]
    row2 = [
        ("AI_Gj", r"AI $G_{j}$"),
        ("AI_CC", r"AI $CC$"),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
        print("\nCoupling vs Distance (Spearman):")

        def _dist_panel(ax, col: str, ylabel: str):
            if col not in sub.columns:
                ax.set_visible(False)
                return
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
            if col.startswith("AI_"):
                ax.axhline(1 / 3, color="0.6", ls=":", lw=0.8, zorder=0)
                ax.set_ylim(0, 1.05)
            ax.set_xlabel(r"Distance [$\mu$m]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)

        for ax, (col, ylabel) in zip(axes[0], row1):
            _dist_panel(ax, col, ylabel)
        for ax, (col, ylabel) in zip(axes[1], row2):
            _dist_panel(ax, col, ylabel)

        cbar = add_age_colorbar(
            fig,
            list(axes.ravel()),
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
# Gj / CC by age group: pooled sides + pair mean (KW + Dunn)
# ---------------------------------------------------------------------------
def plot_coupling_by_age_groups(df: pd.DataFrame, out_dir: Path):
    """
    2×2: top = pooled {x1, x2} for Gj and CC; bottom = pair means.
    Age groups P10–32 / P33–50 / P50+; Kruskal–Wallis + Dunn (Bonferroni).
    """
    sub = df[df["in_pool"]].copy()
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    x_pos = {lab: i + 1 for i, lab in enumerate(group_labels)}

    # (mode, col_or_pair, ylabel) — mode "pair" uses one column; "pool" uses (c1, c2)
    row_pooled = [
        ("pool", ("Gj_12", "Gj_21"), rf"{PARAM_MATH['Gj']} pooled [nS]"),
        ("pool", ("CC_12", "CC_21"), rf"{PARAM_MATH['CC']} pooled"),
    ]
    row_mean = [
        ("pair", "Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]"),
        ("pair", "CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean"),
    ]

    def _samples_for(mode: str, spec) -> list[np.ndarray]:
        samples = []
        for _label, lo, hi, include_mt, _color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            g = sub.loc[mask]
            if mode == "pair":
                vals = g[spec].dropna().to_numpy(dtype=float) if spec in g.columns else np.array([])
            else:
                c1, c2 = spec
                if c1 not in g.columns or c2 not in g.columns:
                    vals = np.array([])
                else:
                    vals = pd.concat([g[c1], g[c2]], ignore_index=True).dropna().to_numpy(
                        dtype=float
                    )
            samples.append(vals)
        return samples

    def _panel(ax, mode: str, spec, ylabel: str, seed_key: str):
        samples = _samples_for(mode, spec)
        for (label, _lo, _hi, _mt, color), vals in zip(AGE_ASYM_BINS, samples):
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(abs(hash(seed_key + label)) % (2**32)).normal(
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
                n_bits = ", ".join(
                    f"{lab} n={len(s)}" for lab, s in zip(group_labels, samples)
                )
                print(
                    f"  {seed_key}: KW H={H:.3f}, {format_p(p)}; "
                    f"Dunn: {dunn_bits}; {n_bits}"
                )
            except Exception as exc:
                stat_txt = "KW n/a"
                print(f"  {seed_key}: KW n/a ({exc})")
        else:
            stat_txt = "KW n/a"
            print(f"  {seed_key}: KW n/a")

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
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), constrained_layout=True)
        print(
            "\nGj / CC by age group — pooled sides & pair mean — "
            "Kruskal–Wallis + Dunn (Bonferroni):"
        )
        for ax, (mode, spec, ylabel) in zip(axes[0], row_pooled):
            key = f"pooled_{spec[0]}"
            _panel(ax, mode, spec, ylabel, key)
        for ax, (mode, spec, ylabel) in zip(axes[1], row_mean):
            _panel(ax, mode, spec, ylabel, spec)

        axes[0, 0].set_title(r"pooled $\{x_1,x_2\}$", fontsize=10, color="0.25", pad=18)
        axes[0, 1].set_title(r"pooled $\{x_1,x_2\}$", fontsize=10, color="0.25", pad=18)
        axes[1, 0].set_title(r"pair mean $(x_1+x_2)/2$", fontsize=10, color="0.25", pad=18)
        axes[1, 1].set_title(r"pair mean $(x_1+x_2)/2$", fontsize=10, color="0.25", pad=18)

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
            r"$G_{j}$ and $CC$ by age group"
            "\n"
            r"top: pooled sides; bottom: pair mean; Kruskal–Wallis; Dunn (Bonferroni)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "07_Gj_CC_by_age_group.png")


# ---------------------------------------------------------------------------
# Pair-mean Gj / CC vs continuous (warped) age — Spearman + LOWESS
# ---------------------------------------------------------------------------
def plot_coupling_vs_age_continuous(df: pd.DataFrame, out_dir: Path):
    """
    Continuous alternative to age-group KW: pair-mean Gj and CC vs warped age.
    Annotate Spearman ρ (rank correlation, not a regression slope); overlay LOWESS.
    """
    sub = df[df["in_pool"]].copy()
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    sub = sub.copy()
    sub["x_warp"] = [
        _warped_age_x(a, bool(mt), vmin, max_age, mt_val)
        for a, mt in zip(sub["age_num"], sub["is_mt"])
    ]
    panels = [
        ("Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]"),
        ("CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean"),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        print("\nPair-mean Gj / CC vs warped age (Spearman + LOWESS):")
        for ax, (col, ylabel) in zip(axes, panels):
            if col not in sub.columns:
                ax.set_visible(False)
                continue
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
            if len(d) >= 5:
                xs, ys = lowess_smooth(
                    d["x_warp"].to_numpy(dtype=float),
                    d[col].to_numpy(dtype=float),
                    frac=0.45,
                )
                ax.plot(xs, ys, color="crimson", lw=1.6, zorder=3, label="LOWESS")
            if len(d) >= 3:
                # Spearman uses chronological age ranks (MT at top), not warp units
                age_for_rank = np.where(
                    d["is_mt"].to_numpy(),
                    mt_val,
                    d["age_num"].to_numpy(dtype=float),
                )
                rho, p = stats.spearmanr(age_for_rank, d[col].to_numpy(dtype=float))
                ax.text(
                    0.04,
                    0.96,
                    f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={len(d)}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="0.25",
                )
                print(f"  {col}: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
            else:
                print(f"  {col}: n={len(d)} (too few)")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Age [postnatal days]")
            _set_warped_age_ticks(ax, vmin, max_age, mt_val, age_ticks, has_mt)
            _style_pub_ax(ax)

        handles = [
            Line2D([0], [0], color="crimson", lw=1.6, label="LOWESS"),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
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
            r"Pair-mean $G_{j}$ and $CC$ vs age (continuous)"
            "\n"
            r"Spearman ρ = rank correlation (not a fit slope); crimson: LOWESS; pool days 1–2",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "10_Gj_CC_vs_age_continuous.png")


# ---------------------------------------------------------------------------
# Coverage: Distance + age histograms on one figure
# ---------------------------------------------------------------------------
def plot_coverage_histograms(df: pd.DataFrame, out_dir: Path):
    """
    One figure, three panels: Distance bins | age (5-day) | age groups.
    Pool days 1–2; one count per electrically coupled pair.
    """
    sub = df[df["in_pool"]].copy()
    if len(sub) == 0:
        print("Skipping coverage histograms: empty pool.")
        return

    numeric = sub.loc[~sub["is_mt"], "age_num"].dropna().to_numpy(dtype=float)
    n_mt = int(sub["is_mt"].sum())
    has_dist = "Distance_um" in sub.columns and sub["Distance_um"].notna().any()
    dists = (
        sub.loc[sub["Distance_um"].notna(), "Distance_um"].to_numpy(dtype=float)
        if has_dist
        else np.array([])
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
        ax_d, ax_a, ax_g = axes

        # --- Distance ---
        if len(dists):
            d_min = float(np.floor(np.min(dists) / 10.0) * 10.0)
            d_max = float(np.ceil(np.max(dists) / 10.0) * 10.0)
            if d_max <= d_min:
                d_max = d_min + 10.0
            width = 15.0
            edges = np.arange(d_min, d_max + width, width)
            if len(edges) < 7:
                width = 10.0
                edges = np.arange(d_min, d_max + width, width)
            ax_d.hist(dists, bins=edges, color="0.55", edgecolor="white", alpha=0.9)
            ax_d.axvline(
                float(np.median(dists)),
                color="darkorange",
                ls="--",
                lw=1.3,
                label=f"median = {np.median(dists):.0f} µm",
            )
            ax_d.axvline(
                float(np.mean(dists)),
                color="crimson",
                lw=1.3,
                label=f"mean = {np.mean(dists):.0f} µm",
            )
            ax_d.text(
                0.98,
                0.96,
                f"$n$ = {len(dists)}\n"
                f"{np.min(dists):.0f}–{np.max(dists):.0f} µm",
                transform=ax_d.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.25",
            )
            ax_d.legend(frameon=False, fontsize=8, loc="upper left")
            print(
                f"\nCoverage histograms — Distance: n={len(dists)}, "
                f"mean={np.mean(dists):.1f}, median={np.median(dists):.1f} µm"
            )
        else:
            ax_d.text(0.5, 0.5, "no Distance", ha="center", va="center", color="0.4")
            print("\nCoverage histograms — Distance: skipped (no column/values)")
        ax_d.set_xlabel(r"Distance [$\mu$m]")
        ax_d.set_ylabel("Number of pairs")
        ax_d.set_title("Distance", fontsize=10, color="0.25")
        ax_d.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_pub_ax(ax_d)

        # --- Age histogram on warped scale (+ Dam after dotted separator) ---
        # ≤P50: 5-day bins on the stretched ruler. >P50: one bar per observed
        # age, same visual width as a 5-day bin (axis grows as needed); minor
        # ticks mark those ages; major labels stay sparse.
        bin_w = 5.0
        if len(numeric) or n_mt:
            if len(numeric):
                a_min = float(np.floor(np.min(numeric) / bin_w) * bin_w)
                max_age = float(np.max(numeric))
                vmin = float(np.min(numeric))
            else:
                a_min = 10.0
                vmin, max_age = 10.0, float(STRETCH_AGE_HI)
            mt_val = max_age + MT_AGE_OFFSET
            age_ticks = (
                sorted({float(a) for a in numeric}) if len(numeric) else [a_min]
            )
            old_ages = [a for a in age_ticks if a > float(STRETCH_AGE_HI)]
            n_old = len(old_ages)

            def _u_young(a: float) -> float:
                """Warped unit for ages ≤ P50 (same stretch as colorbar)."""
                a_c = float(np.clip(float(a), vmin, float(STRETCH_AGE_HI)))
                return age_to_unit(a_c, False, vmin, max_age, mt_val)

            u_tail0 = _u_young(float(STRETCH_AGE_HI))
            # Reference width = one 5-day bin on the stretched ≤P50 ruler
            span_young = max(float(STRETCH_AGE_HI) - vmin, bin_w)
            w_ref = float(STRETCH_FRAC) * bin_w / span_young
            slot = w_ref  # one slot per observed age > P50
            if n_old:
                old_centers = {
                    a: u_tail0 + (i + 0.5) * slot for i, a in enumerate(old_ages)
                }
                old_bar_w = 0.88 * slot
                u_tail1 = u_tail0 + n_old * slot
            else:
                old_centers = {}
                old_bar_w = w_ref
                u_tail1 = u_tail0

            def _u_hist(a: float) -> float:
                """Histogram X: warped ≤P50; equal-width slots for ages >P50."""
                a = float(a)
                if a <= float(STRETCH_AGE_HI):
                    return _u_young(a)
                if a in old_centers:
                    return old_centers[a]
                if old_ages:
                    nearest = min(old_ages, key=lambda x: abs(x - a))
                    return old_centers[nearest]
                return u_tail1

            # Light age-group backgrounds
            band_specs = [
                (10.0, 32.5, "#FB8C00", 0.12, "P10–32", False),
                (32.5, 50.0, "#8E24AA", 0.10, "P33–50", False),
                (50.0, max_age, "#2E0854", 0.08, "P50+", True),
            ]
            for lo_b, hi_b, col_b, alpha_b, lab_b, is_old in band_specs:
                if is_old:
                    if not old_ages and max_age <= float(STRETCH_AGE_HI):
                        continue
                    u_lo = u_tail0
                    u_hi = u_tail1 if old_ages else _u_young(max_age)
                else:
                    lo_c = max(lo_b, a_min)
                    hi_c = min(hi_b, float(STRETCH_AGE_HI), max_age)
                    if hi_c <= lo_c:
                        continue
                    u_lo, u_hi = _u_young(lo_c), _u_young(hi_c)
                if u_hi > u_lo:
                    ax_a.axvspan(
                        u_lo, u_hi, color=col_b, alpha=alpha_b, zorder=0, lw=0
                    )
                    ax_a.text(
                        0.5 * (u_lo + u_hi),
                        0.92,
                        lab_b,
                        transform=ax_a.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=7,
                        color=col_b,
                        alpha=0.85,
                    )

            if old_ages or max_age > STRETCH_AGE_HI:
                ax_a.axvline(u_tail0, color="0.8", ls="--", lw=0.7, zorder=1)

            if len(numeric):
                young = numeric[numeric <= float(STRETCH_AGE_HI)]
                edges = np.arange(a_min, float(STRETCH_AGE_HI) + bin_w * 0.5, bin_w)
                if len(edges) < 2:
                    edges = np.array([a_min, a_min + bin_w])
                if edges[-1] < float(STRETCH_AGE_HI):
                    edges = np.append(edges, float(STRETCH_AGE_HI))
                else:
                    edges[-1] = float(STRETCH_AGE_HI)
                if len(young):
                    counts_y, _ = np.histogram(young, bins=edges)
                    edges_u = np.array([_u_young(float(e)) for e in edges], dtype=float)
                    widths_u = np.maximum(np.diff(edges_u), 1e-4)
                    ax_a.bar(
                        edges_u[:-1],
                        counts_y,
                        width=widths_u,
                        align="edge",
                        color="0.55",
                        edgecolor="white",
                        alpha=0.9,
                        zorder=2,
                    )
                if old_ages:
                    old_arr = numeric[numeric > float(STRETCH_AGE_HI)]
                    old_counts = [int(np.sum(np.isclose(old_arr, a))) for a in old_ages]
                    ax_a.bar(
                        [old_centers[a] for a in old_ages],
                        old_counts,
                        width=old_bar_w,
                        align="center",
                        color="0.55",
                        edgecolor="white",
                        alpha=0.9,
                        zorder=2,
                    )

            majors = major_labeled_ages(age_ticks) or [a_min]
            # Prefer a labeled major near P50 on the young side when both 49 & 51 exist
            tick_pos = [_u_hist(a) for a in majors]
            tick_lab = [_fmt_age_tick(a) for a in majors]

            u_left = _u_young(a_min) - 0.02
            u_right = u_tail1 + 0.03
            if n_mt:
                sep_x = u_tail1 + 0.015
                dam_x = u_tail1 + 0.015 + slot
                dam_w = old_bar_w
                ax_a.axvline(sep_x, color="0.35", ls=":", lw=0.9, zorder=1)
                ax_a.axvspan(
                    dam_x - 0.5 * dam_w,
                    dam_x + 0.5 * dam_w,
                    color="#2E0854",
                    alpha=0.06,
                    zorder=0,
                    lw=0,
                )
                ax_a.bar(
                    dam_x,
                    n_mt,
                    width=dam_w,
                    color="0.35",
                    edgecolor="white",
                    alpha=0.9,
                    align="center",
                    zorder=2,
                )
                tick_pos.append(dam_x)
                tick_lab.append(MT_DISPLAY_LABEL)
                u_right = dam_x + 0.5 * slot + 0.03

            ax_a.set_xticks(tick_pos)
            ax_a.set_xticklabels(tick_lab)
            # Minor насечки: every observed age (+ Dam), same rule as age colorbar
            minor_pos = [_u_hist(a) for a in age_ticks]
            if n_mt:
                minor_pos.append(dam_x)
            if minor_pos:
                ax_a.set_xticks(sorted(set(minor_pos)), minor=True)
            ax_a.set_xlim(u_left, u_right)
        else:
            ax_a.text(0.5, 0.5, "no ages", ha="center", va="center", color="0.4")
        ax_a.set_xlabel("Age [postnatal days]")
        ax_a.set_ylabel("Number of pairs")
        ax_a.set_title("Age (5-day bins; P50+ by age)", fontsize=10, color="0.25")
        ax_a.yaxis.set_major_locator(MaxNLocator(integer=True))
        _style_pub_ax(ax_a)
        if len(numeric) or n_mt:
            # Match colorbar tick styling: short minors, longer majors
            ax_a.tick_params(
                axis="x", which="minor", length=3.0, width=0.8, colors="0.2"
            )
            ax_a.tick_params(
                axis="x", which="major", length=6.0, width=1.0, colors="0.2"
            )

        # --- Age groups (same as KW plots; MT in P50+) ---
        labels = [b[0] for b in AGE_ASYM_BINS]
        colors = [b[4] for b in AGE_ASYM_BINS]
        counts = []
        for _label, lo, hi, include_mt, _c in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            counts.append(int(mask.sum()))
        x = np.arange(len(labels))
        ax_g.bar(x, counts, color=colors, edgecolor="white", width=0.7, alpha=0.9)
        for xi, n in zip(x, counts):
            ax_g.text(
                xi, n + 0.15, str(n), ha="center", va="bottom", fontsize=9, color="0.25"
            )
        ax_g.set_xticks(x)
        ax_g.set_xticklabels(labels)
        ax_g.set_ylabel("Number of pairs")
        ax_g.set_xlabel("")
        ax_g.set_title("Age groups (KW bins)", fontsize=10, color="0.25")
        ax_g.yaxis.set_major_locator(MaxNLocator(integer=True))
        ymax = max(counts) if counts else 1
        ax_g.set_ylim(0, ymax * 1.18 + 0.5)
        _style_pub_ax(ax_g)

        fig.suptitle(
            r"Sample coverage: distance and age"
            "\n"
            r"pool days 1–2; one count per electrically coupled pair",
            fontsize=11,
            color="0.15",
        )
        print(
            f"Coverage histograms — Age: n_numeric={len(numeric)}, "
            f"n_{MT_DISPLAY_LABEL}={n_mt}; "
            + ", ".join(f"{lab}={n}" for lab, n in zip(labels, counts))
        )
        savefig(fig, out_dir, "11_distance_age_histograms.png")


# ---------------------------------------------------------------------------
# AI_Gj (and AI_CC) vs AI_Rin — leak vs junctional asymmetry
# ---------------------------------------------------------------------------
def plot_ai_gj_vs_ai_rin(df: pd.DataFrame, out_dir: Path):
    """
    Does Gj (or CC) asymmetry track Rin asymmetry?
    Scatter AI_coupling vs AI_Rin; identity line; Spearman ρ.
    """
    sub = df[df["in_pool"]].copy()
    if "AI_Rin" not in sub.columns or "AI_Gj" not in sub.columns:
        print("Skipping AI_Gj vs AI_Rin: missing AI columns.")
        return

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        ("AI_Gj", r"AI $G_{j}$"),
        ("AI_CC", r"AI $CC$"),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        print("\nAI coupling vs AI_Rin (Spearman):")
        for ax, (col, ylabel) in zip(axes, panels):
            if col not in sub.columns:
                ax.set_visible(False)
                continue
            d = sub[sub[col].notna() & sub["AI_Rin"].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=0.9, zorder=1)
            ax.scatter(
                d["AI_Rin"],
                d[col],
                c=colors,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            if len(d) >= 3:
                rho, p = stats.spearmanr(d["AI_Rin"], d[col])
                ax.text(
                    0.04,
                    0.96,
                    f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={len(d)}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="0.25",
                )
                print(f"  {col} vs AI_Rin: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
            else:
                print(f"  {col} vs AI_Rin: n={len(d)} (too few)")
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"AI $R_{in}$")
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
            r"Coupling asymmetry vs $R_{in}$ asymmetry"
            "\n"
            r"$AI=(max-min)/(max+min)$; dashed: identity; pool days 1–2",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "08_AI_Gj_CC_vs_AI_Rin.png")


# ---------------------------------------------------------------------------
# Signed ΔGj (and ΔCC) vs ΔRin of the receiver
# ---------------------------------------------------------------------------
def plot_signed_delta_gj_vs_rin(df: pd.DataFrame, out_dir: Path):
    """
    Directional sign check:
      ΔGj = Gj_12 − Gj_21
      ΔRin_recv = Rin_2 − Rin_1
    (for 1→2 the receiver is cell 2). Same-sign quadrants mean larger
    coupling toward the higher-Rin cell.
    """
    need = ["Gj_12", "Gj_21", "Rin_1", "Rin_2"]
    if any(c not in df.columns for c in need):
        print("Skipping signed ΔGj vs ΔRin: missing Gj/Rin columns.")
        return

    sub = df[df["in_pool"]].copy()
    sub["dGj"] = sub["Gj_12"] - sub["Gj_21"]
    sub["dRin_recv"] = sub["Rin_2"] - sub["Rin_1"]
    if "CC_12" in sub.columns and "CC_21" in sub.columns:
        sub["dCC"] = sub["CC_12"] - sub["CC_21"]

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        ("dGj", r"$\Delta G_{j}=G_{j,12}-G_{j,21}$ [nS]"),
        ("dCC", r"$\Delta CC=CC_{12}-CC_{21}$"),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        print("\nSigned Δ coupling vs ΔRin_recv (same-sign / Spearman):")

        for ax, (col, ylabel) in zip(axes, panels):
            if col not in sub.columns:
                ax.set_visible(False)
                continue
            d = sub[sub[col].notna() & sub["dRin_recv"].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            x = d["dRin_recv"].to_numpy(dtype=float)
            y = d[col].to_numpy(dtype=float)

            # soft same-sign quadrant tint
            xpad = max(float(np.nanmax(np.abs(x))) * 1.15, 1.0) if len(x) else 1.0
            ypad = max(float(np.nanmax(np.abs(y))) * 1.15, 0.1) if len(y) else 0.1
            ax.axhspan(0, ypad, xmin=0.5, xmax=1.0, color="0.92", zorder=0)
            ax.axhspan(-ypad, 0, xmin=0.0, xmax=0.5, color="0.92", zorder=0)
            ax.axhline(0, color="0.45", lw=0.8, zorder=1)
            ax.axvline(0, color="0.45", lw=0.8, zorder=1)
            ax.scatter(
                x,
                y,
                c=colors,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )

            same = (x * y) > 0
            nonzero = (x != 0) & (y != 0)
            n_nz = int(nonzero.sum())
            n_same = int((same & nonzero).sum())
            frac = n_same / n_nz if n_nz else np.nan
            # two-sided binomial vs 50%
            if n_nz >= 5:
                binom_p = binom_pvalue(n_same, n_nz, 0.5)
            else:
                binom_p = np.nan

            if len(d) >= 3:
                rho, sp = stats.spearmanr(x, y)
            else:
                rho, sp = np.nan, np.nan

            stat_lines = [
                f"same sign {n_same}/{n_nz}"
                + (f" ({100 * frac:.0f}%)" if n_nz else ""),
                f"binomial {format_p(binom_p)}",
                f"Spearman ρ={rho:.2f}" if np.isfinite(rho) else "Spearman n/a",
                format_p(sp) if np.isfinite(sp) else "",
            ]
            ax.text(
                0.04,
                0.96,
                "\n".join(s for s in stat_lines if s),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="0.25",
            )
            print(
                f"  {col}: same-sign {n_same}/{n_nz} ({frac:.2f}), "
                f"binomial {format_p(binom_p)}; "
                f"Spearman ρ={rho:.3f}, {format_p(sp)}"
            )

            ax.set_xlim(-xpad, xpad)
            ax.set_ylim(-ypad, ypad)
            ax.set_xlabel(r"$\Delta R_{in,recv}=R_{in,2}-R_{in,1}$ [M$\Omega$]")
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
            r"Signed coupling difference vs receiver $\Delta R_{in}$"
            "\n"
            r"shaded: same sign (larger coupling toward higher-$R_{in}$ cell); pool days 1–2",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "09_signed_dGj_dCC_vs_dRin.png")


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


# ---------------------------------------------------------------------------
# Day 1 vs Day 2 × age groups (MW + Bonferroni; KW + Dunn) with stars
# Modes: pair mean AND pooled {x1, x2} as independent measurements
# ---------------------------------------------------------------------------
DAY_AGE_DAY_MARKERS = {1: "o", 2: "s"}  # Day 1 circle, Day 2 square
DAY_AGE_OFFSET = 0.18
MEAN_COLOR = "0.15"  # black-ish mean ± SEM


def _vals_day_age(
    df: pd.DataFrame,
    *,
    mode: str,
    c1: str,
    c2: str,
    mean_col: str,
    day: int,
    lo,
    hi,
    include_mt: bool,
) -> np.ndarray:
    """mode='pair' → pair mean; mode='pool' → concatenate both sides."""
    mask = (df["day"] == day) & _age_bin_mask(df, lo, hi, include_mt)
    g = df.loc[mask]
    if mode == "pair":
        if mean_col not in g.columns:
            return np.array([], dtype=float)
        return g[mean_col].dropna().to_numpy(dtype=float)
    # pooled sides as independent measurements
    if c1 not in g.columns or c2 not in g.columns:
        return np.array([], dtype=float)
    return pd.concat([g[c1], g[c2]], ignore_index=True).dropna().to_numpy(dtype=float)


def _empty_compare(n1: int = 0, n2: int = 0) -> dict:
    return {
        "n1": n1,
        "n2": n2,
        "p": np.nan,
        "test": "n/a",
        "shapiro_p1": None,
        "shapiro_p2": None,
        "mean1": np.nan,
        "mean2": np.nan,
        "sem1": np.nan,
        "sem2": np.nan,
        "median1": np.nan,
        "median2": np.nan,
    }


def _run_day_by_age_mode(
    sub: pd.DataFrame,
    out_dir: Path,
    *,
    mode: str,
    outfile_overview: str,
    outfile_stats: str,
    title_mode: str,
):
    """
    mode: 'pair' | 'pool'
    Day within age: MW/Welch + Bonferroni×3 ages.
    Age within day: KW + Dunn; significant brackets on overview.
    """
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    n_age = len(AGE_ASYM_BINS)
    params = list(STAT_PARAMS)

    day_vs: dict[str, list[dict]] = {}
    age_kw: dict[str, dict[int, dict]] = {}

    print(f"\nDay 1 vs Day 2 × age groups ({title_mode}):")
    for name, c1, c2, mean_col in params:
        if mode == "pair" and mean_col not in sub.columns:
            continue
        if mode == "pool" and (c1 not in sub.columns or c2 not in sub.columns):
            continue

        day_results = []
        print(f"  [{name}] Day1 vs Day2 within age (Bonferroni ×{n_age}):")
        for label, lo, hi, include_mt, _color in AGE_ASYM_BINS:
            v1 = _vals_day_age(
                sub,
                mode=mode,
                c1=c1,
                c2=c2,
                mean_col=mean_col,
                day=1,
                lo=lo,
                hi=hi,
                include_mt=include_mt,
            )
            v2 = _vals_day_age(
                sub,
                mode=mode,
                c1=c1,
                c2=c2,
                mean_col=mean_col,
                day=2,
                lo=lo,
                hi=hi,
                include_mt=include_mt,
            )
            res = (
                compare_two_groups(v1, v2)
                if len(v1) >= 1 and len(v2) >= 1
                else _empty_compare(len(v1), len(v2))
            )
            res["p_adj"] = bonferroni_adjust(res.get("p"), n_age)
            res["age"] = label
            day_results.append(res)
            print(
                f"    {label}: n={res['n1']}/{res['n2']}, {res['test']} "
                f"{format_p(res.get('p'))} → adj {format_p(res['p_adj'])} "
                f"{p_to_stars(res['p_adj'])}"
            )
        day_vs[name] = day_results

        age_kw[name] = {}
        for day in (1, 2):
            samples = [
                _vals_day_age(
                    sub,
                    mode=mode,
                    c1=c1,
                    c2=c2,
                    mean_col=mean_col,
                    day=day,
                    lo=lo,
                    hi=hi,
                    include_mt=include_mt,
                )
                for _lab, lo, hi, include_mt, _c in AGE_ASYM_BINS
            ]
            nonempty = [s for s in samples if len(s) > 0]
            if len(nonempty) >= 2:
                try:
                    H, p = stats.kruskal(*nonempty)
                    dunn = _dunn_posthoc(samples, group_labels)
                except Exception as exc:
                    H, p, dunn = np.nan, np.nan, []
                    print(f"    Day {day} KW n/a ({exc})")
            else:
                H, p, dunn = np.nan, np.nan, []
            age_kw[name][day] = {"H": H, "p": p, "dunn": dunn, "samples": samples}
            dunn_bits = [
                f"{_short_age_pair(a, b)} {format_p(pv)}" for a, b, pv in dunn
            ]
            print(f"    Age KW Day{day}: {format_p(p)}; Dunn: {dunn_bits}")

    # Legend: day = marker shape; age = orange→purple on x; mean = black
    day_handles = [
        Line2D(
            [0],
            [0],
            marker=DAY_AGE_DAY_MARKERS[1],
            color="w",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=7,
            label="Day 1",
        ),
        Line2D(
            [0],
            [0],
            marker=DAY_AGE_DAY_MARKERS[2],
            color="w",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=7,
            label="Day 2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=MEAN_COLOR,
            linestyle="None",
            markersize=6,
            label="mean ± SEM",
        ),
    ]
    age_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=7,
            label=label,
        )
        for label, _lo, _hi, _mt, color in AGE_ASYM_BINS
    ]

    # ----- Overview with brackets / stars -----
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
        axes[0, 2].set_visible(False)

        for ax, (name, c1, c2, mean_col) in zip(axes_flat, params):
            if name not in day_vs:
                ax.set_visible(False)
                continue

            all_y = []
            x_day = {1: {}, 2: {}}
            for i, (label, lo, hi, include_mt, age_color) in enumerate(AGE_ASYM_BINS):
                xc = i + 1
                for day, sign in ((1, -1), (2, +1)):
                    x = xc + sign * DAY_AGE_OFFSET
                    x_day[day][label] = x
                    vals = _vals_day_age(
                        sub,
                        mode=mode,
                        c1=c1,
                        c2=c2,
                        mean_col=mean_col,
                        day=day,
                        lo=lo,
                        hi=hi,
                        include_mt=include_mt,
                    )
                    if len(vals) == 0:
                        continue
                    all_y.append(vals)
                    jitter = np.random.default_rng(
                        abs(hash(f"{mode}{name}{day}{label}")) % (2**32)
                    ).normal(0, 0.04, len(vals))
                    ax.scatter(
                        np.full(len(vals), x) + jitter,
                        vals,
                        c=age_color,
                        marker=DAY_AGE_DAY_MARKERS[day],
                        s=POINT_SIZE,
                        alpha=POINT_ALPHA,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=2,
                    )
                    m, sem, _med = _mean_sem_median(vals)
                    ax.errorbar(
                        [x],
                        [m],
                        yerr=[sem],
                        fmt="o",
                        color=MEAN_COLOR,
                        ecolor=MEAN_COLOR,
                        markersize=5,
                        capsize=3,
                        elinewidth=1.1,
                        zorder=3,
                    )

            if all_y:
                y_all = np.concatenate(all_y)
                y_lo, y_hi = float(np.min(y_all)), float(np.max(y_all))
                span = max(y_hi - y_lo, abs(y_hi) * 0.05, 1e-6)
                ax.set_ylim(y_lo - 0.08 * span, y_hi + 0.55 * span)
            else:
                y_lo, y_hi, span = 0.0, 1.0, 1.0

            # Day1 vs Day2 within age: stars only (no p-text on plot)
            bracket_y = y_hi + 0.08 * span
            step = 0.12 * span
            for res in day_vs[name]:
                stars = p_to_stars(res["p_adj"])
                if stars == "ns":
                    continue
                lab = res["age"]
                x1, x2 = x_day[1][lab], x_day[2][lab]
                _sig_bracket(ax, x1, x2, bracket_y, stars, h=0.035 * span)
                bracket_y += step

            # Dunn age pairs within a day: label "day 1" / "day 2" + stars
            for day in (1, 2):
                info = age_kw.get(name, {}).get(day, {})
                for a, b, pv in info.get("dunn") or []:
                    stars = p_to_stars(pv)
                    if stars == "ns":
                        continue
                    if a not in x_day[day] or b not in x_day[day]:
                        continue
                    _sig_bracket(
                        ax,
                        x_day[day][a],
                        x_day[day][b],
                        bracket_y,
                        f"day {day}\n{stars}",
                        h=0.035 * span,
                        fs=7,
                    )
                    bracket_y += step

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(group_labels)
            ax.set_xlim(0.45, 3.55)
            ax.set_ylabel(PARAM_YLABELS[name])
            ax.set_xlabel("")
            _style_pub_ax(ax)

        fig.legend(
            handles=day_handles + age_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.55),
            frameon=False,
            fontsize=9,
        )
        fig.suptitle(
            rf"Day 1 vs Day 2 by age group ({title_mode})"
            "\n"
            r"colour = age group; marker = day (○ Day 1, □ Day 2); "
            r"black = mean ± SEM; brackets = significant contrasts only",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, outfile_overview)

    # ----- Stats detail figure -----
    mode_tag = "pair mean" if mode == "pair" else r"pooled $\{x_1,x_2\}$"
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 5.2), constrained_layout=True)
        for ax, (name, _c1, _c2, _mc) in zip(axes, params):
            ax.set_axis_off()
            if name not in day_vs:
                continue
            lines = [rf"{PARAM_MATH[name]} {mode_tag}", ""]
            lines.append("Day1 vs Day2 (Bonferroni×3)")
            for res in day_vs[name]:
                lines.append(f"  {res['age']}: n={res['n1']}/{res['n2']}")
                lines.append(f"    {res['test']}: {format_p(res.get('p'))}")
                lines.append(
                    f"    adj {format_p(res['p_adj'])} {p_to_stars(res['p_adj'])}"
                )
            lines.append("")
            for day in (1, 2):
                info = age_kw[name][day]
                lines.append(f"Age KW Day{day}: {format_p(info['p'])}")
                dunn = info.get("dunn") or []
                if not dunn:
                    lines.append("  Dunn: —")
                for a, b, pv in dunn:
                    lines.append(
                        f"  {_short_age_pair(a, b)}: "
                        f"{format_p(pv)} {p_to_stars(pv)}"
                    )
                lines.append("")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                family="monospace",
                color="0.15",
                linespacing=1.25,
            )
        fig.suptitle(
            f"Day × age statistics ({title_mode}; Days 1–2 only)\n"
            "Day within age: Shapiro→Welch or Mann–Whitney, Bonferroni across 3 ages; "
            "Age within day: Kruskal–Wallis + Dunn (Bonferroni)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, outfile_stats)


def plot_day_by_age_pair_mean(df: pd.DataFrame, out_dir: Path):
    """Day×age analysis for pair means only."""
    sub = df[df["day"].isin([1, 2])].copy()
    _run_day_by_age_mode(
        sub,
        out_dir,
        mode="pair",
        outfile_overview="13_day_by_age_overview.png",
        outfile_stats="13b_day_by_age_stats.png",
        title_mode="pair means",
    )


def plot_day_by_age_pooled(df: pd.DataFrame, out_dir: Path):
    """Day×age analysis for pooled sides {x1,x2}."""
    sub = df[df["day"].isin([1, 2])].copy()
    _run_day_by_age_mode(
        sub,
        out_dir,
        mode="pool",
        outfile_overview="14_day_by_age_pooled_overview.png",
        outfile_stats="14b_day_by_age_pooled_stats.png",
        title_mode=r"pooled $\{x_1,x_2\}$",
    )


def plot_day_by_age(df: pd.DataFrame, out_dir: Path):
    """Run Day×age analysis for pair means and for pooled sides."""
    plot_day_by_age_pair_mean(df, out_dir)
    plot_day_by_age_pooled(df, out_dir)

def parse_args():
    p = argparse.ArgumentParser(description="Plot general properties of ES pairs")
    p.add_argument(
        "--folder",
        default=None,
        help="Folder with the Excel file. If omitted, the script asks interactively.",
    )
    p.add_argument(
        "--figures",
        default=None,
        help="Comma-separated figure IDs to run (e.g. 01,07,13). "
        "Use 'all' for everything. If omitted, a selection window/prompt is shown.",
    )
    p.add_argument(
        "--all-figures",
        action="store_true",
        help="Generate all figures without asking (same as --figures all).",
    )
    return p.parse_args()


def build_figure_catalog(df: pd.DataFrame, out_dir: Path) -> list[dict]:
    """
    Registry of selectable figure jobs.
    Each item: id, label, outputs (file names), run (callable).
    Existing output files are overwritten when the job runs.
    """

    def _day_panels():
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

    def _asym_by_age_coupling():
        plot_asymmetry_by_age_bins(
            df,
            out_dir,
            rows=[("CC", ""), ("Gj", "nS")],
            outfile="03b_asymmetry_CC_Gj_by_age.png",
            title="Coupling asymmetry",
        )

    def _asym_by_age_cell():
        plot_asymmetry_by_age_bins(
            df,
            out_dir,
            rows=[("Rin", r"M$\Omega$"), ("Cm", "pF"), ("Vm", "mV")],
            outfile="04b_asymmetry_Rin_Cm_Vm_by_age.png",
            title="Within-pair asymmetry",
        )

    return [
        {
            "id": "01",
            "label": "01  Day 1–2–3 comparison (Gj/CC + Rin/Cm/Vm)",
            "outputs": [
                "01_day_comparison_Gj_CC.png",
                "01_day_comparison_Rin_Cm_Vm.png",
            ],
            "run": _day_panels,
        },
        {
            "id": "01b",
            "label": "01b Day 1 vs Day 2 statistics",
            "outputs": ["01b_day1_vs_day2_stats.png"],
            "run": lambda: plot_day12_statistics(df, out_dir),
        },
        {
            "id": "02",
            "label": "02  AI vs age (continuous colour)",
            "outputs": ["02_AI_vs_age.png"],
            "run": lambda: plot_ai_vs_age(df, out_dir),
        },
        {
            "id": "02b",
            "label": "02b AI by age group (KW + Dunn)",
            "outputs": ["02b_AI_by_age_group.png"],
            "run": lambda: plot_ai_by_age_groups(df, out_dir),
        },
        {
            "id": "03",
            "label": "03/04 Asymmetry clouds (Gj/CC + Rin/Cm/Vm)",
            "outputs": [
                "03_asymmetry_Gj_CC.png",
                "04_asymmetry_Rin_Cm_Vm.png",
            ],
            "run": lambda: plot_asymmetry_clouds(df, out_dir),
        },
        {
            "id": "03b",
            "label": "03b Coupling asymmetry by age bins",
            "outputs": ["03b_asymmetry_CC_Gj_by_age.png"],
            "run": _asym_by_age_coupling,
        },
        {
            "id": "04b",
            "label": "04b Within-pair asymmetry by age bins",
            "outputs": ["04b_asymmetry_Rin_Cm_Vm_by_age.png"],
            "run": _asym_by_age_cell,
        },
        {
            "id": "05",
            "label": "05  AI histograms",
            "outputs": ["05_AI_histograms.png"],
            "run": lambda: plot_ai_histograms(df, out_dir),
        },
        {
            "id": "06",
            "label": "06  Coupling vs Distance",
            "outputs": ["06_coupling_vs_distance.png"],
            "run": lambda: plot_coupling_vs_distance(df, out_dir),
        },
        {
            "id": "07",
            "label": "07  Gj/CC by age group (pooled + pair mean)",
            "outputs": ["07_Gj_CC_by_age_group.png"],
            "run": lambda: plot_coupling_by_age_groups(df, out_dir),
        },
        {
            "id": "08",
            "label": "08  AI_Gj / AI_CC vs AI_Rin",
            "outputs": ["08_AI_Gj_CC_vs_AI_Rin.png"],
            "run": lambda: plot_ai_gj_vs_ai_rin(df, out_dir),
        },
        {
            "id": "09",
            "label": "09  Signed ΔGj/ΔCC vs ΔRin_recv",
            "outputs": ["09_signed_dGj_dCC_vs_dRin.png"],
            "run": lambda: plot_signed_delta_gj_vs_rin(df, out_dir),
        },
        {
            "id": "10",
            "label": "10  Gj/CC vs age continuous (Spearman + LOWESS)",
            "outputs": ["10_Gj_CC_vs_age_continuous.png"],
            "run": lambda: plot_coupling_vs_age_continuous(df, out_dir),
        },
        {
            "id": "11",
            "label": "11  Distance + age coverage histograms",
            "outputs": ["11_distance_age_histograms.png"],
            "run": lambda: plot_coverage_histograms(df, out_dir),
        },
        {
            "id": "13",
            "label": "13  Day×age (pair means) + stats text",
            "outputs": [
                "13_day_by_age_overview.png",
                "13b_day_by_age_stats.png",
            ],
            "run": lambda: plot_day_by_age_pair_mean(df, out_dir),
        },
        {
            "id": "14",
            "label": "14  Day×age (pooled sides) + stats text",
            "outputs": [
                "14_day_by_age_pooled_overview.png",
                "14b_day_by_age_pooled_stats.png",
            ],
            "run": lambda: plot_day_by_age_pooled(df, out_dir),
        },
    ]


def _ask_figures_tk(catalog: list[dict], out_dir: Path) -> list[str] | None:
    """Checkbox window; returns selected ids, [] if none, None if cancelled."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Select figures to generate")
    root.geometry("720x560")
    root.minsize(520, 400)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    ttk.Label(
        frm,
        text=(
            "Choose figures to (re)generate.\n"
            f"Output folder: {out_dir}\n"
            "Existing files with the same name will be overwritten."
        ),
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(0, 8))

    canvas = tk.Canvas(frm, highlightthickness=0)
    scroll = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=inner, anchor=tk.NW)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)

    vars_by_id: dict[str, tk.BooleanVar] = {}
    for job in catalog:
        exists = any((out_dir / f).is_file() for f in job["outputs"])
        mark = "  [exists — will overwrite]" if exists else ""
        var = tk.BooleanVar(value=True)
        vars_by_id[job["id"]] = var
        ttk.Checkbutton(
            inner, text=f"{job['label']}{mark}", variable=var
        ).pack(anchor=tk.W, pady=2)

    result: dict[str, list[str] | None] = {"ids": None}

    def _select_all(val: bool):
        for v in vars_by_id.values():
            v.set(val)

    def _ok():
        result["ids"] = [jid for jid, v in vars_by_id.items() if v.get()]
        root.destroy()

    def _cancel():
        result["ids"] = None
        root.destroy()

    btns = ttk.Frame(root, padding=10)
    btns.pack(fill=tk.X)
    ttk.Button(btns, text="Select all", command=lambda: _select_all(True)).pack(
        side=tk.LEFT, padx=4
    )
    ttk.Button(btns, text="Select none", command=lambda: _select_all(False)).pack(
        side=tk.LEFT, padx=4
    )
    ttk.Button(btns, text="Cancel", command=_cancel).pack(side=tk.RIGHT, padx=4)
    ttk.Button(btns, text="Run selected", command=_ok).pack(side=tk.RIGHT, padx=4)

    root.protocol("WM_DELETE_WINDOW", _cancel)
    root.mainloop()
    return result["ids"]


def _ask_figures_console(catalog: list[dict], out_dir: Path) -> list[str] | None:
    """Console multi-select fallback when tkinter is unavailable."""
    print("\nSelect figures to (re)generate.")
    print(f"Output folder: {out_dir}")
    print("Existing files with the same name will be overwritten.")
    print("  all          — all figures")
    print("  none / Enter empty after numbers — cancel")
    print("  e.g. 01,07,13 or 1 7 13\n")
    for i, job in enumerate(catalog, start=1):
        exists = any((out_dir / f).is_file() for f in job["outputs"])
        mark = " [exists]" if exists else ""
        print(f"  {i:2d}. {job['id']:4s}  {job['label']}{mark}")
    raw = input("\nIDs or numbers > ").strip()
    if not raw:
        return None
    if raw.lower() in {"all", "*"}:
        return [j["id"] for j in catalog]
    if raw.lower() in {"none", "cancel", "q"}:
        return None
    tokens = [t.strip() for t in raw.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    id_set = {j["id"] for j in catalog}
    id_by_num = {str(i): j["id"] for i, j in enumerate(catalog, start=1)}
    selected = []
    for t in tokens:
        if t in id_set:
            selected.append(t)
        elif t in id_by_num:
            selected.append(id_by_num[t])
        else:
            print(f"  Unknown: {t}")
    # preserve catalog order, unique
    order = [j["id"] for j in catalog]
    return [jid for jid in order if jid in set(selected)]


def ask_which_figures(catalog: list[dict], out_dir: Path) -> list[str] | None:
    try:
        return _ask_figures_tk(catalog, out_dir)
    except Exception as exc:
        print(f"Figure selection window unavailable ({exc}); using console prompt.")
        return _ask_figures_console(catalog, out_dir)


def parse_figure_ids(catalog: list[dict], figures_arg: str) -> list[str]:
    """Parse --figures value into ordered catalog ids; exit on unknown ids."""
    raw = figures_arg.strip()
    if raw.lower() == "all":
        return [j["id"] for j in catalog]
    wanted = {t.strip() for t in raw.replace(";", ",").split(",") if t.strip()}
    known = {j["id"] for j in catalog}
    unknown = sorted(wanted - known)
    if unknown:
        print(f"Unknown figure ids: {', '.join(unknown)}")
        print("Known:", ", ".join(j["id"] for j in catalog))
        sys.exit(1)
    return [j["id"] for j in catalog if j["id"] in wanted]


def main():
    args = parse_args()
    folder = ask_data_folder(args.folder)
    excel_path = find_excel(folder)
    out_dir = folder / FIGURES_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {excel_path}")
    df = load_and_prepare(excel_path)
    print(f"Rows after Gj filter: {len(df)}")
    print(f"Days present: {sorted(df['day'].dropna().unique().tolist())}")
    print(
        f"{MT_DISPLAY_LABEL} pairs (Excel {MT_EXCEL_TOKEN}): {int(df['is_mt'].sum())}"
    )
    print(f"Pool days 1–2: {int(df['in_pool'].sum())}")

    catalog = build_figure_catalog(df, out_dir)

    if args.all_figures or (args.figures and args.figures.strip().lower() == "all"):
        selected = [j["id"] for j in catalog]
    elif args.figures:
        selected = parse_figure_ids(catalog, args.figures)
    else:
        selected = ask_which_figures(catalog, out_dir)

    if selected is None:
        print("Cancelled — no figures generated.")
        return
    if not selected:
        print("Nothing selected — no figures generated.")
        return

    jobs = {j["id"]: j for j in catalog}
    print(f"\nGenerating {len(selected)} figure job(s): {', '.join(selected)}")
    for jid in selected:
        job = jobs[jid]
        print(f"\n=== {job['label']} ===")
        job["run"]()
    print("\nDone. Figures are in:", out_dir)


if __name__ == "__main__":
    main()
