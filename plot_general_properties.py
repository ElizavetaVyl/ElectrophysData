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
# Asymmetry / AI / coupling / coverage: Days 1–2 only. Day 3 stays on the
# Day 1–2–3 comparison figure (01) and is not mixed into the analysis pool.
POOL_DAYS = {1, 2}
POOL_NOTE = r"pool days 1–2 (Day 3 excluded)"
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
    # Spike → spikelet transmission (optional Excel columns)
    "CCa": ("amp_ratio_mean_12", "amp_ratio_mean_21"),
    "DelayOnset": ("delay_onset_ms_mean_12", "delay_onset_ms_mean_21"),
    "DelayPeak": ("delay_peak_ms_mean_12", "delay_peak_ms_mean_21"),
}

PARAM_MATH = {
    "Gj": r"$G_{j}$",
    "CC": r"$CC$",
    "Rin": r"$R_{in}$",
    "Cm": r"$C_{m}$",
    "Vm": r"$V_{m}$",
    "CCa": r"$CC_{a}$",
    "DelayOnset": r"$t_{onset}$",
    "DelayPeak": r"$t_{peak}$",
}

PARAM_YLABELS = {
    "Gj": r"$G_{j}$ [nS]",
    "CC": r"$CC$",
    "Rin": r"$R_{in}$ [M$\Omega$]",
    "Cm": r"$C_{m}$ [pF]",
    "Vm": r"$V_{m}$ [mV]",
    "CCa": r"$CC_{a}$ (amp. ratio)",
    "DelayOnset": r"onset delay [ms]",
    "DelayPeak": r"peak delay [ms]",
}

# Marks on CC/Gj axis when CCa is missing for that directed value
MISSING_CCA_MARKER = {
    "marker": "x",
    "color": "0.45",
    "s": 28,
    "alpha": 0.85,
    "linewidths": 1.0,
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


def parse_exclude_flag(value) -> bool:
    """True if Excel Exclude? marks the pair to skip from analysis plots."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return False
        return bool(value) and float(value) != 0.0
    text = str(value).strip().lower()
    # strip zero-width / BOM junk sometimes pasted from Excel
    text = re.sub(r"[\u200b\ufeff]", "", text)
    if not text:
        return False
    return text in {"yes", "y", "true", "1", "exclude", "excluded"}


def _find_exclude_column(columns: list[str]) -> str | None:
    """Match Exclude? / Exclude / exclude? (ASCII or fullwidth ?)."""
    for c in columns:
        n = re.sub(r"[\s_?？]+", "", str(c).strip().lower())
        if n in {"exclude", "excluded"} or n.startswith("exclude"):
            return c
    return None


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
    # Keep rows without Gj for Distance/age coverage (often Exclude?=yes).
    # Coupling / AI / day stats still require both Gj sides via has_gj.
    df["has_gj"] = df["Gj_12"].notna() & df["Gj_21"].notna()

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

    # Optional Exclude? column: yes → keep for coverage hists; skip analysis plots
    excl_col = _find_exclude_column(list(df.columns))
    if excl_col is not None:
        df["excluded"] = df[excl_col].map(parse_exclude_flag)
        df.attrs["exclude_column"] = excl_col
    else:
        df["excluded"] = False
        df.attrs["exclude_column"] = None

    # Drop empty junk rows with neither Gj nor Distance nor age/Dam
    keep = (
        df["has_gj"]
        | (df["Distance_um"].notna() if "Distance_um" in df.columns else False)
        | df["age_num"].notna()
        | df["is_mt"]
    )
    df = df[keep].copy()

    df["in_pool"] = df["day"].isin(POOL_DAYS) & ~df["excluded"] & df["has_gj"]
    # Spike transmission coverage (optional columns)
    cca1, cca2 = PARAM_DIRS["CCa"]
    if cca1 in df.columns and cca2 in df.columns:
        df["has_cca_any"] = df[cca1].notna() | df[cca2].notna()
        df["has_cca_both"] = df[cca1].notna() & df[cca2].notna()
        # Zero-fill asymmetry / AI: missing side → 0; both missing → NaN
        a = df[cca1]
        b = df[cca2]
        any_cca = a.notna() | b.notna()
        a0 = a.fillna(0.0)
        b0 = b.fillna(0.0)
        mx = pd.Series(np.nan, index=df.index, dtype=float)
        mn = pd.Series(np.nan, index=df.index, dtype=float)
        ai = pd.Series(np.nan, index=df.index, dtype=float)
        pm = pd.Series(np.nan, index=df.index, dtype=float)
        mx[any_cca] = np.maximum(a0[any_cca], b0[any_cca])
        mn[any_cca] = np.minimum(a0[any_cca], b0[any_cca])
        denom = mx + mn
        ai[any_cca] = np.where(
            denom[any_cca] != 0,
            (mx[any_cca] - mn[any_cca]) / denom[any_cca],
            np.nan,
        )
        pm[any_cca] = (a0[any_cca] + b0[any_cca]) / 2.0
        df["CCa_max_z"] = mx
        df["CCa_min_z"] = mn
        df["AI_CCa_z"] = ai
        df["CCa_pair_mean_z"] = pm
        df["cca_modality"] = np.select(
            [df["has_cca_both"], df["has_cca_any"]],
            ["both", "one"],
            default="none",
        )
    else:
        df["has_cca_any"] = False
        df["has_cca_both"] = False
        df["cca_modality"] = "none"

    # Membrane time constant approximation: τ ≈ Rin × Cm
    # Units in this Excel: Rin [MΩ], Cm [pF] → τ [ms] = Rin × Cm / 1000
    # (1 MΩ·pF = 10^6 Ω × 10^-12 F = 10^-6 s = 0.001 ms)
    if "Rin_1" in df.columns and "Cm_1" in df.columns:
        df["Tau_1"] = np.where(
            df["Rin_1"].notna() & df["Cm_1"].notna(),
            df["Rin_1"] * df["Cm_1"] / 1000.0,
            np.nan,
        )
    if "Rin_2" in df.columns and "Cm_2" in df.columns:
        df["Tau_2"] = np.where(
            df["Rin_2"].notna() & df["Cm_2"].notna(),
            df["Rin_2"] * df["Cm_2"] / 1000.0,
            np.nan,
        )
    if "Tau_1" in df.columns and "Tau_2" in df.columns:
        both_tau = df["Tau_1"].notna() & df["Tau_2"].notna()
        df["Tau_pair_mean"] = np.where(
            both_tau, (df["Tau_1"] + df["Tau_2"]) / 2.0, np.nan
        )
    return df.reset_index(drop=True)


def included_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day 1–2–3 comparison and day stats: Require Gj_12 & Gj_21 filled,
    drop Exclude?=yes. (Coverage histograms are the only plots that keep Exclude?=yes.)
    """
    mask = df["has_gj"] if "has_gj" in df.columns else True
    if "excluded" in df.columns:
        mask = mask & ~df["excluded"]
    return df[mask].copy()


def analysis_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asymmetry / AI / coupling / day×age: Days 1–2, Exclude?=yes out, Gj both sides.
    Only fig. 11 coverage panels marked “incl. Exclude?=yes” keep excluded rows
    (those may lack Gj and are used solely for Distance/age counts).
    """
    mask = df["day"].isin(POOL_DAYS)
    if "has_gj" in df.columns:
        mask = mask & df["has_gj"]
    if "excluded" in df.columns:
        mask = mask & ~df["excluded"]
    return df[mask].copy()


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


# Fixed-width bins for envelope-style summaries (pre-specified step on x)
GJ_BIN_WIDTH = 0.2  # nS (pair-mean Gj)
CC_BIN_WIDTH = 0.05  # pair-mean CC (AI-vs-strength bins)
CC_BIN_WIDTH_CCA_COV = 0.025  # directed-CC bins for CCa coverage %
DISTANCE_BIN_WIDTH = 20.0  # µm
BIN_POINT_COLOR = "#6E6E6E"



def assign_width_bins(
    x: pd.Series,
    width: float,
    *,
    origin: float = 0.0,
    drop_empty: bool = True,
) -> tuple[pd.Series, list[tuple[str, float, float, int]]]:
    """
    Fixed-width bins on x: [origin + k·w, origin + (k+1)·w), last bin closed on the right.
    Returns (bin label per row; NA where x is NA) and meta
    [(label, lo, hi, n), ...] in order. Empty bins dropped by default.
    """
    out = pd.Series(pd.NA, index=x.index, dtype="object")
    meta: list[tuple[str, float, float, int]] = []
    valid = x.dropna().astype(float)
    if len(valid) < 3 or not np.isfinite(width) or width <= 0:
        return out, meta
    xmin = float(valid.min())
    xmax = float(valid.max())
    start = float(origin + width * np.floor((xmin - origin) / width))
    n_steps = int(np.ceil((xmax - start) / width))
    n_steps = max(n_steps, 1)
    vals = valid.to_numpy(dtype=float)
    idx = np.floor((vals - start) / width).astype(int)
    idx = np.clip(idx, 0, n_steps - 1)
    labels = np.array(
        [f"{start + i * width:g}–{start + (i + 1) * width:g}" for i in range(n_steps)],
        dtype=object,
    )
    assigned = pd.Series(labels[idx], index=valid.index, dtype=object)
    out.loc[assigned.index] = assigned
    for i in range(n_steps):
        lo = start + i * width
        hi = start + (i + 1) * width
        name = labels[i]
        n = int((idx == i).sum())
        if drop_empty and n == 0:
            continue
        meta.append((name, float(lo), float(hi), n))
    return out, meta


def _fmt_bin_edge(v: float, *, kind: str) -> str:
    if kind == "distance":
        return f"{v:.0f}"
    if kind == "gj":
        return f"{v:.2g}"
    if kind == "cc":
        return f"{v:.2f}"
    return f"{v:.2g}"


def _draw_width_bin_panel(
    ax,
    *,
    samples: list[np.ndarray],
    meta: list[tuple[str, float, float, int]],
    ylabel: str,
    edge_kind: str,
    seed_key: str,
    y_lim: tuple[float, float] | None = None,
):
    """Jittered points + mean±SEM + median; light KW on bins with n≥3 only."""
    n_bins = len(meta)
    for i, ((name, lo, hi, _n), vals) in enumerate(zip(meta, samples)):
        if len(vals) == 0:
            continue
        xpos = i + 1
        jitter = np.random.default_rng(abs(hash(seed_key + name)) % (2**32)).normal(
            0, 0.06, len(vals)
        )
        ax.scatter(
            np.full(len(vals), xpos) + jitter,
            vals,
            c=BIN_POINT_COLOR,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        ax.errorbar(
            [xpos],
            [float(np.mean(vals))],
            yerr=[float(stats.sem(vals)) if len(vals) > 1 else 0.0],
            fmt="o",
            color="crimson",
            markersize=6,
            capsize=3,
            zorder=3,
        )
        ax.plot(
            [xpos],
            [float(np.median(vals))],
            marker="D",
            color="royalblue",
            markersize=5,
            zorder=4,
        )

    nonempty = [s for s in samples if len(s) >= 3]
    if len(nonempty) >= 2:
        try:
            _h, p_kw = stats.kruskal(*nonempty)
            ax.text(
                0.98,
                0.96,
                f"KW {format_p(p_kw)}\n(bins n≥3)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.35",
            )
        except ValueError:
            pass

    tick_labels = []
    for _name, lo, hi, n in meta:
        lo_s = _fmt_bin_edge(lo, kind=edge_kind)
        hi_s = _fmt_bin_edge(hi, kind=edge_kind)
        tick_labels.append(f"{lo_s}–{hi_s}\nn={n}")
    ax.set_xticks(range(1, n_bins + 1))
    rot = 0 if n_bins <= 5 else (35 if n_bins <= 10 else 60)
    ax.set_xticklabels(tick_labels, fontsize=7 if n_bins > 6 else 8, rotation=rot)
    if rot:
        ax.tick_params(axis="x", pad=2)
    ax.set_xlim(0.4, n_bins + 0.6)
    ax.set_ylabel(ylabel)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_pub_ax(ax)


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


def _spike_cols_present(df: pd.DataFrame) -> bool:
    need = [
        *PARAM_DIRS["CCa"],
        *PARAM_DIRS["DelayOnset"],
        *PARAM_DIRS["DelayPeak"],
    ]
    return all(c in df.columns for c in need)


def directed_xy(
    df: pd.DataFrame,
    x12: str,
    x21: str,
    y12: str,
    y21: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stack directed 12/21 into long rows.
    Returns (has_both_xy, has_x_missing_y) with columns x, y, age_num, is_mt, side.
    """
    rows_both = []
    rows_miss = []
    for side, xc, yc in (("12", x12, y12), ("21", x21, y21)):
        if xc not in df.columns or yc not in df.columns:
            continue
        for _, r in df.iterrows():
            xv, yv = r[xc], r[yc]
            base = {
                "x": xv,
                "y": yv,
                "age_num": r.get("age_num", np.nan),
                "is_mt": bool(r.get("is_mt", False)),
                "side": side,
            }
            if pd.notna(xv) and pd.notna(yv):
                rows_both.append(base)
            elif pd.notna(xv) and pd.isna(yv):
                rows_miss.append(base)
    return pd.DataFrame(rows_both), pd.DataFrame(rows_miss)


def _scatter_directed_with_missing_rug(
    ax,
    both: pd.DataFrame,
    missing: pd.DataFrame,
    *,
    vmin: float,
    max_age: float,
    mt_val: float,
    mark_missing: bool,
    y_label: str,
    x_label: str,
):
    """Age-coloured scatter for (x,y); optional × rug on x for missing y (CCa)."""
    if len(both):
        colors, _, _ = age_colors(
            both["age_num"].to_numpy(),
            both["is_mt"].to_numpy(),
            vmin,
            max_age,
            mt_val,
        )
        ax.scatter(
            both["x"],
            both["y"],
            c=colors,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        if len(both) >= 3:
            rho, p = stats.spearmanr(both["x"], both["y"])
            ax.text(
                0.04,
                0.96,
                f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={len(both)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="0.25",
            )
    if mark_missing and len(missing):
        # Place missing-CCa marks just above the bottom of the axes
        y0 = both["y"].min() if len(both) else 0.0
        y1 = both["y"].max() if len(both) else 1.0
        span = max(y1 - y0, abs(y1) * 0.05, 1e-6)
        y_rug = y0 - 0.06 * span
        ax.scatter(
            missing["x"],
            np.full(len(missing), y_rug),
            marker=MISSING_CCA_MARKER["marker"],
            c=MISSING_CCA_MARKER["color"],
            s=MISSING_CCA_MARKER["s"],
            alpha=MISSING_CCA_MARKER["alpha"],
            linewidths=MISSING_CCA_MARKER["linewidths"],
            zorder=2,
            clip_on=False,
        )
        ax.text(
            0.98,
            0.04,
            f"× no $CC_a$: n={len(missing)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="0.4",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _style_pub_ax(ax)


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
    df = included_pairs(df)
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
    df = included_pairs(df)
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
    sub = analysis_pool(df)
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
            f"{AI_FORMULA}; X stretched to P50; dotted: AI≈0.33 (~2×); {POOL_NOTE}",
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
    """
    AI by age group (P10–32 / P33–50 / P50+).
    Overview: points + mean±SEM + median; Dunn brackets/stars only (no p-text).
    Separate stats figure: Kruskal–Wallis + Dunn (Bonferroni) details.
    """
    sub = analysis_pool(df)
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    x_pos = {lab: i + 1 for i, lab in enumerate(group_labels)}
    panels = [
        ("AI_Gj", r"AI $G_{j}$"),
        ("AI_CC", r"AI $CC$"),
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_{m}$"),
        ("AI_Vm", r"AI $|V_{m}|$"),
    ]

    # Collect samples + stats once for overview and stats figure
    panel_stats: dict[str, dict] = {}
    print("\nAI by age group — Kruskal–Wallis + Dunn (Bonferroni):")
    for col, _ylabel in panels:
        if col not in sub.columns:
            continue
        samples = []
        ns = []
        for label, lo, hi, include_mt, _color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            vals = sub.loc[mask, col].dropna().to_numpy(dtype=float)
            samples.append(vals)
            ns.append(int(len(vals)))
        nonempty = [s for s in samples if len(s) > 0]
        if len(nonempty) >= 2:
            try:
                H, p = stats.kruskal(*nonempty)
                dunn = _dunn_posthoc(samples, group_labels)
            except Exception as exc:
                H, p, dunn = float("nan"), float("nan"), []
                print(f"  {col}: KW n/a ({exc})")
            else:
                dunn_bits = [
                    f"{_short_age_pair(a, b)} {format_p(pv)}" for a, b, pv in dunn
                ]
                print(f"  {col}: KW H={H:.3f}, {format_p(p)}; Dunn: {dunn_bits}")
        else:
            H, p, dunn = float("nan"), float("nan"), []
            print(f"  {col}: KW n/a")
        panel_stats[col] = {
            "samples": samples,
            "ns": ns,
            "H": H,
            "p": p,
            "dunn": dunn,
        }

    # ----- Clean overview (article figure) -----
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6), constrained_layout=True)
        ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]]
        axes[0, 2].set_visible(False)

        for ax, (col, ylabel) in zip(ax_list, panels):
            if col not in panel_stats:
                ax.set_visible(False)
                continue
            info = panel_stats[col]
            samples = info["samples"]
            y_hi = 0.0
            for (label, _lo, _hi, _mt, color), vals in zip(AGE_ASYM_BINS, samples):
                if len(vals) == 0:
                    continue
                y_hi = max(y_hi, float(np.max(vals)))
                jitter = np.random.default_rng(
                    abs(hash(col + label)) % (2**32)
                ).normal(0, 0.06, len(vals))
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

            ax.axhline(1 / 3, color="0.6", ls=":", lw=0.8, zorder=0)
            ax.set_xticks([x_pos[l] for l in group_labels])
            ax.set_xticklabels(group_labels)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")

            # Significant Dunn pairs only: brackets + stars (no p-text)
            sig_pairs = [
                (a, b, pv)
                for a, b, pv in (info["dunn"] or [])
                if p_to_stars(pv) != "ns"
            ]
            # Order short spans first so stacked brackets stay readable
            sig_pairs.sort(
                key=lambda t: abs(x_pos[t[0]] - x_pos[t[1]])
            )
            y_base = max(y_hi, 1 / 3) + 0.04
            bracket_y = y_base
            step = 0.08
            for a, b, pv in sig_pairs:
                _sig_bracket(
                    ax,
                    x_pos[a],
                    x_pos[b],
                    bracket_y,
                    p_to_stars(pv),
                    h=0.03,
                    fs=10,
                )
                bracket_y += step
            y_top = max(1.05, bracket_y + 0.02) if sig_pairs else 1.05
            ax.set_ylim(0, y_top)
            _style_pub_ax(ax)

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
            Line2D(
                [0],
                [0],
                color="0.6",
                ls=":",
                lw=0.9,
                label=r"AI $=1/3$ (~2×)",
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
            r"AI by age group"
            "\n"
            r"dotted: AI $=1/3$ (~2-fold); brackets: significant Dunn pairs only",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "02b_AI_by_age_group.png")

    # ----- Stats detail figure -----
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.8), constrained_layout=True)
        for ax, (col, ylabel) in zip(axes, panels):
            ax.set_axis_off()
            if col not in panel_stats:
                continue
            info = panel_stats[col]
            lines = [ylabel.replace("$", ""), ""]
            lines.append(f"n = {' / '.join(str(n) for n in info['ns'])}")
            lines.append(f"  ({' / '.join(group_labels)})")
            lines.append("")
            lines.append(f"Kruskal–Wallis: {format_p(info['p'])}")
            if np.isfinite(info["H"]):
                lines.append(f"  H = {info['H']:.3f}")
            lines.append("")
            lines.append("Dunn (Bonferroni)")
            dunn = info["dunn"] or []
            if not dunn:
                lines.append("  —")
            for a, b, pv in dunn:
                lines.append(
                    f"  {_short_age_pair(a, b)}: "
                    f"{format_p(pv)} {p_to_stars(pv)}"
                )
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                family="monospace",
                color="0.15",
                linespacing=1.3,
            )
        fig.suptitle(
            "AI by age group — statistics\n"
            f"Kruskal–Wallis across P10–32 / P33–50 / P50+; "
            f"Dunn post-hoc with Bonferroni; {POOL_NOTE}",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "02b_AI_by_age_group_stats.png")


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
    sub = analysis_pool(df)
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

    def _finish(fig, axes, outfile: str, title: str):
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
        fig.suptitle(f"{title}\n{POOL_NOTE}", fontsize=11, color="0.15")
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
        _finish(fig, axes, "03_asymmetry_Gj_CC.png", "Coupling asymmetry")

        fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
        for ax, name in zip(axes2, ["Rin", "Cm", "Vm"]):
            _panel(ax, name, with_fit=True, unit_note=units[name])
        _finish(
            fig2, axes2, "04_asymmetry_Rin_Cm_Vm.png", "Within-pair asymmetry"
        )


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
    sub = analysis_pool(df)
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

        fig.suptitle(f"{title}\n{POOL_NOTE}", fontsize=11, color="0.15")
        savefig(fig, out_dir, outfile)


# ---------------------------------------------------------------------------
# Coupling vs inter-soma Distance
# ---------------------------------------------------------------------------
def plot_coupling_vs_distance(df: pd.DataFrame, out_dir: Path):
    """
    Gj and CC vs Distance (µm): both directed values + pair mean on the axes.
    Spearman (pair mean and pooled {x12,x21}) reported on a separate stats figure,
    not overlaid on the scatter.
    """
    if "Distance_um" not in df.columns:
        print("Skipping distance plots: no Distance column in Excel.")
        return

    sub = analysis_pool(df)
    sub = sub[sub["Distance_um"].notna()].copy()
    if len(sub) == 0:
        print("Skipping distance plots: no Distance values in pool days 1–2.")
        return

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        ("Gj", "Gj_12", "Gj_21", "Gj_pair_mean", rf"{PARAM_MATH['Gj']} [nS]"),
        ("CC", "CC_12", "CC_21", "CC_pair_mean", rf"{PARAM_MATH['CC']}"),
    ]
    spearman_rows: list[dict] = []

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        print("\nCoupling vs Distance (plot: both sides + mean; Spearman → stats fig):")

        for ax, (name, c1, c2, mean_col, ylabel) in zip(axes, panels):
            need = [c1, c2, mean_col]
            if any(c not in sub.columns for c in need):
                ax.set_visible(False)
                continue
            d = sub[sub[c1].notna() & sub[c2].notna() & sub[mean_col].notna()].copy()
            if len(d) == 0:
                ax.set_visible(False)
                continue
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            x = d["Distance_um"].to_numpy(dtype=float)
            y1 = d[c1].to_numpy(dtype=float)
            y2 = d[c2].to_numpy(dtype=float)
            ym = d[mean_col].to_numpy(dtype=float)

            for xi, a, b, col in zip(x, y1, y2, colors):
                ax.plot(
                    [xi, xi],
                    [a, b],
                    color=col,
                    lw=0.6,
                    alpha=0.35,
                    zorder=1,
                    solid_capstyle="round",
                )
            ax.scatter(
                x,
                y1,
                c=colors,
                s=POINT_SIZE * 0.55,
                alpha=0.45,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
            ax.scatter(
                x,
                y2,
                c=colors,
                s=POINT_SIZE * 0.55,
                alpha=0.45,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
            ax.scatter(
                x,
                ym,
                c=colors,
                s=POINT_SIZE * 1.15,
                alpha=0.9,
                edgecolors="0.15",
                linewidths=0.5,
                zorder=3,
            )

            # Spearman: pair mean (1 / pair) and pooled both sides (2 / pair)
            x_pool = np.concatenate([x, x])
            y_pool = np.concatenate([y1, y2])
            row = {
                "name": name,
                "n_pairs": len(d),
                "rho_mean": np.nan,
                "p_mean": np.nan,
                "rho_pool": np.nan,
                "p_pool": np.nan,
                "n_pool": int(len(y_pool)),
            }
            if len(d) >= 3:
                row["rho_mean"], row["p_mean"] = stats.spearmanr(x, ym)
                row["rho_pool"], row["p_pool"] = stats.spearmanr(x_pool, y_pool)
                print(
                    f"  {name} mean: ρ={row['rho_mean']:.3f}, {format_p(row['p_mean'])}, "
                    f"n_pairs={row['n_pairs']}"
                )
                print(
                    f"  {name} pooled {{x12,x21}}: ρ={row['rho_pool']:.3f}, "
                    f"{format_p(row['p_pool'])}, n={row['n_pool']}"
                )
            else:
                print(f"  {name}: n={len(d)} (too few for Spearman)")
            spearman_rows.append(row)

            ax.set_xlabel(r"Distance [$\mu$m]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.55",
                markeredgecolor="white",
                markersize=5,
                alpha=0.7,
                label=r"directed $x_{12}$, $x_{21}$",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.35",
                markeredgecolor="0.15",
                markersize=7,
                label=r"pair mean $(x_{12}+x_{21})/2$",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.55),
            frameon=False,
            fontsize=9,
        )
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
            r"small points: both directions; large: pair mean; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "06_coupling_vs_distance.png")

    # Separate stats figure (not overlaid on the scatter)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig_s, axes_s = plt.subplots(
            1, max(len(spearman_rows), 1), figsize=(9.5, 3.6), constrained_layout=True
        )
        if len(spearman_rows) == 1:
            axes_s = [axes_s]
        elif len(spearman_rows) == 0:
            axes_s = [axes_s] if not hasattr(axes_s, "__iter__") else list(axes_s)
        for ax, row in zip(axes_s, spearman_rows):
            ax.set_axis_off()
            sym = PARAM_MATH.get(row["name"], row["name"])
            lines = [
                rf"{sym} vs Distance",
                "",
                "Pair mean $(x_{12}+x_{21})/2$",
                f"  n_pairs = {row['n_pairs']}",
                f"  Spearman ρ = {row['rho_mean']:.3f}"
                if np.isfinite(row["rho_mean"])
                else "  Spearman ρ = n/a",
                f"  {format_p(row['p_mean'])}",
                "",
                r"Pooled $\{x_{12},x_{21}\}$ (all directed values)",
                f"  n = {row['n_pool']}",
                f"  Spearman ρ = {row['rho_pool']:.3f}"
                if np.isfinite(row["rho_pool"])
                else "  Spearman ρ = n/a",
                f"  {format_p(row['p_pool'])}",
                "",
                "Note: $(x_{12}+x_{21})$ has the same Spearman as the pair mean.",
            ]
            ax.text(
                0.05,
                0.95,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                color="0.15",
                linespacing=1.35,
            )
        for ax in list(axes_s)[len(spearman_rows) :]:
            ax.set_axis_off()
        fig_s.suptitle(
            "Coupling vs Distance — Spearman statistics\n"
            f"{POOL_NOTE}",
            fontsize=11,
            color="0.15",
        )
        savefig(fig_s, out_dir, "06b_coupling_vs_distance_stats.png")

    plot_coupling_by_distance_bins(df, out_dir)


def plot_coupling_by_distance_bins(df: pd.DataFrame, out_dir: Path):
    """
    Pair-mean Gj / CC by fixed Distance bins (width = DISTANCE_BIN_WIDTH µm).
    Descriptive summary of the cloud envelope; light KW on bins with n≥3.
    """
    if "Distance_um" not in df.columns:
        print("Skipping distance width bins: no Distance column.")
        return
    sub = analysis_pool(df)
    sub = sub[sub["Distance_um"].notna()].copy()
    if len(sub) < 3:
        print("Skipping distance width bins: too few pairs with Distance.")
        return

    bins, meta = assign_width_bins(sub["Distance_um"], DISTANCE_BIN_WIDTH)
    if not meta:
        print("Skipping distance width bins: could not form bins.")
        return
    sub = sub.copy()
    sub["_dist_bin"] = bins

    panels = [
        ("Gj_pair_mean", rf"{PARAM_MATH['Gj']} pair mean [nS]", "gj"),
        ("CC_pair_mean", rf"{PARAM_MATH['CC']} pair mean", "cc"),
    ]
    fig_w = max(10.5, 1.15 * len(meta) + 2.5)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.8), constrained_layout=True)
        print(
            f"\nPair-mean coupling by Distance bins "
            f"(width={DISTANCE_BIN_WIDTH:g} µm):"
        )
        for ax, (ycol, ylabel, _kind) in zip(axes, panels):
            if ycol not in sub.columns:
                ax.set_visible(False)
                continue
            samples = []
            for name, _lo, _hi, _n in meta:
                vals = (
                    sub.loc[sub["_dist_bin"] == name, ycol]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                samples.append(vals)
                med = float(np.median(vals)) if len(vals) else float("nan")
                p75 = float(np.quantile(vals, 0.75)) if len(vals) else float("nan")
                print(
                    f"  {ycol} {name}: n={len(vals)}, median={med:.3g}, P75={p75:.3g}"
                )
            _draw_width_bin_panel(
                ax,
                samples=samples,
                meta=meta,
                ylabel=ylabel,
                edge_kind="distance",
                seed_key=f"dist-{ycol}",
            )
            ax.set_xlabel(rf"Distance bin [$\mu$m] (width={DISTANCE_BIN_WIDTH:g})")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="crimson",
                lw=0,
                markersize=6,
                label="mean ± SEM",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="royalblue",
                lw=0,
                markersize=5,
                label="median",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(
            rf"Pair-mean coupling by Distance bins ({DISTANCE_BIN_WIDTH:g} $\mu$m)"
            "\n"
            r"fixed-width bins on Distance (empty bins omitted); points = pairs; "
            r"descriptive; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "06c_coupling_by_distance_bins.png")


# ---------------------------------------------------------------------------
# Gj / CC by age group: pooled sides + pair mean (KW + Dunn)
# ---------------------------------------------------------------------------
def plot_coupling_by_age_groups(df: pd.DataFrame, out_dir: Path):
    """
    2×2: top = pooled {x1, x2} for Gj and CC; bottom = pair means.
    Age groups P10–32 / P33–50 / P50+; Kruskal–Wallis + Dunn (Bonferroni).
    """
    sub = analysis_pool(df)
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
            rf"top: pooled sides; bottom: pair mean; Kruskal–Wallis; Dunn (Bonferroni); {POOL_NOTE}",
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
    sub = analysis_pool(df)
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
            rf"Spearman ρ = rank correlation (not a fit slope); crimson: LOWESS; {POOL_NOTE}",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "10_Gj_CC_vs_age_continuous.png")


# ---------------------------------------------------------------------------
# Coverage: Distance + age histograms (with / without Exclude?=yes)
# ---------------------------------------------------------------------------
def _coverage_frame(df: pd.DataFrame, include_excluded: bool) -> pd.DataFrame:
    """All days; optionally drop Exclude?=yes rows."""
    if include_excluded or "excluded" not in df.columns:
        return df.copy()
    return df[~df["excluded"]].copy()


def _draw_distance_hist(ax, dists: np.ndarray, panel_title: str):
    if len(dists) == 0:
        ax.text(0.5, 0.5, "no Distance", ha="center", va="center", color="0.4")
        ax.set_title(panel_title, fontsize=10, color="0.25")
        ax.set_xlabel(r"Distance [$\mu$m]")
        ax.set_ylabel("Number of pairs")
        _style_pub_ax(ax)
        return
    d_min = float(np.floor(np.min(dists) / 10.0) * 10.0)
    d_max = float(np.ceil(np.max(dists) / 10.0) * 10.0)
    if d_max <= d_min:
        d_max = d_min + 10.0
    width = 15.0
    edges = np.arange(d_min, d_max + width, width)
    if len(edges) < 7:
        width = 10.0
        edges = np.arange(d_min, d_max + width, width)
    ax.hist(dists, bins=edges, color="0.55", edgecolor="white", alpha=0.9)
    d_mean = float(np.mean(dists))
    ax.axvline(d_mean, color="crimson", lw=1.3)
    ax.text(
        0.98,
        0.96,
        f"$n$ = {len(dists)}\n"
        f"{np.min(dists):.0f}–{np.max(dists):.0f} µm\n"
        f"mean = {d_mean:.0f} µm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.25",
    )
    ax.set_title(panel_title, fontsize=10, color="0.25")
    ax.set_xlabel(r"Distance [$\mu$m]")
    ax.set_ylabel("Number of pairs")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_pub_ax(ax)
    print(f"  Distance [{panel_title}]: n={len(dists)}, mean={d_mean:.1f} µm")


def _draw_age_coverage_hist(ax, frame: pd.DataFrame, panel_title: str):
    """
    Warped age histogram; band labels P10–32 / P33–50 / P50+ with pair counts.
    """
    numeric = frame.loc[~frame["is_mt"], "age_num"].dropna().to_numpy(dtype=float)
    n_mt = int(frame["is_mt"].sum())
    bin_w = 5.0

    ax.set_title(panel_title, fontsize=10, color="0.25")
    ax.set_xlabel("Age [postnatal days]")
    ax.set_ylabel("Number of pairs")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if not (len(numeric) or n_mt):
        ax.text(0.5, 0.5, "no ages", ha="center", va="center", color="0.4")
        _style_pub_ax(ax)
        return

    if len(numeric):
        a_min = float(np.floor(np.min(numeric) / bin_w) * bin_w)
        max_age = float(np.max(numeric))
        vmin = float(np.min(numeric))
    else:
        a_min = 10.0
        vmin, max_age = 10.0, float(STRETCH_AGE_HI)
    mt_val = max_age + MT_AGE_OFFSET
    age_ticks = sorted({float(a) for a in numeric}) if len(numeric) else [a_min]
    old_ages = [a for a in age_ticks if a > float(STRETCH_AGE_HI)]
    n_old = len(old_ages)

    def _u_young(a: float) -> float:
        a_c = float(np.clip(float(a), vmin, float(STRETCH_AGE_HI)))
        return age_to_unit(a_c, False, vmin, max_age, mt_val)

    u_tail0 = _u_young(float(STRETCH_AGE_HI))
    span_young = max(float(STRETCH_AGE_HI) - vmin, bin_w)
    w_ref = float(STRETCH_FRAC) * bin_w / span_young
    slot = w_ref
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
        a = float(a)
        if a <= float(STRETCH_AGE_HI):
            return _u_young(a)
        if a in old_centers:
            return old_centers[a]
        if old_ages:
            nearest = min(old_ages, key=lambda x: abs(x - a))
            return old_centers[nearest]
        return u_tail1

    # Pair counts per age band (for labels above the plot)
    band_ns = []
    for label, lo, hi, include_mt, _c in AGE_ASYM_BINS:
        band_ns.append(int(_age_bin_mask(frame, lo, hi, include_mt).sum()))

    band_specs = [
        (10.0, 32.5, "#FB8C00", 0.12, "P10–32", False, band_ns[0]),
        (32.5, 50.0, "#8E24AA", 0.10, "P33–50", False, band_ns[1]),
        (50.0, max_age, "#2E0854", 0.08, "P50+", True, band_ns[2]),
    ]
    for lo_b, hi_b, col_b, alpha_b, lab_b, is_old, n_band in band_specs:
        if is_old:
            if not old_ages and max_age <= float(STRETCH_AGE_HI) and n_mt == 0:
                continue
            u_lo = u_tail0
            u_hi = u_tail1 if (old_ages or n_mt) else _u_young(max_age)
        else:
            lo_c = max(lo_b, a_min)
            hi_c = min(hi_b, float(STRETCH_AGE_HI), max_age)
            if hi_c <= lo_c:
                continue
            u_lo, u_hi = _u_young(lo_c), _u_young(hi_c)
        if u_hi <= u_lo:
            continue
        ax.axvspan(u_lo, u_hi, color=col_b, alpha=alpha_b, zorder=0, lw=0)
        ax.text(
            0.5 * (u_lo + u_hi),
            1.08,
            f"{lab_b}\n$n_{{ec}}$={n_band}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color=col_b,
            alpha=0.9,
            clip_on=False,
            linespacing=1.15,
        )

    if old_ages or max_age > STRETCH_AGE_HI:
        ax.axvline(u_tail0, color="0.8", ls="--", lw=0.7, zorder=1)

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
            ax.bar(
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
            ax.bar(
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
    tick_pos = [_u_hist(a) for a in majors]
    tick_lab = [_fmt_age_tick(a) for a in majors]

    u_left = _u_young(a_min) - 0.02
    u_right = u_tail1 + 0.03
    if n_mt:
        sep_x = u_tail1 + 0.015
        dam_x = u_tail1 + 0.015 + slot
        dam_w = old_bar_w
        ax.axvline(sep_x, color="0.35", ls=":", lw=0.9, zorder=1)
        ax.axvspan(
            dam_x - 0.5 * dam_w,
            dam_x + 0.5 * dam_w,
            color="#2E0854",
            alpha=0.06,
            zorder=0,
            lw=0,
        )
        ax.bar(
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

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    minor_pos = [_u_hist(a) for a in age_ticks]
    if n_mt:
        minor_pos.append(dam_x)
    if minor_pos:
        ax.set_xticks(sorted(set(minor_pos)), minor=True)
    ax.set_xlim(u_left, u_right)
    _style_pub_ax(ax)
    ax.tick_params(axis="x", which="minor", length=3.0, width=0.8, colors="0.2")
    ax.tick_params(axis="x", which="major", length=6.0, width=1.0, colors="0.2")
    print(
        f"  Age [{panel_title}]: n_numeric={len(numeric)}, "
        f"n_{MT_DISPLAY_LABEL}={n_mt}; "
        + ", ".join(f"{lab} n_ec={n}" for (lab, *_rest), n in zip(AGE_ASYM_BINS, band_ns))
    )


def plot_coverage_histograms(df: pd.DataFrame, out_dir: Path):
    """
    Separate coverage figure (2×2) — the only place Exclude?=yes is retained:
      Distance | Distance (Exclude?=yes omitted)
      Age      | Age (Exclude?=yes omitted)
    Left column may include pairs without Gj (coverage only). All other figures
    use analysis_pool / included_pairs and require filled Gj + no Exclude.
    """
    frame_all = _coverage_frame(df, include_excluded=True)
    frame_keep = _coverage_frame(df, include_excluded=False)

    def _dists(frame: pd.DataFrame) -> np.ndarray:
        if "Distance_um" not in frame.columns:
            return np.array([])
        return frame.loc[frame["Distance_um"].notna(), "Distance_um"].to_numpy(
            dtype=float
        )

    d_all, d_keep = _dists(frame_all), _dists(frame_keep)
    has_age_all = bool((~frame_all["is_mt"] & frame_all["age_num"].notna()).any()) or bool(
        frame_all["is_mt"].any()
    )
    has_age_keep = bool(
        (~frame_keep["is_mt"] & frame_keep["age_num"].notna()).any()
    ) or bool(frame_keep["is_mt"].any())

    if len(d_all) == 0 and len(d_keep) == 0 and not has_age_all and not has_age_keep:
        print("Skipping coverage histograms: no distance/age values.")
        return

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
        print("\nCoverage histograms (Distance / Age × Exclude on/off):")
        _draw_distance_hist(
            axes[0, 0], d_all, r"Distance (all pairs, incl. Exclude?=yes)"
        )
        _draw_distance_hist(
            axes[0, 1], d_keep, r"Distance (Exclude?=yes omitted)"
        )
        _draw_age_coverage_hist(
            axes[1, 0], frame_all, r"Age (all pairs, incl. Exclude?=yes)"
        )
        _draw_age_coverage_hist(
            axes[1, 1], frame_keep, r"Age (Exclude?=yes omitted)"
        )
        fig.suptitle(
            r"Sample coverage: inter-soma distance and postnatal age"
            "\n"
            r"left: all recorded pairs; right: Exclude?=yes omitted; "
            r"age-band labels show $n_{ec}$",
            fontsize=11,
            color="0.15",
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
    sub = analysis_pool(df)
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
            rf"$AI=(max-min)/(max+min)$; dashed: identity; {POOL_NOTE}",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "08_AI_Gj_CC_vs_AI_Rin.png")


# ---------------------------------------------------------------------------
# AI vs pair-mean strength (matched parameter) — exploratory
# ---------------------------------------------------------------------------
def plot_ai_vs_pair_mean(df: pd.DataFrame, out_dir: Path):
    """
    Exploratory: does stronger coupling go with lower asymmetry?
    AI_Gj vs Gj_pair_mean and AI_CC vs CC_pair_mean (matched metrics).
    Note: AI and pair mean are both derived from {x12, x21}, so they are not
    independent — weak pairs can inflate AI under absolute noise. Treat as
    descriptive; prefer cross-parameter checks for inference.
    Also writes fixed-width bin companion figure 12b.
    """
    sub = analysis_pool(df)
    need = ["AI_Gj", "Gj_pair_mean", "AI_CC", "CC_pair_mean"]
    if any(c not in sub.columns for c in need):
        print("Skipping AI vs pair-mean strength: missing AI / pair-mean columns.")
        return

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        (
            "Gj_pair_mean",
            "AI_Gj",
            rf"{PARAM_MATH['Gj']} pair mean [nS]",
            r"AI $G_{j}$",
        ),
        (
            "CC_pair_mean",
            "AI_CC",
            rf"{PARAM_MATH['CC']} pair mean",
            r"AI $CC$",
        ),
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        print("\nAI vs pair-mean strength (matched; Spearman):")
        for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, panels):
            d = sub[sub[xcol].notna() & sub[ycol].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            ax.scatter(
                d[xcol],
                d[ycol],
                c=colors,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            if len(d) >= 3:
                rho, p = stats.spearmanr(d[xcol], d[ycol])
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
                print(f"  {ycol} vs {xcol}: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
            else:
                print(f"  {ycol} vs {xcol}: n={len(d)} (too few)")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(xlabel)
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
            "Coupling asymmetry vs pair-mean strength (matched)"
            "\n"
            + AI_FORMULA
            + r"; exploratory — AI & mean share $\{x_{12},x_{21}\}$ "
            r"(not independent); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "12_AI_vs_pair_mean_strength.png")

    plot_ai_by_strength_bins(df, out_dir)


def plot_ai_by_strength_bins(df: pd.DataFrame, out_dir: Path):
    """
    AI by fixed-width pair-mean strength bins (Gj step GJ_BIN_WIDTH nS;
    CC step CC_BIN_WIDTH). Companion to the scatter; same non-independence caveat.
    """
    sub = analysis_pool(df)
    need = ["AI_Gj", "Gj_pair_mean", "AI_CC", "CC_pair_mean"]
    if any(c not in sub.columns for c in need):
        print("Skipping AI by strength bins: missing columns.")
        return

    panels = [
        ("Gj_pair_mean", "AI_Gj", r"AI $G_{j}$", "gj", GJ_BIN_WIDTH),
        ("CC_pair_mean", "AI_CC", r"AI $CC$", "cc", CC_BIN_WIDTH),
    ]

    # Width chosen after scanning both panels' bin counts
    meta_lens = []
    for xcol, _y, _yl, _k, w in panels:
        d0 = sub[sub[xcol].notna()]
        _b, m = assign_width_bins(d0[xcol], w)
        meta_lens.append(len(m))
    fig_w = max(10.5, 1.15 * max(meta_lens, default=3) + 2.5)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.8), constrained_layout=True)
        print(
            f"\nAI by pair-mean strength bins "
            f"(Gj width={GJ_BIN_WIDTH:g} nS, CC width={CC_BIN_WIDTH:g}):"
        )
        for ax, (xcol, ycol, ylabel, edge_kind, width) in zip(axes, panels):
            d = sub[sub[xcol].notna() & sub[ycol].notna()].copy()
            bins, meta = assign_width_bins(d[xcol], width)
            if not meta:
                ax.set_visible(False)
                print(f"  {ycol}: could not form bins")
                continue
            d = d.copy()
            d["_str_bin"] = bins
            samples = []
            for name, _lo, _hi, _n in meta:
                vals = (
                    d.loc[d["_str_bin"] == name, ycol].dropna().to_numpy(dtype=float)
                )
                samples.append(vals)
                med = float(np.median(vals)) if len(vals) else float("nan")
                p75 = float(np.quantile(vals, 0.75)) if len(vals) else float("nan")
                print(
                    f"  {ycol} {name}: n={len(vals)}, median={med:.3g}, P75={p75:.3g}"
                )
            _draw_width_bin_panel(
                ax,
                samples=samples,
                meta=meta,
                ylabel=ylabel,
                edge_kind=edge_kind,
                seed_key=f"str-{ycol}",
                y_lim=(0, 1.05),
            )
            if edge_kind == "gj":
                ax.set_xlabel(
                    rf"{PARAM_MATH['Gj']} pair-mean bin [nS] (width={width:g})"
                )
            else:
                ax.set_xlabel(rf"{PARAM_MATH['CC']} pair-mean bin (width={width:g})")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="crimson",
                lw=0,
                markersize=6,
                label="mean ± SEM",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="royalblue",
                lw=0,
                markersize=5,
                label="median",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(
            "AI by pair-mean strength bins (matched)"
            "\n"
            + AI_FORMULA
            + rf"; fixed width $G_j$={GJ_BIN_WIDTH:g} nS, $CC$={CC_BIN_WIDTH:g}; "
            r"exploratory — AI & mean share $\{x_{12},x_{21}\}$; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "12b_AI_by_strength_bins.png")


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

    sub = analysis_pool(df)
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
            r"shaded: same sign (larger coupling toward higher-$R_{in}$ cell); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "09_signed_dGj_dCC_vs_dRin.png")


# ---------------------------------------------------------------------------
# Figure 5: AI histograms
# ---------------------------------------------------------------------------
def plot_ai_histograms(df: pd.DataFrame, out_dir: Path):
    sub = analysis_pool(df)
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
            f"AI (asymmetry index)\n{AI_FORMULA}; dotted reference: AI≈0.33 (~2×); {POOL_NOTE}",
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
    sub = analysis_pool(df)
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
    sub = analysis_pool(df)
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


# ---------------------------------------------------------------------------
# Spike transmission: CCa (amp ratio) and delays
# ---------------------------------------------------------------------------
def _nan_to_zero(v) -> float:
    return 0.0 if pd.isna(v) else float(v)


def _pair_mean_zero_fill(a, b) -> float:
    """(a+b)/2 with missing side as 0; NaN if both missing."""
    if pd.isna(a) and pd.isna(b):
        return float("nan")
    return (_nan_to_zero(a) + _nan_to_zero(b)) / 2.0


def _rug_y_for_ax(y_vals: np.ndarray) -> float:
    finite = y_vals[np.isfinite(y_vals)]
    if len(finite) == 0:
        return -0.02
    y0, y1 = float(np.min(finite)), float(np.max(finite))
    span = max(y1 - y0, abs(y1) * 0.05, 1e-6)
    return y0 - 0.06 * span


def _plot_paired_cca_vs_coupling(
    ax,
    sub: pd.DataFrame,
    *,
    x12: str,
    x21: str,
    y12: str,
    y21: str,
    vmin: float,
    max_age: float,
    mt_val: float,
    xlabel: str,
    ylabel: str,
):
    """Directed CCa points + × rug for missing CCa (no pair-joining lines)."""
    colors, _, _ = age_colors(
        sub["age_num"].to_numpy(), sub["is_mt"].to_numpy(), vmin, max_age, mt_val
    )
    y_present = []
    for _, r in sub.iterrows():
        for yc in (y12, y21):
            if yc in r.index and pd.notna(r[yc]):
                y_present.append(float(r[yc]))
    y_rug = _rug_y_for_ax(np.asarray(y_present, dtype=float))

    n_both = n_one = n_none = 0
    for col, (_, r) in zip(colors, sub.iterrows()):
        xa, xb = r.get(x12, np.nan), r.get(x21, np.nan)
        ya, yb = r.get(y12, np.nan), r.get(y21, np.nan)
        for xv, yv in ((xa, ya), (xb, yb)):
            if pd.isna(xv):
                continue
            if pd.notna(yv):
                ax.scatter(
                    [xv], [yv], c=[col], s=POINT_SIZE * 0.7, alpha=0.55,
                    edgecolors="white", linewidths=0.3, zorder=3,
                )
            else:
                ax.scatter(
                    [xv], [y_rug], marker="x", c="0.45", s=28, alpha=0.85,
                    linewidths=1.0, zorder=2, clip_on=False,
                )
        has_a = pd.notna(ya)
        has_b = pd.notna(yb)
        if has_a and has_b:
            n_both += 1
        elif has_a or has_b:
            n_one += 1
        else:
            n_none += 1

    ax.text(
        0.04, 0.96,
        f"both $CC_a$ n={n_both}\none $CC_a$ n={n_one}\nnone n={n_none}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_pub_ax(ax)
    return n_both, n_one, n_none


def _scatter_pair_mean_panel(
    ax,
    sub: pd.DataFrame,
    *,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    vmin: float,
    max_age: float,
    mt_val: float,
    xlabel: str,
    ylabel: str,
):
    m = np.isfinite(x_mean) & np.isfinite(y_mean)
    d = sub.loc[m]
    if len(d) == 0:
        ax.set_visible(False)
        return
    colors, _, _ = age_colors(
        d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
    )
    ax.scatter(
        x_mean[m], y_mean[m], c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
        edgecolors="white", linewidths=0.35, zorder=2,
    )
    if int(m.sum()) >= 3:
        rho, p = stats.spearmanr(x_mean[m], y_mean[m])
        ax.text(
            0.04, 0.96,
            f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={int(m.sum())}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_pub_ax(ax)


def plot_cca_vs_cc_gj(df: pd.DataFrame, out_dir: Path):
    """
    (1) Directed CCa vs CC / Gj (+ × for missing CCa; no pair lines).
    (2) Pair-mean CCa vs mean CC/Gj — both sides present only (Spearman).
    (3) Pair-mean with missing CCa filled as 0 (Spearman).
    """
    if not _spike_cols_present(df):
        print("Skipping CCa vs CC/Gj: spike/spikelet columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    cca12, cca21 = PARAM_DIRS["CCa"]
    panels = [
        ("CC_12", "CC_21", "CC_pair_mean", PARAM_YLABELS["CC"]),
        ("Gj_12", "Gj_21", "Gj_pair_mean", PARAM_YLABELS["Gj"]),
    ]

    # --- directed (no lines) ---
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        print("\nCCa vs CC / Gj (directed):")
        for ax, (x12, x21, _mc, xlab) in zip(axes, panels):
            _plot_paired_cca_vs_coupling(
                ax, sub, x12=x12, x21=x21, y12=cca12, y21=cca21,
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                xlabel=xlab, ylabel=PARAM_YLABELS["CCa"],
            )
        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.45",
                markeredgecolor="white", markersize=7, label=r"directed with $CC_a$",
            ),
            Line2D(
                [0], [0], marker="x", color="0.45", lw=0, markersize=7,
                label=r"no $CC_a$ (×)",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(
            r"$CC_a$ vs $CC$ / $G_j$ (directed)"
            "\n"
            r"× = missing $CC_a$ for that direction; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "15_CCa_vs_CC_Gj.png")

    # --- pair means: both-only and zero-fill ---
    y_both = sub["CCa_pair_mean"].to_numpy(dtype=float) if "CCa_pair_mean" in sub.columns else np.full(len(sub), np.nan)
    y_zero = np.array(
        [_pair_mean_zero_fill(a, b) for a, b in zip(sub[cca12], sub[cca21])],
        dtype=float,
    )
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
        print("\nCCa vs CC / Gj (pair means):")
        for c, (_x12, _x21, mean_col, xlab) in enumerate(panels):
            x_mean = sub[mean_col].to_numpy(dtype=float) if mean_col in sub.columns else np.full(len(sub), np.nan)
            _scatter_pair_mean_panel(
                axes[0, c], sub, x_mean=x_mean, y_mean=y_both,
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                xlabel=xlab + r" pair mean", ylabel=r"$CC_a$ pair mean (both sides)",
            )
            print(
                f"  both-only vs {mean_col}: n="
                f"{int((np.isfinite(x_mean) & np.isfinite(y_both)).sum())}"
            )
            _scatter_pair_mean_panel(
                axes[1, c], sub, x_mean=x_mean, y_mean=y_zero,
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                xlabel=xlab + r" pair mean",
                ylabel=r"$CC_a$ mean (missing→0)",
            )
            print(
                f"  zero-fill vs {mean_col}: n="
                f"{int((np.isfinite(x_mean) & np.isfinite(y_zero)).sum())}"
            )
        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        axes[0, 0].set_title("Both directions have $CC_a$", fontsize=10, color="0.25")
        axes[1, 0].set_title(r"Missing $CC_a$ counted as 0", fontsize=10, color="0.25")
        fig.suptitle(
            r"$CC_a$ vs $CC$ / $G_j$ — pair means"
            "\n"
            r"top: mean only if both $CC_a$; bottom: missing side = 0; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "15b_CCa_vs_CC_Gj_pair_mean.png")


def plot_delays_vs_cca_cc_gj(df: pd.DataFrame, out_dir: Path):
    """
    Onset/peak delay vs CCa, CC, Gj — directed, then pair-mean companion.
    On CC/Gj directed panels: × where CCa is missing.
    """
    if not _spike_cols_present(df):
        print("Skipping delays vs CCa/CC/Gj: spike columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    cca12, cca21 = PARAM_DIRS["CCa"]
    on12, on21 = PARAM_DIRS["DelayOnset"]
    pk12, pk21 = PARAM_DIRS["DelayPeak"]
    row_specs = [
        (on12, on21, "DelayOnset_pair_mean", PARAM_YLABELS["DelayOnset"]),
        (pk12, pk21, "DelayPeak_pair_mean", PARAM_YLABELS["DelayPeak"]),
    ]
    col_specs = [
        (cca12, cca21, "CCa_pair_mean", PARAM_YLABELS["CCa"], False),
        ("CC_12", "CC_21", "CC_pair_mean", PARAM_YLABELS["CC"], True),
        ("Gj_12", "Gj_21", "Gj_pair_mean", PARAM_YLABELS["Gj"], True),
    ]
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
        print("\nDelays vs CCa / CC / Gj (directed):")
        for r, (y12, y21, _ym, ylab) in enumerate(row_specs):
            for c, (x12, x21, _xm, xlab, mark_miss) in enumerate(col_specs):
                ax = axes[r, c]
                both, _miss_y = directed_xy(sub, x12, x21, y12, y21)
                if mark_miss:
                    _both_cca, missing_cca = directed_xy(sub, x12, x21, cca12, cca21)
                else:
                    missing_cca = pd.DataFrame()
                print(
                    f"  {ylab} vs {xlab}: n={len(both)}"
                    + (f", no CCa n={len(missing_cca)}" if mark_miss else "")
                )
                _scatter_directed_with_missing_rug(
                    ax, both, missing_cca,
                    vmin=vmin, max_age=max_age, mt_val=mt_val,
                    mark_missing=mark_miss, y_label=ylab, x_label=xlab,
                )
        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Onset / peak delay vs $CC_a$, $CC$, $G_j$ (directed)"
            "\n"
            r"× on $CC$/$G_j$ panels = no measurable $CC_a$; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "16_delays_vs_CCa_CC_Gj.png")

    # pair-mean companion (both sides required for each mean)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
        print("\nDelays vs CCa / CC / Gj (pair means, both sides):")
        for r, (_y12, _y21, ymean, ylab) in enumerate(row_specs):
            for c, (_x12, _x21, xmean, xlab, _m) in enumerate(col_specs):
                ax = axes[r, c]
                if ymean not in sub.columns or xmean not in sub.columns:
                    ax.set_visible(False)
                    continue
                ym = sub[ymean].to_numpy(dtype=float)
                xm = sub[xmean].to_numpy(dtype=float)
                _scatter_pair_mean_panel(
                    ax, sub, x_mean=xm, y_mean=ym,
                    vmin=vmin, max_age=max_age, mt_val=mt_val,
                    xlabel=xlab + r" pair mean", ylabel=ylab + r" pair mean",
                )
                print(
                    f"  {ymean} vs {xmean}: n="
                    f"{int((np.isfinite(xm) & np.isfinite(ym)).sum())}"
                )
        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Onset / peak delay vs $CC_a$, $CC$, $G_j$ — pair means"
            "\n"
            r"means require both directions present; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "16b_delays_vs_CCa_CC_Gj_pair_mean.png")


def plot_cca_delays_vs_distance(df: pd.DataFrame, out_dir: Path):
    """Directed CCa and delays vs Distance (both sides + pair mean when both)."""
    if not _spike_cols_present(df):
        print("Skipping CCa/delays vs Distance: spike columns not found.")
        return
    if "Distance_um" not in df.columns:
        print("Skipping CCa/delays vs Distance: no Distance column.")
        return
    sub = analysis_pool(df)
    sub = sub[sub["Distance_um"].notna()].copy()
    if len(sub) == 0:
        print("Skipping CCa/delays vs Distance: no Distance values.")
        return
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        (*PARAM_DIRS["CCa"], "CCa_pair_mean", PARAM_YLABELS["CCa"]),
        (*PARAM_DIRS["DelayOnset"], "DelayOnset_pair_mean", PARAM_YLABELS["DelayOnset"]),
        (*PARAM_DIRS["DelayPeak"], "DelayPeak_pair_mean", PARAM_YLABELS["DelayPeak"]),
    ]
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
        print("\nCCa / delays vs Distance:")
        for ax, (c1, c2, mean_col, ylabel) in zip(axes, panels):
            d = sub.copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            x = d["Distance_um"].to_numpy(dtype=float)
            y1 = d[c1].to_numpy(dtype=float) if c1 in d.columns else np.full(len(d), np.nan)
            y2 = d[c2].to_numpy(dtype=float) if c2 in d.columns else np.full(len(d), np.nan)
            ym = (
                d[mean_col].to_numpy(dtype=float)
                if mean_col in d.columns
                else np.full(len(d), np.nan)
            )
            for xi, a, b, col in zip(x, y1, y2, colors):
                if np.isfinite(a) and np.isfinite(b):
                    ax.plot([xi, xi], [a, b], color=col, lw=0.6, alpha=0.35, zorder=1)
                for yv in (a, b):
                    if np.isfinite(yv):
                        ax.scatter(
                            [xi], [yv], c=[col], s=POINT_SIZE * 0.55, alpha=0.45,
                            edgecolors="white", linewidths=0.3, zorder=2,
                        )
            m_ok = np.isfinite(ym) & np.isfinite(x)
            if m_ok.any():
                ax.scatter(
                    x[m_ok], ym[m_ok], c=np.asarray(colors)[m_ok],
                    s=POINT_SIZE * 1.15, alpha=0.9, edgecolors="0.15",
                    linewidths=0.5, zorder=3,
                )
            if int(m_ok.sum()) >= 3:
                rho, p = stats.spearmanr(x[m_ok], ym[m_ok])
                ax.text(
                    0.04, 0.96,
                    f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={int(m_ok.sum())}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
                )
                print(f"  {ylabel}: ρ={rho:.3f}, {format_p(p)}, n_mean={int(m_ok.sum())}")
            else:
                print(f"  {ylabel}: n_mean={int(m_ok.sum())} (too few)")
            ax.set_xlabel(r"Distance [$\mu$m]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.55",
                markeredgecolor="white", markersize=5, alpha=0.7,
                label=r"directed $x_{12}$, $x_{21}$",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.35",
                markeredgecolor="0.15", markersize=7,
                label=r"pair mean (both sides)",
            ),
        ]
        fig.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.55),
            frameon=False, fontsize=9,
        )
        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"$CC_a$ and delays vs inter-soma distance"
            "\n"
            r"small: directed; large: pair mean when both sides present; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "17_CCa_delays_vs_distance.png")


def plot_delay_spread(df: pd.DataFrame, out_dir: Path):
    """Alias: CCa + delays vs age (gray points) — fig 18/19."""
    plot_cca_delays_vs_age(df, out_dir)


def plot_spike_transmission_figures(df: pd.DataFrame, out_dir: Path):
    """Run spike/spikelet CCa & delay figures (15–18 + mean companions)."""
    plot_cca_vs_cc_gj(df, out_dir)
    plot_delays_vs_cca_cc_gj(df, out_dir)
    plot_cca_delays_vs_distance(df, out_dir)
    plot_cca_delays_vs_age(df, out_dir)


def _cca_columns_present(df: pd.DataFrame) -> bool:
    c1, c2 = PARAM_DIRS["CCa"]
    return c1 in df.columns and c2 in df.columns


def directed_cca_vs_rin_recv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CCa vs receiver Rin:
      1→2: CCa_12 vs Rin_2
      2→1: CCa_21 vs Rin_1
    Returns (both, missing_cca) with x=Rin_recv, y=CCa.
    """
    cca12, cca21 = PARAM_DIRS["CCa"]
    rows_both, rows_miss = [], []
    specs = (
        (cca12, "Rin_2", "12"),
        (cca21, "Rin_1", "21"),
    )
    for yc, xc, side in specs:
        if yc not in df.columns or xc not in df.columns:
            continue
        for _, r in df.iterrows():
            xv, yv = r[xc], r[yc]
            base = {
                "x": xv,
                "y": yv,
                "age_num": r.get("age_num", np.nan),
                "is_mt": bool(r.get("is_mt", False)),
                "side": side,
            }
            if pd.notna(xv) and pd.notna(yv):
                rows_both.append(base)
            elif pd.notna(xv) and pd.isna(yv):
                rows_miss.append(base)
    return pd.DataFrame(rows_both), pd.DataFrame(rows_miss)


def _coverage_grouped_bars(
    ax,
    *,
    labels: list[str],
    n_pairs: list[int],
    n_pairs_cca: list[int],
    n_dir: list[int] | None = None,
    n_dir_cca: list[int] | None = None,
    xlabel: str,
):
    """Grouped bars: pairs vs pairs with CCa; optional directed counts as text."""
    x = np.arange(len(labels), dtype=float)
    w = 0.36
    ax.bar(
        x - w / 2,
        n_pairs,
        w,
        color="0.72",
        edgecolor="white",
        label="pairs in bin",
        zorder=2,
    )
    ax.bar(
        x + w / 2,
        n_pairs_cca,
        w,
        color="0.28",
        edgecolor="white",
        label=r"pairs with $CC_a$",
        zorder=2,
    )
    y_top = max(n_pairs + n_pairs_cca + [1])
    for i, (a, b) in enumerate(zip(n_pairs, n_pairs_cca)):
        pct = (100.0 * b / a) if a else 0.0
        ax.text(
            i,
            max(a, b) + 0.04 * y_top,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.25",
        )
        if n_dir is not None and n_dir_cca is not None:
            ax.annotate(
                f"dir {n_dir_cca[i]}/{n_dir[i]}",
                xy=(i, 0),
                xytext=(0, -28),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.5,
                color="0.4",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7 if len(labels) > 6 else 8)
    if len(labels) > 6:
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
    ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, y_top * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_pub_ax(ax)


def _pct_points(ax, *, labels: list[str], pct: list[float], xlabel: str, ylabel: str):
    """Coverage % as independent markers (no connecting line — bins are not a time series)."""
    x = np.arange(len(labels), dtype=float)
    ax.scatter(x, pct, s=46, c="0.25", zorder=2, edgecolors="white", linewidths=0.4)
    for i, p in enumerate(pct):
        ax.text(
            i, min(p + 3.5, 108), f"{p:.0f}%",
            ha="center", va="bottom", fontsize=7.5, color="0.25",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7 if len(labels) > 6 else 8)
    if len(labels) > 6:
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
    ax.set_ylim(0, 110)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    _style_pub_ax(ax)


def _age_axis_colorbar(fig, axes, *, vmin, max_age, mt_val, age_ticks, has_mt):
    """Shared age colour scale (x conveys age; points stay gray)."""
    cbar = add_age_colorbar(
        fig, list(np.atleast_1d(axes).ravel()),
        vmin=vmin, max_age=max_age, mt_val=mt_val,
        age_ticks=age_ticks, has_mt=has_mt,
    )
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
    cbar.set_label("Age [postnatal days]", fontsize=9)
    return cbar


def _scatter_gray_vs_warped_age(
    ax,
    d: pd.DataFrame,
    *,
    c1: str,
    c2: str,
    mean_col: str | None,
    vmin: float,
    max_age: float,
    mt_val: float,
    age_ticks: list[float],
    has_mt: bool,
    ylabel: str,
):
    """Directed + optional pair-mean vs warped age; gray markers (age only on x / colourbar)."""
    xw = np.array(
        [
            _warped_age_x(a, bool(mt), vmin, max_age, mt_val)
            for a, mt in zip(d["age_num"], d["is_mt"])
        ],
        dtype=float,
    )
    y1 = d[c1].to_numpy(dtype=float) if c1 in d.columns else np.full(len(d), np.nan)
    y2 = d[c2].to_numpy(dtype=float) if c2 in d.columns else np.full(len(d), np.nan)
    ym = (
        d[mean_col].to_numpy(dtype=float)
        if mean_col and mean_col in d.columns
        else np.full(len(d), np.nan)
    )
    gray = "0.45"
    for xi, a, b in zip(xw, y1, y2):
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([xi, xi], [a, b], color=gray, lw=0.6, alpha=0.3, zorder=1)
        for yv in (a, b):
            if np.isfinite(yv):
                ax.scatter(
                    [xi], [yv], c=gray, s=POINT_SIZE * 0.55, alpha=0.45,
                    edgecolors="white", linewidths=0.3, zorder=2,
                )
    m_ok = np.isfinite(ym)
    if m_ok.any():
        ax.scatter(
            xw[m_ok], ym[m_ok], c="0.28", s=POINT_SIZE * 1.1, alpha=0.85,
            edgecolors="0.15", linewidths=0.45, zorder=3,
        )
    ax.set_xlabel("Age [postnatal days]")
    ax.set_ylabel(ylabel)
    _set_warped_age_ticks(ax, vmin, max_age, mt_val, age_ticks, has_mt)
    _style_pub_ax(ax)


def plot_cca_delays_vs_age(df: pd.DataFrame, out_dir: Path):
    """
    CCa + onset/peak delay vs warped age on one figure (3 panels).
    Points gray; age encoded only by x-axis (+ shared age colourbar).
    """
    if not _spike_cols_present(df):
        print("Skipping CCa/delays vs age: spike columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        (*PARAM_DIRS["CCa"], "CCa_pair_mean", PARAM_YLABELS["CCa"]),
        (*PARAM_DIRS["DelayOnset"], "DelayOnset_pair_mean", PARAM_YLABELS["DelayOnset"]),
        (*PARAM_DIRS["DelayPeak"], "DelayPeak_pair_mean", PARAM_YLABELS["DelayPeak"]),
    ]
    print("\nCCa / delays vs age (gray points; age on x):")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
        for ax, (c1, c2, mean_col, ylab) in zip(axes, panels):
            d = sub[sub[c1].notna() | sub[c2].notna()].copy()
            print(f"  {ylab}: n_pairs={len(d)}")
            if len(d) == 0:
                ax.set_visible(False)
                continue
            _scatter_gray_vs_warped_age(
                ax, d, c1=c1, c2=c2, mean_col=mean_col,
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                age_ticks=age_ticks, has_mt=has_mt, ylabel=ylab,
            )
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.45",
                markeredgecolor="white", markersize=5, label="directed",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.28",
                markeredgecolor="0.15", markersize=7, label="pair mean",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        _age_axis_colorbar(
            fig, axes, vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        fig.suptitle(
            r"$CC_a$ and delays vs age"
            "\n"
            r"gray points (age = x-axis / colourbar only; no point colour); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "18_CCa_delays_vs_age.png")
        # Alias for catalog / prior 19 naming (savefig closes the figure)
        alias = out_dir / "19_CCa_vs_age.png"
        src = out_dir / "18_CCa_delays_vs_age.png"
        if src.is_file():
            alias.write_bytes(src.read_bytes())
            print(f"{'Overwrote' if alias.is_file() else 'Saved'}: {alias}")


def plot_cca_vs_age(df: pd.DataFrame, out_dir: Path):
    """
    Combined CCa+delays vs age (gray points), plus CCa by age-group KW companion.
    """
    plot_cca_delays_vs_age(df, out_dir)
    if not _cca_columns_present(df):
        return
    sub = analysis_pool(df)
    c1, c2 = PARAM_DIRS["CCa"]
    # Age-group summary with KW (directed CCa pooled within group)
    group_labels = [b[0] for b in AGE_ASYM_BINS]
    samples = []
    print("\nCCa by age group (directed pool):")
    for label, lo, hi, include_mt, color in AGE_ASYM_BINS:
        g = sub.loc[_age_bin_mask(sub, lo, hi, include_mt)]
        vals = pd.concat([g[c1], g[c2]], ignore_index=True).dropna().to_numpy(dtype=float)
        samples.append(vals)
        print(f"  {label}: n_dir={len(vals)}, median={np.median(vals) if len(vals) else float('nan'):.3g}")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        for i, ((label, _lo, _hi, _mt, color), vals) in enumerate(zip(AGE_ASYM_BINS, samples)):
            if len(vals) == 0:
                continue
            xpos = i + 1
            jitter = np.random.default_rng(abs(hash("cca-age" + label)) % (2**32)).normal(
                0, 0.06, len(vals)
            )
            ax.scatter(
                np.full(len(vals), xpos) + jitter, vals, c=color,
                s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors="white", linewidths=0.35, zorder=2,
            )
            ax.errorbar(
                [xpos], [float(np.mean(vals))],
                yerr=[float(stats.sem(vals)) if len(vals) > 1 else 0.0],
                fmt="o", color="crimson", markersize=6, capsize=3, zorder=3,
            )
            ax.plot([xpos], [float(np.median(vals))], marker="D", color="royalblue", markersize=5, zorder=4)
        nonempty = [s for s in samples if len(s) >= 3]
        if len(nonempty) >= 2:
            _h, p_kw = stats.kruskal(*nonempty)
            ax.text(
                0.98, 0.96, f"KW {format_p(p_kw)}\n(bins n≥3)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35",
            )
            print(f"  KW {format_p(p_kw)}")
        ax.set_xticks(range(1, len(group_labels) + 1))
        ax.set_xticklabels(
            [f"{lab}\nn={len(s)}" for lab, s in zip(group_labels, samples)], fontsize=8
        )
        ax.set_ylabel(PARAM_YLABELS["CCa"])
        ax.set_xlabel("Age group")
        _style_pub_ax(ax)
        handles = [
            Line2D([0], [0], marker="o", color="crimson", lw=0, markersize=6, label="mean ± SEM"),
            Line2D([0], [0], marker="D", color="royalblue", lw=0, markersize=5, label="median"),
        ]
        fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
        fig.suptitle(
            r"$CC_a$ by age group (directed values)"
            "\n"
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "19b_CCa_by_age_group.png")


def plot_cca_coverage(df: pd.DataFrame, out_dir: Path):
    """
    CCa coverage counts + % (points):
      - age P10–32 / P33–50 / P50+: pairs with any CCa
      - Distance 20 µm: counts + %
      - directed CC bins width 0.025: % of directions with CCa
    """
    if not _cca_columns_present(df):
        print("Skipping CCa coverage: amp_ratio columns not found.")
        return
    sub = analysis_pool(df)
    cca12, cca21 = PARAM_DIRS["CCa"]
    print("\nCCa coverage:")

    age_lab, age_n, age_cca, age_pct = [], [], [], []
    for label, lo, hi, include_mt, _c in AGE_ASYM_BINS:
        g = sub.loc[_age_bin_mask(sub, lo, hi, include_mt)]
        nt = int(len(g))
        nc = int(g["has_cca_any"].sum()) if "has_cca_any" in g.columns else 0
        age_lab.append(label)
        age_n.append(nt)
        age_cca.append(nc)
        age_pct.append((100.0 * nc / nt) if nt else 0.0)
        print(f"  age {label}: pairs {nt}, with CCa {nc} ({age_pct[-1]:.0f}%)")

    dist_lab, dist_n, dist_cca, dist_pct = [], [], [], []
    if "Distance_um" in sub.columns and sub["Distance_um"].notna().any():
        dsub = sub[sub["Distance_um"].notna()].copy()
        bins, meta = assign_width_bins(dsub["Distance_um"], DISTANCE_BIN_WIDTH)
        dsub["_bin"] = bins
        for name, lo, hi, _n in meta:
            g = dsub.loc[dsub["_bin"] == name]
            nt, nc = int(len(g)), int(g["has_cca_any"].sum())
            dist_lab.append(f"{lo:.0f}–{hi:.0f}")
            dist_n.append(nt)
            dist_cca.append(nc)
            dist_pct.append((100.0 * nc / nt) if nt else 0.0)
            print(f"  dist {name}: pairs {nt}, with CCa {nc} ({dist_pct[-1]:.0f}%)")

    cc_lab, cc_nd, cc_nd_cca, cc_pct = [], [], [], []
    if "CC_12" in sub.columns and "CC_21" in sub.columns:
        rows = []
        for cc_c, cca_c in (("CC_12", cca12), ("CC_21", cca21)):
            for _, r in sub.iterrows():
                if pd.isna(r[cc_c]):
                    continue
                rows.append({"cc": float(r[cc_c]), "has_cca": pd.notna(r[cca_c])})
        if rows:
            long = pd.DataFrame(rows)
            bins, meta = assign_width_bins(long["cc"], CC_BIN_WIDTH_CCA_COV)
            long["_bin"] = bins
            for name, lo, hi, _n in meta:
                g = long.loc[long["_bin"] == name]
                nd = int(len(g))
                ndc = int(g["has_cca"].sum())
                cc_lab.append(f"{lo:g}–{hi:g}")
                cc_nd.append(nd)
                cc_nd_cca.append(ndc)
                cc_pct.append((100.0 * ndc / nd) if nd else 0.0)
                print(f"  CC dir {name}: dir {nd}, with CCa {ndc} ({cc_pct[-1]:.0f}%)")

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        # Age / Distance: counts on top, % points underneath (no connecting lines)
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
        _coverage_grouped_bars(
            axes[0, 0], labels=age_lab, n_pairs=age_n, n_pairs_cca=age_cca, xlabel="Age group",
        )
        axes[0, 0].set_title("Age — counts", fontsize=10, color="0.25")
        _pct_points(
            axes[1, 0], labels=age_lab, pct=age_pct, xlabel="Age group",
            ylabel=r"% pairs with $CC_a$",
        )
        axes[1, 0].set_title("Age — %", fontsize=10, color="0.25")
        if dist_lab:
            _coverage_grouped_bars(
                axes[0, 1], labels=dist_lab, n_pairs=dist_n, n_pairs_cca=dist_cca,
                xlabel=rf"Distance [$\mu$m] (width={DISTANCE_BIN_WIDTH:g})",
            )
            axes[0, 1].set_title("Distance — counts", fontsize=10, color="0.25")
            _pct_points(
                axes[1, 1], labels=dist_lab, pct=dist_pct,
                xlabel=rf"Distance [$\mu$m] (width={DISTANCE_BIN_WIDTH:g})",
                ylabel=r"% pairs with $CC_a$",
            )
            axes[1, 1].set_title("Distance — %", fontsize=10, color="0.25")
        else:
            axes[0, 1].set_visible(False)
            axes[1, 1].set_visible(False)
        axes[0, 0].legend(
            handles=[
                Line2D([0], [0], marker="s", color="0.72", lw=0, markersize=8, label="pairs in bin"),
                Line2D(
                    [0], [0], marker="s", color="0.28", lw=0, markersize=8,
                    label=r"pairs with any $CC_a$",
                ),
            ],
            loc="upper right", frameon=True, fancybox=False, framealpha=0.92,
            edgecolor="0.85", fontsize=7.5, borderpad=0.35,
        )
        axes[1, 0].legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="0.25",
                    markeredgecolor="white", markersize=6, label=r"% per bin (not a trend)",
                ),
            ],
            loc="lower right", frameon=True, fancybox=False, framealpha=0.92,
            edgecolor="0.85", fontsize=7.5, borderpad=0.35,
        )
        fig.suptitle(
            r"$CC_a$ coverage — counts (top) and % (bottom)"
            "\n"
            r"% markers are independent per bin (not connected); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "20_CCa_coverage_counts.png")
        # Same combined counts+% figure under prior 20b name
        p20 = out_dir / "20_CCa_coverage_counts.png"
        p20b = out_dir / "20b_CCa_coverage_percent_age_distance.png"
        if p20.is_file():
            p20b.write_bytes(p20.read_bytes())
            print(f"Overwrote: {p20b}")

        if cc_lab:
            fig, axes = plt.subplots(
                2, 1,
                figsize=(max(8.0, 0.55 * len(cc_lab) + 2), 7.2),
                constrained_layout=True,
                sharex=True,
            )
            # directed counts as light bars + with-CCa overlay text via % panel
            x = np.arange(len(cc_lab), dtype=float)
            axes[0].bar(
                x, cc_nd, color="0.78", edgecolor="white", zorder=1, label="directions in bin",
            )
            axes[0].bar(
                x, cc_nd_cca, color="0.28", edgecolor="white", zorder=2,
                label=r"directions with $CC_a$",
            )
            axes[0].set_ylabel("directions")
            axes[0].set_title("Directed $CC$ — counts", fontsize=10, color="0.25")
            axes[0].legend(loc="upper right", frameon=False, fontsize=8)
            _style_pub_ax(axes[0])
            _pct_points(
                axes[1], labels=cc_lab, pct=cc_pct,
                xlabel=rf"Directed $CC$ bin (width={CC_BIN_WIDTH_CCA_COV:g})",
                ylabel=r"% directions with $CC_a$",
            )
            for i, (nd, ndc) in enumerate(zip(cc_nd, cc_nd_cca)):
                axes[1].annotate(
                    f"{ndc}/{nd}",
                    xy=(i, 0),
                    xytext=(0, -26),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color="0.4",
                )
            axes[1].set_title("Directed $CC$ — %", fontsize=10, color="0.25")
            fig.suptitle(
                r"$CC_a$ coverage by directed $CC$"
                "\n"
                rf"bin width={CC_BIN_WIDTH_CCA_COV:g}; % = independent markers; "
                + POOL_NOTE,
                fontsize=11,
                color="0.15",
            )
            savefig(fig, out_dir, "20c_CCa_coverage_percent_by_CC_directed.png")


def plot_cca_vs_rin_receiver(df: pd.DataFrame, out_dir: Path):
    """
    CCa vs Rin of the receiver (1→2: Rin_2; 2→1: Rin_1).
    × marks: receiver Rin for directions without measurable CCa.
    """
    if not _cca_columns_present(df):
        print("Skipping CCa vs Rin_recv: amp_ratio columns not found.")
        return
    if "Rin_1" not in df.columns or "Rin_2" not in df.columns:
        print("Skipping CCa vs Rin_recv: Rin columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    both, missing = directed_cca_vs_rin_recv(sub)
    print(
        f"\nCCa vs Rin_recv: with CCa n={len(both)}, "
        f"Rin without CCa n={len(missing)}"
    )
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        _scatter_directed_with_missing_rug(
            ax,
            both,
            missing,
            vmin=vmin,
            max_age=max_age,
            mt_val=mt_val,
            mark_missing=True,
            y_label=PARAM_YLABELS["CCa"],
            x_label=r"$R_{in,\mathrm{recv}}$ [M$\Omega$]",
        )
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.45",
                markeredgecolor="white", markersize=7, label=r"directed with $CC_a$",
            ),
            Line2D(
                [0], [0], marker="x", color="0.45", lw=0, markersize=7,
                label=r"$R_{in,recv}$ without $CC_a$",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        cbar = add_age_colorbar(
            fig, [ax], vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"$CC_a$ vs receiver $R_{in}$"
            "\n"
            r"$1\!\to\!2$: $R_{in,2}$; $2\!\to\!1$: $R_{in,1}$; "
            r"× = no measurable $CC_a$; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "21_CCa_vs_Rin_receiver.png")


def plot_cca_coverage_and_rin_figures(df: pd.DataFrame, out_dir: Path):
    """CCa vs age, coverage summaries, CCa vs Rin_recv (19–21)."""
    plot_cca_vs_age(df, out_dir)
    plot_cca_coverage(df, out_dir)
    plot_cca_vs_rin_receiver(df, out_dir)


def plot_cca_by_day(df: pd.DataFrame, out_dir: Path):
    """
    Day 1–2–3 comparison for CCa (directed 12/21; one-sided pairs allowed)
    + Day1 vs Day2 stats (pooled directions and pair mean when both present).
    """
    if not _cca_columns_present(df):
        print("Skipping CCa by day: amp_ratio columns not found.")
        return
    c1, c2 = PARAM_DIRS["CCa"]
    dfp = included_pairs(df)
    dfp = dfp[dfp[c1].notna() | dfp[c2].notna()].copy()
    if len(dfp) == 0:
        print("Skipping CCa by day: no CCa values.")
        return

    vmin, max_age, mt_val, has_mt = age_limits(dfp)
    age_ticks = unique_ages_for_colorbar(dfp)
    n_parts = []
    for day in (1, 2, 3):
        n = int(((dfp["day"] == day) & (dfp[c1].notna() | dfp[c2].notna())).sum())
        n_parts.append(f"Day {day}: $n$={n}")
    n_line = "   |   ".join(n_parts)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
        colors, _, _ = age_colors(
            dfp["age_num"].to_numpy(), dfp["is_mt"].to_numpy(), vmin, max_age, mt_val
        )
        for (_, row), color in zip(dfp.iterrows(), colors):
            if pd.isna(row["day"]):
                continue
            day = int(row["day"])
            if day not in DAY_X:
                continue
            x0 = DAY_X[day]
            x1 = x0 - DAY_PAIR_OFFSET
            x2 = x0 + DAY_PAIR_OFFSET
            y1, y2 = row[c1], row[c2]
            if pd.notna(y1) and pd.notna(y2):
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.22, lw=0.7, zorder=1)
                ax.scatter(
                    [x1, x2], [y1, y2], c=[color, color], s=POINT_SIZE,
                    alpha=POINT_ALPHA, zorder=2, edgecolors="white", linewidths=0.35,
                )
            elif pd.notna(y1):
                ax.scatter(
                    [x1], [y1], c=[color], s=POINT_SIZE, alpha=POINT_ALPHA,
                    zorder=2, edgecolors="white", linewidths=0.35,
                )
            elif pd.notna(y2):
                ax.scatter(
                    [x2], [y2], c=[color], s=POINT_SIZE, alpha=POINT_ALPHA,
                    zorder=2, edgecolors="white", linewidths=0.35,
                )
        ax.set_ylabel(PARAM_YLABELS["CCa"])
        ax.set_xticks([DAY_X[1], DAY_X[2], DAY_X[3]])
        ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2], DAY_TICK_LABELS[3]])
        ax.set_xlim(DAY_X[1] - 0.35, DAY_X[3] + 0.35)
        _style_day_ax(ax)
        cbar = add_age_colorbar(
            fig, [ax], vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"$CC_a$ by recording day"
            "\n"
            + n_line
            + r"; left/right = dir $12$/$21$ (one-sided OK)",
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "22_CCa_by_day.png")

    # Day1 vs Day2 stats: pooled directed + pair mean (both)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4), constrained_layout=True)
        # pooled
        d1p = pool_both_sides(dfp, 1, c1, c2)
        d2p = pool_both_sides(dfp, 2, c1, c2)
        res_p = compare_two_groups(d1p, d2p)
        ax = axes[0]
        rng = np.random.default_rng(0)
        ax.scatter(
            np.full(len(d1p), 1) + rng.normal(0, 0.03, len(d1p)), d1p,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
        ax.scatter(
            np.full(len(d2p), 2) + rng.normal(0, 0.03, len(d2p)), d2p,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
        _plot_group_summary(ax, res_p)
        ax.set_title(r"Pooled $\{CC_{a,12},CC_{a,21}\}$", fontsize=10, color="0.25")
        ax.set_ylabel(PARAM_YLABELS["CCa"])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Day 1", "Day 2"])
        ax.text(
            0.98, 0.96,
            f"{res_p['test']}\n{format_p(res_p['p'])}\nn={res_p['n1']}/{res_p['n2']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.25",
        )
        _style_pub_ax(ax)
        print(
            f"\nCCa Day1 vs Day2 pooled: {res_p['test']} {format_p(res_p['p'])}, "
            f"n={res_p['n1']}/{res_p['n2']}"
        )

        # pair mean both
        d1m = dfp.loc[(dfp["day"] == 1) & dfp["CCa_pair_mean"].notna(), "CCa_pair_mean"].to_numpy(dtype=float) if "CCa_pair_mean" in dfp.columns else np.array([])
        d2m = dfp.loc[(dfp["day"] == 2) & dfp["CCa_pair_mean"].notna(), "CCa_pair_mean"].to_numpy(dtype=float) if "CCa_pair_mean" in dfp.columns else np.array([])
        res_m = compare_two_groups(d1m, d2m)
        ax = axes[1]
        ax.scatter(
            np.full(len(d1m), 1) + rng.normal(0, 0.03, len(d1m)), d1m,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
        ax.scatter(
            np.full(len(d2m), 2) + rng.normal(0, 0.03, len(d2m)), d2m,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
        _plot_group_summary(ax, res_m)
        ax.set_title(r"$CC_a$ pair mean (both sides)", fontsize=10, color="0.25")
        ax.set_ylabel(PARAM_YLABELS["CCa"])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Day 1", "Day 2"])
        ax.text(
            0.98, 0.96,
            f"{res_m['test']}\n{format_p(res_m['p'])}\nn={res_m['n1']}/{res_m['n2']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.25",
        )
        _style_pub_ax(ax)
        print(
            f"CCa Day1 vs Day2 pair mean: {res_m['test']} {format_p(res_m['p'])}, "
            f"n={res_m['n1']}/{res_m['n2']}"
        )

        handles = [
            Line2D([0], [0], marker="o", color="crimson", lw=0, markersize=6, label="mean ± SEM"),
            Line2D([0], [0], marker="D", color="royalblue", lw=0, markersize=5, label="median"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(
            r"Day 1 vs Day 2 — $CC_a$"
            "\n"
            + TEST_RULE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "22b_CCa_day1_vs_day2_stats.png")


def plot_cca_by_day_p10_32(df: pd.DataFrame, out_dir: Path):
    """
    Day 1–2–3 CCa for age P10–32 only: overview + Day1 vs Day2 stats
    + Kruskal–Wallis across days (pooled directions).
    """
    if not _cca_columns_present(df):
        print("Skipping CCa by day P10–32: amp_ratio columns not found.")
        return
    c1, c2 = PARAM_DIRS["CCa"]
    dfp = included_pairs(df)
    # first AGE_ASYM_BINS entry is P10–32
    lab0, lo0, hi0, mt0, _col0 = AGE_ASYM_BINS[0]
    dfp = dfp.loc[_age_bin_mask(dfp, lo0, hi0, mt0)].copy()
    dfp = dfp[dfp[c1].notna() | dfp[c2].notna()].copy()
    if len(dfp) == 0:
        print(f"Skipping CCa by day {lab0}: no CCa values.")
        return

    n_parts = []
    day_pools = {}
    for day in (1, 2, 3):
        n = int(((dfp["day"] == day) & (dfp[c1].notna() | dfp[c2].notna())).sum())
        n_parts.append(f"Day {day}: $n$={n}")
        day_pools[day] = pool_both_sides(dfp, day, c1, c2)
    n_line = "   |   ".join(n_parts)
    print(f"\nCCa by day ({lab0}):")

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
        # overview — gray points (age band fixed)
        ax = axes[0]
        for _, row in dfp.iterrows():
            if pd.isna(row["day"]):
                continue
            day = int(row["day"])
            if day not in DAY_X:
                continue
            x0 = DAY_X[day]
            x1 = x0 - DAY_PAIR_OFFSET
            x2 = x0 + DAY_PAIR_OFFSET
            y1, y2 = row[c1], row[c2]
            if pd.notna(y1) and pd.notna(y2):
                ax.plot([x1, x2], [y1, y2], color="0.55", alpha=0.25, lw=0.7, zorder=1)
                ax.scatter(
                    [x1, x2], [y1, y2], c="0.45", s=POINT_SIZE,
                    alpha=POINT_ALPHA, zorder=2, edgecolors="white", linewidths=0.35,
                )
            elif pd.notna(y1):
                ax.scatter(
                    [x1], [y1], c="0.45", s=POINT_SIZE, alpha=POINT_ALPHA,
                    zorder=2, edgecolors="white", linewidths=0.35,
                )
            elif pd.notna(y2):
                ax.scatter(
                    [x2], [y2], c="0.45", s=POINT_SIZE, alpha=POINT_ALPHA,
                    zorder=2, edgecolors="white", linewidths=0.35,
                )
        # day means
        for day in (1, 2, 3):
            vals = day_pools[day]
            if len(vals) == 0:
                continue
            ax.errorbar(
                [DAY_X[day]], [float(np.mean(vals))],
                yerr=[float(stats.sem(vals)) if len(vals) > 1 else 0.0],
                fmt="o", color="crimson", markersize=6, capsize=3, zorder=4,
            )
            ax.plot(
                [DAY_X[day]], [float(np.median(vals))],
                marker="D", color="royalblue", markersize=5, zorder=5,
            )
        ax.set_ylabel(PARAM_YLABELS["CCa"])
        ax.set_xticks([DAY_X[1], DAY_X[2], DAY_X[3]])
        ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2], DAY_TICK_LABELS[3]])
        ax.set_xlim(DAY_X[1] - 0.35, DAY_X[3] + 0.35)
        _style_day_ax(ax)
        nonempty = [day_pools[d] for d in (1, 2, 3) if len(day_pools[d]) >= 3]
        if len(nonempty) >= 2:
            _h, p_kw = stats.kruskal(*nonempty)
            ax.text(
                0.04, 0.96, f"KW (days) {format_p(p_kw)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
            print(f"  KW across days: {format_p(p_kw)}")
        ax.set_title(rf"Overview — age {lab0}", fontsize=10, color="0.25")

        # Day1 vs Day2 stats
        ax = axes[1]
        _day12_stat_panel(
            ax, day_pools[1], day_pools[2],
            ylabel=PARAM_YLABELS["CCa"],
            title=r"Day 1 vs Day 2 (pooled $\{12,21\}$)",
        )
        handles = [
            Line2D([0], [0], marker="o", color="crimson", lw=0, markersize=6, label="mean ± SEM"),
            Line2D([0], [0], marker="D", color="royalblue", lw=0, markersize=5, label="median"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(
            rf"$CC_a$ by day — age {lab0}"
            "\n"
            + n_line
            + r"; "
            + TEST_RULE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "33_CCa_by_day_P10_32.png")


def plot_cca_vs_cc_by_age(df: pd.DataFrame, out_dir: Path):
    """
    CCa vs CC pair means by age group (P10–32 / P33–50 / P50+), separate Spearman.
    Top: both CCa required; bottom: missing CCa → 0.
    """
    if not _cca_columns_present(df):
        print("Skipping CCa vs CC by age: amp_ratio columns not found.")
        return
    sub = analysis_pool(df)
    if "CC_pair_mean" not in sub.columns:
        print("Skipping CCa vs CC by age: CC_pair_mean missing.")
        return
    y_both = (
        sub["CCa_pair_mean"].to_numpy(dtype=float)
        if "CCa_pair_mean" in sub.columns
        else np.full(len(sub), np.nan)
    )
    y_zero = (
        sub["CCa_pair_mean_z"].to_numpy(dtype=float)
        if "CCa_pair_mean_z" in sub.columns
        else np.array(
            [_pair_mean_zero_fill(a, b) for a, b in zip(sub[PARAM_DIRS["CCa"][0]], sub[PARAM_DIRS["CCa"][1]])],
            dtype=float,
        )
    )
    x_mean = sub["CC_pair_mean"].to_numpy(dtype=float)
    print("\nCCa vs CC by age (Spearman):")

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), constrained_layout=True)
        for row, (y_mean, ylab) in enumerate(
            (
                (y_both, r"$CC_a$ pair mean (both sides)"),
                (y_zero, r"$CC_a$ mean (missing→0)"),
            )
        ):
            for c, (label, lo, hi, include_mt, color) in enumerate(AGE_ASYM_BINS):
                ax = axes[row, c]
                mask = _age_bin_mask(sub, lo, hi, include_mt).to_numpy()
                m = mask & np.isfinite(x_mean) & np.isfinite(y_mean)
                n = int(m.sum())
                if n:
                    ax.scatter(
                        x_mean[m], y_mean[m], c=color, s=POINT_SIZE,
                        alpha=POINT_ALPHA, edgecolors="white", linewidths=0.35, zorder=2,
                    )
                if n >= 3:
                    rho, p = stats.spearmanr(x_mean[m], y_mean[m])
                    ax.text(
                        0.04, 0.96,
                        f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={n}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="0.25",
                    )
                    print(f"  {label} {ylab}: ρ={rho:.3f}, {format_p(p)}, n={n}")
                else:
                    ax.text(
                        0.04, 0.96, f"$n$={n}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="0.25",
                    )
                    print(f"  {label} {ylab}: n={n} (too few)")
                ax.set_xlabel(PARAM_YLABELS["CC"] + r" pair mean")
                if c == 0:
                    ax.set_ylabel(ylab)
                _style_pub_ax(ax)
                if row == 0:
                    ax.set_title(f"{label}\n$n_{{ec}}$={int(mask.sum())}", fontsize=10, color="0.2")
        fig.suptitle(
            r"$CC_a$ vs $CC$ by age group"
            "\n"
            r"top: both $CC_a$; bottom: missing→0; separate Spearman; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "23_CCa_vs_CC_by_age.png")


def _cca_asym_panel(ax, d: pd.DataFrame, *, color=None, colors=None, with_fit: bool = True):
    """CCa max vs min (zero-fill); mark unilateral (min≈0) with open markers."""
    if len(d) == 0:
        return
    x = d["CCa_min_z"].to_numpy(dtype=float)
    y = d["CCa_max_z"].to_numpy(dtype=float)
    uni = d["cca_modality"].to_numpy() == "one" if "cca_modality" in d.columns else (x <= 0)
    both = ~uni
    if colors is not None:
        cols = np.asarray(colors)
        if both.any():
            ax.scatter(
                x[both], y[both], c=cols[both], s=POINT_SIZE, alpha=POINT_ALPHA,
                edgecolors="white", linewidths=0.35, zorder=2,
            )
        if uni.any():
            ax.scatter(
                x[uni], y[uni], facecolors="none", edgecolors=cols[uni],
                s=POINT_SIZE * 1.05, linewidths=1.0, zorder=3,
            )
    else:
        if both.any():
            ax.scatter(
                x[both], y[both], c=color, s=POINT_SIZE, alpha=POINT_ALPHA,
                edgecolors="white", linewidths=0.35, zorder=2,
            )
        if uni.any():
            ax.scatter(
                x[uni], y[uni], facecolors="none", edgecolors=color,
                s=POINT_SIZE * 1.05, linewidths=1.0, zorder=3,
            )
    lim_min, lim_max = _asym_axis_limits(x, y, name="CCa")
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="0.35", ls="--", lw=0.9, zorder=1)
    if with_fit:
        k = _fit_slope_through_origin(x, y)
        if k is not None:
            ax.plot(
                [lim_min, lim_max], [k * lim_min, k * lim_max],
                color="crimson", lw=1.2, zorder=1,
            )
            ax.text(
                0.04, 0.96, f"k = {k:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$CC_a$ min (missing→0)")
    ax.set_ylabel(r"$CC_a$ max (missing→0)")
    _style_pub_ax(ax)


def plot_cca_asymmetry(df: pd.DataFrame, out_dir: Path):
    """
    CCa max vs min like CC/Gj asymmetry clouds.
    Missing direction filled as 0 (unilateral → min=0, AI→1).
    Companion: same by age groups.
    """
    if not _cca_columns_present(df):
        print("Skipping CCa asymmetry: amp_ratio columns not found.")
        return
    if "CCa_min_z" not in df.columns:
        print("Skipping CCa asymmetry: zero-fill columns missing.")
        return
    sub = analysis_pool(df)
    d = sub[sub["CCa_min_z"].notna() & sub["CCa_max_z"].notna()].copy()
    if len(d) == 0:
        print("Skipping CCa asymmetry: no CCa values.")
        return
    vmin, max_age, mt_val, has_mt = age_limits(d)
    age_ticks = unique_ages_for_colorbar(d)
    print(f"\nCCa asymmetry (missing→0): n={len(d)} "
          f"(both={int((d['cca_modality']=='both').sum())}, "
          f"one={int((d['cca_modality']=='one').sum())})")

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
        colors, _, _ = age_colors(
            d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
        )
        _cca_asym_panel(ax, d, colors=colors, with_fit=True)
        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.45",
                markeredgecolor="white", markersize=7, label=r"both $CC_a$",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor="0.35", markersize=7, label=r"one $CC_a$ (min=0)",
            ),
            Line2D([0], [0], color="0.35", ls="--", lw=0.9, label="identity"),
            Line2D([0], [0], color="crimson", lw=1.2, label="fit through 0"),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
        cbar = add_age_colorbar(
            fig, [ax], vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"$CC_a$ asymmetry (max vs min)"
            "\n"
            r"missing direction = 0; open = unilateral; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "24_CCa_asymmetry.png")

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
        all_x, all_y = [], []
        for label, lo, hi, include_mt, color in AGE_ASYM_BINS:
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            g = sub.loc[mask & sub["CCa_min_z"].notna() & sub["CCa_max_z"].notna()]
            if len(g):
                all_x.append(g["CCa_min_z"].to_numpy(dtype=float))
                all_y.append(g["CCa_max_z"].to_numpy(dtype=float))
        if all_x:
            lim_min, lim_max = _asym_axis_limits(
                np.concatenate(all_x), np.concatenate(all_y), name="CCa"
            )
        else:
            lim_min, lim_max = 0.0, 1.0
        for ax, (label, lo, hi, include_mt, color) in zip(axes, AGE_ASYM_BINS):
            mask = _age_bin_mask(sub, lo, hi, include_mt)
            g = sub.loc[mask & sub["CCa_min_z"].notna() & sub["CCa_max_z"].notna()].copy()
            ax.plot(
                [lim_min, lim_max], [lim_min, lim_max],
                color="0.35", ls="--", lw=0.9, zorder=1,
            )
            if len(g):
                x = g["CCa_min_z"].to_numpy(dtype=float)
                y = g["CCa_max_z"].to_numpy(dtype=float)
                uni = g["cca_modality"].to_numpy() == "one"
                both = ~uni
                if both.any():
                    ax.scatter(
                        x[both], y[both], c=color, s=POINT_SIZE, alpha=POINT_ALPHA,
                        edgecolors="white", linewidths=0.35, zorder=2,
                    )
                if uni.any():
                    ax.scatter(
                        x[uni], y[uni], facecolors="none", edgecolors=color,
                        s=POINT_SIZE * 1.05, linewidths=1.0, zorder=3,
                    )
                k = _fit_slope_through_origin(x, y)
                if k is not None:
                    ax.plot(
                        [lim_min, lim_max], [k * lim_min, k * lim_max],
                        color="crimson", lw=1.2, zorder=1,
                    )
                    ax.text(
                        0.04, 0.96, f"k = {k:.2f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="0.25",
                    )
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$CC_a$ min (missing→0)")
            ax.set_ylabel(r"$CC_a$ max (missing→0)")
            ax.set_title(
                f"{label}\n$n$={len(g)} (both={int((g['cca_modality']=='both').sum()) if len(g) else 0})",
                fontsize=10, color="0.2",
            )
            _style_pub_ax(ax)
        fig.suptitle(
            r"$CC_a$ asymmetry by age (missing→0)"
            "\n"
            r"open markers = unilateral; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "24b_CCa_asymmetry_by_age.png")


def _day12_stat_panel(ax, d1: np.ndarray, d2: np.ndarray, *, ylabel: str, title: str):
    """One Day1 vs Day2 panel: jitter + mean±SEM + median + test annotation."""
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]
    if len(d1) == 0 and len(d2) == 0:
        ax.set_visible(False)
        return
    res = compare_two_groups(d1, d2) if (len(d1) and len(d2)) else {
        "n1": len(d1), "n2": len(d2),
        "mean1": float(np.mean(d1)) if len(d1) else np.nan,
        "mean2": float(np.mean(d2)) if len(d2) else np.nan,
        "sem1": float(stats.sem(d1)) if len(d1) > 1 else 0.0,
        "sem2": float(stats.sem(d2)) if len(d2) > 1 else 0.0,
        "median1": float(np.median(d1)) if len(d1) else np.nan,
        "median2": float(np.median(d2)) if len(d2) else np.nan,
        "test": "n/a", "p": np.nan,
    }
    rng = np.random.default_rng(0)
    if len(d1):
        ax.scatter(
            np.full(len(d1), 1) + rng.normal(0, 0.03, len(d1)), d1,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
    if len(d2):
        ax.scatter(
            np.full(len(d2), 2) + rng.normal(0, 0.03, len(d2)), d2,
            s=14, alpha=0.35, c="0.55", edgecolors="none", zorder=1,
        )
    if len(d1) and len(d2):
        _plot_group_summary(ax, res)
    ax.set_title(title, fontsize=10, color="0.25")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Day 1", "Day 2"])
    ax.text(
        0.98, 0.96,
        f"{res['test']}\n{format_p(res.get('p'))}\nn={res['n1']}/{res['n2']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.25",
    )
    _style_pub_ax(ax)
    print(f"  {title}: {res['test']} {format_p(res.get('p'))}, n={res['n1']}/{res['n2']}")


def plot_delays_by_day(df: pd.DataFrame, out_dir: Path):
    """Delay onset/peak Day1 vs Day2 — same template as 22b (pooled + pair mean)."""
    need = [*PARAM_DIRS["DelayOnset"], *PARAM_DIRS["DelayPeak"]]
    if any(c not in df.columns for c in need):
        print("Skipping delays Day1 vs Day2: delay columns not found.")
        return
    dfp = included_pairs(df)
    on1, on2 = PARAM_DIRS["DelayOnset"]
    pk1, pk2 = PARAM_DIRS["DelayPeak"]
    has_any = dfp[on1].notna() | dfp[on2].notna() | dfp[pk1].notna() | dfp[pk2].notna()
    dfp = dfp[has_any].copy()
    if len(dfp) == 0:
        print("Skipping delays Day1 vs Day2: no delay values.")
        return

    print("\nDelays Day1 vs Day2:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2), constrained_layout=True)
        # onset pooled / mean
        _day12_stat_panel(
            axes[0, 0],
            pool_both_sides(dfp, 1, on1, on2),
            pool_both_sides(dfp, 2, on1, on2),
            ylabel=PARAM_YLABELS["DelayOnset"],
            title=r"Onset pooled $\{12,21\}$",
        )
        d1m = dfp.loc[(dfp["day"] == 1) & dfp["DelayOnset_pair_mean"].notna(), "DelayOnset_pair_mean"].to_numpy(dtype=float) if "DelayOnset_pair_mean" in dfp.columns else np.array([])
        d2m = dfp.loc[(dfp["day"] == 2) & dfp["DelayOnset_pair_mean"].notna(), "DelayOnset_pair_mean"].to_numpy(dtype=float) if "DelayOnset_pair_mean" in dfp.columns else np.array([])
        _day12_stat_panel(
            axes[0, 1], d1m, d2m,
            ylabel=PARAM_YLABELS["DelayOnset"],
            title=r"Onset pair mean (both)",
        )
        _day12_stat_panel(
            axes[1, 0],
            pool_both_sides(dfp, 1, pk1, pk2),
            pool_both_sides(dfp, 2, pk1, pk2),
            ylabel=PARAM_YLABELS["DelayPeak"],
            title=r"Peak pooled $\{12,21\}$",
        )
        d1p = dfp.loc[(dfp["day"] == 1) & dfp["DelayPeak_pair_mean"].notna(), "DelayPeak_pair_mean"].to_numpy(dtype=float) if "DelayPeak_pair_mean" in dfp.columns else np.array([])
        d2p = dfp.loc[(dfp["day"] == 2) & dfp["DelayPeak_pair_mean"].notna(), "DelayPeak_pair_mean"].to_numpy(dtype=float) if "DelayPeak_pair_mean" in dfp.columns else np.array([])
        _day12_stat_panel(
            axes[1, 1], d1p, d2p,
            ylabel=PARAM_YLABELS["DelayPeak"],
            title=r"Peak pair mean (both)",
        )
        handles = [
            Line2D([0], [0], marker="o", color="crimson", lw=0, markersize=6, label="mean ± SEM"),
            Line2D([0], [0], marker="D", color="royalblue", lw=0, markersize=5, label="median"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(
            r"Day 1 vs Day 2 — onset / peak delay"
            "\n"
            + TEST_RULE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "25_delays_day1_vs_day2.png")


def plot_cca_modality_by_conductance(df: pd.DataFrame, out_dir: Path):
    """
    None / unilateral / bilateral CCa counts (+ %) by pair-mean CC and Gj bins.
    Shows how many and which modality at each conductance strength.
    """
    if not _cca_columns_present(df):
        print("Skipping CCa modality by conductance: amp_ratio columns not found.")
        return
    sub = analysis_pool(df)
    panels = []
    if "CC_pair_mean" in sub.columns:
        panels.append(("CC_pair_mean", CC_BIN_WIDTH, PARAM_YLABELS["CC"] + r" pair mean", "cc"))
    if "Gj_pair_mean" in sub.columns:
        panels.append(("Gj_pair_mean", GJ_BIN_WIDTH, PARAM_YLABELS["Gj"] + r" pair mean", "gj"))
    if not panels:
        print("Skipping CCa modality by conductance: no CC/Gj pair means.")
        return

    modality_order = ["none", "one", "both"]
    modality_colors = {"none": "0.78", "one": "#F4A261", "both": "#2A9D8F"}
    modality_labels = {
        "none": r"no $CC_a$",
        "one": r"unilateral $CC_a$",
        "both": r"bilateral $CC_a$",
    }

    print("\nCCa modality by conductance:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(
            2, len(panels),
            figsize=(5.8 * len(panels) + 1.8, 8.0),
            constrained_layout=True,
        )
        if len(panels) == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for c, (xcol, width, xlabel, _tag) in enumerate(panels):
            d = sub[sub[xcol].notna()].copy()
            bins, meta = assign_width_bins(d[xcol], width)
            d["_bin"] = bins
            labels = [m[0] for m in meta]
            counts = {m: [] for m in modality_order}
            for name, _lo, _hi, _n in meta:
                g = d.loc[d["_bin"] == name]
                for mod in modality_order:
                    counts[mod].append(int((g["cca_modality"] == mod).sum()))
                tot = int(len(g))
                print(
                    f"  {xcol} {name}: n={tot} "
                    f"none={counts['none'][-1]} one={counts['one'][-1]} both={counts['both'][-1]}"
                )

            x = np.arange(len(labels), dtype=float)
            # counts stacked
            ax = axes[0, c]
            bottom = np.zeros(len(labels), dtype=float)
            for mod in modality_order:
                vals = np.asarray(counts[mod], dtype=float)
                ax.bar(
                    x, vals, bottom=bottom, width=0.85,
                    color=modality_colors[mod], edgecolor="white", linewidth=0.4, zorder=2,
                    label=modality_labels[mod],
                )
                bottom += vals
            tick_labs = [f"{lo:g}–{hi:g}" for _name, lo, hi, _n in meta]
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labs, fontsize=7 if len(labels) > 6 else 8)
            if len(labels) > 6:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(35)
            ax.set_ylabel("pairs")
            ax.set_xlabel(xlabel + rf" (width={width:g})")
            ax.set_title("Counts", fontsize=10, color="0.25")
            _style_pub_ax(ax)

            # percent stacked
            ax = axes[1, c]
            bottom = np.zeros(len(labels), dtype=float)
            totals = sum(np.asarray(counts[m], dtype=float) for m in modality_order)
            for mod in modality_order:
                vals = np.asarray(counts[mod], dtype=float)
                pct = np.where(totals > 0, 100.0 * vals / totals, 0.0)
                ax.bar(
                    x, pct, bottom=bottom, width=0.85,
                    color=modality_colors[mod], edgecolor="white", linewidth=0.4, zorder=2,
                )
                bottom += pct
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labs, fontsize=7 if len(labels) > 6 else 8)
            if len(labels) > 6:
                for lab in ax.get_xticklabels():
                    lab.set_rotation(35)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of pairs in bin")
            ax.set_xlabel(xlabel + rf" (width={width:g})")
            ax.set_title("Percent", fontsize=10, color="0.25")
            _style_pub_ax(ax)

        # Room for legend below panels (avoids overlapping bars / title)
        handles = [
            Line2D(
                [0], [0], marker="s", color=modality_colors[m], lw=0,
                markersize=10, label=modality_labels[m],
            )
            for m in modality_order
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            fontsize=10,
            columnspacing=1.8,
            handletextpad=0.5,
        )
        fig.suptitle(
            r"$CC_a$ modality vs coupling strength"
            "\n"
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "26_CCa_modality_by_conductance.png")


def plot_ai_cca(df: pd.DataFrame, out_dir: Path):
    """
    AI for CCa (missing side → 0) vs AI_CC / AI_Gj, and vs CCa mean (zero-fill).
    """
    if not _cca_columns_present(df) or "AI_CCa_z" not in df.columns:
        print("Skipping AI_CCa: columns not found.")
        return
    sub = analysis_pool(df)
    need = ["AI_CCa_z", "AI_CC", "AI_Gj", "CCa_pair_mean_z"]
    if any(c not in sub.columns for c in need):
        print("Skipping AI_CCa: missing AI / mean columns.")
        return
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)

    print("\nAI_CCa (missing→0):")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
        panels = [
            ("AI_CC", r"AI $CC$", r"AI $CC_a$ (missing→0)"),
            ("AI_Gj", r"AI $G_{j}$", r"AI $CC_a$ (missing→0)"),
            ("CCa_pair_mean_z", r"$CC_a$ mean (missing→0)", r"AI $CC_a$ (missing→0)"),
        ]
        for ax, (xcol, xlab, ylab) in zip(axes, panels):
            d = sub[sub["AI_CCa_z"].notna() & sub[xcol].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            # distinguish unilateral (AI=1 typically)
            uni = d["cca_modality"].to_numpy() == "one" if "cca_modality" in d.columns else np.zeros(len(d), dtype=bool)
            both = ~uni
            if both.any():
                ax.scatter(
                    d.loc[both, xcol], d.loc[both, "AI_CCa_z"],
                    c=np.asarray(colors)[both], s=POINT_SIZE, alpha=POINT_ALPHA,
                    edgecolors="white", linewidths=0.35, zorder=2,
                )
            if uni.any():
                ax.scatter(
                    d.loc[uni, xcol], d.loc[uni, "AI_CCa_z"],
                    facecolors="none", edgecolors=np.asarray(colors)[uni],
                    s=POINT_SIZE * 1.05, linewidths=1.0, zorder=3,
                )
            if xcol.startswith("AI_"):
                ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=0.9, zorder=1)
                ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            if len(d) >= 3:
                rho, p = stats.spearmanr(d[xcol], d["AI_CCa_z"])
                ax.text(
                    0.04, 0.96,
                    f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={len(d)}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
                )
                print(f"  AI_CCa_z vs {xcol}: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
            else:
                print(f"  AI_CCa_z vs {xcol}: n={len(d)} (too few)")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            _style_pub_ax(ax)

        handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="0.45",
                markeredgecolor="white", markersize=7, label=r"both $CC_a$",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor="0.35", markersize=7, label=r"unilateral (AI→1)",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"AI $CC_a$ vs coupling AI / $CC_a$ strength"
            "\n"
            + AI_FORMULA
            + r"; missing $CC_a$ side = 0; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "27_AI_CCa_vs_AI_CC_Gj.png")


def plot_cca_extended_figures(df: pd.DataFrame, out_dir: Path):
    """Figs 23–27: age Spearman, asymmetry, delays day, modality, AI_CCa."""
    plot_cca_vs_cc_by_age(df, out_dir)
    plot_cca_asymmetry(df, out_dir)
    plot_delays_by_day(df, out_dir)
    plot_cca_modality_by_conductance(df, out_dir)
    plot_ai_cca(df, out_dir)


def directed_delay_vs_recv_param(
    df: pd.DataFrame, y12: str, y21: str, x1: str, x2: str
) -> pd.DataFrame:
    """
    Directed delay vs a receiver-side cell parameter.
    1→2 uses x2; 2→1 uses x1 (receiver of that direction).
    """
    rows = []
    for yc, xc, side in ((y12, x2, "12"), (y21, x1, "21")):
        if yc not in df.columns or xc not in df.columns:
            continue
        for _, r in df.iterrows():
            if pd.notna(r[xc]) and pd.notna(r[yc]):
                rows.append(
                    {
                        "x": float(r[xc]),
                        "y": float(r[yc]),
                        "age_num": r.get("age_num", np.nan),
                        "is_mt": bool(r.get("is_mt", False)),
                        "side": side,
                    }
                )
    return pd.DataFrame(rows)


def directed_delay_vs_rin_recv(
    df: pd.DataFrame, y12: str, y21: str
) -> pd.DataFrame:
    """Directed delay vs receiver Rin (1→2: Rin_2; 2→1: Rin_1)."""
    return directed_delay_vs_recv_param(df, y12, y21, "Rin_1", "Rin_2")


def _ols_line_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Simple OLS line over x-range; None if too few finite points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.allclose(x, x[0]):
        return None
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([float(np.min(x)), float(np.max(x))])
    y_line = slope * x_line + intercept
    return x_line, y_line


def plot_delays_vs_rin_receiver(df: pd.DataFrame, out_dir: Path):
    """
    Onset / peak delay vs receiver Rin (directed).
    Straight OLS line as a visual guide for the Spearman trend (ρ annotated on panel).
    """
    need = [*PARAM_DIRS["DelayOnset"], *PARAM_DIRS["DelayPeak"], "Rin_1", "Rin_2"]
    if any(c not in df.columns for c in need):
        print("Skipping delays vs Rin_recv: delay/Rin columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    on12, on21 = PARAM_DIRS["DelayOnset"]
    pk12, pk21 = PARAM_DIRS["DelayPeak"]
    panels = [
        (on12, on21, PARAM_YLABELS["DelayOnset"]),
        (pk12, pk21, PARAM_YLABELS["DelayPeak"]),
    ]
    print("\nDelays vs Rin_recv:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        for ax, (y12, y21, ylab) in zip(axes, panels):
            both = directed_delay_vs_rin_recv(sub, y12, y21)
            print(f"  {ylab}: n={len(both)}")
            _scatter_directed_with_missing_rug(
                ax, both, pd.DataFrame(),
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                mark_missing=False, y_label=ylab,
                x_label=r"$R_{in,\mathrm{recv}}$ [M$\Omega$]",
            )
            if len(both) >= 3:
                line = _ols_line_xy(
                    both["x"].to_numpy(dtype=float),
                    both["y"].to_numpy(dtype=float),
                )
                if line is not None:
                    ax.plot(
                        line[0], line[1], color="0.65", lw=1.6, zorder=2,
                        solid_capstyle="round",
                    )
        handles = [
            Line2D(
                [0], [0], color="0.65", lw=1.6,
                label="OLS line (illustration; ρ = Spearman)",
            ),
        ]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Onset / peak delay vs receiver $R_{in}$"
            "\n"
            r"$1\!\to\!2$: $R_{in,2}$; $2\!\to\!1$: $R_{in,1}$; "
            r"gray line: OLS illustration (not the Spearman slope); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "28_delays_vs_Rin_receiver.png")


def plot_delays_vs_cm_tau_receiver(df: pd.DataFrame, out_dir: Path):
    """
    Onset / peak delay vs receiver Cm and vs receiver τ.
    τ_cell ≈ Rin × Cm; with Rin [MΩ], Cm [pF]: τ [ms] = Rin·Cm/1000.
    """
    need = [
        *PARAM_DIRS["DelayOnset"],
        *PARAM_DIRS["DelayPeak"],
        "Cm_1",
        "Cm_2",
        "Tau_1",
        "Tau_2",
    ]
    if any(c not in df.columns for c in need):
        print("Skipping delays vs Cm/τ: need delay, Cm, and Tau columns.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    on12, on21 = PARAM_DIRS["DelayOnset"]
    pk12, pk21 = PARAM_DIRS["DelayPeak"]
    # rows = delay type; cols = Cm_recv, Tau_recv
    row_specs = [
        (on12, on21, PARAM_YLABELS["DelayOnset"]),
        (pk12, pk21, PARAM_YLABELS["DelayPeak"]),
    ]
    col_specs = [
        ("Cm_1", "Cm_2", r"$C_{m,\mathrm{recv}}$ [pF]"),
        ("Tau_1", "Tau_2", r"$\tau_{\mathrm{recv}}\approx R_{in}C_m$ [ms]"),
    ]
    print("\nDelays vs Cm_recv / τ_recv:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
        for r, (y12, y21, ylab) in enumerate(row_specs):
            for c, (x1, x2, xlab) in enumerate(col_specs):
                ax = axes[r, c]
                both = directed_delay_vs_recv_param(sub, y12, y21, x1, x2)
                print(f"  {ylab} vs {xlab}: n={len(both)}")
                _scatter_directed_with_missing_rug(
                    ax, both, pd.DataFrame(),
                    vmin=vmin, max_age=max_age, mt_val=mt_val,
                    mark_missing=False, y_label=ylab, x_label=xlab,
                )
        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Onset / peak delay vs receiver $C_m$ and $\tau$"
            "\n"
            r"$1\!\to\!2$: cell 2; $2\!\to\!1$: cell 1; "
            r"$\tau$ [ms] $=R_{in}$ [M$\Omega$]$\cdot C_m$ [pF]$/1000$; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "35_delays_vs_Cm_Tau_receiver.png")


def plot_delay_onset_vs_peak(df: pd.DataFrame, out_dir: Path):
    """Onset delay vs peak delay (directed + pair mean)."""
    need = [*PARAM_DIRS["DelayOnset"], *PARAM_DIRS["DelayPeak"]]
    if any(c not in df.columns for c in need):
        print("Skipping onset vs peak: delay columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    on12, on21 = PARAM_DIRS["DelayOnset"]
    pk12, pk21 = PARAM_DIRS["DelayPeak"]
    both, _ = directed_xy(sub, on12, on21, pk12, pk21)
    print(f"\nOnset vs peak delay: directed n={len(both)}")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        ax = axes[0]
        _scatter_directed_with_missing_rug(
            ax, both, pd.DataFrame(),
            vmin=vmin, max_age=max_age, mt_val=mt_val,
            mark_missing=False,
            y_label=PARAM_YLABELS["DelayPeak"],
            x_label=PARAM_YLABELS["DelayOnset"],
        )
        if len(both):
            lo = float(np.nanmin([both["x"].min(), both["y"].min()]))
            hi = float(np.nanmax([both["x"].max(), both["y"].max()]))
            pad = max((hi - lo) * 0.08, 0.05)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", ls="--", lw=0.9, zorder=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(r"Directed $\{12,21\}$", fontsize=10, color="0.25")

        ax = axes[1]
        if (
            "DelayOnset_pair_mean" in sub.columns
            and "DelayPeak_pair_mean" in sub.columns
        ):
            d = sub[
                sub["DelayOnset_pair_mean"].notna() & sub["DelayPeak_pair_mean"].notna()
            ].copy()
            print(f"  pair mean n={len(d)}")
            _scatter_pair_mean_panel(
                ax, d,
                x_mean=d["DelayOnset_pair_mean"].to_numpy(dtype=float),
                y_mean=d["DelayPeak_pair_mean"].to_numpy(dtype=float),
                vmin=vmin, max_age=max_age, mt_val=mt_val,
                xlabel=PARAM_YLABELS["DelayOnset"] + r" pair mean",
                ylabel=PARAM_YLABELS["DelayPeak"] + r" pair mean",
            )
            if len(d):
                lo = float(np.nanmin(
                    [d["DelayOnset_pair_mean"].min(), d["DelayPeak_pair_mean"].min()]
                ))
                hi = float(np.nanmax(
                    [d["DelayOnset_pair_mean"].max(), d["DelayPeak_pair_mean"].max()]
                ))
                pad = max((hi - lo) * 0.08, 0.05)
                ax.plot(
                    [lo - pad, hi + pad], [lo - pad, hi + pad],
                    color="0.35", ls="--", lw=0.9, zorder=1,
                )
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_visible(False)
        ax.set_title(r"Pair mean (both sides)", fontsize=10, color="0.25")

        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Onset delay vs peak delay"
            "\n"
            r"dashed: identity; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "29_delay_onset_vs_peak.png")


def plot_ai_delays_vs_ai_coupling(df: pd.DataFrame, out_dir: Path):
    """AI_DelayOnset / AI_DelayPeak vs AI_CC and AI_Gj (both sides required)."""
    need = ["AI_DelayOnset", "AI_DelayPeak", "AI_CC", "AI_Gj"]
    if any(c not in df.columns for c in need):
        print("Skipping AI delays vs AI coupling: AI columns not found.")
        return
    sub = analysis_pool(df)
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    rows = [
        ("AI_DelayOnset", r"AI onset delay"),
        ("AI_DelayPeak", r"AI peak delay"),
    ]
    cols = [
        ("AI_CC", r"AI $CC$"),
        ("AI_Gj", r"AI $G_{j}$"),
    ]
    print("\nAI delays vs AI coupling:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
        for r, (ycol, ylab) in enumerate(rows):
            for c, (xcol, xlab) in enumerate(cols):
                ax = axes[r, c]
                d = sub[sub[ycol].notna() & sub[xcol].notna()].copy()
                colors, _, _ = age_colors(
                    d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
                )
                ax.plot([0, 1], [0, 1], color="0.35", ls="--", lw=0.9, zorder=1)
                if len(d):
                    ax.scatter(
                        d[xcol], d[ycol], c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
                        edgecolors="white", linewidths=0.35, zorder=2,
                    )
                if len(d) >= 3:
                    rho, p = stats.spearmanr(d[xcol], d[ycol])
                    ax.text(
                        0.04, 0.96,
                        f"Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={len(d)}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=8, color="0.25",
                    )
                    print(f"  {ycol} vs {xcol}: ρ={rho:.3f}, {format_p(p)}, n={len(d)}")
                else:
                    print(f"  {ycol} vs {xcol}: n={len(d)} (too few)")
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                _style_pub_ax(ax)
        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"AI delays vs AI coupling"
            "\n"
            + AI_FORMULA
            + r"; both directions required for each AI; dashed: identity; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "30_AI_delays_vs_AI_CC_Gj.png")


def plot_cca_directionality_vs_rin(df: pd.DataFrame, out_dir: Path):
    """
    Signed ΔCCa vs ΔRin_recv (same template as fig 09).
    Missing CCa side → 0 when the other side is present.
    """
    if not _cca_columns_present(df):
        print("Skipping CCa directionality: amp_ratio columns not found.")
        return
    if "Rin_1" not in df.columns or "Rin_2" not in df.columns:
        print("Skipping CCa directionality: Rin columns not found.")
        return
    sub = analysis_pool(df)
    c1, c2 = PARAM_DIRS["CCa"]
    # zero-fill delta when any CCa present
    a = sub[c1]
    b = sub[c2]
    any_cca = a.notna() | b.notna()
    dcca = pd.Series(np.nan, index=sub.index, dtype=float)
    dcca[any_cca] = a.fillna(0.0)[any_cca] - b.fillna(0.0)[any_cca]
    sub = sub.copy()
    sub["dCCa_z"] = dcca
    sub["dRin_recv"] = sub["Rin_2"] - sub["Rin_1"]
    # also both-only delta
    both = a.notna() & b.notna()
    dcca_both = pd.Series(np.nan, index=sub.index, dtype=float)
    dcca_both[both] = a[both] - b[both]
    sub["dCCa_both"] = dcca_both

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    panels = [
        ("dCCa_z", r"$\Delta CC_a$ (missing→0)"),
        ("dCCa_both", r"$\Delta CC_a$ (both sides)"),
    ]
    print("\nSigned ΔCCa vs ΔRin_recv:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        for ax, (col, ylabel) in zip(axes, panels):
            d = sub[sub[col].notna() & sub["dRin_recv"].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            x = d["dRin_recv"].to_numpy(dtype=float)
            y = d[col].to_numpy(dtype=float)
            xpad = max(float(np.nanmax(np.abs(x))) * 1.15, 1.0) if len(x) else 1.0
            ypad = max(float(np.nanmax(np.abs(y))) * 1.15, 0.05) if len(y) else 0.05
            ax.axhspan(0, ypad, xmin=0.5, xmax=1.0, color="0.92", zorder=0)
            ax.axhspan(-ypad, 0, xmin=0.0, xmax=0.5, color="0.92", zorder=0)
            ax.axhline(0, color="0.45", lw=0.8, zorder=1)
            ax.axvline(0, color="0.45", lw=0.8, zorder=1)
            if len(d):
                ax.scatter(
                    x, y, c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
                    edgecolors="white", linewidths=0.35, zorder=2,
                )
            same = (x * y) > 0
            nonzero = (x != 0) & (y != 0)
            n_nz = int(nonzero.sum())
            n_same = int((same & nonzero).sum())
            frac = n_same / n_nz if n_nz else np.nan
            binom_p = binom_pvalue(n_same, n_nz, 0.5) if n_nz >= 5 else np.nan
            rho, sp = stats.spearmanr(x, y) if len(d) >= 3 else (np.nan, np.nan)
            ax.text(
                0.04, 0.96,
                "\n".join(
                    s for s in [
                        f"same sign {n_same}/{n_nz}"
                        + (f" ({100 * frac:.0f}%)" if n_nz else ""),
                        f"binomial {format_p(binom_p)}",
                        f"Spearman ρ={rho:.2f}" if np.isfinite(rho) else "Spearman n/a",
                        format_p(sp) if np.isfinite(sp) else "",
                    ] if s
                ),
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
            print(
                f"  {col}: same-sign {n_same}/{n_nz}, "
                f"binomial {format_p(binom_p)}; ρ={rho:.3f}, {format_p(sp)}"
            )
            ax.set_xlim(-xpad, xpad)
            ax.set_ylim(-ypad, ypad)
            ax.set_xlabel(r"$\Delta R_{in,recv}=R_{in,2}-R_{in,1}$ [M$\Omega$]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)
        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Signed $\Delta CC_a$ vs receiver $\Delta R_{in}$"
            "\n"
            r"shaded: same sign (larger $CC_a$ toward higher-$R_{in}$ cell); "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "31_CCa_directionality_vs_dRin.png")


def _rank_partial_spearman(
    y: np.ndarray, x: np.ndarray, covars: np.ndarray
) -> tuple[float, float, int, np.ndarray, np.ndarray]:
    """
    Partial Spearman via rank-transform + OLS residualization of y,x on covariates.
    Returns (rho, p, n, y_resid, x_resid).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    C = np.asarray(covars, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(C), axis=1)
    y, x, C = y[m], x[m], C[m]
    n = len(y)
    if n < 5:
        return np.nan, np.nan, n, np.array([]), np.array([])
    ry = stats.rankdata(y).astype(float)
    rx = stats.rankdata(x).astype(float)
    rc = np.column_stack([stats.rankdata(C[:, j]) for j in range(C.shape[1])])
    ones = np.ones((n, 1))
    Z = np.hstack([ones, rc])

    def _resid(v):
        beta, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    yr, xr = _resid(ry), _resid(rx)
    rho, p = stats.spearmanr(yr, xr)
    return float(rho), float(p), n, yr, xr


def plot_cca_cc_partial_correlation(df: pd.DataFrame, out_dir: Path):
    """
    Partial Spearman CCa–CC controlling Distance and age.
    Panels: raw scatter, residual scatter, summary for both-only and missing→0.
    """
    if not _cca_columns_present(df):
        print("Skipping partial CCa–CC: amp_ratio columns not found.")
        return
    if "Distance_um" not in df.columns or "CC_pair_mean" not in df.columns:
        print("Skipping partial CCa–CC: Distance or CC_pair_mean missing.")
        return
    sub = analysis_pool(df)
    sub = sub[sub["Distance_um"].notna() & sub["age_num"].notna() & ~sub["is_mt"]].copy()
    if len(sub) == 0:
        print("Skipping partial CCa–CC: no rows with Distance+age.")
        return
    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)

    specs = []
    if "CCa_pair_mean" in sub.columns:
        specs.append(("CCa_pair_mean", r"$CC_a$ pair mean (both)", "both"))
    if "CCa_pair_mean_z" in sub.columns:
        specs.append(("CCa_pair_mean_z", r"$CC_a$ mean (missing→0)", "zero"))
    if not specs:
        print("Skipping partial CCa–CC: no CCa mean columns.")
        return

    print("\nPartial Spearman CCa–CC | Distance, age:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(
            len(specs), 2, figsize=(10.5, 4.2 * len(specs)), constrained_layout=True
        )
        if len(specs) == 1:
            axes = np.array([axes])

        for r, (ycol, ylab, _tag) in enumerate(specs):
            d = sub[sub[ycol].notna() & sub["CC_pair_mean"].notna()].copy()
            x = d["CC_pair_mean"].to_numpy(dtype=float)
            y = d[ycol].to_numpy(dtype=float)
            cov = np.column_stack(
                [d["Distance_um"].to_numpy(dtype=float), d["age_num"].to_numpy(dtype=float)]
            )
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            ax = axes[r, 0]
            ax.scatter(
                x, y, c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
                edgecolors="white", linewidths=0.35, zorder=2,
            )
            raw_rho, raw_p = (stats.spearmanr(x, y) if len(d) >= 3 else (np.nan, np.nan))
            ax.text(
                0.04, 0.96,
                f"raw Spearman ρ={raw_rho:.2f}\n{format_p(raw_p)}\n$n$={len(d)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
            ax.set_xlabel(PARAM_YLABELS["CC"] + r" pair mean")
            ax.set_ylabel(ylab)
            ax.set_title("Raw", fontsize=10, color="0.25")
            _style_pub_ax(ax)

            rho, p, n, yr, xr = _rank_partial_spearman(y, x, cov)
            ax = axes[r, 1]
            if n >= 5:
                # residual points inherit age colours of complete rows
                m = (
                    np.isfinite(y) & np.isfinite(x)
                    & np.isfinite(d["Distance_um"].to_numpy(dtype=float))
                    & np.isfinite(d["age_num"].to_numpy(dtype=float))
                )
                ax.scatter(
                    xr, yr, c=np.asarray(colors)[m], s=POINT_SIZE, alpha=POINT_ALPHA,
                    edgecolors="white", linewidths=0.35, zorder=2,
                )
            ax.axhline(0, color="0.55", lw=0.7, zorder=1)
            ax.axvline(0, color="0.55", lw=0.7, zorder=1)
            ax.text(
                0.04, 0.96,
                f"partial Spearman ρ={rho:.2f}\n{format_p(p)}\n$n$={n}"
                "\n"
                r"ctrl: Distance, age",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
            ax.set_xlabel(r"residual rank $CC$")
            ax.set_ylabel(r"residual rank $CC_a$")
            ax.set_title("Partial (rank residuals)", fontsize=10, color="0.25")
            _style_pub_ax(ax)
            print(
                f"  {ycol}: raw ρ={raw_rho:.3f} {format_p(raw_p)}; "
                f"partial ρ={rho:.3f} {format_p(p)}, n={n}"
            )

        cbar = add_age_colorbar(
            fig, list(axes.ravel()), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Partial Spearman: $CC_a$ vs $CC$ | Distance, age"
            "\n"
            r"MT excluded from age covariate; rank-residual method; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "32_CCa_CC_partial_Distance_age.png")


def plot_delay_delta_vs_drin(df: pd.DataFrame, out_dir: Path):
    """
    Signed Δ peak delay (and onset) vs ΔRin_recv — same template as fig 09/31.
      Δt_peak = t_peak,12 − t_peak,21
      ΔRin_recv = Rin_2 − Rin_1
    Both delay directions required (no zero-fill).
    """
    need = [*PARAM_DIRS["DelayPeak"], *PARAM_DIRS["DelayOnset"], "Rin_1", "Rin_2"]
    if any(c not in df.columns for c in need):
        print("Skipping Δdelay vs ΔRin: delay/Rin columns not found.")
        return
    sub = analysis_pool(df).copy()
    pk12, pk21 = PARAM_DIRS["DelayPeak"]
    on12, on21 = PARAM_DIRS["DelayOnset"]
    both_pk = sub[pk12].notna() & sub[pk21].notna()
    both_on = sub[on12].notna() & sub[on21].notna()
    sub["dPeak"] = np.where(both_pk, sub[pk12] - sub[pk21], np.nan)
    sub["dOnset"] = np.where(both_on, sub[on12] - sub[on21], np.nan)
    sub["dRin_recv"] = sub["Rin_2"] - sub["Rin_1"]

    vmin, max_age, mt_val, has_mt = age_limits(sub)
    age_ticks = unique_ages_for_colorbar(sub)
    # Peak first (requested); onset companion
    panels = [
        ("dPeak", r"$\Delta t_{peak}=t_{peak,12}-t_{peak,21}$ [ms]"),
        ("dOnset", r"$\Delta t_{onset}=t_{onset,12}-t_{onset,21}$ [ms]"),
    ]
    print("\nSigned Δdelay vs ΔRin_recv:")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        for ax, (col, ylabel) in zip(axes, panels):
            d = sub[sub[col].notna() & sub["dRin_recv"].notna()].copy()
            colors, _, _ = age_colors(
                d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, max_age, mt_val
            )
            x = d["dRin_recv"].to_numpy(dtype=float)
            y = d[col].to_numpy(dtype=float)
            xpad = max(float(np.nanmax(np.abs(x))) * 1.15, 1.0) if len(x) else 1.0
            ypad = max(float(np.nanmax(np.abs(y))) * 1.15, 0.1) if len(y) else 0.1
            ax.axhspan(0, ypad, xmin=0.5, xmax=1.0, color="0.92", zorder=0)
            ax.axhspan(-ypad, 0, xmin=0.0, xmax=0.5, color="0.92", zorder=0)
            ax.axhline(0, color="0.45", lw=0.8, zorder=1)
            ax.axvline(0, color="0.45", lw=0.8, zorder=1)
            if len(d):
                ax.scatter(
                    x, y, c=colors, s=POINT_SIZE, alpha=POINT_ALPHA,
                    edgecolors="white", linewidths=0.35, zorder=2,
                )
            same = (x * y) > 0
            nonzero = (x != 0) & (y != 0)
            n_nz = int(nonzero.sum())
            n_same = int((same & nonzero).sum())
            frac = n_same / n_nz if n_nz else np.nan
            binom_p = binom_pvalue(n_same, n_nz, 0.5) if n_nz >= 5 else np.nan
            rho, sp = stats.spearmanr(x, y) if len(d) >= 3 else (np.nan, np.nan)
            ax.text(
                0.04, 0.96,
                "\n".join(
                    s for s in [
                        f"same sign {n_same}/{n_nz}"
                        + (f" ({100 * frac:.0f}%)" if n_nz else ""),
                        f"binomial {format_p(binom_p)}",
                        f"Spearman ρ={rho:.2f}" if np.isfinite(rho) else "Spearman n/a",
                        format_p(sp) if np.isfinite(sp) else "",
                    ] if s
                ),
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
            )
            print(
                f"  {col}: same-sign {n_same}/{n_nz}, "
                f"binomial {format_p(binom_p)}; ρ={rho:.3f}, {format_p(sp)}"
            )
            ax.set_xlim(-xpad, xpad)
            ax.set_ylim(-ypad, ypad)
            ax.set_xlabel(r"$\Delta R_{in,recv}=R_{in,2}-R_{in,1}$ [M$\Omega$]")
            ax.set_ylabel(ylabel)
            _style_pub_ax(ax)

        cbar = add_age_colorbar(
            fig, list(axes), vmin=vmin, max_age=max_age, mt_val=mt_val,
            age_ticks=age_ticks, has_mt=has_mt,
        )
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.7)
        cbar.set_label("Age [postnatal days]", fontsize=9)
        fig.suptitle(
            r"Signed $\Delta$ delay vs receiver $\Delta R_{in}$"
            "\n"
            r"shaded: same sign; both delay directions required; "
            + POOL_NOTE,
            fontsize=11,
            color="0.15",
        )
        savefig(fig, out_dir, "34_dPeak_dOnset_vs_dRin.png")


def plot_delay_cca_followup_figures(df: pd.DataFrame, out_dir: Path):
    """Figs 28–32."""
    plot_delays_vs_rin_receiver(df, out_dir)
    plot_delay_onset_vs_peak(df, out_dir)
    plot_ai_delays_vs_ai_coupling(df, out_dir)
    plot_cca_directionality_vs_rin(df, out_dir)
    plot_cca_cc_partial_correlation(df, out_dir)


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
            "label": "02b AI by age group (overview + stats)",
            "outputs": [
                "02b_AI_by_age_group.png",
                "02b_AI_by_age_group_stats.png",
            ],
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
            "label": "06  Coupling vs Distance (plot + stats + 20 µm bins)",
            "outputs": [
                "06_coupling_vs_distance.png",
                "06b_coupling_vs_distance_stats.png",
                "06c_coupling_by_distance_bins.png",
            ],
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
            "label": "11  Distance/age coverage (with & without Exclude)",
            "outputs": ["11_distance_age_histograms.png"],
            "run": lambda: plot_coverage_histograms(df, out_dir),
        },
        {
            "id": "12",
            "label": "12  AI vs pair-mean strength (scatter + fixed-width bins)",
            "outputs": [
                "12_AI_vs_pair_mean_strength.png",
                "12b_AI_by_strength_bins.png",
            ],
            "run": lambda: plot_ai_vs_pair_mean(df, out_dir),
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
        {
            "id": "15",
            "label": "15  CCa vs CC/Gj (directed + means)",
            "outputs": [
                "15_CCa_vs_CC_Gj.png",
                "15b_CCa_vs_CC_Gj_pair_mean.png",
            ],
            "run": lambda: plot_cca_vs_cc_gj(df, out_dir),
        },
        {
            "id": "16",
            "label": "16  Delays vs CCa/CC/Gj (directed + means)",
            "outputs": [
                "16_delays_vs_CCa_CC_Gj.png",
                "16b_delays_vs_CCa_CC_Gj_pair_mean.png",
            ],
            "run": lambda: plot_delays_vs_cca_cc_gj(df, out_dir),
        },
        {
            "id": "17",
            "label": "17  CCa & delays vs Distance",
            "outputs": ["17_CCa_delays_vs_distance.png"],
            "run": lambda: plot_cca_delays_vs_distance(df, out_dir),
        },
        {
            "id": "18",
            "label": "18  CCa + delays vs age (gray points)",
            "outputs": [
                "18_CCa_delays_vs_age.png",
                "19_CCa_vs_age.png",
            ],
            "run": lambda: plot_cca_delays_vs_age(df, out_dir),
        },
        {
            "id": "15-18",
            "label": "15–18  All spike transmission (CCa & delays)",
            "outputs": [
                "15_CCa_vs_CC_Gj.png",
                "15b_CCa_vs_CC_Gj_pair_mean.png",
                "16_delays_vs_CCa_CC_Gj.png",
                "16b_delays_vs_CCa_CC_Gj_pair_mean.png",
                "17_CCa_delays_vs_distance.png",
                "18_CCa_delays_vs_age.png",
                "19_CCa_vs_age.png",
            ],
            "run": lambda: plot_spike_transmission_figures(df, out_dir),
        },
        {
            "id": "19",
            "label": "19  CCa+delays vs age + age-group KW",
            "outputs": [
                "18_CCa_delays_vs_age.png",
                "19_CCa_vs_age.png",
                "19b_CCa_by_age_group.png",
            ],
            "run": lambda: plot_cca_vs_age(df, out_dir),
        },
        {
            "id": "20",
            "label": "20  CCa coverage counts + % (age/dist/CC)",
            "outputs": [
                "20_CCa_coverage_counts.png",
                "20b_CCa_coverage_percent_age_distance.png",
                "20c_CCa_coverage_percent_by_CC_directed.png",
            ],
            "run": lambda: plot_cca_coverage(df, out_dir),
        },
        {
            "id": "21",
            "label": "21  CCa vs Rin_receiver (× = no CCa)",
            "outputs": ["21_CCa_vs_Rin_receiver.png"],
            "run": lambda: plot_cca_vs_rin_receiver(df, out_dir),
        },
        {
            "id": "22",
            "label": "22  CCa by day (+ Day1 vs Day2 stats)",
            "outputs": [
                "22_CCa_by_day.png",
                "22b_CCa_day1_vs_day2_stats.png",
            ],
            "run": lambda: plot_cca_by_day(df, out_dir),
        },
        {
            "id": "33",
            "label": "33  CCa by day for age P10–32 (+ stats)",
            "outputs": ["33_CCa_by_day_P10_32.png"],
            "run": lambda: plot_cca_by_day_p10_32(df, out_dir),
        },
        {
            "id": "19-21",
            "label": "19–21  CCa age, coverage, Rin_recv",
            "outputs": [
                "19_CCa_vs_age.png",
                "19b_CCa_by_age_group.png",
                "20_CCa_coverage_counts.png",
                "20b_CCa_coverage_percent_age_distance.png",
                "20c_CCa_coverage_percent_by_CC_directed.png",
                "21_CCa_vs_Rin_receiver.png",
            ],
            "run": lambda: plot_cca_coverage_and_rin_figures(df, out_dir),
        },
        {
            "id": "19-22",
            "label": "19–22  CCa age, coverage, Rin_recv, by day",
            "outputs": [
                "19_CCa_vs_age.png",
                "19b_CCa_by_age_group.png",
                "20_CCa_coverage_counts.png",
                "20b_CCa_coverage_percent_age_distance.png",
                "20c_CCa_coverage_percent_by_CC_directed.png",
                "21_CCa_vs_Rin_receiver.png",
                "22_CCa_by_day.png",
                "22b_CCa_day1_vs_day2_stats.png",
            ],
            "run": lambda: (
                plot_cca_coverage_and_rin_figures(df, out_dir),
                plot_cca_by_day(df, out_dir),
            ),
        },
        {
            "id": "23",
            "label": "23  CCa vs CC by age (Spearman)",
            "outputs": ["23_CCa_vs_CC_by_age.png"],
            "run": lambda: plot_cca_vs_cc_by_age(df, out_dir),
        },
        {
            "id": "24",
            "label": "24  CCa asymmetry max vs min (+ by age)",
            "outputs": [
                "24_CCa_asymmetry.png",
                "24b_CCa_asymmetry_by_age.png",
            ],
            "run": lambda: plot_cca_asymmetry(df, out_dir),
        },
        {
            "id": "25",
            "label": "25  Delay onset/peak Day1 vs Day2",
            "outputs": ["25_delays_day1_vs_day2.png"],
            "run": lambda: plot_delays_by_day(df, out_dir),
        },
        {
            "id": "26",
            "label": "26  CCa modality (none/one/both) by CC/Gj",
            "outputs": ["26_CCa_modality_by_conductance.png"],
            "run": lambda: plot_cca_modality_by_conductance(df, out_dir),
        },
        {
            "id": "27",
            "label": "27  AI_CCa vs AI_CC / AI_Gj",
            "outputs": ["27_AI_CCa_vs_AI_CC_Gj.png"],
            "run": lambda: plot_ai_cca(df, out_dir),
        },
        {
            "id": "23-27",
            "label": "23–27  CCa age Spearman, asym, delays day, modality, AI",
            "outputs": [
                "23_CCa_vs_CC_by_age.png",
                "24_CCa_asymmetry.png",
                "24b_CCa_asymmetry_by_age.png",
                "25_delays_day1_vs_day2.png",
                "26_CCa_modality_by_conductance.png",
                "27_AI_CCa_vs_AI_CC_Gj.png",
            ],
            "run": lambda: plot_cca_extended_figures(df, out_dir),
        },
        {
            "id": "28",
            "label": "28  Delays vs Rin_receiver",
            "outputs": ["28_delays_vs_Rin_receiver.png"],
            "run": lambda: plot_delays_vs_rin_receiver(df, out_dir),
        },
        {
            "id": "29",
            "label": "29  Delay onset vs peak",
            "outputs": ["29_delay_onset_vs_peak.png"],
            "run": lambda: plot_delay_onset_vs_peak(df, out_dir),
        },
        {
            "id": "30",
            "label": "30  AI delays vs AI_CC / AI_Gj",
            "outputs": ["30_AI_delays_vs_AI_CC_Gj.png"],
            "run": lambda: plot_ai_delays_vs_ai_coupling(df, out_dir),
        },
        {
            "id": "31",
            "label": "31  CCa directionality vs ΔRin",
            "outputs": ["31_CCa_directionality_vs_dRin.png"],
            "run": lambda: plot_cca_directionality_vs_rin(df, out_dir),
        },
        {
            "id": "32",
            "label": "32  Partial Spearman CCa–CC | Distance, age",
            "outputs": ["32_CCa_CC_partial_Distance_age.png"],
            "run": lambda: plot_cca_cc_partial_correlation(df, out_dir),
        },
        {
            "id": "28-32",
            "label": "28–32  Delays Rin/onset-peak/AI, CCa dir, partial CC",
            "outputs": [
                "28_delays_vs_Rin_receiver.png",
                "29_delay_onset_vs_peak.png",
                "30_AI_delays_vs_AI_CC_Gj.png",
                "31_CCa_directionality_vs_dRin.png",
                "32_CCa_CC_partial_Distance_age.png",
            ],
            "run": lambda: plot_delay_cca_followup_figures(df, out_dir),
        },
        {
            "id": "34",
            "label": "34  Δ peak/onset delay vs ΔRin",
            "outputs": ["34_dPeak_dOnset_vs_dRin.png"],
            "run": lambda: plot_delay_delta_vs_drin(df, out_dir),
        },
        {
            "id": "35",
            "label": "35  Delays vs Cm_recv and τ_recv",
            "outputs": ["35_delays_vs_Cm_Tau_receiver.png"],
            "run": lambda: plot_delays_vs_cm_tau_receiver(df, out_dir),
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
    print(f"Rows kept (Gj and/or Distance and/or age): {len(df)}")
    print(f"  with Gj both sides: {int(df['has_gj'].sum())}")
    print(f"Days present: {sorted(df['day'].dropna().unique().tolist())}")
    print(
        f"{MT_DISPLAY_LABEL} pairs (Excel {MT_EXCEL_TOKEN}): {int(df['is_mt'].sum())}"
    )
    excl_col = df.attrs.get("exclude_column")
    n_excl = int(df["excluded"].sum()) if "excluded" in df.columns else 0
    if excl_col:
        print(
            f"Exclude column: {excl_col!r} → Exclude?=yes: {n_excl} "
            "(kept for Distance/age coverage only)"
        )
    else:
        print(
            "Exclude column: not found "
            "(expected header Exclude? / Exclude). Treating all as included."
        )
    n_day3 = int(
        ((df["day"] == 3) & df["has_gj"] & ~df["excluded"]).sum()
    )
    print(
        f"Analysis pool (Days 1–2, not excluded, with Gj): {int(df['in_pool'].sum())} "
        f"(Day 3 omitted from asymmetry/AI/coupling: n={n_day3})"
    )

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
