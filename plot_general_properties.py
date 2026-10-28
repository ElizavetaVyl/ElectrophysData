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
DAY_PAIR_OFFSET = 0.06  # left/right shift within a day tick
# light orange (young) -> dark purple (old / MT)
AGE_CMAP = LinearSegmentedColormap.from_list(
    "age_orange_purple",
    ["#FFE0B2", "#FF9800", "#E65100", "#8E24AA", "#4A148C"],
)
SHAPIRO_ALPHA = 0.05
DAY_TICK_LABELS = {1: "день 1", 2: "день 2", 3: "день 3"}

PARAM_DIRS = {
    "Gj": ("Gj_12", "Gj_21"),
    "CC": ("CC_12", "CC_21"),
    "Rin": ("Rin_1", "Rin_2"),
    "Cm": ("Cm_1", "Cm_2"),
}

PARAM_YLABELS = {
    "Gj": r"$G_j$ [nS]",
    "CC": "CC",
    "Rin": r"$R_{in}$ [M$\Omega$]",
    "Cm": r"$C_m$ [pF]",
}

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


def age_scale(df: pd.DataFrame) -> tuple[float, float]:
    numeric = df.loc[~df["is_mt"], "age_num"].dropna()
    vmin = float(numeric.min()) if len(numeric) else 0.0
    vmax = float(numeric.max()) if len(numeric) else 50.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def age_color_values(age_num: np.ndarray, is_mt: np.ndarray, vmax: float) -> np.ndarray:
    """MT maps to vmax so it gets the darkest color on the scale."""
    vals = []
    for a, mt in zip(age_num, is_mt):
        if mt:
            vals.append(vmax)
        elif a is None or (isinstance(a, float) and np.isnan(a)):
            vals.append(vmax)
        else:
            vals.append(float(a))
    return np.asarray(vals, dtype=float)


def age_colors(age_num: np.ndarray, is_mt: np.ndarray, vmin: float, vmax: float):
    norm = Normalize(vmin=vmin, vmax=vmax)
    color_vals = age_color_values(age_num, is_mt, vmax)
    colors = [AGE_CMAP(norm(v)) for v in color_vals]
    return colors, norm, AGE_CMAP


def unique_ages_for_colorbar(df: pd.DataFrame) -> list[float]:
    ages = sorted({float(a) for a in df.loc[~df["is_mt"], "age_num"].dropna().unique()})
    return ages


def add_age_colorbar(fig, norm, cmap, axs, age_ticks: list[float] | None = None, has_mt: bool = False):
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label("Age [P]" + (" (MT = darkest)" if has_mt else ""))
    if age_ticks:
        ticks = [t for t in age_ticks if norm.vmin <= t <= norm.vmax]
        # de-duplicate
        uniq = []
        seen = set()
        for t in ticks:
            key = round(float(t), 6)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(float(t))
        ticklabels = [
            str(int(t)) if abs(t - round(t)) < 1e-9 else f"{t:g}" for t in uniq
        ]
        if has_mt and uniq:
            # mark oldest tick as also MT color endpoint
            ticklabels[-1] = f"{ticklabels[-1]}/MT"
        if uniq:
            cbar.set_ticks(uniq)
            cbar.set_ticklabels(ticklabels)
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
    return {
        "n1": len(a),
        "n2": len(b),
        "mean1": float(np.mean(a)) if len(a) else np.nan,
        "mean2": float(np.mean(b)) if len(b) else np.nan,
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
# Figure 1: day comparison (X=day; pair left/right)
# ---------------------------------------------------------------------------
def plot_day_comparison(df: pd.DataFrame, out_dir: Path):
    params = [
        ("Gj", "Gj_12", "Gj_21"),
        ("CC", "CC_12", "CC_21"),
        ("Rin", "Rin_1", "Rin_2"),
        ("Cm", "Cm_1", "Cm_2"),
    ]
    vmin, vmax = age_scale(df)
    age_ticks = unique_ages_for_colorbar(df)
    has_mt = bool(df["is_mt"].any())

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2), sharex=True, constrained_layout=True)
    norm = cmap = None
    for ax, (name, c1, c2) in zip(axes, params):
        if c1 not in df.columns or c2 not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[df[c1].notna() & df[c2].notna() & df["day"].notna()].copy()
        colors, norm, cmap = age_colors(
            sub["age_num"].to_numpy(), sub["is_mt"].to_numpy(), vmin, vmax
        )
        for (_, row), color in zip(sub.iterrows(), colors):
            day = row["day"]
            x1 = day - DAY_PAIR_OFFSET
            x2 = day + DAY_PAIR_OFFSET
            y1, y2 = row[c1], row[c2]
            ax.plot([x1, x2], [y1, y2], color=color, alpha=0.4, lw=0.8, zorder=1)
            ax.scatter(
                [x1, x2],
                [y1, y2],
                c=[color, color],
                s=20,
                alpha=0.9,
                zorder=2,
                edgecolors="none",
            )
        # n ec per day instead of subplot name
        counts = []
        for day in (1, 2, 3):
            n = int(((sub["day"] == day).sum()))
            counts.append(f"n ec = {n}")
        ax.set_title("   |   ".join(counts), fontsize=9)
        ax.set_ylabel(PARAM_YLABELS[name])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([DAY_TICK_LABELS[1], DAY_TICK_LABELS[2], DAY_TICK_LABELS[3]])
        ax.set_xlabel("")
        ax.set_xlim(0.7, 3.3)
        ax.grid(True, alpha=0.25)

    if norm is not None and cmap is not None:
        add_age_colorbar(fig, norm, cmap, list(axes), age_ticks=age_ticks, has_mt=has_mt)
    fig.suptitle("Pairs by recording day")
    savefig(fig, out_dir, "01_day_comparison_Gj_CC_Rin_Cm.png")


# ---------------------------------------------------------------------------
# Figure 1b: day1 vs day2 statistics
# ---------------------------------------------------------------------------
def plot_day12_statistics(df: pd.DataFrame, out_dir: Path):
    """
    Two rows:
      1) pooled both sides {x1, x2} for day1 vs day2
      2) pair means (x1+x2)/2 for day1 vs day2
    Shapiro -> Welch or Mann-Whitney. No medians on plots.
    """
    params = [
        ("Gj", "Gj_12", "Gj_21", "Gj_pair_mean"),
        ("CC", "CC_12", "CC_21", "CC_pair_mean"),
        ("Rin", "Rin_1", "Rin_2", "Rin_pair_mean"),
        ("Cm", "Cm_1", "Cm_2", "Cm_pair_mean"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    fig.suptitle(
        "Day 1 vs Day 2\n"
        + POOL_FORMULA
        + r"  |  "
        + PAIR_MEAN_FORMULA
        + "\n"
        + TEST_RULE
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
            alpha=0.55,
            color="0.35",
        )
        ax.scatter(
            np.full(len(d2), 2) + np.random.default_rng(1).normal(0, 0.03, len(d2)),
            d2,
            s=14,
            alpha=0.55,
            color="0.35",
        )
        ax.plot(
            [1, 2],
            [res["mean1"], res["mean2"]],
            "o-",
            color="crimson",
            lw=1.8,
            markersize=7,
            zorder=3,
        )
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
            f"n={res['n1']}/{res['n2']}; means {res['mean1']:.3g}, {res['mean2']:.3g}"
        )
        ax.grid(True, alpha=0.25)

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
            alpha=0.65,
            color="0.35",
        )
        ax.scatter(
            np.full(len(s2), 2) + np.random.default_rng(3).normal(0, 0.03, len(s2)),
            s2,
            s=18,
            alpha=0.65,
            color="0.35",
        )
        ax.plot(
            [1, 2],
            [res["mean1"], res["mean2"]],
            "o-",
            color="crimson",
            lw=1.8,
            markersize=7,
            zorder=3,
        )
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
            f"n={res['n1']}/{res['n2']}; means {res['mean1']:.3g}, {res['mean2']:.3g}"
        )
        ax.grid(True, alpha=0.25)

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
        ("AI_Gj", "AI Gj"),
        ("AI_CC", "AI CC"),
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_m$"),
    ]
    vmin, vmax = age_scale(sub)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna()]
        d_num = d[~d["is_mt"]]
        colors_num, norm, cmap = age_colors(
            d_num["age_num"].to_numpy(), d_num["is_mt"].to_numpy(), vmin, vmax
        )
        ax.scatter(d_num["age_num"], d_num[col], c=colors_num, s=22, alpha=0.85, edgecolors="none")
        d_mt = d[d["is_mt"]]
        if len(d_mt):
            ax.axvline(MT_AGE_PLACEHOLDER - 40, color="0.5", ls="--", lw=1)
            mt_color = cmap(norm(vmax))
            ax.scatter(
                np.full(len(d_mt), MT_AGE_PLACEHOLDER),
                d_mt[col],
                c=[mt_color],
                s=28,
                alpha=0.9,
                label="MT",
                zorder=3,
                edgecolors="none",
            )
            ax.legend(frameon=False, loc="upper right")
        ax.set_ylabel(title)
        ax.set_xlabel("Age (P)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Asymmetry index vs age (days 1–2); MT at right of dashed line\n{AI_FORMULA}")
    savefig(fig, out_dir, "02_AI_vs_age.png")

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes2 = axes2.ravel()
    for ax, (col, title) in zip(axes2, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna() & (~sub["is_mt"]) & (sub["age_num"] <= P50_ZOOM_MAX)]
        colors_num, _, _ = age_colors(d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, vmax)
        ax.scatter(d["age_num"], d[col], c=colors_num, s=22, alpha=0.85, edgecolors="none")
        ax.set_ylabel(title)
        ax.set_xlabel("Age (P)")
        ax.set_xlim(0, P50_ZOOM_MAX)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
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
    vmin, vmax = age_scale(sub)

    def _panel(ax, name, ylabel, with_fit: bool, unit_note: str = ""):
        xmin_c, xmax_c = f"{name}_min", f"{name}_max"
        if xmin_c not in sub.columns:
            ax.set_visible(False)
            return
        d = sub[sub[xmin_c].notna() & sub[xmax_c].notna()].copy()
        colors, norm, cmap = age_colors(
            d["age_num"].to_numpy(), d["is_mt"].to_numpy(), vmin, vmax
        )
        ax.scatter(d[xmin_c], d[xmax_c], c=colors, s=28, alpha=0.85, edgecolors="none")
        lim_max = np.nanmax([d[xmin_c].max(), d[xmax_c].max()])
        lim_min = 0.0
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="y = x")
        if with_fit:
            k = _fit_slope_through_origin(d[xmin_c].to_numpy(), d[xmax_c].to_numpy())
            if k is not None:
                ax.plot(
                    [lim_min, lim_max],
                    [lim_min, k * lim_max],
                    color="crimson",
                    lw=1.5,
                    label=f"fit k={k:.2f}",
                )
        xlab = f"{name} min" + (f" [{unit_note}]" if unit_note else "")
        ylab = f"{name} max" + (f" [{unit_note}]" if unit_note else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        return norm, cmap

    age_ticks = unique_ages_for_colorbar(sub)
    has_mt = bool(sub["is_mt"].any())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    norm = cmap = None
    units = {"Gj": "nS", "CC": "", "Rin": r"M$\Omega$", "Cm": "pF"}
    for ax, name, title in zip(axes, ["Gj", "CC"], ["Gj asymmetry", "CC asymmetry"]):
        out = _panel(ax, name, title, with_fit=True, unit_note=units[name])
        if out:
            norm, cmap = out
    if norm is not None:
        add_age_colorbar(
            fig, norm, cmap, list(axes), age_ticks=age_ticks, has_mt=has_mt
        )
    fig.suptitle(f"Asymmetry clouds (days 1–2): max vs min\n{AI_FORMULA}")
    savefig(fig, out_dir, "03_asymmetry_Gj_CC.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for ax, name, title in zip(
        axes2, ["Rin", "Cm"], [r"$R_{in}$ asymmetry", r"$C_m$ asymmetry"]
    ):
        out = _panel(ax, name, title, with_fit=False, unit_note=units[name])
        if out:
            norm, cmap = out
    if norm is not None:
        add_age_colorbar(
            fig2, norm, cmap, list(axes2), age_ticks=age_ticks, has_mt=has_mt
        )
    fig2.suptitle(
        rf"Cell-property asymmetry (days 1–2): $R_{{in}}$ and $C_m$" + f"\n{AI_FORMULA}"
    )
    savefig(fig2, out_dir, "04_asymmetry_Rin_Cm.png")


# ---------------------------------------------------------------------------
# Figure 5: AI histograms
# ---------------------------------------------------------------------------
def plot_ai_histograms(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    metrics = [
        ("AI_Gj", "AI Gj"),
        ("AI_CC", "AI CC"),
        ("AI_Rin", r"AI $R_{in}$"),
        ("AI_Cm", r"AI $C_m$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
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

    plot_day_comparison(df, out_dir)
    plot_day12_statistics(df, out_dir)
    plot_ai_vs_age(df, out_dir)
    plot_asymmetry_clouds(df, out_dir)
    plot_ai_histograms(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
