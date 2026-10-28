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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

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

PARAM_DIRS = {
    "Gj": ("Gj_12", "Gj_21"),
    "CC": ("CC_12", "CC_21"),
    "Rin": ("Rin_1", "Rin_2"),
    "Cm": ("Cm_1", "Cm_2"),
}


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
    # Prefer the plotting-only workbook if several General*.xlsx exist
    candidates = sorted(
        p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")
    )
    plotting = [
        p
        for p in candidates
        if "plotting" in p.name.lower().replace(" ", "_")
    ]
    if len(plotting) == 1:
        print(f"Using Excel: {plotting[0].name}")
        return plotting[0]
    fuzzy = [
        p
        for p in candidates
        if "general" in p.name.lower() and "plotting" in p.name.lower()
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
    # allow P42 / p42
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

    # Skip empty data rows (id may exist, but no Gj)
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
        MT_AGE_PLACEHOLDER if is_mt else age_num for age_num, is_mt in zip(df["age_num"], df["is_mt"])
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
        mx[both] = np.maximum(a[both], b[both])
        mn[both] = np.minimum(a[both], b[both])
        denom = mx + mn
        ai[both] = np.where(denom[both] != 0, (mx[both] - mn[both]) / denom[both], np.nan)
        ratio[both] = np.where(mn[both] != 0, mx[both] / mn[both], np.nan)
        df[f"{name}_max"] = mx
        df[f"{name}_min"] = mn
        df[f"AI_{name}"] = ai
        df[f"{name}_ratio"] = ratio

    df["in_pool"] = df["day"].isin(POOL_DAYS)
    return df.reset_index(drop=True)


def age_colors(age_plot: np.ndarray, is_mt: np.ndarray, vmin: float, vmax: float):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis
    colors = []
    for ap, mt in zip(age_plot, is_mt):
        if mt or (ap is None) or (isinstance(ap, float) and np.isnan(ap)):
            colors.append("#222222")
        else:
            colors.append(cmap(norm(ap)))
    return colors, norm, cmap


def add_age_colorbar(fig, norm, cmap, axs):
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label("Age (postnatal day)")
    return cbar


def savefig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: day comparison (X=day, Y=param; pair values linked)
# ---------------------------------------------------------------------------
def plot_day_comparison(df: pd.DataFrame, out_dir: Path):
    params = [
        ("Gj", "Gj_12", "Gj_21", "Gj"),
        ("CC", "CC_12", "CC_21", "CC"),
        ("Rin", "Rin_1", "Rin_2", r"$R_{in}$"),
    ]
    numeric_ages = df.loc[~df["is_mt"], "age_num"].dropna()
    vmin = float(numeric_ages.min()) if len(numeric_ages) else 0
    vmax = float(numeric_ages.max()) if len(numeric_ages) else 50
    if vmin == vmax:
        vmax = vmin + 1

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, constrained_layout=True)
    norm = cmap = None
    for ax, (name, c1, c2, ylabel) in zip(axes, params):
        if c1 not in df.columns or c2 not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[df[c1].notna() & df[c2].notna() & df["day"].notna()].copy()
        colors, norm, cmap = age_colors(
            sub["age_plot"].to_numpy(), sub["is_mt"].to_numpy(), vmin, vmax
        )
        for (_, row), color in zip(sub.iterrows(), colors):
            x = row["day"]
            y1, y2 = row[c1], row[c2]
            ax.plot([x, x], [y1, y2], color=color, alpha=0.35, lw=0.8, zorder=1)
            ax.scatter([x, x], [y1, y2], c=[color, color], s=18, alpha=0.85, zorder=2, edgecolors="none")
        ax.set_xlabel("Recording day after slice prep")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3])
        ax.set_title(name)
        ax.grid(True, alpha=0.25)

    if norm is not None and cmap is not None:
        add_age_colorbar(fig, norm, cmap, list(axes))
    mt_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=7, label="MT")
    axes[0].legend(handles=[mt_handle], loc="best", frameon=False)
    fig.suptitle("Within-pair values by recording day (color = animal age)")
    savefig(fig, out_dir, "01_day_comparison_Gj_CC_Rin.png")


# ---------------------------------------------------------------------------
# Figure 2: AI vs age (+ P50 zoom)
# ---------------------------------------------------------------------------
def plot_ai_vs_age(df: pd.DataFrame, out_dir: Path):
    # Pooled days 1–2 for summary age trends
    sub = df[df["in_pool"]].copy()
    metrics = [("AI_Gj", "AI Gj"), ("AI_CC", "AI CC"), ("AI_Rin", r"AI $R_{in}$"), ("AI_Cm", r"AI $C_m$")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna()]
        # non-MT
        d_num = d[~d["is_mt"]]
        ax.scatter(d_num["age_num"], d_num[col], c=d_num["age_num"], cmap="viridis", s=22, alpha=0.85)
        # MT at the end
        d_mt = d[d["is_mt"]]
        if len(d_mt):
            ax.axvline(MT_AGE_PLACEHOLDER - 40, color="0.5", ls="--", lw=1)
            ax.scatter(
                np.full(len(d_mt), MT_AGE_PLACEHOLDER),
                d_mt[col],
                c="#222222",
                s=28,
                alpha=0.9,
                label="MT",
                zorder=3,
            )
        ax.set_ylabel(title)
        ax.set_xlabel("Age (P)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        if len(d_mt):
            ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Asymmetry index vs age (days 1–2 only); MT at right of dashed line")
    savefig(fig, out_dir, "02_AI_vs_age.png")

    # Zoom ≤ P50
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes2 = axes2.ravel()
    for ax, (col, title) in zip(axes2, metrics):
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        d = sub[sub[col].notna() & (~sub["is_mt"]) & (sub["age_num"] <= P50_ZOOM_MAX)]
        ax.scatter(d["age_num"], d[col], c=d["age_num"], cmap="viridis", s=22, alpha=0.85)
        ax.set_ylabel(title)
        ax.set_xlabel("Age (P)")
        ax.set_xlim(0, P50_ZOOM_MAX)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    fig2.suptitle(f"AI vs age zoom (≤ P{P50_ZOOM_MAX}, days 1–2, MT excluded)")
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
    # least squares y = k x
    k = float(np.dot(x, y) / np.dot(x, x))
    return k


def plot_asymmetry_clouds(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    numeric_ages = sub.loc[~sub["is_mt"], "age_num"].dropna()
    vmin = float(numeric_ages.min()) if len(numeric_ages) else 0
    vmax = float(numeric_ages.max()) if len(numeric_ages) else 50
    if vmin == vmax:
        vmax = vmin + 1

    def _panel(ax, name, ylabel, with_fit: bool):
        xmin_c, xmax_c = f"{name}_min", f"{name}_max"
        if xmin_c not in sub.columns:
            ax.set_visible(False)
            return
        d = sub[sub[xmin_c].notna() & sub[xmax_c].notna()].copy()
        colors, norm, cmap = age_colors(d["age_plot"].to_numpy(), d["is_mt"].to_numpy(), vmin, vmax)
        ax.scatter(d[xmin_c], d[xmax_c], c=colors, s=28, alpha=0.85, edgecolors="none")
        lim_max = np.nanmax([d[xmin_c].max(), d[xmax_c].max()])
        lim_min = 0.0
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="y = x")
        if with_fit:
            k = _fit_slope_through_origin(d[xmin_c].to_numpy(), d[xmax_c].to_numpy())
            if k is not None:
                ax.plot([lim_min, lim_max], [lim_min, k * lim_max], color="crimson", lw=1.5, label=f"fit k={k:.2f}")
        ax.set_xlabel(f"{name} min")
        ax.set_ylabel(f"{name} max")
        ax.set_title(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        return norm, cmap

    # CC + Gj
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    norm = cmap = None
    for ax, name, title in zip(axes, ["Gj", "CC"], ["Gj asymmetry", "CC asymmetry"]):
        out = _panel(ax, name, title, with_fit=True)
        if out:
            norm, cmap = out
    if norm is not None:
        add_age_colorbar(fig, norm, cmap, list(axes))
    fig.suptitle("Asymmetry clouds (days 1–2): max vs min, color = age; MT dark")
    savefig(fig, out_dir, "03_asymmetry_Gj_CC.png")

    # Rin + Cm
    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for ax, name, title in zip(axes2, ["Rin", "Cm"], [r"$R_{in}$ asymmetry", r"$C_m$ asymmetry"]):
        out = _panel(ax, name, title, with_fit=False)
        if out:
            norm, cmap = out
    if norm is not None:
        add_age_colorbar(fig2, norm, cmap, list(axes2))
    fig2.suptitle(r"Cell-property asymmetry (days 1–2): $R_{in}$ and $C_m$")
    savefig(fig2, out_dir, "04_asymmetry_Rin_Cm.png")


# ---------------------------------------------------------------------------
# Figure 5: AI histograms
# ---------------------------------------------------------------------------
def plot_ai_histograms(df: pd.DataFrame, out_dir: Path):
    sub = df[df["in_pool"]].copy()
    metrics = [("AI_Gj", "AI Gj"), ("AI_CC", "AI CC"), ("AI_Rin", r"AI $R_{in}$"), ("AI_Cm", r"AI $C_m$")]
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
            ax.axvline(np.median(vals), color="darkorange", lw=1.5, ls="--", label=f"median={np.median(vals):.2f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("AI distributions (days 1–2)")
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
    plot_ai_vs_age(df, out_dir)
    plot_asymmetry_clouds(df, out_dir)
    plot_ai_histograms(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
