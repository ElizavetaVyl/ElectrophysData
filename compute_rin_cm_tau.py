"""
Compute Rin, τ and Cm from ABF files (for cells missing Cm in Excel).

Logic matches the lab notebooks:
  - Rin_abs [MΩ]: slope of V–I (mV vs pA) × 1000 on subthreshold sweeps
  - τ [ms]: time to reach 63% of hyperpolarizing ΔV on a selected sweep
  - Cm [pF]: τ / Rin × 1000   ⇔   τ = Rin × Cm / 1000

Usage:
  python compute_rin_cm_tau.py --folder PATH_TO_ABFS
  python compute_rin_cm_tau.py --folder PATH --channels 0,2
  python compute_rin_cm_tau.py --file one_cell.abf --channels 0

Output Excel is written next to the folder/file:
  rin_cm_tau_results.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _import_deps():
    try:
        import pyabf
    except ImportError as e:
        raise SystemExit("Need pyabf: pip install pyabf") from e
    try:
        from scipy.signal import find_peaks
    except ImportError as e:
        raise SystemExit("Need scipy: pip install scipy") from e
    return pyabf, find_peaks


# ---------------------------------------------------------------------------
# Stimulus window helpers
# ---------------------------------------------------------------------------
def stim_windows(abf, voltage_channel: int) -> tuple[int, int, int, int, int]:
    """
    Return (stim_start, stim_end, pre_start, post_stop, within_start)
    using command waveform (same channel mapping as Cell Properties notebooks).
    """
    sample_rate = int(abf.dataRate)
    # ch0 voltage → sweepC on channel 0
    # ch2 voltage → sweepC on channel 1 (notebook quirk)
    if voltage_channel == 0:
        abf.setSweep(0, channel=0)
        stim = np.asarray(abf.sweepC)
    elif voltage_channel == 2:
        abf.setSweep(0, channel=1)
        stim = np.asarray(abf.sweepC)
    else:
        abf.setSweep(0, channel=voltage_channel)
        stim = np.asarray(abf.sweepC)

    nz = np.flatnonzero(np.abs(stim) > 1e-9)
    if len(nz) == 0:
        raise ValueError(f"No stimulus found for voltage channel {voltage_channel}")

    stim_start = int(nz[0])
    stim_end = int(nz[-1])
    pre_pts = int(0.125 * sample_rate)  # 125 ms baseline
    in_pts = int(0.25 * sample_rate)  # 250 ms late in pulse
    after_pts = int(0.5 * sample_rate)  # 500 ms after

    pre_start = max(0, stim_start - pre_pts)
    post_stop = min(len(stim) - 1, stim_end + after_pts)
    within_start = max(stim_start, stim_end - in_pts)
    return stim_start, stim_end, pre_start, post_stop, within_start


def last_subthreshold_sweep(
    abf,
    find_peaks,
    voltage_channel: int,
    stim_start: int,
    stim_end: int,
    ap_height: float = -10.0,
) -> int:
    """Last sweep index before the first AP (or last sweep if none)."""
    last = 0
    for sw in abf.sweepList:
        abf.setSweep(int(sw), channel=voltage_channel)
        peaks, _ = find_peaks(abf.sweepY[stim_start:stim_end], height=ap_height)
        if len(peaks) >= 1:
            return max(0, int(sw) - 1)
        last = int(sw)
    return last


# ---------------------------------------------------------------------------
# Rin
# ---------------------------------------------------------------------------
def compute_rin(
    abf,
    find_peaks,
    voltage_channel: int,
    current_channel: int,
    pre_start: int,
    stim_start: int,
    within_start: int,
    stim_end: int,
    v_lo: float = -85.0,
    v_hi: float = -50.0,
) -> dict:
    """
    Absolute Rin from linear fit of V_during vs I_during on subthreshold sweeps
    with V in [v_lo, v_hi]. Relative Rin = mean(ΔV/ΔI) in that range.
    """
    n_sub = last_subthreshold_sweep(
        abf, find_peaks, voltage_channel, stim_start, stim_end
    )
    voltages, currents, dV, dI, in_range = [], [], [], [], []

    for sw in range(n_sub + 1):
        abf.setSweep(sw, channel=voltage_channel)
        v_pre = float(np.mean(abf.sweepY[pre_start:stim_start]))
        v_dur = float(np.mean(abf.sweepY[within_start:stim_end]))
        abf.setSweep(sw, channel=current_channel)
        i_pre = float(np.mean(abf.sweepY[pre_start:stim_start]))
        i_dur = float(np.mean(abf.sweepY[within_start:stim_end]))

        voltages.append(v_dur)
        currents.append(i_dur)
        dV.append(v_dur - v_pre)
        dI.append(i_dur - i_pre)
        if v_lo <= v_dur <= v_hi:
            in_range.append(len(voltages) - 1)

    rin_rel = np.nan
    if in_range:
        ratios = [
            dV[idx] / dI[idx] for idx in in_range if abs(dI[idx]) > 0.15
        ]
        if ratios:
            rin_rel = float(np.round(np.nanmean(ratios) * 1000.0, 3))  # MΩ

    rin_abs = np.nan
    r2 = np.nan
    v_rest_fit = np.nan
    if len(in_range) >= 2:
        i_arr = np.asarray(currents, dtype=float)[in_range]
        v_arr = np.asarray(voltages, dtype=float)[in_range]
        p, residuals, *_ = np.polyfit(i_arr, v_arr, 1, full=True)
        slope, intercept = float(p[0]), float(p[1])
        rin_abs = float(np.round(slope * 1000.0, 3))  # mV/pA → MΩ
        v_rest_fit = float(np.round(intercept, 3))
        if residuals.size > 0:
            sst = float(np.sum((v_arr - np.mean(v_arr)) ** 2))
            r2 = float(np.round(1.0 - residuals[0] / sst, 3)) if sst > 0 else np.nan

    return {
        "Rin_abs_MOhm": rin_abs,
        "Rin_rel_MOhm": rin_rel,
        "R2_Rin": r2,
        "V_zero_I_mV": v_rest_fit,
        "n_IV_points": len(in_range),
        "n_subthreshold_sweeps": n_sub + 1,
    }


# ---------------------------------------------------------------------------
# τ and Cm
# ---------------------------------------------------------------------------
def compute_tau_cm(
    abf,
    voltage_channel: int,
    pre_start: int,
    post_stop: int,
    rin_mohm: float,
    search_range_mv: tuple[float, float] = (-35.0, -10.0),
) -> dict:
    """
    Pick hyperpolarizing sweep with ΔV in search_range (closest to 0),
    τ = time to 63% of ΔV, Cm = τ / Rin × 1000.
    """
    sample_rate = int(abf.dataRate)
    best_sw = None
    best_abs_dv = np.inf
    trace = None
    v_pre = v_min = np.nan

    pre_end = pre_start + int(0.125 * sample_rate)
    for sw in abf.sweepList:
        abf.setSweep(int(sw), channel=voltage_channel)
        y = np.asarray(abf.sweepY, dtype=float)
        vp = float(np.mean(y[pre_start:pre_end]))
        segment = y[pre_start:post_stop]
        vmin = float(np.min(segment))
        dv = vmin - vp
        if search_range_mv[0] <= dv <= search_range_mv[1]:
            if abs(dv) < best_abs_dv:
                best_abs_dv = abs(dv)
                best_sw = int(sw)
                trace = segment.copy()
                v_pre, v_min = vp, vmin

    if best_sw is None or trace is None:
        return {
            "tau_ms": np.nan,
            "Cm_pF": np.nan,
            "tau_sweep": np.nan,
            "delta_V_mV": np.nan,
            "note": "no sweep with ΔV in search_range",
        }

    v63 = v_pre + 0.63 * (v_min - v_pre)
    tau_samples = np.nan
    for i, v in enumerate(trace):
        if v <= v63:
            tau_samples = i
            break
    if not np.isfinite(tau_samples):
        return {
            "tau_ms": np.nan,
            "Cm_pF": np.nan,
            "tau_sweep": best_sw,
            "delta_V_mV": float(v_min - v_pre),
            "note": "63% level not reached",
        }

    tau_ms = float(tau_samples / sample_rate * 1000.0)
    cm_pf = np.nan
    if np.isfinite(rin_mohm) and rin_mohm != 0:
        cm_pf = float((tau_ms / rin_mohm) * 1000.0)

    return {
        "tau_ms": float(np.round(tau_ms, 3)),
        "Cm_pF": float(np.round(cm_pf, 3)) if np.isfinite(cm_pf) else np.nan,
        "tau_sweep": best_sw,
        "delta_V_mV": float(np.round(v_min - v_pre, 3)),
        "note": "",
    }


def analyze_channel(abf, find_peaks, voltage_channel: int) -> dict:
    current_channel = voltage_channel + 1
    if current_channel >= int(abf.channelCount):
        return {
            "voltage_channel": voltage_channel,
            "error": f"no current channel {current_channel}",
        }
    try:
        stim_start, stim_end, pre_start, post_stop, within_start = stim_windows(
            abf, voltage_channel
        )
    except ValueError as e:
        return {"voltage_channel": voltage_channel, "error": str(e)}

    rin = compute_rin(
        abf,
        find_peaks,
        voltage_channel,
        current_channel,
        pre_start,
        stim_start,
        within_start,
        stim_end,
    )
    rin_for_cm = rin["Rin_abs_MOhm"]
    if not np.isfinite(rin_for_cm):
        rin_for_cm = rin["Rin_rel_MOhm"]

    tau_cm = compute_tau_cm(abf, voltage_channel, pre_start, post_stop, rin_for_cm)

    tau_check = np.nan
    if np.isfinite(rin_for_cm) and np.isfinite(tau_cm["Cm_pF"]):
        tau_check = rin_for_cm * tau_cm["Cm_pF"] / 1000.0

    return {
        "voltage_channel": voltage_channel,
        "current_channel": current_channel,
        "sampling_Hz": int(abf.dataRate),
        **rin,
        "Rin_used_for_Cm_MOhm": rin_for_cm,
        **tau_cm,
        "tau_from_Rin_Cm_ms": (
            float(np.round(tau_check, 3)) if np.isfinite(tau_check) else np.nan
        ),
    }


def analyze_file(path: Path, channels: list[int], pyabf, find_peaks) -> list[dict]:
    abf = pyabf.ABF(str(path))
    rows = []
    for ch in channels:
        if ch >= int(abf.channelCount):
            rows.append(
                {
                    "file": path.name,
                    "abf_id": abf.abfID,
                    "voltage_channel": ch,
                    "error": "channel out of range",
                }
            )
            continue
        row = analyze_channel(abf, find_peaks, ch)
        row["file"] = path.name
        row["abf_id"] = abf.abfID
        rows.append(row)
        status = row.get("error") or row.get("note") or "ok"
        print(
            f"  {path.name}  ch{ch}: "
            f"Rin_abs={row.get('Rin_abs_MOhm', np.nan)}  "
            f"τ={row.get('tau_ms', np.nan)}  "
            f"Cm={row.get('Cm_pF', np.nan)}  [{status}]"
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Rin, τ, Cm from ABF (fill missing Cm)."
    )
    p.add_argument("--folder", default=None, help="Folder with .abf files")
    p.add_argument("--file", default=None, help="Single .abf file")
    p.add_argument(
        "--channels",
        default="0,2",
        help="Voltage channels to analyze (default: 0,2 for pairs)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output Excel path (default: <folder>/rin_cm_tau_results.xlsx)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pyabf, find_peaks = _import_deps()
    channels = [int(x.strip()) for x in args.channels.split(",") if x.strip() != ""]

    if args.file:
        files = [Path(args.file)]
        out_dir = files[0].parent
    elif args.folder:
        folder = Path(args.folder)
        files = sorted(folder.glob("*.abf")) + sorted(folder.glob("*.ABF"))
        files = sorted({f.resolve(): f for f in files}.values())
        out_dir = folder
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select folder with ABF files")
            root.destroy()
        except Exception:
            folder = input("Folder with ABF files: ").strip()
        if not folder:
            raise SystemExit("No folder selected.")
        folder = Path(folder)
        files = sorted(folder.glob("*.abf")) + sorted(folder.glob("*.ABF"))
        files = sorted({f.resolve(): f for f in files}.values())
        out_dir = folder

    if not files:
        raise SystemExit("No .abf files found.")

    print(f"Files: {len(files)}; channels: {channels}")
    rows: list[dict] = []
    for f in files:
        try:
            rows.extend(analyze_file(f, channels, pyabf, find_peaks))
        except Exception as e:
            print(f"  FAIL {f.name}: {e}")
            rows.append({"file": f.name, "error": str(e)})

    df = pd.DataFrame(rows)
    preferred = [
        "file",
        "abf_id",
        "voltage_channel",
        "Rin_abs_MOhm",
        "Rin_rel_MOhm",
        "Rin_used_for_Cm_MOhm",
        "tau_ms",
        "Cm_pF",
        "tau_from_Rin_Cm_ms",
        "delta_V_mV",
        "tau_sweep",
        "R2_Rin",
        "n_IV_points",
        "note",
        "error",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    df = df[cols]

    out_path = Path(args.out) if args.out else out_dir / "rin_cm_tau_results.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(
        "Fill Excel with: Rin_abs_MOhm → Rin; Cm_pF → Cm; "
        "tau_ms → τ (= Rin×Cm/1000)."
    )


if __name__ == "__main__":
    main()
