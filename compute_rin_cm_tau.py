"""
Compute Rin, τ and Cm from ABF files (for cells missing Cm in Excel).

Ported from:
  - FWHM recalculation cuz of bad filtering.ipynb  (IV_Rin + tau_and_capacitance)
  - Cell properties Clean / for one folder          (same IV_Rin / channel windows)

Pipeline (as in FWHM batch cell):
  1) Rin_abs / Rin_rel from subthreshold IV (−85…−50 mV)
  2) τ from 63% ΔV on a hyperpolarizing sweep
  3) Cm = τ / Rin_rel × 1000

Usage:
  python compute_rin_cm_tau.py --folder PATH_TO_ABFS
  python compute_rin_cm_tau.py --file one_cell.abf --channels 0
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
# channel() — FWHM / Cell properties
# ---------------------------------------------------------------------------
def stim_windows(abf, voltage_channel: int) -> tuple[int, int, int, int, int]:
    """
    Return (stim_start, stim_end, pre_start, post_stop, within_start).

    Stimulus is read from sweepC:
      ch0 voltage → setSweep(..., channel=0)
      ch2 voltage → setSweep(..., channel=1)   # notebook quirk
    """
    sample_rate = int(abf.dataRate)
    n_ch = int(abf.channelCount)

    stim_start_ch0 = stim_end_ch0 = None
    stim_start_ch2 = stim_end_ch2 = None

    if n_ch >= 2:
        abf.setSweep(0, channel=0)
        nz0 = np.flatnonzero(np.asarray(abf.sweepC))
        if len(nz0) == 0:
            raise ValueError("No stimulus on command channel 0")
        stim_start_ch0, stim_end_ch0 = int(nz0[0]), int(nz0[-1])

    if n_ch == 4:
        abf.setSweep(0, channel=1)
        nz2 = np.flatnonzero(np.asarray(abf.sweepC))
        if len(nz2) == 0:
            raise ValueError("No stimulus on command channel 1 (for voltage ch2)")
        stim_start_ch2, stim_end_ch2 = int(nz2[0]), int(nz2[-1])

    pre_pts = int(0.125 * sample_rate)  # 125 ms baseline
    in_pts = int(0.25 * sample_rate)  # 250 ms late in pulse
    after_pts = int(0.5 * sample_rate)  # 500 ms after

    if voltage_channel == 0:
        if stim_start_ch0 is None:
            raise ValueError("Need ≥2 channels for voltage ch0")
        start, stop = stim_start_ch0, stim_end_ch0
    elif voltage_channel == 2:
        if stim_start_ch2 is None:
            raise ValueError("Need 4 channels for voltage ch2")
        start, stop = stim_start_ch2, stim_end_ch2
    else:
        raise ValueError(f"Unsupported voltage channel {voltage_channel} (use 0 or 2)")

    pre_start = start - pre_pts
    post_stop = stop + after_pts
    within_start = stop - in_pts
    return start, stop, pre_start, post_stop, within_start


def sweep_1AP(abf, find_peaks, voltage_channel: int, start: int, stop: int, v_level: float = -10.0) -> int:
    """
    Index of last subthreshold sweep (sweep before first AP), as in FWHM sweep_1AP.
    Returned value is used as exclusive upper bound in range(...) in IV_Rin.
    """
    last = 0
    for sweep_num in abf.sweepList:
        abf.setSweep(int(sweep_num), channel=voltage_channel)
        peaks, _ = find_peaks(abf.sweepY[start:stop], height=v_level)
        if len(peaks) >= 1:
            return int(sweep_num) - 1
        last = int(sweep_num)
    return last


# ---------------------------------------------------------------------------
# IV_Rin — FWHM / Cell properties (ΔI threshold 0.15 as in FWHM)
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
) -> dict:
    my_sweep = sweep_1AP(abf, find_peaks, voltage_channel, stim_start, stim_end)
    # FWHM: for sweepNumber in range(my_sweep)  → excludes the last subthreshold sweep
    voltages_list: list[float] = []
    currents_list: list[float] = []
    voltages_delta_list: list[float] = []
    currents_delta_list: list[float] = []
    voltage_range_ind_list: list[int] = []

    for sweep_index, sweep_number in enumerate(range(my_sweep)):
        abf.setSweep(sweep_number, channel=voltage_channel)
        v_pre = float(np.mean(abf.sweepY[pre_start:stim_start]))
        v_dur = float(np.mean(abf.sweepY[within_start:stim_end]))
        voltages_delta_list.append(v_dur - v_pre)
        voltages_list.append(v_dur)
        if -85.0 <= v_dur <= -50.0:
            voltage_range_ind_list.append(sweep_index)

        abf.setSweep(sweep_number, channel=current_channel)
        i_pre = float(np.mean(abf.sweepY[pre_start:stim_start]))
        i_dur = float(np.mean(abf.sweepY[within_start:stim_end]))
        currents_delta_list.append(i_dur - i_pre)
        currents_list.append(i_dur)

    rin_rel = np.nan
    if voltages_delta_list and currents_delta_list:
        rin_relative_list = [
            (x / y if abs(y) > 0.15 else np.nan)
            for x, y in zip(voltages_delta_list, currents_delta_list)
        ]
        if voltage_range_ind_list:
            rin_from_range = [rin_relative_list[i] for i in voltage_range_ind_list]
            rin_rel = float(np.round(np.nanmean(rin_from_range) * 1000.0, 3))

    rin_abs = np.nan
    r2 = np.nan
    v_zero = np.nan
    if len(voltage_range_ind_list) >= 2:
        i_arr = np.asarray(currents_list, dtype=float)[voltage_range_ind_list]
        v_arr = np.asarray(voltages_list, dtype=float)[voltage_range_ind_list]
        p, residuals, *_ = np.polyfit(i_arr, v_arr, 1, full=True)
        slope, intercept = float(p[0]), float(p[1])
        rin_abs = float(np.round(slope * 1000.0, 3))
        v_zero = float(np.round(intercept, 3))
        if residuals.size > 0:
            sst = float(np.sum((v_arr - np.mean(v_arr)) ** 2))
            r2 = float(np.round(1.0 - residuals[0] / sst, 3)) if sst != 0 else np.nan

    return {
        "Rin_abs_MOhm": rin_abs,
        "Rin_rel_MOhm": rin_rel,
        "R2_Rin": r2,
        "V_zero_I_mV": v_zero,
        "n_IV_points": len(voltage_range_ind_list),
        "n_sweeps_in_IV": my_sweep,  # exclusive upper bound as in notebook
    }


# ---------------------------------------------------------------------------
# tau_and_capacitance — FWHM (Cm uses Rin_rel)
# ---------------------------------------------------------------------------
def compute_tau_cm(
    abf,
    voltage_channel: int,
    window_start: int,
    window_stop: int,
    rin_rel_mohm: float,
    sampling_rate: int,
    search_range: tuple[float, float] = (-50.0, -10.0),
) -> dict:
    """
    FWHM call site passes (my_start, my_stop) = stim window, not pre/post.
    Sweep choice: ΔV in search_range, closest to search_range[1] (−10 mV).
    τ = time to 63% of ΔV; Cm = τ / Rin_rel × 1000.
    """
    best_sweep = None
    min_delta = np.inf
    trace = None
    v_pre = v_min = np.nan

    for sweep_num in range(len(abf.sweepList)):
        abf.setSweep(sweep_num, channel=voltage_channel)
        # FWHM literal: scalar at window edges (call uses stim_start / stim_end)
        voltage_pre = float(np.mean(abf.sweepY[window_start]))
        voltage_at_stop = float(np.min(abf.sweepY[window_stop]))
        delta_v = voltage_at_stop - voltage_pre

        if delta_v < 0 and search_range[0] <= delta_v <= search_range[1]:
            if abs(delta_v - search_range[1]) < min_delta:
                min_delta = abs(delta_v - search_range[1])
                best_sweep = sweep_num
                trace = np.asarray(abf.sweepY[window_start:window_stop], dtype=float).copy()
                v_pre, v_min = voltage_pre, voltage_at_stop

    if best_sweep is None or trace is None:
        return {
            "tau_ms": np.nan,
            "Cm_pF": np.nan,
            "tau_sweep": np.nan,
            "delta_V_mV": np.nan,
            "note": "no sweep with ΔV in search_range",
        }

    v_63 = v_pre + 0.63 * (v_min - v_pre)
    tau_samples = None
    for i, v in enumerate(trace):
        if v <= v_63:
            tau_samples = i
            break
    if tau_samples is None:
        return {
            "tau_ms": np.nan,
            "Cm_pF": np.nan,
            "tau_sweep": best_sweep,
            "delta_V_mV": float(np.round(v_min - v_pre, 3)),
            "note": "63% level not reached",
        }

    tau_ms = float(tau_samples / sampling_rate * 1000.0)
    cm_pf = np.nan
    if rin_rel_mohm is not None and not (isinstance(rin_rel_mohm, float) and np.isnan(rin_rel_mohm)):
        if np.isfinite(rin_rel_mohm) and rin_rel_mohm != 0:
            cm_pf = float((tau_ms / rin_rel_mohm) * 1000.0)

    return {
        "tau_ms": float(np.round(tau_ms, 3)),
        "Cm_pF": float(np.round(cm_pf, 3)) if np.isfinite(cm_pf) else np.nan,
        "tau_sweep": best_sweep,
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

    # FWHM: Cm from Rin_rel (not Rin_abs)
    tau_cm = compute_tau_cm(
        abf,
        voltage_channel,
        stim_start,  # my_start
        stim_end,  # my_stop
        rin["Rin_rel_MOhm"],
        int(abf.dataRate),
    )

    return {
        "voltage_channel": voltage_channel,
        "current_channel": current_channel,
        "sampling_Hz": int(abf.dataRate),
        **rin,
        "Rin_used_for_Cm_MOhm": rin["Rin_rel_MOhm"],
        **tau_cm,
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
            f"Rin_rel={row.get('Rin_rel_MOhm', np.nan)}  "
            f"τ={row.get('tau_ms', np.nan)}  "
            f"Cm={row.get('Cm_pF', np.nan)}  [{status}]"
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Rin, τ, Cm from ABF (FWHM / Cell properties logic)."
    )
    p.add_argument("--folder", default=None, help="Folder with .abf files")
    p.add_argument("--file", default=None, help="Single .abf file")
    p.add_argument(
        "--channels",
        default="0,2",
        help="Voltage channels (default: 0,2)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output Excel (default: <folder>/rin_cm_tau_results.xlsx)",
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
    print("Fill Excel: Rin_rel_MOhm (or Rin_abs) → Rin; Cm_pF → Cm; tau_ms → τ.")


if __name__ == "__main__":
    main()
