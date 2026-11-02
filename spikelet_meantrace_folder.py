"""
Folder batch: meantrace spikelet metrics with automatic peak vs shelf choice.

Run:
  python spikelet_meantrace_folder.py
or import analyze_folder() from a notebook.

For each ABF × direction (ch0→ch2, ch2→ch0):
  1) pick primary sweep (first with ≥4 APs, else 3, else 2, else 1)
  2) align AP2+ (AP1 only on 1-AP sweeps) on active AP start
  3) average active/passive snippets (meantrace)
  4) measure peak amp (local peak + prominence) AND plateau/step amp
  5) auto shape = peak | shelf | flat → chosen amp/delay
  6) Excel + over-time plot (amp spike, amp spikelet, ratio, delay)
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime

import numpy as np
import pyabf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

EXPECTED_CHANNELS = 4
SPIKE_HEIGHT = -10  # mV
SPIKE_DISTANCE = 10  # samples

BASELINE_MS = 1.0
WINDOW_MS = 15.0
SMOOTH_MS = 0.3
PEAK_PROMINENCE_MV = 0.05

# Plateau = mean passive in this late window after AP start (ms)
PLATEAU_START_MS = 8.0
PLATEAU_END_MS = 15.0

# Shape rules
SHELF_DROP_FRAC = 0.25  # if (Vmax-Vend)/peak_amp < this → treat as shelf even if peak found
MIN_PLATEAU_AMP_MV = 0.05
MIN_SPIKE_AMP_MV = 1.0

OUT_SUBDIR = "Spikelet_meantrace_peak_shelf"
DIRECTIONS = (
    ("ch0->ch2", 0, 2, "12"),
    ("ch2->ch0", 2, 0, "21"),
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _ms_to_samples(ms, sr):
    return max(1, int(round(float(ms) * float(sr) / 1000.0)))


def _samples_to_ms(n, sr):
    return float(n) * 1000.0 / float(sr)


def _round(v, nd=4):
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fv):
        return None
    return round(fv, nd)


def _stim_channel_for_voltage(voltage_ch):
    if voltage_ch == 0:
        return 0
    if voltage_ch == 2:
        return 1
    raise ValueError(f"unsupported voltage channel {voltage_ch}")


def channel_epochs(abf, voltage_ch):
    sr = int(abf.dataRate)
    points_pre = int(0.125 * sr)
    points_after = int(0.5 * sr)
    stim_ch = _stim_channel_for_voltage(voltage_ch)
    abf.setSweep(sweepNumber=0, channel=stim_ch)
    stim_idx = np.array(np.nonzero(abf.sweepC))[0]
    if len(stim_idx) == 0:
        return None
    start = int(stim_idx[0])
    stop = int(stim_idx[-1])
    return {
        "start": start,
        "stop": stop,
        "pre_start": start - points_pre,
        "post_stop": stop + points_after,
    }


def recording_datetime_str(abf):
    dt = getattr(abf, "abfDateTime", None)
    if dt is None:
        return getattr(abf, "abfDateTimeString", None)
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def cp_infl_points(abf, sweep, voltage_ch, start, stop):
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    smooth = gaussian_filter1d(abf.sweepY[start:stop], 3)
    smooth_d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    return infls + start


def cp_peak_indices(abf, sweep, voltage_ch, start, stop):
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    peaks, _ = find_peaks(
        abf.sweepY[start:stop], height=SPIKE_HEIGHT, distance=SPIKE_DISTANCE
    )
    return peaks + start


def cp_spike_begin(ind_peaks, ind_infls):
    starts = []
    for peak in ind_peaks:
        before = [i for i in ind_infls if i < peak]
        starts.append(before[-2] if len(before) >= 2 else np.nan)
    return starts


def select_primary_sweep(abf, voltage_ch, start, stop):
    """First sweep with ≥4 APs; else 3, else 2, else 1."""
    for min_aps, tier in ((4, ">=4"), (3, "3"), (2, "2"), (1, "1")):
        for sn in abf.sweepList:
            abf.setSweep(sweepNumber=sn, channel=voltage_ch)
            peaks, _ = find_peaks(
                abf.sweepY[start:stop], height=SPIKE_HEIGHT, distance=SPIKE_DISTANCE
            )
            if len(peaks) >= min_aps:
                return int(sn), int(len(peaks)), tier
    return None, 0, None


def _smooth(y, sr, smooth_ms=SMOOTH_MS):
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 5 or sr is None or smooth_ms <= 0:
        return y
    sigma = float(smooth_ms) * float(sr) / 1000.0
    if sigma < 0.5:
        return y
    return gaussian_filter1d(y, sigma=sigma, mode="nearest")


def _local_peak_index(seg, sr, prominence=PEAK_PROMINENCE_MV):
    y_raw = np.asarray(seg, dtype=float).ravel()
    if y_raw.size < 5:
        return None
    y = _smooth(y_raw, sr)
    peaks, _ = find_peaks(y, prominence=float(prominence))
    if peaks.size == 0:
        return None
    return int(peaks[int(np.argmax(y[peaks]))])


def _frac_time_ms(trace, i0, i_end, baseline, amp, sr, frac=0.5):
    """First time (ms from sample 0) when trace reaches baseline+frac*amp after i0."""
    if amp is None or amp == 0 or i_end is None or i_end <= i0:
        return None
    target = float(baseline) + float(frac) * float(amp)
    rising = float(amp) > 0
    seg = np.asarray(trace[i0:i_end + 1], dtype=float)
    for i, v in enumerate(seg):
        if rising and v >= target:
            return _samples_to_ms(i0 + i, sr)
        if (not rising) and v <= target:
            return _samples_to_ms(i0 + i, sr)
    return None


# -----------------------------------------------------------------------------
# Core meantrace metrics
# -----------------------------------------------------------------------------

def build_meantrace(abf, sweep, active_ch, passive_ch, stim_start, stim_stop, sr):
    """
    Align AP2+ (or AP1 if only one AP) to active AP start.
    Returns dict with mean_a, mean_p, n_pre, n_post, n_aligned, or reason.
    """
    n_pre = _ms_to_samples(BASELINE_MS, sr)
    n_post = _ms_to_samples(WINDOW_MS, sr)
    ind_peaks = cp_peak_indices(abf, sweep, active_ch, stim_start, stim_stop)
    ind_infls = cp_infl_points(abf, sweep, active_ch, stim_start, stim_stop)
    ap_starts = cp_spike_begin(ind_peaks, ind_infls)
    n_ap = int(len(ind_peaks))
    if n_ap < 1:
        return None, "no APs in stim window"

    use_ap1 = n_ap == 1
    ap_indices = range(0, 1) if use_ap1 else range(1, n_ap)

    abf.setSweep(sweepNumber=sweep, channel=active_ch)
    y_a = np.asarray(abf.sweepY, dtype=float)
    abf.setSweep(sweepNumber=sweep, channel=passive_ch)
    y_p = np.asarray(abf.sweepY, dtype=float)
    n_y = len(y_a)

    snips_a, snips_p = [], []
    for i in ap_indices:
        i_start = ap_starts[i] if i < len(ap_starts) else np.nan
        if not np.isfinite(i_start):
            continue
        i_start = int(i_start)
        if i_start < n_pre or i_start + n_post >= n_y:
            continue
        # skip if next/prev AP too close (ISI conflict inside window)
        if i + 1 < n_ap:
            nxt = ap_starts[i + 1] if i + 1 < len(ap_starts) else np.nan
            if np.isfinite(nxt) and int(nxt) < i_start + n_post:
                continue
        if i >= 1:
            prev = ap_starts[i - 1] if i - 1 < len(ap_starts) else np.nan
            if np.isfinite(prev) and (i_start - int(prev)) < (n_pre + n_post):
                continue
        snips_a.append(y_a[i_start - n_pre:i_start + n_post])
        snips_p.append(y_p[i_start - n_pre:i_start + n_post])

    if not snips_p:
        return None, "no alignable APs for meantrace"

    mean_a = np.mean(np.vstack(snips_a), axis=0)
    mean_p = np.mean(np.vstack(snips_p), axis=0)
    return {
        "mean_a": mean_a,
        "mean_p": mean_p,
        "snips_a": snips_a,
        "snips_p": snips_p,
        "n_pre": n_pre,
        "n_post": n_post,
        "n_aligned": len(snips_p),
        "n_ap": n_ap,
        "use_ap1": use_ap1,
    }, None


def measure_meantrace(mean_a, mean_p, n_pre, sr):
    """
    Peak + plateau metrics on meantrace; auto shape and chosen values.
    """
    out = {
        "shape": "flat",
        "amp_spike_mV": None,
        "amp_spikelet_peak_mV": None,
        "amp_spikelet_plateau_mV": None,
        "amp_spikelet_mV": None,
        "amp_ratio": None,
        "delay_ms": None,
        "delay_mode": None,
        "peak_prominence_mV": PEAK_PROMINENCE_MV,
        "plateau_window_ms": f"{PLATEAU_START_MS:g}-{PLATEAU_END_MS:g}",
        "skip_reason": None,
        "t_peak_active_ms": None,
        "t_peak_passive_ms": None,
    }
    if mean_a is None or mean_p is None:
        out["skip_reason"] = "no meantrace"
        return out

    ma = np.asarray(mean_a, dtype=float)
    mp = np.asarray(mean_p, dtype=float)
    i1 = min(len(mp), n_pre + _ms_to_samples(WINDOW_MS, sr) + 1)
    if i1 <= n_pre + 2:
        out["skip_reason"] = "window too short"
        return out
    ma = ma[:i1]
    mp = mp[:i1]
    mp_s = _smooth(mp, sr)

    # --- active spike amp on mean ---
    seg_a = ma[n_pre:]
    i_rel_a = _local_peak_index(seg_a, sr, prominence=max(0.5, PEAK_PROMINENCE_MV))
    if i_rel_a is None and len(seg_a):
        i_rel_a = int(np.argmax(seg_a))
    if i_rel_a is None:
        out["skip_reason"] = "no AP peak on meantrace"
        return out
    base_a = float(ma[n_pre])
    amp_a = float(seg_a[i_rel_a]) - base_a
    out["amp_spike_mV"] = _round(amp_a)
    out["t_peak_active_ms"] = _round(_samples_to_ms(i_rel_a, sr))
    if amp_a < MIN_SPIKE_AMP_MV:
        out["skip_reason"] = f"amp_spike < {MIN_SPIKE_AMP_MV} mV"
        return out

    # --- peak spikelet ---
    seg_p = mp[n_pre:]
    base_p = float(np.mean(mp_s[:n_pre])) if n_pre > 0 else float(mp_s[0])
    i_rel_p = _local_peak_index(seg_p, sr, prominence=PEAK_PROMINENCE_MV)
    amp_peak = None
    drop_frac = None
    if i_rel_p is not None:
        i_peak = n_pre + int(i_rel_p)
        amp_peak = float(mp_s[i_peak]) - base_p
        out["amp_spikelet_peak_mV"] = _round(amp_peak)
        out["t_peak_passive_ms"] = _round(_samples_to_ms(i_rel_p, sr))
        # drop from peak to end of window (on smoothed)
        vend = float(mp_s[-1])
        if amp_peak not in (None, 0):
            drop_frac = (float(mp_s[i_peak]) - vend) / abs(float(amp_peak))

    # --- plateau / step amp ---
    i_plat0 = n_pre + _ms_to_samples(PLATEAU_START_MS, sr)
    i_plat1 = n_pre + _ms_to_samples(PLATEAU_END_MS, sr)
    i_plat1 = min(i_plat1, len(mp))
    if i_plat1 > i_plat0 + 2:
        amp_plat = float(np.mean(mp[i_plat0:i_plat1])) - base_p
    else:
        amp_plat = float(np.mean(mp[n_pre:])) - base_p
    out["amp_spikelet_plateau_mV"] = _round(amp_plat)

    # --- shape decision ---
    peaked = (
        i_rel_p is not None
        and amp_peak is not None
        and amp_peak > 0
        and drop_frac is not None
        and drop_frac >= SHELF_DROP_FRAC
    )
    shelf_ok = amp_plat is not None and abs(amp_plat) >= MIN_PLATEAU_AMP_MV

    if peaked:
        shape = "peak"
        amp_s = amp_peak
        delay = _round(_samples_to_ms(int(i_rel_p) - int(i_rel_a), sr))
        delay_mode = "peak_passive - peak_active"
    elif shelf_ok:
        shape = "shelf"
        amp_s = amp_plat
        # delay by 50% rise of step on both cells
        i_end = len(mp) - 1
        t50_a = _frac_time_ms(ma, n_pre, n_pre + int(i_rel_a), base_a, amp_a, sr, 0.5)
        t50_p = _frac_time_ms(mp_s, n_pre, i_end, base_p, amp_plat, sr, 0.5)
        if t50_a is not None and t50_p is not None:
            delay = _round(t50_p - t50_a)
        else:
            delay = None
        delay_mode = "t50_passive - t50_active (plateau amp)"
    else:
        shape = "flat"
        amp_s = None
        delay = None
        delay_mode = None
        out["skip_reason"] = "no peak and plateau amp too small"
        out["shape"] = shape
        return out

    out["shape"] = shape
    out["amp_spikelet_mV"] = _round(amp_s)
    out["amp_ratio"] = _round(amp_s / amp_a) if amp_a else None
    out["delay_ms"] = delay
    out["delay_mode"] = delay_mode
    out["skip_reason"] = None
    return out


def analyze_direction(abf, active_ch, passive_ch, direction, tag):
    epochs = channel_epochs(abf, active_ch)
    if epochs is None:
        return {
            "direction": direction,
            "tag": tag,
            "skip_reason": "no stimulus in sweepC",
        }
    sr = int(abf.dataRate)
    stim_start, stim_stop = epochs["start"], epochs["stop"]
    sweep, n_ap, tier = select_primary_sweep(abf, active_ch, stim_start, stim_stop)
    if sweep is None:
        return {
            "direction": direction,
            "tag": tag,
            "skip_reason": "no sweep with APs",
        }
    mt, reason = build_meantrace(
        abf, sweep, active_ch, passive_ch, stim_start, stim_stop, sr,
    )
    row = {
        "direction": direction,
        "tag": tag,
        "sweep": sweep,
        "sweep_tier": tier,
        "n_AP_active": n_ap,
    }
    if mt is None:
        row["skip_reason"] = reason
        return row
    row["n_AP_aligned"] = mt["n_aligned"]
    row["ap1_fallback"] = mt["use_ap1"]
    metrics = measure_meantrace(mt["mean_a"], mt["mean_p"], mt["n_pre"], sr)
    row.update(metrics)
    row["_plot"] = {
        "mean_a": mt["mean_a"],
        "mean_p": mt["mean_p"],
        "snips_p": mt["snips_p"],
        "n_pre": mt["n_pre"],
        "sr": sr,
        "metrics": metrics,
    }
    return row


def analyze_abf(filepath):
    name = os.path.basename(filepath)
    abf = pyabf.ABF(filepath)
    rec_dt = recording_datetime_str(abf)
    rows = []
    if abf.channelCount != EXPECTED_CHANNELS:
        for direction, _a, _p, tag in DIRECTIONS:
            rows.append({
                "file": name,
                "recording_datetime": rec_dt,
                "direction": direction,
                "tag": tag,
                "skip_reason": f"expected {EXPECTED_CHANNELS} channels, got {abf.channelCount}",
            })
        return rows

    for direction, active, passive, tag in DIRECTIONS:
        r = analyze_direction(abf, active, passive, direction, tag)
        r["file"] = name
        r["recording_datetime"] = rec_dt
        rows.append(r)
    return rows


# -----------------------------------------------------------------------------
# QC + folder outputs
# -----------------------------------------------------------------------------

def _get_agg_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_qc_png(row, out_dir, stem):
    plot = row.get("_plot")
    if not plot:
        return None
    plt = _get_agg_plt()
    mean_a = plot["mean_a"]
    mean_p = plot["mean_p"]
    n_pre = plot["n_pre"]
    sr = plot["sr"]
    m = plot["metrics"]
    t = (np.arange(len(mean_p)) - n_pre) / float(sr) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sn in plot.get("snips_p") or []:
        if len(sn) == len(t):
            ax.plot(t, sn, color="0.6", lw=0.5, alpha=0.35)
    ax.plot(t, mean_p, color="C1", lw=2.2, label="mean passive")
    ax.plot(t, _smooth(mean_p, sr), color="darkorange", lw=1.2, label="smoothed")
    base = float(np.mean(_smooth(mean_p, sr)[:n_pre])) if n_pre else float(mean_p[0])
    ax.axhline(base, color="0.3", lw=0.8, label="baseline")
    ax.axvspan(-BASELINE_MS, 0, color="0.7", alpha=0.25)
    ax.axvspan(0, WINDOW_MS, color="C4", alpha=0.1)
    ax.axvspan(PLATEAU_START_MS, PLATEAU_END_MS, color="C2", alpha=0.12, label="plateau win")
    ax.axvline(0, color="limegreen", ls="--", lw=1)

    ax2 = ax.twinx()
    ax2.plot(t, mean_a, color="C0", lw=1.5, alpha=0.85, label="mean active")
    ax2.set_ylabel("Active Vm (mV)", color="C0")

    shape = m.get("shape")
    title = (
        f"{stem} {row.get('direction')} sweep {row.get('sweep')}  "
        f"shape={shape}  spike={m.get('amp_spike_mV')}  "
        f"spikelet={m.get('amp_spikelet_mV')}  ratio={m.get('amp_ratio')}  "
        f"delay={m.get('delay_ms')}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time from AP start (ms)")
    ax.set_ylabel("Passive Vm (mV)", color="C1")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_{row.get('tag')}_meantrace.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def _parse_dt(s):
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s), fmt)
        except ValueError:
            continue
    return None


def save_over_time_plot(rows, out_path, title=None):
    """4 panels: amp spike, amp spikelet, ratio, delay vs recording time."""
    plt = _get_agg_plt()
    # one point per file per direction
    by_dir = {"ch0->ch2": [], "ch2->ch0": []}
    for r in rows:
        if r.get("_plot") is not None:
            # strip plot payload if still present
            pass
        d = r.get("direction")
        if d not in by_dir:
            continue
        dt = _parse_dt(r.get("recording_datetime"))
        by_dir[d].append((dt, r))

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    if title:
        fig.suptitle(f"{title} — meantrace spikelet (auto peak/shelf)", fontsize=12)

    metrics = (
        ("amp_spike_mV", "AP amp (mV)"),
        ("amp_spikelet_mV", "Spikelet amp chosen (mV)"),
        ("amp_ratio", "Amp ratio (spikelet/spike)"),
        ("delay_ms", "Delay (ms)"),
    )
    styles = {
        "ch0->ch2": ("o-", "C0", "12"),
        "ch2->ch0": ("s-", "C1", "21"),
    }
    n_pts = 0
    for ax, (key, ylab) in zip(axes, metrics):
        for direction, items in by_dir.items():
            items = sorted(
                [(dt, r) for dt, r in items if dt is not None],
                key=lambda x: x[0],
            )
            if not items:
                continue
            xs = [dt for dt, _ in items]
            ys = []
            for _, r in items:
                v = r.get(key)
                try:
                    fv = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    fv = np.nan
                ys.append(fv)
            sty, color, lab = styles[direction]
            ax.plot(xs, ys, sty, color=color, label=lab)
            n_pts += sum(np.isfinite(ys))
            # shape annotation on spikelet amp panel
            if key == "amp_spikelet_mV":
                for x, y, (_, r) in zip(xs, ys, items):
                    if not np.isfinite(y):
                        continue
                    sh = r.get("shape") or "?"
                    ax.annotate(
                        sh[0].upper(), xy=(x, y),
                        textcoords="offset points", xytext=(3, 3),
                        fontsize=7, color=color, alpha=0.8,
                    )
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Recording time")
    fig.autofmt_xdate()
    try:
        fig.tight_layout()
    except Exception:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}  (points={n_pts})")
    return out_path


EXCEL_KEYS = (
    "file",
    "recording_datetime",
    "direction",
    "sweep",
    "sweep_tier",
    "n_AP_active",
    "n_AP_aligned",
    "ap1_fallback",
    "shape",
    "amp_spike_mV",
    "amp_spikelet_peak_mV",
    "amp_spikelet_plateau_mV",
    "amp_spikelet_mV",
    "amp_ratio",
    "delay_ms",
    "delay_mode",
    "peak_prominence_mV",
    "plateau_window_ms",
    "skip_reason",
)


def save_excel(path, rows):
    import pandas as pd

    clean = []
    for r in rows:
        clean.append({k: r.get(k) for k in EXCEL_KEYS})
    # wide summary: one row per file
    by_file = {}
    for r in clean:
        f = r["file"]
        by_file.setdefault(f, {
            "file": f,
            "recording_datetime": r.get("recording_datetime"),
        })
        tag = "12" if r.get("direction") == "ch0->ch2" else "21"
        for k in (
            "sweep", "shape", "amp_spike_mV", "amp_spikelet_peak_mV",
            "amp_spikelet_plateau_mV", "amp_spikelet_mV", "amp_ratio",
            "delay_ms", "delay_mode", "skip_reason", "n_AP_aligned",
        ):
            by_file[f][f"{k}_{tag}"] = r.get(k)

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(clean).to_excel(writer, sheet_name="All_directions", index=False)
        pd.DataFrame(list(by_file.values())).to_excel(
            writer, sheet_name="Summary_by_file", index=False,
        )
    print(f"Saved Excel: {path}")
    return path


def analyze_folder(folder_path, out_dir=None, save_qc=True):
    folder_path = os.path.abspath(folder_path)
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    if out_dir is None:
        out_dir = os.path.join(folder_path, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    abf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".abf"))
    if not abf_files:
        print("No .abf files in", folder_path)
        return None

    all_rows = []
    print(f"Folder: {folder_path}")
    print(f"Output: {out_dir}")
    print(f"Files: {len(abf_files)}")
    for fn in abf_files:
        fp = os.path.join(folder_path, fn)
        print(fn)
        try:
            rows = analyze_abf(fp)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows = [{
                "file": fn,
                "direction": "ch0->ch2",
                "tag": "12",
                "skip_reason": str(exc),
            }]
        stem = os.path.splitext(fn)[0]
        for r in rows:
            shape = r.get("shape")
            print(
                f"  {r.get('direction')}: sweep={r.get('sweep')} shape={shape} "
                f"spike={r.get('amp_spike_mV')} spikelet={r.get('amp_spikelet_mV')} "
                f"ratio={r.get('amp_ratio')} delay={r.get('delay_ms')} "
                f"skip={r.get('skip_reason')}"
            )
            if save_qc:
                try:
                    save_qc_png(r, out_dir, stem)
                except Exception as exc:
                    print(f"  QC plot error: {exc}")
            r.pop("_plot", None)
            all_rows.append(r)

    xlsx = os.path.join(out_dir, f"{folder_name}_spikelet_meantrace.xlsx")
    save_excel(xlsx, all_rows)
    png = os.path.join(out_dir, f"{folder_name}_spikelet_meantrace_over_time.png")
    save_over_time_plot(all_rows, png, title=folder_name)
    return {"excel": xlsx, "plot": png, "out_dir": out_dir, "rows": all_rows}


def _ask_folder():
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Select folder with .abf files")
        root.destroy()
        return path or None
    except Exception as exc:
        print("Folder dialog failed:", exc)
        return None


def main():
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else _ask_folder()
    if not folder:
        print("No folder selected.")
        return
    analyze_folder(folder)


if __name__ == "__main__":
    main()
