"""
Folder batch: meantrace spikelet metrics with automatic peak vs shelf choice.

Run:
  python spikelet_meantrace_folder.py
or import analyze_folder() from a notebook.

For each ABF × direction (ch0→ch2, ch2→ch0):
  1) pick primary sweep (first with ≥4 APs, else 3, else 2; 1-AP ignored)
  2) align AP2+ on active AP start
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
SMOOTH_MS = 0.5
PEAK_PROMINENCE_MV = 0.01

# Shelf step height = mean passive in this late window after AP start (ms)
PLATEAU_START_MS = 8.0
PLATEAU_END_MS = 15.0
# Spikelet / shelf onset from AP start: first time passive reaches this
# fraction of spikelet amplitude (foot of the rise — used for delay_onset)
SPIKELET_ONSET_FRAC = 0.20
# Late plateau landmark (QC / shelf amp context): ~arrival at flat level
SHELF_PLATEAU_START_FRAC = 0.90

# Shape rules: peak only if drop after max ≥ this fraction; else shelf
SHELF_DROP_FRAC = 0.02
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
    """First sweep with ≥4 APs; else 3, else 2. Single-AP sweeps are skipped."""
    for min_aps, tier in ((4, ">=4"), (3, "3"), (2, "2")):
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


def _shelf_plateau_start_rel(mp_s, n_pre, base_p, amp_step, sr, frac=None):
    """
    Index relative to AP start of the **start of the plateau** (not rise onset).

    Primary: first sample after t=0 that reaches
    baseline + SHELF_PLATEAU_START_FRAC * step_amp (default 90% of shelf height).

    Refinement: from that crossing, walk forward while |dV/dt| is still large,
    then take the first sample where the slope flattens (enter the shelf).
    Fallback if 90% never reached: max |dV/dt| sample (rise), then search flatten.
    """
    if frac is None:
        frac = SHELF_PLATEAU_START_FRAC
    if amp_step is None or abs(float(amp_step)) < 1e-9:
        return None
    amp_step = float(amp_step)
    y = np.asarray(mp_s, dtype=float)
    if len(y) <= n_pre + 3:
        return None
    target = float(base_p) + float(frac) * amp_step
    rising = amp_step > 0
    dy = np.diff(y)
    # noise scale for "flat" slope from pre-baseline derivative
    if n_pre >= 3:
        pre_dy = dy[max(0, n_pre - _ms_to_samples(BASELINE_MS, sr)):n_pre]
        slope_thr = max(1e-4, 3.0 * float(np.std(pre_dy))) if len(pre_dy) else 1e-3
    else:
        slope_thr = 1e-3

    i_cross = None
    for i in range(n_pre, len(y)):
        v = float(y[i])
        if rising and v >= target:
            i_cross = i
            break
        if (not rising) and v <= target:
            i_cross = i
            break

    if i_cross is None:
        # never reached 90%: start search after steepest rise
        seg_dy = dy[n_pre:]
        if rising:
            j = int(np.argmax(seg_dy))
        else:
            j = int(np.argmin(seg_dy))
        i_cross = n_pre + j

    # from crossing (or steepest), find first flattened sample
    for i in range(i_cross, len(y) - 1):
        s = float(dy[i])
        if rising and s <= slope_thr:
            return i - n_pre
        if (not rising) and s >= -slope_thr:
            return i - n_pre
    return i_cross - n_pre


def _amp_frac_onset_rel(mp_s, n_pre, base_p, amp, sr, frac=None):
    """
    First sample after AP start (n_pre) where passive reaches
    baseline + frac * amp_spikelet (default SPIKELET_ONSET_FRAC).

    This is the shelf / spikelet onset used for delay_onset.
    """
    if frac is None:
        frac = SPIKELET_ONSET_FRAC
    if amp is None or abs(float(amp)) < 1e-9:
        return None
    y = np.asarray(mp_s, dtype=float)
    if len(y) <= n_pre + 1:
        return None
    amp = float(amp)
    target = float(base_p) + float(frac) * amp
    rising = amp > 0
    for i in range(int(n_pre), len(y)):
        v = float(y[i])
        if rising and v >= target:
            return i - int(n_pre)
        if (not rising) and v <= target:
            return i - int(n_pre)
    return None


# -----------------------------------------------------------------------------
# Core meantrace metrics
# -----------------------------------------------------------------------------

def build_meantrace(abf, sweep, active_ch, passive_ch, stim_start, stim_stop, sr):
    """
    Align AP2+ to active AP start. Single-AP sweeps are not analyzed.
    Returns dict with mean_a, mean_p, n_pre, n_post, n_aligned, or reason.
    """
    n_pre = _ms_to_samples(BASELINE_MS, sr)
    n_post = _ms_to_samples(WINDOW_MS, sr)
    ind_peaks = cp_peak_indices(abf, sweep, active_ch, stim_start, stim_stop)
    ind_infls = cp_infl_points(abf, sweep, active_ch, stim_start, stim_stop)
    ap_starts = cp_spike_begin(ind_peaks, ind_infls)
    n_ap = int(len(ind_peaks))
    if n_ap < 2:
        return None, f"need ≥2 APs for spikelet meantrace (got {n_ap})"

    ap_indices = range(1, n_ap)  # AP1 excluded

    abf.setSweep(sweepNumber=sweep, channel=active_ch)
    y_a = np.asarray(abf.sweepY, dtype=float)
    abf.setSweep(sweepNumber=sweep, channel=passive_ch)
    y_p = np.asarray(abf.sweepY, dtype=float)
    n_y = len(y_a)

    snips_a, snips_p = [], []
    used_starts, used_peaks_a, used_ap_index = [], [], []
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
        used_starts.append(i_start)
        used_peaks_a.append(int(ind_peaks[i]))
        used_ap_index.append(i + 1)

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
        "used_starts": used_starts,
        "used_peaks_a": used_peaks_a,
        "used_ap_index": used_ap_index,
        "y_a": y_a,
        "y_p": y_p,
        "stim_start": stim_start,
        "stim_stop": stim_stop,
        "all_peaks_a": [int(p) for p in ind_peaks],
        "all_starts": [
            int(s) if np.isfinite(s) else None for s in ap_starts
        ],
    }, None


def measure_meantrace(mean_a, mean_p, n_pre, sr):
    """
    Peak + plateau metrics on meantrace; auto shape and chosen amp.

    Delays (both when available):
      delay_onset_ms = t(SPIKELET_ONSET_FRAC * amp from AP start) − t_AP_start
        AP start is t=0 by alignment; this is the primary delay.
      delay_peak_ms  = t_peak_passive − t_peak_active  (kept for comparison)
    """
    out = {
        "shape": "flat",
        "amp_spike_mV": None,
        "amp_spikelet_peak_mV": None,
        "amp_spikelet_plateau_mV": None,
        "amp_spikelet_mV": None,
        "amp_ratio": None,
        "delay_ms": None,
        "delay_onset_ms": None,
        "delay_peak_ms": None,
        "delay_mode": None,
        "peak_prominence_mV": PEAK_PROMINENCE_MV,
        "plateau_window_ms": f"{PLATEAU_START_MS:g}-{PLATEAU_END_MS:g}",
        "skip_reason": None,
        "t_peak_active_ms": None,
        "t_peak_passive_ms": None,
        "t_spikelet_onset_ms": None,
        "t_shelf_onset_ms": None,  # late plateau landmark (~90%)
        "spikelet_onset_frac": SPIKELET_ONSET_FRAC,
        "shelf_plateau_start_frac": SHELF_PLATEAU_START_FRAC,
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
    out["t_AP_onset_ms"] = 0.0  # snippets aligned to AP start
    if amp_a < MIN_SPIKE_AMP_MV:
        out["skip_reason"] = f"amp_spike < {MIN_SPIKE_AMP_MV} mV"
        return out

    # --- peak spikelet ---
    seg_p = mp[n_pre:]
    base_p = float(np.mean(mp_s[:n_pre])) if n_pre > 0 else float(mp_s[0])
    i_rel_p = _local_peak_index(seg_p, sr, prominence=PEAK_PROMINENCE_MV)
    amp_peak = None
    drop_frac = None
    i_peak_abs = None
    if i_rel_p is not None:
        i_peak_abs = n_pre + int(i_rel_p)
        amp_peak = float(mp_s[i_peak_abs]) - base_p
        out["amp_spikelet_peak_mV"] = _round(amp_peak)
        out["t_peak_passive_ms"] = _round(_samples_to_ms(i_rel_p, sr))
        out["i_peak_passive_rel"] = int(i_rel_p)
        # drop from peak to end of window (on smoothed)
        vend = float(mp_s[-1])
        if amp_peak not in (None, 0):
            drop_frac = (float(mp_s[i_peak_abs]) - vend) / abs(float(amp_peak))
    out["i_peak_active_rel"] = int(i_rel_a)
    out["baseline_passive_mV"] = _round(base_p)

    # --- plateau / step amp ---
    i_plat0 = n_pre + _ms_to_samples(PLATEAU_START_MS, sr)
    i_plat1 = n_pre + _ms_to_samples(PLATEAU_END_MS, sr)
    i_plat1 = min(i_plat1, len(mp))
    if i_plat1 > i_plat0 + 2:
        amp_plat = float(np.mean(mp[i_plat0:i_plat1])) - base_p
    else:
        amp_plat = float(np.mean(mp[n_pre:])) - base_p
    out["amp_spikelet_plateau_mV"] = _round(amp_plat)

    # --- shape decision (mutually exclusive: peak XOR shelf) ---
    shelf_ok = amp_plat is not None and abs(amp_plat) >= MIN_PLATEAU_AMP_MV
    # True peak = local max that drops enough toward the late window / end.
    # If late plateau stays close to the "peak", treat as shelf (bump on step).
    peaked = (
        i_rel_p is not None
        and amp_peak is not None
        and amp_peak > 0
        and drop_frac is not None
        and drop_frac >= SHELF_DROP_FRAC
    )
    if peaked and shelf_ok and amp_peak > 0:
        # residual plateau vs peak height
        residual = abs(amp_plat) / abs(amp_peak)
        if residual >= (1.0 - SHELF_DROP_FRAC):
            peaked = False

    if peaked:
        shape = "peak"
        amp_s = amp_peak
    elif shelf_ok:
        shape = "shelf"
        amp_s = amp_plat
    else:
        out["skip_reason"] = "no peak and plateau amp too small"
        out["shape"] = "flat"
        return out

    # Keep only the chosen mode's landmarks / amps for export + QC
    if shape == "peak":
        out["amp_spikelet_plateau_mV"] = None
        out["t_shelf_onset_ms"] = None
        out["i_shelf_onset_rel"] = None
    else:  # shelf
        out["amp_spikelet_peak_mV"] = None
        out["t_peak_passive_ms"] = None
        out["i_peak_passive_rel"] = None
        i_plat_rel = _shelf_plateau_start_rel(mp_s, n_pre, base_p, amp_plat, sr)
        if i_plat_rel is not None:
            out["t_shelf_onset_ms"] = _round(_samples_to_ms(int(i_plat_rel), sr))
            out["i_shelf_onset_rel"] = int(i_plat_rel)

    # --- spikelet onset: frac of spikelet amp from AP start (primary delay) ---
    i_on_p = _amp_frac_onset_rel(mp_s, n_pre, base_p, amp_s, sr, frac=SPIKELET_ONSET_FRAC)
    delay_onset = None
    if i_on_p is not None:
        t_on_p = _samples_to_ms(int(i_on_p), sr)
        out["t_spikelet_onset_ms"] = _round(t_on_p)
        out["i_spikelet_onset_rel"] = int(i_on_p)
        # from AP start (t=0) → just the onset time
        delay_onset = _round(t_on_p)
        out["delay_onset_ms"] = delay_onset

    # --- peak-to-peak delay only for shape=peak ---
    delay_peak = None
    if shape == "peak" and i_rel_p is not None:
        delay_peak = _round(
            _samples_to_ms(int(i_rel_p) - int(i_rel_a), sr)
        )
        out["delay_peak_ms"] = delay_peak

    # primary delay = onset (frac * amp from AP start)
    if delay_onset is not None:
        delay = delay_onset
        delay_mode = (
            f"delay_onset = t({SPIKELET_ONSET_FRAC:g}*amp_spikelet from AP start)"
        )
    elif delay_peak is not None:
        delay = delay_peak
        delay_mode = "delay_peak = peak_passive - peak_active (onset missing)"
    else:
        delay = None
        delay_mode = None

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
            "skip_reason": "no sweep with ≥2 APs (1-AP not analyzed)",
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
    metrics = measure_meantrace(mt["mean_a"], mt["mean_p"], mt["n_pre"], sr)
    row.update(metrics)
    row["_plot"] = {
        "mean_a": mt["mean_a"],
        "mean_p": mt["mean_p"],
        "snips_a": mt["snips_a"],
        "snips_p": mt["snips_p"],
        "n_pre": mt["n_pre"],
        "n_post": mt["n_post"],
        "sr": sr,
        "metrics": metrics,
        "used_starts": mt["used_starts"],
        "used_peaks_a": mt["used_peaks_a"],
        "used_ap_index": mt["used_ap_index"],
        "y_a": mt["y_a"],
        "y_p": mt["y_p"],
        "stim_start": mt["stim_start"],
        "stim_stop": mt["stim_stop"],
        "all_peaks_a": mt["all_peaks_a"],
        "all_starts": mt["all_starts"],
        "active_ch": active_ch,
        "passive_ch": passive_ch,
        "sweep": sweep,
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


def _draw_main_track(ax_a, ax_p, row):
    """Main sweep: active/passive with used APs and detected spikelet peaks."""
    plot = row.get("_plot")
    if not plot:
        ax_a.text(0.5, 0.5, row.get("skip_reason") or "no plot",
                  transform=ax_a.transAxes, ha="center", va="center", color="C3")
        ax_p.axis("off")
        return
    sr = plot["sr"]
    y_a = np.asarray(plot["y_a"], dtype=float)
    y_p = np.asarray(plot["y_p"], dtype=float)
    t = np.arange(len(y_a)) / float(sr)
    ss, se = int(plot["stim_start"]), int(plot["stim_stop"])
    pad = int(0.05 * sr)
    i0 = max(ss - pad, 0)
    i1 = min(se + pad, len(t) - 1)

    ax_a.plot(t[i0:i1], y_a[i0:i1], color="C0", lw=0.9, label=f"active ch{plot['active_ch']}")
    ax_p.plot(t[i0:i1], y_p[i0:i1], color="C1", lw=0.9, label=f"passive ch{plot['passive_ch']}")
    for ax in (ax_a, ax_p):
        ax.axvline(t[min(ss, len(t) - 1)], color="0.4", ls="--", lw=0.8)
        ax.axvline(t[min(se, len(t) - 1)], color="0.4", ls="--", lw=0.8)

    used_set = set(plot.get("used_starts") or [])
    # unused AP starts (gray), used (green) + AP peaks (blue)
    labeled_u = labeled_used = labeled_pk = labeled_sp = False
    labeled_ap_line = False
    for st, pk in zip(plot.get("all_starts") or [], plot.get("all_peaks_a") or []):
        if st is None:
            continue
        if st in used_set:
            continue
        if 0 <= st < len(t):
            ax_a.scatter(
                t[st], y_a[st], c="0.55", s=22, marker="v", zorder=4,
                label="AP start (not used)" if not labeled_u else None,
            )
            labeled_u = True

    m = plot.get("metrics") or {}
    t_peak_p_ms = m.get("t_peak_passive_ms")
    t_on_p_ms = m.get("t_spikelet_onset_ms")
    shape = m.get("shape")
    labeled_on = False
    for st, pk, ap_i in zip(
        plot.get("used_starts") or [],
        plot.get("used_peaks_a") or [],
        plot.get("used_ap_index") or [],
    ):
        if 0 <= st < len(t):
            ax_a.scatter(
                t[st], y_a[st], c="limegreen", s=48, marker="v", zorder=6,
                edgecolors="k", linewidths=0.4,
                label="AP start used" if not labeled_used else None,
            )
            # also mark AP start on passive panel + vertical guide
            ax_p.scatter(
                t[st], y_p[st], c="limegreen", s=36, marker="v", zorder=6,
                edgecolors="k", linewidths=0.3,
                label="AP start" if not labeled_used else None,
            )
            ax_a.axvline(
                t[st], color="limegreen", ls="--", lw=0.8, alpha=0.7,
                label="AP start line" if not labeled_ap_line else None,
            )
            ax_p.axvline(t[st], color="limegreen", ls="--", lw=0.8, alpha=0.7)
            labeled_ap_line = True
            labeled_used = True
            # analysis window on passive
            t0 = t[st]
            ax_p.axvspan(
                t0, t0 + WINDOW_MS / 1000.0, color="C4", alpha=0.12,
            )
        if 0 <= pk < len(t):
            ax_a.scatter(
                t[pk], y_a[pk], c="C0", s=30, marker="o", zorder=6,
                edgecolors="k", linewidths=0.3,
                label="AP peak used" if not labeled_pk else None,
            )
            labeled_pk = True
        # only the chosen shape marker (+ shared frac onset)
        _on_lbl = f"onset {SPIKELET_ONSET_FRAC:.0%} amp"
        if t_on_p_ms is not None and 0 <= st < len(t):
            io = st + _ms_to_samples(t_on_p_ms, sr)
            if 0 <= io < len(t):
                ax_p.scatter(
                    t[io], y_p[io], c="purple", s=36, marker="^", zorder=7,
                    label=_on_lbl if not labeled_on else None,
                )
                labeled_on = True
        if shape == "peak" and t_peak_p_ms is not None and 0 <= st < len(t):
            ip = st + _ms_to_samples(t_peak_p_ms, sr)
            if 0 <= ip < len(t):
                ax_p.scatter(
                    t[ip], y_p[ip], c="darkorange", s=42, marker="x", zorder=7,
                    label="spikelet peak" if not labeled_sp else None,
                )
                labeled_sp = True
        elif shape == "shelf" and m.get("t_shelf_onset_ms") is not None and 0 <= st < len(t):
            ip = st + _ms_to_samples(m["t_shelf_onset_ms"], sr)
            if 0 <= ip < len(t):
                ax_p.scatter(
                    t[ip], y_p[ip], c="C2", s=42, marker="D", zorder=7,
                    label="plateau start" if not labeled_sp else None,
                )
                labeled_sp = True

    ax_a.set_ylabel("Active Vm (mV)")
    ax_p.set_ylabel("Passive Vm (mV)")
    ax_a.set_xlim(t[i0], t[i1])
    ax_p.set_xlim(t[i0], t[i1])
    ax_a.legend(loc="upper right", fontsize=7)
    ax_p.legend(loc="upper right", fontsize=7)
    ax_a.grid(True, alpha=0.25)
    ax_p.grid(True, alpha=0.25)
    ax_a.set_title(
        f"{row.get('direction')}  sweep {row.get('sweep')} ({row.get('sweep_tier')})  "
        f"used AP={row.get('n_AP_aligned')}  shape={shape}",
        fontsize=10,
    )


def _draw_meantrace(ax, row):
    plot = row.get("_plot")
    if not plot:
        ax.text(0.5, 0.5, row.get("skip_reason") or "no meantrace",
                transform=ax.transAxes, ha="center", va="center", color="C3")
        return
    mean_a = plot["mean_a"]
    mean_p = plot["mean_p"]
    n_pre = plot["n_pre"]
    sr = plot["sr"]
    m = plot.get("metrics") or {}
    t = (np.arange(len(mean_p)) - n_pre) / float(sr) * 1000.0
    mp_s = _smooth(mean_p, sr)

    for sn in plot.get("snips_p") or []:
        if len(sn) == len(t):
            ax.plot(t, sn, color="0.6", lw=0.5, alpha=0.3)
    ax.plot(t, mean_p, color="C1", lw=2.2, label="mean passive")
    ax.plot(t, mp_s, color="darkorange", lw=1.2, label="smoothed")
    base = float(np.mean(mp_s[:n_pre])) if n_pre else float(mp_s[0])
    ax.axhline(base, color="0.3", lw=0.8, label="baseline")
    ax.axvspan(-BASELINE_MS, 0, color="0.7", alpha=0.25)
    ax.axvspan(0, WINDOW_MS, color="C4", alpha=0.1)
    ax.axvspan(PLATEAU_START_MS, PLATEAU_END_MS, color="C2", alpha=0.12, label="plateau win")
    ax.axvline(0, color="limegreen", ls="--", lw=1.2, label="AP start")

    # mark onset + only the chosen shape landmark
    shape = m.get("shape")
    if m.get("t_spikelet_onset_ms") is not None:
        ton_sp = float(m["t_spikelet_onset_ms"])
        io = n_pre + _ms_to_samples(ton_sp, sr)
        io = min(max(io, 0), len(mp_s) - 1)
        _on_lbl = f"onset {SPIKELET_ONSET_FRAC:.0%} amp"
        ax.scatter(
            t[io], mp_s[io], c="purple", s=70, marker="^", zorder=8,
            edgecolors="k", linewidths=0.5, label=_on_lbl,
        )
        ax.axvline(ton_sp, color="purple", ls="--", lw=1.0)
        ax.annotate(
            f"{_on_lbl} {ton_sp:.2f} ms",
            (t[io], mp_s[io]), textcoords="offset points", xytext=(6, -12),
            fontsize=8, color="purple",
        )
    if shape == "peak" and m.get("t_peak_passive_ms") is not None:
        tp = float(m["t_peak_passive_ms"])
        ip = n_pre + _ms_to_samples(tp, sr)
        ip = min(max(ip, 0), len(mp_s) - 1)
        ax.scatter(
            t[ip], mp_s[ip], c="darkorange", s=70, marker="o", zorder=8,
            edgecolors="k", linewidths=0.5, label="spikelet peak",
        )
        ax.axvline(tp, color="darkorange", ls=":", lw=1.0)
        if m.get("amp_spikelet_peak_mV") is not None:
            ax.annotate(
                f"peak {m['amp_spikelet_peak_mV']:.2f} mV",
                (t[ip], mp_s[ip]), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="0.15",
            )
    elif shape == "shelf" and m.get("t_shelf_onset_ms") is not None:
        ton = float(m["t_shelf_onset_ms"])
        ip = n_pre + _ms_to_samples(ton, sr)
        ip = min(max(ip, 0), len(mp_s) - 1)
        ax.scatter(
            t[ip], mp_s[ip], c="C2", s=55, marker="D", zorder=7,
            edgecolors="k", linewidths=0.5, label="plateau start",
        )
        ax.axvline(ton, color="C2", ls=":", lw=0.9)

    ax2 = ax.twinx()
    ax2.plot(t, mean_a, color="C0", lw=1.4, alpha=0.85, label="mean active")
    # AP start at t=0 (alignment reference)
    i0_ap = min(max(n_pre, 0), len(mean_a) - 1)
    ax2.scatter(
        t[i0_ap], mean_a[i0_ap], c="limegreen", s=70, marker="v", zorder=9,
        edgecolors="k", linewidths=0.5, label="AP start",
    )
    if m.get("t_peak_active_ms") is not None:
        ta = float(m["t_peak_active_ms"])
        ia = n_pre + _ms_to_samples(ta, sr)
        ia = min(max(ia, 0), len(mean_a) - 1)
        ax2.scatter(
            t[ia], mean_a[ia], c="C0", s=55, marker="o", zorder=8,
            edgecolors="k", linewidths=0.4, label="AP peak",
        )
    ax2.set_ylabel("Active Vm (mV)", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    ax.set_xlabel("Time from AP start (ms)")
    ax.set_ylabel("Passive Vm (mV)", color="C1")
    ax.set_title(
        f"Meantrace  shape={shape}  spike={m.get('amp_spike_mV')}  "
        f"spikelet={m.get('amp_spikelet_mV')}  ratio={m.get('amp_ratio')}  "
        f"d_on={m.get('delay_onset_ms')}  d_pk={m.get('delay_peak_ms')}",
        fontsize=9,
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


def save_file_qc_png(rows_12_21, out_dir, stem):
    """
    One QC figure per file: directions 12 and 21 stacked.
    Each direction: main sweep track (active+passive) + meantrace with detected peak.
    """
    by_tag = {}
    for r in rows_12_21:
        by_tag[r.get("tag")] = r
    if not by_tag:
        return None

    plt = _get_agg_plt()
    fig = plt.figure(figsize=(14, 12))
    # 2 directions × (active, passive, meantrace) = 6 rows
    gs = fig.add_gridspec(6, 1, height_ratios=[1.0, 1.0, 1.2, 1.0, 1.0, 1.2], hspace=0.35)
    order = (("12", "ch0→ch2"), ("21", "ch2→ch0"))
    for di, (tag, label) in enumerate(order):
        row = by_tag.get(tag) or {
            "tag": tag, "direction": label, "skip_reason": "missing direction",
        }
        ax_a = fig.add_subplot(gs[di * 3 + 0])
        ax_p = fig.add_subplot(gs[di * 3 + 1], sharex=ax_a)
        ax_m = fig.add_subplot(gs[di * 3 + 2])
        _draw_main_track(ax_a, ax_p, row)
        _draw_meantrace(ax_m, row)
        ax_a.set_title(
            f"{stem}  |  {row.get('direction') or label}",
            fontsize=11, fontweight="bold",
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_spikelet_QC.png")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")
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


def _file_tail_digits(name, n=4):
    """Last digits of filename stem (for over-time labels)."""
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits[-n:] if len(digits) >= n else digits
    return stem[-n:] if stem else "?"


def _as_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _scrub_negative_metrics(row):
    """
    Drop negative amp/ratio/delay values (invalid for this analysis).
    Returns list of (key, bad_value) that were cleared — for red-X plotting.
    """
    keys = (
        "amp_spike_mV",
        "amp_spikelet_mV",
        "amp_spikelet_peak_mV",
        "amp_spikelet_plateau_mV",
        "amp_ratio",
        "delay_ms",
        "delay_onset_ms",
        "delay_peak_ms",
    )
    cleared = []
    for k in keys:
        v = _as_float(row.get(k))
        if np.isfinite(v) and v < 0:
            cleared.append((k, float(v)))
            row[k] = None
    if cleared and row.get("skip_reason") is None:
        row["negatives_cleared"] = ",".join(f"{k}={v:.4g}" for k, v in cleared)
    return cleared


def save_over_time_plot(rows, out_path, title=None):
    """
    Over-time figure: one subplot per metric; both directions overlaid
    (different colors). Valid points connected; negatives marked red ×.
    Annotate last digits of filename next to points.
    """
    plt = _get_agg_plt()
    dir_style = (
        ("ch0->ch2", "C0", "12 ch0→ch2"),
        ("ch2->ch0", "C1", "21 ch2→ch0"),
    )
    metrics = (
        ("amp_spike_mV", "AP amp (mV)"),
        ("amp_spikelet_mV", "Spikelet amp (mV)"),
        ("amp_ratio", "Ratio spikelet/spike"),
        ("delay_onset_ms", "Delay onset (ms)"),
        ("delay_peak_ms", "Delay peak (ms)"),
    )

    # scrub negatives in-place (Excel + plot share the same cleaned values)
    neg_marks = []  # (dt, key, value, direction, file_tail)
    metric_keys = {k for k, _ in metrics}
    for r in rows:
        dt = _parse_dt(r.get("recording_datetime"))
        cleared = _scrub_negative_metrics(r)
        if dt is None or not cleared:
            continue
        tail = _file_tail_digits(r.get("file"))
        d = r.get("direction")
        for k, v in cleared:
            if k in metric_keys:
                neg_marks.append((dt, k, v, d, tail))

    by_dir = {"ch0->ch2": [], "ch2->ch0": []}
    for r in rows:
        d = r.get("direction")
        if d not in by_dir:
            continue
        dt = _parse_dt(r.get("recording_datetime"))
        if dt is None:
            continue
        by_dir[d].append((dt, r))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 14), sharex=True)
    if title:
        fig.suptitle(
            f"{title} — meantrace spikelet over time (12 & 21 overlay)",
            fontsize=12,
        )

    n_pts = 0
    labeled_neg = False
    for ax, (key, ylab) in zip(axes, metrics):
        any_pts = False
        for direction, color, label in dir_style:
            items = sorted(by_dir[direction], key=lambda x: x[0])
            if not items:
                continue
            xs, ys, tails = [], [], []
            for dt, r in items:
                fv = _as_float(r.get(key))
                xs.append(dt)
                ys.append(fv if np.isfinite(fv) else np.nan)
                tails.append(_file_tail_digits(r.get("file")))
            # line only through finite (non-negative already scrubbed) points
            ax.plot(xs, ys, "o-", color=color, label=label, lw=1.4, ms=5)
            n_pts += int(np.sum(np.isfinite(ys)))
            any_pts = True
            for x, y, tail in zip(xs, ys, tails):
                if not np.isfinite(y):
                    continue
                ax.annotate(
                    tail, xy=(x, y), textcoords="offset points",
                    xytext=(3, 4), fontsize=6, color=color, alpha=0.9,
                )
                if key == "amp_spikelet_mV":
                    # tiny shape tag under file number
                    pass
        # red × for cleared negatives of this metric
        for dt, k, v, d, tail in neg_marks:
            if k != key:
                continue
            ax.scatter(
                [dt], [v], marker="x", c="red", s=55, zorder=8, linewidths=1.5,
                label="negative (removed)" if not labeled_neg else None,
            )
            ax.annotate(
                tail, xy=(dt, v), textcoords="offset points",
                xytext=(3, -9), fontsize=6, color="red", alpha=0.9,
            )
            labeled_neg = True
            any_pts = True
        if not any_pts:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

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
    "shape",
    "amp_spike_mV",
    "amp_spikelet_peak_mV",
    "amp_spikelet_plateau_mV",
    "amp_spikelet_mV",
    "amp_ratio",
    "delay_ms",
    "delay_onset_ms",
    "delay_peak_ms",
    "delay_mode",
    "t_AP_onset_ms",
    "t_spikelet_onset_ms",
    "t_peak_active_ms",
    "t_peak_passive_ms",
    "t_shelf_onset_ms",
    "spikelet_onset_frac",
    "shelf_plateau_start_frac",
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
            "delay_ms", "delay_onset_ms", "delay_peak_ms", "delay_mode",
            "t_spikelet_onset_ms", "t_shelf_onset_ms", "t_peak_passive_ms",
            "skip_reason", "n_AP_aligned",
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
                f"ratio={r.get('amp_ratio')} "
                f"d_on={r.get('delay_onset_ms')} d_pk={r.get('delay_peak_ms')} "
                f"skip={r.get('skip_reason')}"
            )
        if save_qc:
            try:
                save_file_qc_png(rows, out_dir, stem)
            except Exception as exc:
                print(f"  QC plot error: {exc}")
                import traceback
                traceback.print_exc()
        for r in rows:
            r.pop("_plot", None)
            all_rows.append(r)

    # over-time first: scrubs negatives (red ×) in-place, then Excel uses cleaned rows
    png = os.path.join(out_dir, f"{folder_name}_spikelet_meantrace_over_time.png")
    save_over_time_plot(all_rows, png, title=folder_name)
    xlsx = os.path.join(out_dir, f"{folder_name}_spikelet_meantrace.xlsx")
    save_excel(xlsx, all_rows)
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
