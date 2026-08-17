"""Core logic for coupling coefficient batch analysis (imported by notebook).

Notebook: ``CC calculation (master project).ipynb`` (cells 0-3).
Batch loop and Excel writers live in the notebook, not here.
"""

import os
import statistics
import traceback
from datetime import datetime

import numpy as np
import pyabf
import scipy.signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# =============================================================================
# CONFIG
# =============================================================================

EXPECTED_CHANNELS = 4  # double_cciv: 2x Vm + 2x I

CC_SPIKE_HEIGHT = -10  # mV; AP if find_peaks sees Vm above this
CC_SPIKE_DISTANCE = 10  # samples; min distance between peaks
CC_MIN_DELTA_I_PA = 10  # pA; skip CC if |stim current step| smaller
CC_MIN_DELTA_V_MV = 10  # mV; skip CC if |delta_V_active| (pre-post) is smaller

RIN_VMIN = -80  # mV; primary I–V window (mean Vm in stim epoch)
RIN_VMAX = -50  # mV
RIN_VMIN_FALLBACK = -95  # mV; wider window if too few points in primary
RIN_VMAX_FALLBACK = -50  # mV
RIN_MIN_POINTS = 2  # minimum sweeps for V(I) linear fit
# Unit conversion: R [MΩ] = (ΔV [mV] / ΔI [pA]) × 1000  OR  slope(V vs I) [mV/pA] × 1000

# Cell properties (Cell Properties for one folder.ipynb); separate from CC Rin
CP_SPIKE_HEIGHT = -10  # mV
CP_SPIKE_DISTANCE = 10  # samples
CP_MIN_APS = 4  # sweep must have >= this many APs in stim window
CP_RIN_VMIN = -85  # mV; linear I–V window for Rin_abs / Rin_rel
CP_RIN_VMAX = -50  # mV
CP_RIN_REL_MIN_DI_PA = 0.15  # pA; skip per-sweep Rin_rel if |delta I| smaller

SAVE_QC_PLOTS = True  # save AP + I–V + tau/Cm PNGs (cell-properties style QC)
CELL_PROPS_PLOTS_SUBDIR = "Cell_properties_plots"  # AP / Rin / tau figures
QC_PLOTS_SUBDIR = CELL_PROPS_PLOTS_SUBDIR  # backward-compatible alias
SAVE_CC_PLOTS = True  # save CC QC PNGs when cc_plots_dir is passed
SAVE_CC_TRACE_PLOTS = True  # *_CC_traces.png (sweep QC for CC/Gj)
SAVE_CC_VPOST_PLOTS = True  # *_CC_vs_Vpost.png per file
CC_PLOTS_SUBDIR = "CC_plots"  # subfolder for coupling-coefficient figures
PLOT_DPI = 100  # PNG resolution (lower = faster writes; was 150)
CC_VM_FIT_LINEAR = True  # solid line on folder CC_norm vs Vm
CC_VM_FIT_QUADRATIC = True  # dashed 2nd-order poly (needs ≥3 points)
CC_VM_CMAP = "viridis"  # file color = recording order (first → last)

# Spikelet coupling (AP2+ on first >=4 AP sweep, else 3, else 2)
SPIKELET_BASELINE_MS = 1.0  # passive mean Vm in [t_start-1ms, t_start); not used as t=0
SPIKELET_PEAK_MS = 15.0  # search passive peak in [t_start, t_start+15ms]
SPIKELET_NOISE_K = 1.5  # detect if amp_spikelet > k × robust noise (MAD)
SPIKELET_MIN_AMP_MV = 0.15  # extra floor so tiny bumps still need 0.15 mV
SPIKELET_MIN_APS = 2  # need AP2, so at least 2 APs on the sweep
SAVE_SPIKELET_PLOTS = True
SPIKELET_PLOTS_SUBDIR = "Spikelet_plots"

# Tau / Cm (Tau Cm calculations.ipynb); Rin = same CC/Gj rin_for_channel
TCM_VMIN_LIMIT = -90  # mV; skip sweep if min Vm in post epoch is below this
TCM_DV_SEARCH_MIN = -35  # mV; delta_V = mean(post) - mean(pre) search range
TCM_DV_SEARCH_MAX = -10  # mV
TCM_PRE_MS = 0.125  # s; prestim mean window before stim (125 ms)

CELL_PROPS_FIELD_SUFFIXES = (
    "AP21_ratio",
    "init_freq_Hz",
    "late_freq_Hz",
    "mean_freq_Hz",
    "FWHM_ms",
    "delay_AP1_ms",
    "Rin_abs_MOhm",
    "Rin_rel_MOhm",
    "V_rest_mV",
    "hold_V_mV",
    "inj_current_pA",
    "R2_abs_Rin",
    "props_sweep",
    "props_skip_reason",
)

SPIKELET_SUMMARY_SUFFIXES = (
    "sweep",
    "n_AP_active",
    "sweep_tier",
    "n_AP_used",
    "n_spikelet_detected",
    "mean_amp_active_mV",
    "mean_amp_spikelet_mV",
    "mean_amp_ratio",
    "mean_delay_ms",
    "avg_amp_active_mV",
    "avg_amp_spikelet_mV",
    "avg_amp_ratio",
    "avg_delay_ms",
    "metric_source",
    "rms_noise_mV",
    "skip_reason",
)

SPIKELET_AP_KEYS = (
    "file",
    "recording_datetime",
    "direction",
    "sweep",
    "ap_index",
    "t_start_ms",
    "t_peak_active_ms",
    "t_peak_passive_ms",
    "amp_active_mV",
    "baseline_passive_mV",
    "amp_spikelet_mV",
    "amp_ratio",
    "delay_ms",
    "detected",
    "used_in_average",
    "skip_reason",
    "metric_source",
)


def empty_spikelet_summary_fields():
    out = {}
    for tag in ("12", "21"):
        for key in SPIKELET_SUMMARY_SUFFIXES:
            out[f"spikelet_{key}_{tag}"] = None
    return out


TAU_CM_FIELD_SUFFIXES = (
    "tau_ms",
    "Cm_pF",
    "V_pre_mV",
    "V_post_mV",
    "V_post_min_mV",
    "delta_V_mV",
    "tau_sweep",
    "tau_selection_note",
    "tau_skip_reason",
    "Cm_skip_reason",
)


def empty_cell_props_fields():
    """Default None values for cell-property columns on both channels."""
    out = {}
    for ch in ("ch0", "ch2"):
        for key in CELL_PROPS_FIELD_SUFFIXES:
            out[f"{key}_{ch}"] = None
    return out


def empty_tau_cm_fields():
    out = {}
    for ch in ("ch0", "ch2"):
        for key in TAU_CM_FIELD_SUFFIXES:
            out[f"{key}_{ch}"] = None
    return out


# Per-sweep columns only (file-level metrics -> File_summary sheet)
ALL_DATA_SWEEP_KEYS = (
    "file",
    "recording_datetime",
    "sweep",
    "direction",
    "cur_step_pA",
    "delta_V_active_mV",
    "delta_V_passive_mV",
    "Vm_active_stim_mV",
    "CC",
    "CC_norm",
    "CC_skip_reason",
    "Gj_sweep_nS",
    "Gj_sweep_skip_reason",
)


def sweep_only_row(row_dict):
    return {k: row_dict.get(k) for k in ALL_DATA_SWEEP_KEYS}


def time_period_borders(data_rate):
    sr = int(data_rate)
    if sr == 20000:
        return dict(
            start10=0, end10=2800, start11=6000, end11=12800,
            start20=28000, end20=30800, start21=34000, end21=40800,
            Rtime_ch0=8000, Rtime_ch2=36000,
        )
    if sr == 10000:
        return dict(
            start10=0, end10=1400, start11=3000, end11=6400,
            start20=14000, end20=15400, start21=16600, end21=20000,
            Rtime_ch0=4000, Rtime_ch2=18000,
        )
    raise ValueError(f"Unsupported sampling rate: {sr} Hz")


def recording_datetime_str(abf):
    dt = getattr(abf, "abfDateTime", None)
    if dt is None:
        return getattr(abf, "abfDateTimeString", None)
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


def mean_delta_voltage(sweep_y, pre_start, pre_end, post_start, post_end):
    pre_seg = sweep_y[pre_start:pre_end]
    post_seg = sweep_y[post_start:post_end]
    if len(pre_seg) == 0 or len(post_seg) == 0:
        raise ValueError(
            f"empty epoch window (pre {pre_start}:{pre_end}, post {post_start}:{post_end})"
        )
    pre = statistics.mean(pre_seg)
    post = statistics.mean(post_seg)
    return pre - post


def n_spikes_in_window(sweep_y, win_start, win_end, height=CC_SPIKE_HEIGHT, distance=CC_SPIKE_DISTANCE):
    peaks, _ = find_peaks(sweep_y[win_start:win_end], height=height, distance=distance)
    return len(peaks)


def cc_sweep_indices_direction(
    abf,
    active_ch,
    passive_ch,
    pre_start,
    pre_end,
    post_start,
    post_end,
    height=CC_SPIKE_HEIGHT,
    distance=CC_SPIKE_DISTANCE,
):
    """Subthreshold sweeps: no AP on active or passive in stimulus period (pre_start:post_end)."""
    indices = []
    for sn in abf.sweepList:
        abf.setSweep(sweepNumber=sn, channel=active_ch)
        if n_spikes_in_window(abf.sweepY, pre_start, post_end, height, distance) > 0:
            break
        abf.setSweep(sweepNumber=sn, channel=passive_ch)
        if n_spikes_in_window(abf.sweepY, pre_start, post_end, height, distance) > 0:
            break
        indices.append(sn)
    return indices


# backward-compatible name (single channel, given window)
def cc_sweep_indices(abf, channel, win_start, win_end, height=CC_SPIKE_HEIGHT, distance=CC_SPIKE_DISTANCE):
    indices = []
    for sn in abf.sweepList:
        abf.setSweep(sweepNumber=sn, channel=channel)
        if n_spikes_in_window(abf.sweepY, win_start, win_end, height, distance) > 0:
            break
        indices.append(sn)
    return indices


def cc_skip_reason(delta_i, delta_v_active, passive_sweep_y, pre_start, pre_end, post_start, post_end):
    if n_spikes_in_window(passive_sweep_y, pre_start, post_end) > 0:
        return (
            f"spike on passive cell in stimulus period "
            f"(samples {pre_start}-{post_end})"
        )
    if abs(delta_i) < CC_MIN_DELTA_I_PA:
        return (
            f"|delta_I| < {CC_MIN_DELTA_I_PA} pA in stim current channel "
            f"(pre {pre_start}-{pre_end}, post {post_start}-{post_end})"
        )
    if abs(delta_v_active) < CC_MIN_DELTA_V_MV:
        return (
            f"|delta_V_active| < {CC_MIN_DELTA_V_MV} mV "
            f"(pre {pre_start}-{pre_end}, post {post_start}-{post_end})"
        )
    return None


def coupling_block(abf, sweeps, active_ch, passive_ch, cur_ch, windows):
    s10, e10, s11, e11 = windows
    rows = []
    for sn in sweeps:
        abf.setSweep(sweepNumber=sn, channel=active_ch)
        dva = mean_delta_voltage(abf.sweepY, s10, e10, s11, e11)
        vm_stim = float(statistics.mean(abf.sweepY[s11:e11]))
        abf.setSweep(sweepNumber=sn, channel=cur_ch)
        di = mean_delta_voltage(abf.sweepY, s10, e10, s11, e11)
        abf.setSweep(sweepNumber=sn, channel=passive_ch)
        dvp = mean_delta_voltage(abf.sweepY, s10, e10, s11, e11)
        reason = cc_skip_reason(di, dva, abf.sweepY, s10, e10, s11, e11)
        cc = None if reason else round(dvp / dva, 4)
        rows.append({
            "sweep": sn,
            "cur_step_pA": round(di, 1),
            "delta_V_active_mV": round(dva, 4),
            "delta_V_passive_mV": round(dvp, 4),
            "Vm_active_stim_mV": round(vm_stim, 4),
            "CC": cc,
            "CC_norm": None,
            "CC_skip_reason": reason,
        })
    return rows


def cc_normalize_block(block, cc_mean):
    """Set CC_norm = CC / file mean CC for this direction (valid sweeps only)."""
    if cc_mean in (None, 0):
        return block
    for r in block:
        cc = r.get("CC")
        r["CC_norm"] = round(cc / cc_mean, 4) if cc is not None else None
    return block


def last_subthreshold_sweep(
    abf, voltage_ch, spike_start, spike_end,
    height=CC_SPIKE_HEIGHT, distance=CC_SPIKE_DISTANCE,
):
    """
    Last sweep index before the first AP on voltage_ch in [spike_start:spike_end].

    If no AP in any sweep -> last sweep in the file.
    If sweep 0 already has an AP -> None (no usable subthreshold sweeps).
    """
    last_ok = None
    for sn in abf.sweepList:
        abf.setSweep(sweepNumber=sn, channel=voltage_ch)
        if n_spikes_in_window(abf.sweepY, spike_start, spike_end, height, distance) > 0:
            return last_ok
        last_ok = sn
    return last_ok


def collect_iv_points(
    abf, voltage_ch, current_ch, v_start, v_end, i_index, vmin, vmax, sweep_max=None,
):
    """Absolute (I at Rtime, mean Vm in stim epoch) for linear Rin fit."""
    currents, voltages = [], []
    for sn in abf.sweepList:
        if sweep_max is not None and sn > sweep_max:
            break
        abf.setSweep(sweepNumber=sn, channel=voltage_ch)
        vm = statistics.mean(abf.sweepY[v_start:v_end])
        if vmin <= vm <= vmax:
            voltages.append(vm)
            abf.setSweep(sweepNumber=sn, channel=current_ch)
            currents.append(abf.sweepY[i_index])
    return currents, voltages


def collect_rin_delta_points(
    abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end, vmin, vmax,
    sweep_max=None,
):
    """Per sweep: ΔV/ΔI if mean post-stim Vm in [vmin, vmax] and |ΔI| large enough."""
    delta_is, delta_vs = [], []
    for sn in abf.sweepList:
        if sweep_max is not None and sn > sweep_max:
            break
        abf.setSweep(sweepNumber=sn, channel=voltage_ch)
        v_post = statistics.mean(abf.sweepY[post_start:post_end])
        if not (vmin <= v_post <= vmax):
            continue
        v_pre = statistics.mean(abf.sweepY[pre_start:pre_end])
        abf.setSweep(sweepNumber=sn, channel=current_ch)
        i_pre = statistics.mean(abf.sweepY[pre_start:pre_end])
        i_post = statistics.mean(abf.sweepY[post_start:post_end])
        di = i_post - i_pre
        if abs(di) < CC_MIN_DELTA_I_PA:
            continue
        delta_is.append(di)
        delta_vs.append(v_post - v_pre)
    return delta_is, delta_vs


def rin_r2(currents, voltages):
    """Linear fit V(I); inputs pA and mV -> Rin in MΩ (= slope [mV/pA] × 1000)."""
    if len(currents) < RIN_MIN_POINTS:
        return None, None
    try:
        fit = np.polyfit(currents, voltages, 1, full=True)
        rin = np.round(fit[0][0] * 1000.0, 3)
        sse = fit[1][0]
        sst = ((np.array(voltages) - statistics.mean(voltages)) ** 2).sum()
        r2 = None if sst == 0 else np.round(1 - sse / sst, 3)
        return rin, r2
    except (IndexError, ZeroDivisionError, np.linalg.LinAlgError, TypeError, ValueError):
        return None, None


def rin_from_delta(di, dv):
    """Rin [MΩ] from one step: (ΔV [mV] / ΔI [pA]) × 1000."""
    if abs(di) < CC_MIN_DELTA_I_PA:
        return None
    return round(dv / di * 1000.0, 3)


def collect_cc_iv_series(
    abf, voltage_ch, current_ch, v_start, v_end, i_index, spike_start=None, spike_end=None,
):
    """All-sweep absolute (I, V) for Rin QC plot; flags spike-free and Vm windows."""
    if spike_start is None:
        spike_start = v_start
    if spike_end is None:
        spike_end = v_end
    sweep_max = last_subthreshold_sweep(abf, voltage_ch, spike_start, spike_end)

    currents, voltages = [], []
    flags_primary, flags_fallback, flags_subth = [], [], []
    for sn in abf.sweepList:
        abf.setSweep(sweepNumber=sn, channel=voltage_ch)
        vm = float(statistics.mean(abf.sweepY[v_start:v_end]))
        abf.setSweep(sweepNumber=sn, channel=current_ch)
        i_val = float(abf.sweepY[i_index])
        voltages.append(vm)
        currents.append(i_val)
        is_sub = sweep_max is not None and sn <= sweep_max
        flags_subth.append(is_sub)
        flags_primary.append(is_sub and RIN_VMIN <= vm <= RIN_VMAX)
        flags_fallback.append(is_sub and RIN_VMIN_FALLBACK <= vm <= RIN_VMAX_FALLBACK)
    return {
        "currents": currents,
        "voltages": voltages,
        "in_primary": flags_primary,
        "in_fallback": flags_fallback,
        "subthreshold": flags_subth,
        "sweep_max_subth": sweep_max,
    }


def rin_for_channel(
    abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end, i_index
):
    """
    Rin for CC / Gj / Cm (priority), using only sweeps before the first AP
    on voltage_ch in the post epoch [post_start:post_end]:

    1) Absolute: linear fit V vs I, mean post Vm in -80...-50 mV (>=2 points)
    2) Absolute: same fit, mean post Vm in -95...-50 mV (>=2 points)
    3) Relative: dV/dI (mean if several) for sweeps with mean Vm in -95...-50
       (>=1 point; no linear fit)
    """
    stim_label = f"stim Vm samples {post_start}-{post_end}, I at sample {i_index}"
    plot = {"mode": None, "rin_mohm": None, "r2": None, "vm_range": None}

    sweep_max = last_subthreshold_sweep(abf, voltage_ch, post_start, post_end)
    if sweep_max is None:
        return (
            None, None, 0,
            (
                f"no subthreshold sweeps (AP already on sweep 0 in "
                f"[{post_start}:{post_end}]) ({stim_label})"
            ),
            None, None, plot,
        )
    sub_note = f"sweeps 0...{sweep_max} (before first AP)"

    # --- 1) Absolute, primary Vm window (>=2 points) ---
    currents, voltages = collect_iv_points(
        abf, voltage_ch, current_ch, post_start, post_end, i_index,
        RIN_VMIN, RIN_VMAX, sweep_max=sweep_max,
    )
    n_primary = len(voltages)
    if n_primary >= RIN_MIN_POINTS:
        rin, r2 = rin_r2(currents, voltages)
        if rin is not None:
            plot.update(
                mode="abs_linear", rin_mohm=rin, r2=r2,
                vm_range=f"{RIN_VMIN}:{RIN_VMAX} mV",
            )
            return (
                rin, r2, n_primary, None,
                f"{RIN_VMIN}:{RIN_VMAX} mV",
                f"Rin_abs primary ({sub_note})",
                plot,
            )

    # --- 2) Absolute, extended Vm window (>=2 points) ---
    currents_fb, voltages_fb = collect_iv_points(
        abf, voltage_ch, current_ch, post_start, post_end, i_index,
        RIN_VMIN_FALLBACK, RIN_VMAX_FALLBACK, sweep_max=sweep_max,
    )
    n_fb = len(voltages_fb)
    if n_fb >= RIN_MIN_POINTS:
        rin, r2 = rin_r2(currents_fb, voltages_fb)
        if rin is not None:
            note = (
                f"Rin_abs extended ({RIN_VMIN_FALLBACK}...{RIN_VMAX_FALLBACK} mV, "
                f"{n_fb} points; {sub_note})"
            )
            plot.update(
                mode="abs_linear", rin_mohm=rin, r2=r2,
                vm_range=f"{RIN_VMIN_FALLBACK}:{RIN_VMAX_FALLBACK} mV",
            )
            return (
                rin, r2, n_fb, None,
                f"{RIN_VMIN_FALLBACK}:{RIN_VMAX_FALLBACK} mV", note, plot,
            )

    # --- 3) Relative: only -95...-50, >=1 point, no linear fit ---
    dis, dvs = collect_rin_delta_points(
        abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end,
        RIN_VMIN_FALLBACK, RIN_VMAX_FALLBACK, sweep_max=sweep_max,
    )
    if len(dis) >= 1:
        ratios = [dv / di for di, dv in zip(dis, dvs)]
        rin = round(statistics.mean(ratios) * 1000.0, 3)
        note = (
            f"Rin_rel dV/dI ({len(dis)} sweep(s), "
            f"Vm in [{RIN_VMIN_FALLBACK}, {RIN_VMAX_FALLBACK}] mV; {sub_note})"
        )
        plot.update(mode="rel_delta", rin_mohm=rin, vm_range=note)
        return (
            rin, None, len(dis), None,
            f"{RIN_VMIN_FALLBACK}:{RIN_VMAX_FALLBACK} mV", note, plot,
        )

    return (
        None, None, max(n_primary, n_fb),
        (
            f"no Rin_abs (>= {RIN_MIN_POINTS} points in "
            f"[{RIN_VMIN},{RIN_VMAX}] or [{RIN_VMIN_FALLBACK},{RIN_VMAX_FALLBACK}] mV) "
            f"and no Rin_rel (>=1 sweep in [{RIN_VMIN_FALLBACK},{RIN_VMAX_FALLBACK}] mV) "
            f"among {sub_note} ({stim_label})"
        ),
        None, None, plot,
    )


# =============================================================================
# Cell properties (from Cell Properties for one folder.ipynb)
# =============================================================================


def _stim_channel_for_voltage(voltage_ch):
    """Channel index whose sweepC carries the stimulus command for voltage_ch."""
    if voltage_ch == 0:
        return 0
    if voltage_ch == 2:
        return 1
    raise ValueError(f"unsupported voltage channel {voltage_ch}")


def channel_epochs(abf, voltage_ch):
    """Stim start/stop and analysis windows from sweepC (125 ms pre, 250 ms late stim, 500 ms post)."""
    sr = int(abf.dataRate)
    points_pre = int(0.125 * sr)
    points_in = int(0.25 * sr)
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
        "within_point": stop - points_in,
    }


def cp_sweep_4aps(abf, voltage_ch, start, stop, v_level=CP_SPIKE_HEIGHT):
    for sweep_num in abf.sweepList:
        abf.setSweep(sweepNumber=sweep_num, channel=voltage_ch)
        peaks, _ = find_peaks(
            abf.sweepY[start:stop], height=v_level, distance=CP_SPIKE_DISTANCE
        )
        if len(peaks) >= CP_MIN_APS:
            return sweep_num
    return None


def cp_sweep_1ap(abf, voltage_ch, start, stop, v_level=CP_SPIKE_HEIGHT):
    """Last sweep before the first sweep with >=1 AP (for subthreshold I–V / Rin_abs)."""
    last_no_ap = 0
    for sweep_num in abf.sweepList:
        abf.setSweep(sweepNumber=sweep_num, channel=voltage_ch)
        peaks, _ = find_peaks(
            abf.sweepY[start:stop], height=v_level, distance=CP_SPIKE_DISTANCE
        )
        if len(peaks) >= 1:
            return sweep_num - 1 if sweep_num > 0 else 0
        last_no_ap = sweep_num
    return last_no_ap


def cp_infl_points(abf, sweep, voltage_ch, start, stop):
    if sweep is None:
        return np.array([], dtype=int)
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    smooth = gaussian_filter1d(abf.sweepY[start:stop], 3)
    smooth_d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    return infls + start


def cp_peak_indices(abf, sweep, voltage_ch, start, stop, v_level=CP_SPIKE_HEIGHT):
    if sweep is None:
        return np.array([], dtype=int)
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    peaks, _ = find_peaks(
        abf.sweepY[start:stop], height=v_level, distance=CP_SPIKE_DISTANCE
    )
    return peaks + start


def cp_spike_begin(ind_peaks, ind_infls):
    start_ap_ind, end_ap_ind = [], []
    if len(ind_peaks) == 0 or len(ind_infls) == 0:
        return start_ap_ind, end_ap_ind

    for peak in ind_peaks:
        infl_before = [i for i in ind_infls if i < peak]
        if len(infl_before) >= 2:
            start_ap_ind.append(infl_before[-2])
        else:
            start_ap_ind.append(np.nan)

        infl_after = [i for i in ind_infls if i > peak]
        if len(infl_after) >= 2:
            end_ap_ind.append(infl_after[1])
        else:
            end_ap_ind.append(np.nan)

    return start_ap_ind, end_ap_ind


def cp_fwhm_details(abf, sweep, voltage_ch, ap_start, ap_end, ap_peak, sampling_rate, height=0.5):
    """
    FWHM of 1st AP relative to AP start baseline (not prominence).

    half_level = V(ap_start) + height * (V(peak) - V(ap_start))
    Width = time between rising and falling crossings of half_level.
    """
    if sweep is None or not np.isfinite(ap_start) or not np.isfinite(ap_end):
        return None, None, None
    if not np.isfinite(ap_peak):
        return None, None, None

    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    i0 = int(ap_start)
    i1 = int(ap_end)
    ip = int(ap_peak)
    y = abf.sweepY
    if i0 < 0 or i1 <= i0 + 1 or ip <= i0 or ip >= i1:
        return None, None, None

    v_base = float(y[i0])
    v_peak = float(y[ip])
    amp = v_peak - v_base
    if amp <= 0:
        return None, None, None

    half_level = v_base + height * amp

    def _cross_up(j0, j1, level):
        for i in range(j0, j1):
            y0, y1 = float(y[i]), float(y[i + 1])
            if y0 < level <= y1:
                if y1 == y0:
                    return float(i)
                return i + (level - y0) / (y1 - y0)
        return None

    def _cross_down(j0, j1, level):
        for i in range(j0, j1):
            y0, y1 = float(y[i]), float(y[i + 1])
            if y0 >= level > y1:
                if y0 == y1:
                    return float(i)
                return i + (y0 - level) / (y0 - y1)
        return None

    left = _cross_up(i0, ip, half_level)
    right = _cross_down(ip, i1 - 1, half_level)
    if left is None or right is None or right <= left:
        return None, None, None

    fwhm_ms = round(((right - left) / sampling_rate) * 1000, 3)
    left_i = int(round(left))
    right_i = int(round(right))
    return fwhm_ms, (left_i, right_i), half_level


def cp_fwhm_1st_spike(abf, sweep, voltage_ch, ap_start, ap_end, ap_peak, sampling_rate, height=0.5):
    fwhm_ms, _, _ = cp_fwhm_details(
        abf, sweep, voltage_ch, ap_start, ap_end, ap_peak, sampling_rate, height
    )
    return fwhm_ms


def cp_ap21_ratio(abf, sweep, voltage_ch, ap_start, ap_peaks):
    if sweep is None or len(ap_peaks) < 2:
        return None
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    amplitudes = [
        abf.sweepY[int(peak)] - abf.sweepY[int(start)]
        for peak, start in zip(ap_peaks, ap_start)
        if np.isfinite(start)
    ]
    if len(amplitudes) < 2 or amplitudes[0] == 0:
        return None
    return round(amplitudes[1] / amplitudes[0], 2)


def cp_frequencies(ap_peaks, sampling_rate):
    if len(ap_peaks) < 2:
        return None, None, None
    ap_peaks = np.asarray(ap_peaks, dtype=int)
    freq_12 = round(sampling_rate / (ap_peaks[1] - ap_peaks[0]), 2)
    freq_late = round(sampling_rate / (ap_peaks[-1] - ap_peaks[-2]), 2)
    if len(ap_peaks) >= 3:
        freq_list = [
            sampling_rate / (ap_peaks[i + 1] - ap_peaks[i])
            for i in range(1, len(ap_peaks) - 1)
        ]
        mean_freq = round(statistics.mean(freq_list), 2) if freq_list else None
    else:
        mean_freq = None
    return freq_12, freq_late, mean_freq


def cp_delay_1st_ap(stim_start, ap_peaks, sampling_rate):
    if len(ap_peaks) == 0:
        return None
    return round(((ap_peaks[0] - stim_start) / sampling_rate) * 1000, 2)


def cp_collect_iv_rin_data(abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end):
    """I–V points and fit for cell-properties Rin_abs / Rin_rel (returns dict for metrics + plots)."""
    last_sub_sweep = cp_sweep_1ap(abf, voltage_ch, pre_end, post_end)
    voltages_list, currents_list = [], []
    voltages_delta_list, currents_delta_list = [], []
    voltage_range_ind_list = []

    for sweep_index, sn in enumerate(range(last_sub_sweep + 1)):
        abf.setSweep(sweepNumber=sn, channel=voltage_ch)
        voltage_pre = float(np.mean(abf.sweepY[pre_start:pre_end]))
        voltage_post = float(np.mean(abf.sweepY[post_start:post_end]))
        voltage_delta = voltage_post - voltage_pre
        voltages_delta_list.append(voltage_delta)
        voltages_list.append(voltage_post)
        if CP_RIN_VMIN <= voltage_post <= CP_RIN_VMAX:
            voltage_range_ind_list.append(sweep_index)

        abf.setSweep(sweepNumber=sn, channel=current_ch)
        current_pre = float(np.mean(abf.sweepY[pre_start:pre_end]))
        current_post = float(np.mean(abf.sweepY[post_start:post_end]))
        currents_delta_list.append(current_post - current_pre)
        currents_list.append(current_post)

    rin_relative_list = [
        x / y if abs(y) > CP_RIN_REL_MIN_DI_PA else np.nan
        for x, y in zip(voltages_delta_list, currents_delta_list)
    ]

    if voltage_range_ind_list:
        rin_from_range = [rin_relative_list[i] for i in voltage_range_ind_list]
        rin_rel = round(float(np.nanmean(rin_from_range)) * 1000, 3)
    else:
        rin_rel = None

    rin_abs = None
    v_rest = None
    r2_abs = None
    slope = None
    intercept = None

    if len(voltage_range_ind_list) >= 2:
        currents_ar = np.array(currents_list)[voltage_range_ind_list]
        voltages_ar = np.array(voltages_list)[voltage_range_ind_list]
        try:
            fit = np.polyfit(currents_ar, voltages_ar, 1, full=True)
            slope, intercept = fit[0]
            rin_abs = round(float(slope) * 1000, 3)
            v_rest = round(float(intercept), 3)
            sse = fit[1][0]
            sst = np.sum((voltages_ar - np.mean(voltages_ar)) ** 2)
            r2_abs = round(1 - sse / sst, 3) if sst != 0 else None
        except (IndexError, ZeroDivisionError, np.linalg.LinAlgError, TypeError, ValueError):
            pass

    return {
        "currents": currents_list,
        "voltages": voltages_list,
        "range_indices": voltage_range_ind_list,
        "rin_rel": rin_rel,
        "rin_abs": rin_abs,
        "v_rest": v_rest,
        "r2_abs": r2_abs,
        "slope": slope,
        "intercept": intercept,
        "n_subthreshold_sweeps": last_sub_sweep + 1,
    }


def cp_iv_rin(abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end):
    """Rin_abs (V–I slope), Rin_rel (mean deltaV/deltaI), V_rest (intercept at I=0)."""
    data = cp_collect_iv_rin_data(
        abf, voltage_ch, current_ch, pre_start, pre_end, post_start, post_end
    )
    return data["rin_rel"], data["rin_abs"], data["v_rest"], data["r2_abs"]


def cp_iv_values(abf, sweep, voltage_ch, current_ch, pre_start, pre_end, in_start, in_end):
    """Hold potential (pre-stim Vm) and injection current on the >=4 AP sweep."""
    if sweep is None:
        return None, None
    abf.setSweep(sweepNumber=sweep, channel=voltage_ch)
    hold_v = round(float(np.mean(abf.sweepY[pre_start:pre_end])), 3)
    abf.setSweep(sweepNumber=sweep, channel=current_ch)
    current_pre = float(np.mean(abf.sweepY[pre_start:pre_end]))
    current_in = float(np.mean(abf.sweepY[in_start:in_end]))
    inj = round(current_in - current_pre, 3)
    return hold_v, inj


def compute_cell_properties(abf, voltage_ch, current_ch):
    """Per-cell firing and I–V metrics (Cell Properties notebook logic)."""
    label = f"ch{voltage_ch}"
    epochs = channel_epochs(abf, voltage_ch)
    if epochs is None:
        return {f"props_skip_reason_{label}": "no stimulus detected in sweepC"}

    sr = int(abf.dataRate)
    start = epochs["start"]
    stop = epochs["stop"]
    pre_start = epochs["pre_start"]
    post_stop = epochs["post_stop"]
    within = epochs["within_point"]

    sweep_4 = cp_sweep_4aps(abf, voltage_ch, start, stop)
    if sweep_4 is None:
        return {f"props_skip_reason_{label}": f"no sweep with >={CP_MIN_APS} APs in stim window"}

    ind_infls = cp_infl_points(abf, sweep_4, voltage_ch, start, stop)
    ind_peaks = cp_peak_indices(abf, sweep_4, voltage_ch, start, stop)
    if len(ind_peaks) < CP_MIN_APS:
        return {
            f"props_skip_reason_{label}": (
                f"sweep {sweep_4} has {len(ind_peaks)} peaks (< {CP_MIN_APS})"
            )
        }

    ap_start, ap_end = cp_spike_begin(ind_peaks, ind_infls)
    if not ap_start or not np.isfinite(ap_start[0]) or not np.isfinite(ap_end[0]):
        return {f"props_skip_reason_{label}": "could not define 1st AP borders (inflection points)"}

    rin_rel, rin_abs, v_rest, r2_abs = cp_iv_rin(
        abf, voltage_ch, current_ch, pre_start, start, within, stop
    )
    hold_v, inj = cp_iv_values(
        abf, sweep_4, voltage_ch, current_ch, pre_start, start, within, stop
    )
    freq_12, freq_late, mean_freq = cp_frequencies(ind_peaks, sr)
    ch_tag = f"ch{voltage_ch}"

    return {
        f"AP21_ratio_{ch_tag}": cp_ap21_ratio(abf, sweep_4, voltage_ch, ap_start, ind_peaks),
        f"init_freq_Hz_{ch_tag}": freq_12,
        f"late_freq_Hz_{ch_tag}": freq_late,
        f"mean_freq_Hz_{ch_tag}": mean_freq,
        f"FWHM_ms_{ch_tag}": cp_fwhm_1st_spike(
            abf, sweep_4, voltage_ch, ap_start[0], ap_end[0], ind_peaks[0], sr
        ),
        f"delay_AP1_ms_{ch_tag}": cp_delay_1st_ap(start, ind_peaks, sr),
        f"Rin_abs_MOhm_{ch_tag}": rin_abs,
        f"Rin_rel_MOhm_{ch_tag}": rin_rel,
        f"V_rest_mV_{ch_tag}": v_rest,
        f"hold_V_mV_{ch_tag}": hold_v,
        f"inj_current_pA_{ch_tag}": inj,
        f"R2_abs_Rin_{ch_tag}": r2_abs,
        f"props_sweep_{ch_tag}": sweep_4,
        f"props_skip_reason_{ch_tag}": None,
    }


def cell_properties_for_file(abf):
    """Cell properties for both paired cells; keys suffixed _ch0 and _ch2."""
    out = empty_cell_props_fields()
    for voltage_ch, current_ch in ((0, 1), (2, 3)):
        props = compute_cell_properties(abf, voltage_ch, current_ch)
        out.update(props)
    return out


# =============================================================================
# Spikelet coupling (AP2+; first >=4 AP sweep, else 3, else 2)
# =============================================================================


def _ms_to_samples(ms, sr):
    return max(1, int(round(float(ms) * float(sr) / 1000.0)))


def _samples_to_ms(n_samples, sr):
    return (float(n_samples) / float(sr)) * 1000.0


def _spikelet_local_peak_index(seg):
    """Index of max after AP start.

    Relaxed vs v1 (which required a strict interior local max):
    - first sample -> not a spikelet (no rise after start)
    - last sample OK if still rising (peak may sit at window edge)
    - otherwise argmax in the window
    """
    if seg is None or len(seg) < 2:
        return None
    i = int(np.argmax(seg))
    if i <= 0:
        return None
    if i >= len(seg) - 1:
        return i if float(seg[-1]) > float(seg[-2]) else None
    return i


def _passive_rms_prestim(abf, sweep, passive_ch, pre_start, stim_start):
    """Noise on passive before stim: MAD (robust) instead of std."""
    abf.setSweep(sweepNumber=sweep, channel=passive_ch)
    i0 = max(int(pre_start), 0)
    i1 = max(int(stim_start), i0 + 3)
    pre = np.asarray(abf.sweepY[i0:i1], dtype=float)
    if len(pre) < 3:
        return None
    med = float(np.median(pre))
    mad = float(np.median(np.abs(pre - med)))
    if mad > 0:
        return 1.4826 * mad
    return float(np.std(pre - np.mean(pre)))


def spikelet_amp_passes(amp, rms):
    """True if spikelet amplitude clears the relaxed noise gate."""
    if amp is None or amp <= 0:
        return False, "amp_spikelet_le_0"
    noise_bar = (SPIKELET_NOISE_K * rms) if rms is not None else SPIKELET_MIN_AMP_MV
    need = max(noise_bar, SPIKELET_MIN_AMP_MV)
    if amp < need:
        return False, (
            f"below_noise (amp={amp:.4f} < max({SPIKELET_NOISE_K}*noise, "
            f"{SPIKELET_MIN_AMP_MV})={need:.4f})"
        )
    return True, None


def select_spikelet_sweep(abf, voltage_ch, start, stop):
    """
    Same idea as cell properties: first sweep with >=4 APs.
    If none: first with >=3, then >=2. Returns (sweep, n_peaks, tier) or (None, 0, None).
    """
    for min_aps, tier in ((4, ">=4"), (3, "3"), (SPIKELET_MIN_APS, "2")):
        for sweep_num in abf.sweepList:
            abf.setSweep(sweepNumber=sweep_num, channel=voltage_ch)
            peaks, _ = find_peaks(
                abf.sweepY[start:stop], height=CP_SPIKE_HEIGHT, distance=CP_SPIKE_DISTANCE
            )
            if len(peaks) >= min_aps:
                return sweep_num, int(len(peaks)), tier
    return None, 0, None


def _empty_spikelet_metrics(skip_reason):
    return {
        "sweep": None,
        "n_AP_active": 0,
        "sweep_tier": None,
        "n_AP_used": 0,
        "n_spikelet_detected": 0,
        "mean_amp_active_mV": None,
        "mean_amp_spikelet_mV": None,
        "mean_amp_ratio": None,
        "mean_delay_ms": None,
        "avg_amp_active_mV": None,
        "avg_amp_spikelet_mV": None,
        "avg_amp_ratio": None,
        "avg_delay_ms": None,
        "metric_source": None,
        "rms_noise_mV": None,
        "skip_reason": skip_reason,
    }


def _round_or_none(val, nd=4):
    if val is None:
        return None
    try:
        if not np.isfinite(val):
            return None
    except TypeError:
        return None
    return round(float(val), nd)


def analyze_spikelets_direction(abf, active_ch, passive_ch, direction):
    """
    Spikelet metrics on one direction (active -> passive).

    AP1 excluded. Baseline = mean passive in 1 ms before AP start.
    Peak search = [t_start, t_start+SPIKELET_PEAK_MS]. Delay = (peak_p - peak_a) / SR * 1000.
    Detected if a peak after AP start, amp > 0, and amp clears MAD noise gate.

    t=0 is the 2nd-last d2V inflection before the active peak (same as FWHM).
    The 1 ms before t=0 is only the passive baseline, not the AP start.
    """
    epochs = channel_epochs(abf, active_ch)
    if epochs is None:
        return [], _empty_spikelet_metrics("no stimulus detected in sweepC"), None

    sr = int(abf.dataRate)
    stim_start, stim_stop = epochs["start"], epochs["stop"]
    pre_start = epochs["pre_start"]
    n_pre = _ms_to_samples(SPIKELET_BASELINE_MS, sr)
    n_post = _ms_to_samples(SPIKELET_PEAK_MS, sr)

    sweep, n_ap, tier = select_spikelet_sweep(abf, active_ch, stim_start, stim_stop)
    if sweep is None:
        return [], _empty_spikelet_metrics(
            f"no sweep with >={SPIKELET_MIN_APS} APs in stim window"
        ), None

    ind_infls = cp_infl_points(abf, sweep, active_ch, stim_start, stim_stop)
    ind_peaks = cp_peak_indices(abf, sweep, active_ch, stim_start, stim_stop)
    ap_starts, _ap_ends = cp_spike_begin(ind_peaks, ind_infls)
    n_ap = int(len(ind_peaks))
    if n_ap < SPIKELET_MIN_APS:
        return [], _empty_spikelet_metrics(
            f"sweep {sweep} has {n_ap} peaks (< {SPIKELET_MIN_APS})"
        ), None

    rms = _passive_rms_prestim(abf, sweep, passive_ch, pre_start, stim_start)

    abf.setSweep(sweepNumber=sweep, channel=active_ch)
    y_a = np.asarray(abf.sweepY, dtype=float)
    abf.setSweep(sweepNumber=sweep, channel=passive_ch)
    y_p = np.asarray(abf.sweepY, dtype=float)
    n_y = len(y_a)

    ap_rows = []
    snips_a, snips_p = [], []
    amps_a_for_avg = []

    for i in range(1, n_ap):  # skip AP1 (index 0)
        i_peak_a = int(ind_peaks[i])
        i_start = ap_starts[i] if i < len(ap_starts) else np.nan
        row = {
            "ap_index": i + 1,
            "sweep": sweep,
            "t_start_ms": None,
            "t_peak_active_ms": _round_or_none(_samples_to_ms(i_peak_a, sr), 4),
            "t_peak_passive_ms": None,
            "amp_active_mV": None,
            "baseline_passive_mV": None,
            "amp_spikelet_mV": None,
            "amp_ratio": None,
            "delay_ms": None,
            "detected": False,
            "used_in_average": False,
            "skip_reason": None,
            "metric_source": "individual",
        }

        if not np.isfinite(i_start):
            row["skip_reason"] = "no_ap_start"
            ap_rows.append(row)
            continue
        i_start = int(i_start)
        row["t_start_ms"] = _round_or_none(_samples_to_ms(i_start, sr), 4)

        if i_start < n_pre or i_start + n_post >= n_y:
            row["skip_reason"] = "window_out_of_trace"
            ap_rows.append(row)
            continue

        isi_ok = True
        if i + 1 < n_ap:
            next_start = ap_starts[i + 1] if i + 1 < len(ap_starts) else np.nan
            if np.isfinite(next_start) and int(next_start) < i_start + n_post:
                isi_ok = False
        prev_start = ap_starts[i - 1] if i - 1 < len(ap_starts) else np.nan
        if np.isfinite(prev_start) and (i_start - int(prev_start)) < (n_pre + n_post):
            isi_ok = False

        v_base_a = float(y_a[i_start])
        v_peak_a = float(y_a[i_peak_a])
        amp_a = v_peak_a - v_base_a
        row["amp_active_mV"] = _round_or_none(amp_a, 4)
        if amp_a <= 0:
            row["skip_reason"] = "amp_active_le_0"
            ap_rows.append(row)
            continue

        baseline = float(np.mean(y_p[i_start - n_pre:i_start]))
        row["baseline_passive_mV"] = _round_or_none(baseline, 4)
        seg_p = y_p[i_start:i_start + n_post + 1]
        i_rel = _spikelet_local_peak_index(seg_p)
        if i_rel is None:
            row["skip_reason"] = "no_local_peak"
        else:
            i_peak_p = i_start + i_rel
            amp_p = float(y_p[i_peak_p]) - baseline
            row["t_peak_passive_ms"] = _round_or_none(_samples_to_ms(i_peak_p, sr), 4)
            row["amp_spikelet_mV"] = _round_or_none(amp_p, 4)
            row["delay_ms"] = _round_or_none(_samples_to_ms(i_peak_p - i_peak_a, sr), 4)
            if amp_a != 0:
                row["amp_ratio"] = _round_or_none(amp_p / amp_a, 4)
            ok, why = spikelet_amp_passes(amp_p, rms)
            if not ok:
                row["skip_reason"] = why
            else:
                row["detected"] = True
                row["skip_reason"] = None

        if isi_ok:
            snips_a.append(y_a[i_start - n_pre:i_start + n_post])
            snips_p.append(y_p[i_start - n_pre:i_start + n_post])
            amps_a_for_avg.append(amp_a)
            row["used_in_average"] = True
        else:
            if row["skip_reason"] is None and not row["detected"]:
                row["skip_reason"] = "isi_too_short"
            elif not row["detected"] and row["skip_reason"]:
                pass
            elif row["detected"] and not isi_ok:
                row["used_in_average"] = False

        ap_rows.append(row)

    detected = [r for r in ap_rows if r["detected"]]
    metrics = _empty_spikelet_metrics(None)
    metrics.update({
        "sweep": sweep,
        "n_AP_active": n_ap,
        "sweep_tier": tier,
        "n_AP_used": len(ap_rows),
        "n_spikelet_detected": len(detected),
        "rms_noise_mV": _round_or_none(rms, 4),
    })
    if detected:
        metrics["mean_amp_active_mV"] = _round_or_none(
            statistics.mean([r["amp_active_mV"] for r in detected]), 4
        )
        metrics["mean_amp_spikelet_mV"] = _round_or_none(
            statistics.mean([r["amp_spikelet_mV"] for r in detected]), 4
        )
        metrics["mean_amp_ratio"] = _round_or_none(
            statistics.mean([r["amp_ratio"] for r in detected if r["amp_ratio"] is not None]), 4
        )
        metrics["mean_delay_ms"] = _round_or_none(
            statistics.mean([r["delay_ms"] for r in detected if r["delay_ms"] is not None]), 4
        )
        metrics["metric_source"] = "individual"

    mean_a = mean_p = None
    avg_detected = False
    if snips_p:
        arr_p = np.vstack(snips_p)
        arr_a = np.vstack(snips_a)
        mean_p = np.mean(arr_p, axis=0)
        mean_a = np.mean(arr_a, axis=0)
        base_avg = float(np.mean(mean_p[:n_pre]))
        seg_avg = mean_p[n_pre:]
        i_rel_p = _spikelet_local_peak_index(seg_avg)
        i_rel_a = _spikelet_local_peak_index(mean_a[n_pre:])
        amp_a_avg = (
            _round_or_none(statistics.mean(amps_a_for_avg), 4) if amps_a_for_avg else None
        )
        metrics["avg_amp_active_mV"] = amp_a_avg
        if i_rel_p is not None:
            amp_p_avg = float(seg_avg[i_rel_p]) - base_avg
            metrics["avg_amp_spikelet_mV"] = _round_or_none(amp_p_avg, 4)
            if amp_a_avg not in (None, 0):
                metrics["avg_amp_ratio"] = _round_or_none(amp_p_avg / amp_a_avg, 4)
            if i_rel_a is not None:
                metrics["avg_delay_ms"] = _round_or_none(
                    _samples_to_ms(i_rel_p - i_rel_a, sr), 4
                )
            ok, _ = spikelet_amp_passes(amp_p_avg, rms)
            avg_detected = ok
        if not detected:
            if avg_detected:
                metrics["metric_source"] = "average"
                metrics["skip_reason"] = None
                for r in ap_rows:
                    if r["used_in_average"]:
                        r["metric_source"] = "average"
            else:
                metrics["metric_source"] = None
                metrics["skip_reason"] = "no spikelet on AP2+ (individual or average)"
    else:
        if not detected:
            metrics["skip_reason"] = metrics["skip_reason"] or "no AP2+ windows for average"
        mean_a = mean_p = None

    plot_meta = {
        "direction": direction,
        "sweep": sweep,
        "active_ch": active_ch,
        "passive_ch": passive_ch,
        "stim_start": stim_start,
        "stim_stop": stim_stop,
        "pre_start": pre_start,
        "sr": sr,
        "n_pre": n_pre,
        "n_post": n_post,
        "ind_peaks": ind_peaks,
        "ap_starts": ap_starts,
        "ap_rows": ap_rows,
        "mean_a": mean_a,
        "mean_p": mean_p,
        "snips_a": snips_a,
        "snips_p": snips_p,
        "tier": tier,
        "n_ap": n_ap,
        "rms": rms,
        "metrics": metrics,
    }
    return ap_rows, metrics, plot_meta


def _spikelet_row(row_dict):
    return {k: row_dict.get(k) for k in SPIKELET_AP_KEYS}


def save_spikelet_qc_plot(abf, plot_meta, plots_dir, stem):
    """Sweep overlay (active+passive, one axis) + aligned AP2+ mean (twin scales)."""
    if not plot_meta:
        return None
    import os

    os.makedirs(plots_dir, exist_ok=True)
    plt = _get_agg_plt()
    sweep = plot_meta["sweep"]
    active_ch = plot_meta["active_ch"]
    passive_ch = plot_meta["passive_ch"]
    sr = plot_meta["sr"]
    n_pre = plot_meta["n_pre"]
    n_post = plot_meta["n_post"]
    direction = plot_meta["direction"]

    abf.setSweep(sweepNumber=sweep, channel=active_ch)
    t = abf.sweepX
    y_a = np.asarray(abf.sweepY, dtype=float)
    abf.setSweep(sweepNumber=sweep, channel=passive_ch)
    y_p = np.asarray(abf.sweepY, dtype=float)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    ax_ov, ax_avg = axes

    ss = int(plot_meta["stim_start"])
    se = int(plot_meta["stim_stop"])
    i_left = max(int(plot_meta["pre_start"]), 0)
    i_right = min(se + int(0.05 * sr), len(t) - 1)

    ax_ov.plot(t, y_a, color="C0", lw=1.0, label=f"active ch{active_ch}")
    ax_ov.plot(t, y_p, color="C1", lw=1.0, label=f"passive ch{passive_ch}")
    ax_ov.axvline(t[min(ss, len(t) - 1)], color="0.4", ls="--", lw=0.8)
    ax_ov.axvline(t[min(se, len(t) - 1)], color="0.4", ls="--", lw=0.8)

    starts = plot_meta["ap_starts"]
    peaks = plot_meta["ind_peaks"]
    for i, ip in enumerate(peaks):
        ip = int(ip)
        if 0 <= ip < len(t):
            ax_ov.scatter(
                t[ip], y_a[ip], c="C3", s=28, zorder=5, marker="o",
                label="AP peak" if i == 0 else None,
            )
        if i < len(starts) and np.isfinite(starts[i]):
            i0 = int(starts[i])
            if 0 <= i0 < len(t):
                ax_ov.scatter(
                    t[i0], y_a[i0], c="limegreen", s=28, zorder=5, marker="v",
                    label="AP start (t=0, 2nd inflection before peak)" if i == 0 else None,
                )
            if i >= 1 and 0 <= i0 < len(t):
                t0 = t[i0]
                ax_ov.axvspan(t0 - SPIKELET_BASELINE_MS / 1000.0, t0, color="0.7", alpha=0.25)
                ax_ov.axvspan(t0, t0 + SPIKELET_PEAK_MS / 1000.0, color="C4", alpha=0.12)

    labeled_sp = False
    for r in plot_meta["ap_rows"]:
        if r.get("t_peak_passive_ms") is None:
            continue
        tp = r["t_peak_passive_ms"] / 1000.0
        idx = min(max(int(round(tp * sr)), 0), len(y_p) - 1)
        ax_ov.scatter(
            t[idx], y_p[idx], c="darkorange", s=40, zorder=6, marker="x",
            label="spikelet peak" if not labeled_sp else None,
        )
        labeled_sp = True

    src = (plot_meta.get("metrics") or {}).get("metric_source")
    ax_ov.set_ylabel("Vm (mV)")
    ax_ov.set_title(
        f"{stem} — {direction}  sweep {sweep} ({plot_meta.get('tier')}, "
        f"n_AP={plot_meta.get('n_ap')}, source={src})"
    )
    ax_ov.legend(loc="upper right", fontsize=7)
    ax_ov.set_xlim(t[i_left], t[i_right])
    ax_ov.grid(True, alpha=0.25)

    t_snip = (np.arange(-n_pre, n_post) / float(sr)) * 1000.0
    snips_p = plot_meta.get("snips_p") or []
    snips_a = plot_meta.get("snips_a") or []
    n_avg = sum(1 for sn in snips_p if len(sn) == len(t_snip))
    for sn in snips_p:
        if len(sn) == len(t_snip):
            ax_avg.plot(t_snip, sn, color="C1", lw=0.7, alpha=0.35)
    if plot_meta.get("mean_p") is not None and len(plot_meta["mean_p"]) == len(t_snip):
        ax_avg.plot(
            t_snip, plot_meta["mean_p"], color="C1", lw=2.2,
            label=f"mean spikelet (n={n_avg})",
        )
    ax_avg.axvline(0, color="limegreen", ls="--", lw=1.2, label="t=0 AP start (inflection)")
    ax_avg.axvspan(
        -SPIKELET_BASELINE_MS, 0, color="0.7", alpha=0.25,
        label="passive baseline 1 ms (not t=0)",
    )
    ax_avg.axvspan(0, SPIKELET_PEAK_MS, color="C4", alpha=0.12)
    ax_avg.set_xlabel("Time from active AP start (ms)")
    ax_avg.set_ylabel("Passive Vm (mV)", color="C1")
    ax_avg.tick_params(axis="y", labelcolor="C1")

    ax_avg2 = ax_avg.twinx()
    for sn in snips_a:
        if len(sn) == len(t_snip):
            ax_avg2.plot(t_snip, sn, color="C0", lw=0.6, alpha=0.25)
    if plot_meta.get("mean_a") is not None and len(plot_meta["mean_a"]) == len(t_snip):
        ax_avg2.plot(
            t_snip, plot_meta["mean_a"], color="C0", lw=1.6, alpha=0.9,
            label=f"mean AP (n={n_avg})",
        )
    ax_avg2.set_ylabel("Active Vm (mV)", color="C0")
    ax_avg2.tick_params(axis="y", labelcolor="C0")

    handles, labels = ax_avg.get_legend_handles_labels()
    h2, l2 = ax_avg2.get_legend_handles_labels()
    ax_avg.legend(handles + h2, labels + l2, loc="upper left", fontsize=7)
    ax_avg.set_title(
        f"Aligned AP2+ (not AP1): thin=each spike, thick=mean of {n_avg} "
        "(passive left, AP right)"
    )
    ax_avg.grid(True, alpha=0.3)

    fig.tight_layout()
    tag = direction.replace(">", "")
    path = os.path.join(plots_dir, f"{stem}_{tag}_spikelets.png")
    _savefig_white(fig, path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def spikelets_for_file(abf, name, rec_dt, plots_dir=None, stem=None):
    """Both directions. Returns (ap_rows, summary_fields, plot_paths)."""
    summary = empty_spikelet_summary_fields()
    all_rows = []
    plot_paths = []
    stem = stem or _abf_stem(name)

    for direction, active, passive, tag in (
        ("ch0->ch2", 0, 2, "12"),
        ("ch2->ch0", 2, 0, "21"),
    ):
        try:
            ap_rows, metrics, meta = analyze_spikelets_direction(
                abf, active, passive, direction
            )
        except Exception as exc:
            metrics = _empty_spikelet_metrics(str(exc))
            ap_rows, meta = [], None
        for r in ap_rows:
            all_rows.append(_spikelet_row({
                "file": name,
                "recording_datetime": rec_dt,
                "direction": direction,
                **r,
            }))
        for key, val in metrics.items():
            summary[f"spikelet_{key}_{tag}"] = val
        if plots_dir and SAVE_SPIKELET_PLOTS and meta:
            try:
                p = save_spikelet_qc_plot(abf, meta, plots_dir, stem)
                if p:
                    plot_paths.append(p)
            except Exception as exc:
                print(f"  Spikelet plot skip ({direction}): {exc}")
                traceback.print_exc()

    return all_rows, summary, plot_paths


# =============================================================================
# Tau / Cm (Tau Cm calculations.ipynb)
# =============================================================================


def cc_epochs_for_tau(borders, data_rate, voltage_ch, sweep_len=None):
    """Epochs for tau/Cm on double_cciv: Vm channels 0 and 2, CC fixed borders.

    Tau search starts at pre_end (end of pre-stim epoch), not at post_start.
    """
    sr = int(data_rate)
    if voltage_ch == 0:
        pre_start, pre_end = borders["start10"], borders["end10"]
        post_start, post_end = borders["start11"], borders["end11"]
    elif voltage_ch == 2:
        pre_start, pre_end = borders["start20"], borders["end20"]
        post_start, post_end = borders["start21"], borders["end21"]
    else:
        raise ValueError(f"unsupported voltage channel {voltage_ch}")
    # tau clock starts at end of pre epoch (onset of step), not at post mean window
    stim_start = pre_end
    post_stop = post_end + int(0.5 * sr)
    if sweep_len is not None:
        post_stop = min(post_stop, sweep_len)
    return {
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
        "stim_start": stim_start,
        "post_stop": post_stop,
        "voltage_ch": voltage_ch,
    }


def compute_tau_cm(abf, voltage_ch, rin_MOhm, borders=None):
    """
    Prefer sweep with ΔV=mean(post)-mean(pre) in [-35, -10] mV and post_min >= -90
    (closest |ΔV| to zero). If none: sweep with ΔV closest to -35 mV (post_min may be
    below -90; values still exported).

    v_63 = V_pre + 0.63 * (V_post - V_pre);
    tau [ms] from pre_end; Cm = tau_ms / Rin * 1000 (CC/Gj Rin).
    """
    ch = f"ch{voltage_ch}"
    empty = {f"{k}_{ch}": None for k in TAU_CM_FIELD_SUFFIXES}

    if borders is not None:
        abf.setSweep(sweepNumber=abf.sweepList[0], channel=voltage_ch)
        epochs = cc_epochs_for_tau(borders, abf.dataRate, voltage_ch, sweep_len=len(abf.sweepY))
    else:
        ce = channel_epochs(abf, voltage_ch)
        if ce is None:
            return {**empty, f"tau_skip_reason_{ch}": "no stimulus detected in sweepC"}
        sr = int(abf.dataRate)
        epochs = {
            "pre_start": ce["pre_start"],
            "pre_end": ce["pre_start"] + int(TCM_PRE_MS * sr),
            "post_start": ce["start"],
            "post_end": ce["stop"],
            "stim_start": ce["pre_start"] + int(TCM_PRE_MS * sr),  # pre_end
            "post_stop": ce["post_stop"],
        }

    sr = int(abf.dataRate)
    pre_start = epochs["pre_start"]
    pre_end = epochs["pre_end"]
    post_start = epochs["post_start"]
    post_end = epochs["post_end"]
    stim_start = epochs["stim_start"]  # tau clock origin = pre_end
    post_stop = epochs["post_stop"]
    epoch_label = (
        f"ch{voltage_ch} V_pre[{pre_start}:{pre_end}] "
        f"V_post[{post_start}:{post_end}] tau_from@{stim_start}(=pre_end) "
        f"tau→[{stim_start}:{post_stop}]"
    )

    # cand: (sweep, v_pre, v_post, delta_v, post_min)
    in_range = []
    all_sweeps = []

    for sweep_num in abf.sweepList:
        abf.setSweep(sweepNumber=sweep_num, channel=voltage_ch)
        y = abf.sweepY
        if pre_end > len(y) or post_end > len(y) or post_stop > len(y):
            continue
        if post_start >= post_end:
            continue

        sweep_v_pre = float(np.mean(y[pre_start:pre_end]))
        sweep_v_post = float(np.mean(y[post_start:post_end]))
        post_min = float(np.min(y[post_start:post_end]))
        sweep_dv = sweep_v_post - sweep_v_pre
        cand = (sweep_num, sweep_v_pre, sweep_v_post, sweep_dv, post_min)
        all_sweeps.append(cand)

        if (
            TCM_DV_SEARCH_MIN <= sweep_dv <= TCM_DV_SEARCH_MAX
            and post_min >= TCM_VMIN_LIMIT
        ):
            in_range.append(cand)

    selection_note = None
    if in_range:
        best_sweep, v_pre, v_post, delta_v, post_min = min(in_range, key=lambda c: abs(c[3]))
        selection_note = (
            f"in-range [{TCM_DV_SEARCH_MIN},{TCM_DV_SEARCH_MAX}] mV, "
            f"post_min>={TCM_VMIN_LIMIT}; "
            f"delta_V={delta_v:.3f} mV; V_post={v_post:.3f} mV; "
            f"V_post_min={post_min:.3f} mV"
        )
    elif all_sweeps:
        # fallback: closest to -35; post_min may be < -90 — still export
        target = float(TCM_DV_SEARCH_MIN)
        best_sweep, v_pre, v_post, delta_v, post_min = min(
            all_sweeps, key=lambda c: abs(c[3] - target)
        )
        selection_note = (
            f"fallback: closest to {target:g} mV "
            f"(no in-range with post_min>={TCM_VMIN_LIMIT}); "
            f"delta_V={delta_v:.3f} mV; V_post={v_post:.3f} mV; "
            f"V_post_min={post_min:.3f} mV"
        )
    else:
        reason = f"no sweeps with valid pre/post epochs ({post_start}:{post_end})"
        return {
            **empty,
            f"tau_skip_reason_{ch}": reason,
            f"_tau_plot_{ch}": {
                "sweep": None,
                "voltage_ch": voltage_ch,
                "pre_start": pre_start,
                "pre_end": pre_end,
                "post_start": post_start,
                "post_end": post_end,
                "stim_start": stim_start,
                "post_stop": post_stop,
                "epoch_label": epoch_label,
                "skip_reason": reason,
                "rin_MOhm": rin_MOhm,
            },
        }

    abf.setSweep(sweepNumber=best_sweep, channel=voltage_ch)
    v_63 = v_pre + 0.63 * (v_post - v_pre)
    tau_samples = None
    trace = abf.sweepY[stim_start:post_stop]
    for i, v in enumerate(trace):
        if v <= v_63:
            tau_samples = i
            break

    plot_base = {
        "sweep": best_sweep,
        "voltage_ch": voltage_ch,
        "pre_start": pre_start,
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_end,
        "stim_start": stim_start,
        "post_stop": post_stop,
        "epoch_label": epoch_label,
        "v_pre": v_pre,
        "v_post": v_post,
        "v_63": v_63,
        "delta_v": delta_v,
        "post_min": post_min,
        "selection_note": selection_note,
        "rin_MOhm": rin_MOhm,
    }

    common_fields = {
        f"V_pre_mV_{ch}": round(v_pre, 3),
        f"V_post_mV_{ch}": round(v_post, 3),
        f"V_post_min_mV_{ch}": round(post_min, 3),
        f"delta_V_mV_{ch}": round(delta_v, 3),
        f"tau_sweep_{ch}": best_sweep,
        f"tau_selection_note_{ch}": selection_note,
    }

    if tau_samples is None:
        reason = "trace never reached 63% of delta_V for tau"
        return {
            **empty,
            **common_fields,
            f"tau_skip_reason_{ch}": reason,
            f"_tau_plot_{ch}": {**plot_base, "tau_ms": None, "cm_pF": None, "skip_reason": reason},
        }

    tau_ms = round((tau_samples / sr) * 1000, 3)
    tau_index = stim_start + tau_samples
    cm_pF = None
    cm_skip = None
    if rin_MOhm is None:
        cm_skip = "missing Rin for Cm"
    elif rin_MOhm == 0:
        cm_skip = "Rin = 0"
    else:
        cm_pF = round((tau_ms / rin_MOhm) * 1000, 3)

    return {
        **common_fields,
        f"tau_ms_{ch}": tau_ms,
        f"Cm_pF_{ch}": cm_pF,
        f"tau_skip_reason_{ch}": None,
        f"Cm_skip_reason_{ch}": cm_skip,
        f"_tau_plot_{ch}": {
            **plot_base,
            "tau_index": tau_index,
            "tau_ms": tau_ms,
            "cm_pF": cm_pF,
            "cm_skip": cm_skip,
        },
    }


def tau_cm_for_file(abf, rin_ch0, rin_ch2, borders=None):
    """Tau and Cm for ch0 and ch2 using CC/Gj Rin."""
    out = empty_tau_cm_fields()
    plot_meta = {}
    for voltage_ch, rin in ((0, rin_ch0), (2, rin_ch2)):
        ch = f"ch{voltage_ch}"
        result = compute_tau_cm(abf, voltage_ch, rin, borders=borders)
        plot_key = f"_tau_plot_{ch}"
        if plot_key in result:
            plot_meta[ch] = result.pop(plot_key)
        out.update({k: v for k, v in result.items() if not k.startswith("_")})
    out["_tau_plot_meta"] = plot_meta
    return out


# =============================================================================
# QC plots (saved to disk)
# =============================================================================


def _get_agg_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "savefig.transparent": False,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })
    return plt


def _savefig_white(fig, path, dpi=None):
    """Save PNG with opaque white figure/axes background (readable axes)."""
    if dpi is None:
        dpi = PLOT_DPI
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")
        ax.tick_params(colors="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight",
        facecolor="white", edgecolor="none", transparent=False,
    )


def cell_props_plots_dir(folder_path):
    """Return ``folder_path/Cell_properties_plots``, creating it if needed."""
    path = os.path.join(folder_path, CELL_PROPS_PLOTS_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def qc_plots_dir(folder_path):
    """Alias for ``cell_props_plots_dir`` (backward compatible)."""
    return cell_props_plots_dir(folder_path)


def cc_plots_dir(folder_path):
    """Return ``folder_path/CC_plots``, creating it if needed."""
    path = os.path.join(folder_path, CC_PLOTS_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def spikelet_plots_dir(folder_path):
    """Return ``folder_path/Spikelet_plots``, creating it if needed."""
    path = os.path.join(folder_path, SPIKELET_PLOTS_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _abf_stem(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def _save_ap_qc_plot(abf, voltage_ch, plots_dir, stem):
    """Vm trace on >=4 AP sweep: peaks, AP starts, stim lines, 1st-spike FWHM."""
    plt = _get_agg_plt()

    epochs = channel_epochs(abf, voltage_ch)
    if epochs is None:
        return None

    sr = int(abf.dataRate)
    start, stop = epochs["start"], epochs["stop"]
    pre_start, post_stop = epochs["pre_start"], epochs["post_stop"]

    sweep_4 = cp_sweep_4aps(abf, voltage_ch, start, stop)
    if sweep_4 is None:
        return None

    ind_infls = cp_infl_points(abf, sweep_4, voltage_ch, start, stop)
    ind_peaks = cp_peak_indices(abf, sweep_4, voltage_ch, start, stop)
    if len(ind_peaks) == 0:
        return None

    ap_start, ap_end = cp_spike_begin(ind_peaks, ind_infls)
    abf.setSweep(sweepNumber=sweep_4, channel=voltage_ch)
    t = abf.sweepX
    y = abf.sweepY

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t[pre_start:post_stop], y[pre_start:post_stop], color="C0", lw=1, label="Vm")
    ax.scatter(t[ind_peaks], y[ind_peaks], color="red", s=40, zorder=5, label="peaks")
    starts = [int(s) for s in ap_start if np.isfinite(s)]
    if starts:
        ax.scatter(
            t[starts], y[starts], color="limegreen", s=40, zorder=5, label="AP start"
        )
    ax.axvline(start / sr, color="gray", ls=":", lw=1, alpha=0.8)
    ax.axvline(stop / sr, color="gray", ls=":", lw=1, alpha=0.8)

    if ap_start and np.isfinite(ap_start[0]) and ap_end and np.isfinite(ap_end[0]):
        fwhm_ms, fwhm_idx, fwhm_h = cp_fwhm_details(
            abf, sweep_4, voltage_ch, ap_start[0], ap_end[0], ind_peaks[0], sr
        )
        if fwhm_idx is not None and fwhm_h is not None:
            t0 = fwhm_idx[0] / sr
            t1 = fwhm_idx[1] / sr
            ax.hlines(
                fwhm_h, t0, t1,
                colors="darkorange",
                lw=5,
                zorder=6,
                label=f"FWHM 1st AP ({fwhm_ms} ms)" if fwhm_ms is not None else "FWHM 1st AP",
            )
            ax.plot([t0, t1], [fwhm_h, fwhm_h], "o", color="darkorange", ms=8, zorder=7)
            # mark AP baseline and peak used for 50% level
            i_ap0 = int(ap_start[0])
            i_pk = int(ind_peaks[0])
            ax.axhline(y[i_ap0], color="limegreen", ls="--", lw=1, alpha=0.7)
            ax.axhline(y[i_pk], color="red", ls=":", lw=1, alpha=0.5)
            ax.axhline(fwhm_h, color="darkorange", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vm (mV)")
    ax.set_title(f"{stem} — ch{voltage_ch} sweep {sweep_4} (>= {CP_MIN_APS} APs)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{stem}_ch{voltage_ch}_AP.png")
    _savefig_white(fig, path)
    plt.close(fig)
    return path


def _save_rin_qc_plot(abf, voltage_ch, current_ch, plots_dir, stem, borders, rin_mohm, r2, vm_range_label, rin_note):
    """Single I–V QC plot for Rin used in CC, Gj, and Cm."""
    plt = _get_agg_plt()

    if voltage_ch == 0:
        v_start, v_end, i_index = borders["start11"], borders["end11"], borders["Rtime_ch0"]
    else:
        v_start, v_end, i_index = borders["start21"], borders["end21"], borders["Rtime_ch2"]

    if voltage_ch == 0:
        spike_start, spike_end = borders["start11"], borders["end11"]
    else:
        spike_start, spike_end = borders["start21"], borders["end21"]
    series = collect_cc_iv_series(
        abf, voltage_ch, current_ch, v_start, v_end, i_index,
        spike_start=spike_start, spike_end=spike_end,
    )
    currents = np.array(series["currents"])
    voltages = np.array(series["voltages"])
    if len(currents) == 0:
        return None

    primary = np.array(series["in_primary"])
    fallback = np.array(series["in_fallback"]) & ~primary
    subth = np.array(series.get("subthreshold", [True] * len(currents)))
    spiked = ~subth
    other = subth & ~primary & ~fallback

    fig, ax = plt.subplots(figsize=(8, 6))
    if spiked.any():
        ax.scatter(
            currents[spiked], voltages[spiked], color="0.75", marker="x", s=50,
            label="after first AP (excluded)",
        )
    if other.any():
        ax.scatter(currents[other], voltages[other], color="lightgray", s=50, label="other Vm (subth)")
    if fallback.any():
        ax.scatter(
            currents[fallback], voltages[fallback], color="C1", s=60,
            label=f"fallback {RIN_VMIN_FALLBACK}...{RIN_VMAX_FALLBACK} mV",
        )
    if primary.any():
        ax.scatter(
            currents[primary], voltages[primary], color="C0", s=60,
            label=f"primary {RIN_VMIN}...{RIN_VMAX} mV",
        )

    if rin_mohm is not None:
        fit_mask = primary
        if fit_mask.sum() < RIN_MIN_POINTS:
            fit_mask = primary | fallback
        if fit_mask.sum() >= RIN_MIN_POINTS and r2 is not None:
            i_fit = np.linspace(currents[fit_mask].min(), currents[fit_mask].max(), 100)
            slope_mV_pA = rin_mohm / 1000.0
            intercept = float(np.mean(voltages[fit_mask] - slope_mV_pA * currents[fit_mask]))
            ax.plot(
                i_fit, slope_mV_pA * i_fit + intercept, color="red", lw=2,
                label=f"Rin = {rin_mohm} MΩ, R² = {r2}",
            )
        else:
            ax.plot([], [], color="red", lw=2, label=f"Rin = {rin_mohm} MΩ ({rin_note or 'delta'})")

    ax.set_xlabel("Current at Rtime (pA)")
    ax.set_ylabel("Mean Vm in stim epoch (mV)")
    ax.set_title(f"{stem} — ch{voltage_ch} I–V (Rin for CC / Gj / Cm)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{stem}_ch{voltage_ch}_IV_Rin.png")
    _savefig_white(fig, path)
    plt.close(fig)
    return path


def _save_tau_cm_qc_plot(abf, voltage_ch, plots_dir, stem, plot_meta):
    """Vm sweep for tau/Cm QC — saved even when tau/Cm failed (diagnostic)."""
    plt = _get_agg_plt()

    ch = f"ch{voltage_ch}"
    meta = plot_meta.get(ch)
    if not meta:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "no tau/Cm metadata", ha="center", va="center")
        ax.set_title(f"{stem} — ch{voltage_ch} tau/Cm (no data)")
        path = os.path.join(plots_dir, f"{stem}_ch{voltage_ch}_tau_Cm.png")
        _savefig_white(fig, path)
        plt.close(fig)
        return path

    if meta.get("sweep") is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        msg = meta.get("skip_reason", "tau/Cm not computed")
        if meta.get("epoch_label"):
            msg = f"{msg}\n{meta['epoch_label']}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True, fontsize=10)
        ax.set_title(f"{stem} — ch{voltage_ch} tau/Cm SKIPPED")
        path = os.path.join(plots_dir, f"{stem}_ch{voltage_ch}_tau_Cm.png")
        _savefig_white(fig, path)
        plt.close(fig)
        return path

    abf.setSweep(sweepNumber=meta["sweep"], channel=voltage_ch)
    ps, pe = meta["pre_start"], meta["post_stop"]
    stim = meta["stim_start"]
    t = abf.sweepX[ps:pe]
    y = abf.sweepY[ps:pe]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, y, color="C0", lw=1, label=f"Vm ch{voltage_ch}")
    if meta.get("pre_end") is not None:
        ax.axvspan(
            abf.sweepX[meta["pre_start"]], abf.sweepX[meta["pre_end"]],
            color="green", alpha=0.08, label="pre epoch",
        )
    if meta.get("post_end") is not None:
        ax.axvspan(
            abf.sweepX[meta["post_start"]], abf.sweepX[meta["post_end"]],
            color="blue", alpha=0.08, label="post epoch",
        )
    ax.axvline(abf.sweepX[stim], color="gray", ls=":", lw=1, alpha=0.8, label="tau start (pre_end)")
    if meta.get("v_pre") is not None:
        ax.axhline(meta["v_pre"], color="green", ls="--", lw=1.5, label=f"V_pre = {meta['v_pre']:.2f} mV")
    if meta.get("v_post") is not None:
        ax.axhline(meta["v_post"], color="blue", ls="--", lw=1.5, label=f"V_post = {meta['v_post']:.2f} mV")
    if meta.get("v_63") is not None:
        dv = meta.get("delta_v")
        dv_str = f" (ΔV={dv:.2f} mV)" if dv is not None else ""
        ax.axhline(
            meta["v_63"], color="darkorange", ls="--", lw=1.5,
            label=f"63% = {meta['v_63']:.2f} mV{dv_str}",
        )
    if meta.get("tau_index") is not None and meta.get("tau_ms") is not None:
        ax.scatter(
            abf.sweepX[meta["tau_index"]], abf.sweepY[meta["tau_index"]],
            color="purple", s=60, zorder=5, label=f"tau = {meta['tau_ms']:.3f} ms",
        )
        ax.axvline(abf.sweepX[meta["tau_index"]], color="purple", ls=":", alpha=0.5)

    rin = meta.get("rin_MOhm")
    tau_ms = meta.get("tau_ms")
    cm_pF = meta.get("cm_pF")
    cm_skip = meta.get("cm_skip")
    if tau_ms is not None and rin is not None and cm_pF is not None:
        cm_line = f"Cm = {cm_pF:.3f} pF  (tau/Rin×1000; Rin={rin} MΩ)"
    elif cm_skip:
        cm_line = f"Cm n/a ({cm_skip}; Rin={rin})"
    else:
        cm_line = meta.get("skip_reason", "")

    tau_str = f"{tau_ms:.3f} ms" if tau_ms is not None else "n/a"
    epoch_lbl = meta.get("epoch_label", "")
    sel_note = meta.get("selection_note") or ""
    title = (
        f"{stem} — ch{voltage_ch} sweep {meta['sweep']}  |  tau={tau_str}  |  {cm_line}\n"
        f"{epoch_lbl}"
    )
    if sel_note:
        title = f"{title}\n{sel_note}"
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vm (mV)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{stem}_ch{voltage_ch}_tau_Cm.png")
    _savefig_white(fig, path)
    plt.close(fig)
    return path


def save_qc_plots(
    abf, filepath, plots_dir, borders,
    rin_ch0, r2_ch0, rin_note_0,
    rin_ch2, r2_ch2, rin_note_2,
    tau_plot_meta=None,
):
    """Save AP + Rin I–V + tau/Cm QC PNGs into plots_dir."""
    os.makedirs(plots_dir, exist_ok=True)
    stem = _abf_stem(filepath)
    saved = []

    for voltage_ch, current_ch, rin, r2, note in (
        (0, 1, rin_ch0, r2_ch0, rin_note_0),
        (2, 3, rin_ch2, r2_ch2, rin_note_2),
    ):
        for saver, extra in (
            (_save_ap_qc_plot, (abf, voltage_ch, plots_dir, stem)),
            (_save_rin_qc_plot, (abf, voltage_ch, current_ch, plots_dir, stem, borders, rin, r2, None, note)),
            (_save_tau_cm_qc_plot, (abf, voltage_ch, plots_dir, stem, tau_plot_meta or {})),
        ):
            try:
                path = saver(*extra)
                if path:
                    saved.append(path)
            except Exception as exc:
                print(f"  QC plot skip ({saver.__name__} ch{voltage_ch}): {exc}")

    return saved


def _cc_direction_epochs(borders, direction):
    """Return pre_start, pre_end, post_start, post_end, rtime, active_ch, passive_ch, label."""
    if direction == "ch0->ch2":
        return (
            borders["start10"], borders["end10"],
            borders["start11"], borders["end11"],
            borders["Rtime_ch0"],
            0, 2, "CC12 (ch0→ch2)",
        )
    if direction == "ch2->ch0":
        return (
            borders["start20"], borders["end20"],
            borders["start21"], borders["end21"],
            borders["Rtime_ch2"],
            2, 0, "CC21 (ch2→ch0)",
        )
    raise ValueError(direction)


def _cc_block_vpost(abf, block, active_ch, post_start, post_end):
    """Mean Vm in post epoch (active cell) for each row in coupling block."""
    vposts = []
    for r in block:
        sn = r["sweep"]
        abf.setSweep(sweepNumber=sn, channel=active_ch)
        vposts.append(float(statistics.mean(abf.sweepY[post_start:post_end])))
    return vposts


def _attach_gj_for_plot(block, rin_passive):
    """Copy coupling rows and fill Gj_sweep_nS from Rin_passive when missing."""
    out = []
    for r in block:
        rr = dict(r)
        gj = rr.get("Gj_sweep_nS")
        skip = rr.get("Gj_sweep_skip_reason")
        if gj is None and rin_passive is not None:
            try:
                gj, skip = gj_nS(rr.get("CC"), rin_passive)
            except Exception:
                gj, skip = None, "gj_plot_error"
        rr["Gj_sweep_nS"] = gj
        rr["Gj_sweep_skip_reason"] = skip
        out.append(rr)
    return out


def _save_cc_traces_plot(abf, borders, block_02, block_20, plots_dir, stem):
    """Two panels: selected CC sweeps (valid vs skipped). Spike-cut sweeps omitted."""
    plt = _get_agg_plt()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    for ax, block, direction in (
        (axes[0], block_02, "ch0->ch2"),
        (axes[1], block_20, "ch2->ch0"),
    ):
        pre_s, pre_e, post_s, post_e, rtime, active_ch, passive_ch, label = (
            _cc_direction_epochs(borders, direction)
        )
        t_pre0 = abf.sweepX[pre_s]
        t_pre1 = abf.sweepX[min(pre_e, len(abf.sweepX) - 1)]
        t_post0 = abf.sweepX[post_s]
        t_post1 = abf.sweepX[min(post_e, len(abf.sweepX) - 1)]
        t_rt = abf.sweepX[min(rtime, len(abf.sweepX) - 1)]

        ax.axvspan(t_pre0, t_pre1, color="green", alpha=0.12, label="pre epoch")
        ax.axvspan(t_post0, t_post1, color="blue", alpha=0.12, label="post epoch")
        ax.axvline(t_rt, color="purple", ls="--", lw=1.5, label=f"Rtime (I sample @{rtime})")

        plotted_valid = False
        plotted_skip = False
        view_start = pre_s
        view_end = min(post_e + int(0.1 * abf.dataRate), len(abf.sweepY))

        for r in block:
            sn = r["sweep"]
            cc = r["CC"]
            skipped = cc is None
            color = "0.55" if skipped else "C0"
            alpha = 0.55 if skipped else 0.85
            lw = 1.0 if skipped else 1.4

            abf.setSweep(sweepNumber=sn, channel=active_ch)
            t = abf.sweepX[view_start:view_end]
            y_a = abf.sweepY[view_start:view_end]
            lbl = None
            if skipped and not plotted_skip:
                lbl = "skipped (no CC)"
                plotted_skip = True
            elif (not skipped) and not plotted_valid:
                lbl = "valid CC"
                plotted_valid = True
            ax.plot(t, y_a, color=color, alpha=alpha, lw=lw, label=lbl)

            abf.setSweep(sweepNumber=sn, channel=passive_ch)
            y_p = abf.sweepY[view_start:view_end]
            ax.plot(t, y_p, color=color, alpha=alpha * 0.7, lw=0.9, ls="--")

            mid = (post_s + post_e) // 2
            abf.setSweep(sweepNumber=sn, channel=active_ch)
            y_ann = float(abf.sweepY[mid])
            if cc is None:
                cc_txt = "skip"
            else:
                cc_txt = f"{cc:.3f}"
                gj = r.get("Gj_sweep_nS")
                if gj is not None:
                    cc_txt += f" Gj={gj:.2f}"
            ax.annotate(
                f"s{sn}:{cc_txt}",
                xy=(abf.sweepX[mid], y_ann),
                fontsize=7,
                color=color,
                alpha=0.9,
            )

        ax.set_ylabel("Vm (mV)")
        ax.set_title(
            f"{stem} — {label}  |  solid=active ch{active_ch}, dashed=passive ch{passive_ch}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"{stem}_CC_traces.png")
    _savefig_white(fig, path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def _save_cc_vs_vpost_plot(abf, borders, block_02, block_20, plots_dir, stem):
    """CC vs mean active Vm in post epoch; Gj on twin axis when present."""
    plt = _get_agg_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    for ax, block, direction, title in (
        (axes[0], block_02, "ch0->ch2", "CC12 (ch0→ch2)"),
        (axes[1], block_20, "ch2->ch0", "CC21 (ch2→ch0)"),
    ):
        _pre_s, _pre_e, post_s, post_e, _rtime, active_ch, _passive_ch, _ = (
            _cc_direction_epochs(borders, direction)
        )
        vposts = _cc_block_vpost(abf, block, active_ch, post_s, post_e)

        xv, yv, xs, ys = [], [], [], []
        xg, yg = [], []
        for r, vp in zip(block, vposts):
            sw_label = f"s{r['sweep']}"
            if r["CC"] is not None:
                xv.append(vp)
                yv.append(r["CC"])
                ax.annotate(
                    sw_label, (vp, r["CC"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="0.2", zorder=4,
                )
                gj = r.get("Gj_sweep_nS")
                if gj is not None:
                    xg.append(vp)
                    yg.append(gj)
            else:
                xs.append(vp)
                ys.append(0.0)
                ax.annotate(
                    sw_label, (vp, 0.0),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="0.45", zorder=4,
                )

        if xv:
            colors = ["C3" if c < 0 else "C0" for c in yv]
            ax.scatter(xv, yv, c=colors, s=55, zorder=3, label="valid CC")
        if xs:
            ax.scatter(xs, ys, c="0.5", marker="x", s=55, zorder=3, label="skipped")
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_xlabel(f"Mean Vm active ch{active_ch} in post [{post_s}:{post_e}] (mV)")
        ax.set_ylabel("CC", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.set_title(f"{stem} — {title}")

        if xg:
            ax2 = ax.twinx()
            ax2.scatter(xg, yg, c="C3", marker="s", s=36, zorder=2, alpha=0.85, label="Gj")
            ax2.set_ylabel("Gj (nS)", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")
            handles, labels = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(handles + h2, labels + l2, loc="best", fontsize=8)
        else:
            ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"{stem}_CC_vs_Vpost.png")
    _savefig_white(fig, path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def save_cc_qc_plots(
    abf, filepath, plots_dir, borders,
    block_02, block_20,
    rin_ch0=None, rin_ch2=None,
):
    """Save CC QC figures into CC_plots (traces + CC/Gj vs Vm)."""
    if not plots_dir:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), CC_PLOTS_SUBDIR)
    os.makedirs(plots_dir, exist_ok=True)
    stem = _abf_stem(filepath)
    block_02 = _attach_gj_for_plot(block_02, rin_ch2)
    block_20 = _attach_gj_for_plot(block_20, rin_ch0)
    saved = []
    jobs = []
    if SAVE_CC_TRACE_PLOTS:
        jobs.append((_save_cc_traces_plot, (abf, borders, block_02, block_20, plots_dir, stem)))
    if SAVE_CC_VPOST_PLOTS:
        jobs.append((_save_cc_vs_vpost_plot, (abf, borders, block_02, block_20, plots_dir, stem)))
    for saver, args in jobs:
        try:
            path = saver(*args)
            if path:
                saved.append(path)
        except Exception as exc:
            print(f"  CC plot skip ({saver.__name__}): {exc}")
            traceback.print_exc()
    return saved


def gj_nS(cc, rin_passive_MOhm):
    """
    Gj [nS] = (CC/(1-CC)) * (1/Rin_passive [MOhm]) * 1000.

    Rin_passive: for Gj12 (ch0→ch2) use Rin of cell 2 (ch2);
                 for Gj21 (ch2→ch0) use Rin of cell 1 (ch0).
    Requires 0 < CC < 1.
    """
    if cc is None or rin_passive_MOhm is None or rin_passive_MOhm == 0:
        return None, "missing CC or Rin"
    if cc <= 0:
        return None, f"CC <= 0 (got {cc})"
    if cc >= 1:
        return None, f"CC >= 1 (got {cc})"
    try:
        return round((cc / (1.0 - cc)) * (1000.0 / rin_passive_MOhm), 4), None
    except (ZeroDivisionError, TypeError):
        return None, "Gj calculation error"



def mean_valid_cc(cc_list):
    vals = [c for c in cc_list if c is not None]
    return round(statistics.mean(vals), 4) if vals else None


def n_valid_cc(cc_list):
    return sum(c is not None for c in cc_list)


def n_negative_cc(cc_list):
    """Count of valid (non-None) CC values that are negative."""
    return sum(c is not None and c < 0 for c in cc_list)


def _parse_recording_datetime(value):
    """Parse ABF recording time from summary row to datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def folder_summary_complete_rows(summary_rows):
    """
    Files with Rin1, Rin2, CC12, CC21, Gj12, Gj21 all computed, sorted by time.

    Returns list of (datetime, row_dict).
    """
    keys = ("Rin1_MOhm", "Rin2_MOhm", "CC12", "CC21", "Gj12_nS", "Gj21_nS")
    out = []
    for row in summary_rows:
        if any(row.get(k) is None for k in keys):
            continue
        dt = _parse_recording_datetime(row.get("recording_datetime"))
        if dt is None:
            continue
        out.append((dt, row))
    out.sort(key=lambda item: item[0])
    return out


def save_folder_summary_plot(summary_rows, out_path, title=None):
    """
    One figure (3 panels) vs recording time for files with full Rin + CC + Gj:

    1) Rin1 and Rin2
    2) CC12 (left) and Gj12 (right)
    3) CC21 (left) and Gj21 (right)
    """
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    pairs = folder_summary_complete_rows(summary_rows)
    if not pairs:
        return None

    times = [dt for dt, _ in pairs]
    rin1 = [r["Rin1_MOhm"] for _, r in pairs]
    rin2 = [r["Rin2_MOhm"] for _, r in pairs]
    cc12 = [r["CC12"] for _, r in pairs]
    cc21 = [r["CC21"] for _, r in pairs]
    gj12 = [r["Gj12_nS"] for _, r in pairs]
    gj21 = [r["Gj21_nS"] for _, r in pairs]
    names = [r.get("file", "") for _, r in pairs]

    plt = _get_agg_plt()
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if title:
        fig.suptitle(title, fontsize=12)

    ax0 = axes[0]
    ax0.plot(times, rin1, "o-", color="C0", label="Rin1 (ch0)")
    ax0.plot(times, rin2, "s-", color="C1", label="Rin2 (ch2)")
    ax0.set_ylabel("Rin (MΩ)")
    ax0.set_title(f"Rin1 / Rin2  ({len(pairs)} files with full Rin+CC+Gj)")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    def _cc_gj_panel(ax, cc_vals, gj_vals, cc_label, gj_label, panel_title):
        ax.plot(times, cc_vals, "o-", color="C0", label=cc_label)
        ax.set_ylabel("CC", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(times, gj_vals, "s--", color="C3", label=gj_label)
        ax2.set_ylabel("Gj (nS)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.3)
        lines = ax.get_lines() + ax2.get_lines()
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc="best", fontsize=8)
        return ax2

    _cc_gj_panel(axes[1], cc12, gj12, "CC12", "Gj12", "CC12 / Gj12 (ch0→ch2)")
    _cc_gj_panel(axes[2], cc21, gj21, "CC21", "Gj21", "CC21 / Gj21 (ch2→ch0)")

    locator = AutoDateLocator()
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axes[2].set_xlabel("Recording time")
    for i, (t, name) in enumerate(zip(times, names)):
        stem = os.path.splitext(str(name))[0]
        if stem:
            axes[0].annotate(
                stem, (t, rin1[i]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=6, alpha=0.7, rotation=30,
            )

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _savefig_white(fig, out_path)
    plt.close(fig)
    return out_path


def file_cc_norm_curves(all_rows, direction):
    """
    Per-file (Vm, CC_norm) curves for one direction, sorted by Vm within file.

    Returns list of (file_name, recording_datetime, vm_list, cc_norm_list).
    """
    by_file = {}
    file_dt = {}
    for row in all_rows:
        if row.get("direction") != direction:
            continue
        cc_n = row.get("CC_norm")
        vm = row.get("Vm_active_stim_mV")
        if cc_n is None or vm is None:
            continue
        fname = row.get("file") or "?"
        by_file.setdefault(fname, []).append((float(vm), float(cc_n)))
        if fname not in file_dt:
            file_dt[fname] = _parse_recording_datetime(row.get("recording_datetime"))

    curves = []
    for fname, pairs in by_file.items():
        pairs.sort(key=lambda p: p[0])
        vms = [p[0] for p in pairs]
        norms = [p[1] for p in pairs]
        curves.append((fname, file_dt.get(fname), vms, norms))
    curves.sort(key=lambda c: (c[1] is None, c[1] or datetime.min, c[0]))
    return curves


def _cc_vm_file_color_map(all_rows):
    """
    Color 0..1 by recording time (same file → same color on both panels).

    Returns (fname -> pos, first_label, last_label).
    """
    seen = {}
    for row in all_rows:
        fname = row.get("file") or "?"
        if fname not in seen:
            seen[fname] = _parse_recording_datetime(row.get("recording_datetime"))
    items = sorted(
        seen.items(),
        key=lambda it: (it[1] is None, it[1] or datetime.min, it[0]),
    )
    n = max(len(items) - 1, 1)
    pos = {fname: (i / n) for i, (fname, _dt) in enumerate(items)}

    def _lbl(fname, dt):
        stem = os.path.splitext(str(fname))[0]
        if dt is None:
            return stem
        return f"{stem}  {dt.strftime('%H:%M')}"

    first_lbl = _lbl(*items[0]) if items else ""
    last_lbl = _lbl(*items[-1]) if items else ""
    return pos, first_lbl, last_lbl


def _poly_cc_vs_vm(vm, cc, degree, n_grid=80):
    """Return (x_fit, y_fit, coeffs, r2) or Nones if the poly is not defined."""
    vm = np.asarray(vm, dtype=float)
    cc = np.asarray(cc, dtype=float)
    ok = np.isfinite(vm) & np.isfinite(cc)
    vm, cc = vm[ok], cc[ok]
    if len(vm) < degree + 1 or float(np.ptp(vm)) < 1e-9:
        return None, None, None, None
    if len(np.unique(np.round(vm, 6))) < degree + 1:
        return None, None, None, None
    coeffs = np.polyfit(vm, cc, degree)
    pred = np.polyval(coeffs, vm)
    ss_res = float(np.sum((cc - pred) ** 2))
    ss_tot = float(np.sum((cc - np.mean(cc)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    x = np.linspace(float(vm.min()), float(vm.max()), n_grid)
    y = np.polyval(coeffs, x)
    return x, y, coeffs, float(r2)


def _linear_cc_vs_vm(vm, cc, n_grid=80):
    """Return (x_fit, y_fit, slope, intercept, r2) or Nones if a line is not defined."""
    x, y, coeffs, r2 = _poly_cc_vs_vm(vm, cc, 1, n_grid=n_grid)
    if x is None:
        return None, None, None, None, None
    return x, y, float(coeffs[0]), float(coeffs[1]), r2


def save_folder_cc_norm_vs_vm_plot(all_rows, out_path, title=None):
    """
    Folder overview: CC_norm vs Vm, color = recording order (colorbar).

    Solid = linear fit; dashed = 2nd-order polynomial (if ≥3 Vm points).
    Two panels: CC12 (ch0→ch2) and CC21 (ch2→ch0).
    """
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    plt = _get_agg_plt()
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, layout="constrained")
    except TypeError:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    if title:
        fig.suptitle(title, fontsize=12)

    file_pos, first_lbl, last_lbl = _cc_vm_file_color_map(all_rows)
    try:
        cmap = plt.colormaps.get_cmap(CC_VM_CMAP)
    except Exception:
        cmap = plt.cm.get_cmap(CC_VM_CMAP)

    panels = (
        ("ch0->ch2", "CC12 (ch0→ch2)"),
        ("ch2->ch0", "CC21 (ch2→ch0)"),
    )
    any_data = False
    for ax, (direction, panel_title) in zip(axes, panels):
        curves = file_cc_norm_curves(all_rows, direction)
        if not curves:
            ax.set_title(f"{panel_title} — no data")
            ax.grid(True, alpha=0.3)
            continue
        any_data = True

        for fname, _dt, vms, norms in curves:
            color = cmap(file_pos.get(fname, 0.5))
            ax.scatter(vms, norms, color=color, s=28, zorder=3, alpha=0.9)
            if CC_VM_FIT_LINEAR:
                x1, y1, _s, _b, _r2 = _linear_cc_vs_vm(vms, norms)
                if x1 is not None:
                    ax.plot(x1, y1, "-", color=color, lw=1.3, alpha=0.85)
            if CC_VM_FIT_QUADRATIC:
                x2, y2, _c, _r2 = _poly_cc_vs_vm(vms, norms, 2)
                if x2 is not None:
                    ax.plot(x2, y2, "--", color=color, lw=1.5, alpha=0.9)

        ax.axhline(1.0, color="0.45", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel("Vm active during stim (mV)")
        ax.set_title(f"{panel_title} — {len(curves)} file(s)")
        ax.grid(True, alpha=0.3)
        style_handles = []
        if CC_VM_FIT_LINEAR:
            style_handles.append(Line2D([0], [0], color="0.25", ls="-", lw=1.4, label="linear"))
        if CC_VM_FIT_QUADRATIC:
            style_handles.append(
                Line2D([0], [0], color="0.25", ls="--", lw=1.5, label="quadratic")
            )
        if style_handles:
            ax.legend(handles=style_handles, loc="best", fontsize=8)

    if not any_data:
        plt.close(fig)
        return None

    axes[0].set_ylabel("CC_norm (CC / file mean)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("recording order (first → last)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([first_lbl, last_lbl])
    cbar.ax.tick_params(labelsize=7)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _savefig_white(fig, out_path)
    plt.close(fig)
    return out_path


def build_file_summary_row(
    name,
    rec_dt,
    rin_ch0,
    rin_ch2,
    rin_note_0,
    rin_note_2,
    rin_skip_0,
    rin_skip_2,
    cc_mean_02,
    cc_mean_20,
    n_avg_02,
    n_avg_20,
    gj_02,
    gj_20,
    gj_skip_02,
    gj_skip_20,
    n_sel_02,
    n_sel_20,
    n_neg_02=0,
    n_neg_20=0,
    cell_props=None,
    file_skip_reason=None,
):
    """One row per file for Excel sheet File_summary (CC12/Gj12 = ch0->ch2)."""
    row = {
        "file": name,
        "recording_datetime": rec_dt,
        "Rin1_MOhm": rin_ch0,
        "Rin2_MOhm": rin_ch2,
        "Rin1_note": rin_note_0,
        "Rin2_note": rin_note_2,
        "Rin1_skip_reason": rin_skip_0,
        "Rin2_skip_reason": rin_skip_2,
        "CC12": cc_mean_02,
        "CC21": cc_mean_20,
        "Gj12_nS": gj_02,
        "Gj21_nS": gj_20,
        "Gj12_skip_reason": gj_skip_02,
        "Gj21_skip_reason": gj_skip_20,
        "n_CC_avg_12": n_avg_02,
        "n_CC_avg_21": n_avg_20,
        "n_CC_neg_12": n_neg_02,
        "n_CC_neg_21": n_neg_20,
        "CC_used_neg_12": "Yes" if n_neg_02 > 0 else "No",
        "CC_used_neg_21": "Yes" if n_neg_20 > 0 else "No",
        "CC_neg_of_avg_12": f"{n_neg_02}/{n_avg_02}",
        "CC_neg_of_avg_21": f"{n_neg_20}/{n_avg_20}",
        "n_CC_selected_12": n_sel_02,
        "n_CC_selected_21": n_sel_20,
    }
    if cell_props:
        row.update(cell_props)
    if file_skip_reason:
        row["file_skip_reason"] = file_skip_reason
    return row


def _skipped_file_row(name, rec_dt, reason):
    return [sweep_only_row({
        "file": name,
        "recording_datetime": rec_dt,
        "sweep": None,
        "direction": None,
        "cur_step_pA": None,
        "delta_V_active_mV": None,
        "delta_V_passive_mV": None,
        "CC": None,
        "CC_skip_reason": reason,
        "Gj_sweep_nS": None,
        "Gj_sweep_skip_reason": reason,
    })]


def analyze_abf_file(filepath, plots_dir=None, cc_plots_dir_path=None, spikelet_plots_dir_path=None):
    """Return (per_sweep_rows, file_summary_row, qc_plot_paths, spikelet_rows)."""
    name = os.path.basename(filepath)
    abf = pyabf.ABF(filepath)
    rec_dt = recording_datetime_str(abf)
    plot_paths = []

    if abf.channelCount != EXPECTED_CHANNELS:
        reason = f"expected {EXPECTED_CHANNELS} channels, got {abf.channelCount}"
        empty_props = {
            **empty_cell_props_fields(),
            **empty_tau_cm_fields(),
            **empty_spikelet_summary_fields(),
        }
        return (
            _skipped_file_row(name, rec_dt, reason),
            build_file_summary_row(
                name, rec_dt, None, None, None, None, reason, reason,
                None, None, 0, 0, None, None, reason, reason, 0, 0,
                cell_props=empty_props,
                file_skip_reason=reason,
            ),
            plot_paths,
            [],
        )

    cell_props = empty_cell_props_fields()
    try:
        cell_props = cell_properties_for_file(abf)
    except Exception as exc:
        msg = str(exc)
        cell_props["props_skip_reason_ch0"] = msg
        cell_props["props_skip_reason_ch2"] = msg

    b = time_period_borders(abf.dataRate)
    w0 = (b["start10"], b["end10"], b["start11"], b["end11"])
    w2 = (b["start20"], b["end20"], b["start21"], b["end21"])

    sweeps_ch0 = cc_sweep_indices_direction(
        abf, 0, 2, b["start10"], b["end10"], b["start11"], b["end11"]
    )
    sweeps_ch2 = cc_sweep_indices_direction(
        abf, 2, 0, b["start20"], b["end20"], b["start21"], b["end21"]
    )

    block_02 = coupling_block(abf, sweeps_ch0, 0, 2, 1, w0)
    block_20 = coupling_block(abf, sweeps_ch2, 2, 0, 3, w2)

    cc_list_02 = [r["CC"] for r in block_02]
    cc_list_20 = [r["CC"] for r in block_20]
    cc_mean_02 = mean_valid_cc(cc_list_02)
    cc_mean_20 = mean_valid_cc(cc_list_20)
    cc_normalize_block(block_02, cc_mean_02)
    cc_normalize_block(block_20, cc_mean_20)
    n_avg_02 = n_valid_cc(cc_list_02)
    n_avg_20 = n_valid_cc(cc_list_20)
    n_neg_02 = n_negative_cc(cc_list_02)
    n_neg_20 = n_negative_cc(cc_list_20)

    rin_ch0, r2_ch0, n0, rin_skip_0, rin_range_0, rin_note_0, _ = rin_for_channel(
        abf, 0, 1, b["start10"], b["end10"], b["start11"], b["end11"], b["Rtime_ch0"]
    )
    rin_ch2, r2_ch2, n2, rin_skip_2, rin_range_2, rin_note_2, _ = rin_for_channel(
        abf, 2, 3, b["start20"], b["end20"], b["start21"], b["end21"], b["Rtime_ch2"]
    )

    tau_cm = empty_tau_cm_fields()
    tau_plot_meta = {}
    try:
        tau_cm_raw = tau_cm_for_file(abf, rin_ch0, rin_ch2, borders=b)
        tau_plot_meta = tau_cm_raw.pop("_tau_plot_meta", {})
        tau_cm = {k: v for k, v in tau_cm_raw.items() if not k.startswith("_")}
    except Exception as exc:
        msg = str(exc)
        tau_cm["tau_skip_reason_ch0"] = msg
        tau_cm["tau_skip_reason_ch2"] = msg

    spikelet_rows = []
    spikelet_summary = empty_spikelet_summary_fields()
    try:
        sp_dir = spikelet_plots_dir_path
        if SAVE_SPIKELET_PLOTS and not sp_dir:
            sp_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), SPIKELET_PLOTS_SUBDIR)
        spikelet_rows, spikelet_summary, sp_paths = spikelets_for_file(
            abf, name, rec_dt,
            plots_dir=sp_dir if SAVE_SPIKELET_PLOTS else None,
            stem=_abf_stem(filepath),
        )
        plot_paths.extend(sp_paths)
    except Exception as exc:
        print(f"  Spikelet analysis error: {exc}")
        traceback.print_exc()
        spikelet_summary = empty_spikelet_summary_fields()
        spikelet_summary["spikelet_skip_reason_12"] = str(exc)
        spikelet_summary["spikelet_skip_reason_21"] = str(exc)

    file_summary_extra = {
        **cell_props,
        **tau_cm,
        **spikelet_summary,
        "Rin_ch0_MOhm": rin_ch0,
        "R2_ch0": r2_ch0,
        "n_IV_ch0": n0,
        "Rin_ch0_skip_reason": rin_skip_0,
        "Rin_ch0_note": rin_note_0,
        "Rin_vm_range_ch0": rin_range_0,
        "Rin_ch2_MOhm": rin_ch2,
        "R2_ch2": r2_ch2,
        "n_IV_ch2": n2,
        "Rin_ch2_skip_reason": rin_skip_2,
        "Rin_ch2_note": rin_note_2,
        "Rin_vm_range_ch2": rin_range_2,
    }

    gj_02, gj_skip_02 = gj_nS(cc_mean_02, rin_ch2)
    gj_20, gj_skip_20 = gj_nS(cc_mean_20, rin_ch0)

    summary_row = build_file_summary_row(
        name,
        rec_dt,
        rin_ch0,
        rin_ch2,
        rin_note_0,
        rin_note_2,
        rin_skip_0,
        rin_skip_2,
        cc_mean_02,
        cc_mean_20,
        n_avg_02,
        n_avg_20,
        gj_02,
        gj_20,
        gj_skip_02,
        gj_skip_20,
        len(sweeps_ch0),
        len(sweeps_ch2),
        n_neg_02=n_neg_02,
        n_neg_20=n_neg_20,
        cell_props={
            **file_summary_extra,
            "CC_mean_ch0to2": cc_mean_02,
            "CC_mean_ch2to0": cc_mean_20,
            "n_CC_avg_ch0to2": n_avg_02,
            "n_CC_avg_ch2to0": n_avg_20,
            "n_CC_neg_ch0to2": n_neg_02,
            "n_CC_neg_ch2to0": n_neg_20,
            "CC_used_neg_ch0to2": "Yes" if n_neg_02 > 0 else "No",
            "CC_used_neg_ch2to0": "Yes" if n_neg_20 > 0 else "No",
            "CC_neg_of_avg_ch0to2": f"{n_neg_02}/{n_avg_02}",
            "CC_neg_of_avg_ch2to0": f"{n_neg_20}/{n_avg_20}",
            "Gj_ch0to2_nS": gj_02,
            "Gj_ch0to2_skip_reason": gj_skip_02,
            "Gj_ch2to0_nS": gj_20,
            "Gj_ch2to0_skip_reason": gj_skip_20,
            "n_CC_sweeps_ch0to2": len(sweeps_ch0),
            "n_CC_sweeps_ch2to0": len(sweeps_ch2),
        },
    )

    rows = []
    for r in block_02:
        gj_sw, gj_sw_skip = gj_nS(r["CC"], rin_ch2)
        rows.append(sweep_only_row({
            "file": name,
            "recording_datetime": rec_dt,
            "direction": "ch0->ch2",
            "Gj_sweep_nS": gj_sw,
            "Gj_sweep_skip_reason": gj_sw_skip,
            **r,
        }))
    for r in block_20:
        gj_sw, gj_sw_skip = gj_nS(r["CC"], rin_ch0)
        rows.append(sweep_only_row({
            "file": name,
            "recording_datetime": rec_dt,
            "direction": "ch2->ch0",
            "Gj_sweep_nS": gj_sw,
            "Gj_sweep_skip_reason": gj_sw_skip,
            **r,
        }))

    if not rows:
        no_sw_reason = "no sweeps without spikes on active or passive in stimulus period"
        rows.append(sweep_only_row({
            "file": name,
            "recording_datetime": rec_dt,
            "CC_skip_reason": no_sw_reason,
        }))
        summary_row["file_skip_reason"] = no_sw_reason

    if plots_dir and SAVE_QC_PLOTS:
        try:
            qc_paths = save_qc_plots(
                abf,
                filepath,
                plots_dir,
                b,
                rin_ch0,
                r2_ch0,
                rin_note_0,
                rin_ch2,
                r2_ch2,
                rin_note_2,
                tau_plot_meta=tau_plot_meta,
            )
            plot_paths.extend(qc_paths)
        except Exception as exc:
            print(f"  QC plots error: {exc}")

    if SAVE_CC_PLOTS:
        cc_dir = cc_plots_dir_path
        if not cc_dir:
            cc_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), CC_PLOTS_SUBDIR)
        try:
            cc_paths = save_cc_qc_plots(
                abf,
                filepath,
                cc_dir,
                b,
                block_02,
                block_20,
                rin_ch0=rin_ch0,
                rin_ch2=rin_ch2,
            )
            plot_paths.extend(cc_paths)
        except Exception as exc:
            print(f"  CC plots error: {exc}")
            traceback.print_exc()

    return rows, summary_row, plot_paths, spikelet_rows
