"""Drop-in replacements for spikelet detection/plotting in cc_calc_core.py.

Copy the constants and functions below over the same names in cc_calc_core.py,
then reload notebook cell 0.

Changes vs first version:
- peak window 5 ms -> 15 ms
- noise gate 3*RMS -> 1.5 * MAD (less strict, less inflated by artifacts)
- allow peak at the END of the window if the trace is still rising
- reject only if the max is at the first sample (no rise after AP start)
- QC figure: active and passive on separate subplots (different y scales)
"""

# --- replace these constants in cc_calc_core.py ---
SPIKELET_BASELINE_MS = 1.0
SPIKELET_PEAK_MS = 15.0  # was 5, then 8
SPIKELET_NOISE_K = 1.5  # was 3; 1.5 x noise
SPIKELET_MIN_AMP_MV = 0.15  # extra floor so tiny bumps in ultra-low noise still need 0.15 mV


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


def _spikelet_local_peak_index(seg):
    """Index of max after AP start.

    Relaxed vs v1 (which required a strict interior local max):
    - first sample -> not a spikelet (no rise after start)
    - last sample OK if still rising (peak may sit at 15 ms edge)
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


def save_spikelet_qc_plot(abf, plot_meta, plots_dir, stem):
    """Active / passive on separate y-scales; snippets panel with twin axis."""
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
        3, 1, figsize=(12, 10), sharex=False,
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.0]},
    )
    ax_a, ax_p, ax_avg = axes

    ss = int(plot_meta["stim_start"])
    se = int(plot_meta["stim_stop"])
    i_left = max(int(plot_meta["pre_start"]), 0)
    i_right = min(se + int(0.05 * sr), len(t) - 1)

    ax_a.plot(t, y_a, color="C0", lw=1.0, label=f"active ch{active_ch}")
    ax_p.plot(t, y_p, color="C1", lw=1.0, label=f"passive ch{passive_ch}")
    for ax in (ax_a, ax_p):
        ax.axvline(t[min(ss, len(t) - 1)], color="0.4", ls="--", lw=0.8)
        ax.axvline(t[min(se, len(t) - 1)], color="0.4", ls="--", lw=0.8)

    starts = plot_meta["ap_starts"]
    peaks = plot_meta["ind_peaks"]
    for i, ip in enumerate(peaks):
        ip = int(ip)
        if 0 <= ip < len(t):
            ax_a.scatter(
                t[ip], y_a[ip], c="C3", s=28, zorder=5, marker="o",
                label="AP peak" if i == 0 else None,
            )
        if i < len(starts) and np.isfinite(starts[i]):
            i0 = int(starts[i])
            if 0 <= i0 < len(t):
                ax_a.scatter(
                    t[i0], y_a[i0], c="limegreen", s=28, zorder=5, marker="v",
                    label="AP start" if i == 0 else None,
                )
            if i >= 1 and 0 <= i0 < len(t):
                t0 = t[i0]
                ax_p.axvspan(t0 - SPIKELET_BASELINE_MS / 1000.0, t0, color="0.7", alpha=0.25)
                ax_p.axvspan(t0, t0 + SPIKELET_PEAK_MS / 1000.0, color="C4", alpha=0.12)

    labeled_sp = False
    for r in plot_meta["ap_rows"]:
        if r.get("t_peak_passive_ms") is None:
            continue
        tp = r["t_peak_passive_ms"] / 1000.0
        idx = min(max(int(round(tp * sr)), 0), len(y_p) - 1)
        ax_p.scatter(
            t[idx], y_p[idx], c="darkorange", s=40, zorder=6, marker="x",
            label="spikelet peak" if not labeled_sp else None,
        )
        labeled_sp = True

    src = (plot_meta.get("metrics") or {}).get("metric_source")
    ax_a.set_ylabel("Active Vm (mV)")
    ax_p.set_ylabel("Passive Vm (mV)")
    ax_a.set_title(
        f"{stem} — {direction}  sweep {sweep} ({plot_meta.get('tier')}, "
        f"n_AP={plot_meta.get('n_ap')}, source={src})"
    )
    ax_a.legend(loc="upper right", fontsize=7)
    ax_p.legend(loc="upper right", fontsize=7)
    ax_a.set_xlim(t[i_left], t[i_right])
    ax_p.set_xlim(t[i_left], t[i_right])
    ax_a.grid(True, alpha=0.25)
    ax_p.grid(True, alpha=0.25)

    t_snip = (np.arange(-n_pre, n_post) / float(sr)) * 1000.0
    for sn in plot_meta.get("snips_p") or []:
        if len(sn) == len(t_snip):
            ax_avg.plot(t_snip, sn, color="C1", lw=0.7, alpha=0.35)
    if plot_meta.get("mean_p") is not None and len(plot_meta["mean_p"]) == len(t_snip):
        ax_avg.plot(t_snip, plot_meta["mean_p"], color="C1", lw=2.2, label="mean spikelet")
    ax_avg.axvline(0, color="limegreen", ls="--", lw=1.0)
    ax_avg.axvspan(-SPIKELET_BASELINE_MS, 0, color="0.7", alpha=0.25)
    ax_avg.axvspan(0, SPIKELET_PEAK_MS, color="C4", alpha=0.12)
    ax_avg.set_xlabel("Time from AP start (ms)")
    ax_avg.set_ylabel("Passive Vm (mV)", color="C1")
    ax_avg.tick_params(axis="y", labelcolor="C1")

    if plot_meta.get("mean_a") is not None and len(plot_meta["mean_a"]) == len(t_snip):
        ax_avg2 = ax_avg.twinx()
        ax_avg2.plot(t_snip, plot_meta["mean_a"], color="C0", lw=1.4, alpha=0.8, label="mean AP")
        ax_avg2.set_ylabel("Active Vm (mV)", color="C0")
        ax_avg2.tick_params(axis="y", labelcolor="C0")

    ax_avg.set_title("AP2+ aligned to active AP start (passive scale left, AP scale right)")
    ax_avg.legend(loc="upper left", fontsize=7)
    ax_avg.grid(True, alpha=0.3)

    fig.tight_layout()
    tag = direction.replace(">", "")
    path = os.path.join(plots_dir, f"{stem}_{tag}_spikelets.png")
    _savefig_white(fig, path)
    plt.close(fig)
    return path


# In analyze_spikelets_direction, replace the k*RMS check with:
#
#     ok, why = spikelet_amp_passes(amp_p, rms)
#     if not ok:
#         row["skip_reason"] = why
#     else:
#         row["detected"] = True
#         row["skip_reason"] = None
#
# and for the average trace:
#
#     ok, _ = spikelet_amp_passes(amp_p_avg, rms)
#     avg_detected = ok
#
# Also set SPIKELET_PEAK_MS = 15.0 and SPIKELET_NOISE_K = 1.5 at the top of cc_calc_core.py.
