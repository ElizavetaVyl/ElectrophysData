"""Restore per-file CC/Gj QC plots in cc_calc_core.py.

Paste the CONFIG flags and functions below over the same names, then reload
notebook cell 0. Also replace the `if cc_plots_dir_path and SAVE_CC_PLOTS`
block in analyze_abf_file with the snippet at the bottom.

Why they vanished: SAVE_CC_TRACE_PLOTS was set False to speed up the batch.
CC vs sweep was removed on purpose earlier — do not restore that figure.

What this restores:
  CC_plots/{stem}_CC_traces.png     — selected sweeps, pre/post, CC (+Gj) labels
  CC_plots/{stem}_CC_vs_Vpost.png   — CC vs Vm, Gj on a twin axis when available

Folder plots are unchanged:
  {folder}_Rin_CC_Gj_over_time.png
  {folder}_CC_norm_vs_Vm.png
"""

import os
import traceback

# --- replace these flags in cc_calc_core.py ---
SAVE_CC_PLOTS = True
SAVE_CC_TRACE_PLOTS = True  # was False (speed); needed for sweep QC
SAVE_CC_VPOST_PLOTS = True
CC_PLOTS_SUBDIR = "CC_plots"


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


# --- replace the CC-plot call at the end of analyze_abf_file with this ---
#
#     if SAVE_CC_PLOTS:
#         cc_dir = cc_plots_dir_path
#         if not cc_dir:
#             cc_dir = os.path.join(
#                 os.path.dirname(os.path.abspath(filepath)), CC_PLOTS_SUBDIR
#             )
#         try:
#             cc_paths = save_cc_qc_plots(
#                 abf, filepath, cc_dir, b, block_02, block_20,
#                 rin_ch0=rin_ch0, rin_ch2=rin_ch2,
#             )
#             plot_paths.extend(cc_paths)  # extend, do not assign
#         except Exception as exc:
#             print(f"  CC plots error: {exc}")
#             traceback.print_exc()
#
# Notebook cell 3 must still pass cc_plots_dir_path=cc_dir (or None: fallback
# next to the ABF). Unpack 4 values if spikelets are enabled:
#     detail, summary, plot_paths, spikelet_rows = analyze_abf_file(...)
