"""Replace save_folder_cc_norm_vs_vm_plot in cc_calc_core.py.

Do not connect CC vs Vm points with a polyline. Sweep order is not voltage
order, so a broken line zigzags and is not a Vm dependence.

Default: scatter + ordinary least-squares linear fit per file.
Why linear, not quadratic / LOWESS:
  - typically 2–5 CC sweeps per file (before first AP)
  - quadratic / LOWESS overfit that n
  - slope is readable: d(CC_norm)/dVm  (1/mV)

If Vm has no range (all points at the same voltage), only scatter is drawn.
Legend: slope and R² so a bad linear description is obvious.

Requires existing helpers in cc_calc_core.py:
  file_cc_norm_curves, _get_agg_plt, _savefig_white
"""

import os

import numpy as np


def _linear_cc_vs_vm(vm, cc, n_grid=80):
    """Return (x_fit, y_fit, slope, intercept, r2) or Nones if a line is not defined."""
    vm = np.asarray(vm, dtype=float)
    cc = np.asarray(cc, dtype=float)
    ok = np.isfinite(vm) & np.isfinite(cc)
    vm, cc = vm[ok], cc[ok]
    if len(vm) < 2 or float(np.ptp(vm)) < 1e-9:
        return None, None, None, None, None
    slope, intercept = np.polyfit(vm, cc, 1)
    yhat = slope * vm + intercept
    ss_res = float(np.sum((cc - yhat) ** 2))
    ss_tot = float(np.sum((cc - np.mean(cc)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    x = np.linspace(float(vm.min()), float(vm.max()), n_grid)
    y = slope * x + intercept
    return x, y, float(slope), float(intercept), float(r2)


def save_folder_cc_norm_vs_vm_plot(all_rows, out_path, title=None):
    """
    Folder overview: CC_norm vs Vm — scatter + linear fit, one color per file.

    Two panels: CC12 (ch0→ch2) and CC21 (ch2→ch0).
    """
    plt = _get_agg_plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    if title:
        fig.suptitle(title, fontsize=12)

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
        try:
            import matplotlib.colormaps as cmaps
            cmap = cmaps["tab10"] if len(curves) <= 10 else cmaps["viridis"]
        except ImportError:
            cmap = plt.cm.get_cmap("tab10" if len(curves) <= 10 else "viridis")

        for i, (fname, _dt, vms, norms) in enumerate(curves):
            color = cmap(i / max(len(curves) - 1, 1))
            label = os.path.splitext(str(fname))[0]
            ax.scatter(vms, norms, color=color, s=28, zorder=3, alpha=0.9)
            xfit, yfit, slope, _b, r2 = _linear_cc_vs_vm(vms, norms)
            if xfit is not None:
                ax.plot(
                    xfit,
                    yfit,
                    "-",
                    color=color,
                    lw=1.4,
                    alpha=0.85,
                    label=f"{label}  s={slope:.3f}  R²={r2:.2f}",
                )
            else:
                ax.plot([], [], "-", color=color, label=label)

        ax.axhline(1.0, color="0.45", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("Vm active during stim (mV)")
        ax.set_title(f"{panel_title} — {len(curves)} file(s), linear fit")
        ax.grid(True, alpha=0.3)
        if len(curves) <= 12:
            ax.legend(loc="best", fontsize=6, ncol=2 if len(curves) > 6 else 1)

    if not any_data:
        plt.close(fig)
        return None

    axes[0].set_ylabel("CC_norm (CC / file mean)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _savefig_white(fig, out_path)
    plt.close(fig)
    return out_path
