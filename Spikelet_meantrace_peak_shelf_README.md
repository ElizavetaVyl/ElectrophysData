# Spikelet meantrace (peak / shelf)

Standalone folder analysis — **not** wired into the main CC pipeline.

## What it does

For each `.abf` × direction (`ch0→ch2`, `ch2→ch0`):

1. Primary sweep = first with ≥4 APs (else 3 → 2). **1-AP sweeps are skipped** (not analyzed)
2. Align **AP2+ only** to active AP start
3. Average active/passive snippets (**meantrace**)
4. Measure:
   - **peak** spikelet amp (local peak, prominence 0.025 mV, need drop after top)
   - **plateau** amp = mean passive in **8–15 ms** − baseline
5. Auto **shape** (mutually exclusive — one only):
   - `peak` if local max with enough after-peak drop **and** late plateau is clearly below the peak
   - else `shelf` if step amp ≥ 0.05 mV (peak columns cleared)
   - else `flat`
6. Chosen `amp_spikelet_mV` / `amp_ratio` by shape (`peak` or `shelf`).
7. **Two delays** (both exported when available):
   - `delay_onset_ms` (**primary**) = time from **AP start** to first sample where  
     passive reaches **15% of spikelet amplitude** (foot of the rise / shelf onset)
   - `delay_peak_ms` = peak_passive − peak_active (kept for comparison)  
   `delay_ms` = `delay_onset_ms` when available

Also: mean **AP amp** on the same meantrace. Late plateau landmark (~90% step + flatten) is still marked for QC on shelf.

## Run

```bash
python spikelet_meantrace_folder.py
# or
python spikelet_meantrace_folder.py /path/to/abf/folder
```

A **clickable checklist** asks which `.abf` files are the **baseline period**.
Normalization (per direction): `value_norm = value / mean(baseline files)`.

Or notebook: `Spikelet meantrace peak-shelf.ipynb` — lists files, then opens the same picker.
You can also pass `baseline_files=[...]` / set `BASELINE_FILES` in the notebook.

## Outputs

Folder `Spikelet_meantrace_peak_shelf/`:

| File | Content |
|------|---------|
| `*_spikelet_meantrace.xlsx` | **Short**; **Baseline_means**; All_directions; Summary_by_file; **Meantraces** (time from AP onset=0, Vm spike & spikelet) |
| `*_spikelet_QC.png` | Per file: **12 above, 21 below** — main track + meantrace |
| `*_spikelet_meantrace_over_time.png` | Raw over time; **baseline period shaded**; 12 & 21 overlay |
| `*_spikelet_meantrace_over_time_norm.png` | Normalized over time (baseline = 1; baseline period shaded) |

`t_rel_min`: last baseline file = **0**; each file before **−3 min**, each after **+3 min** (by file order).

QC marks:
- green ▾ = AP start **used** in meantrace
- gray ▾ = AP start not used
- blue ○ = AP peak used
- purple ▲ = **onset 15% amp** (from AP start; primary delay)
- orange × / ○ = spikelet peak (peak-to-peak delay)
- green ◆ = late plateau start (~90%; QC landmark)

## Tunable constants

In `spikelet_meantrace_folder.py`:

- `SMOOTH_MS` (default 0.5)
- `PEAK_PROMINENCE_MV` (default 0.025)
- `PEAK_SEARCH_AFTER_AP_MS` (default 7.0) — spikelet peak only in this window after AP peak
- `SHELF_DROP_FRAC` (default 0.05) — peak only if drop after max ≥ this; else shelf
- `PLATEAU_START_AFTER_AP_PEAK_MS` (default 8) — shelf amp window starts 8 ms after AP peak
- `PLATEAU_DUR_MS` (default 7) — shelf amp window length
- `SPIKELET_ONSET_FRAC` (default 0.15) — onset / primary delay from AP start
- `SHELF_PLATEAU_START_FRAC` (default 0.90) — late plateau QC landmark
- `MIN_PLATEAU_AMP_MV` (default 0.05)
