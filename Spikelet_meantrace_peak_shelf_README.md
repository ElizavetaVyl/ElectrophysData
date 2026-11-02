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

Or notebook: `Spikelet meantrace peak-shelf.ipynb` (cells 0 → 1).

## Outputs

Folder `Spikelet_meantrace_peak_shelf/`:

| File | Content |
|------|---------|
| `*_spikelet_meantrace.xlsx` | `All_directions` + `Summary_by_file` |
| `*_spikelet_QC.png` | Per file: **12 above, 21 below**. Each block = main sweep track + meantrace with onset / peak / plateau marks |
| `*_spikelet_meantrace_over_time.png` | Over time: **5 panels** (amp spike, amp spikelet, ratio, delay_onset, delay_peak); **12 & 21 overlaid** (different colors); file-number labels; negatives as red × |

Negative amp/ratio/delay values are cleared from Excel and marked red × on the over-time plot.

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
- `PEAK_PROMINENCE_MV` (default 0.01)
- `SHELF_DROP_FRAC` (default 0.02) — peak only if drop after max ≥ this; else shelf
- `PLATEAU_START_MS` / `PLATEAU_END_MS` (default 8–15) — window for shelf **height**
- `SPIKELET_ONSET_FRAC` (default 0.15) — onset / primary delay from AP start
- `SHELF_PLATEAU_START_FRAC` (default 0.90) — late plateau QC landmark
- `MIN_PLATEAU_AMP_MV` (default 0.05)
