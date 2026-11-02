# Spikelet meantrace (peak / shelf)

Standalone folder analysis — **not** wired into the main CC pipeline.

## What it does

For each `.abf` × direction (`ch0→ch2`, `ch2→ch0`):

1. Primary sweep = first with ≥4 APs (else 3 → 2). **1-AP sweeps are skipped** (not analyzed)
2. Align **AP2+ only** to active AP start
3. Average active/passive snippets (**meantrace**)
4. Measure:
   - **peak** spikelet amp (local peak, prominence 0.05 mV, need drop after top)
   - **plateau** amp = mean passive in **8–15 ms** − baseline
5. Auto **shape**:
   - `peak` if peaked event with enough after-peak drop
   - else `shelf` if step amp ≥ 0.05 mV
   - else `flat`
6. Chosen `amp_spikelet_mV` / `amp_ratio` by shape (`peak` or `shelf`).
7. **Two delays** (both exported when available):
   - `delay_onset_ms` = spikelet onset − AP onset  
     Spikelet onset = inflection (`d²V/dt²` sign change) before peak/anchor on meantrace  
     — same idea as AP `spike_begin` / `cp_infl_points`
   - `delay_peak_ms` = peak_passive − peak_active  
   `delay_ms` = primary (peak delay if shape=peak, else onset delay)

Also: mean **AP amp** on the same meantrace. For shelf amp, plateau start (~90% step + flatten) is still marked for QC.

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
| `*_spikelet_meantrace_over_time.png` | Over time: **12 (5 panels) above, 21 (5 panels) below** — amp spike, amp spikelet, ratio, delay_onset, delay_peak |

On spikelet-amp over-time panels, letter **P** / **S** marks peak vs shelf.

QC marks:
- green ▾ = AP start **used** in meantrace
- gray ▾ = AP start not used
- blue ○ = AP peak used
- purple ▲ = **spikelet onset** (d² inflection on meantrace)
- orange × / ○ = spikelet peak
- green ◆ = plateau start (shelf amp landmark)

## Tunable constants

In `spikelet_meantrace_folder.py`:

- `PEAK_PROMINENCE_MV` (default 0.05)
- `SHELF_DROP_FRAC` (default 0.25) — peak must drop this fraction after max
- `PLATEAU_START_MS` / `PLATEAU_END_MS` (default 8–15) — window for shelf **height**
- `SHELF_PLATEAU_START_FRAC` (default 0.90) — plateau start near 90% of shelf height
- `MIN_PLATEAU_AMP_MV` (default 0.05)
