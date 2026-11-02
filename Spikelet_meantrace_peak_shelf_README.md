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
6. Chosen `amp_spikelet_mV`, `amp_ratio`, `delay_ms`:
   - peak → delay = peak_passive − peak_active
   - shelf → amp = step height (late window − baseline);  
     **onset** = first time passive reaches `SHELF_ONSET_FRAC` (10%) of step above baseline;  
     delay = **shelf_onset − peak_active** (not mid-plateau)

Also: mean **AP amp** on the same meantrace.

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
| `*_spikelet_QC.png` | Per file: **12 above, 21 below**. Each block = main sweep track (used APs + detected peaks) + meantrace with marked peak/plateau |
| `*_spikelet_meantrace_over_time.png` | Over time: **12 (4 panels) above, 21 (4 panels) below** — amp spike, amp spikelet, ratio, delay |

On spikelet-amp over-time panels, letter **P** / **S** marks peak vs shelf.

QC main track marks:
- green ▾ = AP start **used** in meantrace
- gray ▾ = AP start not used
- blue ○ = AP peak used
- orange × = detected spikelet peak (shape=peak)
- green ◆ = **shelf onset** (shape=shelf; start of the step, not mid-plateau)

## Tunable constants

In `spikelet_meantrace_folder.py`:

- `PEAK_PROMINENCE_MV` (default 0.05)
- `SHELF_DROP_FRAC` (default 0.25) — peak must drop this fraction after max
- `PLATEAU_START_MS` / `PLATEAU_END_MS` (default 8–15) — window for shelf **height**
- `SHELF_ONSET_FRAC` (default 0.10) — onset at 10% of shelf height
- `MIN_PLATEAU_AMP_MV` (default 0.05)
