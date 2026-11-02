# Spikelet meantrace (peak / shelf)

Standalone folder analysis — **not** wired into the main CC pipeline.

## What it does

For each `.abf` × direction (`ch0→ch2`, `ch2→ch0`):

1. Primary sweep = first with ≥4 APs (else 3 → 2 → 1)
2. Align AP2+ (AP1 only on 1-AP sweeps) to active AP start
3. Average active/passive snippets (**meantrace**)
4. Measure:
   - **peak** spikelet amp (local peak, prominence 0.05 mV, need drop after top)
   - **plateau** amp = mean passive in **8–15 ms** − baseline
5. Auto **shape**:
   - `peak` if peaked event with enough after-peak drop
   - else `shelf` if plateau amp ≥ 0.05 mV
   - else `flat`
6. Chosen `amp_spikelet_mV`, `amp_ratio`, `delay_ms`:
   - peak → delay = peak_passive − peak_active
   - shelf → delay = t50_passive − t50_active (using plateau amp)

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
| `*_12_meantrace.png` / `*_21_meantrace.png` | QC meantrace |
| `*_spikelet_meantrace_over_time.png` | amp spike, amp spikelet, ratio, delay vs time |

On the over-time spikelet-amp panel, letter **P** / **S** marks peak vs shelf.

## Tunable constants

In `spikelet_meantrace_folder.py`:

- `PEAK_PROMINENCE_MV` (default 0.05)
- `SHELF_DROP_FRAC` (default 0.25) — peak must drop this fraction after max
- `PLATEAU_START_MS` / `PLATEAU_END_MS` (default 8–15)
- `MIN_PLATEAU_AMP_MV` (default 0.05)
