# Coupling coefficient batch analysis — documentation

Logic lives in **`cc_calc_core.py`**. The notebook **`CC calculation.ipynb`** imports it, picks folders, and exports Excel.

**Protocol:** Clampex `double_cciv`, 4 ADC channels — ch0/ch2 = mV (cells), ch1/ch3 = pA (stim current).

---

## Project layout

| File | Role |
|------|------|
| `cc_calc_core.py` | All analysis: CC, Rin, Gj, QC rules |
| `CC calculation.ipynb` | UI, batch loop, optional QC plots |
| This README | Guide for future users |

---

## Notebook cells (run in order)

### Cell 0 — Imports and reload

Loads libraries and `cc_calc_core`.  
`importlib.reload(cc)` reloads the module after you edit thresholds in `.py` without restarting Jupyter.

### Cell 1 — Print CONFIG

Shows current thresholds (channels, spike detection, Rin windows, Gj formula).  
**Edit numbers in `cc_calc_core.py`**, not in this cell.

### Cell 2 — Folder, output path, and plot folders

Either:

- **Option A:** set `FOLDER_PATH`, `OUTPUT_EXCEL`, and optional plot parent paths at the top of the cell (no dialog), or
- **Option B:** run the cell — **tkinter** dialogs pick the folder and `.xlsx` save path.

Sets `folder_path`, `output_excel`, and (when enabled) plot folders next to the ABFs:

- `Cell_properties_plots/`
- `CC_plots/`
- `Spikelet_plots/`

Default output name: `{folder_name}_CC_data.xlsx` if you cancel the save dialog.

### Cell 3 — Analysis blocks + batch

Opens an **Analysis blocks** window (it can sit behind Jupyter). Choose which blocks to run:

| Block | What it computes |
|-------|------------------|
| `cc` | CC, CC_norm, Gj, Rin, CC plots using **all** sweeps before the first sweep with at least one spike |
| `cc_last_pos` | CC, CC_norm, Gj, Rin, CC plots using the **last positive** sweep before the first sweep with at least one spike |
| `cc_most_neg` | CC, CC_norm, Gj, Rin, CC plots using the **most negative** sweep before the first sweep with at least one spike |
| `cc_neg_pos` | CC, CC_norm, Gj, Rin, CC plots using the **most negative** sweep and the **most positive** sweep before the first sweep with at least one spike |
| `cell_props` | AP21 ratio, firing, FWHM, V_rest, Cell Properties Rin |
| `tau_cm` | tau, Cm |
| `spikelets` | spike / spikelet metrics, QC PNGs, folder spikelet plots |

Each CC block saves its results into a separate output folder under the ABF folder:

- `CC_multi_sweeps/`
- `CC_last_positive_pre_spike/`
- `CC_most_negative_pre_spike/`
- `CC_most_negative_most_positive_pre_spike/`

You can select **several CC blocks in one run**. The notebook then loops all selected modes and writes **Excel + CC folder plots** into each mode folder so you can compare graphs side by side. Cell properties / tau / spikelets still run once on the first selected CC mode.

For each `.abf`:

1. Calls `analyze_abf_file(...)`
2. Appends rows to lists for Excel and folder plots
3. Writes **six** sheets: `Summary_short`, `File_summary`, `All_data`, `CC_neg_pos`, `Spikelets`, `Spikelet_sweeps`

Errors on one file do not stop the rest.

Preset blocks without the window: set `ANALYSIS_BLOCKS = {...}` in cell 2.

### Cell 4 — Visual QC (optional)

Set `INSPECT_FILE` to one recording path.  
Plots ch0 and ch2, epoch boundary lines, detected peaks, and which sweeps pass CC selection (blue = included, red = excluded).

---

## `cc_calc_core.py` — block by block

### CONFIG (top of file)

Tunable constants:

| Constant | Purpose |
|----------|---------|
| `EXPECTED_CHANNELS` | Must be 4 |
| `CC_SPIKE_HEIGHT` | mV; `find_peaks` height threshold for APs |
| `CC_SPIKE_DISTANCE` | samples; minimum distance between peaks |
| `CC_MIN_DELTA_I_PA` | pA; skip CC if \|ΔI\| on stim channel is smaller |
| `CC_SMOOTH_MS` | ms; Gaussian σ for **CC only** (default **7.0**). Does not affect spikelets |
| `SPIKELET_PEAK_SMOOTH_MS` | ms; spikelet peak search only (default **0.3**) |
| `RIN_VMIN`, `RIN_VMAX` | Primary Vm window for I–V points (−80…−50 mV) |
| `RIN_VMIN_FALLBACK`, `RIN_VMAX_FALLBACK` | Wider window if too few points (−95…−50 mV) |
| `RIN_MIN_POINTS` | Minimum sweeps for Rin linear fit (≥ 2) |

---

### `time_period_borders(data_rate)`

Returns **sample indices** (not seconds) for prestim/post-stim epochs on ch0 and ch2, plus `Rtime_ch0` / `Rtime_ch2` (one sample index for current used in Rin I–V).

Separate tables for **10 kHz** and **20 kHz**; other rates raise an error.

---

### `recording_datetime_str(abf)`

Reads recording time from the ABF header (`abfDateTime`) for Excel column `recording_datetime`.

---

### `mean_delta_voltage(...)`

**ΔV = mean(prestim) − mean(poststim)** over two index ranges.  
Used for CC on both voltage and current channels. When `CC_SMOOTH_MS > 0`, each trace is lightly Gaussian-smoothed before the means (spike detection still uses raw traces).

---

### `n_spikes_in_window(...)`

Counts peaks in a Vm trace segment using `scipy.signal.find_peaks` (height and distance from CONFIG).

---

### `cc_sweep_indices_direction(...)`

Chooses **which sweeps are used for CC**:

- Starts at sweep 0, increments sweep by sweep
- **Stops** at the first sweep where:
  - **active** cell has ≥1 peak in the **post-stim** window, or
  - **passive** cell has ≥1 peak in the **full ΔV window** (`pre_start` → `post_end`)

Replaces manual “number of sweeps without spikes”.

---

### `cc_sweep_indices(...)` (legacy)

Older helper: active channel only, post-stim window. Kept for compatibility; batch analysis uses `cc_sweep_indices_direction`.

---

### `cc_skip_reason(...)`

Returns **why CC is not calculated** for one sweep (`None` if OK):

1. Spike on passive cell in ΔV window (with sample index ranges for prestim/post-stim)
2. \|ΔI\| below `CC_MIN_DELTA_I_PA` on stim channel
3. ΔV on active cell = 0

Stored in `CC_skip_reason`; `CC` is empty when skipped.

---

### `coupling_block(...)`

For each selected sweep and direction:

- Computes **ΔV_active**, **ΔV_passive**, **ΔI** (stim current channel)
- **CC = ΔV_passive / ΔV_active** if no skip reason
- Returns one dict per sweep

**Directions:**

| Direction | Active (denominator) | Passive (numerator) | Current ch | Epoch window |
|-----------|----------------------|---------------------|------------|--------------|
| ch0→ch2 | ch0 | ch2 | ch1 | `w0` (ch0 epoch) |
| ch2→ch0 | ch2 | ch0 | ch3 | `w2` (ch2 epoch) |

---

### `cc_normalize_block(...)`

After mean CC is computed **per file and per direction**:

\[
CC\_norm = \frac{CC_{sweep}}{mean(CC\ \text{of this file in this direction})}
\]

- stored on each sweep row in `All_data`
- used for folder plots **CC_norm vs Vm** and CC slope-over-time
- if mean CC is 0 or missing → `CC_norm` stays empty

---

### `collect_iv_points(...)`

Builds one **(I, V)** point per sweep for **Rin**:

- **V** = mean Vm in stim epoch (`v_start`–`v_end`)
- **I** = current at `i_index` (`Rtime_ch0` or `Rtime_ch2`)
- Only sweeps with mean Vm in `(vmin, vmax)`

Uses **all sweeps in the file**, not only CC-selected sweeps.

---

### `rin_r2(currents, voltages)`

**Rin = slope** of linear fit **V vs I** (`np.polyfit`, degree 1) → **MΩ**.  
Also returns **R²** (goodness of fit). Needs ≥ `RIN_MIN_POINTS` pairs.

---

### `rin_for_channel(...)`

Rin for one cell (voltage + current channel):

1. Try Vm window **−80…−50 mV** (linear V–I fit, ≥2 points)
2. If too few points or fit fails → try **−95…−45 mV** (linear fit)
3. If still &lt;2 points but **exactly 1** sweep with mean Vm in **−95…−50 mV** → **Rin = ΔV/ΔI × 1000** MΩ

**Linear fit units:** polyfit gives slope **dV/dI in mV/pA**; multiply by **1000** to get **MΩ** (same factor as ΔV/ΔI). Already applied in `rin_r2`.

Skip messages describe **too few I–V points** or **fit failed**, including stim-window sample indices.

---

### `gj_nS(cc, rin_passive_MOhm)`

Gap junction conductance in **nanosiemens**:

```
Gj [nS] = (CC / (1 − CC)) × (1000 / Rin_passive [MΩ])
```

- **ch0→ch2:** uses **Rin_ch2** (passive cell in that direction)
- **ch2→ch0:** uses **Rin_ch0**

Returns `(value, skip_reason)`. Requires **0 < CC < 1** and **Rin_passive > 0** (negative Rin from a bad I–V fit would otherwise yield unphysical negative Gj).

---

### `mean_valid_cc(cc_list)`

Mean of non-null CC values per direction → file-level `CC_mean_ch0to2`, `CC_mean_ch2to0`.

---

### `_skipped_file_row(...)`

One Excel row when the **whole file** is skipped (e.g. wrong channel count), with reasons filled in.

---

### `analyze_abf_file(filepath)` — main pipeline per file

1. Open ABF, read datetime
2. Check 4 channels
3. Get epoch sample indices
4. Run selected analysis blocks (`cc`, `cc_last_pos`, `cc_most_neg`, `cc_neg_pos`, `cell_props`, `tau_cm`, `spikelets`)
5. For CC: choose one sweep-selection mode, then run both directions and compute `CC_norm`, Rin, mean CC, Gj
6. For cell properties: compute ch0/ch2 fields on the ≥4 AP sweep
7. For tau/Cm: pick best sweep, export tau/Cm metadata
8. For spikelets: per-AP rows, per-sweep rows, file-level spikelet summary, QC PNGs
9. Build one `File_summary` row and per-sweep rows for `All_data`

Returns:

- per-sweep CC rows → **`All_data`**
- one file summary row → **`File_summary`**
- spikelet AP rows → **`Spikelets`**
- spikelet sweep rows → **`Spikelet_sweeps`**

---

## Excel sheets

The batch export writes **five** sheets.

### `Summary_short` — compact overview

Short user-facing cut from `File_summary` plus selected spikelet sweeps.

Includes:

- file / recording time / `analysis_blocks`
- main CC, Gj, Rin, cell properties, tau/Cm
- `props_sweep_*`, `inj_current_pA_*`
- `tau_sweep_*`, `delta_V_mV_*`, `V_post_min_mV_*`
- folder `mode_spikelet_baseline_mV`
- two **meantrace** spikelet column groups per direction:
  - `primary_*` = primary / prominent sweep (≥4 AP, then fallback rules)
  - `near_mode_vm_*` = sweep nearest the folder mode baseline

Column fill colors in Excel:

- `primary_*` = one color
- `near_mode_vm_*` = another color

Use this sheet for quick reading. Full detail stays in the other sheets.

### `File_summary` — one row per file

All file-level metrics computed once per recording:

- CC12/21, Gj12/21, Rin
- cell properties for ch0/ch2
- tau/Cm for ch0/ch2
- spikelet summary fields `spikelet_*_12` and `spikelet_*_21`
- skip reasons and notes

### `All_data` — per sweep × direction only

One row per **sweep × direction**. Per-sweep CC/Gj columns only:

- `file`, `recording_datetime`, `sweep`, `direction`
- `cur_step_pA`, `delta_V_active_mV`, `delta_V_passive_mV`, `Vm_active_stim_mV`
- `CC`, **`CC_norm`**, `CC_skip_reason`
- `Gj_sweep_nS`, `Gj_sweep_skip_reason`

File-level metrics (Rin, cell properties, tau/Cm, spikelet summary) are **not** repeated here — see `File_summary`.

### `CC_neg_pos` — most negative / most positive pre-spike sweeps

One row per **file × direction × sweep role** (up to 4 rows per file). Always computed for every successfully opened ABF, independent of which CC mode checkbox is selected for the main run.

| Column | Meaning |
|--------|---------|
| `direction` | `ch0->ch2` or `ch2->ch0` |
| `sweep_role` | `most_negative` or `most_positive` (pre-spike subthreshold sweep by injected current step) |
| `sweep` | Sweep index used |
| `cur_step_pA` | ΔI in the stim current channel (smoothed, same as CC) |
| `delta_V_active_mV`, `delta_V_passive_mV` | Pre−post mean Vm on active and passive cells (**CC_SMOOTH_MS** Gaussian smooth) |
| `Vm_active_stim_mV` | Mean active Vm in post-stim window |
| `CC` | `delta_V_passive / delta_V_active` |
| `Gj_nS` | From this sweep’s CC and passive-cell Rin |
| `CC_smooth_ms` | Smoothing σ used for delta-V (currently `CC_SMOOTH_MS`) |

### `Spikelets` vs `Spikelet_sweeps`

| | **`Spikelets`** | **`Spikelet_sweeps`** |
|---|-----------------|----------------------|
| Granularity | **1 row = 1 AP** | **1 row = 1 sweep + direction** |
| Detail level | highest | aggregated per sweep |
| Typical use | inspect individual APs | compare sweeps, primary flag, sweep means |
| Contains | peak times, amp spike/spikelet, ratio, delays, baseline, flags | n_AP, n_detected, sweep means, avg/meantrace metrics, `is_primary` |

**`Spikelets`** is the most detailed spikelet sheet (AP level).

**`Spikelet_sweeps`** is the best sheet for sweep-level spikelet results, including meantrace metrics and which sweep was primary.

---

## Units (why ×1000 appears)

All inputs from ABF: **Vm in mV**, **I in pA**, **sample rate in Hz**.

| Quantity | Formula | ×1000 meaning |
|----------|---------|---------------|
| **Rin [MΩ]** | slope(V vs I) or ΔV/ΔI, then **×1000** | mV/pA → MΩ (1 mV/pA = 1000 MΩ) |
| **tau [ms]** | (samples / Hz) **×1000** | seconds → milliseconds |
| **FWHM [ms]** | (width in samples / Hz) **×1000** | seconds → milliseconds (same as tau; **not** related to Rin) |
| **Cm [pF]** | tau_ms / Rin_MΩ **×1000** | from C = τ/R: ms and MΩ → pF |
| **Gj [nS]** | CC/(1−CC) **×1000** / Rin_MΩ | nS = 1000 / MΩ |

No unit error between tau and FWHM — both only convert samples→ms. Cm uses a **different** ×1000 (ms·MΩ → pF). Rin uses ×1000 (mV/pA → MΩ).

---

One row per `.abf` file. Mean **CC** and **Gj** use only sweeps with a valid CC (not skipped).

| Column | Meaning |
|--------|---------|
| `file`, `recording_datetime` | File identity |
| `Rin1_MOhm`, `Rin2_MOhm` | Input resistance ch0 and ch2 |
| `Rin1_note`, `Rin2_note` | e.g. single-point Rin comment |
| `CC12`, `CC21` | Mean CC ch0→ch2 and ch2→ch0 |
| `Gj12_nS`, `Gj21_nS` | Gj from mean CC and passive-cell Rin |
| `n_CC_avg_12`, `n_CC_avg_21` | Sweeps included in CC/Gj mean |
| `n_CC_selected_12`, `n_CC_selected_21` | Subthreshold sweeps selected before per-sweep skips |
| `AP21_ratio_ch0/ch2` | 2nd / 1st AP amplitude on ≥4 AP sweep |
| `init_freq_Hz_*`, `late_freq_Hz_*`, `mean_freq_Hz_*` | Firing frequencies (Cell Properties logic) |
| `FWHM_ms_*`, `delay_AP1_ms_*` | First spike width and latency from stim start |
| `Rin_abs_MOhm_*`, `Rin_rel_MOhm_*` | Cell Properties Rin (V–I slope / mean ΔV/ΔI); **not** CC/Gj Rin |
| `V_rest_mV_*` | V intercept at I=0 from abs Rin fit |
| `hold_V_mV_*`, `inj_current_pA_*` | Hold potential and injection on ≥4 AP sweep |
| `R2_abs_Rin_*`, `props_sweep_*`, `props_skip_reason_*` | Fit quality, analysis sweep, skip reason |
| `tau_ms_*`, `Cm_pF_*` | Membrane time constant and capacitance (Tau Cm notebook) |
| `V_pre_mV_*`, `V_post_mV_*`, `delta_V_mV_*`, `V_post_min_mV_*` | Mean Vm in pre/post epochs on tau sweep; minimum Vm in post epoch used for sweep selection |
| `tau_sweep_*`, `tau_selection_note_*`, `tau_skip_reason_*`, `Cm_skip_reason_*` | Sweep used; selection note; skip reasons |

**CC Rin** stays in `Rin1_MOhm` / `Rin2_MOhm` on `File_summary`.

---

## Cell properties (per file, both cells)

Same logic as **Cell Properties for one folder.ipynb**, computed for **ch0** and **ch2**:

- Sweep with **≥4 APs** in the stim window (from `sweepC` epoch detection)
- AP2/1 ratio, initial / late / mean firing frequency, FWHM (1st spike), delay to 1st AP
- **Rin_abs** (linear V–I, Vm −85…−50 mV), **Rin_rel** (mean ΔV/ΔI), **V_rest**
- **hold_V_mV** (pre-stim Vm), **inj_current_pA** (on the ≥4 AP sweep)

These are **in addition to** CC Rin used for Gj (`Rin_ch0_MOhm` / `Rin_ch2_MOhm`). No inline plots in batch mode.

### QC plots (saved PNGs)

When `SAVE_QC_PLOTS = True`, cell-property QC PNGs go to **`<data_folder>/Cell_properties_plots/`** with an opaque **white** background:

| Filename pattern | Content |
|------------------|---------|
| `{file}_ch0_AP.png` | ≥4 AP sweep QC |
| `{file}_ch0_IV_Rin.png` | Rin I–V (CC / Gj / Cm) |
| `{file}_ch0_tau_Cm.png` | tau/Cm: V_pre, V_post, 63%, tau from pre_end |
| Same for `ch2` | |

Set `SAVE_QC_PLOTS = False` in `cc_calc_core.py` to disable.  
CC plots go to **`CC_plots/`**. Spikelet QC goes to **`Spikelet_plots/`**. Optional CC inspect plot remains in notebook cell 4.

### Tau / Cm

**Rin = same as CC/Gj** (`rin_for_channel`).

- **V_pre** = mean Vm in pre epoch (`start10:end10` ch0, `start20:end20` ch2)
- **V_post** = mean Vm in post epoch (`start11:end11` / `start21:end21`)
- **V_post_min** = minimum Vm in the post epoch on the selected sweep (`V_post_min_mV_*`)
- Select sweep where **ΔV = V_post − V_pre** is in **−35…−10 mV** (closest to zero)
- Prefer sweeps with **min Vm in post epoch ≥ −90 mV**
- **v_63** = V_pre + 0.63 × (V_post − V_pre)
- **tau** = time from **stim_start** (post epoch onset) to first Vm ≤ v_63
- **Cm [pF]** = tau_ms / Rin_MΩ × 1000

Saved metadata per channel:

- `tau_sweep_*` — which sweep was used
- `delta_V_mV_*` — ΔV on that sweep
- `V_post_min_mV_*` — minimum Vm in post epoch
- `inj_current_pA_*` is **not** exported for tau/Cm (that field belongs to cell properties)

Config in `cc_calc_core.py`: `TCM_VMIN_LIMIT`, `TCM_DV_SEARCH_MIN/MAX`.

QC plot: `{file}_ch0_tau_Cm.png` — trace, pre/post epochs, V_pre, V_post, 63% level, tau point.

---

## Spikelet analysis

Directions:

- **12** = ch0 → ch2 (active ch0, passive ch2)
- **21** = ch2 → ch0

### Primary sweep selection

1. first sweep with **≥ 4 APs**
2. else first with **3 APs**
3. else first with **2 APs**
4. else **1-AP fallback** only if there is no multi-spike sweep

This primary sweep is used for QC PNGs and the main file-level spikelet summary.

### Nearest mode Vm sweep

Additionally, the code picks the sweep whose spikelet baseline (passive mean 1 ms before AP start) is closest to the folder **mode baseline Vm** (KDE peak of all sweep-mean baselines). Used in extra folder plots and in `Summary_short` columns `near_mode_vm_*`.

### Scheme A — original per-AP spikelets

For each analyzed AP (normally AP2+, except 1-AP fallback):

- baseline = passive mean in **1 ms before t = 0**
- search window = **15 ms** after AP start
- Gaussian smooth σ = **0.3 ms**
- local peak with prominence **≥ 0.15 mV**
- noise gate **OFF**
- if `amp_spikelet / amp_spike > 1` → flagged as error; kept in Excel but omitted from ratio plots
- negative delays are kept and marked

Results: sheet **`Spikelets`** (AP level) and aggregated fields on **`Spikelet_sweeps`**.

### Scheme B — meantrace (averaged signal)

Parallel scheme on aligned mean traces (AP2+):

- window = **15 ms** after AP start
- smooth σ = **0.3 ms**
- local peak with prominence **≥ 0.05 mV**
- noise gate **OFF**
- spikelet amplitude taken from the **smoothed mean passive trace**
- delay = passive peak time − active mean AP peak time
- ratio = amp_spikelet / amp_spike

These values appear on **`Spikelet_sweeps`** and in file-level fields `spikelet_meantrace10_*` on **`File_summary`**.

### Spikelet QC PNG (`Spikelet_plots/`)

Primary sweep only: `{stem}_12_spikelets.png`, `{stem}_21_spikelets.png`

Three panels:

1. raw active + passive overlay
2. aligned AP mean traces (original 15 ms scheme)
3. meantrace panel: thin aligned traces, thick mean passive, thin smoothed passive in another color, active mean, peak markers

Folder plots (in `Spikelet_plots/`, not among ABF files):

- `*_spikelet_over_time.png`
- `*_spikelet_over_time_near_<modeVm>mV.png`
- `*_spikelet_meantrace10_over_time*.png`
- `*_spikelet_vs_Vm.png`
- `*_spikelet_meantrace10_vs_Vm.png`

Current spikelet constants in `cc_calc_core.py`:

| Constant | Value |
|----------|-------|
| `SPIKELET_PEAK_MS` | 15 ms |
| `SPIKELET_PEAK_SMOOTH_MS` | 0.3 ms |
| `SPIKELET_MIN_PROMINENCE_MV` | 0.15 mV (per-AP) |
| `SPIKELET_MEANTRACE_PEAK_MS` | 15 ms |
| `SPIKELET_MEANTRACE_MIN_PROMINENCE_MV` | 0.05 mV |
| `SPIKELET_USE_NOISE_GATE` | False |

---

| Analysis | Which sweeps | Criterion |
|----------|--------------|-----------|
| **CC** | Prefix from sweep 0 until first “bad” sweep | No AP on active (post-stim) or passive (full ΔV window); then per-sweep ΔI and passive spike checks |
| **Rin** | All sweeps in file | Mean Vm in stim window inside mV bands; linear V–I fit |
| **Spikelets** | Sweeps with ≥2 APs from first such sweep onward; primary = rules above | per-AP local peak + meantrace on aligned mean |

---

## Typical workflow

1. Adjust CONFIG in `cc_calc_core.py`
2. Run notebook cells **0 → 3**
3. Use cell **4** on questionable files
4. Read `*_skip_reason` columns in Excel for QC

## Dependencies

```
pip install pyabf pandas matplotlib scipy openpyxl
```

`openpyxl` is required for `.xlsx` export.
