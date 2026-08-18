# Coupling coefficient batch analysis — documentation

Logic lives in **`cc_calc_core.py`**. The notebook **`CC calculation (master project).ipynb`** imports it, picks folders, and exports Excel.

**Protocol:** Clampex `double_cciv`, 4 ADC channels — ch0/ch2 = mV (cells), ch1/ch3 = pA (stim current).

---

## Project layout

| File | Role |
|------|------|
| `cc_calc_core.py` | All analysis: CC, Rin, Gj, QC rules |
| `CC calculation (master project).ipynb` | UI, batch loop, optional QC plots |
| This README | Guide for future users |

---

## Notebook cells (run in order)

### Cell 0 — Imports and reload

Loads libraries and `cc_calc_core`.  
`importlib.reload(cc)` reloads the module after you edit thresholds in `.py` without restarting Jupyter.

### Cell 1 — Print CONFIG

Shows current thresholds (channels, spike detection, Rin windows, Gj formula).  
**Edit numbers in `cc_calc_core.py`**, not in this cell.

### Cell 2 — Folder and output path

Either:

- **Option A:** set `FOLDER_PATH` and optionally `OUTPUT_EXCEL` at the top of the cell (no dialog), or
- **Option B:** run the cell — **tkinter** dialogs pick the folder and `.xlsx` save path (same pattern as *Cell Properties for one folder*).

Sets `folder_path` and `output_excel`. Default output name: `CC_results.xlsx` in the chosen folder if you cancel the save dialog.

### Cell 3 — Batch analysis

For each `.abf`:

1. Calls `analyze_abf_file(filepath)`
2. Appends rows to one list
3. Writes two sheets to Excel: **`All_data`** (per sweep) and **`File_summary`** (one row per file)

Errors on one file do not stop the rest.

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
Used for CC on both voltage and current channels.

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

Returns `(value, skip_reason)`.

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
4. Select CC sweeps (both directions)
5. Run `coupling_block` for ch0→ch2 and ch2→ch0
6. Compute Rin for ch0 and ch2
7. Mean CC and file-level Gj per direction
8. Build `file_fields` (Rin, Gj, means, sweep counts — repeated on every sweep row)
9. One row per sweep per direction; per-sweep `Gj_sweep_nS`
10. If no CC sweeps → one row with explanatory skip message

Returns a **list of dicts** → flattened into Excel sheet **`All_data`**.

---

## Excel sheets

### `All_data` — per sweep only

One row per **sweep × direction**. Columns: `file`, `recording_datetime`, `sweep`, `direction`, `cur_step_pA`, `delta_V_*`, `CC`, `CC_skip_reason`, `Gj_sweep_nS`, `Gj_sweep_skip_reason`.

File-level metrics (Rin, Gj means, cell properties, tau/Cm) are **not** repeated here — see `File_summary`.

### `File_summary` — one row per file

All metrics computed once per recording: CC/Gj means, Rin, cell properties, tau/Cm, etc.

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
| `V_pre_mV_*`, `V_post_mV_*`, `delta_V_mV_*` | Mean Vm in pre/post CC epochs on tau sweep |
| `tau_sweep_*`, `tau_skip_reason_*`, `Cm_skip_reason_*` | Sweep used; skip reasons |

**CC Rin** stays in `Rin1_MOhm` / `Rin2_MOhm` (and `Rin_ch0_MOhm` on `All_data`) with the existing linear-fit exceptions.

---

## Excel sheet `All_data`

One table for all files. Each row is roughly **one sweep × direction**, with file-level columns repeated on every row.

| Column group | Meaning |
|--------------|---------|
| `file`, `recording_datetime` | File identity and ABF header time |
| `sweep`, `direction` | Sweep index; `ch0->ch2` or `ch2->ch0` |
| `cur_step_pA`, `delta_V_*_mV`, `CC`, `CC_skip_reason` | Per-sweep coupling |
| `Rin_ch0_MOhm`, `R2_ch0`, `n_IV_ch0`, `Rin_ch0_skip_reason`, `Rin_vm_range_ch0` | File-level Rin cell 0 |
| Same for `ch2` | File-level Rin cell 2 |
| `CC_mean_*`, `n_CC_avg_*`, `Gj_*_nS`, `Gj_*_skip_reason` | File-level mean CC (valid sweeps only), Gj |
| `Rin_ch0_note`, `Rin_ch2_note` | Single-point Rin comment when applicable |
| `Gj_sweep_nS`, `Gj_sweep_skip_reason` | Gj from that sweep’s CC |
| `n_CC_sweeps_ch0to2`, `n_CC_sweeps_ch2to0` | Count of sweeps used for CC |

---

## Cell properties (per file, both cells)

Same logic as **Cell Properties for one folder.ipynb**, computed for **ch0** and **ch2**:

- Sweep with **≥4 APs** in the stim window (from `sweepC` epoch detection)
- AP2/1 ratio, initial / late / mean firing frequency, FWHM (1st spike), delay to 1st AP
- **Rin_abs** (linear V–I, Vm −85…−50 mV), **Rin_rel** (mean ΔV/ΔI), **V_rest**
- **hold_V_mV** (pre-stim Vm), **inj_current_pA** (on the ≥4 AP sweep)

These are **in addition to** CC Rin used for Gj (`Rin_ch0_MOhm` / `Rin_ch2_MOhm`). No inline plots in batch mode.

### QC plots (saved PNGs)

When `SAVE_QC_PLOTS = True`, PNGs go to **`<data_folder>/QC_plots/`** with an opaque **white** background:

| Filename pattern | Content |
|------------------|---------|
| `{file}_ch0_AP.png` | ≥4 AP sweep QC |
| `{file}_ch0_IV_Rin.png` | Rin I–V (CC / Gj / Cm) |
| `{file}_ch0_tau_Cm.png` | tau/Cm: V_pre, V_post, 63%, tau from pre_end |
| Same for `ch2` | |

Set `SAVE_QC_PLOTS = False` in `cc_calc_core.py` to disable. Optional CC inspect plot remains in notebook cell 4.

### Tau / Cm

**Rin = same as CC/Gj** (`rin_for_channel`).

- **V_pre** = mean Vm in pre epoch (`start10:end10` ch0, `start20:end20` ch2)
- **V_post** = mean Vm in post epoch (`start11:end11` / `start21:end21`)
- Select sweep where **ΔV = V_post − V_pre** is in **−35…−10 mV** (closest to zero)
- Require **min Vm in post epoch ≥ −90 mV**
- **v_63** = V_pre + 0.63 × (V_post − V_pre)
- **tau** = time from **stim_start** (post epoch onset) to first Vm ≤ v_63
- **Cm [pF]** = tau_ms / Rin_MΩ × 1000

Config in `cc_calc_core.py`: `TCM_VMIN_LIMIT`, `TCM_DV_SEARCH_MIN/MAX`.

QC plot: `{file}_ch0_tau_Cm.png` — trace, pre/post epochs, V_pre, V_post, 63% level, tau point.

---

| Analysis | Which sweeps | Criterion |
|----------|--------------|-----------|
| **CC** | Prefix from sweep 0 until first “bad” sweep | No AP on active (post-stim) or passive (full ΔV window); then per-sweep ΔI and passive spike checks |
| **Rin** | All sweeps in file | Mean Vm in stim window inside mV bands; linear V–I fit |

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
