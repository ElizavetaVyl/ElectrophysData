# CC / spikelet batch analysis: what is calculated and where it is saved

This document describes the current behavior of:

- `cc_calc_core.py`
- `CC calculation (master project).ipynb`

It is intended as a short reference for what the code computes, which sweeps are used, and what each Excel sheet contains.

## General workflow

For each ABF file, the batch notebook can run 4 blocks:

- `cc` - CC / Gj / Rin
- `cell_props` - cell properties
- `tau_cm` - tau / Cm
- `spikelets` - spike / spikelet analysis

The selected blocks are recorded in Excel in `analysis_blocks`.

Plots are saved next to the ABFs:

- `Cell_properties_plots/`
- `CC_plots/`
- `Spikelet_plots/`

## Cell properties

Calculated separately for `ch0` and `ch2`.

### Main outputs

- `AP21_ratio`
- `init_freq_Hz`
- `late_freq_Hz`
- `mean_freq_Hz`
- `FWHM_ms`
- `delay_AP1_ms`
- `Rin_abs_MOhm`
- `Rin_rel_MOhm`
- `V_rest_mV`
- `hold_V_mV`

### Which sweep is used

Cell properties use the first sweep with `>= 4` APs in the stimulus window.

Saved fields:

- `props_sweep_ch0`, `props_sweep_ch2`
- `inj_current_pA_ch0`, `inj_current_pA_ch2`
- `props_skip_reason_ch0`, `props_skip_reason_ch2`

## Tau / Cm

Calculated separately for `ch0` and `ch2`.

### Main outputs

- `tau_ms`
- `Cm_pF`
- `delta_V_mV`
- `V_post_min_mV`

### Which sweep is used

The code selects the best sweep for tau/Cm according to the tau/Cm rules in `cc_calc_core.py`.

Saved fields:

- `tau_sweep_ch0`, `tau_sweep_ch2`
- `tau_selection_note_ch0`, `tau_selection_note_ch2`
- `tau_skip_reason_ch0`, `tau_skip_reason_ch2`
- `Cm_skip_reason_ch0`, `Cm_skip_reason_ch2`
- `delta_V_mV_ch0`, `delta_V_mV_ch2`
- `V_post_min_mV_ch0`, `V_post_min_mV_ch2`

## CC / Gj / Rin

### Per-sweep CC sheet values

Saved for each analyzed sweep:

- `cur_step_pA`
- `delta_V_active_mV`
- `delta_V_passive_mV`
- `Vm_active_stim_mV`
- `CC`
- `CC_norm`
- `Gj_sweep_nS`
- skip reasons if a value could not be calculated

### File-level summary values

Saved per file:

- `CC12`, `CC21`
- `Gj12_nS`, `Gj21_nS`
- `Rin_ch0_MOhm`, `Rin_ch2_MOhm`
- `R2_ch0`, `R2_ch2`
- `n_IV_ch0`, `n_IV_ch2`
- negative-CC bookkeeping fields

## Spikelet analysis

Spikelets are analyzed in both directions:

- `12` = `ch0 -> ch2`
- `21` = `ch2 -> ch0`

### Sweep choice for the primary spikelet QC / summary

Primary sweep choice:

1. first sweep with `>= 4` APs
2. else first sweep with `3` APs
3. else first sweep with `2` APs
4. else fallback to `1`-AP sweeps only if there is no multi-spike sweep

That selected sweep is the `primary` / prominent sweep used in QC and in the main file-level spikelet summary.

### Original spikelet scheme

For each eligible AP:

- baseline = passive mean in `1 ms` before `t = 0`
- search window = `15 ms` after AP start
- search uses Gaussian smoothing with `sigma = 0.3 ms`
- detection requires a local peak with post-peak drop:
  - `SPIKELET_MIN_PROMINENCE_MV = 0.15`
- noise gate is OFF for the original per-AP spikelet scheme

Notes:

- if `amp_spikelet / amp_spike > 1`, the point is marked as error
- such values stay in Excel but are omitted from ratio plots
- negative delays are kept and marked

### Meantrace spikelet scheme

This is the averaged-signal scheme used on the aligned mean passive trace.

Current settings:

- window = first `15 ms` after AP start
- smoothing sigma = `0.3 ms`
- detection = local peak with post-peak drop
- `SPIKELET_MEANTRACE_MIN_PROMINENCE_MV = 0.05`
- noise gate OFF

For meantrace values:

- spikelet amplitude is taken from the smoothed mean passive trace
- delay is the difference between passive spikelet peak and active mean AP peak
- ratio = `amp_spikelet / amp_spike`

### Extra "nearest mode Vm" selection

The code also computes the sweep whose spikelet baseline is closest to the folder-wide baseline with the highest density (`mode Vm`, from KDE / histogram fallback).

This is used for the extra spikelet plots and for the compact Excel sheet.

## Excel sheets

## 1) `Summary_short`

Compact user-facing sheet.

Contains:

- file / recording time
- selected analysis blocks
- main CC / Gj / Rin values
- main cell-property values
- sweep used for cell properties + injected current
- main tau / Cm values
- tau sweep + `delta_V_mV` + `V_post_min_mV`
- `mode_spikelet_baseline_mV`
- two meantrace spikelet column groups:
  - `primary_*` = primary / prominent sweep
  - `near_mode_vm_*` = sweep nearest the folder mode baseline

For each spikelet group and each direction (`12`, `21`), the compact sheet stores:

- sweep number
- baseline
- meantrace spikelet amplitude
- peak delay
- amplitude ratio
- detected / skip reason

Column coloring:

- `primary_*` columns: one fill color
- `near_mode_vm_*` columns: another fill color

## 2) `File_summary`

Full per-file summary sheet.

Contains:

- file-level CC / Gj / Rin
- all cell-property values
- all tau / Cm values
- all spikelet summary values for both directions
- skip reasons and notes

This is the complete per-file export.

## 3) `All_data`

Per-sweep CC / Gj sheet.

One row per analyzed sweep with:

- current step
- delta V values
- active Vm during stimulus
- CC
- normalized CC
- sweep Gj
- skip reasons

This sheet is for detailed CC inspection across sweeps.

## 4) `Spikelets`

Per-AP spikelet sheet.

One row per analyzed AP with:

- sweep
- AP index
- active and passive peak times
- 10% times
- active amplitude
- spikelet amplitude
- amplitude ratio
- delay values
- baseline
- quality flags
- skip reasons

This is the most detailed spikelet sheet.

## 5) `Spikelet_sweeps`

Per-sweep spikelet sheet.

One row per spikelet-analyzed sweep and direction with:

- number of APs
- number of detected spikelets
- original-scheme sweep means
- average-trace metrics
- meantrace metrics
- baseline
- metric source
- fallback flags
- primary-sweep flag
- skip reasons

This is the best sheet if you want spikelet results aggregated per sweep rather than per AP.

## QC plot notes

Primary spikelet QC PNG has 3 panels:

1. raw active + passive sweep overlay
2. aligned AP mean panel for the original spikelet scheme
3. meantrace panel

The meantrace panel now shows:

- all aligned passive traces as thin lines
- thick mean passive trace
- thin smoothed passive trace in another color
- active mean trace
- mean AP peak marker
- smoothed spikelet peak marker

