# Whole-cell electrophysiology and cell-culture statistics

Jupyter notebooks used to measure passive and active properties of patch-clamp recordings (Axon ABF files) and to run the group statistics for those measurements and for live/dead, cell-size, and cell-density counts.

Replace the fields below before the repository is cited from a paper.

- **Manuscript:** [title]
- **Authors:** [names]
- **Journal / year:** [journal, year]
- **DOI:** [doi]

Raw ABF recordings are not included. Each notebook reads either a folder of `.abf` files or numbers typed into the notebook.

## Notebooks in this release

| Notebook | What it does | Input | Output |
| --- | --- | --- | --- |
| `Cell Properties for one folder.ipynb` | Active and passive properties for every ABF in one folder | Folder of `.abf` files | `results_depolarisation.xlsx` and one PNG per file |
| `FWHM recalculation cuz of bad filtering.ipynb` | Same spike helpers, plus a batch estimate of membrane time constant and capacitance | Folder of `.abf` files | Excel with tau, capacitance, and the sweep used |
| `Tau Cm calculations.ipynb` | Function only: membrane time constant and capacitance from one ABF object | Called from other notebooks | tau (ms), capacitance (pF), sweep index |
| `TRN RBs all files in a folder.ipynb` | Rebound burst / low-threshold spike metrics after the current step | Folder of `.abf` files | `results.xlsx` and one QC figure per file |
| `LIVE DEAD Cell Statistics calculations.ipynb` | Within-group time course and between-group tests for viability, size, and density | Numbers typed into the cells | Printed tables |
| `Statistics for 2 Groups.ipynb` | Two-group descriptive statistics, normality, Mann–Whitney, effect sizes | Excel with columns `group1`, `group2` | Excel of test results |
| `Statistics for 3 groups.ipynb` | Three-group Kruskal–Wallis, pairwise tests, multiplicity correction | Excel or CSV with three columns | Printed pairwise table |

Other `.ipynb` files in this repository are earlier working copies. They are not required for the analyses above.

## Setup

Python 3.10, 3.11, or 3.12. `tkinter` is used for the folder dialog and ships with the standard Windows and macOS installers. On Linux, install the Tk bindings for your Python (for example `python3-tk` on Debian/Ubuntu).

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

A clean install of this file on Python 3.12 resolved to numpy 1.26.4, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2, pyABF 2.3.8, statsmodels 0.15.0, scikit-posthocs 0.17.1, and notebook 7.6.4. Imports used by the seven notebooks were checked against that environment. The notebooks were not re-run, because they need the original ABF files and a desktop folder dialog.

`PySimpleGUI` is pinned to 4.60.5.1, a re-upload of the last PySimpleGUI 4 release (LGPL-3.0). Releases 5 and 6 stop on import unless a license key is set. The batch notebooks import the package; the folder dialog itself uses tkinter.

The ABF notebooks were written on Windows. Paths such as `C:\Users\user\Documents\IMBIT\...` are hard-coded. Change them before running. Folder dialogs need a local desktop session.

## ABF layout

Recordings are opened with [pyABF](https://github.com/swharden/pyABF). The notebooks expect:

- voltage on channel 0 or channel 2, selected by `my_search_ch`;
- the current command on the next channel (`my_search_ch + 1`);
- the stimulus as the samples where that command trace is not zero;
- for a 4-channel file, the command timing of the second cell is read from channel index 1.

Windows taken around the step:

- 125 ms immediately before the step (baseline);
- the last 250 ms of the step (steady level);
- 500 ms after the step in the cell-properties notebook, 750 ms after the step in the rebound-burst notebook.

Voltage is treated as mV and current as pA. Input resistance in MΩ is `ΔV / ΔI × 1000`. Capacitance in pF is `tau_ms / Rin_MΩ × 1000`.

## Cell properties for one folder

Run the cells from top to bottom.

1. Set `folder_path` to the directory where the Excel file and the PNG traces should be written. The notebook creates `results_depolarisation.xlsx` there.
2. Set `my_search_ch` to `0` or `2`.
3. Choose the folder that contains the `.abf` files in the dialog.

For each file the notebook:

- takes the first sweep with at least four spikes above −10 mV;
- finds spike start and end from sign changes of the second derivative after a Gaussian smooth (σ = 3 samples): the start is the second-to-last inflection before the peak, the end is the second inflection after the peak;
- writes half-width of the first spike (`scipy.signal.peak_widths`, half height), the amplitude ratio of spike 2 to spike 1, the mean of the later successive amplitude ratios, delay of the first spike from step onset, initial firing rate (spikes 1–2), late rate (last two spikes), mean rate of the intermediate intervals, and the late/initial ratio;
- estimates input resistance from the sweeps before the first spike. Relative Rin is the mean of `ΔV/ΔI` on steps whose voltage during the pulse lies between −85 and −50 mV (steps with `|ΔI| ≤ 0.15` pA are skipped). Absolute Rin is the slope of voltage against current in that same voltage window, reported with the fit R² and the zero-current intercept;
- records holding voltage and injected current on the sweep used for the spikes.

Columns written to Excel: file name, AP2/AP1 ratio, attenuation trend, initial frequency (Hz), late frequency (Hz), mean frequency (Hz), frequency ratio, FWHM (ms), delay of the first spike (ms), absolute and relative Rin (MΩ), resting voltage from the fit (mV), holding voltage (mV), injected current (pA), R² of the absolute Rin fit.

A failed file is printed and skipped. The table is saved once after the loop.

## FWHM recalculation notebook

`FWHM recalculation cuz of bad filtering.ipynb` defines the same spike, Rin, and FWHM helpers as the cell-properties notebook, and also defines `tau_and_capacitance`.

The batch cell does not write FWHM. It loops over a chosen folder and saves only:

- `my_tau_ms`
- `my_capacitance_pF`
- `my_best_sweep`

Set `folder_path` (output directory) and `my_search_ch` before the loop. In the copy in this repository the batch channel is `2`, and the Excel file is named `Tau_mPFC_Ch2.xlsx`.

The tau routine in this notebook searches sweeps whose voltage step falls between −50 and −10 mV and keeps the sweep closest to −10 mV. Tau is the time from the analysed index to 63% of that voltage change. Capacitance uses the relative Rin (MΩ) from `IV_Rin`.

Half-width of the first spike for a whole folder is the FWHM column of `Cell Properties for one folder.ipynb`.

## Tau and capacitance

`Tau Cm calculations.ipynb` contains only `tau_and_capacitance`. It does not open files. Pass a pyABF object, the voltage channel, the index before the step, the index after the step, and Rin in MΩ.

The function:

- keeps the sweep whose negative voltage change lies in `search_range` (default −35 to −10 mV) and is closest to zero;
- takes the baseline as the mean of the first 125 ms from `pre_start`, and the peak change as the minimum between `pre_start` and `post_stop`;
- sets tau to the time from `pre_start` until the trace reaches 63% of that change;
- converts tau and Rin to capacitance in pF.

The capacitance line reads a name `Rin_abs`, which is not an argument of the function. Define `Rin_abs` in the notebook before the call, or edit that line to use the argument `Rin_rel`. The copy of this function in `FWHM recalculation cuz of bad filtering.ipynb` already uses `Rin_rel`.

Sampling rate defaults to 10 kHz when the argument is omitted.

## Rebound bursts (TRN)

`TRN RBs all files in a folder.ipynb` measures the rebound that follows the current step. The analysed window starts at the end of the step. Sweep number and voltage channel are set in the notebook (`my_sweep = 0`, `my_search_ch = 0` in this copy).

For each ABF it estimates:

- start of the rebound from the minimum of the smoothed first derivative before the first spike;
- end of the rebound from the later region where the smoothed first and second derivatives are near zero;
- width of the low-threshold spike (LTS) in ms, number of spikes in the burst, initial rate (first two spikes), rate across all spikes, and rate excluding the first interval;
- voltage at spike threshold, amplitude of the first spike, LTS amplitude, and the amplitude ratio of spike 2 to spike 1;
- holding voltage, injected current, and membrane potential from the command and voltage channels.

The QC figure has three panels (voltage, first derivative, second derivative) and is saved under `folder_path` using the ABF id as the file name. Numeric results are appended to `results.xlsx` in that same folder. Create the empty workbook by running the table cell before the loop. A failed file is printed and skipped.

## Live/dead, size, and density

`LIVE DEAD Cell Statistics calculations.ipynb` is four self-contained analyses. Replace the lists at the top of a cell with the values for that run. Group labels in the output are `ctrl` and `drug`. Days that exist in only one group are omitted from the between-group test.

| Cell | Data as stored in the notebook | Within each group, across days | Between groups, each day |
| --- | --- | --- | --- |
| Viability | percent viable, one list per day | Kruskal–Wallis, then Dunn with Holm | Mann–Whitney, Holm across days |
| Cell size | values divided by the day-1 mean of that group | Kruskal–Wallis, then Dunn with Holm | Mann–Whitney, Holm across days |
| Cell density, parametric | counts as entered | one-way ANOVA, Tukey HSD | unpaired t-test, Holm across days |
| Cell density, non-parametric | the same density lists | Kruskal–Wallis, then Dunn with Holm | Mann–Whitney, Holm across days |

Stars printed next to a p value: `****` < 0.0001, `***` < 0.001, `**` < 0.01, `*` < 0.05, otherwise `ns`. Results are printed, not written to a file.

## Two groups

`Statistics for 2 Groups.ipynb` reads the first sheet of an Excel file. It uses two numeric columns named `group1` and `group2`. Empty cells are dropped.

For each column it reports n, mean, sample standard deviation, median, interquartile range, minimum, and maximum; a Shapiro–Wilk test; and a D’Agostino–Pearson test when n ≥ 20.

The comparison is a two-sided Mann–Whitney U test. Effect sizes, computed on the same pair for every results row:

- Cliff’s delta, with |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large;
- Cohen’s d (pooled standard deviation) and Hedges’ g, with |effect| < 0.2 negligible, < 0.5 small, < 0.8 medium, otherwise large.

The output path is set by `results.to_excel(...)` near the end of the first cell. The import cell is below the analysis cell; run the imports first, or move them above the analysis.

## Three groups

`Statistics for 3 groups.ipynb` reads an Excel sheet or a CSV (`use_excel`). Set `input_file` and `cols` to the three column names. Non-numeric cells are dropped.

The notebook prints n, median, and IQR per column, a Shapiro–Wilk test (n ≥ 3), and a Kruskal–Wallis test. Effect size is epsilon-squared, `(H − k + 1) / (n − k)`.

Each pair of columns then gets:

- a two-sided Mann–Whitney U test;
- a permutation test of the difference in medians (`n_permutations = 10000`, p = `(count + 1) / (n_perm + 1)`);
- Cliff’s delta.

Mann–Whitney p values are corrected across the three pairs by Bonferroni and by Benjamini–Hochberg. The star column uses the Bonferroni p value with thresholds 0.05, 0.01, and 0.001, each divided by the number of pairs. The pairwise table is displayed. Saving it to Excel is left commented at the bottom of the cell.

## Citation

[Add the citation of the paper, or of this repository, once the manuscript is accepted.]

## License

No license is set yet. Add a `LICENSE` file before public release so others can reuse the notebooks. A permissive license such as MIT or BSD-3-Clause is a common choice for analysis code; the journal may also ask for one.
