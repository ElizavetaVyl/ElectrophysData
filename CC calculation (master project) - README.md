# CC calculation (master project) — README

Документ описывает, **что и как обрабатывает** batch-анализ:

- `cc_calc_core.py`
- `CC calculation (master project).ipynb`

> **Примечание:** этот файл раньше был только локально (`C:\Users\user\ElectrophysData\...`) и **не был в GitHub**. Сейчас он добавлен в репозиторий и обновлён под текущую версию кода.

---

## Как запускать

1. Положить `cc_calc_core.py` и notebook в одну папку с ABF.
2. **Kernel → Restart**, ячейки **0 → 3**.
3. В cell 3 выбрать блоки анализа (CC, cell properties, tau/Cm, spikelets).

Папки с графиками создаются рядом с ABF:

- `Cell_properties_plots/`
- `CC_plots/`
- `Spikelet_plots/`

---

## Блоки анализа

| Блок | Что считает |
|------|-------------|
| `cc` | CC, CC_norm, Gj, Rin |
| `cell_props` | AP21 ratio, частоты, FWHM, V_rest, Rin по I–V |
| `tau_cm` | tau, Cm |
| `spikelets` | spike / spikelet, QC, Excel, folder plots |

Выбранные блоки записываются в Excel: `analysis_blocks`.

---

## CC и CC_norm

### CC на одном sweep

\[
CC = \frac{\Delta V_{passive}}{\Delta V_{active}}
\]

- `delta_V_active_mV` — изменение Vm на **active** канале между pre/post окнами стимула
- `delta_V_passive_mV` — то же на **passive** канале
- если |ΔI| или |ΔV_active| слишком малы → sweep пропускается (`CC_skip_reason`)

### CC_norm

**Отдельно для каждого файла и каждого направления** (`12` = ch0→ch2, `21` = ch2→ch0):

1. Берутся все **валидные** CC sweep'ов этого файла в данном направлении
2. Считается **средний CC** по файлу (`CC12` или `CC21` в `File_summary`)
3. Для каждого sweep:

\[
CC\_norm = \frac{CC_{sweep}}{mean(CC\ \text{файла в этом направлении})}
\]

- если средний CC = 0 или None → `CC_norm` не считается
- `CC_norm` используется в графиках **CC_norm vs Vm** и для slope-over-time

### Gj

На уровне файла:

\[
Gj = \frac{CC}{Rin\ passive\ cell} \times 1000\ \text{(nS)}
\]

(с проверками 0 < CC < 1)

---

## Cell properties

Отдельно для **ch0** и **ch2**.

### Какой sweep

Первый sweep с **≥ 4 AP** в окне стимула.

### Что сохраняется

- `AP21_ratio`, частоты, `FWHM_ms`, `delay_AP1_ms`
- `V_rest_mV`, `Rin_abs/rel`, `hold_V_mV`
- **`props_sweep_ch0/ch2`** — номер sweep для расчёта
- **`inj_current_pA_ch0/ch2`** — инъекция тока на этом sweep

---

## Tau / Cm

Отдельно для **ch0** и **ch2**.

### Что сохраняется

- `tau_ms`, `Cm_pF`
- **`tau_sweep_ch0/ch2`** — какой sweep использован
- **`delta_V_mV_ch0/ch2`** — дельта напряжения для расчёта
- **`V_post_min_mV_ch0/ch2`** — минимальное Vm в post-окне (используется в расчёте delta V)
- `tau_selection_note`, skip reasons

---

## Spikelet analysis

Два направления:

- **12** = ch0 → ch2 (active ch0, passive ch2)
- **21** = ch2 → ch0

### Primary sweep (prominent)

Порядок выбора:

1. первый sweep с **≥ 4 AP**
2. иначе с **3 AP**
3. иначе с **2 AP**
4. иначе fallback на **1 AP**, только если нет multi-spike sweep

Этот sweep — **primary**: QC PNG, основной file-level summary.

### Nearest mode Vm

Дополнительно выбирается sweep, чей **spikelet baseline** (passive, 1 ms до t=0) ближе всего к **mode Vm** папки (KDE плотности baseline по всем sweep).

Используется в extra plots и в `Summary_short` (колонки `near_mode_vm_*`).

### Схема 1: per-AP (original)

На каждый AP (обычно AP2+, кроме 1-AP fallback):

- baseline = mean passive **1 ms до t=0**
- окно поиска = **15 ms** после AP start
- smooth σ = **0.3 ms**, local peak + prominence **≥ 0.15 mV**
- noise gate **OFF**
- ratio > 1 → ошибка, в ratio plots не идёт
- negative delay → помечается, но сохраняется

### Схема 2: meantrace (средний сигнал)

На **aligned mean passive trace** (AP2+):

- окно = **15 ms** после t=0
- smooth σ = **0.3 ms**
- prominence **≥ 0.05 mV**, noise gate **OFF**
- **амплитуда spikelet** — со **сглаженного** mean passive
- delay = t_peak_passive − t_peak_active на mean traces
- ratio = amp_spikelet / amp_spike

QC PNG (3-й сабплот): тонкие aligned traces, **толстый mean**, **тонкий smoothed** другим цветом, маркеры пиков.

---

## Excel — какие листы и что в них

### 1. `Summary_short` (короткая выжимка)

Для быстрого просмотра. Цвета колонок:

- **`primary_*`** — primary / prominent sweep (синий фон)
- **`near_mode_vm_*`** — sweep ближайший к mode baseline (оранжевый фон)

Содержит: CC, Gj, Rin, cell props, tau/Cm metadata, meantrace spikelet для обоих направлений.

### 2. `File_summary` (полная сводка по файлам)

**Одна строка = один ABF файл.** Всё file-level:

- CC12/21, Gj, Rin
- все cell properties
- все tau/Cm
- все spikelet summary поля (`spikelet_*_12`, `spikelet_*_21`)
- skip reasons, notes

### 3. `All_data` (CC по sweep'ам)

**Одна строка = один sweep × direction.**

- `cur_step_pA`, delta V, Vm
- `CC`, **`CC_norm`**
- `Gj_sweep_nS`
- skip reasons

Для детального разбора CC по sweep'ам.

### 4. `Spikelets` vs `Spikelet_sweeps`

| | **Spikelets** | **Spikelet_sweeps** |
|---|---------------|---------------------|
| **Гранулярность** | **1 строка = 1 AP** | **1 строка = 1 sweep + direction** |
| **Детальность** | максимальная | агрегированная |
| **Что внутри** | времена пиков, amp spike/spikelet, ratio, delay, baseline, flags на **каждый AP** | n_AP, n_detected, **средние по sweep**, avg/meantrace metrics, `is_primary` |
| **Когда смотреть** | нужен разбор **отдельных AP** | нужны **итоги по sweep** или сравнение sweep'ов |

**Spikelets** — самый подробный spikelet-лист (уровень AP).

**Spikelet_sweeps** — промежуточный уровень между AP и file summary; удобен для primary vs других sweep'ов и meantrace по sweep.

### 5. Полный список листов

1. `Summary_short`
2. `File_summary`
3. `All_data`
4. `Spikelets`
5. `Spikelet_sweeps`

---

## QC spikelet PNG

`*_12_spikelets.png`, `*_21_spikelets.png` в `Spikelet_plots/`:

1. raw active + passive overlay
2. aligned AP mean (original scheme, 15 ms)
3. meantrace panel (aligned traces + mean + smoothed)

---

## Константы (cc_calc_core.py)

| Параметр | Значение |
|----------|----------|
| SPIKELET_PEAK_MS | 15 ms |
| SPIKELET_PEAK_SMOOTH_MS | 0.3 ms |
| SPIKELET_MIN_PROMINENCE_MV | 0.15 mV (per-AP) |
| SPIKELET_MEANTRACE_PEAK_MS | 15 ms |
| SPIKELET_MEANTRACE_MIN_PROMINENCE_MV | 0.05 mV |
| SPIKELET_USE_NOISE_GATE | False |

---

*Последнее обновление: соответствует ветке с meantrace QC, Summary_short и parallel spikelet schemes.*
