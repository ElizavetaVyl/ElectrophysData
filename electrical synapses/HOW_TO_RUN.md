# Как запустить notebook

GitHub **не запускает** Jupyter. Нужно скачать файлы на компьютер и открыть notebook локально (Jupyter / VS Code / Cursor).

Эта папка — готовый набор для batch-анализа CC / cell properties / tau-Cm / spikelets.

## Какие файлы обязательны

Положите **в одну папку** (эта папка уже так собрана):

| Файл | Нужен? |
|------|--------|
| `cc_calc_core.py` | **да** — весь расчёт |
| `CC calculation.ipynb` | **да** — запуск |
| `CC calculation - README.md` | нет, только описание |
| `HOW_TO_RUN.md` | нет, только инструкция |

ABF-файлы **не** должны лежать в этой папке. Папку с `.abf` выберете в cell 2.

## Как скачать с GitHub

Пока изменения в ветке PR: `cursor/cc-vs-vm-linear-fit-b101`  
(не берите только `main`, там старая версия без spikelets / Summary_short).

### Вариант A — ZIP (проще)

1. Откройте репозиторий на GitHub.
2. Слева от зелёной кнопки **Code** выберите ветку  
   `cursor/cc-vs-vm-linear-fit-b101`.
3. **Code → Download ZIP**.
4. Распакуйте архив.
5. Откройте папку `electrical synapses`.

### Вариант B — git clone

```
git clone https://github.com/ElizavetaVyl/ElectrophysData.git
cd ElectrophysData
git checkout cursor/cc-vs-vm-linear-fit-b101
```

Дальше работайте из папки `electrical synapses`.

## Как открыть notebook

1. Установите зависимости один раз:

```
pip install pyabf pandas matplotlib scipy openpyxl
```

2. Откройте **именно** файл  
   `electrical synapses/CC calculation.ipynb`  
   (не старую копию с рабочего стола / другой папки).

3. **Kernel → Restart**.

4. Запускайте ячейки по порядку: **0 → 1 → 2 → 3**.

5. В cell 1 проверьте строку `cc_calc_core file:` — путь должен указывать на  
   `electrical synapses/cc_calc_core.py`.

6. Cell 2: выберите папку с `.abf` и куда сохранить Excel.

7. Cell 3: окно **Analysis blocks** может быть за Jupyter.  
   **Select all** или нужные блоки → **OK**.

## Куда пишутся результаты

Рядом с папкой ABF (не обязательно внутри `electrical synapses`):

- Excel: `{имя_папки}_CC_data.xlsx`
- `Cell_properties_plots/`
- `CC_plots/`
- `Spikelet_plots/`

## Если «как раньше» / старые графики

Значит Jupyter открыл старый `.ipynb` или старый `.py`.  
Снова скачайте оба файла из этой папки, положите вместе, **Kernel → Restart**.
