# soc-arrhenius-kalman-priors

MATLAB pipeline and result tables for the paper **Temperature-Parametrised Kalman Priors and Learned Corrections for Lithium-Ion State-of-Charge Estimation Across Temperatures and Cells** (IEEE TTE submission candidate). Four Kalman filters on a 1-RC Thevenin model with Arrhenius temperature parametrisation, three residual learners (BiLSTM, PI-LSTM, LSBoost), and simplex NNLS stacking are evaluated on eleven protocols across three public datasets.

## Requirements
- MATLAB with Deep Learning Toolbox and Statistics and Machine Learning Toolbox (GPU optional)
- Python 3 + NumPy for `edge_benchmark/`

## Datasets
See `data/README_DATA.md` for DOIs. Put data folders next to `matlab/SOC_Pipeline_v3.m`. Braun data are CC BY-NC 4.0 — do not redistribute the raw files from this repository.

## How to run
From the MATLAB folder, open `RUN_ALL.m` and set `stage` to `'A'` … `'E'` (see `matlab/RUN_GUIDE.md`). Use `preset='express'` for a short smoke run. Full stage times are listed in the guide.

## Result folders ↔ protocols
| Short | Folder |
|---|---|
| LG-L | `results/results_lg_loco` |
| S-L | `results/results_samsung_loco` |
| S-X | `results/results_samsung_cross_baseshekf` |
| LG-T | `results/results_lg_loto` |
| S-T | `results/results_samsung_loto` |
| LG-E | `results/results_lg_tempx` |
| S-E | `results/results_samsung_tempx` |
| LG-S | `results/results_lg_sparse` |
| S-S | `results/results_samsung_sparse` |
| LG-C | `results/results_lg_lcyc_us06` |
| B-W | `results/results_braun_wltpseen` |
| B-R | `results/results_braun_regime` |

Folders without a prior suffix use the SPKF prior; `_baseshekf` = SHEKF prior; `_nearest_baseshekf` = nearest-temperature lookup.

## Randomness
Seeds are fixed in the pipeline, but retraining still moves RMSE slightly. The LG-T raw LSTM+KF reference is about 1.24% or 1.69% depending on the run — both are reported in the paper.

## Edge benchmark
`edge_benchmark/edge_benchmark.py` times NumPy float32 networks. Valid UNO Q numbers are aarch64 with peak RSS tens of MB (see `edge_runs_summary.txt`). Do not commit Mac `arm64` / multi-GB RSS JSONs as board results.

## Licence
MIT for code. Dataset licences remain with their publishers (Braun: CC BY-NC 4.0).

## Cite
Paper (when published) + this repository (Zenodo DOI after release; replace placeholder in `CITATION.cff`).
