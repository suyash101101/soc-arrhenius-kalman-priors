# SOC_Pipeline_v3.3: run guide (rework round)

Same folder layout as v3.2 (data folders next to `SOC_Pipeline_v3.m`). Everything is driven by `RUN_ALL.m`:
set `stage`, run, upload every `results_*` folder and `console_*.txt`. Finished runs are skipped, MATLAB can be restarted.

## What changed (maps to the review document)
| Review item | Change in code |
|---|---|
| 1 SPKF | sigma-point scaling default alpha=1, kappa=1 (legacy 1e-3/0 still run as a diagnostic); min eig(P) logged for EKF/SPKF/IEKF/SHEKF; Cholesky failures counted (`results_spkf_diag*`, `results_pmin.txt`) |
| 2 init. error | every segment starts with a random 10-20 % SOC error (`cfg.initErr`); P0(1,1) = `cfg.p0soc` |
| 3 convergence | time to stay within 2 % for 60 s (`results_conv_rows.tex`) |
| 4 LSTM+KF capacity | uses QcAll(eT), same capacity model as the filters (note printed) |
| 5 Coulomb counting | ideal / initial-error only / realistic offset+gain (`results_cc_rows.tex`) |
| 6 seeds | 5 seeds for BiLSTM, PI-LSTM, PI-LSTM lambda=0 and Fusion (`results_seedtab_rows.tex`, `results_seeds_all.csv`) |
| 7 leave-one-cycle-out | new protocol `lcyc` (US06 held out); Mixed files are used for fit/val |
| 8 statistics | whole-segment bootstrap and per-segment Wilcoxon (30-min units for Braun) |
| 9 Arrhenius | parameter oracle, tau-law ablation, multi-start ECM spread, optional ECM bootstrap (`results_oracle*`, `results_ecm*`) |
| robustness | initial shift, noise, current offset, **current gain +2 %**; Table XIV `%` bug fixed |

## Stages (RUN_ALL.m, `stage = ...`)
| stage | content | est. time (CPU, 5 seeds) |
|---|---|---|
| A | MAIN: lg loco, samsung loco, samsung cross, lg loto, lg tempx, samsung tempx (SHEKF base, extras on) | 25-35 h |
| C | SPKF base, extras off, the six runs + lcyc (needed for the validation-based prior choice and the supplement) | 10-14 h |
| B | lg leave-one-cycle-out (US06) | 3-4 h |
| D | nearest-temperature lookup, 1 seed, 4 runs | 2-3 h |
| E | supplement: sparse, samsung loto, braun, SPKF variants | 6-10 h |

Order: A, C, B, D, E. Stage A is the minimum for the main paper. `preset = 'express'` (3 seeds, shorter training) for a first look.
Estimates are not measured; a GPU is used automatically by `trainNetwork`.
Delete `cache_*.mat` if you change anything in `cfg`.

## What to send back
All `results_*` folders (without `cache_*.mat`) and the `console_*.txt` files. Keep `results_lg_loco_baseshekf/models.mat`; the cross run needs it.

## Honest caveat
v3.3 was smoke-tested in Octave with stubbed networks on a tiny subset: the logic and file outputs run end to end, but no value from that test is meaningful and MATLAB-only calls (trainNetwork, fitrensemble) were not exercised. If a run fails, send the console text.
