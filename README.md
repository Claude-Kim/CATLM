# CATLM (Cat Adaptive Tiny Language Model) Python Agent & Simulator

`src/catlm_simulator.py` is a reference Python implementation based on the CATLM Concept Definition.
`src/sit_core.py` provides the environment-agnostic SIT engine used by the simulator.

## Features

- Models 4 initial trait values (activity / sociability / appetite / cowardice) and 16 state values
- Applies 10 user actions via a 10×16 action-state impact matrix
- Computes state changes using trait multipliers (`0.6, 0.8, 1.0, 1.3, 1.6`)
- Calculates forward-looking collapse forecast intensity `Ct` (idle-drift projection) and determines stage (Normal / Caution / Warning / Collapse)
- Links collapse threshold `θ` to the cowardice trait; SURVIVAL/NORMAL mode transitions apply hysteresis
- Irreversible capacity decay permanently shrinks the feasible action set
- `SITCore`: Mixture-of-Experts latent strategy state `z_t` (SAFE / GREEDY / REPAIR) with EMA resistance (λ=0.25); detects Structural Inference Transitions via baseline displacement persistence
- Generates dialogue via the 512-token `DialogueBank` (tone + tag weighted selection + emoticon)

## Running

```bash
python3 src/catlm_simulator.py
```

Outputs a per-tick log with columns:
`t  mode  Ct(fwd)  Ct(now)  stage  action  cap  alpha  safe_w  dstr  SIT#`

- `safe_w` — SAFE expert weight in z_t (high value = defensive strategy dominant)
- `dstr` — SIT displacement streak (ticks since z left current attractor basin)
- `SIT#` — cumulative SIT event count

## Experiments

Three experiments support the paper's empirical claims. Run all from the project root.

### EXP-001: SEED_SWEEP — H1 Effect-Size Analysis

Paired seed sweep comparing Irreversible (decay enabled) vs Ablated (decay=0) conditions.

```bash
python3 src/exp_001_seed_sweep.py [--seeds N] [--ticks N] [--n-boot N] \
                                   [--eps EPS] [--cowardice N] [--sociability N] \
                                   [--neglect TICKS] [--no-pdf]
```

Outputs → `results/exp-001/`:
- `exp-001_per_seed.csv` — per-seed raw data (all metrics, both conditions)
- `exp-001_summary.txt` — Δ statistics + percentile bootstrap 95% CI report
- `exp-001_robustness.png/pdf` — histogram panels with CI bands (fig6)

### EXP-002: ROBUSTNESS_GRID — Parameter Robustness Sweep

Maps the region where H1 effects hold across a grid of SIT hyperparameters (`eps`, `k_persist`, `H`, `decay_crisis`).

```bash
python3 src/exp_002_robustness_grid.py [--seeds N] [--workers N] [--no-pdf]
```

Outputs → `results/exp-002/`:
- `exp-002_grid.csv` — per-cell summary statistics
- `exp-002_heatmaps.pdf` — heatmaps: eps×k per (H, decay, metric)
- `exp-002_readme.md` — effect-region summary

### EXP-003: NULL_FPR — False Positive Rate via Permutation

Permutation-based FPR measurement for the SIT detector. Tests three null conditions:
`NULL_CT_PERMUTE` / `NULL_OBS_PERMUTE` / `NULL_ZPREF_RANDOM`.

Key finding: real CATLM dynamics exhibit **attractor stability** — genuine strategy states persist longer than permuted baselines (real SIT count < null SIT count).

```bash
python3 src/exp_003_null_fpr.py [--seeds N] [--ticks N] [--neglect N] [--no-pdf]
```

Outputs → `results/exp-003/`:
- `exp-003_null_counts.csv` — per-seed SIT counts (real vs 3 null conditions)
- `exp-003_summary.json` — FPR, mean/median/std, Δmean, p-value (strict & exploratory)
- `exp-003_plots.png/pdf` — violin+histogram, FPR bar chart, Δmean panel

## Paper Figures

Generates all publication figures (PNG + PDF) from a single run.

```bash
python3 src/paper_figures.py [--output figures/] [--ticks 120] [--seed 42]
```

Generated → `figures/`:
| File | Description |
|------|-------------|
| `fig1_main_trajectory` | Ct / z_t / capacity / care_alpha over time |
| `fig2_moe_resistance` | EMA resistance: z̃_t (instantaneous) vs z_t (resistant) |
| `fig3_capacity_ablation` | Irreversible decay vs ablated control condition (H1) |
| `fig4_trait_sensitivity` | Mean Ct and SIT events across cowardice / sociability levels |
| `fig5_z_simplex` | z_t trajectory on SAFE–GREEDY–REPAIR simplex |

## File Structure

```
src/
  catlm_simulator.py        — CATLMAgent, CatProfile, CatState, DialogueBank, demo
  sit_core.py               — SITCore, SITConfig, SITStepResult (environment-agnostic)
  exp_001_seed_sweep.py     — EXP-001: paired seed sweep for H1 effect-size
  exp_002_robustness_grid.py — EXP-002: hyperparameter robustness grid
  exp_003_null_fpr.py       — EXP-003: permutation-based FPR measurement
  paper_figures.py          — publication figure generation (fig1–fig5)
  catlm_web.py              — Streamlit web demo
  run_init.py               — reproducible run-seed + deterministic trait roll helper
dialogue_bank_512.json      — token bank for dialogue generation
figures/                    — generated paper figures (PNG + PDF)
results/                    — experiment output (CSV, JSON, plots)
  exp-001/
  exp-002/
  exp-003/
CATLM_concept_define.md
```

## Extension Points

- **Game balance**: Adjust `ACTIONS_MATRIX` labels and `CAPACITY_DECAY_*` constants in `catlm_simulator.py`
- **SIT tuning**: Modify `SITConfig` fields (`eps`, `k_persist`, `lam`, `insight_phi`) in `CATLMAgent.__init__`; `sit_core.py` is environment-agnostic and requires no changes
- **Word pool**: Expand `dialogue_bank_512.json` (schema: `tokens[].{id, text, category, tone[], intensity, tags[]}`)
- **Batch replay**: Call `step(user_action=action)` directly; `tick(hours=n)` delegates internally for legacy compatibility
- **Recovery items (BM)**: Implement capacity recovery logic in `_tick_one()` — current capacity decay is unidirectional
- **Web demo**: Run `streamlit run src/catlm_web.py` for an interactive browser-based interface

## Related Paper
[Structural Inference Transitions Under Irreversible Survival Constraints](https://zenodo.org/records/18780274)
