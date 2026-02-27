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

## File Structure

```
src/
  catlm_simulator.py   — CATLMAgent, CatProfile, CatState, DialogueBank, demo
  sit_core.py          — SITCore, SITConfig, SITStepResult (environment-agnostic)
dialogue_bank_512.json — token bank for dialogue generation
CATLM_concept_define.md
```

## Extension Points

- **Game balance**: Adjust `ACTIONS_MATRIX` labels and `CAPACITY_DECAY_*` constants in `catlm_simulator.py`
- **SIT tuning**: Modify `SITConfig` fields (`eps`, `k_persist`, `lam`, `insight_phi`) in `CATLMAgent.__init__`; `sit_core.py` is environment-agnostic and requires no changes
- **Word pool**: Expand `dialogue_bank_512.json` (schema: `tokens[].{id, text, category, tone[], intensity, tags[]}`)
- **Batch replay**: Call `step(user_action=action)` directly; `tick(hours=n)` delegates internally for legacy compatibility
- **Recovery items (BM)**: Implement capacity recovery logic in `_tick_one()` — current capacity decay is unidirectional

## Related Paper
[Structural Inference Transitions Under Irreversible Survival Constraints](https://zenodo.org/records/18780274)
