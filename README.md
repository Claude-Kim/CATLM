# CATLM (Cat Adaptive Tiny Language Model) Python Agent & Simulator

`catlm_simulator.py` is a reference Python implementation based on the CATLM Concept Definition.

## Features

- Models 4 initial trait values (activity / sociability / appetite / cowardice) and 16 state values
- Applies 10 user actions via a 10×16 action-state impact matrix
- Computes state changes using trait multipliers (`0.6, 0.8, 1.0, 1.3, 1.6`)
- Calculates collapse forecast intensity `Ct` and determines stage (Normal / Caution / Warning / Collapse)
- Links collapse threshold `θ` to the cowardice trait
- Generates simple dialogue (tone + word pool + emoticon + cumulative relationship alpha)

## Running

```bash
python3 catlm_simulator.py
```

Outputs a log of sequentially applied demo scenarios: action, Ct, stage, and dialogue.

## Extension Points

- Replace qualitative labels in the current matrix (`+strong`, `-medium`, etc.) with real game balance data
- Extract the word pool into an external JSON/CSV resource and expand to 512+ tokens
- Extend the simulation loop to support event-log-based batch replay
- Integrate collapse-stage recovery conditions (BM items) with the actual in-game economy system

## Related Paper
Structural Inference Transitions Under Irreversible Survival Constraints (https://zenodo.org/records/18780274)
