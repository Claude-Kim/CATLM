# EXP-002: ROBUSTNESS_GRID — Summary

## Configuration
- Grid cells: 400  (5 eps × 4 k × 4 H × 5 decay)
- Seeds per cell: 5
- n_ticks: 120  |  neglect: 40 IDLE ticks
- Profile: {'name': 'Cat', 'activity': 3, 'sociability': 5, 'appetite': 3, 'cowardice': 5}
- Scenario: 40 IDLE → auto-policy
- Δ = Irreversible (B) − Ablated (A, decay=0)  [paired by seed]

## Sweep Ranges
- eps: [0.15, 0.2, 0.25, 0.3, 0.35]
- k_persist: [2, 3, 4, 5]
- H (FORECAST_HORIZON): [1, 3, 5, 7]
- decay_crisis: [0.0, 0.02, 0.04, 0.06, 0.08]
- decay_overload = decay_crisis × 0.5

## Effect Region (cells where decay_crisis > 0, n=320)

| Metric        | Frac(Δ>0) | Mean Δ    | Best (eps,k)  |
|---------------|-----------|-----------|---------------|
| mean Ct (↓good)  | 0.000     | -0.00167  | eps=0.15 k=3  (-0.00167) |
| SIT count        | 0.000     | +0.00000  | eps=0.15 k=2  (+0.00000) |
| Normal ticks     | 0.000     | +0.00000  | eps=0.15 k=2  (+0.00000) |
| α_T final        | 0.000     | +0.00000  | eps=0.15 k=2  (+0.00000) |

## Interpretation
- Frac(Δ>0) > 0.5 → majority of seeds show positive effect direction.
- **ΔCt < 0** is the primary H1 effect: irreversible decay leads to lower mean
  collapse forecast intensity (capacity-restricted actions maintain lower state).
- Δsit ≈ 0: SIT events rarely fire at eps ∈ {0.15…0.35} because d_base_max ≈ 0.07
  in this scenario (EMA smoothing + baseline tracking keeps d_base small).
  → This is by design: the game-default eps=0.25 produces rare structural transitions.
- Cells with decay_crisis=0 always show Δ≈0 (sanity check).
- See exp-002_heatmaps.pdf for full visual breakdown.
- eps×k summary page shows P(Δ>0) averaged over all H and non-zero decay.

## Files
- `exp-002_grid.csv`        — full per-cell statistics
- `exp-002_heatmaps.pdf`    — multi-page heatmaps (eps×k, faceted by H and decay)
- `exp-002_H{1,3,5,7}.png` — individual H-page PNGs
- `exp-002_summary_p_pos.png` — P(Δ>0) summary across all conditions
