# CATLM — Claude Code Guide

## Project Overview

**Cat Adaptive Tiny Language Model** is an on-device lightweight AI character adaptation system for the mobile game *SNAX Cats*.
Theoretical foundation: *Structural Inference Transitions Under Irreversible Survival Constraints* (SIT paper, pending arXiv publication).

Key files:
- [src/catlm_simulator.py](src/catlm_simulator.py) — main reference implementation
- [src/sit_core.py](src/sit_core.py) — SIT core (environment-agnostic MoE z_t + EMA hysteresis + insight detection)
- [CATLM_concept_define.md](CATLM_concept_define.md) — domain spec document

## Running

```bash
python3 src/catlm_simulator.py
```

Output columns: `t  mode  Ct(fwd)  Ct(now)  stage  action  cap  alpha  safe_w  dstr  SIT#`

## Architecture

### Data Models

```
CatProfile     fixed trait values — activity / sociability / appetite / cowardice (each 1~5)
CatState       mutable state values — 16 fields, range 0~255, clamp() required
CATLMAgent     agent — above two models + simulation loop; holds SITCore instance
Mode           NORMAL | SURVIVAL
DialogueToken  word token — id / text / category / tones / intensity / tags
EmoticonRule   emoticon rule — emoji / tone_weights
DialogueBank   512-token dialogue bank — JSON load, 3 indexes (_by_tone / _by_category / _by_tag)
SITConfig      frozen dataclass — all SITCore hyperparameters (lam, eps, k_persist, insight_phi, …)
SITCore        environment-agnostic SIT engine — R^3 MoE z_t, EMA resistance, SIT/insight detection
SITStepResult  per-tick output from SITCore.step() — z_pref, z, d_base, sit_event, insight_*, …
```

### Core Constants (modify with care)

| Constant | Meaning | Default |
|------|------|--------|
| `IMPACT_SCALE` | label → integer conversion (`+강`=36 … `-강`=-36) | fixed |
| `TRAIT_MULTIPLIER` | trait value (1~5) → multiplier (0.6~1.6) | fixed |
| `CAPACITY_DECAY_ON_CRISIS` | irreversible capacity loss per collapse-regime tick (× SCALE effective value) | 0.06 |
| `CAPACITY_DECAY_ON_OVERLOAD` | additional loss on fatigue/irritation overload | 0.03 |
| `CAPACITY_DECAY_SCALE` | global capacity loss scale (game balance tuning) | 0.65 |
| `SALVATION_COOLDOWN_HOURS` | u_t cooldown | 6 |
| `SALVATION_CAPACITY_SHIELD` | capacity loss reduction when u_t=1 | 0.08 |
| `ATTACHMENT_GAIN` | A_t integration intensity baseline | 0.015 |
| `MODE_EXIT_HYSTERESIS` | SURVIVAL exit hysteresis margin | 0.08 |
| `CRISIS_STREAK_FOR_CAPACITY` | consecutive collapse-regime ticks required to trigger capacity loss | 2 |
| `_BASE_TONE_PRIOR` | prior distribution over 8 tones (all 1.0) | fixed |

SIT detection parameters (formerly `SIT_EPS` / `SIT_PERSIST_K` module-level constants) are now fields of `SITConfig` and configured in `CATLMAgent.__init__`:

| SITConfig field | Meaning | Default |
|------|------|--------|
| `lam` | EMA smoothing for z_t (resistance) | 0.25 |
| `eps` | displacement threshold for SIT detection (R^3) | 0.25 |
| `k_persist` | consecutive ticks displaced required to fire SIT | 3 |
| `require_collapse_regime` | gate SIT detection to collapse regime only | True |
| `insight_phi` | SAFE weight threshold for "insight" | 0.7 |
| `insight_k_persist` | consecutive insight ticks to attain insight | 3 |

### Step API (v1.1)

```
tick(hours)                  ← legacy compatibility API, routes to step()
step(user_action, hours)     ← public game integration API, returns report
_tick_one(user_action)       ← internal 1-hour frame, returns detailed report
```

**`step()` return report structure:**
```python
{
  "t": int,
  "mode": "normal"|"survival",
  "Ct": float,               # forward-looking collapse forecast (H=3 idle projection)
  "Ct_instant": float,       # instantaneous collapse score (for UI display)
  "theta": float,
  "stage": str,              # "정상"|"주의"|"경고"|"붕괴"
  "action": str,
  "u_t": int,                # 0 or 1 (origin salvation)
  "Y": int,                  # survival proxy (0 or 1)
  "S_t": float,              # trust LLR proxy
  "E_t": int,                # collapse+intervention event flag
  "A_t": float,              # attachment integration intensity
  "alpha": float,            # care_alpha
  "capacity": float,
  "crisis_streak": int,
  "z_pref": tuple,           # instantaneous MoE preference (safe, greedy, repair)
  "z": tuple,                # resistant z_t after EMA (safe, greedy, repair)
  "safe_w": float,           # z[0] — SAFE expert weight
  "insight_now": bool,       # safe_w >= insight_phi this tick
  "insight_attained": bool,  # insight persisted >= k steps (latched True)
  "insight_streak": int,
  "sit_d_base": float,       # ||z - baseline||
  "sit_disp_streak": int,    # consecutive ticks displaced
  "sit_event": bool,         # SIT fired this tick
  "sit_count": int,          # total SIT events so far
  "sit_flag": int,           # 1 if sit_event else 0 (legacy compat)
}
```

### _tick_one() Execution Order (1 tick = 1 hour)

```
1.  t++
2.  Natural drift (hunger+6, boredom+5, loneliness+3×sociability_multiplier, ...)
3.  _gate_mode() → Ct vs θ → NORMAL/SURVIVAL transition (hysteresis applied)
4.  crisis_streak update (consecutive collapse-regime tick counter)
5.  Action decision: adopt user_action if provided, else auto-select by mode policy
6.  _origin_salvation() → u_t (stochastic, cooldown-controlled)
7.  _origin_attachment_message(act) → m_t → A_t  ← based on actual action
8.  apply_action(act)
9.  capacity decay (crisis_streak ≥ 2 + overload conditions) × CAPACITY_DECAY_SCALE, shield applied if u_t=1
10. If u_t=1: immediate hunger/irritation relief
11. state.clamp()
12. salvation_cooldown decrement
13. Y_t = _survival_proxy() → _surv_stats update
14. S_t = _trust_llr_proxy() (LLR, add-1 smoothing)
15. care_alpha += η·E_t·S_t + κ·A_t  (η=0.035, κ=0.020)
16. _sit_obs_features() → obs (hunger/curiosity/fatigue/anxiety/health_bad/skin_bad, normalized)
17. SITCore.step(obs, Ct, theta):
      a. compute_z_pref(obs, Ct) → z̃_t  (R^3 softmax over SAFE/GREEDY/REPAIR logits)
      b. update_z(z̃_t) → z_t  (EMA: z ← (1-λ)z + λz̃, λ=0.25)
      c. _update_sit(Ct, theta) → sit_event, reason, d_base
      d. _update_insight() → insight_now, insight_attained, insight_streak
18. If sit_event: append (t, reason) to sit_events
19. Return report
```

**Changes from v1.1 to v2 (SIT core separation):**

| Item | v1.1 | v2 |
|-----|-----|------|
| z_t space | R^5 (explore_drive, care_drive, survival_pressure, social_need, capacity_state) | R^3 MoE (SAFE, GREEDY, REPAIR) |
| z_t update | direct formula | EMA resistance: z ← (1-λ)z + λz̃, λ=0.25 |
| SIT detection | `_compute_z()` + `_detect_sit()` inside CATLMAgent | delegated to `SITCore` in sit_core.py |
| SIT constants | module-level `SIT_EPS`, `SIT_Z_STREAK_K`, `Z_DIM` | `SITConfig` fields |
| Insight | not defined | SAFE weight ≥ φ (default 0.7) persisting k steps |
| Serialization | `z_baseline`, `z_disp_streak` in snapshot | `sit_state` dict via `SITCore.dump_state()` |

### Collapse Forecast Intensity Ct Formula

Per the SIT paper (Section 3), Ct = 1 − P_surv(t + H) is approximated via weighted state signals:

```
Ct = hunger×0.25 + depression×0.20 + irritation×0.15
   + skin_degradation×0.15 + loneliness×0.15×(sociability_multiplier)
   + health_degradation×0.10
   + hunger×0.03×(appetite_multiplier)
   + boredom×0.05×(activity_multiplier)   ← activity_bonus
```

Collapse threshold θ: `0.5 - (cowardice_multiplier - 1.0) × 0.15`, range [0.25, 0.65]

### Irreversible Capacity Decay Structure

Per the SIT paper (Section 2), irreversible structural degradation (ΔW_irrev) permanently reduces the feasible inference topology. In CATLM this is modeled as:

```
capacity: 1.0 → 0.0 (unidirectional decay, no recovery)

capacity < 0.75 : EXPLORE, COSTUME unavailable
capacity < 0.55 : TRAIN additionally unavailable
capacity < 0.40 : GIFT additionally unavailable
capacity < 0.25 : only FEED / SNACK / PET / GROOM / IDLE / PLAY allowed
```

### SITCore: MoE Gating Logits

`SITCore.compute_z_pref()` computes SAFE / GREEDY / REPAIR logits from normalized obs features:

```
SAFE   logit = w_safe_ct×Ct + w_safe_threat×threat + w_safe_hunger×hunger
GREEDY logit = w_greedy_hunger×hunger + w_greedy_curiosity×curiosity + w_greedy_ct×Ct + w_greedy_threat×threat
REPAIR logit = w_repair_healthbad×health_bad + w_repair_skinbad×skin_bad + w_repair_ct×Ct + w_repair_fatigue×fatigue

threat = clamp(0.6×anxiety + 0.4×health_bad)
```

All obs features are normalized to [0, 1] by `_sit_obs_features()`. Capacity is **not** included in gating logits (capacity only restricts `_available_actions()`).

### Dialogue Generation System (Dialogue Bank)

#### File Structure (JSON)

```json
{
  "tokens": [
    {
      "id": "tok_001",
      "text": "배고파",
      "category": "demand-whine",
      "tone": ["complaint", "whine"],
      "intensity": "medium",
      "tags": ["hunger", "food"]
    }
  ],
  "emoticons": [
    {
      "emoji": "😺",
      "tone_weights": {"happy": 1.0, "recovery": 0.3}
    }
  ]
}
```

#### dialogue() Routing

```
dialogue_bank present → sample_dialogue(self)   ← 512-token path
dialogue_bank absent  → inline word_bank fallback (legacy)
```

#### sample_dialogue() Processing Flow

```
1. _tone_from_signals()      → state values + traits + Ct + care_alpha → tone probability distribution → softmax sampling
2. _derive_tags_from_agent() → state threshold exceeded + traits + care_alpha extremes → tag list (max 6)
3. tone_pool filtering       → extract bank._by_tone[tone] index
4. weight calculation        → intensity × collapse-regime flag + tag overlap × 0.35 + category bias + care_alpha relation-tag adjustment
5. _weighted_choice()        → sample 1st token
6. 2nd token (stochastic)    → 55% chance if Ct < 0.55, 25% chance otherwise for additional sentence
7. sample_emoticon()         → tone + collapse-regime flag + care_alpha → emoticon
```

#### Key Tone Weight Signals

| Signal | Affected Tones |
|-----|--------|
| Ct ≥ θ | danger↑↑, alert↑, complaint↑, happy↓, excited↓ |
| hunger↑ | complaint↑↑ |
| boredom↑ / loneliness↑ | whine↑↑ |
| fatigue↑ / depression↑ | lethargy↑↑ |
| excitement↑ / curiosity↑ | excited↑↑ |
| happiness↑ / satisfaction↑ | happy↑↑ |
| care_alpha↑ | happy↑, recovery↑, complaint↓ |
| care_alpha↓ | complaint↑, lethargy↑ |

## Variable Naming Conventions

| Symbol | Code Variable | Description |
|------|----------|------|
| αt | `care_alpha` | origin dependency weight (cumulative care trust); see SIT paper §5 |
| u_t | `u_t` (local in tick) | origin intervention signal (0 or 1) |
| m_t | `m_t` (local in tick) | origin attachment message |
| A_t | `A_t` (local in tick) | attachment integration intensity |
| Y_t | `Y` (local in tick) | short-term survival proxy (0 or 1) |
| S_t | `S_t` (local in tick) | trust LLR proxy |
| E_t | `E_t` (local in tick) | collapse + intervention event flag |
| z̃_t | `sit_out.z_pref` | instantaneous MoE preference (SAFE, GREEDY, REPAIR); see SIT paper §4 |
| z_t | `self.z` / `sit_out.z` | resistant strategy state (R^3, EMA-smoothed); see SIT paper §4 |
| Ct | return value of `crisis_score()` | collapse forecast intensity; see SIT paper §3 |
| θ | return value of `crisis_threshold()` | collapse threshold; see SIT paper §3 |

## Extension Guide

### Adding a New Trait
1. Add entry to `Trait` enum
2. Add field to `CatProfile`
3. Add mapping rule to `_trait_for_action_state()`
4. Reflect trait weight in `crisis_score()` or `crisis_threshold()`
5. Reflect trait-based tone weights in `_tone_from_signals()`

### Adding a New Action
1. Add entry to `Action` enum
2. Add `{State: label}` dictionary to `ACTIONS_MATRIX` (label must be one of the `IMPACT_SCALE` keys)
3. Determine capacity constraint applicability in `_available_actions()`
4. Review scoring in `_policy_normal()` / `_policy_survival()`

### Tuning SITCore Parameters
- Adjust `SITConfig` fields in `CATLMAgent.__init__` — no changes to `sit_core.py` needed
- `eps` sweep suggestion: {0.20, 0.25, 0.30} (R^3 max distance = √2 ≈ 1.41)
- To run a reversible control condition (H1 ablation): set `CAPACITY_DECAY_ON_CRISIS = 0` and `CAPACITY_DECAY_ON_OVERLOAD = 0` — `SITCore` stays identical across conditions
- `require_collapse_regime=False` makes SIT fire anywhere in state space (exploratory, not paper-default)

### Expanding the Word Pool
- Write `dialogue_bank_512.json` then call `attach_dialogue_bank(cat, path)`
- JSON schema: `tokens[].{id, text, category, tone[], intensity, tags[]}` + `emoticons[].{emoji, tone_weights{}}`
- 9 categories: positive-emotion / negative-emotion / neutral-action / demand-whine / alert-refusal / food-related / exploration-related / recovery-response / character-unique
- If `dialogue_bank` is absent, automatically falls back to legacy inline word_bank

### Extending Batch Replay
- Implement event-log-based replay by directly calling `step(user_action=action)`
- `tick(hours=n)` maintains legacy compatibility — delegates internally to `step(user_action=None, hours=n)`

## Notes

- `apply_action()` does **not** update `care_alpha` — updates occur only within `_tick_one()`
- `capacity` decays unidirectionally — recovery logic not yet implemented (planned for BM item integration)
- `capacity` loss only triggers after `crisis_streak >= CRISIS_STREAK_FOR_CAPACITY` — single collapse-regime ticks incur no loss
- `_gate_mode()` applies hysteresis — SURVIVAL entry and exit thresholds differ (`θ` vs `θ - 0.08`)
- `sit_events` is a list of `(t, reason_str)` tuples — recorded only when SIT conditions are met; `reason` includes d_base, disp_streak, Ct, theta, baseline, z
- `SITCore._disp_streak` resets to 0 after each confirmed SIT event (new attractor accepted)
- `insight_attained` latches to `True` once and never resets within a session (permanent milestone)
- `_trust_llr_proxy()` applies add-1 smoothing — LLR near 0 in early ticks is expected
- `DialogueBank._by_tone` and similar indexes start with underscore but are directly accessed in `sample_dialogue()` — not a public API
- `_tone_from_signals()` and `_pick_tone()` differ in logic: the former samples via softmax probability distribution, the latter uses if-else hard decisions
- `tick()` is the legacy API — use `step(user_action)` in new code
- Legacy snapshots (without `sit_state` key) trigger `SITCore.reset()` on load — z_t restarts from uniform (1/3, 1/3, 1/3)
