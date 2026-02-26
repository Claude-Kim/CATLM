# CATLM — Claude Code Guide

## Project Overview

**Cat Adaptive Tiny Language Model** is an on-device lightweight AI character adaptation system for the mobile game *SNAX Cats*.
Theoretical foundation: *Structural Inference Transitions Under Irreversible Survival Constraints* (SIT paper, pending arXiv publication).

Key files:
- [catlm_simulator.py](catlm_simulator.py) — main reference implementation
- [CATLM_개념_정의서.md](CATLM_개념_정의서.md) — domain spec document

## Running

```bash
python3 catlm_simulator.py
```

Output columns: `mode  Ct  stage  dialogue  cap  alpha  z  |A|  theta  SIT#`

## Architecture

### Data Models

```
CatProfile     fixed trait values — activity / sociability / appetite / cowardice (each 1~5)
CatState       mutable state values — 16 fields, range 0~255, clamp() required
CATLMAgent     agent — above two models + simulation loop
Mode           NORMAL | SURVIVAL
DialogueToken  word token — id / text / category / tones / intensity / tags
EmoticonRule   emoticon rule — emoji / tone_weights
DialogueBank   512-token dialogue bank — JSON load, 3 indexes (_by_tone / _by_category / _by_tag)
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
| `SIT_PERSIST_K` | minimum persistence ticks required for SIT detection | 3 |
| `SIT_EPS` | latent inference configuration change threshold for SIT detection | 0.35 |
| `MODE_EXIT_HYSTERESIS` | SURVIVAL exit hysteresis margin | 0.08 |
| `CRISIS_STREAK_FOR_CAPACITY` | consecutive collapse-regime ticks required to trigger capacity loss | 2 |
| `_BASE_TONE_PRIOR` | prior distribution over 8 tones (all 1.0) | fixed |

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
  "action": str,                   # actual action name executed
  "action_source": "player"|"auto",
  "deltas": {state: int, ...},     # only non-zero state changes included
  "crisis": {"Ct", "theta", "stage", "streak"},
  "mode": "normal"|"survival",
  "capacity": {"before", "after", "loss"},
  "origin": {"u_t", "m_t", "A_t", "S_t", "Y"},
  "alpha": {"before", "after", "d_salvation", "d_attachment"},
  "sit_count": int,
}
```

### _tick_one() Execution Order (1 tick = 1 hour)

```
1. t++
2. Natural drift (hunger+6, boredom+5, loneliness+3×sociability_multiplier, ...)
3. _gate_mode() → Ct vs θ → NORMAL/SURVIVAL transition (hysteresis applied)
4. crisis_streak update (consecutive collapse-regime tick counter)
5. Action decision: adopt user_action if provided, else auto-select by mode policy
6. _origin_salvation() → u_t (stochastic, cooldown-controlled)
7. _origin_attachment_message(act) → m_t → A_t  ← based on actual action
8. apply_action(act)
9. capacity decay (crisis_streak ≥ 2 + overload conditions) × CAPACITY_DECAY_SCALE, shield applied if u_t=1
10. If u_t=1: immediate hunger/irritation relief
11. state.clamp()
12. salvation_cooldown decrement
13. Y_t = _survival_proxy() → _surv_stats update
14. S_t = _trust_llr_proxy() (LLR, add-1 smoothing)
15. care_alpha += η·E_t·S_t + κ·A_t  (η=0.035, κ=0.020)
16. z_t = (explore_drive, care_drive) update
17. SIT detection: mode transition + |Δz| > ε + persist ≥ k → record sit_events
18. Return report
```

**Changes from v1 to v1.1:**

| Item | v1 | v1.1 |
|-----|-----|------|
| SURVIVAL exit condition | `Ct < θ` | `Ct ≤ θ - 0.08` (hysteresis) |
| capacity loss trigger | immediately on each collapse-regime tick | triggers after 2+ consecutive collapse-regime ticks |
| capacity loss magnitude | default value as-is | × 0.65 scale applied |
| m_t computation basis | policy-predicted action | actual executed action |
| public API | `tick()` | `step(user_action)` added |

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
| z_t | `self.z` | latent inference configuration vector (explore_drive, care_drive); see SIT paper §4 |
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
- `_update_care_alpha()` is currently unused (legacy, pre-tick() integration implementation)
- `capacity` decays unidirectionally — recovery logic not yet implemented (planned for BM item integration)
- `capacity` loss only triggers after `crisis_streak >= CRISIS_STREAK_FOR_CAPACITY` — single collapse-regime ticks incur no loss
- `_gate_mode()` applies hysteresis — SURVIVAL entry and exit thresholds differ (`θ` vs `θ - 0.08`)
- `sit_events` is a list of `(t, old_mode, new_mode)` tuples — recorded only when SIT conditions are met (mode transition + |Δz| > ε + persist ≥ k)
- `_trust_llr_proxy()` applies add-1 smoothing — LLR near 0 in early ticks is expected
- `import math` / `import json` located at top of file
- `DialogueBank._by_tone` and similar indexes start with underscore but are directly accessed in `sample_dialogue()` — not a public API
- `_tone_from_signals()` and `_pick_tone()` differ in logic: the former samples via softmax probability distribution, the latter uses if-else hard decisions
- `tick()` is the legacy API — use `step(user_action)` in new code
