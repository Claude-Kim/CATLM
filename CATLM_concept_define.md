# CATLM Concept Definition

> **Cat Adaptive Tiny Language Model**
> SNAX Cats On-Device AI Character Adaptation System
> v1.0 | 2026

---

## 0. SNAX Cats Matrix Summary

| Item | Value | Meaning |
|------|----|------|
| Initial trait values | 4 traits × 5 levels | 625 unique cat personality combinations |
| State values | 16 states × 256 range | Emotional/physical states: hunger, boredom, irritation, etc. |
| User actions | 10 actions | Influence state values via a 10×16 matrix |
| Word pool | 512 tokens + 16 emoticons | Vocabulary pool for dialogue generation |

---

## 1. Definition

**CATLM** is an on-device lightweight AI-based character adaptation system for the mobile game SNAX Cats.

It analyzes the player's gameplay patterns in real time to dynamically generate each cat's state and dialogue. Designed as an optional enhancement layer independent of the core game loop, the game functions normally even when CATLM is disabled.

---

## 2. Purpose

- Reduce repetitive-play fatigue
- Build long-term retention through cumulative player–character relationships
- Differentiate individual cat personalities to strengthen collection motivation
- Implement re-engagement incentives via the collapse forecast system

---

## 3. Core Components

### 3-1. Initial Trait Values (4 traits × 5 levels)

Immutable values that define the unique personality of each cat. **625 combinations** possible in total.

| Trait | Description | Primary Influence |
|------|------|---------|
| Activity | Drive for movement and exploration | Intensity of exploration/play responses |
| Sociability | Affinity with the player and other cats | Social responses, loneliness sensitivity |
| Appetite | Intensity of desire for food and snacks | Intensity of food-related responses |
| Cowardice | Sensitivity and wariness toward stimuli | Collapse threshold θ |

---

### 3-2. State Values (16 states × 256 range)

The cat's current condition, varying in real time based on user actions and time elapsed.

| # | State | Primary Triggers |
|---|------|-----------|
| 1 | Hunger | Feeding missed, time elapsed |
| 2 | Boredom | Play missed, neglect |
| 3 | Fatigue | Excessive exploration or training |
| 4 | Irritation | Compounded accumulation of hunger + boredom |
| 5 | Depression | Prolonged neglect |
| 6 | Happiness | Accumulated positive interactions |
| 7 | Satisfaction | Needs fulfilled |
| 8 | Curiosity | Return from exploration, gift, new item |
| 9 | Aggression | Irritation + wariness exceeds threshold |
| 10 | Anxiety | Unfamiliar stimuli, cowardice trait response |
| 11 | Loneliness | Neglect; amplified in high-sociability cats |
| 12 | Self-esteem | Accumulated gifts, petting, praise |
| 13 | Health | Linked to feeding + grooming states |
| 14 | Skin condition | Degrades when grooming is missed |
| 15 | Wariness | Costume, unfamiliar placement, initial gift response |
| 16 | Excitement | Snack, return from exploration, events |

---

### 3-3. User Actions (10 actions)

Actions the player can perform on a cat. Each action is defined by a **10×16 matrix** of effects on state values.

| # | Action | Primary State Effects |
|---|------|-------------|
| 1 | FEED | hunger↓ satisfaction↑ health↑ |
| 2 | SNACK | happiness↑↑ excitement↑↑ |
| 3 | PLAY | boredom↓ happiness↑ fatigue↑ |
| 4 | GROOM | skin condition↑ health↑ anxiety↓ |
| 5 | PET | satisfaction↑ self-esteem↑ loneliness↓ |
| 6 | EXPLORE | curiosity↑↑ fatigue↑ |
| 7 | GIFT | self-esteem↑↑ curiosity↑ |
| 8 | TRAIN | self-esteem↑ health↑ (long-term) fatigue↑ |
| 9 | COSTUME | wariness↑↑ anxiety↑ excitement↑ |
| 10 | IDLE | hunger↑ boredom↑ depression↑ loneliness↑ |

---

### 3-4. 10×16 Action-State Matrix (Draft)

Impact labels use the scale: **+strong, +medium, +weak, -weak, -medium, -strong, 0 (no effect)**.

| Action↓ State→ | Hunger | Boredom | Fatigue | Irritation | Depression | Happiness | Satisfaction | Curiosity | Aggression | Anxiety | Loneliness | Self-esteem | Health | Skin | Wariness | Excitement |
|------------|--------|--------|------|------|------|------|------|--------|--------|------|--------|--------|------|------|--------|------|
| FEED | -strong | 0 | 0 | -weak | 0 | +weak | +medium | 0 | 0 | 0 | 0 | 0 | +weak | 0 | 0 | 0 |
| SNACK | -medium | 0 | 0 | -weak | 0 | +strong | +weak | 0 | 0 | 0 | 0 | +weak | 0 | 0 | 0 | +strong |
| PLAY | 0 | -strong | +weak | -medium | -weak | +medium | +weak | +weak | -weak | -weak | -medium | 0 | 0 | 0 | 0 | +medium |
| GROOM | 0 | 0 | 0 | -weak | 0 | +weak | +weak | 0 | 0 | -weak | 0 | 0 | +medium | +strong | 0 | 0 |
| PET | 0 | -weak | 0 | -weak | -weak | +weak | +medium | 0 | -weak | -weak | -weak | +medium | 0 | 0 | 0 | 0 |
| EXPLORE | 0 | -medium | +strong | 0 | 0 | 0 | 0 | +strong | 0 | +weak | 0 | 0 | 0 | 0 | +weak | +weak |
| GIFT | 0 | -weak | 0 | 0 | -weak | +medium | +weak | +strong | 0 | 0 | -weak | +strong | 0 | 0 | +weak | +medium |
| TRAIN | 0 | -medium | +medium | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | +medium | +medium | 0 | 0 | 0 |
| COSTUME | 0 | 0 | 0 | +weak | 0 | 0 | 0 | +weak | 0 | +medium | 0 | 0 | 0 | 0 | +strong | +medium |
| IDLE | +strong | +strong | 0 | +medium | +strong | -strong | -strong | 0 | +weak | +weak | +strong | -medium | -weak | -weak | 0 | 0 |

---

### 3-5. Trait Weight Rules

> **Final state change = base action value × trait multiplier**

The same action produces different response intensities depending on a cat's trait values.

**Per-trait application rules**

- **Activity (1–5)**: Applied to EXPLORE (curiosity, fatigue), PLAY (boredom reduction), IDLE (boredom/irritation accumulation).
- **Sociability (1–5)**: Applied to PET/PLAY (happiness, satisfaction), IDLE (loneliness, depression accumulation), GIFT (self-esteem).
- **Appetite (1–5)**: Applied to FEED/SNACK (happiness, excitement), IDLE (hunger accumulation), SNACK (satisfaction).
- **Cowardice (1–5)**: Applied to COSTUME (wariness, anxiety), GIFT (wariness toward unfamiliar stimuli), EXPLORE (anxiety accumulation), new cat placement (wariness).

```text
Trait value 1  →  ×0.6
Trait value 2  →  ×0.8
Trait value 3  →  ×1.0  (baseline)
Trait value 4  →  ×1.3
Trait value 5  →  ×1.6
```

**Examples**

- Cat with Appetite 5 + SNACK → excitement response ×1.6 (very strong reaction)
- Cat with Cowardice 5 + COSTUME → wariness ×1.6 (immediate hissing response possible)
- Cat with Sociability 5 + IDLE → loneliness accumulation rate ×1.6

---

## 4. Collapse Forecast System

### 4-1. Collapse Forecast Intensity Formula

Collapse forecast intensity Ct is computed not from a single state value, but as a composite threshold structure over multiple state values.

```text
Ct = collapse forecast intensity (0.0 ~ 1.0)
θ  = collapse threshold
Agent enters collapse regime when Ct ≥ θ
```

```text
Ct = (hunger × 0.25) + (depression × 0.20) + (irritation × 0.15)
   + (skin degradation × 0.15) + (loneliness × 0.15) + (health degradation × 0.10)

Ct range: 0.0 ~ 1.0
```

### 4-2. Collapse Stage Definitions

| Stage | Ct Range | Cat Response | Player Feedback |
|------|---------|-----------|---------|
| Normal | 0.0 ~ 0.3 | Standard behavior | No gauge shown |
| Caution | 0.3 ~ 0.5 | Crouching, leg trembling | Yellow gauge |
| Warning | 0.5 ~ 0.7 | Hissing, lying flat, skin color change | Orange flashing |
| Collapse | 0.7 ~ 1.0 | Hiding, aggression, illness | Red notification + BM integration |

### 4-3. Trait Linkage

- **Higher Cowardice** → collapse threshold θ decreases (enters collapse regime more easily)
- **Higher Sociability** → loneliness weight increases
- **Higher Appetite** → hunger weight increases
- **Higher Activity** → boredom additionally reflected in Ct

### 4-4. Collapse Resolution Conditions

| Stage | Exit Condition |
|------|----------|
| Caution → Normal | 1–2 key actions |
| Warning → Caution | 3–5 key actions + time elapsed |
| Collapse → Warning | Combined actions + recovery item use (BM integration) |

---

## 5. Dialogue Generation System

### 5-1. Processing Flow

```text
Step 1: Determine emotional tone (based on state value combination)
        ↓
Step 2: Apply speech filter (based on trait values)
        ↓
Step 3: Filter word pool (weighted selection from 512 tokens)
        ↓
Step 4: Select emoticon (tone-matched from 16 types)
        ↓
Output: 1–2 short dialogue sentences + emoticon
```

### 5-2. Tone Classification (8 types)

| Tone | Primary Triggers | Speech Characteristics |
|----|-----------|---------|
| Happy | happiness↑ satisfaction↑ | Positive, affectionate |
| Whine | boredom↑ loneliness↑ | Clingy, childlike |
| Complaint | hunger↑ irritation↑ | Blunt, demanding |
| Alert | wariness↑ anxiety↑ | Distancing, questioning |
| Lethargy | depression↑ fatigue↑ | Listless, unresponsive |
| Danger | Ct ≥ 0.5 | Hissing, aggressive warning |
| Excited | excitement↑ curiosity↑ | High-energy, overactive |
| Recovery | Action received after collapse | Cautious positivity |

### 5-3. Trait-Based Speech Filter (Secondary Filter)

| Trait | High Value | Low Value |
|------|--------|--------|
| Activity | Short, high-energy | Relaxed, unhurried |
| Sociability | Expressive, talkative | Reserved, brief responses |
| Appetite | Frequently mentions food | Indifferent to food |
| Cowardice | Passive, questioning | Bold, direct |

### 5-4. 512-Token Word Pool Categories

| Category | Token Count |
|---------|--------|
| Positive-emotion words | 80 |
| Negative-emotion words | 80 |
| Neutral-action words | 80 |
| Demand-whine words | 60 |
| Alert-refusal words | 60 |
| Food-related words | 40 |
| Exploration-related words | 40 |
| Recovery-response words | 40 |
| Character-unique words | 32 |

### 5-5. 16 Emoticon Distribution

| Emoticon | Linked Tone | Emoticon | Linked Tone |
|--------|--------|--------|--------|
| 😺 | Happy | 😿 | Depression, lethargy |
| 😸 | Excited | 😾 | Irritation, complaint |
| 😹 | Over-excited | 🐱 | Neutral, observing |
| 😻 | Peak satisfaction | 💤 | Fatigue, lying flat |
| 😼 | Arrogant, high self-esteem | ❤️ | High-sociability happiness |
| 😽 | Affectionate, petting response | 💢 | Aggression, danger |
| 🙀 | Startled, alert | 🐟 | Appetite-related |
| ✨ | Recovery, positive shift | ❓ | Cowardice, anxiety |

### 5-6. Cumulative Relationship Formation (αt)

As the player's care pattern accumulates, dialogue tone becomes personalized. This corresponds to the origin dependency weight αt defined in the SIT paper (§5).

```text
High αt (consistent care)
→ Recovery and positive word pool weights increase
→ e.g., "I think I can trust you now"

Low αt (irregular care)
→ Alert and whine word pool weights increase
→ e.g., "You'll just disappear again anyway"
```

---

## 6. Key Performance Indicators

| KPI | Target | Measurement Method |
|-----|--------|---------|
| Avg. session duration | +15% or more vs. without CATLM | A/B test, session log comparison |
| D7 retention | +5–8 pp vs. industry average | A/B test, 7-day return rate |
| Re-engagement rate for collapse-regime users | +20% or more vs. non-exposed users | Collapse stage entry event log analysis |

---

## 7. Risk Management

| Risk | Mitigation |
|--------|---------|
| AI implementation failure | CATLM is an optional enhancement layer — game functions normally when disabled |
| KPI underperformance | Phased A/B test validation, early parameter adjustment |
| On-device performance issues | Lightweight probability-weight-based filtering — designed to minimize compute overhead |
| Content quality degradation | Pre-curated word pool + QA review stage |

---

## 8. Theoretical Foundation

This system is grounded in the paper *Structural Inference Transitions Under Irreversible Survival Constraints* (pending arXiv publication).

The core concepts of the paper have been applied in lightweight form, adapted to the game context.

| Paper Concept | CATLM Application |
|---------|-----------|
| Collapse forecast intensity Ct | Collapse forecast gauge |
| Irreversible functional loss (ΔW_irrev) | Illness, depression, skin disease |
| Collapse threshold θ | Condition for entering the Collapse stage |
| Origin dependency weight αt | Cumulative player care pattern structure |

---

*CATLM Concept Definition Document v1.0 — SNAX Cats*

(Related documents)
- [[Part I __ Structural Inference Transitions Under Irreversible Survival Constraints]]
