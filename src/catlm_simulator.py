from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Dict, List, Optional, Tuple


class Trait(str, Enum):
    ACTIVITY = "activity"
    SOCIABILITY = "sociability"
    APPETITE = "appetite"
    COWARDICE = "cowardice"


class State(str, Enum):
    HUNGER = "hunger"
    BOREDOM = "boredom"
    FATIGUE = "fatigue"
    IRRITATION = "irritation"
    DEPRESSION = "depression"
    HAPPINESS = "happiness"
    SATISFACTION = "satisfaction"
    CURIOSITY = "curiosity"
    AGGRESSION = "aggression"
    ANXIETY = "anxiety"
    LONELINESS = "loneliness"
    SELF_ESTEEM = "self_esteem"
    HEALTH = "health"
    SKIN = "skin"
    VIGILANCE = "vigilance"
    EXCITEMENT = "excitement"


class Action(str, Enum):
    FEED = "feed"
    SNACK = "snack"
    PLAY = "play"
    GROOM = "groom"
    PET = "pet"
    EXPLORE = "explore"
    GIFT = "gift"
    TRAIN = "train"
    COSTUME = "costume"
    IDLE = "idle"


class Mode(str, Enum):
    NORMAL = "normal"
    SURVIVAL = "survival"


# irreversible capacity settings (v1)
CAPACITY_INIT = 1.0
CAPACITY_MIN = 0.0
CAPACITY_DECAY_ON_CRISIS = 0.06     # irreversible loss per crisis-tick
CAPACITY_DECAY_ON_OVERLOAD = 0.03   # small irreversible loss when fatigue/irritation very high

# salvation settings (origin intervention u_t)
SALVATION_COOLDOWN_HOURS = 6
SALVATION_CAPACITY_SHIELD = 0.08    # reduces irreversible capacity loss for one tick when u_t=1
SALVATION_SUCCESS_BOOST = 0.12      # increases short-horizon survival proxy when u_t=1

# attachment settings (origin message m_t)
ATTACHMENT_GAIN = 0.015            # cumulative integration strength

# SIT detection settings
SIT_PERSIST_K = 3
SIT_EPS = 0.35  # for z_{t+1} - z_t magnitude threshold (proxy)

# game-stability knobs (v1.1)
MODE_EXIT_HYSTERESIS = 0.08          # SURVIVAL→NORMAL hysteresis band
CRISIS_STREAK_FOR_CAPACITY = 2       # min consecutive crisis ticks before capacity decays
CAPACITY_DECAY_SCALE = 0.65          # multiplier applied to all capacity losses (tune < 1 to slow decay)


IMPACT_SCALE: Dict[str, int] = {
    "+강": 36,
    "+중": 24,
    "+약": 12,
    "-약": -12,
    "-중": -24,
    "-강": -36,
    "0": 0,
}

TRAIT_MULTIPLIER = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6}

ACTIONS_MATRIX: Dict[Action, Dict[State, str]] = {
    Action.FEED: {
        State.HUNGER: "-강", State.IRRITATION: "-약", State.HAPPINESS: "+약", State.SATISFACTION: "+중", State.HEALTH: "+약"
    },
    Action.SNACK: {
        State.HUNGER: "-중", State.IRRITATION: "-약", State.HAPPINESS: "+강", State.SATISFACTION: "+약", State.SELF_ESTEEM: "+약", State.EXCITEMENT: "+강"
    },
    Action.PLAY: {
        State.BOREDOM: "-강", State.FATIGUE: "+약", State.IRRITATION: "-중", State.DEPRESSION: "-약", State.HAPPINESS: "+중", State.SATISFACTION: "+약", State.CURIOSITY: "+약", State.AGGRESSION: "-약", State.ANXIETY: "-약", State.LONELINESS: "-중", State.EXCITEMENT: "+중"
    },
    Action.GROOM: {
        State.IRRITATION: "-약", State.HAPPINESS: "+약", State.SATISFACTION: "+약", State.ANXIETY: "-약", State.HEALTH: "+중", State.SKIN: "+강"
    },
    Action.PET: {
        State.BOREDOM: "-약", State.IRRITATION: "-약", State.DEPRESSION: "-약", State.HAPPINESS: "+약", State.SATISFACTION: "+중", State.AGGRESSION: "-약", State.ANXIETY: "-약", State.LONELINESS: "-약", State.SELF_ESTEEM: "+중"
    },
    Action.EXPLORE: {
        State.BOREDOM: "-중", State.FATIGUE: "+강", State.CURIOSITY: "+강", State.ANXIETY: "+약", State.VIGILANCE: "+약", State.EXCITEMENT: "+약"
    },
    Action.GIFT: {
        State.BOREDOM: "-약", State.DEPRESSION: "-약", State.HAPPINESS: "+중", State.SATISFACTION: "+약", State.CURIOSITY: "+강", State.LONELINESS: "-약", State.SELF_ESTEEM: "+강", State.VIGILANCE: "+약", State.EXCITEMENT: "+중"
    },
    Action.TRAIN: {
        State.BOREDOM: "-중", State.FATIGUE: "+중", State.SELF_ESTEEM: "+중", State.HEALTH: "+중"
    },
    Action.COSTUME: {
        State.IRRITATION: "+약", State.CURIOSITY: "+약", State.ANXIETY: "+중", State.VIGILANCE: "+강", State.EXCITEMENT: "+중"
    },
    Action.IDLE: {
        State.HUNGER: "+강", State.BOREDOM: "+강", State.IRRITATION: "+중", State.DEPRESSION: "+강", State.HAPPINESS: "-강", State.SATISFACTION: "-강", State.AGGRESSION: "+약", State.ANXIETY: "+약", State.LONELINESS: "+강", State.SELF_ESTEEM: "-중", State.HEALTH: "-약", State.SKIN: "-약"
    },
}


# ----------------------------
# Dialogue bank structures
# ----------------------------

@dataclass(frozen=True)
class DialogueToken:
    id: str
    text: str
    category: str
    tones: Tuple[str, ...]
    intensity: str                 # "약" | "중" | "강"
    tags: Tuple[str, ...]


@dataclass(frozen=True)
class EmoticonRule:
    emoji: str
    tone_weights: Dict[str, float]


class DialogueBank:
    def __init__(self, tokens: List[DialogueToken], emoticons: List[EmoticonRule], meta: Dict[str, Any]):
        self.tokens = tokens
        self.emoticons = emoticons
        self.meta = meta

        self._by_tone: Dict[str, List[int]] = {}
        self._by_category: Dict[str, List[int]] = {}
        self._by_tag: Dict[str, List[int]] = {}

        for i, tok in enumerate(tokens):
            for t in tok.tones:
                self._by_tone.setdefault(t, []).append(i)
            self._by_category.setdefault(tok.category, []).append(i)
            for tg in tok.tags:
                self._by_tag.setdefault(tg, []).append(i)

    @staticmethod
    def load_from_json(path: str) -> "DialogueBank":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens: List[DialogueToken] = []
        for t in data["tokens"]:
            tokens.append(
                DialogueToken(
                    id=str(t["id"]),
                    text=str(t["text"]),
                    category=str(t["category"]),
                    tones=tuple(t["tone"]),
                    intensity=str(t["intensity"]),
                    tags=tuple(t.get("tags", [])),
                )
            )

        emoticons: List[EmoticonRule] = []
        for e in data.get("emoticons", []):
            emoticons.append(
                EmoticonRule(
                    emoji=str(e["emoji"]),
                    tone_weights=dict(e.get("tone_weights", {})),
                )
            )

        meta = {k: v for k, v in data.items() if k not in {"tokens", "emoticons"}}
        return DialogueBank(tokens=tokens, emoticons=emoticons, meta=meta)


# ----------------------------
# Weighting utilities
# ----------------------------

_INTENSITY_WEIGHT = {"약": 0.9, "중": 1.0, "강": 1.1}

_BASE_TONE_PRIOR = {
    "행복": 1.0, "투정": 1.0, "불평": 1.0, "경계": 1.0,
    "무기력": 1.0, "위험": 1.0, "흥분": 1.0, "회복": 1.0
}


def _safe_softmax(weights: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    keys = list(weights.keys())
    xs = [weights[k] / max(1e-9, temperature) for k in keys]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) if exps else 1.0
    return {k: exps[i] / s for i, k in enumerate(keys)}


def _weighted_choice(rng: random.Random, items: List[Any], weights: List[float]) -> Any:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += max(0.0, w)
        if acc >= r:
            return item
    return items[-1]


# ----------------------------
# CATLMAgent dialogue integration
# ----------------------------

def attach_dialogue_bank(agent: "CATLMAgent", bank_path: str) -> None:
    """Call once after CATLMAgent is created to enable 512-token dialogue."""
    agent.dialogue_bank = DialogueBank.load_from_json(bank_path)


def _tone_from_signals(agent: "CATLMAgent") -> str:
    s = agent.state.values
    Ct = agent.crisis_score()
    theta = agent.crisis_threshold()

    w = dict(_BASE_TONE_PRIOR)

    if Ct >= theta:
        w["위험"] *= 2.4
        w["경계"] *= 1.6
        w["불평"] *= 1.3
        w["무기력"] *= 1.2
        w["행복"] *= 0.5
        w["흥분"] *= 0.6
    else:
        w["회복"] *= 1.2

    hunger = s[State.HUNGER] / 255
    boredom = s[State.BOREDOM] / 255
    fatigue = s[State.FATIGUE] / 255
    irritation = s[State.IRRITATION] / 255
    depression = s[State.DEPRESSION] / 255
    happiness = s[State.HAPPINESS] / 255
    satisfaction = s[State.SATISFACTION] / 255
    anxiety = s[State.ANXIETY] / 255
    loneliness = s[State.LONELINESS] / 255
    curiosity = s[State.CURIOSITY] / 255
    excitement = s[State.EXCITEMENT] / 255
    vigilance = s[State.VIGILANCE] / 255

    w["불평"] *= (1.0 + 1.6 * hunger + 1.2 * irritation)
    w["투정"] *= (1.0 + 1.5 * boredom + 1.8 * loneliness)
    w["무기력"] *= (1.0 + 1.7 * fatigue + 1.5 * depression)
    w["경계"] *= (1.0 + 1.6 * vigilance + 1.6 * anxiety)
    w["위험"] *= (1.0 + 2.0 * max(0.0, Ct - 0.5))
    w["행복"] *= (1.0 + 2.0 * happiness + 1.6 * satisfaction)
    w["흥분"] *= (1.0 + 1.8 * excitement + 1.2 * curiosity)
    w["회복"] *= (1.0 + 1.0 * (1.0 - Ct))

    sociability = TRAIT_MULTIPLIER[agent.profile.sociability]
    activity = TRAIT_MULTIPLIER[agent.profile.activity]
    appetite = TRAIT_MULTIPLIER[agent.profile.appetite]
    cowardice = TRAIT_MULTIPLIER[agent.profile.cowardice]

    w["투정"] *= (0.9 + 0.4 * sociability)
    w["행복"] *= (0.9 + 0.3 * sociability)
    w["흥분"] *= (0.9 + 0.4 * activity)
    w["경계"] *= (0.9 + 0.45 * cowardice)
    w["불평"] *= (0.9 + 0.35 * appetite)

    a = max(0.0, min(1.0, agent.care_alpha))
    w["행복"] *= (0.8 + 0.8 * a)
    w["회복"] *= (0.8 + 0.9 * a)
    w["불평"] *= (1.2 - 0.6 * a)
    w["무기력"] *= (1.1 - 0.4 * a)

    probs = _safe_softmax(w, temperature=1.0)
    tones = list(probs.keys())
    return _weighted_choice(agent.rng, tones, [probs[t] for t in tones])


def _derive_tags_from_agent(agent: "CATLMAgent") -> List[str]:
    s = agent.state.values
    tags: List[str] = []

    if s[State.HUNGER] > 165:
        tags += ["배고픔", "먹이"]
    if s[State.LONELINESS] > 165:
        tags += ["외로움", "관계"]
    if s[State.BOREDOM] > 165:
        tags += ["심심함", "놀이"]
    if s[State.ANXIETY] > 150 or s[State.VIGILANCE] > 150:
        tags += ["불안", "경계"]
    if s[State.DEPRESSION] > 160 or s[State.FATIGUE] > 170:
        tags += ["무기력", "휴식"]
    if s[State.IRRITATION] > 165:
        tags += ["짜증", "불편"]

    if agent.profile.appetite >= 4:
        tags += ["먹이"]
    if agent.profile.activity >= 4:
        tags += ["탐험"]
    if agent.profile.cowardice >= 4:
        tags += ["경계"]

    if agent.care_alpha > 0.7:
        tags += ["신뢰", "안정"]
    elif agent.care_alpha < 0.3:
        tags += ["거리", "경고"]

    seen: set = set()
    uniq: List[str] = []
    for t in tags:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:6]


def sample_dialogue(agent: "CATLMAgent") -> str:
    """Produce 1-2 sentences + emoticon from the 512-token bank."""
    if not hasattr(agent, "dialogue_bank"):
        raise RuntimeError("Dialogue bank not attached. Call attach_dialogue_bank(agent, bank_path) first.")

    bank: DialogueBank = agent.dialogue_bank
    tone = _tone_from_signals(agent)
    tags = _derive_tags_from_agent(agent)

    tone_pool = bank._by_tone.get(tone, list(range(len(bank.tokens))))

    Ct = agent.crisis_score()
    theta = agent.crisis_threshold()
    in_crisis = Ct >= theta

    weights: List[float] = []
    items: List[int] = []

    for idx in tone_pool:
        tok = bank.tokens[idx]
        w = 1.0

        if in_crisis:
            w *= {"강": 1.25, "중": 1.10, "약": 0.90}.get(tok.intensity, 1.0)
        else:
            w *= {"약": 1.10, "중": 1.00, "강": 0.95}.get(tok.intensity, 1.0)

        overlap = sum(1 for t in tags if t in tok.tags)
        if overlap > 0:
            w *= (1.0 + 0.35 * overlap)

        if tok.category in {"경계거부어", "부정감정어"} and not in_crisis:
            w *= 0.92
        if tok.category in {"긍정감정어"} and in_crisis:
            w *= 0.85

        a = max(0.0, min(1.0, agent.care_alpha))
        if "관계" in tok.tags or "신뢰" in tok.tags:
            w *= (0.9 + 0.7 * a)
        if "거리" in tok.tags or "경고" in tok.tags:
            w *= (1.0 + 0.7 * (0.5 - a))

        weights.append(w)
        items.append(idx)

    idx1 = _weighted_choice(agent.rng, items, weights)
    tok1 = bank.tokens[idx1]

    p_second = 0.55 if Ct < 0.55 else 0.25
    second_text = ""
    if agent.rng.random() < p_second:
        weights2: List[float] = []
        items2: List[int] = []
        for idx, w0 in zip(items, weights):
            if idx == idx1:
                continue
            tok = bank.tokens[idx]
            new_tag_bonus = 1.0
            if tags:
                has_new = any(t in tok.tags for t in tags) and not any(t in tok1.tags for t in tags)
                if has_new:
                    new_tag_bonus = 1.15
            weights2.append(w0 * new_tag_bonus * 0.9)
            items2.append(idx)

        if items2:
            idx2 = _weighted_choice(agent.rng, items2, weights2)
            second_text = bank.tokens[idx2].text

    emoji = sample_emoticon(agent, tone=tone)

    if second_text:
        return f"{tok1.text} {second_text} {emoji}".strip()
    return f"{tok1.text} {emoji}".strip()


def sample_emoticon(agent: "CATLMAgent", tone: Optional[str] = None) -> str:
    if not hasattr(agent, "dialogue_bank"):
        return "🐱"
    bank: DialogueBank = agent.dialogue_bank
    if tone is None:
        tone = _tone_from_signals(agent)

    Ct = agent.crisis_score()
    theta = agent.crisis_threshold()
    in_crisis = Ct >= theta

    emojis = [e.emoji for e in bank.emoticons] or ["🐱"]
    if not bank.emoticons:
        return "🐱"

    weights: List[float] = []
    for e in bank.emoticons:
        w = 0.05 + e.tone_weights.get(tone, 0.0)
        if in_crisis:
            if e.emoji in {"💢", "🙀", "😾", "❓"}:
                w *= 1.35
            if e.emoji in {"❤️", "😻"}:
                w *= 0.70
        else:
            a = max(0.0, min(1.0, agent.care_alpha))
            if e.emoji in {"❤️", "😻", "😽", "✨"}:
                w *= (0.9 + 0.5 * a)
        weights.append(w)

    return _weighted_choice(agent.rng, emojis, weights)


@dataclass
class CatProfile:
    name: str
    activity: int
    sociability: int
    appetite: int
    cowardice: int

    def as_dict(self) -> Dict[Trait, int]:
        return {
            Trait.ACTIVITY: self.activity,
            Trait.SOCIABILITY: self.sociability,
            Trait.APPETITE: self.appetite,
            Trait.COWARDICE: self.cowardice,
        }


@dataclass
class CatState:
    values: Dict[State, int] = field(default_factory=lambda: {
        State.HUNGER: 70,
        State.BOREDOM: 70,
        State.FATIGUE: 60,
        State.IRRITATION: 65,
        State.DEPRESSION: 60,
        State.HAPPINESS: 160,
        State.SATISFACTION: 155,
        State.CURIOSITY: 120,
        State.AGGRESSION: 45,
        State.ANXIETY: 60,
        State.LONELINESS: 75,
        State.SELF_ESTEEM: 120,
        State.HEALTH: 185,
        State.SKIN: 180,
        State.VIGILANCE: 70,
        State.EXCITEMENT: 110,
    })

    def clamp(self) -> None:
        for key, value in self.values.items():
            self.values[key] = max(0, min(255, int(round(value))))


class CATLMAgent:
    def __init__(self, profile: CatProfile, rng_seed: int | None = None):
        self.profile = profile
        self.state = CatState()
        self.rng = random.Random(rng_seed)

        # origin-weight (alpha candidate)
        self.care_alpha = 0.5

        # irreversible feasible-set reduction proxy
        self.capacity = CAPACITY_INIT  # 1.0 -> full capacity, 0.0 -> collapsed capability
        self.salvation_cooldown = 0

        # gating / SIT logging
        self.mode: Mode = Mode.NORMAL
        self.mode_persist = 0
        self.sit_events: List[Tuple[int, Mode, Mode]] = []  # (t, old, new)

        # simple latent config proxy z_t (vector in R^2) to detect transition magnitude
        self.z = (0.5, 0.5)  # (explore_drive, care_drive)

        # time index (for logging)
        self.t = 0

        # consecutive crisis-tick counter (v1.1: capacity only decays after CRISIS_STREAK_FOR_CAPACITY ticks)
        self.crisis_streak = 0

        # running trust stats (for S_t proxy)
        self._surv_stats = {
            "u1": {"n": 0, "alive": 0},
            "u0": {"n": 0, "alive": 0},
        }

    def _available_actions(self) -> List[Action]:
        """Feasible action set shrinks irreversibly with capacity loss."""
        base = [a for a in Action]
        if self.capacity < 0.75:
            # lose 'fun/novelty' first
            base = [a for a in base if a not in {Action.EXPLORE, Action.COSTUME}]
        if self.capacity < 0.55:
            base = [a for a in base if a not in {Action.TRAIN}]
        if self.capacity < 0.40:
            base = [a for a in base if a not in {Action.GIFT}]
        if self.capacity < 0.25:
            # survival-only minimal set remains
            base = [a for a in base if a in {Action.FEED, Action.SNACK, Action.PET, Action.GROOM, Action.IDLE, Action.PLAY}]
        return base

    def _policy_normal(self) -> Action:
        """Normal mode: pursue curiosity/affect balancing."""
        s = self.state.values
        candidates = self._available_actions()
        # light heuristic scoring
        scores: Dict[Action, float] = {a: 0.0 for a in candidates}
        for a in candidates:
            m = ACTIONS_MATRIX[a]
            # bias toward reducing boredom/loneliness and increasing happiness
            scores[a] += (m.get(State.BOREDOM, "0").startswith("-")) * (s[State.BOREDOM] / 255) * 0.8
            scores[a] += (m.get(State.LONELINESS, "0").startswith("-")) * (s[State.LONELINESS] / 255) * 0.8
            scores[a] += (m.get(State.CURIOSITY, "0") in {"+중", "+강"}) * 0.4
            scores[a] += (m.get(State.HAPPINESS, "0") in {"+중", "+강"}) * 0.5
            scores[a] -= (m.get(State.FATIGUE, "0") in {"+중", "+강"}) * (s[State.FATIGUE] / 255) * 0.6
        return max(scores, key=scores.get)

    def _policy_survival(self) -> Action:
        """Survival mode: minimize collapse drivers."""
        s = self.state.values
        candidates = self._available_actions()
        scores: Dict[Action, float] = {a: 0.0 for a in candidates}
        for a in candidates:
            m = ACTIONS_MATRIX[a]
            # prioritize hunger/health/irritation control
            scores[a] += (m.get(State.HUNGER, "0") in {"-중", "-강"}) * (s[State.HUNGER] / 255) * 1.2
            scores[a] += (m.get(State.HEALTH, "0") in {"+약", "+중", "+강"}) * ((255 - s[State.HEALTH]) / 255) * 1.0
            scores[a] += (m.get(State.IRRITATION, "0") in {"-중", "-강"}) * (s[State.IRRITATION] / 255) * 0.8
            scores[a] += (m.get(State.ANXIETY, "0") in {"-약", "-중", "-강"}) * (s[State.ANXIETY] / 255) * 0.6
            # avoid fatigue spikes under low capacity
            if self.capacity < 0.6 and m.get(State.FATIGUE, "0") in {"+중", "+강"}:
                scores[a] -= 1.0
        return max(scores, key=scores.get) if scores else Action.IDLE

    def _gate_mode(self) -> Tuple[Mode, float, float]:
        """Gating based on collapse forecast Ct and threshold theta, with hysteresis on exit (v1.1)."""
        Ct = self.crisis_score()
        theta = self.crisis_threshold()
        if self.mode == Mode.SURVIVAL:
            # require Ct to drop below (theta - hysteresis) before returning to NORMAL
            exit_theta = max(0.0, theta - MODE_EXIT_HYSTERESIS)
            new_mode = Mode.NORMAL if Ct <= exit_theta else Mode.SURVIVAL
        else:
            new_mode = Mode.SURVIVAL if Ct >= theta else Mode.NORMAL
        return new_mode, Ct, theta

    def _origin_salvation(self, Ct: float, theta: float) -> int:
        """Salvation signal u_t triggers only in collapse regime and when cooldown allows."""
        if self.salvation_cooldown > 0:
            return 0
        if Ct >= theta:
            # make it probabilistic: more likely when Ct is high, and when care_alpha is high
            p = min(0.95, 0.25 + 0.6 * Ct + 0.2 * (self.care_alpha - 0.5))
            u = 1 if self.rng.random() < p else 0
            if u == 1:
                self.salvation_cooldown = SALVATION_COOLDOWN_HOURS
            return u
        return 0

    def _origin_attachment_message(self, action: Action) -> str:
        """Attachment channel m_t: non-lethal contextual exchange independent of immediate survival."""
        # Keep it simple: messages exist even outside crisis; richer when caring interactions happen.
        if action in {Action.PET, Action.GROOM, Action.GIFT, Action.PLAY}:
            return self.rng.choice(["함께 있는 시간", "손길의 기억", "약속", "루틴", "안정감"])
        return self.rng.choice(["주변 소리", "새로운 냄새", "시간의 흐름", "공간의 패턴", "조용한 생각"])

    def _attachment_proxy(self, m_t: str) -> float:
        """A_t proxy (v1): cumulative integration strength, boosted by semantic coherence."""
        # toy proxy: consistent tokens raise integration, shuffled/noise would lower in ablation
        coherent = 1.0 if m_t in {"함께 있는 시간", "손길의 기억", "약속", "루틴", "안정감"} else 0.6
        return ATTACHMENT_GAIN * coherent

    def _survival_proxy(self) -> int:
        """Binary proxy for short-horizon survival (Y_t). Here: not in '붕괴' stage."""
        return 0 if self.crisis_stage() == "붕괴" else 1

    def _trust_llr_proxy(self) -> float:
        """S_t proxy: log-likelihood ratio from running survival stats under u=1 vs u=0."""
        # add-one smoothing to avoid div0
        u1 = self._surv_stats["u1"]
        u0 = self._surv_stats["u0"]
        p1 = (u1["alive"] + 1) / (u1["n"] + 2)
        p0 = (u0["alive"] + 1) / (u0["n"] + 2)
        return math.log(p1 / p0)

    def _trait_for_action_state(self, action: Action, state: State) -> Trait | None:
        if state in {State.CURIOSITY, State.FATIGUE, State.BOREDOM} and action in {Action.EXPLORE, Action.PLAY, Action.IDLE}:
            return Trait.ACTIVITY
        if state in {State.HAPPINESS, State.SATISFACTION, State.LONELINESS, State.DEPRESSION, State.SELF_ESTEEM} and action in {Action.PET, Action.PLAY, Action.GIFT, Action.IDLE}:
            return Trait.SOCIABILITY
        if state in {State.HUNGER, State.HAPPINESS, State.EXCITEMENT, State.SATISFACTION} and action in {Action.FEED, Action.SNACK, Action.IDLE}:
            return Trait.APPETITE
        if state in {State.VIGILANCE, State.ANXIETY} and action in {Action.COSTUME, Action.EXPLORE, Action.GIFT}:
            return Trait.COWARDICE
        return None

    def apply_action(self, action: Action) -> Dict[State, int]:
        deltas: Dict[State, int] = {state: 0 for state in State}
        action_map = ACTIONS_MATRIX[action]

        for state, level in action_map.items():
            base_delta = IMPACT_SCALE[level]
            trait_kind = self._trait_for_action_state(action, state)
            multiplier = 1.0
            if trait_kind is not None:
                trait_value = self.profile.as_dict()[trait_kind]
                multiplier = TRAIT_MULTIPLIER[trait_value]
            delta = int(round(base_delta * multiplier))
            deltas[state] = delta
            self.state.values[state] += delta

        self._apply_derived_effects()
        self.state.clamp()

        # NOTE: care_alpha update moved to tick() where we can incorporate salvation/trust/attachment
        return deltas

    def _apply_derived_effects(self) -> None:
        if self.state.values[State.IRRITATION] > 180 and self.state.values[State.VIGILANCE] > 170:
            self.state.values[State.AGGRESSION] += 12
        if self.state.values[State.HUNGER] > 170 and self.state.values[State.SKIN] < 80:
            self.state.values[State.HEALTH] -= 10

    def tick(self, hours: int = 1) -> None:
        """Backward-compatible wrapper: autonomous tick(s), discards the report."""
        self.step(user_action=None, hours=hours)

    def step(self, user_action: Optional[Action], hours: int = 1) -> Dict[str, Any]:
        """Run one or more ticks, optionally injecting a player action on the first tick.

        Returns the report dict from the last tick.
        """
        report: Dict[str, Any] = {}
        for i in range(hours):
            report = self._tick_one(user_action=user_action if i == 0 else None)
        return report

    def _tick_one(self, user_action: Optional[Action] = None) -> Dict[str, Any]:
        """Execute one hour-tick. Returns a rich report dict."""
        self.t += 1

        # 1. natural drift
        self.state.values[State.HUNGER] += 6
        self.state.values[State.BOREDOM] += 5
        self.state.values[State.LONELINESS] += int(3 * TRAIT_MULTIPLIER[self.profile.sociability])
        self.state.values[State.DEPRESSION] += 2
        self.state.values[State.EXCITEMENT] -= 6
        self.state.values[State.HAPPINESS] -= 4
        self.state.values[State.SATISFACTION] -= 3

        # 2. gating — with hysteresis on SURVIVAL→NORMAL exit (v1.1)
        new_mode, Ct, theta = self._gate_mode()
        old_mode = self.mode

        if new_mode == self.mode:
            self.mode_persist += 1
        else:
            self.mode = new_mode
            self.mode_persist = 1

        # 3. crisis streak counter (v1.1)
        if Ct >= theta:
            self.crisis_streak += 1
        else:
            self.crisis_streak = 0

        # 4. action selection (player override or autonomous policy)
        if user_action is not None:
            act = user_action
        elif self.mode == Mode.SURVIVAL:
            act = self._policy_survival()
        else:
            act = self._policy_normal()

        # 5. origin channels
        u_t = self._origin_salvation(Ct, theta)
        m_t = self._origin_attachment_message(act)
        A_t = self._attachment_proxy(m_t)

        # 6. irreversible capacity loss — only after CRISIS_STREAK_FOR_CAPACITY consecutive crisis ticks (v1.1)
        cap_loss = 0.0
        if Ct >= theta and self.crisis_streak >= CRISIS_STREAK_FOR_CAPACITY:
            cap_loss += CAPACITY_DECAY_ON_CRISIS
        if self.state.values[State.FATIGUE] > 175 or self.state.values[State.IRRITATION] > 175:
            cap_loss += CAPACITY_DECAY_ON_OVERLOAD

        # salvation shields capacity loss
        if u_t == 1:
            cap_loss = max(0.0, cap_loss - SALVATION_CAPACITY_SHIELD)

        # apply capacity loss with global scale knob (v1.1)
        if cap_loss > 0:
            self.capacity = max(CAPACITY_MIN, self.capacity - cap_loss * CAPACITY_DECAY_SCALE)

        # 7. apply selected action
        self.apply_action(act)

        # 8. salvation immediate effect: reduce hunger/irritation
        if u_t == 1:
            self.state.values[State.HUNGER] = max(0, self.state.values[State.HUNGER] - int(18 * (1.0 + self.care_alpha)))
            self.state.values[State.IRRITATION] = max(0, self.state.values[State.IRRITATION] - 10)

        # 9. clamp
        self.state.clamp()

        # 10. salvation cooldown tick-down
        if self.salvation_cooldown > 0:
            self.salvation_cooldown -= 1

        # 11. trust stats + care_alpha update
        Y = self._survival_proxy()
        key = "u1" if (Ct >= theta and u_t == 1) else "u0"
        self._surv_stats[key]["n"] += 1
        self._surv_stats[key]["alive"] += Y

        S_t = self._trust_llr_proxy()
        E_t = 1 if (Ct >= theta and u_t == 1) else 0

        eta_alpha = 0.035
        kappa_alpha = 0.020
        self.care_alpha = max(0.0, min(1.0, self.care_alpha + eta_alpha * E_t * S_t + kappa_alpha * A_t))

        # 12. latent config proxy z_t update
        explore_drive = min(1.0, max(0.0, 0.6 * (self.state.values[State.CURIOSITY] / 255) + 0.4 * (self.state.values[State.EXCITEMENT] / 255)))
        care_drive = min(1.0, max(0.0, 0.5 * self.care_alpha + 0.5 * (1.0 - Ct)))
        z_new = (explore_drive, care_drive)

        # 13. SIT detection
        dz = ((z_new[0] - self.z[0]) ** 2 + (z_new[1] - self.z[1]) ** 2) ** 0.5
        if old_mode != self.mode and dz > SIT_EPS and self.mode_persist >= SIT_PERSIST_K:
            self.sit_events.append((self.t, old_mode, self.mode))

        self.z = z_new

        return {
            "t": self.t,
            "mode": self.mode.value,
            "Ct": Ct,
            "theta": theta,
            "stage": self.crisis_stage(),
            "action": act.value,
            "u_t": u_t,
            "Y": Y,
            "S_t": round(S_t, 4),
            "E_t": E_t,
            "A_t": round(A_t, 4),
            "alpha": round(self.care_alpha, 4),
            "capacity": round(self.capacity, 4),
            "crisis_streak": self.crisis_streak,
            "z": tuple(round(v, 4) for v in z_new),
            "sit_count": len(self.sit_events),
        }

    def crisis_score(self) -> float:
        hunger = self.state.values[State.HUNGER] / 255
        depression = self.state.values[State.DEPRESSION] / 255
        irritation = self.state.values[State.IRRITATION] / 255
        skin_bad = (255 - self.state.values[State.SKIN]) / 255
        loneliness = self.state.values[State.LONELINESS] / 255
        health_bad = (255 - self.state.values[State.HEALTH]) / 255

        activity_bonus = (self.state.values[State.BOREDOM] / 255) * 0.05 * TRAIT_MULTIPLIER[self.profile.activity]

        ct = (
            hunger * 0.25
            + depression * 0.20
            + irritation * 0.15
            + skin_bad * 0.15
            + loneliness * 0.15 * TRAIT_MULTIPLIER[self.profile.sociability]
            + health_bad * 0.10
            + hunger * 0.03 * TRAIT_MULTIPLIER[self.profile.appetite]
            + activity_bonus
        )
        return max(0.0, min(1.0, ct))

    def crisis_threshold(self) -> float:
        base_theta = 0.5
        cowardice_shift = (TRAIT_MULTIPLIER[self.profile.cowardice] - 1.0) * 0.15
        return max(0.25, min(0.65, base_theta - cowardice_shift))

    def crisis_stage(self) -> str:
        ct = self.crisis_score()
        if ct < 0.3:
            return "정상"
        if ct < 0.5:
            return "주의"
        if ct < 0.7:
            return "경고"
        return "붕괴"

    def dialogue(self) -> str:
        if hasattr(self, "dialogue_bank"):
            return sample_dialogue(self)
        # fallback: inline word bank (pre-512 legacy)
        tone = self._pick_tone()
        word_bank = {
            "행복": ["좋아", "포근해", "고마워"],
            "투정": ["심심해", "같이 놀자", "혼자 싫어"],
            "불평": ["배고파", "기분 별로야", "불편해"],
            "경계": ["이건 뭐야?", "조금 무서워", "지켜볼래"],
            "무기력": ["지쳤어", "가만히 있고 싶어", "축 처져"],
            "위험": ["하악", "건드리지 마", "지금 예민해"],
            "흥분": ["우와!", "신나!", "더 하자"],
            "회복": ["조금 나아졌어", "다시 믿어볼게", "괜찮아지는 중이야"],
        }
        emoticons = {
            "행복": "😺", "투정": "😿", "불평": "😾", "경계": "🙀",
            "무기력": "💤", "위험": "💢", "흥분": "😸", "회복": "✨"
        }
        words = word_bank[tone][:]
        if self.profile.appetite >= 4 and tone in {"행복", "불평", "흥분"}:
            words.append("간식 생각나")
        if self.profile.cowardice >= 4 and tone in {"경계", "위험"}:
            words.append("...괜찮을까?")
        if self.care_alpha > 0.7:
            words.append("요즘은 믿어도 될 것 같아")
        elif self.care_alpha < 0.3:
            words.append("어차피 또 사라질 거잖아")
        sentence = self.rng.choice(words)
        follow = self.rng.choice([w for w in words if w != sentence]) if len(words) > 1 else ""
        return f"{sentence}. {follow} {emoticons[tone]}".strip()

    def _pick_tone(self) -> str:
        if self.crisis_score() >= 0.5:
            return "위험"
        s = self.state.values
        if s[State.HAPPINESS] > 170 and s[State.SATISFACTION] > 160:
            return "행복"
        if s[State.BOREDOM] > 165 or s[State.LONELINESS] > 165:
            return "투정"
        if s[State.HUNGER] > 160 or s[State.IRRITATION] > 160:
            return "불평"
        if s[State.VIGILANCE] > 150 or s[State.ANXIETY] > 150:
            return "경계"
        if s[State.DEPRESSION] > 160 or s[State.FATIGUE] > 170:
            return "무기력"
        if s[State.EXCITEMENT] > 170 and s[State.CURIOSITY] > 150:
            return "흥분"
        return "회복"

    def _update_care_alpha(self, action: Action) -> None:
        caring_actions = {Action.FEED, Action.PLAY, Action.GROOM, Action.PET, Action.GIFT}
        if action in caring_actions:
            self.care_alpha = min(1.0, self.care_alpha + 0.04)
        elif action == Action.IDLE:
            self.care_alpha = max(0.0, self.care_alpha - 0.06)

    def dump_full_state(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "profile": {
                "name": self.profile.name,
                "activity": self.profile.activity,
                "sociability": self.profile.sociability,
                "appetite": self.profile.appetite,
                "cowardice": self.profile.cowardice,
            },
            "state": {k.value: int(v) for k, v in self.state.values.items()},
            "care_alpha": float(self.care_alpha),
            "capacity": float(self.capacity),
            "mode": self.mode.value,
            "mode_persist": int(self.mode_persist),
            "z": [float(self.z[0]), float(self.z[1])],
            "crisis_streak": int(getattr(self, "crisis_streak", 0)),
            "salvation_cooldown": int(getattr(self, "salvation_cooldown", 0)),
            "surv_stats": {k: dict(v) for k, v in self._surv_stats.items()},
            "sit_events": [[t, old.value, new.value] for t, old, new in self.sit_events],
            "rng_state": _rng_state_to_str(self.rng),
        }

    def load_full_state(self, snap: Dict[str, Any]) -> None:
        self.t = int(snap["t"])

        # profile (traits)
        p = snap["profile"]
        self.profile.name = p["name"]
        self.profile.activity = int(p["activity"])
        self.profile.sociability = int(p["sociability"])
        self.profile.appetite = int(p["appetite"])
        self.profile.cowardice = int(p["cowardice"])

        # state vector
        for k, v in snap["state"].items():
            self.state.values[State(k)] = int(v)

        self.care_alpha = float(snap["care_alpha"])
        self.capacity = float(snap["capacity"])
        self.mode = Mode(snap["mode"])
        self.mode_persist = int(snap.get("mode_persist", 0))
        self.z = (float(snap["z"][0]), float(snap["z"][1]))

        self.crisis_streak = int(snap.get("crisis_streak", 0))
        self.salvation_cooldown = int(snap.get("salvation_cooldown", 0))
        raw = snap.get("surv_stats", self._surv_stats)
        self._surv_stats = {k: dict(v) for k, v in raw.items()}
        self.sit_events = [(int(t), Mode(old), Mode(new)) for t, old, new in snap.get("sit_events", [])]

        _rng_state_from_str(self.rng, snap["rng_state"])


def _rng_state_to_str(rng: random.Random) -> str:
    version, internalstate, gauss_next = rng.getstate()
    return json.dumps({"version": version, "internalstate": list(internalstate), "gauss_next": gauss_next})


def _rng_state_from_str(rng: random.Random, s: str) -> None:
    d = json.loads(s)
    rng.setstate((d["version"], tuple(d["internalstate"]), d["gauss_next"]))


def run_demo(seed: int = 42) -> List[Tuple[str, float, str, str, float, float, str, int, float, int]]:
    cat = CATLMAgent(
        profile=CatProfile(name="Nabi", activity=4, sociability=5, appetite=3, cowardice=4),
        rng_seed=seed,
    )

    logs = []
    # run multiple ticks; the agent will pick actions internally now
    for _ in range(12):
        cat.tick(hours=1)
        logs.append((
            cat.mode.value,
            round(cat.crisis_score(), 3),
            cat.crisis_stage(),
            cat.dialogue(),
            round(cat.capacity, 3),
            round(cat.care_alpha, 3),
            f"z={tuple(round(v, 3) for v in cat.z)}",
            len(cat._available_actions()),
            round(cat.crisis_threshold(), 3),
            len(cat.sit_events),
        ))
    return logs


if __name__ == "__main__":
    output = run_demo()
    print("mode\tCt\tstage\tdialogue\tcap\talpha\tz\t|A|\ttheta\tSIT#")
    for row in output:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\t{row[6]}\t{row[7]}\t{row[8]}\t{row[9]}")
