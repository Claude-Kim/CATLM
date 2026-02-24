from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List, Tuple


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
        self.care_alpha = 0.5

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
        self._update_care_alpha(action)
        return deltas

    def _apply_derived_effects(self) -> None:
        if self.state.values[State.IRRITATION] > 180 and self.state.values[State.VIGILANCE] > 170:
            self.state.values[State.AGGRESSION] += 12
        if self.state.values[State.HUNGER] > 170 and self.state.values[State.SKIN] < 80:
            self.state.values[State.HEALTH] -= 10

    def tick(self, hours: int = 1) -> None:
        for _ in range(hours):
            self.state.values[State.HUNGER] += 6
            self.state.values[State.BOREDOM] += 5
            self.state.values[State.LONELINESS] += int(3 * TRAIT_MULTIPLIER[self.profile.sociability])
            self.state.values[State.DEPRESSION] += 2
            self.state.values[State.EXCITEMENT] -= 6
            self.state.values[State.HAPPINESS] -= 4
            self.state.values[State.SATISFACTION] -= 3
        self.state.clamp()

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


def run_demo(seed: int = 42) -> List[Tuple[str, float, str, str]]:
    cat = CATLMAgent(
        profile=CatProfile(name="Nabi", activity=4, sociability=5, appetite=3, cowardice=4),
        rng_seed=seed,
    )
    scenario = [Action.PET, Action.FEED, Action.PLAY, Action.EXPLORE, Action.IDLE, Action.IDLE, Action.GROOM, Action.GIFT]

    logs: List[Tuple[str, float, str, str]] = []
    for act in scenario:
        cat.apply_action(act)
        cat.tick(hours=2)
        logs.append((act.value, round(cat.crisis_score(), 3), cat.crisis_stage(), cat.dialogue()))
    return logs


if __name__ == "__main__":
    output = run_demo()
    print("action\tCt\tstage\tdialogue")
    for row in output:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")
