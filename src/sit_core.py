# sit_core.py
# Clean, paper-aligned SIT core for CATLM v2.x
#
# Core commitments (paper v2 / our agreements):
# - Ct is forward-looking collapse forecast (provided by environment/agent).
# - z_t is *strategy-selection state* (Mixture-of-Experts gating weights).
# - Resistance / hysteresis is implemented by EMA on z_t with λ = 0.25.
# - Insight is a *dominance pattern*: SAFE >= 0.7 (and optionally persists k steps).
# - SIT event is a persistent structural shift: ||z - baseline|| > eps for k steps,
#   optionally only counted inside collapse regime (Ct >= theta).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import math


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) if exps else 1.0
    return [e / s for e in exps]


def _l2(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass(frozen=True)
class SITConfig:
    # Resistance (EMA) for z_t
    lam: float = 0.25  # fixed constant as agreed

    # SIT detection
    eps: float = 0.25           # tuned for z in R^3 (safe/greedy/repair)
    k_persist: int = 3          # consecutive ticks z must stay displaced → SIT fires
    k_stable: int = 15          # consecutive ticks z must stay stable → baseline updates
                                # Must be > k_persist; with EMA λ=0.25, k_stable<5 lets
                                # baseline caterpillar-crawl with z and block SIT detection.
    require_collapse_regime: bool = True

    # Insight definition
    insight_phi: float = 0.7
    insight_k_persist: int = 3  # require insight to persist for k steps to "attain"

    # Gating logits weights (paper-aligned, collapse-aware)
    # Obs features should be normalized into [0,1] where possible.
    w_safe_ct: float = 2.0
    w_safe_threat: float = 1.2   # anxiety/health_bad etc.
    w_safe_hunger: float = -0.4  # safe tends to reduce risky foraging

    w_greedy_hunger: float = 1.1
    w_greedy_curiosity: float = 0.9
    w_greedy_ct: float = -2.2
    w_greedy_threat: float = -0.8

    w_repair_healthbad: float = 1.3
    w_repair_skinbad: float = 0.9
    w_repair_ct: float = 0.6
    w_repair_fatigue: float = -0.3

    # Optional small bias terms
    b_safe: float = 0.0
    b_greedy: float = 0.0
    b_repair: float = 0.0


@dataclass
class SITStepResult:
    Ct: float
    theta: float
    in_collapse: bool

    z_pref: Tuple[float, float, float]   # instantaneous preference \tilde{z}_t
    z: Tuple[float, float, float]        # resistant strategy state z_t

    d_base: float
    disp_streak: int

    sit_event: bool
    sit_reason: str

    safe_weight: float
    insight_now: bool
    insight_attained: bool
    insight_streak: int


class SITCore:
    """
    SIT core is environment-agnostic:
    - caller provides Ct, theta, and observation features (normalized).
    - SITCore returns z_t (resistant), SIT event flag/reason, and insight flags.
    """

    def __init__(self, cfg: SITConfig):
        self.cfg = cfg

        # stateful strategy z_t (R^3 over experts)
        self.z: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)

        # SIT baseline (attractor reference) and displacement streak
        self._baseline: Tuple[float, float, float] = self.z
        self._disp_streak: int = 0
        self._stable_streak: int = 0

        # Insight persistence
        self._insight_streak: int = 0
        self._insight_attained: bool = False

    def reset(self) -> None:
        self.z = (1 / 3, 1 / 3, 1 / 3)
        self._baseline = self.z
        self._disp_streak = 0
        self._stable_streak = 0
        self._insight_streak = 0
        self._insight_attained = False

    # -----------------------------
    # 1) Collapse-aware gating
    # -----------------------------
    def compute_z_pref(self, obs: Dict[str, float], Ct: float) -> Tuple[float, float, float]:
        """
        Compute instantaneous strategy preference \tilde{z}_t in R^3:
          experts = [SAFE, GREEDY, REPAIR]

        Required obs keys (normalized [0,1] recommended):
          hunger, curiosity, health_bad, skin_bad, anxiety, fatigue

        Notes:
        - Ct is the main driver (future collapse forecast).
        - This function must not include direct "capacity" in logits
          if you want to avoid circularity (capacity should restrict feasible actions elsewhere).
        """
        cfg = self.cfg
        hunger = _clamp01(obs.get("hunger", 0.0))
        curiosity = _clamp01(obs.get("curiosity", 0.0))
        health_bad = _clamp01(obs.get("health_bad", 0.0))
        skin_bad = _clamp01(obs.get("skin_bad", 0.0))
        anxiety = _clamp01(obs.get("anxiety", 0.0))
        fatigue = _clamp01(obs.get("fatigue", 0.0))

        threat = _clamp01(0.6 * anxiety + 0.4 * health_bad)

        l_safe = (
            cfg.b_safe
            + cfg.w_safe_ct * Ct
            + cfg.w_safe_threat * threat
            + cfg.w_safe_hunger * hunger
        )

        l_greedy = (
            cfg.b_greedy
            + cfg.w_greedy_hunger * hunger
            + cfg.w_greedy_curiosity * curiosity
            + cfg.w_greedy_ct * Ct
            + cfg.w_greedy_threat * threat
        )

        l_repair = (
            cfg.b_repair
            + cfg.w_repair_healthbad * health_bad
            + cfg.w_repair_skinbad * skin_bad
            + cfg.w_repair_ct * Ct
            + cfg.w_repair_fatigue * fatigue
        )

        w = _softmax([l_safe, l_greedy, l_repair])
        return (w[0], w[1], w[2])

    # -----------------------------
    # 2) Resistance dynamics for z_t
    # -----------------------------
    def update_z(self, z_pref: Tuple[float, float, float]) -> Tuple[float, float, float]:
        lam = self.cfg.lam
        z_new = tuple((1 - lam) * z + lam * zp for z, zp in zip(self.z, z_pref))
        s = sum(z_new) or 1.0
        z_new = tuple(v / s for v in z_new)
        self.z = z_new
        return z_new

    # -----------------------------
    # 3) SIT + Insight detectors
    # -----------------------------
    def _update_sit(self, Ct: float, theta: float) -> Tuple[bool, str, float]:
        cfg = self.cfg
        d_base = _l2(self.z, self._baseline)
        in_collapse = Ct >= theta

        if d_base > cfg.eps:
            self._disp_streak += 1
            self._stable_streak = 0
        else:
            # Don't immediately follow z — wait for k_stable stable ticks
            # before accepting the new position as baseline.
            # k_stable >> k_persist so that gradual drift (irreversible
            # condition) can accumulate enough displacement before the
            # baseline catches up.
            self._disp_streak = 0
            self._stable_streak += 1
            if self._stable_streak >= cfg.k_stable:
                self._baseline = self.z
                self._stable_streak = 0

        sit_ok = self._disp_streak >= cfg.k_persist
        if cfg.require_collapse_regime:
            sit_ok = sit_ok and in_collapse

        if not sit_ok:
            return (False, "", d_base)

        reason = (
            f"d_base={d_base:.3f}>(eps={cfg.eps}), "
            f"disp_streak={self._disp_streak}>=(k={cfg.k_persist}), "
            f"Ct={Ct:.3f}, theta={theta:.3f}, collapse_regime={in_collapse}, "
            f"baseline={tuple(round(v,3) for v in self._baseline)}, z={tuple(round(v,3) for v in self.z)}"
        )

        # Accept new attractor; reset both streaks
        self._baseline = self.z
        self._disp_streak = 0
        self._stable_streak = 0
        return (True, reason, d_base)

    def _update_insight(self) -> Tuple[bool, bool, int]:
        cfg = self.cfg
        safe_w = self.z[0]
        insight_now = safe_w >= cfg.insight_phi

        if insight_now:
            self._insight_streak += 1
        else:
            self._insight_streak = 0

        if (not self._insight_attained) and self._insight_streak >= cfg.insight_k_persist:
            self._insight_attained = True

        return insight_now, self._insight_attained, self._insight_streak

    # -----------------------------
    # One-step API
    # -----------------------------
    def step(self, obs: Dict[str, float], Ct: float, theta: float) -> SITStepResult:
        in_collapse = Ct >= theta

        z_pref = self.compute_z_pref(obs, Ct)
        z = self.update_z(z_pref)

        sit_event, sit_reason, d_base = self._update_sit(Ct, theta)
        insight_now, insight_attained, insight_streak = self._update_insight()

        return SITStepResult(
            Ct=Ct,
            theta=theta,
            in_collapse=in_collapse,
            z_pref=z_pref,
            z=z,
            d_base=d_base,
            disp_streak=self._disp_streak,
            sit_event=sit_event,
            sit_reason=sit_reason,
            safe_weight=z[0],
            insight_now=insight_now,
            insight_attained=insight_attained,
            insight_streak=insight_streak,
        )

    # -----------------------------
    # Serialization helpers
    # -----------------------------
    def dump_state(self) -> dict:
        return {
            "z": list(self.z),
            "baseline": list(self._baseline),
            "disp_streak": self._disp_streak,
            "stable_streak": self._stable_streak,
            "insight_streak": self._insight_streak,
            "insight_attained": self._insight_attained,
        }

    def load_state(self, snap: dict) -> None:
        self.z = tuple(float(v) for v in snap["z"])
        self._baseline = tuple(float(v) for v in snap["baseline"])
        self._disp_streak = int(snap["disp_streak"])
        self._stable_streak = int(snap.get("stable_streak", 0))
        self._insight_streak = int(snap["insight_streak"])
        self._insight_attained = bool(snap["insight_attained"])
