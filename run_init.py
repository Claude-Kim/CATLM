# === Run-start init block: deterministic random traits from seed ===
# Drop this near your demo/main entrypoint (or in catlm_simulator.py).
# It creates:
#  - a reproducible run_seed
#  - a deterministic trait roll (1..5)
#  - a nice header print that shows seed + traits + multipliers

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import random

from catlm_simulator import CatProfile, CATLMAgent, TRAIT_MULTIPLIER


@dataclass(frozen=True)
class RunInit:
    run_seed: int
    profile: "CatProfile"
    multipliers: Tuple[float, float, float, float]  # act, soc, app, cow


def _roll_trait_1to5(rng: random.Random, mode: str = "uniform") -> int:
    if mode == "uniform":
        return rng.randint(1, 5)

    # Slightly more "normal-ish": center-weighted (3 is more common than 1/5)
    # Feel free to tune weights.
    if mode == "centered":
        values = [1, 2, 3, 4, 5]
        weights = [1, 2, 4, 2, 1]
        return rng.choices(values, weights=weights, k=1)[0]

    # Optional: more “varied personalities” (edges more common)
    if mode == "spiky":
        values = [1, 2, 3, 4, 5]
        weights = [3, 2, 1, 2, 3]
        return rng.choices(values, weights=weights, k=1)[0]

    raise ValueError(f"Unknown trait roll mode: {mode}")


def init_run_profile(
    name: str = "Nabi",
    run_seed: Optional[int] = None,
    trait_roll_mode: str = "centered",
    force_traits: Optional[Tuple[int, int, int, int]] = None,  # (activity,sociability,appetite,cowardice)
) -> RunInit:
    """
    Deterministic init:
      - If run_seed is provided, same seed => same traits.
      - If run_seed is None, we pick a random seed (but we print it so it can be replayed).
    """
    if run_seed is None:
        run_seed = random.randint(0, 2_147_483_647)

    rng = random.Random(run_seed)

    if force_traits is not None:
        a, s, ap, c = force_traits
    else:
        a = _roll_trait_1to5(rng, trait_roll_mode)
        s = _roll_trait_1to5(rng, trait_roll_mode)
        ap = _roll_trait_1to5(rng, trait_roll_mode)
        c = _roll_trait_1to5(rng, trait_roll_mode)

    profile = CatProfile(name=name, activity=a, sociability=s, appetite=ap, cowardice=c)
    mults = (
        float(TRAIT_MULTIPLIER[a]),
        float(TRAIT_MULTIPLIER[s]),
        float(TRAIT_MULTIPLIER[ap]),
        float(TRAIT_MULTIPLIER[c]),
    )
    return RunInit(run_seed=run_seed, profile=profile, multipliers=mults)


def print_run_header(init: RunInit) -> None:
    a = init.profile.activity
    s = init.profile.sociability
    ap = init.profile.appetite
    c = init.profile.cowardice
    ma, ms, map_, mc = init.multipliers

    print("=" * 72)
    print(f"RUN SEED: {init.run_seed}")
    print(f"CAT: {init.profile.name}")
    print(f"TRAITS (a/s/ap/c): {a}/{s}/{ap}/{c}")
    print(f"MULTIPLIERS: act={ma:.2f}  soc={ms:.2f}  app={map_:.2f}  cow={mc:.2f}")
    print("=" * 72)


# --- Example usage in your run/demo ---
def run_demo(seed: Optional[int] = None):
    init = init_run_profile(name="Nabi", run_seed=seed, trait_roll_mode="centered")
    print_run_header(init)

    cat = CATLMAgent(profile=init.profile, rng_seed=init.run_seed)

    # Optional: if you attached dialogue bank, do it here.
    # attach_dialogue_bank(cat, "dialogue_bank_512.json")

    # Now proceed with your scenario loop...
    # return logs or print lines
    return cat


if __name__ == "__main__":
    cat = run_demo(seed=42)
    print("\nRunning 12 ticks with step() API:")
    print(f"{'t':>3}  {'mode':<8}  {'Ct':>6}  {'stage':>6}  {'action':<8}  {'cap':>6}  {'alpha':>7}  {'streak':>6}  {'SIT#':>4}")
    print("-" * 72)
    for _ in range(12):
        r = cat.step(user_action=None)
        print(
            f"{r['t']:>3}  {r['mode']:<8}  {r['Ct']:>6.3f}  {r['stage']:>6}"
            f"  {r['action']:<8}  {r['capacity']:>6.3f}  {r['alpha']:>7.4f}"
            f"  {r['crisis_streak']:>6}  {r['sit_count']:>4}"
        )