"""
exp_001_seed_sweep.py
=====================
EXP-001: SEED_SWEEP — Paired seed sweep for H1 effect-size analysis.

Compares Irreversible (paper default: capacity decay enabled) vs Ablated
(CAPACITY_DECAY_ON_CRISIS = CAPACITY_DECAY_ON_OVERLOAD = 0) conditions
across N random seeds using a paired design.

Metrics collected per seed (Δ = Irreversible − Ablated):
  sit_count    — total SIT events fired
  mean_Ct      — mean collapse forecast intensity
  normal_ticks — ticks spent in NORMAL mode  (survival proxy)
  alpha_final  — care_alpha at last tick
  alpha_slope  — mean(α, last-quarter) − mean(α, first-quarter)

Statistical analysis:
  Percentile bootstrap 95% CI for the mean of each paired difference.
  CI that excludes zero → evidence of genuine H1 effect.

Scenario: profile cowardice=5, sociability=5
          40 ticks IDLE (forced neglect) → 80 ticks auto-policy
          → ensures capacity decay fires and SIT events are informative.

Outputs (results/exp-001/):
  exp-001_per_seed.csv       — per-seed raw data (all metrics, both conditions)
  exp-001_summary.txt        — Δ statistics + bootstrap CI report
  exp-001_robustness.png/pdf — histogram panels with CI bands (fig6)

Run from project root:
  python3 src/exp_001_seed_sweep.py [--seeds N] [--ticks N] [--n-boot N]
                                    [--eps EPS] [--cowardice N] [--sociability N]
                                    [--neglect TICKS] [--no-pdf]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import catlm_simulator as _sim_module
from catlm_simulator import CATLMAgent, CatProfile, Action, TRAIT_MULTIPLIER
from sit_core import SITCore, SITConfig

# ── Experiment defaults ───────────────────────────────────────────────────────

DEFAULT_SEEDS       = 200
DEFAULT_N_TICKS     = 120
DEFAULT_NEGLECT     = 40       # forced IDLE at start → creates capacity-decay conditions
DEFAULT_EPS         = 0.25    # paper-default SIT threshold
DEFAULT_COWARDICE   = 5       # θ ≈ 0.41 — easier collapse entry
DEFAULT_SOCIABILITY = 5       # strong loneliness weight in Ct
DEFAULT_N_BOOT      = 2000

# Profile kwargs (overridden by CLI)
PROFILE_KWARGS = dict(name="Cat", activity=3, sociability=DEFAULT_SOCIABILITY,
                      appetite=3, cowardice=DEFAULT_COWARDICE)

OUT_DIR = os.path.join(_ROOT_DIR, "results", "exp-001")


# ── Simulation runners ────────────────────────────────────────────────────────

def _run_simulation(
    profile: CatProfile,
    n_ticks: int,
    seed: int,
    user_actions: Optional[List[Optional[Action]]] = None,
    sit_config: Optional[SITConfig] = None,
) -> List[Dict[str, Any]]:
    agent = CATLMAgent(profile=profile, rng_seed=seed)
    if sit_config is not None:
        agent.sit = SITCore(sit_config)
    logs: List[Dict[str, Any]] = []
    for i in range(n_ticks):
        action = (user_actions[i] if user_actions and i < len(user_actions) else None)
        logs.append(agent.step(user_action=action))
    return logs


def _run_simulation_ablated(
    profile: CatProfile,
    n_ticks: int,
    seed: int,
    user_actions: Optional[List[Optional[Action]]] = None,
    sit_config: Optional[SITConfig] = None,
) -> List[Dict[str, Any]]:
    """Run with irreversible capacity decay disabled (ablated control)."""
    orig_crisis   = _sim_module.CAPACITY_DECAY_ON_CRISIS
    orig_overload = _sim_module.CAPACITY_DECAY_ON_OVERLOAD
    _sim_module.CAPACITY_DECAY_ON_CRISIS   = 0.0
    _sim_module.CAPACITY_DECAY_ON_OVERLOAD = 0.0
    try:
        return _run_simulation(profile, n_ticks, seed,
                               user_actions=user_actions, sit_config=sit_config)
    finally:
        _sim_module.CAPACITY_DECAY_ON_CRISIS   = orig_crisis
        _sim_module.CAPACITY_DECAY_ON_OVERLOAD = orig_overload


# ── Core experiment logic ─────────────────────────────────────────────────────

def _bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap 95% CI for the mean of paired differences.
    Returns (mean, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(deltas)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(deltas, size=n, replace=True).mean()
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(deltas.mean()), ci_lo, ci_hi


def _per_run_metrics(logs: List[Dict[str, Any]], n_ticks: int) -> Dict[str, float]:
    """Extract scalar summary metrics from a single simulation run."""
    cts    = np.array([r["Ct"]    for r in logs])
    alphas = np.array([r["alpha"] for r in logs])
    modes  = [r["mode"] for r in logs]

    normal_ticks = sum(1 for m in modes if m == "normal")

    q = max(1, n_ticks // 4)
    alpha_slope = float(alphas[-q:].mean() - alphas[:q].mean())

    return {
        "sit_count":    float(logs[-1]["sit_count"]),
        "mean_Ct":      float(cts.mean()),
        "normal_ticks": float(normal_ticks),
        "alpha_final":  float(alphas[-1]),
        "alpha_slope":  alpha_slope,
    }


def run_paired_experiment(
    profile: CatProfile,
    n_seeds: int = DEFAULT_SEEDS,
    n_ticks: int = DEFAULT_N_TICKS,
    sit_config: Optional[SITConfig] = None,
    user_actions: Optional[List[Optional[Action]]] = None,
    n_boot: int = DEFAULT_N_BOOT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Paired seed sweep comparing Irreversible vs Ablated conditions.

    For each seed s in range(n_seeds):
      1. Run Irreversible (paper default capacity decay).
      2. Run Ablated      (CAPACITY_DECAY_ON_CRISIS = CAPACITY_DECAY_ON_OVERLOAD = 0).
      Both runs share the same seed → paired design.

    Returns dict with:
      per_seed  — list of per-seed dicts (irrev_*, ablated_*, delta_*)
      summary   — per-metric {delta_mean, ci_lo, ci_hi, ci_zero_excluded}
    """
    if verbose:
        print(f"  Paired experiment: n_seeds={n_seeds}, n_ticks={n_ticks}")

    metric_keys = ["sit_count", "mean_Ct", "normal_ticks", "alpha_final", "alpha_slope"]
    records: List[Dict[str, Any]] = []

    for seed in range(n_seeds):
        if verbose and seed % 50 == 0:
            print(f"    seed {seed}/{n_seeds} …")

        logs_i = _run_simulation(profile, n_ticks=n_ticks, seed=seed,
                                 user_actions=user_actions, sit_config=sit_config)
        logs_a = _run_simulation_ablated(profile, n_ticks=n_ticks, seed=seed,
                                         user_actions=user_actions, sit_config=sit_config)

        mi = _per_run_metrics(logs_i, n_ticks)
        ma = _per_run_metrics(logs_a, n_ticks)

        row: Dict[str, Any] = {"seed": seed}
        for k in metric_keys:
            row[f"irrev_{k}"]   = mi[k]
            row[f"ablated_{k}"] = ma[k]
            row[f"delta_{k}"]   = mi[k] - ma[k]
        records.append(row)

    rng_boot = np.random.default_rng(0)
    summary: Dict[str, Dict[str, Any]] = {}
    for k in metric_keys:
        deltas = np.array([r[f"delta_{k}"] for r in records])
        mean, lo, hi = _bootstrap_ci(deltas, n_boot=n_boot, rng=rng_boot)
        summary[k] = {
            "delta_mean":       mean,
            "ci_lo":            lo,
            "ci_hi":            hi,
            "ci_zero_excluded": not (lo <= 0.0 <= hi),
        }

    if verbose:
        print(f"\n  ── Δ(Irreversible − Ablated), 95% Bootstrap CI  (n={n_seeds} seeds) ──")
        for k, v in summary.items():
            flag = "  *** CI excludes 0" if v["ci_zero_excluded"] else ""
            print(f"    Δ{k:<15}  {v['delta_mean']:+.4f}  "
                  f"95%CI [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]{flag}")

    return {"per_seed": records, "summary": summary}


# ── Figure ────────────────────────────────────────────────────────────────────

def figure_seed_robustness(
    experiment_result: Dict[str, Any],
    title: str = "H1 Effect-Size: Irreversible − Ablated  (Paired Seed Sweep)",
) -> plt.Figure:
    """
    One panel per metric. Each panel shows:
      - histogram of per-seed Δ = (Irreversible − Ablated)
      - vertical line at mean
      - shaded 95% bootstrap CI band
      - red dashed zero reference
    """
    records = experiment_result["per_seed"]
    summary = experiment_result["summary"]
    n       = len(records)

    panels = [
        ("sit_count",    "Δ SIT events",          "ΔSIT count"),
        ("normal_ticks", "Δ ticks in NORMAL mode", "ΔSurvival time (normal ticks)"),
        ("alpha_final",  "Δ α_t at t=T",          "Δα_t final"),
        ("mean_Ct",      "Δ mean C_t",             "ΔMean collapse forecast"),
        ("alpha_slope",  "Δ α slope (Q4−Q1)",      "Δα growth slope"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5.2))
    fig.suptitle(f"{title}  (n={n} seeds)", fontsize=11, y=1.02)

    C_irrev   = "#1565C0"
    C_zero    = "#E53935"
    C_ci_fill = "#BBDEFB"

    for ax, (k, xlabel, panel_title) in zip(axes, panels):
        deltas = np.array([r[f"delta_{k}"] for r in records])
        s      = summary[k]

        ax.hist(deltas, bins=35, color="#90A4AE", edgecolor="white",
                linewidth=0.5, alpha=0.85, density=True)

        ax.axvspan(s["ci_lo"], s["ci_hi"], color=C_ci_fill, alpha=0.55,
                   label=f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]")
        ax.axvline(s["delta_mean"], color=C_irrev, linewidth=2.0,
                   label=f"mean = {s['delta_mean']:+.3f}")
        ax.axvline(0.0, color=C_zero, linewidth=1.3, linestyle="--",
                   alpha=0.75, label="zero (null)")

        star = "\n★ CI excludes 0" if s["ci_zero_excluded"] else ""
        title_color = C_irrev if s["ci_zero_excluded"] else "black"
        ax.set_title(f"{panel_title}{star}", fontsize=9, color=title_color)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density" if k == "sit_count" else "", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ── Output writers ────────────────────────────────────────────────────────────

def save_csv(result: Dict[str, Any], path: str) -> None:
    records = result["per_seed"]
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  → {path}")


def save_summary_txt(
    result: Dict[str, Any],
    path: str,
    n_seeds: int,
    n_ticks: int,
    eps: float,
    cowardice: int,
    sociability: int,
    neglect: int,
    n_boot: int,
    require_collapse_regime: bool = True,
) -> None:
    summary = result["summary"]
    theta_approx = 0.5 - (TRAIT_MULTIPLIER[cowardice] - 1.0) * 0.15
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f"CATLM EXP-001: Seed-Sweep Experiment Summary\n"
            f"=============================================\n"
            f"Profile:  activity=3, sociability={sociability}, "
            f"appetite=3, cowardice={cowardice}  (θ ≈ {theta_approx:.3f})\n"
            f"Scenario: {neglect} ticks IDLE → {n_ticks - neglect} ticks auto-policy\n"
            f"n_seeds:  {n_seeds}\n"
            f"n_ticks:  {n_ticks}\n"
            f"SIT eps:  {eps}  (require_collapse_regime={require_collapse_regime})\n"
            f"n_boot:   {n_boot}\n\n"
            f"Δ = Irreversible − Ablated  (paired design, 95% percentile bootstrap CI)\n\n"
        )
        for k, v in summary.items():
            flag = "  *** CI EXCLUDES 0" if v["ci_zero_excluded"] else ""
            fh.write(
                f"  Δ{k:<15}  mean={v['delta_mean']:+.4f}  "
                f"CI=[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]{flag}\n"
            )
    print(f"  → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-001: SEED_SWEEP — Paired irreversible vs ablated seed sweep"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, metavar="N",
                        help=f"Number of seeds (default: {DEFAULT_SEEDS}; "
                             "recommend 500 for publication)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_N_TICKS,
                        help=f"Simulation length in ticks (default: {DEFAULT_N_TICKS})")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, metavar="N",
                        help=f"Bootstrap resamples for CI (default: {DEFAULT_N_BOOT})")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, metavar="EPS",
                        help=f"SIT displacement threshold (default: {DEFAULT_EPS} = "
                             "game default; paper-consistent)")
    parser.add_argument("--cowardice", type=int, default=DEFAULT_COWARDICE,
                        choices=[1, 2, 3, 4, 5], metavar="N",
                        help=f"Cowardice trait (default: {DEFAULT_COWARDICE}; "
                             "lowers θ → easier crisis entry, shows H1 effect)")
    parser.add_argument("--sociability", type=int, default=DEFAULT_SOCIABILITY,
                        choices=[1, 2, 3, 4, 5], metavar="N",
                        help=f"Sociability trait (default: {DEFAULT_SOCIABILITY}; "
                             "raises loneliness weight in Ct)")
    parser.add_argument("--neglect", type=int, default=DEFAULT_NEGLECT, metavar="TICKS",
                        help=f"Forced IDLE ticks at start (default: {DEFAULT_NEGLECT}; "
                             "creates crisis conditions where capacity decay fires)")
    parser.add_argument("--output", default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF output (PNG only)")
    parser.add_argument("--sit-anytime", action="store_true",
                        help="Count SIT events outside collapse regime (diagnostic: "
                             "checks whether collapse-filter is the reason SIT=0).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    sit_cfg = SITConfig(
        lam=0.25,
        eps=args.eps,
        k_persist=3,
        k_stable=15,
        require_collapse_regime=(not args.sit_anytime),
        insight_phi=0.7,
        insight_k_persist=3,
    )

    profile = CatProfile(
        name="Cat",
        activity=3,
        sociability=args.sociability,
        appetite=3,
        cowardice=args.cowardice,
    )

    neglect_n = max(0, args.neglect)
    user_actions: List[Optional[Action]] = (
        [Action.IDLE] * neglect_n
        + [None] * max(0, args.ticks - neglect_n)
    )

    theta_approx = 0.5 - (TRAIT_MULTIPLIER[args.cowardice] - 1.0) * 0.15

    print("=" * 68)
    print("EXP-001: SEED_SWEEP — Irreversible vs Ablated (H1 effect-size)")
    print("=" * 68)
    print(f"  seeds={args.seeds}  ticks={args.ticks}  eps={args.eps}  n_boot={args.n_boot}")
    print(f"  Profile: sociability={args.sociability}, cowardice={args.cowardice}  "
          f"(θ ≈ {theta_approx:.3f})")
    print(f"  Scenario: {neglect_n} ticks IDLE → {args.ticks - neglect_n} ticks auto-policy")
    if args.sit_anytime:
        print("  [DIAGNOSTIC] --sit-anytime: require_collapse_regime=False")
    print(f"  Output: {args.output}")
    print()

    t0 = time.time()

    result = run_paired_experiment(
        profile=profile,
        n_seeds=args.seeds,
        n_ticks=args.ticks,
        sit_config=sit_cfg,
        user_actions=user_actions,
        n_boot=args.n_boot,
        verbose=True,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    print("\n[saving CSV]")
    save_csv(result, os.path.join(args.output, "exp-001_per_seed.csv"))

    # ── Save summary report ───────────────────────────────────────────────────
    print("[saving summary]")
    save_summary_txt(
        result,
        os.path.join(args.output, "exp-001_summary.txt"),
        n_seeds=args.seeds,
        n_ticks=args.ticks,
        eps=args.eps,
        cowardice=args.cowardice,
        sociability=args.sociability,
        neglect=neglect_n,
        n_boot=args.n_boot,
        require_collapse_regime=(not args.sit_anytime),
    )

    # ── Render figure ─────────────────────────────────────────────────────────
    print("[rendering figure]")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.linewidth": 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig = figure_seed_robustness(result)

    png_path = os.path.join(args.output, "exp-001_robustness.png")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  → {png_path}")

    if not args.no_pdf:
        pdf_path = os.path.join(args.output, "exp-001_robustness.pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        print(f"  → {pdf_path}")

    plt.close(fig)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"All outputs saved to {args.output}/")


if __name__ == "__main__":
    main()
