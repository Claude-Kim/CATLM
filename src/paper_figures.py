"""
paper_figures.py
================
Data generation and figure rendering module for the SIT paper:
"Structural Inference Transitions Under Irreversible Survival Constraints"

Usage (from project root):
    python3 src/paper_figures.py [--output figures/] [--ticks 120] [--seed 42]

Generated figures (PNG + PDF):
    fig1_main_trajectory   — Ct / z_t / capacity / care_alpha over time
    fig2_moe_resistance    — EMA resistance: z̃_t (instantaneous) vs z_t (resistant)
    fig3_capacity_ablation — Irreversible decay vs. ablated control condition (H1)
    fig4_trait_sensitivity — Mean Ct and SIT events across cowardice / sociability levels
    fig5_z_simplex         — z_t trajectory on SAFE–GREEDY–REPAIR simplex
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── path setup so this file can be run directly ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")           # headless / non-interactive rendering
import matplotlib.pyplot as plt
import numpy as np

import catlm_simulator as _sim_module
from catlm_simulator import CATLMAgent, CatProfile, Action
from sit_core import SITCore, SITConfig


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (publication-friendly)
# ─────────────────────────────────────────────────────────────────────────────
_C = {
    "safe":       "#2196F3",   # blue
    "greedy":     "#FF9800",   # orange
    "repair":     "#4CAF50",   # green
    "Ct":         "#E53935",   # red
    "theta":      "#9E9E9E",   # grey
    "capacity":   "#7B1FA2",   # purple
    "alpha":      "#00796B",   # teal
    "survival_bg":"#FFEBEE",   # light-red fill for SURVIVAL regions
    "sit":        "#E91E63",   # pink — SIT event markers
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "lines.linewidth":    1.6,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runners
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    profile: CatProfile,
    n_ticks: int = 120,
    seed: int = 42,
    user_actions: Optional[List[Optional[Action]]] = None,
    sit_config: Optional[SITConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run n_ticks and return a list of per-tick report dicts.
    Pass sit_config to override the default SITCore parameters (e.g. lower eps
    for paper-demonstration runs where SIT events should be visible).
    """
    agent = CATLMAgent(profile=profile, rng_seed=seed)
    if sit_config is not None:
        agent.sit = SITCore(sit_config)
    logs: List[Dict[str, Any]] = []
    for i in range(n_ticks):
        action = (user_actions[i] if user_actions and i < len(user_actions) else None)
        logs.append(agent.step(user_action=action))
    return logs


def run_simulation_ablated(
    profile: CatProfile,
    n_ticks: int = 120,
    seed: int = 42,
    user_actions: Optional[List[Optional[Action]]] = None,
    sit_config: Optional[SITConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run with irreversible capacity decay disabled (H1 ablation / reversible control).
    Temporarily patches module-level constants then restores them.
    """
    orig_crisis   = _sim_module.CAPACITY_DECAY_ON_CRISIS
    orig_overload = _sim_module.CAPACITY_DECAY_ON_OVERLOAD
    _sim_module.CAPACITY_DECAY_ON_CRISIS   = 0.0
    _sim_module.CAPACITY_DECAY_ON_OVERLOAD = 0.0
    try:
        return run_simulation(profile, n_ticks, seed,
                              user_actions=user_actions, sit_config=sit_config)
    finally:
        _sim_module.CAPACITY_DECAY_ON_CRISIS   = orig_crisis
        _sim_module.CAPACITY_DECAY_ON_OVERLOAD = orig_overload


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract(logs: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in logs])


def _shade_survival(ax, t: np.ndarray, mode_list: List[str]) -> None:
    """Fill SURVIVAL-mode spans with a light-red background."""
    in_surv = False
    start   = None
    first   = True
    for ti, m in zip(t, mode_list):
        if m == "survival" and not in_surv:
            start  = ti
            in_surv = True
        elif m != "survival" and in_surv:
            label = "Survival mode" if first else None
            ax.axvspan(start, ti, color=_C["survival_bg"], alpha=0.55,
                       zorder=0, label=label)
            first   = False
            in_surv = False
    if in_surv:
        label = "Survival mode" if first else None
        ax.axvspan(start, t[-1], color=_C["survival_bg"], alpha=0.55,
                   zorder=0, label=label)


def _mark_sit(ax, t: np.ndarray, y_vals: np.ndarray,
              sit_mask: np.ndarray, **scatter_kw) -> None:
    """Mark SIT-event ticks with × on the given y-axis series."""
    idx = np.where(sit_mask)[0]
    if len(idx):
        kw = dict(marker="x", s=90, color=_C["sit"],
                  zorder=5, linewidths=2, label="SIT event")
        kw.update(scatter_kw)
        ax.scatter(t[idx], y_vals[idx], **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main trajectory (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def figure_main_trajectory(
    logs: List[Dict[str, Any]],
    title: str = "CATLM — Main Trajectory",
) -> plt.Figure:
    """
    4 stacked panels sharing the x-axis (time t):
      [0] Ct (forecast) and Ct_instant vs θ threshold + SURVIVAL shading + SIT events
      [1] z_t MoE weights (SAFE / GREEDY / REPAIR) stacked area
      [2] Irreversible structural capacity W_t
      [3] Origin dependency weight α_t (care_alpha)
    """
    t         = _extract(logs, "t")
    Ct        = _extract(logs, "Ct")
    Ct_now    = _extract(logs, "Ct_instant")
    theta     = _extract(logs, "theta")
    mode      = [r["mode"] for r in logs]
    sit_mask  = np.array([r["sit_event"] for r in logs], dtype=bool)
    z_safe    = np.array([r["z"][0] for r in logs])
    z_greedy  = np.array([r["z"][1] for r in logs])
    z_repair  = np.array([r["z"][2] for r in logs])
    capacity  = _extract(logs, "capacity")
    alpha     = _extract(logs, "alpha")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=13, y=0.99)

    # ── Panel 0: Ct ────────────────────────────────────────────────────────
    ax = axes[0]
    _shade_survival(ax, t, mode)
    ax.plot(t, theta,  color=_C["theta"], linestyle=":", linewidth=1.2,
            label="θ (collapse threshold)")
    ax.plot(t, Ct_now, color=_C["Ct"], linestyle="--", alpha=0.45,
            linewidth=1.2, label="$C_t$ instant")
    ax.plot(t, Ct,     color=_C["Ct"], label="$C_t$ forecast (H=3)")
    _mark_sit(ax, t, Ct, sit_mask)
    ax.set_ylabel("Collapse score")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.set_title("Collapse Forecast Intensity $C_t$ and Mode Transitions")

    # ── Panel 1: z_t stacked ───────────────────────────────────────────────
    ax = axes[1]
    ax.stackplot(t, z_safe, z_greedy, z_repair,
                 labels=["SAFE", "GREEDY", "REPAIR"],
                 colors=[_C["safe"], _C["greedy"], _C["repair"]],
                 alpha=0.78)
    for sit_t in t[sit_mask]:
        ax.axvline(sit_t, color=_C["sit"], linewidth=1.1, alpha=0.65, zorder=3)
    ax.set_ylabel("MoE weight $z_t$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_title("Strategy State $z_t$ (Mixture-of-Experts: SAFE / GREEDY / REPAIR)")

    # ── Panel 2: Capacity ──────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, capacity, color=_C["capacity"], label="$W_t$ (capacity)")
    ax.fill_between(t, capacity, alpha=0.15, color=_C["capacity"])
    # capacity thresholds
    thresholds = [(0.75, "EXPLORE off"), (0.55, "TRAIN off"),
                  (0.40, "GIFT off"), (0.25, "minimal set")]
    for th, lbl in thresholds:
        ax.axhline(th, color="#BDBDBD", linestyle="--", linewidth=0.8)
        ax.text(t[-1] * 1.01, th, lbl, va="center", fontsize=7, color="#757575")
    ax.set_ylabel("Capacity")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Irreversible Structural Capacity $W_t$")

    # ── Panel 3: α_t ───────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, alpha, color=_C["alpha"], label="$\\alpha_t$ (care trust)")
    ax.fill_between(t, alpha, alpha=0.15, color=_C["alpha"])
    ax.set_ylabel("$\\alpha_t$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Tick $t$ (hours)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Origin Dependency Weight $\\alpha_t$")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — EMA resistance: z̃_t vs z_t
# ─────────────────────────────────────────────────────────────────────────────

def figure_moe_resistance(
    logs: List[Dict[str, Any]],
    title: str = "EMA Resistance: $\\tilde{z}_t$ (Instantaneous Preference) vs $z_t$ (Resistant State)",
) -> plt.Figure:
    """
    3-panel plot, one per expert (SAFE / GREEDY / REPAIR).
    Dashed = z̃_t (instantaneous softmax output).
    Solid  = z_t  (EMA-smoothed resistant state, λ=0.25).
    """
    t        = _extract(logs, "t")
    mode     = [r["mode"] for r in logs]
    sit_mask = np.array([r["sit_event"] for r in logs], dtype=bool)

    experts = [
        (0, _C["safe"],   "SAFE"),
        (1, _C["greedy"], "GREEDY"),
        (2, _C["repair"], "REPAIR"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=11, y=0.99)

    for ax, (idx, color, label) in zip(axes, experts):
        z_pref = np.array([r["z_pref"][idx] for r in logs])
        z      = np.array([r["z"][idx]      for r in logs])

        _shade_survival(ax, t, mode)
        ax.plot(t, z_pref, linestyle="--", alpha=0.60, color=color,
                linewidth=1.2, label=f"$\\tilde{{z}}_t$ ({label})")
        ax.plot(t, z,     color=color, label=f"$z_t$ ({label})")
        for sit_t in t[sit_mask]:
            ax.axvline(sit_t, color=_C["sit"], linewidth=1.0, alpha=0.55, zorder=3)
        ax.set_ylabel("Weight")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{label} Expert")

    axes[-1].set_xlabel("Tick $t$ (hours)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Capacity ablation (H1: reversible control)
# ─────────────────────────────────────────────────────────────────────────────

def figure_capacity_ablation(
    logs_irrev: List[Dict[str, Any]],
    logs_ablated: List[Dict[str, Any]],
    title: str = "H1 Ablation: Irreversible Capacity Decay vs. Reversible Control",
) -> plt.Figure:
    """
    2 × 2 panel comparing the irreversible (paper) condition vs. ablated control:
      [0,0] W_t (capacity)
      [0,1] Ct  (collapse forecast)
      [1,0] Cumulative SIT events
      [1,1] α_t (care trust)
    """
    t = _extract(logs_irrev, "t")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(title, fontsize=12, y=0.99)

    C_irrev   = "#1565C0"   # dark blue — irreversible
    C_ablated = "#EF6C00"   # dark orange — ablated

    def _panel(ax, key, ylabel, title_str, ylim=(0, 1.05)):
        irrev   = _extract(logs_irrev,   key)
        ablated = _extract(logs_ablated, key)
        ax.plot(t, irrev,   color=C_irrev,   label="Irreversible (paper)")
        ax.plot(t, ablated, color=C_ablated, label="Ablated (no decay)",
                linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(title_str)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.set_xlabel("Tick $t$ (hours)")

    _panel(axes[0, 0], "capacity", "Capacity $W_t$",
           "Structural Capacity $W_t$")
    _panel(axes[0, 1], "Ct", "Collapse score $C_t$",
           "Collapse Forecast $C_t$")
    _panel(axes[1, 1], "alpha", "$\\alpha_t$",
           "Origin Dependency Weight $\\alpha_t$")

    # Cumulative SIT events (step plot)
    ax = axes[1, 0]
    cum_irrev   = np.cumsum([r["sit_event"] for r in logs_irrev])
    cum_ablated = np.cumsum([r["sit_event"] for r in logs_ablated])
    ax.step(t, cum_irrev,   where="post", color=C_irrev,
            label=f"Irreversible (n={cum_irrev[-1]})")
    ax.step(t, cum_ablated, where="post", color=C_ablated,
            linestyle="--", label=f"Ablated (n={cum_ablated[-1]})")
    ax.set_ylabel("Cumulative SIT events")
    ax.set_title("Cumulative SIT Events")
    ax.legend(fontsize=8)
    ax.set_xlabel("Tick $t$ (hours)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Trait sensitivity (cowardice × sociability)
# ─────────────────────────────────────────────────────────────────────────────

def figure_trait_sensitivity(
    n_ticks: int = 120,
    n_seeds: int = 10,
) -> plt.Figure:
    """
    2 × 2 box-plot grid:
      [0,*] Vary cowardice (1–5), fixed sociability=3
      [1,*] Vary sociability (1–5), fixed cowardice=3
    Columns: mean Ct | total SIT count
    """

    def _collect(vary: str, levels: List[int]) -> Dict[int, Dict[str, List[float]]]:
        results: Dict[int, Dict[str, List[float]]] = {}
        for level in levels:
            ct_means: List[float] = []
            sit_counts: List[float] = []
            for seed in range(n_seeds):
                if vary == "cowardice":
                    profile = CatProfile(name="Cat", activity=3, sociability=3,
                                         appetite=3, cowardice=level)
                else:
                    profile = CatProfile(name="Cat", activity=3, sociability=level,
                                         appetite=3, cowardice=3)
                logs = run_simulation(profile, n_ticks=n_ticks, seed=seed)
                ct_means.append(float(np.mean([r["Ct"] for r in logs])))
                sit_counts.append(float(logs[-1]["sit_count"]))
            results[level] = {"ct": ct_means, "sit": sit_counts}
        return results

    levels = [1, 2, 3, 4, 5]
    print("    Collecting cowardice data …")
    cow = _collect("cowardice", levels)
    print("    Collecting sociability data …")
    soc = _collect("sociability", levels)

    def _boxplot(ax, results, levels, xlabel, ylabel, title, cmap, key):
        data = [results[lv][key] for lv in levels]
        bp = ax.boxplot(data, positions=levels, widths=0.6, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        colors = plt.get_cmap(cmap)(np.linspace(0.35, 0.85, len(levels)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(levels)
        ax.grid(axis="y", alpha=0.3)
        # Fix near-empty appearance by setting data-driven y-limits
        all_vals = [v for lv in levels for v in results[lv][key]]
        if key == "sit":
            ax.set_ylim(-0.5, max(all_vals) + 0.5)
        else:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.10 if hi > lo else 0.05
            ax.set_ylim(lo - margin, hi + margin)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Trait Sensitivity Analysis  (n={n_seeds} seeds × {n_ticks} ticks each)",
        fontsize=12, y=0.99,
    )

    _boxplot(axes[0, 0], cow, levels,
             "Cowardice level", "Mean $C_t$",
             "Cowardice vs Mean Collapse Forecast $C_t$", "Blues", key="ct")
    _boxplot(axes[0, 1], cow, levels,
             "Cowardice level", "Total SIT events",
             "Cowardice vs Total SIT Events", "Blues", key="sit")

    _boxplot(axes[1, 0], soc, levels,
             "Sociability level", "Mean $C_t$",
             "Sociability vs Mean Collapse Forecast $C_t$", "Greens", key="ct")
    _boxplot(axes[1, 1], soc, levels,
             "Sociability level", "Total SIT events",
             "Sociability vs Total SIT Events", "Greens", key="sit")

    # Annotate effect direction
    for ax in (axes[0, 0], axes[0, 1]):
        ax.text(0.98, 0.02,
                "Higher cowardice → lower θ → easier collapse entry",
                transform=ax.transAxes, fontsize=7, ha="right", color="#616161")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.text(0.98, 0.02,
                "Higher sociability → stronger loneliness weight in $C_t$",
                transform=ax.transAxes, fontsize=7, ha="right", color="#616161")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — z_t trajectory on SAFE–GREEDY–REPAIR simplex (ternary plot)
# ─────────────────────────────────────────────────────────────────────────────

def _ternary_to_xy(s: float, g: float, r: float) -> Tuple[float, float]:
    """
    Barycentric (SAFE, GREEDY, REPAIR) → 2-D Cartesian.
    Vertex positions:
      SAFE   = (0.5,  √3/2)  — top
      GREEDY = (0,    0)     — bottom-left
      REPAIR = (1,    0)     — bottom-right
    """
    x = r + 0.5 * s
    y = s * math.sqrt(3) / 2
    return x, y


def _draw_simplex(ax) -> None:
    """Draw the equilateral triangle and internal iso-z grid."""
    h = math.sqrt(3) / 2
    # Triangle edges
    tri_x = [0.5, 0.0, 1.0, 0.5]
    tri_y = [h,   0.0, 0.0, h  ]
    ax.plot(tri_x, tri_y, color="#424242", linewidth=1.8, zorder=2)

    # Iso-weight grid lines at 0.25, 0.50, 0.75
    for iso in [0.25, 0.50, 0.75]:
        rem = 1.0 - iso
        # iso-SAFE  (s = iso)
        p1 = _ternary_to_xy(iso, rem, 0)
        p2 = _ternary_to_xy(iso, 0,   rem)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#BDBDBD", linewidth=0.6, linestyle="--", zorder=1)
        # iso-GREEDY  (g = iso)
        p1 = _ternary_to_xy(rem, iso, 0)
        p2 = _ternary_to_xy(0,   iso, rem)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#BDBDBD", linewidth=0.6, linestyle="--", zorder=1)
        # iso-REPAIR  (r = iso)
        p1 = _ternary_to_xy(rem, 0,   iso)
        p2 = _ternary_to_xy(0,   rem, iso)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="#BDBDBD", linewidth=0.6, linestyle="--", zorder=1)
        # Tick labels on edges
        ax.text(p2[0] - 0.04, p2[1], f"{iso:.2f}",
                fontsize=6, color="#9E9E9E", ha="right", va="center")

    # Vertex labels
    off = 0.07
    h   = math.sqrt(3) / 2
    ax.text(0.5,    h + off,  "SAFE",   ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=_C["safe"])
    ax.text(0 - off, -0.05,  "GREEDY", ha="center", va="top",
            fontsize=12, fontweight="bold", color=_C["greedy"])
    ax.text(1 + off, -0.05,  "REPAIR", ha="center", va="top",
            fontsize=12, fontweight="bold", color=_C["repair"])


def figure_z_simplex(
    logs: List[Dict[str, Any]],
    title: str = "Strategy State $z_t$ on SAFE–GREEDY–REPAIR Simplex",
) -> plt.Figure:
    """
    Ternary simplex plot.
    Colour encodes time (viridis); thicker/brighter segments in SURVIVAL mode.
    × = SIT event;  ○ = start;  ■ = end.
    """
    z_arr    = np.array([r["z"] for r in logs])            # (T, 3)
    mode     = [r["mode"] for r in logs]
    sit_mask = np.array([r["sit_event"] for r in logs], dtype=bool)
    T        = len(logs)

    xs, ys = np.array([_ternary_to_xy(*z) for z in z_arr]).T
    cmap   = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, T))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))
    fig.suptitle(title, fontsize=11, y=0.99)

    _draw_simplex(ax)

    # Trajectory segments coloured by time
    for i in range(T - 1):
        surv = (mode[i] == "survival")
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color=colors[i],
                alpha=0.85 if surv else 0.50,
                linewidth=2.2 if surv else 1.2,
                zorder=3)

    # SIT event markers
    sit_idx = np.where(sit_mask)[0]
    if len(sit_idx):
        ax.scatter(xs[sit_idx], ys[sit_idx],
                   marker="X", s=110, color=_C["sit"], zorder=6,
                   edgecolors="white", linewidths=0.6, label="SIT event")

    # Start / end
    ax.scatter(xs[0],  ys[0],  marker="o", s=90, color="#212121",
               zorder=7, edgecolors="white", linewidths=1, label="Start")
    ax.scatter(xs[-1], ys[-1], marker="s", s=90, color="#212121",
               zorder=7, edgecolors="white", linewidths=1, label="End")

    # Colourbar (time)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=1, vmax=T))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Tick $t$", fontsize=9)

    # Mode legend proxy
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="#888888", linewidth=2.2, label="SURVIVAL (thick)"),
        Line2D([0], [0], color="#888888", linewidth=1.2, alpha=0.5,
               label="NORMAL (thin)"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=9,
              ncol=4, framealpha=0.85)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SIT paper figures from CATLM simulations."
    )
    parser.add_argument("--output", default="figures",
                        help="Output directory (default: figures/)")
    parser.add_argument("--ticks", type=int, default=120,
                        help="Simulation length in ticks/hours (default: 120)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--seeds-sensitivity", type=int, default=10,
                        help="Seeds per level in trait sensitivity (default: 10)")
    parser.add_argument("--eps", type=float, default=0.04,
                        help="SIT displacement threshold eps for paper demos "
                             "(default: 0.04; game default is 0.25). "
                             "Lower values produce more frequent SIT events.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF output (PNG only)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    _setup_style()

    # ── Profile & scenario setup ────────────────────────────────────────────
    # "crisis" scenario: high sociability (strong loneliness weight in Ct),
    # high cowardice (low θ = easier collapse entry), followed by active care.
    # This naturally generates SIT events for paper demonstration.
    profile = CatProfile(
        name="Nabi",
        activity=4,
        sociability=5,
        appetite=3,
        cowardice=5,   # θ ≈ 0.41 — low threshold, easier collapse entry
    )

    # First half: no care (IDLE forced) → second half: auto-policy care
    neglect_ticks = args.ticks // 3
    user_actions: List[Optional[Action]] = (
        [Action.IDLE] * neglect_ticks
        + [None] * (args.ticks - neglect_ticks)
    )

    def _save(fig: plt.Figure, stem: str) -> None:
        for ext in (["png"] if args.no_pdf else ["png", "pdf"]):
            path = os.path.join(args.output, f"{stem}.{ext}")
            fig.savefig(path, bbox_inches="tight")
            print(f"      → {path}")
        plt.close(fig)

    # ── 1. Main simulation ──────────────────────────────────────────────────
    # SIT config for paper demonstrations (lower eps than game default)
    # Paper demo SIT config:
    #   eps=0.04 (game default=0.25) — lower threshold to visualise structural transitions
    #   require_collapse_regime=False — capture BOTH crisis-entry and recovery transitions
    #   Game uses require_collapse_regime=True (SIT gated to collapse regime only)
    paper_sit_cfg = SITConfig(
        lam=0.25,
        eps=args.eps,
        k_persist=3,
        require_collapse_regime=False,
        insight_phi=0.7,
        insight_k_persist=3,
    )
    print(f"      SIT config: eps={args.eps} (game default=0.25), k_persist=3, "
          f"require_collapse_regime=False")

    print(f"[1/5] Running main simulation  (seed={args.seed}, ticks={args.ticks}) …")
    print(f"      Scenario: {neglect_ticks} ticks IDLE neglect → {args.ticks - neglect_ticks} ticks auto-care")
    logs = run_simulation(profile, n_ticks=args.ticks, seed=args.seed,
                          user_actions=user_actions, sit_config=paper_sit_cfg)
    n_sit = logs[-1]["sit_count"]
    print(f"      {n_sit} SIT event(s) detected.")

    # ── Fig 1 ───────────────────────────────────────────────────────────────
    print("[2/5] Figure 1: Main trajectory …")
    _save(
        figure_main_trajectory(
            logs,
            title=f"CATLM Main Trajectory  (seed={args.seed}, ticks={args.ticks})",
        ),
        "fig1_main_trajectory",
    )

    # ── Fig 2 ───────────────────────────────────────────────────────────────
    print("[3/5] Figure 2: MoE EMA resistance …")
    _save(figure_moe_resistance(logs), "fig2_moe_resistance")

    # ── Fig 3 ───────────────────────────────────────────────────────────────
    print("[4/5] Figure 3: Capacity ablation …")
    logs_ablated = run_simulation_ablated(profile, n_ticks=args.ticks, seed=args.seed,
                                          user_actions=user_actions,
                                          sit_config=paper_sit_cfg)
    _save(figure_capacity_ablation(logs, logs_ablated), "fig3_capacity_ablation")

    # ── Fig 4 ───────────────────────────────────────────────────────────────
    print(f"[5/5] Figure 4: Trait sensitivity  (n_seeds={args.seeds_sensitivity}) …")
    _save(
        figure_trait_sensitivity(
            n_ticks=args.ticks,
            n_seeds=args.seeds_sensitivity,
        ),
        "fig4_trait_sensitivity",
    )

    # ── Fig 5 ───────────────────────────────────────────────────────────────
    print("[5/5] Figure 5: z_t simplex …")
    _save(figure_z_simplex(logs), "fig5_z_simplex")

    print(f"\nAll figures saved to ./{args.output}/")
    print("  (Seed-sweep experiment → run: python3 src/exp_001_seed_sweep.py)")


if __name__ == "__main__":
    main()
