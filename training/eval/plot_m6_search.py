"""M6 v2 paired-ablation analysis (fixed-target visual discrimination) -- target-goal success +
mean SPL with 95% CIs + probability-of-improvement.

Reads training/eval/m6_search.csv (columns: arm,seed,level,episode,success,spl,steps_to_goal,
decoy_visit,pit_falls,jump_uses,near_frac,overlap_frac), where every (arm, seed) ran the SAME
held-out layout set (paired, byte-verified via m6_pairing_<arm>.csv). Produces:
  - training/eval/m6_search.png : four panels --
      (0) target-goal success rate per arm, 95% Wilson CI (headline; chance=1/3 line for ray1)
      (1) success rate per difficulty level per arm
      (2) target-goal SPL (fixed red target) per arm, 95% stratified-bootstrap CI (mean, see note)
      (3) decoy-visit rate per difficulty level per arm (discrimination-failure tracker)
  - training/eval/m6_poi.csv    : probability-of-improvement of pixel over ray1 (PRIMARY) and
                                  pixel over rayC (STEELMAN), on SPL over the 12 seed x level cells.

PRE-REGISTERED decision rule (locked before the full run; see .superpowers/sdd/progress.md "Stage B"):
  PRIMARY: PoI(pixel > ray1, SPL) >= 0.75 AND ray1's success CI contains 1/3 (confound-detector holds).
  STEELMAN (not gated, reported for context): PoI(pixel > rayC, SPL).

Metric note: this reports MEAN SPL (not IQM SPL as M5 did), with the CI still computed via rliable's
stratified bootstrap. SPL here is heavily zero-inflated on the low-success arm (ray1 fails ~64% of
episodes -> SPL=0), so the interquartile mean collapses toward zero (IQM SPL 0.026 vs mean 0.162 on
ray1) and would misrepresent the colour-blind arm relative to the headline numbers quoted throughout
the ledger and results doc. Mean SPL is the number that is consistent end to end.

PoI is computed over the 12 (seed x level) cells per arm -- each cell's mean SPL is treated as one
independent "run" (a single task) -- matching how the full-ablation headline PoI numbers were derived,
NOT the raw per-episode matrix M5 used (raw episodes give a materially different, less conservative PoI).

Honesty note: with 3 seeds/arm (12 seed x level cells), rliable's interval estimates + PoI are the tools
designed for this few-run regime, replacing a bare mean +/- SE.
"""
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rliable import library as rly, metrics

CSV = "training/eval/m6_search.csv"
ARMS = ["m6_pixel", "m6_ray1", "m6_rayc"]
LABELS = {
    "m6_pixel": "pixel (from-scratch CNN)",
    "m6_ray1": "ray1 (single-tag, chance floor)",
    "m6_rayc": "rayC (hand-told colour tags)",
}
COLORS = {"m6_pixel": "#4C78A8", "m6_ray1": "#F58518", "m6_rayc": "#54A24B"}
CHANCE = 1.0 / 3.0
POI_GATE = 0.75  # PRIMARY gate: PoI(pixel > ray1) on SPL, seed x level cells
# Bootstrap reps: stable well below the canonical 50k for a coarse >=0.75 gate + few-seed CIs.
# probability_of_improvement is O(tasks^2) per rep, so keep its reps lower. Bump for a final figure.
SPL_REPS = 5000
POI_REPS = 2000


def wilson_ci(successes, n, z=1.959963985):
    """95% Wilson score interval for a binomial proportion -- the same convention used for every
    B1-B5 probe gate in the ledger (e.g. probe5's Wilson CI (0.512, 0.700))."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - margin, center + margin


def seed_matrix(df, arm, level=None):
    d = df[df.arm == arm]
    if level is not None:
        d = d[d.level == level]
    d = d.sort_values(["seed", "level", "episode"])
    seeds = sorted(d.seed.unique())
    if not seeds:
        return np.empty((0, 0))
    # rows = seeds (runs), cols = held-out episodes (tasks); paired -> same tasks across arms
    return np.stack([d[d.seed == s]["spl"].to_numpy() for s in seeds])


def cell_matrix(df, arm, col):
    """rows = the 12 (seed, level) cells, each treated as one independent run; cols = 1 task
    (that cell's mean of `col`). This is the unit the headline PoI numbers were computed on."""
    d = df[df.arm == arm]
    seeds = sorted(d.seed.unique())
    levels = sorted(d.level.unique())
    rows = [[d[(d.seed == s) & (d.level == lv)][col].mean()] for s in seeds for lv in levels]
    return np.array(rows)


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def _ci(x):
    r = np.asarray(x).ravel()
    return float(r[0]), float(r[-1])


def main():
    df = pd.read_csv(CSV)
    order = [a for a in ARMS if not df[df.arm == a].empty]
    if not order:
        print("no arm data in", CSV)
        return 1

    levels = sorted(df.level.unique())

    # --- headline: target-goal success rate + 95% Wilson CI ---
    succ, succ_ci = {}, {}
    for a in order:
        d = df[df.arm == a]
        n = len(d)
        k = int(d["success"].sum())
        succ[a] = k / n
        succ_ci[a] = wilson_ci(k, n)

    # --- target-goal SPL: mean + 95% stratified-bootstrap CI (rliable) ---
    mean_fn = lambda x: np.array([np.mean(x)])
    spl_scores = {a: seed_matrix(df, a) for a in order}
    spl_agg, spl_cis = rly.get_interval_estimates(spl_scores, mean_fn, reps=SPL_REPS)

    # --- decoy-visit rate: headline (overall) + per difficulty level ---
    decoy = {a: df[df.arm == a]["decoy_visit"].mean() for a in order}
    decoy_by_level = {a: [df[(df.arm == a) & (df.level == lv)]["decoy_visit"].mean() for lv in levels]
                       for a in order}

    # --- success per difficulty level ---
    succ_by_level = {a: [df[(df.arm == a) & (df.level == lv)]["success"].mean() for lv in levels]
                      for a in order}

    # --- probability of improvement on SPL, over the 12 seed x level cells: primary + steelman ---
    poi_pairs = {}
    if "m6_pixel" in order:
        if "m6_ray1" in order:
            poi_pairs["pixel_vs_ray1_PRIMARY"] = (
                cell_matrix(df, "m6_pixel", "spl"), cell_matrix(df, "m6_ray1", "spl"))
        if "m6_rayc" in order:
            poi_pairs["pixel_vs_rayc_STEELMAN"] = (
                cell_matrix(df, "m6_pixel", "spl"), cell_matrix(df, "m6_rayc", "spl"))
    poi, poi_ci = {}, {}
    if poi_pairs:
        poi, poi_ci = rly.get_interval_estimates(poi_pairs, metrics.probability_of_improvement, reps=POI_REPS)

    poi_vals = {k: _scalar(v) for k, v in poi.items()}
    pd.DataFrame({
        "pair": list(poi.keys()),
        "poi": [poi_vals[k] for k in poi],
        "ci_lo": [_ci(poi_ci[k])[0] for k in poi],
        "ci_hi": [_ci(poi_ci[k])[1] for k in poi],
    }).to_csv("training/eval/m6_poi.csv", index=False)

    # --- figure: 2x2 ---
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 0: target-goal success rate + 95% Wilson CI (headline; PRIMARY + confound-detector)
    for i, a in enumerate(order):
        lo, hi = succ_ci[a]
        ax0.barh(i, hi - lo, left=lo, height=0.5, color=COLORS[a], alpha=0.75)
        ax0.vlines(succ[a], i - 0.22, i + 0.22, color="k", alpha=0.6)
    ax0.axvline(CHANCE, color="gray", linestyle="--", linewidth=1, label="chance (1/3)")
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels([LABELS[a] for a in order])
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("success rate")
    ax0.set_title("target-goal success (fixed red target) -- 95% Wilson CI", fontsize=10)
    ax0.legend(fontsize=8, loc="lower right")
    ax0.grid(True, axis="x", alpha=0.25)

    # Panel 1: success rate per difficulty level
    for a in order:
        ax1.plot(levels, succ_by_level[a], "-o", color=COLORS[a], label=LABELS[a])
    ax1.set_xlabel("difficulty level (L0 open -> L3 hidden/elevated/crowded)")
    ax1.set_ylabel("success rate")
    ax1.set_xticks(levels)
    ax1.set_ylim(0, 1)
    ax1.set_title("success per difficulty level", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    # Panel 2: target-goal SPL (fixed red target) -- mean + 95% stratified-bootstrap CI (rliable)
    for i, a in enumerate(order):
        lo, hi = _ci(spl_cis[a])
        ax2.barh(i, hi - lo, left=lo, height=0.5, color=COLORS[a], alpha=0.75)
        ax2.vlines(_scalar(spl_agg[a]), i - 0.22, i + 0.22, color="k", alpha=0.6)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([LABELS[a] for a in order])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("mean SPL")
    ax2.set_title("target-goal SPL (fixed red target) -- 95% stratified-bootstrap CI", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25)

    # Panel 3: decoy-visit rate per difficulty level (discrimination-failure tracker, kept separate from SPL)
    for a in order:
        ax3.plot(levels, decoy_by_level[a], "-o", color=COLORS[a], label=LABELS[a])
    ax3.set_xlabel("difficulty level (L0 open -> L3 hidden/elevated/crowded)")
    ax3.set_ylabel("decoy-visit rate")
    ax3.set_xticks(levels)
    ax3.set_ylim(0, 1)
    ax3.set_title("decoy-visit rate per difficulty level", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "M6 v2 ablation -- target-goal success + target-goal SPL (fixed red target)\n"
        "paired held-out layouts, 3 seeds/arm x 4 levels x 25 episodes",
        fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig("training/eval/m6_search.png", dpi=130)

    # --- console summary ---
    print("success rate (95% Wilson CI):",
          {LABELS[a]: (round(succ[a], 3), tuple(round(x, 3) for x in succ_ci[a])) for a in order})
    print("mean SPL (95% stratified-bootstrap CI):",
          {LABELS[a]: (round(_scalar(spl_agg[a]), 3), tuple(round(x, 3) for x in _ci(spl_cis[a])))
           for a in order})
    print("decoy-visit rate:", {LABELS[a]: round(decoy[a], 3) for a in order})
    print("PoI (SPL, seed x level cells):", {k: round(v, 3) for k, v in poi_vals.items()})

    primary_key = "pixel_vs_ray1_PRIMARY"
    ray1_ci_contains_chance = ("m6_ray1" in succ_ci) and (succ_ci["m6_ray1"][0] <= CHANCE <= succ_ci["m6_ray1"][1])
    passed = (primary_key in poi_vals and poi_vals[primary_key] >= POI_GATE and ray1_ci_contains_chance)
    print("PRIMARY PASS" if passed else "PRIMARY INCONCLUSIVE")
    print("confound-detector (ray1 CI contains 1/3):", ray1_ci_contains_chance)
    print("wrote training/eval/m6_search.png + training/eval/m6_poi.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
