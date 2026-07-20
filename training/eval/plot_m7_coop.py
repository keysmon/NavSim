"""M7 cooperative-sacrifice analysis (MA-POCA credit assignment vs naive reward-sharing vs selfish
per-agent PPO) -- C1 task success + within-success cooperative fraction, with 95% CIs and
probability-of-improvement.

Reads training/eval/m7_coop.csv (columns: arm,seed,lesson,episode,success,steps,plate_hold_frac,
holder_idx,scorer_idx,both_on_plate_frac), where every (arm, seed) ran the SAME paired held-out
episode set at lesson 1 / C1 (byte-verified via m7_pairing_<arm>.csv). 75 rows/arm = 3 seeds x 25
episodes. arm in {poca, shared, selfish}; success in {0,1}; scorer_idx == -1 on failure.

Produces:
  - training/eval/m7_coop.png : four panels --
      (0) C1 SUCCESS rate per arm, 95% Wilson CI, with per-seed points overlaid (the CLEAN POSITIVE;
          per-seed dominance pre-empts the n=3 seed-clustering objection)
      (1) within-success COOPERATIVE fraction per arm, 95% Wilson CI (the REFUTED discriminator --
          the CIs overlap heavily; poca is nominally LOWEST, opposite the pre-registered prediction)
      (2) probability-of-improvement on success: poca-vs-shared, poca-vs-selfish, shared-vs-selfish
          (context), 0.5 chance line + 0.75 strong-effect bar
      (3) mean plate_hold_frac per arm (corroborating: selfish -- zero coop incentive -- holds MOST)
  - training/eval/m7_stats.csv : every computed number, long format (metric, group, value, ci_lo,
    ci_hi, n).

Cooperation discriminator definition (within successful episodes only):
  cooperative = holder_idx != scorer_idx and scorer_idx != -1  (one agent holds the plate while the
    OTHER crosses the door and scores -- a division of labour)
  solo        = holder_idx == scorer_idx                       (tap-and-sprint: the same agent taps
    the plate and crosses alone before the door re-closes)

PRE-REGISTERED claim (fixed before the eval): MA-POCA counterfactual credit assignment would show the
HIGHEST cooperative fraction (learned sacrifice); selfish ~0; shared in between. VERDICT: refuted --
poca is nominally the lowest, and the coop-fraction difference is ~1.2 sigma (a 2-proportion z-test),
i.e. NOT significant given the small success counts (53/24/26). The competence result (success) is
the clean positive; the cooperation discriminator did not validate the credit-assignment story at
this solo-solvable rung. See docs/M7-results.md.

Honesty note (mirrors M6's PoI-construction disclosure): PoI on BINARY success is mechanically
~ 0.5 + 0.5*(success-rate gap) and its ties (failure==failure) pull it toward 0.5, so it carries no
information beyond the success CIs -- the headline rests on the fully separated Wilson success CIs, and
PoI is reported only for method-parity with M5/M6.
"""
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rliable import library as rly, metrics

CSV = "training/eval/m7_coop.csv"
LESSON = 1  # C1 -- the only rung both learnable AND evaluated (C2 was 0/100 for all arms; see doc)
ARMS = ["poca", "shared", "selfish"]
LABELS = {
    "poca": "poca (MA-POCA credit)",
    "shared": "shared (naive reward-sharing)",
    "selfish": "selfish (per-agent PPO)",
}
COLORS = {"poca": "#4C78A8", "shared": "#F58518", "selfish": "#54A24B"}
SEED_COLORS = ["#333333", "#777777", "#bbbbbb"]
POI_STRONG = 0.75  # the M5/M6 pre-registered "strong effect" bar (reference line only, not a gate here)
POI_REPS = 5000


def wilson_ci(successes, n, z=1.959963985):
    """95% Wilson score interval for a binomial proportion -- the convention used for every M-series
    probe gate (matches plot_m6_search.py)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - margin, center + margin


def two_prop_z(k1, n1, k2, n2):
    """2-proportion pooled z-test (arm1 - arm2). Used to test whether poca's lower coop-fraction is
    significant or within-noise."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


def seed_matrix(df, arm, col="success"):
    """rows = seeds (runs), cols = paired held-out episodes (tasks) -> same tasks across arms."""
    d = df[df.arm == arm].sort_values(["seed", "episode"])
    seeds = sorted(d.seed.unique())
    return np.stack([d[d.seed == s][col].to_numpy() for s in seeds])


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def _ci(x):
    r = np.asarray(x).ravel()
    return float(r[0]), float(r[-1])


def main():
    df = pd.read_csv(CSV)
    df = df[df.lesson == LESSON]
    order = [a for a in ARMS if not df[df.arm == a].empty]
    if not order:
        print("no arm data in", CSV)
        return 1
    seeds = sorted(df.seed.unique())

    # --- (0) headline: C1 success rate + 95% Wilson CI + per-seed rates ---
    succ, succ_ci, succ_k, succ_n, per_seed = {}, {}, {}, {}, {}
    for a in order:
        d = df[df.arm == a]
        n, k = len(d), int(d.success.sum())
        succ[a], succ_ci[a], succ_k[a], succ_n[a] = k / n, wilson_ci(k, n), k, n
        per_seed[a] = [df[(df.arm == a) & (df.seed == s)]["success"].mean() for s in seeds]

    # --- (1) cooperation discriminator: within-success cooperative fraction + Wilson CI ---
    coop_frac, coop_ci, coop_k, coop_n = {}, {}, {}, {}
    for a in order:
        s = df[(df.arm == a) & (df.success == 1)]
        ns = len(s)
        kc = int(((s.holder_idx != s.scorer_idx) & (s.scorer_idx != -1)).sum())
        coop_k[a], coop_n[a] = kc, ns
        coop_frac[a] = kc / ns if ns else 0.0
        coop_ci[a] = wilson_ci(kc, ns)

    # significance of poca's lower coop-fraction (2-prop z-test vs each comparator)
    coop_z = {}
    if "poca" in order:
        for b in ("shared", "selfish"):
            if b in order and coop_n["poca"] and coop_n[b]:
                coop_z[f"poca_vs_{b}"] = two_prop_z(coop_k["poca"], coop_n["poca"], coop_k[b], coop_n[b])

    # --- (2) probability-of-improvement on success (per-episode paired matrix; method-parity only) ---
    poi_pairs = {}
    if "poca" in order and "shared" in order:
        poi_pairs["poca_vs_shared"] = (seed_matrix(df, "poca"), seed_matrix(df, "shared"))
    if "poca" in order and "selfish" in order:
        poi_pairs["poca_vs_selfish"] = (seed_matrix(df, "poca"), seed_matrix(df, "selfish"))
    if "shared" in order and "selfish" in order:
        poi_pairs["shared_vs_selfish"] = (seed_matrix(df, "shared"), seed_matrix(df, "selfish"))
    poi, poi_ci = ({}, {})
    if poi_pairs:
        poi, poi_ci = rly.get_interval_estimates(poi_pairs, metrics.probability_of_improvement, reps=POI_REPS)
    poi_vals = {k: _scalar(v) for k, v in poi.items()}

    # --- corroborating + mechanism metrics ---
    hold_all = {a: df[df.arm == a]["plate_hold_frac"].mean() for a in order}
    hold_succ = {a: df[(df.arm == a) & (df.success == 1)]["plate_hold_frac"].mean() for a in order}
    steps_succ, steps_solo, steps_coop = {}, {}, {}
    for a in order:
        s = df[(df.arm == a) & (df.success == 1)]
        steps_succ[a] = s["steps"].mean()
        steps_solo[a] = s[s.holder_idx == s.scorer_idx]["steps"].mean()
        steps_coop[a] = s[(s.holder_idx != s.scorer_idx) & (s.scorer_idx != -1)]["steps"].mean()

    # ============================== stats CSV (long format) ==============================
    rows = []
    for a in order:
        rows.append(("success_rate", LABELS[a], succ[a], succ_ci[a][0], succ_ci[a][1], succ_n[a]))
    for a in order:
        rows.append(("coop_fraction", LABELS[a], coop_frac[a], coop_ci[a][0], coop_ci[a][1], coop_n[a]))
    for a in order:
        rows.append(("plate_hold_frac_all", LABELS[a], hold_all[a], "", "", succ_n[a]))
        rows.append(("plate_hold_frac_success", LABELS[a], hold_succ[a], "", "", coop_n[a]))
    for a in order:
        rows.append(("mean_steps_success", LABELS[a], steps_succ[a], "", "", coop_n[a]))
        rows.append(("mean_steps_solo_success", LABELS[a], steps_solo[a], "", "", coop_k[a] and (coop_n[a] - coop_k[a])))
        rows.append(("mean_steps_coop_success", LABELS[a], steps_coop[a], "", "", coop_k[a]))
    for a in order:
        for i, s in enumerate(seeds):
            rows.append((f"per_seed_success_s{s}", LABELS[a], per_seed[a][i], "", "", 25))
    for k in poi_vals:
        rows.append(("poi_success", k, poi_vals[k], _ci(poi_ci[k])[0], _ci(poi_ci[k])[1], ""))
    for k, zz in coop_z.items():
        rows.append(("coop_frac_z_test", k, zz, "", "", ""))
    pd.DataFrame(rows, columns=["metric", "group", "value", "ci_lo", "ci_hi", "n"]).to_csv(
        "training/eval/m7_stats.csv", index=False)

    # ================================== figure: 2x2 ==================================
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 0: C1 success rate + 95% Wilson CI + per-seed points (the clean positive)
    for i, a in enumerate(order):
        lo, hi = succ_ci[a]
        ax0.barh(i, hi - lo, left=lo, height=0.5, color=COLORS[a], alpha=0.75)
        ax0.vlines(succ[a], i - 0.25, i + 0.25, color="k", alpha=0.7)
        for j, s in enumerate(seeds):
            ax0.plot(per_seed[a][j], i, "o", color=SEED_COLORS[j % len(SEED_COLORS)],
                     markersize=6, markeredgecolor="white", markeredgewidth=0.6,
                     label=f"seed {s}" if i == 0 else None, zorder=5)
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels([LABELS[a] for a in order])
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("C1 success rate")
    ax0.set_title("C1 task success -- 95% Wilson CI (bars) + per-seed rate (dots)\n"
                  "the clean positive: poca is top arm in every seed", fontsize=10)
    ax0.legend(fontsize=8, loc="lower right", title="per seed")
    ax0.grid(True, axis="x", alpha=0.25)
    ax0.invert_yaxis()

    # Panel 1: within-success cooperative fraction + 95% Wilson CI (the refuted discriminator)
    for i, a in enumerate(order):
        lo, hi = coop_ci[a]
        ax1.barh(i, hi - lo, left=lo, height=0.5, color=COLORS[a], alpha=0.75)
        ax1.vlines(coop_frac[a], i - 0.25, i + 0.25, color="k", alpha=0.7)
        ax1.text(hi + 0.02, i, f"{coop_frac[a]:.2f}  (n={coop_n[a]})", va="center", fontsize=8)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([LABELS[a] for a in order])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("cooperative fraction of successes (holder != scorer)")
    ax1.set_title("cooperation discriminator -- 95% Wilson CI (REFUTED)\n"
                  "pre-registered: poca HIGHEST; observed: lowest, ~1.2 sigma",
                  fontsize=9)
    ax1.grid(True, axis="x", alpha=0.25)
    ax1.invert_yaxis()

    # Panel 2: probability-of-improvement on success + CIs
    pk = [k for k in ("poca_vs_shared", "poca_vs_selfish", "shared_vs_selfish") if k in poi_vals]
    for i, k in enumerate(pk):
        lo, hi = _ci(poi_ci[k])
        ax2.barh(i, hi - lo, left=lo, height=0.5, color="#4C78A8" if k.startswith("poca") else "#999999",
                 alpha=0.75)
        ax2.vlines(poi_vals[k], i - 0.25, i + 0.25, color="k", alpha=0.7)
        ax2.text(hi + 0.01, i, f"{poi_vals[k]:.2f}", va="center", fontsize=8)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax2.axvline(POI_STRONG, color="firebrick", linestyle=":", linewidth=1, label="strong-effect bar (0.75)")
    ax2.set_yticks(range(len(pk)))
    ax2.set_yticklabels([k.replace("_", " ") for k in pk])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("P(improvement) on success")
    ax2.set_title("probability-of-improvement on success\n"
                  "(binary-tie-diluted; headline rests on the separated success CIs)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.invert_yaxis()

    # Panel 3: mean plate_hold_frac per arm (corroborating: selfish holds MOST)
    x = np.arange(len(order))
    ax3.bar(x - 0.2, [hold_all[a] for a in order], width=0.38, color=[COLORS[a] for a in order],
            alpha=0.85, label="all episodes")
    ax3.bar(x + 0.2, [hold_succ[a] for a in order], width=0.38, color=[COLORS[a] for a in order],
            alpha=0.45, hatch="//", label="successful episodes")
    ax3.set_xticks(x)
    ax3.set_xticklabels(order, fontsize=9)
    ax3.set_ylabel("mean plate_hold_frac")
    ax3.set_title("corroborating: mean plate-hold fraction\n"
                  "selfish (zero coop incentive) holds MOST; poca not elevated", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "M7 cooperative sacrifice -- MA-POCA credit vs naive reward-sharing vs selfish PPO\n"
        "paired held-out C1 (hard 4.0 s dwell), 3 seeds/arm x 25 episodes = 75 episodes/arm",
        fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.89)
    fig.savefig("training/eval/m7_coop.png", dpi=130)

    # ================================== console summary ==================================
    print("C1 success rate (95% Wilson CI):",
          {LABELS[a]: (round(succ[a], 3), tuple(round(x, 3) for x in succ_ci[a])) for a in order})
    print("per-seed success:", {LABELS[a]: [round(v, 2) for v in per_seed[a]] for a in order})
    print("coop fraction (95% Wilson CI, n=successes):",
          {LABELS[a]: (round(coop_frac[a], 3), tuple(round(x, 3) for x in coop_ci[a]), coop_n[a]) for a in order})
    print("coop-fraction 2-prop z (poca lower?):", {k: round(v, 3) for k, v in coop_z.items()})
    print("plate_hold_frac (all | success):",
          {LABELS[a]: (round(hold_all[a], 3), round(hold_succ[a], 3)) for a in order})
    print("mean steps on success (all | solo | coop):",
          {LABELS[a]: (round(steps_succ[a], 0), round(steps_solo[a], 0), round(steps_coop[a], 0)) for a in order})
    print("PoI on success:", {k: (round(poi_vals[k], 3), tuple(round(x, 3) for x in _ci(poi_ci[k]))) for k in poi_vals})
    print("wrote training/eval/m7_coop.png + training/eval/m7_stats.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
