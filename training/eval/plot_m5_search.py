"""M5 paired-ablation analysis — IQM SPL + stratified-bootstrap CIs + probability-of-improvement.

Reads training/eval/m5_search.csv (columns: arm,seed,level,episode,success,spl,steps_to_goal,
pit_falls,jump_uses,near_frac,overlap_frac), where every (arm, seed) ran the SAME held-out layout
set (paired). Produces:
  - training/eval/m5_search.png : left = IQM SPL per arm with 95% stratified-bootstrap CIs (headline);
                                  right = IQM SPL per difficulty level per arm.
  - training/eval/m5_poi.csv    : probability-of-improvement of the primary over each single-ablation.

PRE-REGISTERED decision rule (locked before any real run): PASS iff
  PoI(primary vs no-LSTM) >= 0.75  AND  PoI(primary vs no-RND) >= 0.75
else INCONCLUSIVE -> add seeds 4-5. (baseline = neither component; shown for context, not gated.)

Honesty note: with 3 seeds/arm the CIs are stratified-bootstrap over a SMALL run count; rliable's
interval estimates + probability-of-improvement are exactly the tools designed for this few-run regime,
which is why they replace a bare mean +/- SE.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rliable import library as rly, metrics, plot_utils

CSV = "training/eval/m5_search.csv"
ARMS = ["m5_primary", "m5_nolstm", "m5_nornd", "m5_baseline"]
LABELS = {"m5_primary": "LSTM+RND", "m5_nolstm": "RND only (no LSTM)",
          "m5_nornd": "LSTM only (no RND)", "m5_baseline": "baseline (neither)"}
COLORS = {"m5_primary": "#4C78A8", "m5_nolstm": "#F58518",
          "m5_nornd": "#54A24B", "m5_baseline": "#B279A2"}
POI_GATE = 0.75
# Bootstrap reps: stable well below the canonical 50k for a coarse >=0.75 gate + few-seed CIs.
# probability_of_improvement is O(tasks^2) per rep, so keep its reps lower. Bump for a final figure.
IQM_REPS = 5000
POI_REPS = 2000


def main():
    df = pd.read_csv(CSV)

    def score_matrix(arm, level=None):
        d = df[df.arm == arm]
        if level is not None:
            d = d[d.level == level]
        d = d.sort_values(["seed", "level", "episode"])
        seeds = sorted(d.seed.unique())
        if not seeds:
            return np.empty((0, 0))
        # rows = seeds (runs), cols = held-out episodes (tasks); paired -> same tasks across arms
        return np.stack([d[d.seed == s]["spl"].to_numpy() for s in seeds])

    scores = {a: score_matrix(a) for a in ARMS if not df[df.arm == a].empty}
    if not scores:
        print("no arm data in", CSV)
        return 1

    iqm = lambda x: np.array([metrics.aggregate_iqm(x)])
    agg, cis = rly.get_interval_estimates(scores, iqm, reps=IQM_REPS)

    # --- probability of improvement: primary vs each SINGLE ablation ---
    poi, poi_ci = {}, {}
    if "m5_primary" in scores:
        pairs = {f"primary_vs_{a.split('_', 1)[1]}": (scores["m5_primary"], scores[a])
                 for a in ("m5_nolstm", "m5_nornd") if a in scores}
        if pairs:
            poi, poi_ci = rly.get_interval_estimates(
                pairs, metrics.probability_of_improvement, reps=POI_REPS)

    # probability_of_improvement returns a SCALAR aggregate, so poi[k] is 0-d and poi_ci[k] is shape (2,)
    # or (2,1) — ravel to be robust to either.
    def _scalar(x):
        return float(np.asarray(x).ravel()[0])

    def _ci(x):
        r = np.asarray(x).ravel()
        return float(r[0]), float(r[-1])

    poi_vals = {k: _scalar(poi[k]) for k in poi}
    pd.DataFrame({
        "pair": list(poi.keys()),
        "poi": [poi_vals[k] for k in poi],
        "ci_lo": [_ci(poi_ci[k])[0] for k in poi],
        "ci_hi": [_ci(poi_ci[k])[1] for k in poi],
    }).to_csv("training/eval/m5_poi.csv", index=False)

    # --- figure ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    order = [a for a in ARMS if a in scores]
    plot_utils.plot_interval_estimates(
        {LABELS[a]: agg[a] for a in order}, {LABELS[a]: cis[a] for a in order},
        metric_names=["IQM SPL"], algorithms=[LABELS[a] for a in order], ax=ax0)
    ax0.set_title("M5 ablation — IQM SPL (95% stratified-bootstrap CI)\n"
                  "paired held-out layouts, 3 seeds/arm", fontsize=10)

    levels = sorted(df.level.unique())
    for a in order:
        ys = []
        for lv in levels:
            m = score_matrix(a, lv)
            ys.append(float(metrics.aggregate_iqm(m)) if m.size else np.nan)
        ax1.plot(levels, ys, "-o", color=COLORS[a], label=LABELS[a])
    ax1.set_xlabel("difficulty level (L0 open -> L3 hidden/elevated/crowded)")
    ax1.set_ylabel("IQM SPL")
    ax1.set_xticks(levels)
    ax1.set_ylim(0, 1)
    ax1.set_title("IQM SPL per difficulty level", fontsize=10)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("training/eval/m5_search.png", dpi=130)

    # --- pre-registered verdict ---
    print("IQM SPL:", {LABELS[a]: round(_scalar(agg[a]), 3) for a in order})
    print("PoI (primary > ablation):", {k: round(v, 3) for k, v in poi_vals.items()})
    passed = bool(poi_vals) and all(v >= POI_GATE for v in poi_vals.values())
    print("PASS" if passed else "INCONCLUSIVE -> add seeds 4-5")
    print("wrote training/eval/m5_search.png + training/eval/m5_poi.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
