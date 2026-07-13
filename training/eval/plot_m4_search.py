"""Render the M4 ablation eval: reach rate + early-window exploration vs goal visibility, per arm.
The teeth are the reach-rate CURVE across visibility -- the mid points (0.6, 0.35) show the primary
degrading gracefully while ablations fall off a cliff."""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("training/eval/m4_search.csv")))
arms = ["m4_primary", "m4_nolstm", "m4_nocuriosity", "m4_baseline"]
labels = {"m4_primary": "LSTM+curiosity (primary)", "m4_nolstm": "no LSTM",
          "m4_nocuriosity": "no curiosity", "m4_baseline": "baseline (neither)"}
colors = {"m4_primary": "#4C78A8", "m4_nolstm": "#F58518",
          "m4_nocuriosity": "#54A24B", "m4_baseline": "#B279A2"}

goals = defaultdict(list); cover = defaultdict(list); vis = defaultdict(list)
for r in rows:
    a = r["arm"]
    vis[a].append(float(r["vis_frac"]))
    goals[a].append(float(r["goals_per_agent_per_1k"]))
    cover[a].append(float(r["coverage_early_cells_per_agent"]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for a in arms:
    if a not in vis:
        continue
    order = sorted(range(len(vis[a])), key=lambda i: vis[a][i])
    x = [vis[a][i] for i in order]
    ax1.plot(x, [goals[a][i] for i in order], "-o", color=colors[a], label=labels[a])
    ax2.plot(x, [cover[a][i] for i in order], "-o", color=colors[a], label=labels[a])
ax1.set_ylabel("goals / agent / 1k steps")
ax1.set_title("M4 hidden-goal search: reach rate vs goal visibility (right = visible, left = hidden)")
ax1.legend(fontsize=8)
ax2.set_ylabel("cells visited / agent (first 500 steps)")
ax2.set_xlabel("goal visibility (fraction of arena diagonal)")
ax2.set_title("Early-window exploration (higher = searches more)")
plt.tight_layout()
plt.savefig("training/eval/m4_search.png", dpi=130)
print("wrote training/eval/m4_search.png")
