"""Render the M4 ablation result HONESTLY.

Top panel: per-arm mean reward on the hidden rung (L3) of training, the cleanest per-arm signal
(each summary is already a window-average, so the phase-mean is stable). This is the headline.
Bottom panels: the in-engine eval sweep (reach rate + early-window exploration vs goal visibility) at
the fixed hardest config. The reach metric is LOW and noisy here (compass-free ray search) and is
non-monotonic in visibility, so it is shown as exploratory, not as the primary evidence.

Caveat baked into the figure: n = 1 training seed per arm, so NO difference (incl. the baseline gap)
can be attributed to the mechanism vs. seed/init variance. Error bars are within-run temporal SE only.
"""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

arms = ["m4_primary", "m4_nolstm", "m4_nocuriosity", "m4_baseline"]
labels = {"m4_primary": "LSTM+curiosity", "m4_nolstm": "curiosity only\n(no LSTM)",
          "m4_nocuriosity": "LSTM only\n(no curiosity)", "m4_baseline": "baseline\n(neither)"}
colors = {"m4_primary": "#4C78A8", "m4_nolstm": "#F58518",
          "m4_nocuriosity": "#54A24B", "m4_baseline": "#B279A2"}

# --- per-arm hidden-rung (L3) training reward ---
l3 = {r["arm"]: r for r in csv.DictReader(open("training/eval/m4_l3_reward.csv"))}

# --- eval sweep ---
rows = list(csv.DictReader(open("training/eval/m4_search.csv")))
goals = defaultdict(list); cover = defaultdict(list); vis = defaultdict(list)
for r in rows:
    a = r["arm"]
    vis[a].append(float(r["vis_frac"]))
    goals[a].append(float(r["goals_per_agent_per_1k"]))
    cover[a].append(float(r["coverage_early_cells_per_agent"]))

fig, (axb, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 10))

# Panel 1: hidden-rung training reward per arm (headline, honest)
xs = list(range(len(arms)))
means = [float(l3[a]["mean_l3_reward"]) for a in arms]
ses = [float(l3[a]["se_within_run"]) for a in arms]
axb.bar(xs, means, yerr=ses, capsize=5, color=[colors[a] for a in arms])
axb.set_xticks(xs)
axb.set_xticklabels([labels[a].replace("\n", " ") for a in arms], fontsize=9)
axb.set_ylabel("mean reward on hidden rung (L3)")
axb.set_title("M4 ablation: hidden-goal-search training reward per arm\n"
              "(n=1 seed/arm — bars are within-run SE, NOT a mechanism-effect CI; "
              "baseline is stably lowest, mechanism arms cluster)", fontsize=10)
for i, a in enumerate(arms):
    axb.text(i, means[i] + ses[i] + 0.03, "%.2f" % means[i], ha="center", fontsize=9)

# Panel 2: eval reach rate vs visibility (exploratory)
for a in arms:
    if a not in vis:
        continue
    order = sorted(range(len(vis[a])), key=lambda i: vis[a][i])
    x = [vis[a][i] for i in order]
    ax1.plot(x, [goals[a][i] for i in order], "-o", color=colors[a],
             label=labels[a].replace("\n", " "))
ax1.set_ylabel("goals / agent / 1k steps")
ax1.set_xlabel("goal visibility (fraction of arena diagonal; right=visible, left=hidden)")
ax1.set_title("Eval reach rate vs visibility — LOW + noisy + non-monotonic (exploratory, not the headline)",
              fontsize=10)
ax1.legend(fontsize=8)

# Panel 3: eval early-window coverage vs visibility
for a in arms:
    if a not in vis:
        continue
    order = sorted(range(len(vis[a])), key=lambda i: vis[a][i])
    x = [vis[a][i] for i in order]
    ax2.plot(x, [cover[a][i] for i in order], "-o", color=colors[a],
             label=labels[a].replace("\n", " "))
ax2.set_ylabel("cells visited / agent (first 500 steps)")
ax2.set_xlabel("goal visibility (fraction of arena diagonal)")
ax2.set_title("Eval early-window exploration vs visibility", fontsize=10)

plt.tight_layout()
plt.savefig("training/eval/m4_search.png", dpi=130)
print("wrote training/eval/m4_search.png")
