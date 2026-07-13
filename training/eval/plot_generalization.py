"""Render the M3 generalization CSV as a grouped bar chart (diagonal vs off-diagonal)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("training/eval/m3_generalization.csv")))
tags = [r["tag"] for r in rows]
goals = [float(r["goals_per_agent_per_1k"]) for r in rows]
near = [float(r["near_frac"]) for r in rows]
fallbacks = [int(r["fallbacks"]) for r in rows]
colors = ["#4C78A8" if t.startswith("diag") else "#F58518" for t in tags]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.bar(tags, goals, color=colors)
ax1.set_ylabel("goals / agent / 1k steps")
ax1.set_title("M3 generalization (blue=trained diagonal, orange=held-out off-diagonal)")
# Flag any bar whose placement hit the ClearPoint fallback -- its collision numbers are suspect (honesty).
for i, f in enumerate(fallbacks):
    if f > 0:
        ax1.text(i, goals[i], "!%d" % f, ha="center", va="bottom", fontsize=8, color="#B00")

ax2.bar(tags, near, color=colors)
ax2.set_ylabel("near-encounter frame frac")
ax2.set_title("Collision proxy (lower = better)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("training/eval/m3_generalization.png", dpi=130)
print("wrote training/eval/m3_generalization.png")
