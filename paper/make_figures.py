"""Figure 1 for the Phase 7 paper draft. Numbers from goals/typed-decision-heads/plans/phase7_options.md."""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

pts = [  # label, tokens/item, accuracy, marker, text offset
    ("E4B head (one pass)", 93, 77.2, "o", (6, -12)),
    ("E4B-it one letter", 121, 77.4, "s", (6, 4)),
    ("E4B-it thinking", 521, 80.8, "s", (-78, -12)),
    ("31B-it one letter", 124, 85.7, "D", (6, 4)),
    ("31B-it thinking (600 items)", 495, 85.8, "D", (-150, 6)),
]
casc = [(211, 80.0, 1.2), (260, 80.9, 1.1), (310, 81.3, 0.8), (361, 81.4, 0.4)]   # head router, 3 seeds
base = [(240, 79.3), (297, 79.8), (351, 80.5), (401, 80.7)]                          # E4B-it letter-confidence router
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for name, x, y, m, off in pts:
    ax.plot(x, y, m, color="black", ms=6)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=8)
ax.errorbar([c[0] for c in casc], [c[1] for c in casc], yerr=[c[2] for c in casc], fmt="-o", color="C0", ms=4,
            capsize=2, label="Cascade, head router (20-50% escalated)")
ax.plot([b[0] for b in base], [b[1] for b in base], "--^", color="C1", ms=4, label="Cascade, chat-confidence router")
ax.set_xlabel("Tokens per question (read + written)")
ax.set_ylabel("Accuracy, 10 unseen tasks (%)")
ax.set_xlim(50, 560); ax.set_ylim(75, 88)
ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("fig_tokens_accuracy.pdf")
print("saved")
