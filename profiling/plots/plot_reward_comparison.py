#!/usr/bin/env python3
"""Plot pass@8 and reward over steps for all three APPS ablation runs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nothink_pass = [0.664, 0.684, 0.664, 0.652, 0.707]
think_pass = [0.359, 0.375, 0.336, 0.328, 0.406, 0.406]
rr_pass = [0.359, 0.387, 0.340, 0.324, 0.387, 0.402, 0.406, 0.395, 0.422]

nothink_reward = [0.478, 0.514, 0.504, 0.510, 0.539]
think_reward = [0.269, 0.304, 0.264, 0.323, 0.323, None]
rr_reward = [0.269, 0.306, 0.262, 0.269, 0.326, 0.319, 0.334, None, None]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# pass@8
ax1.plot(range(1, len(nothink_pass)+1), nothink_pass, 'o-', color='#2ca02c', lw=2, ms=6, label='Nothink')
ax1.plot(range(1, len(think_pass)+1), think_pass, 's-', color='#1f77b4', lw=2, ms=6, label='Think')
ax1.plot(range(1, len(rr_pass)+1), rr_pass, '^-', color='#ff7f0e', lw=2, ms=6, label='Round-Robin')
ax1.set_ylabel('pass@8', fontsize=12)
ax1.set_title('APPS Introductory — Think vs Nothink vs Round-Robin', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.25, 0.80)

# reward
nt_r = [(i+1, v) for i, v in enumerate(nothink_reward) if v is not None]
th_r = [(i+1, v) for i, v in enumerate(think_reward) if v is not None]
rr_r = [(i+1, v) for i, v in enumerate(rr_reward) if v is not None]

ax2.plot([x[0] for x in nt_r], [x[1] for x in nt_r], 'o-', color='#2ca02c', lw=2, ms=6, label='Nothink')
ax2.plot([x[0] for x in th_r], [x[1] for x in th_r], 's-', color='#1f77b4', lw=2, ms=6, label='Think')
ax2.plot([x[0] for x in rr_r], [x[1] for x in rr_r], '^-', color='#ff7f0e', lw=2, ms=6, label='Round-Robin')
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Avg Reward', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.20, 0.60)

plt.tight_layout()
out = "/projects/I20240005/rnouaj/skyrl/profiling_results/reward_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
