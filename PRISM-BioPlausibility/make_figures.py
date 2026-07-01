"""Generate lab-notebook figures from the recorded experimental results."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
TEAL, INK, MUT, AMB = "#0E8C7F", "#1C2B33", "#9AA8AD", "#E08A3C"

assays = ["CCDB_Adkar", "CCDB_Tripathi", "BLAT", "GFP", "SPG1", "PTEN",
          "UBC9", "NUD15", "AICDA", "POLG"]
naive = [0.653, 0.778, 0.698, 0.640, 0.609, 0.725, 0.671, 0.750, 0.683, 0.600]
learn = [0.773, 0.836, 0.750, 0.664, 0.646, 0.751, 0.718, 0.799, 0.697, 0.636]
gbm   = [0.956, 0.929, 0.780, 0.726, 0.881, 0.812, 0.734, 0.867, 0.736, 0.691]

# Fig 1: strategy comparison (mean + worst-case)
strat = ["naive\naverage", "reliability\nweighted", "best single", "learned\nlogistic", "learned\nGBM"]
mean = [0.681, 0.713, 0.715, 0.727, 0.811]
worst = [0.600, 0.610, 0.606, 0.636, 0.691]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(strat)); w = 0.38
ax.bar(x - w/2, mean, w, label="mean AUC", color=TEAL)
ax.bar(x + w/2, worst, w, label="worst-case AUC", color=MUT)
ax.axhline(0.5, ls="--", c="#cccccc", lw=1)
ax.set_xticks(x); ax.set_xticklabels(strat, fontsize=9)
ax.set_ylim(0.45, 0.88); ax.set_ylabel("AUC"); ax.legend(frameon=False)
ax.set_title("Fig 1. How to combine signals (10-assay out-of-fold)")
for i, v in enumerate(mean): ax.text(i - w/2, v + .005, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig1_strategy_comparison.png", dpi=150); plt.close()

# Fig 2: per-assay naive vs GBM
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(assays)); w = 0.4
ax.bar(x - w/2, naive, w, label="naive average", color=MUT)
ax.bar(x + w/2, gbm, w, label="learned GBM", color=TEAL)
ax.axhline(0.5, ls="--", c="#cccccc", lw=1)
ax.set_xticks(x); ax.set_xticklabels(assays, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("AUC"); ax.set_ylim(0.45, 1.0); ax.legend(frameon=False)
ax.set_title("Fig 2. Per-assay: naive average vs learned GBM")
plt.tight_layout(); plt.savefig("figures/fig2_per_assay.png", dpi=150); plt.close()

# Fig 3: single-signal AUC (CcdB Adkar)
sig = ["pLDDT\ncontext", "ESM", "hydropathy", "sec.\nstruct", "aggregation", "burial\n(WCN)", "Grantham", "BLOSUM62"]
val = [0.806, 0.722, 0.642, 0.626, 0.600, 0.591, 0.532, 0.524]
fig, ax = plt.subplots(figsize=(8, 4))
colors = [TEAL if v >= 0.65 else MUT for v in val]
ax.barh(range(len(sig))[::-1], val, color=colors)
ax.set_yticks(range(len(sig))[::-1]); ax.set_yticklabels(sig, fontsize=8)
ax.axvline(0.5, ls="--", c="#cccccc", lw=1); ax.set_xlim(0.45, 0.85)
ax.set_xlabel("single-signal AUC (CcdB Adkar)")
ax.set_title("Fig 3. Which signals carry the power")
for i, v in enumerate(val): ax.text(v + .004, len(sig)-1-i, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig3_single_signals.png", dpi=150); plt.close()

# Fig 4: iterative improvement of the learned mean AUC
steps = ["hand-crafted\nlogistic", "+ ESM\n+ WCN burial", "+ non-linear\n(GBM)"]
vals = [0.682, 0.727, 0.811]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(steps, vals, "-o", color=TEAL, lw=2, ms=9)
for i, v in enumerate(vals): ax.text(i, v + .006, f"{v:.3f}", ha="center", fontsize=10, color=INK)
ax.set_ylim(0.64, 0.84); ax.set_ylabel("mean AUC (10 assays)")
ax.set_title("Fig 4. Iterative improvement of the mechanism")
plt.tight_layout(); plt.savefig("figures/fig4_iterative.png", dpi=150); plt.close()

# Fig 5: verification controls (CcdB Adkar GBM)
labs = ["dataset\nfolds", "random\n5-fold", "shuffled\nlabels"]
vals = [0.956, 0.958, 0.479]
cols = [TEAL, TEAL, AMB]
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar(labs, vals, color=cols)
ax.axhline(0.5, ls="--", c="#cccccc", lw=1)
ax.set_ylim(0, 1.05); ax.set_ylabel("AUC")
ax.set_title("Fig 5. Leakage controls (CcdB GBM = 0.956 is real)")
for i, v in enumerate(vals): ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig5_controls.png", dpi=150); plt.close()

print("wrote figures/fig1..fig5")
