# Empirical findings: how to combine bioplausibility signals

We downloaded real CcdB DMS data (genbio-ai ProteinGYM-DMS) + the AlphaFold
structure, computed every biophysical signal per variant, and compared
combination strategies **out-of-sample** using the dataset's own 5-fold splits.
Tool: `empirical_bioplausibility.py`. Ground truth: measured DMS fitness.

## Result (CcdB Adkar 2012, 1,176 variants, 5-fold CV)

Single signals (AUC / |Spearman|):
| signal | AUC | |rho| |
|---|---|---|
| **structural region (pLDDT, where the edit lands)** | **0.806** | **0.526** |
| secondary structure | 0.620 | 0.224 |
| hydropathy | 0.606 | 0.191 |
| aggregation | 0.600 | 0.235 |
| grantham | 0.532 | 0.038 |
| BLOSUM62 | 0.524 | 0.059 |
| burial (CA-contact proxy) | 0.505 | 0.030 |

Combination strategies (out-of-fold):
| strategy | AUC | |rho| |
|---|---|---|
| (a) **naive average of all signals** | **0.558** | 0.109 |
| (b) reliability-weighted average | 0.672 | 0.313 |
| (c) learned logistic (CV) | 0.691 | 0.350 |
| best single signal | 0.806 | 0.526 |

Replicate (CcdB Tripathi 2016, 1,663) confirms by Spearman: structural region
strongest (0.355), naive average (0.231) does not beat it.

## What this means (the answer to "should we just average many metrics?")

1. **No. Naive averaging is actively bad.** It scored 0.558, worse than the best
   single signal (0.806) and worse than several individual signals. Averaging
   dilutes one or two strong signals with many weak / noisy ones.
2. **Weighting beats averaging.** Reliability-weighting (0.672) and learned
   weights (0.691) clearly beat the naive average (0.558). So combine by validated
   weight, not equal weight.
3. **Signal quality matters more than the combination rule.** One signal,
   structural context (where the edit lands, read from pLDDT), carried most of the
   predictive power. Hand-crafted physicochemical signals (Grantham, BLOSUM,
   burial proxy) were near chance on this assay. The strong earlier result
   (FVS 0.735) came from the ESM language-model signal, which is far stronger than
   any single hand-crafted feature here.

## Recommendation for our bioplausibility measure
- **Do not flat-average a long list of metrics.** Anchor on the few validated
  strong signals: structural context (pLDDT / where the edit lands), the ESM
  language-model marginal, and real ΔΔG (RaSP) when available.
- **Combine by cross-validated learned weights** (or reliability weighting), with
  weak physicochemical signals as small-weight complements.
- **Always validate out-of-sample** on DMS with the dataset's CV folds, as here.

## Breadth + depth: 10 assays, 9 protein classes (multi_assay_analysis.py)

Same comparison, now across toxin, resistance enzyme, fluorescent protein, binding
domain, phosphatase, SUMO enzyme, hydrolase, deaminase, and virus, with bootstrap
95% CIs and a cross-assay paired test.

Cross-assay summary (mean AUC over 10 assays / worst case):
| strategy | mean AUC | worst-case AUC |
|---|---|---|
| naive average | 0.632 | 0.565 |
| reliability-weighted | 0.661 | 0.594 |
| best single signal | 0.685 | 0.594 |
| **learned (cross-validated)** | **0.682** | **0.613** |

Headline statistics:
- **Learned weighting beats naive averaging in 10 / 10 assays** (sign test p ≈ 0.002),
  mean improvement +0.049 AUC.
- Naive averaging is the **worst** strategy on both mean and worst case, everywhere.
- Learned weighting ties the best single signal on the mean (0.682 vs 0.685) and has
  the **best worst case** (0.613 vs 0.594). Since you cannot know in advance which
  single signal will be best for a new protein, the learned combination is the
  robust, generalizable choice.

Conclusion (now supported across breadth and depth): **do not naive-average; learn
cross-validated weights.** The modest absolute AUCs (~0.68) reflect that these are
weak hand-crafted signals; adding the strong signals (ESM marginal, structural
context, real ΔΔG) raises the ceiling, but the combination rule is settled.

## Improving the mechanism: ESM signal + real burial + non-linear combiner

Two upgrades (add the ESM2 language-model signal; replace the crude burial proxy
with Weighted Contact Number, a validated CA-only RSA surrogate) plus a non-linear
combiner (gradient-boosted trees) transformed the result. 10 assays, out-of-fold:

| strategy | mean AUC | worst-case AUC |
|---|---|---|
| naive average | 0.681 | 0.600 |
| reliability-weighted | 0.713 | 0.610 |
| best single signal | 0.715 | 0.606 |
| learned logistic (CV) | 0.727 | 0.636 |
| **learned GBM (CV)** | **0.811** | **0.691** |

- Adding ESM + WCN burial lifted the learned-logistic mean from 0.682 to 0.727.
- A **non-linear combiner (gradient boosting)** then jumped it to **0.811 mean /
  0.691 worst** by capturing interactions among signals (e.g. "buried AND severe
  AND low-ESM") that a linear model cannot.
- The burial fix worked: WCN burial single-feature AUC rose 0.505 -> 0.591.

### Verified, not leakage (CcdB Adkar, GBM AUC 0.956)
- Dataset folds 0.956 vs fresh **random 5-fold 0.958** -> no fold-structure leakage.
- **Spearman(prediction, fitness) = +0.766** -> strong continuous corroboration.
- **Shuffled-labels control = 0.479** (~0.5) -> no spurious predictive power.
- Strong single features: structural context (pLDDT) 0.806, ESM 0.722; the
  hand-crafted physicochemical features remain weak (Grantham 0.53, BLOSUM 0.52).

### Final recommended mechanism
Strong signals (structural context, ESM marginal, burial-WCN, hydropathy) combined
by a **non-linear, cross-validated learned model (gradient boosting)** — NOT a naive
average, and not even a linear weighting. Validate out-of-sample on DMS with the
dataset's folds and a shuffle control.

## Honest caveats
- The burial term used a crude CA-contact proxy and underperformed the literature
  expectation (RSA is normally a top feature). Proper DSSP-based solvent
  accessibility is the fix, and is the highest-value signal upgrade.
- Binary AUC on Tripathi was degenerate (discretized labels); Spearman is the
  robust metric and was used to confirm the pattern.
- Validation target is functional tolerance (DMS), a proxy for "naturalness."
  A complementary check is RQ4: rank natural-ortholog substitutions vs random.
