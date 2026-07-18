# Robustness Audit — Bioplausibility Metric

*What was fragile, what each fix did, and the honest headline numbers with
uncertainty. All numbers are leave-one-protein-out (LOPO) on 10 ProteinGym DMS
assays / 9 protein classes / 36,374 single-and-multi variants, unless stated.*

Summary figure: `robustness_summary.png`. Production model:
`data/enhanced_model_leakfree.joblib` (version `leakfree-v2`).

---

## Headline numbers (leak-free, deployment-realistic)

| metric | zero-shot ESM2 | **enhanced** | lift (95% CI) |
|---|---|---|---|
| mean LOPO AUC | 0.660 | **0.719** [0.690, 0.747] | **+0.059 [0.022, 0.102]** |
| mean LOPO Spearman | 0.295 | **0.406** [0.347, 0.465] | **+0.111 [0.042, 0.193]** |

The enhanced metric beats zero-shot ESM2 on **8/9 proteins** by both metrics; the
paired lift CI excludes zero on **8/9** proteins for Spearman and **6/9** for AUC.
BLAT is the one protein where the two are statistically indistinguishable (ESM2
already scores a well-characterized β-lactamase well).

Everything below is the evidence that these numbers are real and not an artifact.

---

## 1. Transductive ESM normalization — fixed (`leakfree_pipeline.py`)

**Was fragile:** the previous build rank-normalized ESM2 scores *within each
assay's whole variant population*. That is transductive — it assumes you score a
full batch at once, which you never do at deployment (you score one produced
molecule). It also discarded the magnitude of the ESM log-likelihood ratio.

**Fix:** keep ESM2 wt-marginal **raw** (per-variant log-ratio in nats), pointwise.
Any scaling used downstream is fit train-only inside each LOPO fold.

**Effect:** mean LOPO AUC **0.703 → 0.719**. Removing the transductive step both
eliminated the batch-normalization leak *and raised* the honest number, because the
raw magnitude carries signal that rank-within-batch threw away. (Panel A.)

## 2. ProteinGym-standard Spearman added (`metric_comparison.csv`)

AUC on a binarized label is half the field-standard evaluation. Added per-protein
**Spearman ρ** between continuous P(tolerated) and the continuous DMS score — the
ProteinGym primary metric — computed LOPO. The enhanced metric's relative gain is
*larger* on Spearman (+0.111) than on AUC (+0.059).

## 3. Bootstrap 95% CIs on every headline (`bootstrap_cis.csv`, `bootstrap_forest.png`)

2,000 within-protein resamples for per-protein CIs; 5,000 cluster-over-proteins
resamples for pooled CIs. Point estimates are now intervals; the paired-lift CIs
give the significance test above. Pooled lifts both exclude zero.

## 4. Model + feature ablation (`ablation_table.csv`, `ablation_figure.png`)

- **Combiner ladder:** ESM2-only 0.660 → L2-logistic 0.687 → gradient boosting
  0.719. Both combiners beat zero-shot; the non-linear model adds a further real
  step (biophysical signals interact non-linearly with ESM2). (Panel D.)
- **Leave-one-feature-out:** dropping *any* of the 7 features hurts both metrics —
  no dead weight. ESM2 is the backbone (−0.036 AUC when removed); the six
  biophysical signals each add a small complementary increment.
- **LOPO permutation importance:** ESM2 dominates (0.100 AUC drop when shuffled,
  ~3× the next); no single biophysical feature dominates. (Panel E.)
- **plddt_region confirmed as leakage, not signal:** adding it back *hurts* under
  LOPO (−0.011 AUC, −0.018 Spearman). Its value in the original was random-fold
  position memorization. Dropping it was correct.

## 5. Residual-leakage audit (`leakage_audit.csv`, `leakage_audit.png`)

Independent stress test — within each protein, leave-position-out CV:

| | AUC |
|---|---|
| position-identity only, **random-fold** CV | 0.828 *(pure memorization)* |
| position-identity only, **held-out-position** CV | **0.555** *(≈ chance)* |
| **enhanced metric**, held-out-position CV | **0.707** |

This is the definitive picture: the inflated ~0.81/0.96 numbers reported earlier
were a model learning *which position → which label*, which evaporates the moment
positions are held out. The enhanced metric, in contrast, holds at 0.707 under the
same held-out-position CV — it carries genuine substitution-dependent signal, not
position identity. (Panel C.)

## 6. Calibration robustness (`calibration_metrics.csv`, `calibration_robustness.png`)

Refit the calibrator **train-only** per LOPO fold (not pooled). Honest finding:

| calibration | mean per-protein ECE | aggregate ECE |
|---|---|---|
| raw GBM probabilities | **0.118** | **0.050** |
| isotonic (fit on other proteins) | 0.126 | 0.062 |
| Platt (fit on other proteins) | 0.126 | 0.067 |

Post-hoc calibration fit on *other* proteins does **not** transfer to a held-out
protein — it slightly worsens ECE. The earlier 0.038→0.027 improvement came from a
*pooled* (transductive) calibrator that had seen the test protein's distribution.
**Deployment rule:** use the GBM's native probabilities; recalibrate on-target only
if per-protein labels become available. Ranking (AUC) is identical across all three
(monotone maps), so this is purely about probability quality.

## 7. Seed stability (`seed_stability.csv`, `seed_stability.png`)

12 random seeds: mean LOPO AUC **0.719 ± 0.001** (total spread 0.004), Spearman
**0.406 ± 0.002**. Run-to-run variance is ~40× smaller than the improvement over
ESM2. The headline is seed-independent. (Panel F.)

## 8. Production model refit

`data/enhanced_model_leakfree.joblib` — `HistGradientBoostingClassifier`
(max_depth 3, 200 iters, lr 0.06, L2 1.0) on all 36,374 variants, 7 leak-free
pointwise features, **raw probabilities** (no post-hoc calibrator, per §6).
Metric: `P(tolerated) = model.predict_proba(x)[:, 1]`; Proto constraint =
`1 − P(tolerated)`.

---

## Still outstanding — compute-gated, NOT done here

These require GPU folding/inverse-folding and cannot run on the current CPU-only
box (no remote compute configured). They are the honest remaining gaps:

1. **Reference-free structural validation** — scTM self-consistency
   (ProteinMPNN/SolubleMPNN → ESMFold round-trip) and folded-variant pLDDT/pAE for
   *de novo* produced molecules with no wild-type. The panel constraints for this
   are plumbed (`proto_panel.py`) but unvalidated — no DMS label exists for a
   reference-free design.
2. **Functional (FVS) half** — replace the ΔΔG stub with real RaSP/FoldX and
   UniProt/M-CSA catalytic residues; validate against activity-specific DMS.
3. **Broader benchmark** — extend beyond 9 proteins to more Pfam families and to
   indel / clinical ProteinGym sets.

The statistical-robustness work above is complete and deployment-honest for the
**referenced** regime (produced molecule vs a natural wild-type). The
**reference-free** regime remains the main scientific gap.
