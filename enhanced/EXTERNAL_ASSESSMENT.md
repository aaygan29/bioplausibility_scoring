# Is the PRISM bioplausibility metric a good one? — external assessment

Reviewer note on the PRISM-BioPlausibility project (BPS-P / FVS / rBPV, Proto integration).

## What the project set out to do
Score whether a protein edit is (1) **bioplausible** (looks like a natural mutation) and
(2) **functionally viable** (still works), to flag the biosecurity-dangerous case: an edit
that evades an AI screen AND is plausible AND still functions.

## Two design choices that are correct
- **BPS-P = weighted sum** (soft "looks natural") vs **FVS = product of necessary factors**
  (P_prod · V_fold · F_site · E) is the right modelling split. A conjunction correctly
  zeroes on a single catastrophic failure (frameshift, active-site deletion) where an
  average would report "75% fine."
- The **empirical study is the strongest part**: real DMS ground truth (ProteinGym, 10
  assays / 9 protein classes), out-of-fold evaluation, a shuffle-label control, a paired
  sign test, and the clear conclusion "do not naive-average; learn cross-validated weights."
  That finding (learned > naive in 10/10) is sound and well-supported.

## The problem: the headline AUC is inflated by POSITION LEAKAGE
The reported ceiling — GBM 0.811 mean, **0.956 on CcdB**, and the pLDDT single-signal 0.806 —
is not an honest estimate of performance on *new edits*, because of how the two strongest
signals behave under the chosen cross-validation.

1. **The pLDDT-context signal is a pure function of position.** It is `1 - pLDDT[i]/100`:
   every one of the ~19 substitutions at a residue gets the *identical* value. Verified:
   max distinct values at any single position = **1**.
2. **CcdB is saturation mutagenesis** — every position appears many times, with a
   near-shared label (destabilizing positions kill most substitutions).
3. **Random / dataset-fold CV puts the same positions in train and test.** So a model can
   memorize "position → label." A predictor using *position identity alone* scores
   **AUC 1.000** under random-fold CV and **0.424 (chance)** under position-grouped CV.
   The project's own leakage checks (random re-split, shuffle labels, Spearman) do **not**
   catch this, because random re-splitting reproduces the same position overlap.

### The honest number
Re-run with **GroupKFold on edited position** (no position in both train and test):

| CV scheme | GBM (pLDDT + ESM) AUC | "position alone" AUC |
|---|---|---|
| random / dataset fold (project) | **0.956** | 1.000 |
| position-grouped (honest)       | **0.785** | 0.424 |

The GBM keeps real skill (0.785 ≫ chance) — mostly from ESM, which *does* vary per
substitution (mean ~12 distinct values/position) — but **~0.17 AUC of the headline was
position memorization**, not edit-level discrimination.

## Why this matters for the metric's stated purpose
The tool is meant to score a *specific edit* an attacker proposes at a chosen site. In that
use, the model has never seen labels at that residue — exactly the grouped-CV regime, not
the random-fold regime. So 0.78, not 0.95, is the number that reflects deployment. The
ranking of conclusions (structure + ESM strong; hand-crafted physicochemical weak; learned
> naive) survives; only the absolute ceiling is overstated.

## Verdict
Directionally good, quantitatively over-optimistic.
- **Keep:** additive-vs-multiplicative split; DMS validation; learned non-linear combiner;
  shuffle/Spearman controls; the rBPV algebra (applicability × reliability gating is the
  right robustness idea).
- **Fix before publishing a number:**
  1. Report **position-grouped (leave-position-out) CV** as the primary metric. Keep
     random-fold only as an upper bound, labelled as such.
  2. Treat the **pLDDT-context term with suspicion** — as a per-position feature it cannot
     discriminate among substitutions at a site and is the main leakage vector. Either drop
     it from the edit-level score or replace with a variant-dependent structural term
     (real ΔΔG / RaSP, or folded-variant pLDDT ratio), which the docs already flag as TODO.
  3. The **FVS branch is still unvalidated** — V_fold uses a Grantham-scaled ddG *stub* and
     F_site needs real UniProt active sites. FVS AUC is at chance until those are wired, as
     the notebook honestly states. No FVS claim should be made yet.
  4. ESM rank-normalization is **transductive** (uses the test-fold feature distribution);
     fit it train-only to remove a smaller, second leakage.

## Bottom line on "is the current metric good?"
As a **research finding about how to combine signals**: yes, credible and useful.
As a **deployable bioplausibility score with the reported accuracy**: not yet — the ~0.81/0.96
numbers are inflated by position leakage under random-fold CV; the honest edit-level AUC is
~0.78, and the FVS half is not yet validated at all.
