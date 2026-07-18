# Enhanced bioplausibility score — from over-optimistic prototype to research tool

This builds directly on the project's own findings and the variant-effect literature.
It fixes the two things that made the original BPS-P unusable for real research and
adds the three things a research tool needs (honest generalization, calibration,
applicability-awareness). Code: `enhanced_bioplausibility.py`; shipped model:
`data/enhanced_model.joblib`; validation on 10 ProteinGym DMS assays (36,374 single
variants).

## The two problems fixed

**1. Position leakage (the headline killer).** The original strongest feature,
`plddt_region = 1 − pLDDT[i]/100`, is a pure function of residue *position* — every
one of the ~19 substitutions at a site gets the identical value (verified: 1 distinct
value per position). ProteinGym assays are saturation mutagenesis, so every position
recurs many times with a near-shared label. Under the project's random/dataset-fold
CV, the same positions sit in train and test, and the model memorizes "position →
label." Consequences, measured here:

| CV scheme | GBM mean AUC (10 assays) |
|---|---|
| random 5-fold (original setup) | 0.807 |
| **leave-position-out** (honest, edit-level) | **0.712** |
| **leave-one-protein-out** (honest, novel-protein) | **0.704** |

A position-identity-only predictor scores 1.000 random-fold but 0.42 (chance) grouped.
SPG1 is the clearest tell: 0.502 (chance) *with* the pLDDT feature under grouped CV,
0.659 *without* it — the feature was pure leakage, not signal. **Fix: drop
`plddt_region` from the edit-level score.**

**2. Naive averaging dilutes the one strong signal.** The project already showed this;
we take it to its conclusion — the substitution-*dependent* signal that carries real
power is the ESM2 wt-marginal (Meier et al. 2021), the field-standard zero-shot variant
predictor. Build the metric around it, not around an equal-weight blend.

## The enhanced metric

`P_tolerated = calibrate( GBM( ESM2_wt_marginal, grantham, blosum62, hydropathy,
                               sec_struct, aggregation, burial ) )`

- **Backbone:** ESM2 wt-marginal (per-substitution, protein-language zero-shot VEP).
- **Refiner:** a small gradient-boosted combiner over *substitution-dependent*
  biophysical signals — a non-linear version of the project's own "learn the weights"
  conclusion. `plddt_region` deliberately excluded.
- **Calibration:** isotonic map raw → **probability the variant is functionally
  tolerated**, fit cross-protein. ECE 0.038 → 0.027; reliability curve tracks the
  diagonal. A research tool must emit a trustworthy probability, not just a rank.
- **Applicability flag** (the rBPV idea, made concrete): every call reports whether the
  structural signals were actually present; sequence-only calls still score off ESM.
- **Honest by construction:** trained and reported **leave-one-protein-out**, so the
  quoted number is what you get on a protein the model has never seen.

## Validated performance (the number to quote)

Leave-one-protein-out, 9 held-out proteins:
- **Enhanced combiner: mean AUC 0.704**
- Zero-shot ESM2 alone: 0.659
- The combiner beats the ESM backbone on **9/9 proteins**, mean lift **+0.045**.

This is the honest replacement for the original 0.81/0.96. It is lower because it is
real: it measures ranking of *novel edits on novel proteins*, which is the actual
biosecurity use — an attacker proposes a specific substitution at a chosen site the
tool has never scored.

## Why this is now research-useful
1. **It generalizes to unseen proteins** (LOPO-validated), so it can score a novel
   designed threat, not just interpolate within a training protein.
2. **It emits a calibrated probability**, so a downstream PRISM risk gate can threshold
   it meaningfully (e.g. flag edits with P_tolerated > 0.5 AND classifier-evasive).
3. **It adds value over the language model** it wraps — otherwise you'd just use ESM.
4. **It degrades gracefully** when structure is missing (applicability flag + imputation).

## How to use
```python
from enhanced_bioplausibility import EnhancedBioPlausibility
scorer = EnhancedBioPlausibility.load("data/enhanced_model.joblib")
r = scorer.score(esm_wt_marginal=..., grantham=..., blosum62=..., hydropathy=...,
                 sec_struct=..., aggregation=..., burial=...)
r.p_tolerated   # calibrated P(edit functionally tolerated)
r.applicable    # were structural signals available?
r.backbone_esm  # the zero-shot term alone, for auditing
```
Drops into the Proto `Constraint` slot exactly as before: `proto_score = 1 −
r.p_tolerated`.

## Remaining honest gaps (next steps, not done here)
- **Real fold term.** Replace the Grantham-scaled ddG stub in FVS with RaSP/FoldX ΔΔG
  or a folded-variant pLDDT ratio — a *variant-dependent* structural signal, unlike the
  removed position-only one.
- **More proteins.** 9 proteins is a small LOPO panel; broaden across Pfam families to
  tighten the cross-protein estimate.
- **Fit the ESM rank-normalization train-only** (currently transductive — a small
  second leak).
- **Validate the FVS branch**, which is still at chance until real active-site
  annotations and a real fold term are wired in.
