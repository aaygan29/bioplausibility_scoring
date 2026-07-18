# Measuring whether a produced protein is structurally + functionally plausible
### A Proto Constraint for AI-generated proteins — where we started, what changed, why it's better, and how to strengthen it

**Goal (restated).** Once an AI *produces* a protein (or a structure), return one
trustworthy number: how likely is this thing to actually fold and function? Expressed
in **Proto** (Hie et al., 2026), the biological design language, this is a single
**Constraint** — `0.0 = keep, 1.0 = reject` — that any Proto Optimizer can minimize.
Code: `enhanced_proto.py` (adapter) + `enhanced_bioplausibility.py` (metric) +
`data/enhanced_model.joblib` (fitted model).

---

## 1. Where we started

The original score, **BPS-P**, answered "does this *edit* look natural?" as a weighted
**average** of four hand-crafted signals (Grantham, active-site avoidance, codon usage,
structural confidence via pLDDT). A companion **FVS** multiplied necessary factors
(produced × folds × active-site-intact). The reported ceiling looked strong: single-
signal pLDDT AUC 0.806, learned combiner 0.811, and up to 0.956 on CcdB.

Two problems made those numbers unusable for real research:

1. **Position leakage.** The strongest feature, `plddt_region = 1 − pLDDT[i]/100`, is a
   pure function of *residue position* — identical for all ~19 substitutions at a site
   (verified: 1 distinct value per position). ProteinGym is saturation mutagenesis, so
   every position recurs; under random/dataset-fold CV the same positions sit in train
   and test and the model memorizes "position → label." A position-identity-only
   predictor scores **1.000** random-fold but **0.42 (chance)** under position-grouped
   CV. The headline was largely memorization, not edit discrimination.
2. **Naive averaging** diluted the one signal that actually varies per substitution and
   generalizes — the protein language model (ESM2 wt-marginal; Meier et al. 2021).

---

## 2. What we did

Rebuilt the metric around the substitution-dependent, generalizing signal, removed the
leakage, and added what a *research* tool needs.

- **Backbone = ESM2 wt-marginal** — the field-standard zero-shot variant-effect
  predictor; per-substitution, not per-position.
- **Refiner = gradient-boosted combiner** over the *substitution-dependent* biophysical
  signals (Grantham, BLOSUM62, hydropathy, secondary-structure propensity, aggregation,
  burial). This is the project's own "learn the weights, don't average" conclusion taken
  to a non-linear model.
- **Dropped `plddt_region`** — the position-only leakage vector. (SPG1 binding domain:
  0.502 = chance *with* it under honest CV → 0.659 *without* it.)
- **Isotonic calibration** → the output is a genuine **P(variant tolerated)**, not an
  index. ECE 0.038 → 0.027, cross-protein. This is what lets Proto treat
  `1 − P_tolerated` as an honest constraint rather than an arbitrary penalty.
- **Applicability flag** — every score reports whether structural signals were present
  (sequence-only calls still score off ESM); operationalizes the rBPV robustness idea.
- **Honest by construction** — trained and reported **leave-one-protein-out (LOPO)**, so
  the quoted number is what you get on a protein the model has never seen.

Then wired it into Proto: `enhanced_proto.py` exposes
`enhanced_bioplausibility_constraint(seq, scorer=...)` returning `(1 − P_tolerated,
metadata)` — a drop-in for `Constraint(function=...)`. Verified end-to-end on the real
CcdB AF2 structure: a conservative far edit is kept (0.43), a radical edit at the
gyrase-interaction site is rejected (0.65).

---

## 3. Why it is currently better

| | Original BPS-P | Enhanced metric |
|---|---|---|
| Strongest feature | `plddt_region` (position-only — leaks) | ESM2 wt-marginal (per-substitution) |
| Combination | naive / weighted average | calibrated GBM combiner |
| Reported CV | random/dataset fold (leaky) | leave-position-out **and** leave-one-protein-out |
| Output | a 0–1 index | a **calibrated probability** |
| Novel-protein AUC | not measured (0.81/0.96 inflated) | **0.704**, beats zero-shot ESM (0.659) on **9/9** proteins |
| Missing data | silent | explicit applicability flag |

The honest **0.704** is lower than the old 0.81/0.96 *because it is real*: it measures
ranking of novel edits on **proteins never seen in training** — exactly the deployment
case (an attacker, or a generative model, produces a specific protein the scorer has
never scored). Validated on **10 ProteinGym DMS assays across 9 protein classes**
(toxin, resistance enzyme, phosphatase, hydrolase, SUMO enzyme, deaminase, binding
domain, fluorescent protein, viral polyprotein), **36,374 single variants** — a robust,
diverse benchmark.

---

## 4. The important honesty: "edit" vs "produced structure"

What is validated today scores a **produced protein *relative to a natural reference***
(the PRISM adversarial-edit case) — because DMS ground truth *is* edit-relative. Your
stated goal also includes **de novo produced structures with no wild-type**. That is a
strictly harder, **reference-free** problem, and it is NOT yet validated here (there is
no DMS label for a protein that has no wild-type). `enhanced_proto.py` already exposes
both regimes behind one interface; the reference-free path needs the additions in §5
before any number should be quoted for it.

---

## 5. How to strengthen this (the roadmap)

**A. Reference-free structural plausibility (the biggest gap for de novo designs).**
Add signals that judge a produced *structure* on its own, no wild-type needed:
- **Self-consistency (scTM/scRMSD):** inverse-fold the produced backbone (ProteinMPNN /
  SolubleMPNN), re-fold the sequences (ESMFold/AF2), measure agreement. This is the
  standard de-novo-design plausibility check and needs no reference.
- **Folded-variant confidence:** fold the *produced* sequence and read its own pLDDT/
  pAE — a variant-DEPENDENT structural term (unlike the removed position-only one).
- **Real ΔΔG** (RaSP/FoldX) or Boltz-2 co-folding confidence for the fold factor in FVS.

**B. Validate the functional (FVS) half.** It is still at chance: the fold term is a
Grantham-scaled stub and active sites are hand-annotated for a few proteins. Wire real
ΔΔG + UniProt/M-CSA catalytic residues, then validate against activity-specific DMS
assays (not just stability).

**C. Broaden the benchmark.** 9 proteins is a small LOPO panel. Extend across Pfam
families and add ProteinGym's indel/clinical sets to tighten the cross-protein estimate
and test transfer to fold classes unlike the training set.

**D. Close residual leaks.** Fit the ESM rank-normalization train-only (currently
transductive). Re-report every number under LOPO as the primary metric.

**E. Make it a first-class Proto module.** Publish `enhanced_proto` as a reusable Proto
Constraint library so a design program can write, literally,
`Constraint(function=enhanced_bioplausibility_constraint, weight=w)` alongside an
evasion constraint — the optimizer then searches directly for produced proteins that are
plausible AND functional (AND, in the PRISM risk study, classifier-evasive). The metric
becomes a shared, calibrated scoring primitive inside the design language, which is the
end state you described: AI evaluating the structural + functional bioplausibility of
what AI produces.

---

## 6. How to use (Proto)
```python
from enhanced_proto import EnhancedScorer, enhanced_bioplausibility_constraint
scorer = EnhancedScorer.from_files(wt, "AF2/PROT.pdb", "data/enhanced_model.joblib",
                                   esm_fn=my_esm2_wt_marginal)   # plug in your ESM2
# inside a Proto program:
#   Constraint(inputs=[seg],
#              function=lambda s, **_: enhanced_bioplausibility_constraint(s, scorer=scorer),
#              weight=1.0)
# The optimizer MINIMIZES 1 - P_tolerated, i.e. maximizes calibrated plausibility.
```

---

## 7. The experiment's actual shape: a constraint PANEL (proto_panel.py)

The scoring tool is one part of a larger experiment: a protein/genomic **foundation
model produces a biomolecule**, and we then evaluate whether that produced thing will
actually be functional — structurally and otherwise. Proto is the right frame precisely
because it lets us score **one produced construct from many independent constraints** at
once, rather than collapsing everything into a single hand-tuned number.

`proto_panel.py` implements this directly. A `ProtoPanel` holds several `Constraint`
objects, each scoring the same produced molecule on Proto's 0=keep..1=reject scale, each
reporting its own applicability, and `evaluate()` returns every constraint's score plus a
tunable weighted aggregate. Default panel:

| constraint | regime | what it judges | status |
|---|---|---|---|
| `calibrated_plausibility` | referenced | 1 − P(tolerated), the enhanced metric | **validated, LOPO AUC 0.704** |
| `functional_viability` | referenced | produced AND folds AND active-site intact | needs real ΔΔG + sites |
| `self_structure_confidence` | reference-free | 1 − mean pLDDT of the PRODUCED fold | plumbed; needs folding call |
| `aggregation` | reference-free | hydrophobic-run / aggregation propensity | heuristic (swap in TANGO) |
| `naturalness_esm` | reference-free | squashed ESM2 pseudo-likelihood of the sequence | plumbed; needs ESM2 |

Worked demo (three produced CcdB-scale molecules) — aggregate rises monotonically with
badness while each constraint contributes its own view:

| produced molecule | plausibility | self-structure | aggregation | naturalness | **aggregate** |
|---|---|---|---|---|---|
| good design (conservative) | 0.432 | 0.045 | 0.43 | 0.08 | **0.247** |
| risky (radical @ active site) | 0.649 | 0.331 | 0.43 | 0.08 | **0.399** |
| broken (low-confidence fold) | 0.841 | 0.570 | 0.42 | 0.08 | **0.530** |

Why the panel form matters for this experiment:
- **Independent evidence.** A design can pass one constraint and fail another; the panel
  surfaces *which* axis flags it, instead of hiding it in an average.
- **De novo coverage.** Reference-free constraints score a produced protein with no
  wild-type — the general generative case — while referenced ones add power when the
  produced molecule is a modified natural protein.
- **Graceful degradation.** A constraint with missing inputs self-skips; the panel still
  returns a score over the applicable ones (with `n_applicable` reported).
- **Ablatable weights.** Per-constraint weights let you ask "which constraint actually
  predicts functionality?" directly, and feed the aggregate to a Proto Optimizer (to
  design toward plausibility) or a risk gate (to flag implausible produced threats).

This is the end state you described: a produced biomolecule, evaluated by AI from a
structural + functional panel of constraints, expressed in Proto so the same object is
scored from many angles at once.
