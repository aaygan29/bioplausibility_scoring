# Bioplausibility scoring as a Proto Constraint

This wires our bioplausibility / functional-viability scoring into **Proto** (Hie
et al., 2026), so the metric becomes a reusable, shareable scoring module inside a
real design framework instead of a one-off script. Code: `proto_bioplausibility.py`.

## How Proto works (the part we plug into)
Proto builds a design from four primitives:
- **Segment / Construct**: the region(s) being designed (here, a protein).
- **Generator**: proposes candidate sequences each round (point mutations).
- **Constraint**: scores a candidate from **0.0 (perfect) to 1.0 (worst)**.
- **Optimizer**: searches sequence space to **minimize the total constraint score**.

`Constraint(inputs=[seg], function=fn, function_config={...}, weight=w)` is the slot
our metric fills. The Optimizer (e.g. `MCMCOptimizer`) minimizes the weighted sum.

## The one orientation rule
Proto constraints are **minimized** (0 = best). Our scores run the other way
(0..1, higher = more plausible / more functional). So every adapter inverts:

```
proto_constraint_score = 1 - our_score
```

A low Proto score therefore means "more plausible AND more functional" = keep.

## What the module provides
1. **`BioPlausibilityScorer`** — loads one protein's context once (AF2 structure,
   per-residue pLDDT, active-site residues, optional RaSP ddG table) and scores any
   variant fast. Three axes, each in [0,1], higher = better:
   - `plausibility(var)` — additive BPS-P: does the edit look natural?
   - `functionality(var)` — multiplicative FVS: produced AND folds AND active-site intact.
   - `robust(var)` — functionality gates, plausibility refines (ROBUST_METRIC.md form).
2. **Three Proto constraint adapters** (return `(score, metadata)`, 0 = best):
   - `bioplausibility_constraint` — penalize unnatural edits.
   - `functional_viability_constraint` — penalize broken proteins.
   - `robust_viability_constraint` — penalize implausible OR non-functional edits.
3. **`build_prism_program(...)`** — a full Proto program: a protein Segment, a
   point-mutation Generator, our Constraints, and an MCMC Optimizer. With an optional
   `classifier_evasion_fn`, it searches the **PRISM dangerous corner**: edits that are
   evasive AND plausible AND still functional.

## How this maps to PRISM (Vrinda's framework)
- Our metric is exactly the **Constraint** primitive; her mutation proposer is the
  **Generator**; her GA/MCMC search is the **Optimizer**. The pipeline already is a
  Proto program.
- Running all three constraints at once (evasion + plausibility + functionality)
  operationalizes her **three-tier risk grading**: the optimizer is steered toward
  tier-1 threats (evades and stays functional) by construction.
- Per-constraint weights let RQ2 (which constraints matter) be tuned and ablated
  directly inside one program.

## Minimal usage
```python
from proto_bioplausibility import BioPlausibilityScorer, build_prism_program

scorer = BioPlausibilityScorer.from_files(
    wt=WT_SEQUENCE,
    pdb_path="AF2/MY_PROTEIN.pdb",
    active_sites_json="data/active_sites.json", assay_key="MY_PROTEIN",
    rasp_csv="data/rasp_MY_PROTEIN.csv",   # optional; else a Grantham-scaled stub
)
program, construct = build_prism_program(WT_SEQUENCE, scorer,
                                         classifier_evasion_fn=my_esm2_toxicity)
program.run()
for seq in construct.joined_sequences:
    print(seq.sequence)
```

## Status / how to run
- The scorer kernel and the three adapters run today (verified on a synthetic
  structure: a conservative far edit scores low/keep, a radical active-site edit
  scores high/reject).
- `build_prism_program` needs `pip install proto-language` (the generator class
  name for proteins may differ by version; adjust the import if so).
- The real-data self-test in `__main__` needs the ProteinGym CSVs + AF2 structures
  (currently not on disk after the data folder was moved); restore that folder to
  re-run the CcdB check.
- For real `functionality` numbers, supply a RaSP ddG CSV (`RASP_SETUP.md`); without
  it the fold term uses the Grantham-scaled stub.

## Reference
Hie, B., et al. (2026). A high-level programming language for generative biology
(Proto). bioRxiv. https://www.biorxiv.org/content/10.64898/2026.06.22.733870v1
