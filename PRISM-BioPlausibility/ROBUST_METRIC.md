# The Robust Bioplausibility–Viability Score (rBPV)

A single equation that tracks the plausibility/viability of molecules produced by
an altered genetic code — robust *across molecule classes* (toxins, viral capsids,
enzymes), robust to *missing data*, and robust to *unreliable components*.

Motivated directly by the experiments (see README): no single signal is reliable
everywhere (Grantham ~0; ESM2-35M inverted; F_site helps only when edits hit the
functional region). Robustness therefore must be built into the *algebra*, not
hoped for from one good feature.

---

## 1. Components (each mapped to [0,1])

Two algebraic groups — necessary conditions that GATE, and graded evidence that REFINES.

**Necessary (conjunctive) — group N:**
- `P_prod` production integrity: synonymous/missense=1; nonsense/frameshift that
  loses any functional residue → 0 (DNA→protein stage).
- `V_fold` fold viability: folded fraction from ΔΔG (RaSP) or a PLM-likelihood proxy.
- `F_site` functional-site integrity: 3D-distance-weighted active-site preservation.

**Evidence (graded) — group E:**
- `S_evo` evolutionary: position-specific conservation (MSA/PSSM) blended with Grantham.
- `S_nat` naturalness: PLM pseudo-perplexity / wt-marginal — **ensembled over scales**.
- `S_expr` expression: codon adaptation (nucleotide-level edits only).

---

## 2. Three robustness mechanisms (the core idea)

Each component k carries two extra scalars beyond its score `s_k`:

- **Applicability `a_k ∈ [0,1]`** — is the data present AND relevant here?
  e.g. `a_Fsite` = fraction of edits within interaction range (λ) of a functional
  residue (→0 when the mutated region misses the active site, as in AICDA/POLG);
  `a_expr` = 1 only for nucleotide-level edits; `a_evo` = MSA depth term.
- **Reliability `r_k ∈ [0,1]`** — validated predictive power for this molecule
  *class*, estimated out-of-sample (e.g. component AUC on held-out homologs).
  This is what auto-kills the ESM2-35M signal (low r) while keeping 150M/650M.
- **Prior weight `w_k`** — learned (Sec. 4).

Effective weight: `ω_k = a_k · r_k · w_k`. A component that is inapplicable
(a=0), unreliable (r→0), or down-weighted drops out gracefully — it never injects
noise or a false penalty. This is the single most important robustness property.

---

## 3. The equation

**Gate (soft-AND, tunable hardness via generalized mean, p ≤ 0):**
```
G = ( Σ_{k∈N} ω_k · s_k^p  /  Σ_{k∈N} ω_k )^{1/p}        (Eq. 1)
```
p→0 ⇒ weighted geometric mean; p→−∞ ⇒ min. Use p∈[−3,0]: any included s_k→0
drives G→0 (true gate), but missing components (ω_k=0) are excluded.

**Evidence (robust weighted mean of available graded signals):**
```
V = Σ_{k∈E} ω_k · s_k  /  Σ_{k∈E} ω_k                    (Eq. 2)
```

**Combine — evidence refines but cannot resurrect a failed gate:**
```
rBPV_raw = G · ( λ + (1−λ)·V ) ,   λ ≈ 0.5               (Eq. 3)
```

**Calibrate (comparable across molecule classes):**
```
rBPV = σ( β0 + β1 · logit(rBPV_raw) )    (Platt; or isotonic) (Eq. 4)
```

---

## 4. Learned, out-of-sample weights (anti-overfit, anti-fragile)

Fit `w_k` (and the calibration) by logistic regression / gradient-boosted trees
on functional labels, **with k-fold cross-validation**; report held-out AUC.
The learned weights ARE the reliability ranking (answers the project's RQ2) and
automatically suppress weak components. Interpretable floor = equal weights +
geometric/arithmetic means (Eqs. 1–2 with ω_k=a_k).

**Ensemble naturalness (robust to model capacity — the 35M-vs-150M lesson):**
```
S_nat = Σ_j r_j · rank_norm( PLM_j(m') )  /  Σ_j r_j      (Eq. 5)
```
over PLM scales j (35M/150M/650M). rank_norm makes scales comparable; r_j
down-weights weak models. A single capacity choice can no longer break the score.

---

## 5. Why this is "most robust"
- **Graceful degradation:** missing/irrelevant components vanish (a_k), never harm.
- **Self-limiting unreliable signals:** r_k and learned w_k zero-out weak features
  (ESM2-35M, off-region F_site) instead of letting them invert the score.
- **No single point of failure:** evidence is an ensemble mean; naturalness is a
  multi-scale ensemble; gates are conjunctive so one strong-but-wrong signal can't
  fake viability.
- **Hard biology preserved:** production/fold/active-site catastrophes still gate to ~0.
- **Cross-class comparability:** calibration (Eq. 4) puts toxin, viral, enzyme
  scores on one scale.
- **Honest evaluation:** all weights learned with CV, reported out-of-sample.

## 6. Generality across "products of genetic code"
The framework is modality-agnostic — swap component implementations per molecule:
- proteins: V_fold=ΔΔG/pLDDT, F_site=catalytic geometry, S_nat=protein LM.
- structured RNA: V_fold=secondary-structure ΔΔG, F_site=motif preservation, S_nat=RNA LM.
- the genetic-code stage (P_prod, S_expr) is shared by all coding sequences.
The algebra (Eqs. 1–5) is unchanged; only the per-component scorers differ.
