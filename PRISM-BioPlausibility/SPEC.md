# BPS-P: A Biological Plausibility Score for Protein Foundation Models

**Mathematical specification, validation protocol, and model-integration path.**
Anchored on Krishnan et al. (`s_bio`, genomic), SafeProtein (Fan et al. 2025),
and the structure-vs-function findings of Ikonomova et al. (2026) and
Niloy et al. (ESM-2 substitution study, 2026).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `x`  | wild-type (reference) amino-acid sequence, length `L` |
| `x'` | adversarial / mutated variant of `x` |
| `E`  | set of edited residue positions, `E = {i : x_i != x'_i}`, `m = |E|` |
| `A`  | annotated functional residues of `x` (catalytic + binding), from UniProt |
| `s_k` | component score `k`, each mapped to `[0,1]` (1 = maximally plausible) |
| `w_k` | weight of component `k`, `sum_k w_k = 1` |

---

## 1. The composite

This mirrors the genomic `s_bio` (arithmetic mean of constraint terms in `[0,1]`),
extended with a **structural / functional** term — the piece the genomic work did
not need but protein work does, because Ikonomova et al. show sequence-level
plausibility does **not** imply functional retention.

```
BPS-P(x, x') = sum_k  w_k * s_k(x, x')          (Eq. 1)
```

Default weights `w_k = 1/K` (matches the s_bio precedent so the baseline is not
arbitrary). Section 4 replaces these with **learned** weights.

The four components below are Vrinda's three named constraints (Grantham,
active-site, codon) **plus** the structural term that makes the metric honest.

---

## 2. Component definitions

### 2.1 Grantham conservatism — `s_gram`

Grantham distance between residues `a,b` is computed from its three
physicochemical properties (composition `c`, polarity `p`, volume `v`),
**not** a memorized lookup table (this is the original 1974 definition):

```
G(a,b) = rho * sqrt( alpha*(c_a - c_b)^2
                   + beta *(p_a - p_b)^2
                   + gamma*(v_a - v_b)^2 )         (Eq. 2)

alpha = 1.833,  beta = 0.1018,  gamma = 0.000399,  rho = 50.723
```

Range ~5 (Leu-Ile) to 215 (Cys-Trp). Per-edit conservatism, averaged:

```
s_gram(x,x') = (1/m) * sum_{i in E} ( 1 - G(x_i, x'_i) / G_max )   (Eq. 3)
G_max = 215.   If m = 0, s_gram = 1.
```

### 2.2 Active-site preservation — `s_act`

Catalytic residues should weigh more than mere binding residues. Let
`W(i)` be a per-residue functional weight (default: catalytic = 1.0,
binding = 0.5, else 0):

```
s_act(x,x') = 1 - ( sum_{i in E} W(i) ) / ( sum_{i in A} W(i) )    (Eq. 4)
```

Clamped to `[0,1]`. If `A` empty, `s_act = 1` (no annotated core to break).

### 2.3 Codon adaptation — `s_codon`

Back-translate the variant to a coding sequence for the host organism, then
take the Codon Adaptation Index (already in `[0,1]`):

```
CAI = ( prod_{j=1}^{n} w_codon(j) )^(1/n)                          (Eq. 5)
s_codon(x,x') = CAI(codons(x'))
```

`w_codon(j)` = relative adaptiveness of codon `j` from the host's reference
table (RSCU-derived). **Must** be a real host table (e.g. human / E. coli);
the placeholder in code is flagged.

> Note: amino-acid-level attacks can hold the codon table fixed; this term
> matters when the threat model edits nucleotides (the genomic-protein bridge).

### 2.4 Structural / functional validity — `s_struct`

The term that operationalizes Ikonomova: a variant can be sequence-plausible
yet fold-broken. Use predicted pLDDT (ESMFold / Boltz), **relative** to WT so
naturally low-confidence proteins are not penalized:

```
s_struct(x,x') = min( 1, pLDDT(x') / pLDDT(x) )                    (Eq. 6)
```

Optionally substitute pTM or a TM-score(model(x'), model(x)) if structures
are available — same `[0,1]` shape.

---

## 3. Why this shape (provenance)

| Component | Borrowed from | Adds over prior work |
|---|---|---|
| `s_gram`  | Grantham 1974; transition/transversion idea in Krishnan s_bio | residue-level evolutionary cost |
| `s_act`   | new (Vrinda) | functional-core integrity, absent from s_bio |
| `s_codon` | s_bio GC/codon term (genomic) | expression plausibility |
| `s_struct`| Ikonomova 2026; SafeProtein RMSD proxy | catches plausible-but-nonfunctional — the core gap |
| mean form | Krishnan s_bio (arithmetic mean) | identical aggregation for comparability |

---

## 4. Validating the metric (turns a formula into a result)

### 4.1 Construct validity
Correlate BPS-P with measured fitness on **ProteinGym** DMS assays
(use viral-fitness + resistance-enzyme assays as functional analogs of the
threat classes — see overlap note in README). Report Spearman rho per assay.

### 4.2 Internal validity (non-redundancy)
Component correlation matrix + VIF. If two components collinear, drop or merge.

### 4.3 Predictive validity & weight learning
Fit logistic regression predicting the ground-truth label
(`functional / still-active` vs not) from the four component scores:

```
P(functional | x') = sigmoid( b0 + sum_k beta_k * s_k )            (Eq. 7)
w_k  := beta_k / sum_j |beta_j|     (normalized learned weights)
```

The learned `beta_k` **are the answer to Vrinda's RQ2** — the ranking of which
biological constraint most limits adversarial success. Report AUC.

### 4.4 Calibration
Reliability diagram (BPS-P bin vs. fraction actually functional),
Brier score, ECE. A metric a model will consume must be calibrated.

---

## 5. Integrating into a model's predictive work

**Tier 1 — inference-time second opinion.** Flag inputs where the model says
`benign` but BPS-P is high (plausible). No retraining. Deployment story.

**Tier 2 — differentiable surrogate (the real enhancement).** Grantham/UniProt/
BLAST are non-differentiable. Train `f_theta(emb(x')) ~= BPS-P` on ESM-2
embeddings. Then:
  - **Defense:** add `lambda * f_theta` as an adversarial-training regularizer
    (penalize confident-benign on high-plausibility variants) — the missing
    protein-model defense.
  - **White-box attack:** FGSM/PGD in embedding space constrained to high
    `f_theta` (biologically plausible) — the missing white-box study.

**Tier 3 — auxiliary head.** Co-train a BPS-P head with the classifier
(multi-task) so the representation is plausibility-aware. Future work.

---

## 6b. Empirical result (real data, CcdB toxin)

First real run on `CCDB_ECOLI_Adkar_2012` (1,176 variants, ProteinGym labels):

| Config | AUC |
|---|---|
| Grantham only (other components stubbed) | 0.53 |
| + real AF2 pLDDT structural term (`s_struct_af2`) | **0.718** |
| ESM2-650M zero-shot (ProteinGym leaderboard, same assay) | 0.761 |

Two findings that **drive the v2 design** below:
1. The structural term carries almost all the signal (learned weight 0.91);
   position-agnostic Grantham is near-useless (negative coefficient).
   => the metric needs **position-specific** terms, not static matrices.
2. A WT-structure-only BPS-P (0.718) already approaches full ESM2-650M (0.761),
   using no learned protein model — a strong, cheap, interpretable baseline.

---

## 7. BPS-P v2 — literature-informed enhancement

Recent VEP literature (2024-25) converges on three points:
- **Sequence conservation beats stability alone** for variant effect; static
  physicochemical matrices (Grantham) underperform position-specific signals
  [Leveraging structure for VEP, Sci. Direct 2025; ESM1b, Nat. Genet. 2023].
- **ΔΔG stability** is a direct, complementary biophysical-plausibility signal
  [structure-informed VEP].
- **PLM pseudo-perplexity** = a calibrated "naturalness" score; pseudo-perplexity
  *shift* under mutation is itself a robustness signal [OFS pseudo-perplexity,
  PRX Life 2024; Niloy et al. 2026 used exactly this on ESM-2].

### v2 components, grouped by plausibility axis

```
BPS-P_v2 = w_evo*S_evo + w_struct*S_struct + w_func*S_func
         + w_nat*S_nat  + w_expr*S_expr            (Eq. 8)
```

| Axis | Component | Definition | Data (on disk?) |
|---|---|---|---|
| Evolutionary | `S_evo` | position-specific: MSA frequency / PSSM delta at edited site, blended with Grantham cost | MSA available on ProteinGym |
| Structural | `S_struct` | predicted ΔΔG (sign+magnitude) → [0,1]; fallback = AF2 pLDDT tolerance proxy (done) | AF2 structures on disk |
| Functional | `S_func` | active-site / catalytic preservation, weighted | UniProt (to wire) |
| Naturalness | `S_nat` | pseudo-perplexity ratio ppl(wt)/ppl(var) from an **independent** PLM | needs a 2nd model |
| Expression | `S_expr` | codon adaptation index (nucleotide threat only) | host table |

### Critical design note — avoid circularity
`S_nat` must NOT come from the same model under attack (ESM2-650M), or the
metric and the target share failure modes. Use a different-family PLM
(e.g. a ProGen/RITA variant) or hold ESM2 pseudo-perplexity out as a separate
diagnostic, not a scoring term.

### Why `S_evo` is the priority add
The CcdB run shows static Grantham fails because it ignores *where* the edit
lands in the conservation profile. The single highest-value upgrade is the
position-specific evolutionary term from the MSA — likely to lift AUC most.

### Aggregation upgrade
Replace the plain mean with the **learned-weight** form (Eq. 7) by default,
and report a non-linear variant (gradient-boosted trees on the 5 components)
as an upper bound — the genomic `s_bio` mean is the interpretable floor.

---

## 6. Open items to verify before publication
1. Exact ProteinGym assay count and the specific viral/resistance assays usable
   as labels (fetch blocked here — confirm on proteingym.org).
2. SafeProtein-Bench composition (429 proteins: toxin vs viral split).
3. Real host codon table for `s_codon`.
4. UniProt active-site annotation coverage across SafeProtein-Bench targets.
