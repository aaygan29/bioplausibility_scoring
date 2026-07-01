# Aayush — Lab Notebook: Bioplausibility & Functional-Viability Scoring

Project: PRISM (biologically-grounded adversarial robustness of protein foundation
models). This notebook records, in order, everything built and tested, with the
math, rationale, results, figures, and exact steps to reproduce.

---

## 0. Objective

Build a measure that scores whether a protein edit is (a) **bioplausible** (looks
like a natural mutation) and (b) **functionally viable** (the edited protein would
still work). The dangerous biosecurity case is an edit that evades an AI screening
model AND is plausible AND still functions. We need a validated scoring tool for
that, and an answer to: how should many biological signals be combined into one
number?

---

## 1. Iterative log (what we did, in order)

**Step 1 — Two metrics, two different algebras.**
- Bioplausibility (BPS-P): a weighted **average** of biological signals (Grantham,
  active-site, codon, structural). Appropriate because "looks natural" is a soft
  overall impression.
- Functional viability (FVS): a **product** (logical AND) of necessary conditions
  (produced AND folds AND active-site intact). Rationale: function is a chain; one
  broken link kills it, so averaging is wrong.
- Verified the gating: an edit that deletes the active site scores 0 under FVS,
  where an average would wrongly say "mostly fine."

**Step 2 — First validation on a real toxin (CcdB).**
- Additive plausibility alone was useless (AUC ~0.49). Adding a real structural
  term (AlphaFold pLDDT) lifted it to 0.72. First evidence that structure carries
  the signal.

**Step 3 — FVS with a real fold/function signal.**
- The ideal fold term (RaSP ddG) could not run locally (blocked install + bit-rotted
  Colab), so we used the ESM2 language-model variant score as a stand-in. On CcdB:
  FVS AUC 0.735, slightly beating the language model alone (0.727), and far above the
  additive plausibility (0.491). Structure-aware multiplicative design wins.

**Step 4 — Robustness across protein classes (rBPV).**
- Combined the signals into a robust score and tested across toxin, virus, enzyme.
  Found that no single signal is reliable everywhere, and a combined, reliability-
  weighted, cross-validated score holds up best.

**Step 5 — Model-capacity finding.**
- Small language models are unreliable (CcdB AUC 0.478 at 35M vs 0.727 at 150M; the
  virus flipped sign). Use >=150M or an ensemble.

**Step 6 — Literature-grounded extra constraints.**
- Added burial (solvent accessibility), BLOSUM62, hydropathy, secondary structure,
  aggregation, and a conservation hook, plus a per-edit analysis tool that explains
  WHY an edit scores as it does. (Sources in Section 6.)

**Step 7 — The empirical question: average, or something smarter?**
- Downloaded real DMS data and tested combination rules out-of-sample. **Naive
  averaging was the WORST strategy.** This is the key finding.

**Step 8 — Breadth + depth.**
- Scaled to 10 assays across 9 protein classes with bootstrap CIs and a paired test.
  Learned weighting beat naive averaging in 10/10 assays.

**Step 9 — Improving the mechanism (and verifying it).**
- Added the ESM signal, fixed burial with Weighted Contact Number, and used a
  non-linear combiner (gradient boosting). Mean AUC rose 0.682 -> 0.727 -> 0.811.
  Verified the strong CcdB number (0.956) with a random-split re-run, a Spearman
  check, and a label-shuffle control.

**Step 10 — Proto integration.**
- Installed Proto (programming language for biological design) and wrote our metric
  as a Proto Constraint, so the search for evasive+plausible+functional edits is one
  composable program.

---

## 2. The central questions (what the experiment answers)

- **Q1.** How should many bioplausibility signals be combined: naive average,
  weighted average, or a learned model?
- **Q2.** Does the answer hold across different protein types?
- **Q3.** Which signals actually carry the predictive power?

These map to the project's RQ2 (which constraints matter most) and RQ4 (how
realistic / tolerable the edits are).

---

## 3. Ground truth and why it is valid

We validate against **deep mutational scanning (DMS)**: wet-lab experiments that
measure the functional effect of thousands of single-residue variants. Rationale:
naturally plausible substitutions tend to be tolerated, and biophysical +
evolutionary signals are known to predict DMS fitness. So the plausibility signals
are validated against measured functional tolerance (a proxy for naturalness).

---

## 4. Data (exact sources and links)

**DMS assays (labels).** ProteinGym (Notin et al., 2023, https://proteingym.org)
via HuggingFace mirror `genbio-ai/ProteinGYM-DMS`, folder `singles_substitutions/`.
Download: `https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/resolve/main/singles_substitutions/<ASSAY>.tsv`.
Columns: `sequences` (variant sequence), `labels` (DMS fitness, continuous),
`fold_id` (0-4 cross-validation fold).

The 10 assays:

| Assay | Class | UniProt | n |
|---|---|---|---|
| CCDB_ECOLI_Adkar_2012 | bacterial toxin | P62554 | 1,176 |
| CCDB_ECOLI_Tripathi_2016 | toxin (replicate) | P62554 | 1,663 |
| BLAT_ECOLX_Deng_2012 | beta-lactamase (resistance) | P62593 | 4,996 |
| GFP_AEQVI_Sarkisyan_2016 | fluorescent protein | P42212 | 1,084 |
| SPG1_STRSG_Olson_2014 | binding domain (protein G) | P06654 | 1,045 |
| PTEN_HUMAN_Matreyek_2021 | phosphatase | P60484 | 5,083 |
| UBC9_HUMAN_Weile_2017 | SUMO E2 ligase | P63279 | 2,563 |
| NUD15_HUMAN_Suiter_2020 | nucleotide hydrolase | Q9NV35 | 2,844 |
| AICDA_HUMAN_Gajula_2014_3cycles | cytidine deaminase | Q9GZX7 | 209 |
| POLG_CXB3N_Mattenberger_2021 | Coxsackievirus polyprotein | P03313 | 15,711 |

**Structures.** AlphaFold DB (https://alphafold.ebi.ac.uk). Resolve UniProt accession
from entry name via `https://rest.uniprot.org/uniprotkb/search?query=id:<ENTRY>&fields=accession&format=tsv`,
then download from `https://alphafold.ebi.ac.uk/api/prediction/<ACCESSION>` (field
`pdbUrl`). Obtained 9/10; POLG (2,185 aa) had no single model so used sequence-only.

---

## 5. Method: the signals (math, per variant)

Edits `E = {i : wt[i] != var[i]}`, edit `(i,a,b)`. Each signal in `[0,1]`, higher =
more tolerable; multi-mutant variants average over `E`. Wild type reconstructed as
the per-column consensus residue across all variant sequences.

- **Grantham** (chemistry): `G = rho*sqrt(alpha*dc^2+beta*dp^2+gamma*dv^2)`,
  `alpha=1.833, beta=0.1018, gamma=0.000399, rho=50.723`; `s = mean(1 - G/215)`.
- **BLOSUM62** (evolutionary tolerance): `s = mean clip((BLOSUM62(a,b)+4)/15,0,1)`.
- **Hydropathy** (Kyte-Doolittle): `s = mean(1 - burial_i*min(1,|KD_a-KD_b|/9))`.
- **Secondary structure** (Chou-Fasman): penalize loss of helix/sheet propensity and
  proline insertion.
- **Aggregation** (TANGO idea): penalize local increase in hydrophobic beta-propensity.
- **Burial = Weighted Contact Number** (RSA surrogate, Lin 2008/Yeh 2014):
  `WCN_i = sum_{j!=i} 1/d_ij^2` over CA atoms, normalized to [0,1];
  `s = mean(1 - burial_i*severity(a,b))`.
- **Structural context (pLDDT)**: `s = mean(1 - pLDDT_i/100)`.
- **ESM2 marginal** (naturalness, Meier 2021): one forward pass of ESM2-150M on the
  wild type; `esm(var) = sum_E [log p(b|wt) - log p(a|wt)]`; rank-normalized per assay.

---

## 6. Method: combining the signals (the comparison)

Evaluated out-of-fold using the dataset's `fold_id`:
- **(a) naive average** of all signals.
- **(b) reliability-weighted**: weight each signal by `max(0, train AUC - 0.5)`.
- **(c) learned logistic** regression.
- **(d) learned GBM**: HistGradientBoosting (depth 3, 150 iters, lr 0.08), non-linear.

Metrics: AUC (functional vs broken), Spearman vs continuous fitness, bootstrap 95%
CI (600 resamples), cross-assay mean and worst-case, paired sign test (learned vs
naive). Binary label = above median fitness.

---

## 7. Results

**Single signals (Q3).** [Figure 3] On CcdB: structural context (pLDDT) 0.806,
ESM 0.722, hydropathy 0.642, secondary structure 0.626, aggregation 0.600, burial
0.591, Grantham 0.532, BLOSUM62 0.524. A few signals carry most of the power.

**Combining signals (Q1, Q2).** [Figure 1, Figure 2] 10 assays, out-of-fold:

| strategy | mean AUC | worst-case AUC |
|---|---|---|
| naive average | 0.681 | 0.600 |
| reliability-weighted | 0.713 | 0.610 |
| best single signal | 0.715 | 0.606 |
| learned logistic (CV) | 0.727 | 0.636 |
| learned GBM (CV) | **0.811** | **0.691** |

Learned logistic beats naive average in **10/10 assays** (sign test p ~ 0.002),
mean +0.046 AUC. Per-assay GBM AUC: CcdB 0.956/0.929, BLAT 0.780, GFP 0.726,
SPG1 0.881, PTEN 0.812, UBC9 0.734, NUD15 0.867, AICDA 0.736, POLG 0.691.

**Iterative improvement of the mechanism.** [Figure 4]
hand-crafted logistic 0.682 -> + ESM + WCN burial 0.727 -> + non-linear GBM 0.811.

**Verification of the strong CcdB number (no leakage).** [Figure 5]
Dataset folds 0.956 vs random 5-fold 0.958 (no fold leakage); Spearman(prediction,
fitness) +0.766; shuffled-labels control 0.479 (~0.5, no spurious power).

Figures (PNG files in `figures/`):
- Fig 1: `fig1_strategy_comparison.png` — strategy comparison (mean + worst-case).
- Fig 2: `fig2_per_assay.png` — per-assay naive vs GBM.
- Fig 3: `fig3_single_signals.png` — single-signal AUC.
- Fig 4: `fig4_iterative.png` — iterative improvement.
- Fig 5: `fig5_controls.png` — leakage controls.

---

## 8. Answers and recommendation

- **Q1:** Do NOT naive-average (worst on every metric and class). Use a **non-linear,
  cross-validated learned model (gradient boosting)** on the signals. It captures
  interactions (e.g. buried AND severe AND low-ESM) that averaging cannot.
- **Q2:** Holds across 9 protein classes; learned > naive in 10/10.
- **Q3:** Structural context (pLDDT) and ESM carry most of the power; hand-crafted
  physicochemical features are weak alone but help inside the non-linear model.

Recommended measure: strong signals (structural context, ESM marginal, WCN burial,
hydropathy, plus the weaker physicochemical ones) combined by a cross-validated
gradient-boosting model, always validated out-of-sample on DMS with a shuffle control.

---

## 9. Exact reproduction

```bash
pip3 install --user numpy scipy scikit-learn biopython torch fair-esm
mkdir -p data/proteingym data/structures data/esm
base=https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/resolve/main/singles_substitutions
for A in CCDB_ECOLI_Adkar_2012 CCDB_ECOLI_Tripathi_2016 BLAT_ECOLX_Deng_2012 \
  GFP_AEQVI_Sarkisyan_2016 SPG1_STRSG_Olson_2014 PTEN_HUMAN_Matreyek_2021 \
  UBC9_HUMAN_Weile_2017 NUD15_HUMAN_Suiter_2020 AICDA_HUMAN_Gajula_2014_3cycles \
  POLG_CXB3N_Mattenberger_2021; do curl -sL -o data/proteingym/$A.tsv $base/$A.tsv; done
# structures: resolve UniProt -> AlphaFold (Section 4) into data/structures/<ENTRY>.pdb
python3 compute_esm_scores.py        # cache ESM signal
python3 multi_assay_analysis.py      # full comparison + stats
python3 empirical_bioplausibility.py data/proteingym/CCDB_ECOLI_Adkar_2012.tsv data/structures/CCDB_ECOLI.pdb
python3 make_figures.py              # regenerate figures
```
Code: `bps_p.py`, `fvs.py`, `constraints_extra.py`, `compute_esm_scores.py`,
`multi_assay_analysis.py`, `empirical_bioplausibility.py`, `make_figures.py`,
`proto_bioplausibility.py` (Proto Constraint). Proto installs in a Python 3.12 venv:
`python3.12 -m venv protoenv; protoenv/bin/pip install "git+https://github.com/evo-design/proto-language.git"`.

---

## 10. Limitations

DMS is a proxy for naturalness (a direct natural-ortholog vs random check is future
work, RQ4). ESM rank-normalization is transductive (no labels, but uses test feature
distribution). GBM can overfit small assays (AICDA n=209); use the linear model there.
Burial uses WCN (CA-only); true DSSP/Biotite SASA could improve it. POLG was
sequence-only. Labels binarized at the median; Spearman reported alongside.

---

## 11. Sources

- Grantham 1974, Science. https://doi.org/10.1126/science.185.4154.862
- Henikoff & Henikoff 1992, PNAS (BLOSUM). https://doi.org/10.1073/pnas.89.22.10915
- Kyte & Doolittle 1982, JMB. https://doi.org/10.1016/0022-2836(82)90515-0
- Chou & Fasman 1978, Annu Rev Biochem. https://doi.org/10.1146/annurev.bi.47.070178.001343
- Fernandez-Escamilla et al. 2004, Nat Biotechnol (TANGO). https://doi.org/10.1038/nbt1012
- Yeh et al. 2014, Mol Biol Evol (Weighted Contact Number). https://doi.org/10.1093/molbev/mst196
- Pucci et al. 2018, Sci Rep. https://doi.org/10.1038/s41598-018-22531-2
- Conservation + solvent accessibility, 2025, bioRxiv. https://www.biorxiv.org/content/10.1101/2025.02.03.636212
- Meier et al. 2021, NeurIPS (ESM variant effects). https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html
- Lin et al. 2023, Science (ESM2). https://doi.org/10.1126/science.ade2574
- Notin et al. 2023, ProteinGym. https://proteingym.org
- Jumper et al. 2021, Nature (AlphaFold2). https://doi.org/10.1038/s41586-021-03819-2
- Varadi et al. 2022, NAR (AlphaFold DB). https://doi.org/10.1093/nar/gkab1061
- Data mirror. https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS
- Proto: Hie et al. 2026, bioRxiv. https://www.biorxiv.org/content/10.64898/2026.06.22.733870v1
