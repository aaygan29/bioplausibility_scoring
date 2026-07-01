# Empirical study: how to build and combine a bioplausibility measure

A full, reproducible record. A scientist starting from scratch should be able to
re-run everything from this document. No prior context assumed.

---

## 1. The central question

We are building a "bioplausibility" measure: a single number that says how
believable a protein edit is as a real, naturally-tolerable change (as opposed to
a random, structurally destructive one). This sits inside a biosecurity project
(PRISM) where we ask whether small, natural-looking edits can fool an AI screening
model while still producing a working protein.

The initial measure was a simple average of a few hand-crafted signals (Grantham
distance, codon usage, active-site preservation, structural confidence). The
practical question this study answers:

**Q1. How should many bioplausibility signals be combined into one measure:
naive average, weighted average, or a learned model?**
**Q2. Does the answer hold across different protein types (breadth)?**
**Q3. Which signals actually carry the predictive power?**

These map onto the project's research questions: which biological constraints
matter most (RQ2), and how realistic / tolerable adversarial edits are (RQ4).

---

## 2. Ground truth and why it is valid

We need an objective answer for "is this edit tolerable." We use **deep mutational
scanning (DMS)**: wet-lab experiments that make thousands of single-residue
variants of a protein and measure each variant's function (stability, activity,
fluorescence, binding, or organismal fitness, depending on the assay).

Rationale: biologically plausible, natural substitutions tend to be functionally
tolerated, and the variant-effect literature shows biophysical and evolutionary
signals predict DMS fitness (Pucci 2018; "conservation + solvent accessibility are
almost all you need", bioRxiv 2025). So we validate the plausibility signals
against measured functional tolerance. This is a proxy for "naturalness," and a
complementary check (ranking natural-ortholog substitutions vs random) is noted in
Limitations.

---

## 3. Data: exact sources and how to get them

### 3.1 DMS assays (the labels)
Source: ProteinGym (Notin et al., NeurIPS 2023, https://proteingym.org), via the
HuggingFace mirror **genbio-ai/ProteinGYM-DMS**, folder `singles_substitutions/`.

Download pattern (one file per assay):
```
https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/resolve/main/singles_substitutions/<ASSAY>.tsv
```
File format: tab-separated, 3 columns.
- `sequences` : the full amino-acid sequence of the variant.
- `labels`    : the measured DMS fitness score (continuous; sign/scale assay-specific).
- `fold_id`   : integer 0-4, a pre-defined cross-validation fold.

The 10 assays used (breadth across protein classes):

| Assay file | Protein class | UniProt | n variants |
|---|---|---|---|
| CCDB_ECOLI_Adkar_2012 | bacterial toxin | P62554 | 1,176 |
| CCDB_ECOLI_Tripathi_2016 | toxin (replicate) | P62554 | 1,663 |
| BLAT_ECOLX_Deng_2012 | beta-lactamase (antibiotic resistance) | P62593 | 4,996 |
| GFP_AEQVI_Sarkisyan_2016 | green fluorescent protein | P42212 | 1,084 |
| SPG1_STRSG_Olson_2014 | protein G B1 (binding domain) | P06654 | 1,045 |
| PTEN_HUMAN_Matreyek_2021 | phosphatase | P60484 | 5,083 |
| UBC9_HUMAN_Weile_2017 | SUMO E2 ligase | P63279 | 2,563 |
| NUD15_HUMAN_Suiter_2020 | nucleotide hydrolase | Q9NV35 | 2,844 |
| AICDA_HUMAN_Gajula_2014_3cycles | cytidine deaminase | Q9GZX7 | 209 |
| POLG_CXB3N_Mattenberger_2021 | Coxsackievirus polyprotein | P03313 | 15,711 |

Original DMS papers: Adkar 2012; Tripathi 2016; Deng 2012; Sarkisyan 2016;
Olson 2014; Matreyek 2021; Weile 2017; Suiter 2020; Gajula 2014; Mattenberger 2021.

### 3.2 Structures (for structural signals)
Source: AlphaFold Protein Structure Database (Jumper et al. 2021;
Varadi et al. 2022), https://alphafold.ebi.ac.uk.

Resolution steps:
1. Entry name (e.g. `CCDB_ECOLI`) = first two underscore tokens of the assay file.
2. Resolve to UniProt accession:
   `https://rest.uniprot.org/uniprotkb/search?query=id:<ENTRY>&fields=accession&format=tsv`
3. Get the model URL from the API and download:
   `https://alphafold.ebi.ac.uk/api/prediction/<ACCESSION>` -> JSON field `pdbUrl`
   (e.g. `https://alphafold.ebi.ac.uk/files/AF-P62554-F1-model_v6.pdb`).

Structures obtained for 9/10 proteins. POLG (P03313, 2,185 aa) returned no single
model (AlphaFold fragments very large proteins), so POLG used sequence-only signals.

### 3.3 On-disk layout
```
data/proteingym/<ASSAY>.tsv        # labels
data/structures/<ENTRY>.pdb        # AlphaFold structures
data/esm/<ASSAY>.npy               # cached ESM scores (Section 5.8)
```

---

## 4. Software environment

- macOS. Python 3.9 (system) for the analysis: `numpy, scipy, scikit-learn,
  biopython, torch, fair-esm`.
- ESM2-150M weights auto-download via `fair-esm` on first use.
- Proto installed separately in a Python 3.12 venv (Section 9).

---

## 5. The signals (math, per variant)

Notation: wild type `wt`, variant `var`, edited positions
`E = { i : wt[i] != var[i] }`, edit `(i, a, b)` with `a=wt[i]`, `b=var[i]`.
Every signal is mapped to `[0,1]` with **higher = more tolerable / more plausible**,
and for multi-mutant variants is averaged over the edits in `E`.

**Wild-type reconstruction.** The genbio files give variant sequences but not the
reference. For single-mutant saturation assays the wild-type residue is the most
common residue at each column, so `wt[i] = mode_s(s[i])` over all variant sequences.

### 5.1 Grantham distance (chemistry)
`G(a,b) = rho * sqrt( alpha*(c_a-c_b)^2 + beta*(p_a-p_b)^2 + gamma*(v_a-v_b)^2 )`
with `alpha=1.833, beta=0.1018, gamma=0.000399, rho=50.723` and per-residue
composition c, polarity p, volume v (Grantham, Science 1974). Range ~5..215.
`s_grantham = mean_E ( 1 - G(a,b)/215 )`.

### 5.2 BLOSUM62 (evolutionary substitution tolerance)
`s_blosum = mean_E clip( (BLOSUM62(a,b) + 4) / 15 , 0, 1)` (Henikoff & Henikoff
1992; matrix from Biopython). Complements Grantham: BLOSUM is how often evolution
accepts a swap, Grantham is the physicochemical distance.

### 5.3 Hydropathy (burial-weighted)
Kyte-Doolittle scale KD (1982). `s_hyd = mean_E ( 1 - burial_i * min(1, |KD_a-KD_b|/9) )`.
A large hydrophobicity change is penalized more when the residue is buried.

### 5.4 Secondary-structure compatibility
Chou-Fasman helix Pa and sheet Pb propensities (1978).
`drop = max(0, max(Pa_a,Pb_a) - max(Pa_b,Pb_b)) / 1.7`;
`s_ss = mean_E clip( 1 - drop - 0.6*[b==P and a!=P], 0, 1)` (proline breaks helices/sheets).

### 5.5 Aggregation propensity
TANGO idea (Fernandez-Escamilla 2004): destabilizing edits can nucleate
beta-aggregation. Over a 5-residue window centred at the edit,
`agg(seq) = mean(max(0,KD))/4.5 * mean(Pb)/1.7`;
`s_agg = 1 - min(1, 3*(agg(var) - agg(wt)))`.

### 5.6 Burial via Weighted Contact Number (solvent accessibility surrogate)
Solvent accessibility is the top structural predictor of variant effect, but true
SASA needs all-atom parsing. We use the CA-only **Weighted Contact Number**
(Lin 2008; Yeh 2014):
`WCN_i = sum_{j != i} 1 / d_ij^2` over CA coordinates; normalize across residues to
`[0,1]` using the 5th-95th percentile -> `burial_i` (1 = buried).
`s_burial = mean_E ( 1 - burial_i * severity(a,b) )` where
`severity = 0.4*(G/215) + 0.25*[charge flip] + 0.20*[H-bond donor/acceptor lost] + 0.15*(|dVolume|/167)`.

### 5.7 Structural context via pLDDT
AlphaFold per-residue pLDDT confidence. `s_plddt = mean_E ( 1 - pLDDT_i/100 )`.
Edits in low-confidence (flexible / disordered) regions are better tolerated.

### 5.8 ESM2 language-model naturalness
ESM2-150M (`esm2_t30_150M_UR50D`; Lin et al. 2023). wt-marginal score
(Meier et al. 2021): one forward pass on the wild type gives log-probabilities;
`esm(var) = sum_{i in E} [ log p(b | wt) - log p(a | wt) ]` at position i.
Rank-normalized to `[0,1]` per assay so it is comparable to the other signals.
Cached to `data/esm/<assay>.npy` (`compute_esm_scores.py`).

---

## 6. Combination strategies (the thing under test)

Each strategy turns the signal vector into one score, evaluated **out-of-fold**
using the dataset's `fold_id` (train on 4 folds, predict the held-out fold).

- **(a) Naive average**: mean of all signals (equal weight).
- **(b) Reliability-weighted**: weight signal j by `max(0, AUC_j - 0.5)` measured on
  the training folds, normalize, weighted sum. Down-weights signals that do not
  generalize.
- **(c) Learned logistic**: logistic regression on the signals.
- **(d) Learned GBM**: `HistGradientBoostingClassifier(max_depth=3, max_iter=150,
  learning_rate=0.08)` — a non-linear model that can use interactions between signals.

Binary label `y_bin = (label > median(label))` per assay (mean if the median is
degenerate). Single signals are oriented as `max(AUC, 1-AUC)` for a fair "best single".

---

## 7. Evaluation and statistics

- **AUC** (area under ROC): probability the score ranks a functional variant above a
  broken one. 0.5 = chance, 1.0 = perfect.
- **Spearman rho**: rank correlation of the score with the continuous DMS fitness.
- **Bootstrap 95% CI**: 600 resamples of the variants, AUC recomputed each time.
- **Cross-assay**: mean AUC and worst-case AUC across the 10 assays; a paired sign
  test of learned vs naive (how many of 10 assays improve), and the mean delta.
- **Leakage controls** (Section 8.3): random-fold re-run, label-shuffle, Spearman.

How this answers the questions: Q1 is the (a)-(d) comparison; Q2 is the cross-assay
summary and sign test; Q3 is the single-signal AUC table.

---

## 8. Results

### 8.1 Single signals (which carry the power, Q3)
On CcdB Adkar (1,176 variants), single-feature AUC:
structural context (pLDDT) **0.806**, ESM **0.722**, hydropathy 0.642,
secondary structure 0.626, aggregation 0.600, burial (WCN) 0.591,
Grantham 0.532, BLOSUM62 0.524.
=> A couple of signals (structural context, ESM) carry most of the power; the
hand-crafted physicochemical features are individually near chance.

### 8.2 Combining signals (Q1, Q2): 10 assays, out-of-fold

| strategy | mean AUC | worst-case AUC |
|---|---|---|
| naive average | 0.681 | 0.600 |
| reliability-weighted | 0.713 | 0.610 |
| best single signal | 0.715 | 0.606 |
| learned logistic (CV) | 0.727 | 0.636 |
| **learned GBM (CV)** | **0.811** | **0.691** |

Paired test: **learned logistic beats naive average in 10/10 assays**
(sign test p ~ 0.002), mean improvement +0.046 AUC.

Per-assay GBM AUC: CcdB 0.956 / 0.929, BLAT 0.780, GFP 0.726, SPG1 0.881,
PTEN 0.812, UBC9 0.734, NUD15 0.867, AICDA 0.736, POLG 0.691.

### 8.3 Verifying the strong CcdB number (no leakage)
GBM hit 0.956 on CcdB Adkar, so we checked it:
- Dataset folds **0.956** vs fresh **random 5-fold 0.958** -> no fold-structure leakage.
- **Spearman(prediction, continuous fitness) = +0.766** -> independent corroboration.
- **Shuffled-labels control = 0.479** (~0.5) -> the model has no spurious ability to
  fit random labels, so the signal is genuine.

---

## 9. Answers to the central questions

- **Q1 (how to combine):** Do NOT naive-average. It was the worst strategy on every
  metric and in every protein class. Weighting helps; a **non-linear learned model
  (gradient boosting), cross-validated**, is best (0.811 mean / 0.691 worst), because
  the signals interact (e.g. "buried AND chemically severe AND low-ESM" is far worse
  than any one alone), which a linear average cannot represent.
- **Q2 (breadth):** The result holds across 9 protein classes (toxin, resistance
  enzyme, fluorescent protein, binding domain, phosphatase, SUMO enzyme, hydrolase,
  deaminase, virus). Learned > naive in 10/10.
- **Q3 (which signals):** Structural context (pLDDT) and the ESM language model carry
  most of the power; the hand-crafted physicochemical features are weak individually
  but add value inside the non-linear model.

Design recommendation for the bioplausibility measure: assemble the strong signals
(structural context, ESM marginal, WCN burial, hydropathy, plus the weaker
physicochemical ones), combine with a cross-validated gradient-boosting model, and
always validate out-of-sample on DMS with a shuffle control.

---

## 10. Exact reproduction steps

```bash
# 0. environment (Python 3.9)
pip3 install --user numpy scipy scikit-learn biopython torch fair-esm

# 1. data: DMS assays (TSV)
mkdir -p data/proteingym data/structures data/esm
base=https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/resolve/main/singles_substitutions
for A in CCDB_ECOLI_Adkar_2012 CCDB_ECOLI_Tripathi_2016 BLAT_ECOLX_Deng_2012 \
         GFP_AEQVI_Sarkisyan_2016 SPG1_STRSG_Olson_2014 PTEN_HUMAN_Matreyek_2021 \
         UBC9_HUMAN_Weile_2017 NUD15_HUMAN_Suiter_2020 \
         AICDA_HUMAN_Gajula_2014_3cycles POLG_CXB3N_Mattenberger_2021; do
  curl -sL -o data/proteingym/$A.tsv $base/$A.tsv
done

# 2. structures: resolve UniProt -> AlphaFold (see Section 3.2), save as
#    data/structures/<ENTRY>.pdb  (the download loop is in the project history)

# 3. cache ESM signals (one forward pass per assay)
python3 compute_esm_scores.py

# 4. run the full analysis (single signals, 4 combiners, bootstrap CIs, paired test)
python3 multi_assay_analysis.py

# 5. single-assay deep dive + verification controls
python3 empirical_bioplausibility.py data/proteingym/CCDB_ECOLI_Adkar_2012.tsv \
        data/structures/CCDB_ECOLI.pdb
```

Code map: `bps_p.py` (Grantham, structural, BPS-P), `fvs.py` (severity, structure
load, functional viability), `constraints_extra.py` (BLOSUM, hydropathy, secondary
structure, aggregation, burial), `compute_esm_scores.py` (ESM cache),
`multi_assay_analysis.py` (signals + 4 combiners + stats),
`empirical_bioplausibility.py` (single-assay + controls).

Proto (for using the metric as a design Constraint) installs in a Python 3.12 venv:
```bash
python3.12 -m venv protoenv
protoenv/bin/pip install "git+https://github.com/evo-design/proto-language.git"
```
(The PyPI release 0.1.0 is missing its `proto_tools` dependency; install from source.)

---

## 11. Limitations and honest caveats

- DMS fitness is a proxy for "naturalness." A direct realism check (natural-ortholog
  vs random substitution distributions, Kolmogorov-Smirnov) is future work (PRISM RQ4).
- ESM rank-normalization is transductive (uses the test feature distribution, no
  labels). It is standard for feature scaling but should ideally be fit on train only.
- The GBM can overfit small assays (AICDA, n=209); for small proteins prefer the
  regularized logistic model.
- Burial uses Weighted Contact Number, a CA-only surrogate. True DSSP / Biotite SASA
  could improve it (Biotite is now installed but not yet swapped in).
- POLG used sequence-only signals (no single AlphaFold model for 2,185 aa).
- Labels were binarized at the median for AUC; Spearman on continuous fitness is the
  more faithful metric and was reported alongside.
- Single-mutant focus; multi-mutant variants are handled by averaging per-edit
  signals and summing the ESM marginal over edits.

---

## 12. References (links)

- Grantham (1974), Science. https://doi.org/10.1126/science.185.4154.862
- Henikoff & Henikoff (1992), PNAS (BLOSUM). https://doi.org/10.1073/pnas.89.22.10915
- Kyte & Doolittle (1982), JMB. https://doi.org/10.1016/0022-2836(82)90515-0
- Chou & Fasman (1978), Annu Rev Biochem. https://doi.org/10.1146/annurev.bi.47.070178.001343
- Fernandez-Escamilla et al. (2004), Nat Biotechnol (TANGO). https://doi.org/10.1038/nbt1012
- Lin et al. (2008) / Yeh et al. (2014), Weighted Contact Number. https://doi.org/10.1093/molbev/mst196
- Pucci et al. (2018), Sci Rep. https://doi.org/10.1038/s41598-018-22531-2
- Conservation + solvent accessibility (2025), bioRxiv. https://www.biorxiv.org/content/10.1101/2025.02.03.636212
- Meier et al. (2021), NeurIPS (ESM variant effects). https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html
- Lin et al. (2023), Science (ESM2 / ESMFold). https://doi.org/10.1126/science.ade2574
- Notin et al. (2023), ProteinGym. https://proteingym.org
- Jumper et al. (2021), Nature (AlphaFold2). https://doi.org/10.1038/s41586-021-03819-2
- Varadi et al. (2022), NAR (AlphaFold DB). https://doi.org/10.1093/nar/gkab1061
- Data mirror: https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS
- Proto: Hie et al. (2026), bioRxiv. https://www.biorxiv.org/content/10.64898/2026.06.22.733870v1
