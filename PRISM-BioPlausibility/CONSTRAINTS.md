# Constraint library: biological signals for bioplausibility & function

The metric now scores an edit on a richer set of biologically-grounded signals.
Each returns [0,1] with **higher = more plausible / more tolerable**, and inverts
to a Proto constraint as `proto = 1 - score`. Code: `constraints_extra.py`,
wired into `BioPlausibilityScorer` (`proto_bioplausibility.py`).

## Core signals (BPS-P / FVS)
| Signal | Measures | Basis |
|---|---|---|
| Grantham | chemical cost of the swap | Grantham, 1974 |
| Active-site (F_site) | 3D-distance-weighted preservation of functional residues | this work |
| Fold (V_fold) | folded fraction from ΔΔG stability | Blaabjerg et al., 2023 |
| Production (P_prod) | frameshift / premature-stop integrity | this work |
| Codon | expression plausibility | Sharp & Li, 1987 |

## New signals (this addition)
| Signal | Measures | Why it matters | Basis |
|---|---|---|---|
| **Burial** | severity scaled by how buried the residue is (CA contact-number proxy for solvent accessibility) | Solvent accessibility is the single most predictive structural feature; buried mutations are far more deleterious | Pucci et al., 2018; "conservation + RSA are almost all you need", 2025 |
| **BLOSUM62** | evolutionary substitution tolerance | Complements Grantham: chemistry vs. how often evolution actually accepts a swap | Henikoff & Henikoff, 1992 |
| **Hydropathy** | hydrophobicity change, weighted by burial | Burying a polar residue or exposing the core is a classic destabilizer | Kyte & Doolittle, 1982 |
| **Secondary structure** | loss of helix/sheet propensity; Pro/Gly insertion | Proline breaks helices/sheets; matches PRISM's "secondary-structure compatibility" | Chou & Fasman, 1978 |
| **Aggregation** | increase in local hydrophobic β-sheet propensity | Destabilizing edits can nucleate aggregation, which kills function | TANGO; Fernandez-Escamilla et al., 2004 |
| **Conservation** (hook) | how often nature uses the mutant residue at that column (MSA frequency) | Position-specific conservation is among the strongest deleteriousness signals | ConSurf; Hopf et al., 2017 |

## Structural / functional analysis tool
`scorer.analyze(variant)` returns a per-edit diagnostic that **explains** the score:
position, wild-type/mutant, Grantham, BLOSUM62, burial, hydropathy change, helix &
sheet propensity change, distance to the active site, and flags for proline
insertion and disulfide breakage. Use it to understand *why* an edit is tolerated
or rejected, not just the final number.

## How to use
```python
scorer.extras(variant)                 # all new signals as a dict
scorer.extended_plausibility(variant)  # core plausibility blended with extras
scorer.analyze(variant)                # per-edit structural/functional explanation

# as Proto constraints (0 = best):
from proto_bioplausibility import make_extra_constraint, extended_plausibility_constraint
make_extra_constraint("burial")        # or blosum62, hydropathy, secondary_structure, aggregation, conservation
```

## Not yet implemented (clear next steps)
- **Epistasis / coevolution** (EVmutation / Potts): needs an MSA-derived coupling
  model; the strongest remaining signal (Hopf et al., 2017). The `conservation`
  hook already reads an MSA, so this is the natural extension.
- **Real ΔΔG** via RaSP for `V_fold` (see `RASP_SETUP.md`).
- **DSSP-based** solvent accessibility and secondary structure (currently
  approximated from CA geometry and residue propensities).

## References
- Grantham, R. (1974). Science. https://doi.org/10.1126/science.185.4154.862
- Henikoff, S. & Henikoff, J. (1992). Amino acid substitution matrices (BLOSUM). PNAS. https://doi.org/10.1073/pnas.89.22.10915
- Kyte, J. & Doolittle, R. (1982). A simple method for displaying hydropathy. JMB. https://doi.org/10.1016/0022-2836(82)90515-0
- Chou, P. & Fasman, G. (1978). Empirical predictions of protein conformation. Annu. Rev. Biochem. https://doi.org/10.1146/annurev.bi.47.070178.001343
- Fernandez-Escamilla, A.-M., et al. (2004). TANGO: aggregation prediction. Nat. Biotechnol. https://doi.org/10.1038/nbt1012
- Pucci, F., et al. (2018). Deleterious variants and structural stability. Sci. Rep. https://doi.org/10.1038/s41598-018-22531-2
- Hopf, T. A., et al. (2017). Mutation effects predicted from sequence co-variation (EVmutation). Nat. Biotechnol. https://doi.org/10.1038/nbt.3769
- Conservation + solvent accessibility for mutational effects (2025). bioRxiv. https://www.biorxiv.org/content/10.1101/2025.02.03.636212
