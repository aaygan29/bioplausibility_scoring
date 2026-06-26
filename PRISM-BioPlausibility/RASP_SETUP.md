# Running RaSP to get real ddG (powers the V_fold term)

RaSP predicts how much each mutation destabilizes the fold (ddG, kcal/mol).
We feed that into `V_fold` (the folded-fraction term of FVS).

## Easiest path — Colab (no local install)
1. Open the RaSP Colab: github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg  -> "Colab".
2. Upload the AF2 PDB(s) you already have, e.g.:
   `~/Downloads/ProteinGym/ProteinGym_AF2_structures/CCDB_ECOLI.pdb`
3. Run the saturation-mutagenesis cell. It outputs a CSV with every
   position x amino acid ddG (101 residues x 19 = ~1900 rows for CcdB).
4. Download that CSV to `~/Desktop/PRISM-BioPlausibility/data/rasp_CCDB_ECOLI.csv`.

## Local path (if you prefer)
torch 2.8.0 is already installed. RaSP also needs:
  - `reduce` (adds hydrogens to PDBs) — `conda install -c bioconda reduce`
  - RaSP repo + pretrained weights from the GitHub release.
Follow the repo README; point it at the AF2 PDBs.

## Then wire it in (already coded)
`ddg.py` reads the RaSP CSV; `validate_fvs.py` uses it:

```bash
python3 validate_fvs.py \
  data/CCDB_ECOLI_Adkar_2012.csv \
  ../../Downloads/ProteinGym/ProteinGym_AF2_structures/CCDB_ECOLI.pdb \
  data/rasp_CCDB_ECOLI.csv
```

Without the RaSP CSV it falls back to the Grantham-scaled ddG stub (weak —
for plumbing only). With it, V_fold becomes real thermodynamics.

## Active-site residues (powers the F_site term)
For CcdB use UniProt P62554; pull the residues annotated "Site" / "Binding" /
"Active site" and put them in `data/active_sites.json` as
`{"CCDB_ECOLI": {"<0based_index>": 1.0, ...}}` (catalytic=1.0, binding=0.5).
