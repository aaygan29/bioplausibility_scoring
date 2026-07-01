"""
validate_fvs.py — validate the Functional Viability Score on a ProteinGym assay.

Usage:
  python3 validate_fvs.py <assay.csv> <structure.pdb> [rasp_ddg.csv] [active_sites.json]

- <assay.csv>     ProteinGym DMS substitution file (mutant, mutated_sequence, DMS_score, DMS_score_bin)
- <structure.pdb> matching AF2 structure (for F_site coords)
- [rasp_ddg.csv]  RaSP saturation output -> real V_fold (else Grantham stub)
- [active_sites]  JSON {assay_key: {idx0: weight}} -> real F_site (else empty)

Reports Spearman(FVS, DMS_score) and AUC(FVS, DMS_score_bin), with the same
metric on the additive BPS-P for direct comparison.
"""
from __future__ import annotations
import csv
import json
import sys

import numpy as np

from fvs import fvs, load_af2_ca, rasp_ddg_stub
from ddg import load_rasp_table, ddg_for_variant
from bps_p import bps_p

try:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    _SCI = True
except Exception:
    _SCI = False


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    assay_csv, pdb = sys.argv[1], sys.argv[2]
    rasp_csv = sys.argv[3] if len(sys.argv) > 3 else None
    sites_json = sys.argv[4] if len(sys.argv) > 4 else None

    ca, _ = load_af2_ca(pdb)
    rasp = load_rasp_table(rasp_csv) if rasp_csv else None
    assay_key = assay_csv.split("/")[-1].replace(".csv", "")
    active = {}
    if sites_json:
        allsites = json.load(open(sites_json))
        # match by UniProt-ish prefix
        for k, v in allsites.items():
            if k in assay_key:
                active = {int(i): float(w) for i, w in v.items()}
                break

    rows = list(csv.DictReader(open(assay_csv)))
    fvs_scores, bps_scores, fit, lab = [], [], [], []
    for r in rows:
        var = r["mutated_sequence"]
        wt = list(var)
        for mut in r["mutant"].split(":"):
            wt[int(mut[1:-1]) - 1] = mut[0]
        wt = "".join(wt)

        if rasp is not None:
            d = ddg_for_variant(wt, var, rasp)
            if d is None:
                d = rasp_ddg_stub(wt, var)
        else:
            d = rasp_ddg_stub(wt, var)

        fvs_scores.append(fvs(wt, var, ca, active, ddg=d).score)
        bps_scores.append(bps_p(wt, var, functional_weights=active).score)
        fit.append(float(r["DMS_score"]))
        lab.append(int(r["DMS_score_bin"]) if r.get("DMS_score_bin") not in (None, "") else None)

    fvs_scores = np.array(fvs_scores); bps_scores = np.array(bps_scores)
    fit = np.array(fit); lab = np.array([x for x in lab]) if None not in lab else None

    print("=" * 60)
    print(f"FVS vs BPS-P on {assay_key}   (RaSP={'yes' if rasp else 'STUB'}, "
          f"active_site={'yes' if active else 'EMPTY'})")
    print("=" * 60)
    if not _SCI:
        print("install scipy scikit-learn for stats"); return
    print(f"  Construct (Spearman vs DMS_score):  FVS={spearmanr(fvs_scores, fit)[0]:+.3f}   "
          f"BPS-P={spearmanr(bps_scores, fit)[0]:+.3f}")
    if lab is not None and len(set(lab)) == 2:
        print(f"  Predictive (AUC vs DMS_score_bin):  FVS={roc_auc_score(lab, fvs_scores):.3f}   "
              f"BPS-P={roc_auc_score(lab, bps_scores):.3f}")
    print(f"  (reference: ESM2-650M zero-shot AUC on CcdB = 0.761)")
    print("=" * 60)


if __name__ == "__main__":
    main()
