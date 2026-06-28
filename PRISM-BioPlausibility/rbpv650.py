"""
rbpv650.py — rBPV with ESM2-650M added to the naturalness ensemble.

650M (the most capable scale) is used where sequence length fits CPU memory
(<=1000 aa: CcdB, F7YBW8, AICDA). POLG (2185 aa) keeps 150M. This tests whether
adding capacity to the S_nat ensemble (Eq. 5) lifts the robust floor.
"""
from __future__ import annotations
import csv, math
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

from run_all import wt_marginal, reconstruct_wt, load_af2_ca, active_for, SUB, STR
from fvs import f_site
from bps_p import s_grantham

TARGETS = [
    ("CCDB_ECOLI_Adkar_2012",          "CCDB_ECOLI",   "toxin"),
    ("CCDB_ECOLI_Tripathi_2016",       "CCDB_ECOLI",   "toxin-rep"),
    ("F7YBW8_MESOW_Aakre_2015",        "F7YBW8_MESOW", "ParE-multi"),
    ("F7YBW8_MESOW_Ding_2023",         "F7YBW8_MESOW", "ParE-multi-rep"),
    ("AICDA_HUMAN_Gajula_2014_3cycles","AICDA_HUMAN",  "enzyme"),
    ("POLG_CXB3N_Mattenberger_2021",   "POLG_CXB3N",   "virus"),
]

def rank_norm(x):
    x = np.asarray(x, float)
    return (rankdata(x) - 1) / max(1, len(x) - 1)

def features(assay, struct):
    rows = list(csv.DictReader(open(f"{SUB}/{assay}.csv")))
    wt = reconstruct_wt(rows[0]); L = len(wt)
    variants = [r["mutated_sequence"] for r in rows]
    s35 = wt_marginal(wt, variants, "esm2_t12_35M_UR50D")
    scales = [("35M", np.array([s35[v] for v in variants]))]
    if L <= 2300:  # 150M handles POLG (2185aa) fine per probe.py
        s150 = wt_marginal(wt, variants, "esm2_t30_150M_UR50D")
        scales.append(("150M", np.array([s150[v] for v in variants])))
    if L <= 1000:
        s650 = wt_marginal(wt, variants, "esm2_t33_650M_UR50D")
        scales.append(("650M", np.array([s650[v] for v in variants])))
    used = "+".join(n for n, _ in scales)
    s_nat = np.mean([rank_norm(v) for _, v in scales], axis=0)  # Eq.5 ensemble
    best = scales[-1][1]                       # largest available scale
    v_fold = 1.0 / (1.0 + np.exp(-best / 3.0))
    ca, _ = load_af2_ca(f"{STR}/{struct}.pdb"); active = active_for(assay)
    fst, sevo = [], []
    for r in rows:
        wtl = reconstruct_wt(r); var = r["mutated_sequence"]
        fst.append(f_site(wtl, var, ca, active)); sevo.append(s_grantham(wtl, var))
    X = np.column_stack([s_nat, v_fold, np.array(fst), np.array(sevo)])
    y = np.array([int(r["DMS_score_bin"]) for r in rows])
    return X, y, used

def cv_auc(X, y):
    if len(set(y)) < 2: return float("nan")
    return float(np.mean(cross_val_score(
        LogisticRegression(max_iter=2000), X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")))

def main():
    print(f"{'assay':<32}{'scales':<14}{'n':>6}  {'rBPV+650(cv)':>13}")
    print("-" * 70)
    rb = []
    for assay, struct, klass in TARGETS:
        X, y, used = features(assay, struct)
        a = cv_auc(X, y); rb.append(a)
        print(f"{assay[:31]:<32}{used:<14}{len(y):>6}  {a:>13.3f}", flush=True)
    rb = [v for v in rb if not math.isnan(v)]
    print("-" * 70)
    print(f"rBPV+650  mean={np.mean(rb):.3f}  worst={np.min(rb):.3f}")
    print("(compare rbpv.py without 650M: mean=0.711 worst=0.572)")

if __name__ == "__main__":
    main()
