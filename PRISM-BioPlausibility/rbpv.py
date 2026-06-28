"""
rbpv.py — Robust Bioplausibility-Viability Score (rBPV), validated.

Implements the robust combination from ROBUST_METRIC.md: a cross-validated,
calibrated weighted model over an applicability-masked, multi-scale component set.
Demonstrates the core claim: the robust combination beats any single component on
WORST-CASE and average across diverse molecule classes.

Components per variant:
  S_nat   : ESM2 wt-marginal naturalness (ENSEMBLE over 35M + 150M scales)
  V_fold  : sigmoid(ESM2-150M) folded-tolerance proxy
  F_site  : 3D active-site preservation (applicability-masked)
  S_evo   : Grantham conservatism
Labels: ProteinGym DMS_score_bin. Metric: 5-fold CV held-out AUC.
"""
from __future__ import annotations
import csv, json, math
import numpy as np
from scipy.stats import rankdata, spearmanr
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
    wt = reconstruct_wt(rows[0])
    variants = [r["mutated_sequence"] for r in rows]
    big = "esm2_t30_150M_UR50D" if len(wt) <= 1200 else "esm2_t12_35M_UR50D"
    s150 = wt_marginal(wt, variants, big)
    s35 = wt_marginal(wt, variants, "esm2_t12_35M_UR50D")
    ca, _ = load_af2_ca(f"{STR}/{struct}.pdb")
    active = active_for(assay)

    e150 = np.array([s150[v] for v in variants])
    e35 = np.array([s35[v] for v in variants])
    # ENSEMBLE naturalness: rank-normalized mean of scales (Eq. 5, equal r here)
    s_nat = 0.5 * rank_norm(e150) + 0.5 * rank_norm(e35)
    v_fold = 1.0 / (1.0 + np.exp(-e150 / 3.0))
    fst, sevo = [], []
    for r in rows:
        wtl = reconstruct_wt(r); var = r["mutated_sequence"]
        fst.append(f_site(wtl, var, ca, active))
        sevo.append(s_grantham(wtl, var))
    X = np.column_stack([s_nat, v_fold, np.array(fst), np.array(sevo)])
    y = np.array([int(r["DMS_score_bin"]) for r in rows])
    return X, y, ["S_nat(ens)", "V_fold", "F_site", "S_evo"]

def cv_auc(X, y):
    if len(set(y)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")))

def main():
    print(f"{'assay':<32}{'class':<16}{'n':>6}  "
          f"{'S_nat':>7}{'V_fold':>7}{'F_site':>7}{'S_evo':>7}  {'rBPV(cv)':>9}")
    print("-" * 100)
    singles_worst, rbpv_worst, singles_all, rbpv_all = {}, [], {f:[] for f in
        ["S_nat(ens)","V_fold","F_site","S_evo"]}, []
    for assay, struct, klass in TARGETS:
        X, y, names = features(assay, struct)
        single = {names[i]: roc_auc_score(y, X[:, i]) if len(set(y)) == 2 else float("nan")
                  for i in range(X.shape[1])}
        # orient each single feature the same way AUC<0.5 -> flip for fair "best single"
        single_oriented = {k: max(v, 1 - v) for k, v in single.items()}
        rb = cv_auc(X, y)
        print(f"{assay[:31]:<32}{klass:<16}{len(y):>6}  "
              f"{single['S_nat(ens)']:>7.3f}{single['V_fold']:>7.3f}"
              f"{single['F_site']:>7.3f}{single['S_evo']:>7.3f}  {rb:>9.3f}")
        for k in singles_all: singles_all[k].append(single_oriented[k])
        rbpv_all.append(rb)
    print("-" * 100)
    print("ROBUSTNESS SUMMARY (across the 6 assays):")
    for k, vs in singles_all.items():
        vs = [v for v in vs if not math.isnan(v)]
        print(f"  best-single {k:<11} mean={np.mean(vs):.3f}  worst={np.min(vs):.3f}")
    rb = [v for v in rbpv_all if not math.isnan(v)]
    print(f"  rBPV (CV combo)         mean={np.mean(rb):.3f}  worst={np.min(rb):.3f}")
    print("\nRobustness = high WORST-CASE + high mean. The combo should lift the floor.")

if __name__ == "__main__":
    main()
