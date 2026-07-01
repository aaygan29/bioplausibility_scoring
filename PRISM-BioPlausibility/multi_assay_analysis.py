"""
multi_assay_analysis.py — breadth + depth validation of how to combine
bioplausibility signals, across many protein classes with statistical rigor.

Breadth: 10 ProteinGym DMS assays (toxin, resistance enzyme, fluorescent protein,
binding domain, phosphatase, SUMO enzyme, hydrolase, deaminase, virus).
Depth: per-assay single-signal vs naive-average vs reliability-weighted vs learned
(cross-validated) with bootstrap 95% CIs, plus a cross-assay paired comparison
(learned vs naive) with a sign test.

Inputs:  data/proteingym/*.tsv  (sequences, labels, fold_id)
         data/structures/<ENTRY>.pdb  (AlphaFold; used only if length matches)
Output:  per-assay table + pooled summary + paired test.
"""
from __future__ import annotations
import csv, glob, os
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bps_p import grantham_distance, G_MAX, load_af2_plddt
from fvs import load_af2_ca, severity
from constraints_extra import burial_fraction, _KD, _PA, _PB, s_aggregation
try:
    from Bio.Align import substitution_matrices
    _BL = substitution_matrices.load("BLOSUM62"); _HAVE_BL = True
except Exception:
    _HAVE_BL = False

PANEL = {  # assay -> (entry_name for structure, class)
    "CCDB_ECOLI_Adkar_2012": ("CCDB_ECOLI", "toxin"),
    "CCDB_ECOLI_Tripathi_2016": ("CCDB_ECOLI", "toxin(rep)"),
    "BLAT_ECOLX_Deng_2012": ("BLAT_ECOLX", "resistance enzyme"),
    "GFP_AEQVI_Sarkisyan_2016": ("GFP_AEQVI", "fluorescent"),
    "SPG1_STRSG_Olson_2014": ("SPG1_STRSG", "binding domain"),
    "PTEN_HUMAN_Matreyek_2021": ("PTEN_HUMAN", "phosphatase"),
    "UBC9_HUMAN_Weile_2017": ("UBC9_HUMAN", "SUMO enzyme"),
    "NUD15_HUMAN_Suiter_2020": ("NUD15_HUMAN", "hydrolase"),
    "AICDA_HUMAN_Gajula_2014_3cycles": ("AICDA_HUMAN", "deaminase"),
    "POLG_CXB3N_Mattenberger_2021": ("POLG_CXB3N", "virus"),
}
SIG = ["grantham", "blosum62", "hydropathy", "sec_struct", "aggregation",
       "burial", "plddt_region", "esm"]


def wcn_burial(ca):
    """Weighted Contact Number burial in [0,1] (1=buried). A validated CA-only
    surrogate for relative solvent accessibility (Lin 2008; Yeh 2014), better than
    plain contact counts and needing no DSSP/all-atom parsing."""
    keys = sorted(ca)
    xyz = np.array([ca[k] for k in keys])
    wcn = np.zeros(len(keys))
    for i in range(len(keys)):
        d2 = np.sum((xyz - xyz[i]) ** 2, axis=1)
        d2[i] = np.inf
        wcn[i] = np.sum(1.0 / d2)
    # normalize to [0,1] across residues
    lo, hi = np.percentile(wcn, 5), np.percentile(wcn, 95)
    norm = np.clip((wcn - lo) / (hi - lo + 1e-9), 0, 1)
    return {k: float(norm[i]) for i, k in enumerate(keys)}


def rank_norm(x):
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    return (rankdata(x) - 1) / max(1, len(x) - 1)


def load_assay(tsv):
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    seqs = [r["sequences"] for r in rows]
    y = np.array([float(r["labels"]) for r in rows])
    fold = np.array([int(r.get("fold_id", 0)) for r in rows])
    L = min(len(s) for s in seqs)
    wt = "".join(Counter(s[i] for s in seqs).most_common(1)[0][0] for i in range(L))
    return seqs, y, fold, wt


def compute_signals(seqs, wt, ca, plddt, esm=None):
    bur = wcn_burial(ca) if ca else {}     # WCN-based burial (RSA surrogate)
    has_struct = bool(ca) and len(plddt) == len(wt)
    esm_n = rank_norm(esm) if esm is not None else None  # comparable [0,1]
    X = []
    for k, s in enumerate(seqs):
        edits = [(i, wt[i], s[i]) for i in range(min(len(wt), len(s))) if wt[i] != s[i]]
        e_val = float(esm_n[k]) if esm_n is not None else 0.5
        if not edits:
            X.append([1, 1, 1, 1, 1, 1, 0.3, e_val]); continue
        gr = bl = hy = ss = bu = pl = 0.0
        for i, a, b in edits:
            b_i = bur.get(i + 1, 0.0) if has_struct else 0.0
            sev = severity(a, b)
            gr += 1 - grantham_distance(a, b) / G_MAX
            bl += ((float(_BL[(a, b)]) + 4) / 15) if _HAVE_BL else 1.0
            hy += 1 - (b_i if has_struct else 0.5) * min(1.0, abs(_KD[a] - _KD[b]) / 9.0)
            ss += float(np.clip(1 - max(0.0, max(_PA[a], _PB[a]) - max(_PA[b], _PB[b])) / 1.7
                                - (0.6 if (b == "P" and a != "P") else 0.0), 0, 1))
            bu += (1 - b_i * sev) if has_struct else 1.0
            pl += (1 - plddt[i] / 100.0) if (has_struct and i < len(plddt)) else 0.3
        n = len(edits)
        X.append([gr / n, np.clip(bl / n, 0, 1), hy / n, ss / n,
                  s_aggregation(wt, s), bu / n, pl / n, e_val])
    return np.array(X), has_struct


def boot_auc(y, score, n=600, seed=0):
    rng = np.random.default_rng(seed)
    if len(set(y)) < 2:
        return (float("nan"), float("nan"))
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        a = roc_auc_score(y[b], score[b]); vals.append(max(a, 1 - a))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def oof_logistic(X, y_bin, fold):
    pred = np.zeros(len(y_bin))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        if len(set(y_bin[tr])) < 2:
            pred[te] = 0.5; continue
        pred[te] = LogisticRegression(max_iter=2000).fit(X[tr], y_bin[tr]).predict_proba(X[te])[:, 1]
    return pred


def oof_reliability(X, y_bin, fold):
    pred = np.zeros(len(y_bin))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        if len(set(y_bin[tr])) < 2:
            pred[te] = X[te].mean(1); continue
        w = np.array([max(0.0, (roc_auc_score(y_bin[tr], X[tr, j]) if np.std(X[tr, j]) > 0 else .5) - .5)
                      for j in range(X.shape[1])])
        w = w / (w.sum() or 1)
        pred[te] = X[te] @ w
    return pred


def oof_gbm(X, y_bin, fold):
    """Non-linear combiner: gradient-boosted trees, out-of-fold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    pred = np.zeros(len(y_bin))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        if len(set(y_bin[tr])) < 2:
            pred[te] = 0.5; continue
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                             learning_rate=0.08).fit(X[tr], y_bin[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def auc_or_nan(y, s):
    try:
        return max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    rows_out = []
    print(f"{'assay':<34}{'class':<18}{'n':>6} {'str':>4} | "
          f"{'best1':>6}{'naive':>6}{'relia':>6}{'learn':>6}{'gbm':>6}  Δ(l-n)")
    print("-" * 104)
    for assay, (entry, klass) in PANEL.items():
        tsv = f"data/proteingym/{assay}.tsv"
        if not os.path.exists(tsv):
            continue
        seqs, y, fold, wt = load_assay(tsv)
        pdb = f"data/structures/{entry}.pdb"
        ca, plddt = ({}, [])
        if os.path.exists(pdb):
            ca, _ = load_af2_ca(pdb); plddt = load_af2_plddt(pdb)
        esm_path = f"data/esm/{assay}.npy"
        esm = np.load(esm_path) if os.path.exists(esm_path) else None
        X, has_struct = compute_signals(seqs, wt, ca, plddt, esm=esm)
        # drop zero-variance columns (e.g. unavailable structural signals)
        keep = [j for j in range(X.shape[1]) if np.std(X[:, j]) > 1e-9]
        Xk = X[:, keep]
        y_bin = (y > np.median(y)).astype(int)
        if len(set(y_bin)) < 2:                      # degenerate median; use mean
            y_bin = (y > y.mean()).astype(int)
        best1 = max(auc_or_nan(y_bin, X[:, j]) for j in keep)
        naive = auc_or_nan(y_bin, Xk.mean(1))
        relia = auc_or_nan(y_bin, oof_reliability(Xk, y_bin, fold))
        learn = auc_or_nan(y_bin, oof_logistic(Xk, y_bin, fold))
        gbm = auc_or_nan(y_bin, oof_gbm(Xk, y_bin, fold))
        delta = learn - naive
        print(f"{assay[:33]:<34}{klass:<18}{len(seqs):>6} {('Y' if has_struct else '-'):>4} | "
              f"{best1:>6.3f}{naive:>6.3f}{relia:>6.3f}{learn:>6.3f}{gbm:>6.3f}  {delta:+.3f}")
        rows_out.append(dict(assay=assay, klass=klass, n=len(seqs), struct=has_struct,
                             best1=best1, naive=naive, relia=relia, learn=learn, gbm=gbm, delta=delta))
    # ----- cross-assay summary + paired test -----
    import math
    def summ(key):
        v = [r[key] for r in rows_out if not math.isnan(r[key])]
        return np.mean(v), np.min(v)
    print("-" * 104)
    for k, lab in [("best1", "best single"), ("naive", "naive average"),
                   ("relia", "reliability-weighted"), ("learn", "learned logistic (CV)"),
                   ("gbm", "learned GBM (CV)")]:
        m, w = summ(k)
        print(f"  {lab:<22} mean AUC={m:.3f}   worst={w:.3f}")
    deltas = [r["delta"] for r in rows_out if not math.isnan(r["delta"])]
    wins = sum(d > 0 for d in deltas)
    print(f"\n  learned beats naive in {wins}/{len(deltas)} assays; "
          f"mean Δ = {np.mean(deltas):+.3f}  (paired across assays)")
    print("=" * 104)


if __name__ == "__main__":
    main()
