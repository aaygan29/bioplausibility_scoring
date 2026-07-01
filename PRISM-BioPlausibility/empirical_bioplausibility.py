"""
empirical_bioplausibility.py — empirically decide how to COMBINE bioplausibility
signals, using real DMS data with the dataset's own cross-validation folds.

Answers the practical question: should we just average many metrics, or weight them?
We compare, out-of-sample:
  (a) each single signal alone
  (b) naive average of all signals (equal weights)
  (c) learned weights (cross-validated logistic regression)
  (d) reliability-weighted average (down-weight signals that don't generalize)

Ground truth: measured DMS fitness (functional tolerance). Biophysical plausibility
signals are expected to predict tolerance (Frazer/CG 2025; Pucci 2018).

Data format (genbio-ai ProteinGYM-DMS tsv): columns sequences, labels, fold_id.
Single-mutant saturation assays -> wild type = per-position consensus.

Usage:  python3 empirical_bioplausibility.py data/proteingym/CCDB_ECOLI_Adkar_2012.tsv data/structures/CCDB_ECOLI.pdb
"""
from __future__ import annotations
import csv, sys
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bps_p import grantham_distance, G_MAX, load_af2_plddt
from fvs import load_af2_ca, severity
from constraints_extra import burial_fraction, _KD, _PA, _PB, s_aggregation


def load_assay(tsv):
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    seqs = [r["sequences"] for r in rows]
    y = np.array([float(r["labels"]) for r in rows])
    fold = np.array([int(r.get("fold_id", 0)) for r in rows])
    L = len(seqs[0])
    wt = "".join(Counter(s[i] for s in seqs).most_common(1)[0][0] for i in range(L))
    return seqs, y, fold, wt


def signals(seqs, wt, ca, plddt):
    """Per-variant biophysical signals, each in [0,1], higher = more tolerable."""
    bur = burial_fraction(ca) if ca else {}
    names = ["grantham", "blosum62", "burial", "hydropathy", "sec_struct",
             "aggregation", "plddt_region", "not_active_proxy"]
    try:
        from Bio.Align import substitution_matrices
        BL = substitution_matrices.load("BLOSUM62"); haveBL = True
    except Exception:
        haveBL = False
    X = []
    for s in seqs:
        e = [(i, wt[i], s[i]) for i in range(len(wt)) if i < len(s) and wt[i] != s[i]]
        if not e:
            X.append([1.0] * len(names)); continue
        i, a, b = e[0]                      # single-mutant assays
        b_i = bur.get(i + 1, 0.0)
        sev = severity(a, b)
        gr = 1 - grantham_distance(a, b) / G_MAX
        bl = (float(BL[(a, b)]) + 4) / 15 if haveBL else 1.0
        burial_s = 1 - b_i * sev
        hyd = 1 - b_i * min(1.0, abs(_KD[a] - _KD[b]) / 9.0)
        ss = 1 - min(1.0, max(0.0, max(_PA[a], _PB[a]) - max(_PA[b], _PB[b])) / 1.7
                     - (0.0 if b != "P" else 0.6))
        ss = float(np.clip(ss, 0, 1))
        agg = s_aggregation(wt, s)
        pl = 1 - (plddt[i] / 100.0 if i < len(plddt) else 0.7)
        nap = 1.0                           # placeholder active-site proxy (no annot here)
        X.append([gr, np.clip(bl, 0, 1), burial_s, hyd, ss, agg, pl, nap])
    return np.array(X), names


def cv_predictions(X, y_bin, fold):
    """Out-of-fold logistic predictions using the dataset's folds."""
    pred = np.zeros(len(y_bin))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        if len(set(y_bin[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y_bin[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return pred


def reliability_weighted(X, y_bin, fold, names):
    """Average signals weighted by each signal's out-of-fold AUC (down-weight weak ones)."""
    pred = np.zeros(len(y_bin))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        if len(set(y_bin[tr])) < 2:
            continue
        w = []
        for j in range(X.shape[1]):
            try:
                a = roc_auc_score(y_bin[tr], X[tr, j])
            except Exception:
                a = 0.5
            w.append(max(0.0, a - 0.5))       # reliability above chance
        w = np.array(w); w = w / (w.sum() or 1)
        pred[te] = X[te] @ w
    return pred


def evaluate(name, score, y_bin, y_cont):
    try:
        auc = roc_auc_score(y_bin, score)
    except Exception:
        auc = float("nan")
    rho = spearmanr(score, y_cont)[0]
    return name, max(auc, 1 - auc), rho   # orient single signals fairly


def main():
    tsv, pdb = sys.argv[1], sys.argv[2]
    seqs, y, fold, wt = load_assay(tsv)
    ca, _ = load_af2_ca(pdb); plddt = load_af2_plddt(pdb)
    if len(wt) != len(plddt):
        print(f"warn: WT len {len(wt)} != structure {len(plddt)}; sequence signals only")
    X, names = signals(seqs, wt, ca, plddt)
    thr = np.median(y); y_bin = (y > thr).astype(int)

    print("=" * 64)
    print(f"{tsv.split('/')[-1]}   n={len(seqs)}  folds={sorted(set(fold))}")
    print("=" * 64)
    print("Single signals (oriented):           AUC    |Spearman|")
    for j, nm in enumerate(names):
        _, a, r = evaluate(nm, X[:, j], y_bin, y)
        print(f"  {nm:18s}                {a:.3f}    {abs(r):.3f}")
    best_single = max(roc_auc_score(y_bin, X[:, j]) for j in range(X.shape[1]))
    best_single = max(best_single, 1 - min(roc_auc_score(y_bin, X[:, j]) for j in range(X.shape[1])))

    print("\nCombination strategies (out-of-fold):  AUC    |Spearman|")
    naive = X.mean(axis=1)
    _, a_naive, r_naive = evaluate("naive average", naive, y_bin, y)
    print(f"  (a) naive average           {a_naive:.3f}    {abs(r_naive):.3f}")
    rw = reliability_weighted(X, y_bin, fold, names)
    _, a_rw, r_rw = evaluate("reliability-weighted", rw, y_bin, y)
    print(f"  (b) reliability-weighted    {a_rw:.3f}    {abs(r_rw):.3f}")
    lc = cv_predictions(X, y_bin, fold)
    a_lc = roc_auc_score(y_bin, lc); r_lc = spearmanr(lc, y)[0]
    print(f"  (c) learned (logistic CV)   {a_lc:.3f}    {abs(r_lc):.3f}")
    print(f"\n  best single signal AUC = {max(a for _, a, _ in [evaluate(n, X[:,j], y_bin, y) for j,n in enumerate(names)]):.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
