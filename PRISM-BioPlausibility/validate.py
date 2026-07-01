"""
validate.py — validation harness for BPS-P (SPEC.md section 4).

Implements the three validity layers + calibration. Runs on a synthetic
labeled set so the statistics work end-to-end today; swap `load_dataset()`
for real ProteinGym DMS assays (use viral-fitness + resistance-enzyme assays
as functional-analog labels — see README overlap note).

Outputs:
  - Spearman rho (construct validity)
  - component correlation matrix + VIF (internal validity)
  - logistic-regression AUC + learned weights (predictive validity, RQ2 answer)
  - Brier score + ECE (calibration)
"""
from __future__ import annotations
import numpy as np

from bps_p import bps_p, BPSWeights, load_af2_plddt

try:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    _HAVE_SCI = True
except Exception:
    _HAVE_SCI = False


def load_proteingym(csv_path: str, functional_quantile: float = 0.5,
                    pdb_path: str = None):
    """REAL loader for a ProteinGym DMS substitution assay CSV.

    Expects columns: mutant, mutated_sequence, DMS_score.
    Reconstructs WT by reverting each single mutant (e.g. 'A24G' -> pos24 back to A).
    label = 1 if DMS_score above the assay's functional_quantile (kept functional).

    Target assays confirmed as SafeProtein analogs (see README overlap table):
      POLG_CXB3N_Mattenberger_2021   (viral fitness)
      CCDB_ECOLI_Adkar_2012          (bacterial toxin, activity)
      F7YBW8_MESOW_Aakre_2015        (toxin, organismal fitness)
      AICDA_HUMAN_Gajula_2014        (enzyme activity)
    """
    import csv
    rows_in = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rows_in.append(row)
    scores = np.array([float(r["DMS_score"]) for r in rows_in])
    thresh = np.quantile(scores, functional_quantile)
    plddt = load_af2_plddt(pdb_path) if pdb_path else None

    rows = []
    for r in rows_in:
        var = r["mutated_sequence"]
        # reconstruct WT from the mutation code(s), e.g. "A24G" or "A24G:L30P"
        wt = list(var)
        for mut in r["mutant"].split(":"):
            wt_aa, pos = mut[0], int(mut[1:-1]) - 1  # ProteinGym is 1-based
            if 0 <= pos < len(wt):
                wt[pos] = wt_aa
        wt = "".join(wt)
        fitness = float(r["DMS_score"])
        # prefer ProteinGym's own binary label if present, else quantile threshold
        label = int(r["DMS_score_bin"]) if r.get("DMS_score_bin") not in (None, "") \
            else int(fitness > thresh)
        # functional_weights left empty here — populate from UniProt per target (TODO)
        rows.append((wt, var, {}, label, fitness))
    return rows, plddt


def load_dataset(n: int = 400, seed: int = 42):
    """SYNTHETIC stand-in. Returns list of (wt, var, functional_weights, label, fitness).
    Replace with a ProteinGym loader: map DMS fitness -> label via the assay's
    functional threshold, keep continuous fitness for construct validity."""
    rng = np.random.default_rng(seed)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for _ in range(n):
        L = 40
        wt = "".join(rng.choice(list(aas), L))
        n_edits = int(rng.integers(1, 5))
        var = list(wt)
        sites = rng.choice(L, n_edits, replace=False)
        for s in sites:
            var[s] = rng.choice(list(aas))
        var = "".join(var)
        active = {int(rng.integers(0, L)): 1.0, int(rng.integers(0, L)): 0.5}
        # synthetic ground-truth fitness: conservative + active-site-preserving -> fit
        r = bps_p(wt, var, functional_weights=active)
        fitness = r.score + rng.normal(0, 0.08)
        label = int(fitness > 0.6)
        rows.append((wt, var, active, label, fitness))
    return rows, None


def main():
    import sys
    # args: validate.py <assay.csv> [structure.pdb]
    if len(sys.argv) > 1:
        pdb = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"Loading real assay: {sys.argv[1]}")
        if pdb:
            print(f"Using AF2 structure: {pdb}")
        data, plddt = load_proteingym(sys.argv[1], pdb_path=pdb)
    else:
        print("No CSV given — using synthetic data. "
              "Run: python3 validate.py <assay.csv> [structure.pdb]")
        data, plddt = load_dataset()
    X, y, fit = [], [], []
    for wt, var, active, label, fitness in data:
        r = bps_p(wt, var, functional_weights=active, plddt_per_residue=plddt)
        X.append([r.components[k] for k in ("grantham", "active_site", "codon", "structural")])
        y.append(label)
        fit.append(fitness)
    X = np.array(X); y = np.array(y); fit = np.array(fit)
    bps_scores = X.mean(axis=1)  # equal-weight baseline composite

    print("=" * 60)
    src = sys.argv[1].split("/")[-1] if len(sys.argv) > 1 else "SYNTHETIC demo"
    print(f"BPS-P VALIDATION REPORT  ({src})")
    print("=" * 60)

    if not _HAVE_SCI:
        print("\n[!] scipy/sklearn not installed — install for full stats:")
        print("    pip install scipy scikit-learn")
        print(f"\nBaseline composite range: {bps_scores.min():.3f}..{bps_scores.max():.3f}")
        return

    # 4.1 construct validity
    rho, p = spearmanr(bps_scores, fit)
    print(f"\n[4.1] Construct validity  Spearman rho = {rho:.3f} (p={p:.1e})")

    # 4.2 internal validity
    names = ["grantham", "active_site", "codon", "structural"]
    corr = np.corrcoef(X.T)
    print("\n[4.2] Component correlation matrix:")
    print("        " + "  ".join(f"{n[:6]:>6s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n[:6]:>6s} " + "  ".join(f"{corr[i,j]:6.2f}" for j in range(4)))
    # VIF
    print("      VIF:")
    for i, n in enumerate(names):
        others = np.delete(X, i, axis=1)
        if np.std(X[:, i]) == 0:
            print(f"    {n:12s} n/a (constant)"); continue
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), others], X[:, i], rcond=None)
        pred = np.c_[np.ones(len(X)), others] @ coef
        ss_res = np.sum((X[:, i] - pred) ** 2)
        ss_tot = np.sum((X[:, i] - X[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        print(f"    {n:12s} {vif:6.2f}")

    # 4.3 predictive validity + learned weights (RQ2 answer)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    coef = {n: float(c) for n, c in zip(names, clf.coef_[0])}
    w = BPSWeights.from_logreg(coef)
    print(f"\n[4.3] Predictive validity  AUC = {auc:.3f}")
    print("      Logistic coefficients (raw):")
    for n in names:
        print(f"    {n:12s} {coef[n]:+.3f}")
    print("      => Learned weights (RQ2 constraint ranking):")
    for n, val in [("grantham", w.grantham), ("active_site", w.active_site),
                   ("codon", w.codon), ("structural", w.structural)]:
        print(f"    {n:12s} {val:.3f}")

    # 4.4 calibration
    brier = brier_score_loss(y, proba)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(proba, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(proba[mask].mean() - y[mask].mean())
    print(f"\n[4.4] Calibration  Brier = {brier:.3f}   ECE = {ece:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
