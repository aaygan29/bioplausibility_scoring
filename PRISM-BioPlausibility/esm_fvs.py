"""
esm_fvs.py — real, local variant-effect signal for the V_fold term via ESM2.

Field-standard zero-shot VEP: ESM2 wt-marginal log-likelihood ratio per variant
(Meier et al. 2021). One forward pass over the WT sequence; per-variant score =
sum_i [ log p(mut_i) - log p(wt_i) ]. Higher = more tolerated.

We use ESM2-150M, deliberately DIFFERENT from the ESM2-650M classifier under
attack, to avoid circularity. This is an interim substitute for RaSP ddG in the
V_fold slot (documented as such), not Rosetta ddG.

Outputs FVS (with ESM V_fold + real F_site + P_prod) vs BPS-P vs ESM-alone,
validated against ProteinGym DMS labels.

Usage:
  python3 esm_fvs.py <assay.csv> <structure.pdb> [active_sites.json] [model]
"""
from __future__ import annotations
import csv, json, sys, math
import numpy as np
import torch
import esm

from fvs import load_af2_ca, f_site, p_prod_missense
from bps_p import bps_p

try:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    _SCI = True
except Exception:
    _SCI = False


def esm_wt_marginal_scores(wt: str, variants, model_name="esm2_t30_150M_UR50D"):
    """Return {variant_seq: score} via ESM2 wt-marginal (one forward pass)."""
    print(f"loading {model_name} (first run downloads weights)...")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([("wt", wt)])
    with torch.no_grad():
        logits = model(toks)["logits"]  # [1, L+2, V]
    logp = torch.log_softmax(logits, dim=-1)[0]  # [L+2, V]
    idx = alphabet.tok_to_idx

    scores = {}
    for var in variants:
        s = 0.0
        for i, (a, b) in enumerate(zip(wt, var)):
            if a == b:
                continue
            pos = i + 1  # BOS offset
            s += float(logp[pos, idx[b]] - logp[pos, idx[a]])
        scores[var] = s
    return scores


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    assay_csv, pdb = sys.argv[1], sys.argv[2]
    sites_json = sys.argv[3] if len(sys.argv) > 3 else None
    model_name = sys.argv[4] if len(sys.argv) > 4 else "esm2_t30_150M_UR50D"

    rows = list(csv.DictReader(open(assay_csv)))
    # reconstruct WT once
    var0 = rows[0]["mutated_sequence"]
    wt = list(var0)
    for mut in rows[0]["mutant"].split(":"):
        wt[int(mut[1:-1]) - 1] = mut[0]
    wt = "".join(wt)

    variants = [r["mutated_sequence"] for r in rows]
    esm_scores = esm_wt_marginal_scores(wt, variants, model_name)

    ca, _ = load_af2_ca(pdb)
    active = {}
    if sites_json:
        allsites = json.load(open(sites_json))
        key = assay_csv.split("/")[-1].replace(".csv", "")
        for k, v in allsites.items():
            if k != "_README" and k in key:
                active = {int(i): float(w) for i, w in v.items()}; break

    # map ESM score -> V_fold-like [0,1] via sigmoid (more tolerated -> higher)
    def v_from_esm(s):  # scale ~3 log-units
        return 1.0 / (1.0 + math.exp(-s / 3.0))

    fvs_s, esm_s, bps_s, fit, lab = [], [], [], [], []
    for r in rows:
        var = r["mutated_sequence"]
        wt_l = list(var)
        for mut in r["mutant"].split(":"):
            wt_l[int(mut[1:-1]) - 1] = mut[0]
        wt_l = "".join(wt_l)
        e = esm_scores[var]
        vfold = v_from_esm(e)
        fs = f_site(wt_l, var, ca, active)
        fvs_s.append(1.0 * vfold * fs)  # P_prod=1 (missense), E omitted (AA-level)
        esm_s.append(e)
        bps_s.append(bps_p(wt_l, var, functional_weights=active).score)
        fit.append(float(r["DMS_score"]))
        lab.append(int(r["DMS_score_bin"]) if r.get("DMS_score_bin") not in (None, "") else None)

    fvs_s, esm_s, bps_s = map(np.array, (fvs_s, esm_s, bps_s))
    fit = np.array(fit); lab = np.array(lab) if None not in lab else None

    print("=" * 64)
    print(f"FVS (ESM2 V_fold + F_site) on {assay_csv.split('/')[-1]}")
    print(f"model={model_name}  active_site={'yes' if active else 'EMPTY'}")
    print("=" * 64)
    if not _SCI:
        print("install scipy scikit-learn"); return
    print("  Construct validity (Spearman vs DMS_score):")
    print(f"    FVS        {spearmanr(fvs_s, fit)[0]:+.3f}")
    print(f"    ESM2 alone {spearmanr(esm_s, fit)[0]:+.3f}")
    print(f"    BPS-P      {spearmanr(bps_s, fit)[0]:+.3f}")
    if lab is not None and len(set(lab)) == 2:
        print("  Predictive (AUC vs DMS_score_bin):")
        print(f"    FVS        {roc_auc_score(lab, fvs_s):.3f}")
        print(f"    ESM2 alone {roc_auc_score(lab, esm_s):.3f}")
        print(f"    BPS-P      {roc_auc_score(lab, bps_s):.3f}")
    print(f"  reference: ESM2-650M zero-shot AUC (ProteinGym) = 0.761")
    print("=" * 64)


if __name__ == "__main__":
    main()
