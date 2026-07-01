"""
run_all.py — robust multi-target, multi-angle evaluation of FVS.

Angles:
  1. Generalization across protein classes (toxin / virus / enzyme).
  2. Test-retest across replicate assays (CCDB x2, F7YBW8 x2).
  3. Component ablation: ESM2 V_fold alone  vs  FVS (V_fold x F_site).
  4. Baselines: ESM2 alone, additive BPS-P, ProteinGym ESM2-650M reference.
  5. Multi-mutant stress (F7YBW8 rows are mostly double/triple mutants).

Loads ESM2 once, scores each assay's WT with a single wt-marginal forward pass.
Long proteins (POLG 2185aa) fall back to a smaller model to fit CPU memory.
"""
from __future__ import annotations
import csv, json, math, sys
import numpy as np
import torch, esm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fvs import load_af2_ca, f_site
from bps_p import bps_p

PG = "/Users/aayushgandhi/Downloads/ProteinGym"
SUB = f"{PG}/DMS_ProteinGym_substitutions"
STR = f"{PG}/ProteinGym_AF2_structures"
SITES = json.load(open("data/active_sites.json"))

# (assay, structure_key, class)
TARGETS = [
    ("CCDB_ECOLI_Adkar_2012",          "CCDB_ECOLI",   "toxin"),
    ("CCDB_ECOLI_Tripathi_2016",       "CCDB_ECOLI",   "toxin (replicate)"),
    ("F7YBW8_MESOW_Aakre_2015",        "F7YBW8_MESOW", "toxin/ParE (multi-mut)"),
    ("F7YBW8_MESOW_Ding_2023",         "F7YBW8_MESOW", "toxin/ParE (multi-mut, rep)"),
    ("AICDA_HUMAN_Gajula_2014_3cycles","AICDA_HUMAN",  "enzyme/deaminase"),
    ("POLG_CXB3N_Mattenberger_2021",   "POLG_CXB3N",   "virus/Coxsackie"),
]

_MODELS = {}
def get_model(name):
    if name not in _MODELS:
        m, a = esm.pretrained.__dict__[name]()
        m.eval()
        _MODELS[name] = (m, a, a.get_batch_converter())
    return _MODELS[name]

def wt_marginal(wt, variants, name):
    model, alphabet, bc = get_model(name)
    _, _, toks = bc([("wt", wt)])
    with torch.no_grad():
        logp = torch.log_softmax(model(toks)["logits"], dim=-1)[0]
    idx = alphabet.tok_to_idx
    out = {}
    for var in variants:
        s = 0.0
        for i, (a, b) in enumerate(zip(wt, var)):
            if a != b:
                s += float(logp[i + 1, idx[b]] - logp[i + 1, idx[a]])
        out[var] = s
    return out

def reconstruct_wt(row):
    wt = list(row["mutated_sequence"])
    for mut in row["mutant"].split(":"):
        wt[int(mut[1:-1]) - 1] = mut[0]
    return "".join(wt)

def active_for(assay):
    key = assay
    for k, v in SITES.items():
        if k != "_README" and k in key:
            return {int(i): float(w) for i, w in v.items()}
    return {}

def v_from_esm(s):
    return 1.0 / (1.0 + math.exp(-s / 3.0))

def evaluate(assay, struct, klass):
    rows = list(csv.DictReader(open(f"{SUB}/{assay}.csv")))
    wt = reconstruct_wt(rows[0])
    name = "esm2_t12_35M_UR50D" if len(wt) > 1200 else "esm2_t30_150M_UR50D"
    variants = [r["mutated_sequence"] for r in rows]
    scores = wt_marginal(wt, variants, name)
    ca, _ = load_af2_ca(f"{STR}/{struct}.pdb")
    active = active_for(assay)

    esm_s, vfold_s, fvs_s, bps_s, fit, lab = [], [], [], [], [], []
    for r in rows:
        var = r["mutated_sequence"]; wtl = reconstruct_wt(r)
        e = scores[var]; vf = v_from_esm(e); fs = f_site(wtl, var, ca, active)
        esm_s.append(e); vfold_s.append(vf); fvs_s.append(vf * fs)
        bps_s.append(bps_p(wtl, var, functional_weights=active).score)
        fit.append(float(r["DMS_score"]))
        lab.append(int(r["DMS_score_bin"]) if r.get("DMS_score_bin") not in (None, "") else -1)
    fit = np.array(fit); lab = np.array(lab)
    has_lab = (lab >= 0).all() and len(set(lab)) == 2
    def auc(x): return roc_auc_score(lab, x) if has_lab else float("nan")
    def rho(x): return spearmanr(x, fit)[0]
    return {
        "assay": assay, "class": klass, "n": len(rows), "model": name.split("_")[1],
        "active": "Y" if active else "-",
        "rho_fvs": rho(fvs_s), "rho_vfold": rho(vfold_s), "rho_esm": rho(esm_s), "rho_bps": rho(bps_s),
        "auc_fvs": auc(fvs_s), "auc_vfold": auc(vfold_s), "auc_esm": auc(esm_s), "auc_bps": auc(bps_s),
    }

def main():
    results = []
    for assay, struct, klass in TARGETS:
        print(f"... {assay}", flush=True)
        try:
            results.append(evaluate(assay, struct, klass))
        except Exception as ex:
            print(f"    FAILED: {ex}")
    print("\n" + "=" * 110)
    print(f"{'assay':<34}{'class':<22}{'n':>6}{'mdl':>5}{'AS':>3}  "
          f"{'AUC_fvs':>8}{'AUC_vf':>7}{'AUC_esm':>8}{'AUC_bps':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['assay'][:33]:<34}{r['class']:<22}{r['n']:>6}{r['model']:>5}{r['active']:>3}  "
              f"{r['auc_fvs']:>8.3f}{r['auc_vfold']:>7.3f}{r['auc_esm']:>8.3f}{r['auc_bps']:>8.3f}")
    print("-" * 110)
    print("Spearman (construct validity):")
    print(f"{'assay':<34}{'':22}{'':6}{'':5}{'':3}  "
          f"{'rho_fvs':>8}{'rho_vf':>7}{'rho_esm':>8}{'rho_bps':>8}")
    for r in results:
        print(f"{r['assay'][:33]:<34}{'':22}{'':6}{'':5}{'':3}  "
              f"{r['rho_fvs']:>8.3f}{r['rho_vfold']:>7.3f}{r['rho_esm']:>8.3f}{r['rho_bps']:>8.3f}")
    print("=" * 110)
    # F_site contribution summary (where active sites in mutated region)
    lift = [(r['assay'], r['auc_fvs'] - r['auc_vfold']) for r in results if r['active'] == 'Y']
    print("F_site lift over V_fold-alone (AUC):")
    for a, d in lift:
        print(f"  {a[:40]:<42}{d:+.3f}")

if __name__ == "__main__":
    main()
