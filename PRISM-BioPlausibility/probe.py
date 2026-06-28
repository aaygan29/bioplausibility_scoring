"""Targeted robustness probes: POLG with larger model; CcdB model-size sensitivity."""
import csv, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from run_all import wt_marginal, reconstruct_wt, SUB

def probe(assay, model):
    rows = list(csv.DictReader(open(f"{SUB}/{assay}.csv")))
    wt = reconstruct_wt(rows[0])
    sc = wt_marginal(wt, [r["mutated_sequence"] for r in rows], model)
    esm = np.array([sc[r["mutated_sequence"]] for r in rows])
    fit = np.array([float(r["DMS_score"]) for r in rows])
    lab = np.array([int(r["DMS_score_bin"]) for r in rows])
    print(f"{assay[:30]:<32} {model.split('_')[1]:>5}  "
          f"rho={spearmanr(esm,fit)[0]:+.3f}  AUC={roc_auc_score(lab,esm):.3f}")

print("=== POLG (virus) model-capacity test ===")
probe("POLG_CXB3N_Mattenberger_2021", "esm2_t12_35M_UR50D")
probe("POLG_CXB3N_Mattenberger_2021", "esm2_t30_150M_UR50D")
print("=== CcdB model-size sensitivity ===")
probe("CCDB_ECOLI_Adkar_2012", "esm2_t12_35M_UR50D")
probe("CCDB_ECOLI_Adkar_2012", "esm2_t30_150M_UR50D")
