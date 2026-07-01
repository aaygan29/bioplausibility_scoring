"""
compute_esm_scores.py — precompute ESM2 wt-marginal scores per variant for each
assay and cache them, so the empirical comparison can include the language-model
signal. One forward pass per assay (wt-marginal; Meier et al. 2021), summed over
the edited positions for multi-mutants.
"""
import csv, os, sys
from collections import Counter
import numpy as np
import torch, esm

MODEL = "esm2_t30_150M_UR50D"
ASSAYS = [
    "CCDB_ECOLI_Adkar_2012", "CCDB_ECOLI_Tripathi_2016", "BLAT_ECOLX_Deng_2012",
    "GFP_AEQVI_Sarkisyan_2016", "SPG1_STRSG_Olson_2014", "PTEN_HUMAN_Matreyek_2021",
    "UBC9_HUMAN_Weile_2017", "NUD15_HUMAN_Suiter_2020",
    "AICDA_HUMAN_Gajula_2014_3cycles", "POLG_CXB3N_Mattenberger_2021",
]
os.makedirs("data/esm", exist_ok=True)
_model = _alpha = _bc = None


def load_model():
    global _model, _alpha, _bc
    if _model is None:
        _model, _alpha = esm.pretrained.__dict__[MODEL]()
        _model.eval(); _bc = _alpha.get_batch_converter()
    return _model, _alpha, _bc


def wt_marginal(wt, seqs):
    model, alphabet, bc = load_model()
    _, _, toks = bc([("wt", wt)])
    with torch.no_grad():
        logp = torch.log_softmax(model(toks)["logits"], dim=-1)[0]
    idx = alphabet.tok_to_idx
    out = np.zeros(len(seqs))
    for k, s in enumerate(seqs):
        tot = 0.0
        for i in range(min(len(wt), len(s))):
            if wt[i] != s[i]:
                tot += float(logp[i + 1, idx[s[i]]] - logp[i + 1, idx[wt[i]]])
        out[k] = tot
    return out


def main():
    for a in ASSAYS:
        tsv = f"data/proteingym/{a}.tsv"
        out = f"data/esm/{a}.npy"
        if not os.path.exists(tsv) or os.path.exists(out):
            print(f"skip {a}"); continue
        rows = list(csv.DictReader(open(tsv), delimiter="\t"))
        seqs = [r["sequences"] for r in rows]
        L = min(len(s) for s in seqs)
        wt = "".join(Counter(s[i] for s in seqs).most_common(1)[0][0] for i in range(L))
        print(f"{a}: n={len(seqs)} L={L} ...", flush=True)
        np.save(out, wt_marginal(wt, seqs))
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
