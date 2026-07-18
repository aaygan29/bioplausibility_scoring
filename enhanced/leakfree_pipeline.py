"""
leakfree_pipeline.py — rebuild the pooled feature matrix with strictly
POINTWISE features and all normalization fit TRAIN-ONLY, so the reported
leave-one-protein-out (LOPO) numbers reflect true deployment (score one produced
molecule at a time, with no access to a test-batch distribution).

Two things are fixed relative to the earlier build:
  1. ESM2 wt-marginal is kept RAW (the per-variant log-likelihood-ratio, in nats)
     instead of rank-normalized within each assay. Rank-within-assay is
     transductive: it assumes you see the whole variant population at once. At
     deployment you see one molecule, so the feature must be pointwise.
  2. Any feature scaling used by the linear combiner / calibrator is fit on the
     TRAINING proteins only and applied to the held-out protein.

The biophysical signals (grantham, blosum62, hydropathy, sec_struct, aggregation,
burial) are already pointwise functions of (wt, variant) and are reused verbatim
from the project modules. plddt_region is intentionally excluded (position-only
leakage vector, established earlier).

Output: FEAT_COLS matrix + metadata columns, cached to /tmp/pooled_leakfree.parquet.
"""
from __future__ import annotations
import csv, os
from collections import Counter
import numpy as np
import pandas as pd

from bps_p import grantham_distance, G_MAX, load_af2_plddt
from fvs import load_af2_ca, severity
from constraints_extra import _KD, _PA, _PB, s_aggregation
from multi_assay_analysis import PANEL, wcn_burial, load_assay
try:
    from Bio.Align import substitution_matrices
    _BL = substitution_matrices.load("BLOSUM62"); _HAVE_BL = True
except Exception:
    _HAVE_BL = False

# pointwise features only (no plddt_region; esm kept RAW)
FEAT_COLS = ["esm_raw", "grantham", "blosum62", "hydropathy",
             "sec_struct", "aggregation", "burial"]


def compute_pointwise(seqs, wt, ca, plddt, esm_raw):
    """Per-variant features. esm_raw is the RAW wt-marginal log-ratio (nats)."""
    bur = wcn_burial(ca) if ca else {}
    has_struct = bool(ca) and len(plddt) == len(wt)
    rows = []
    for k, s in enumerate(seqs):
        edits = [(i, wt[i], s[i]) for i in range(min(len(wt), len(s))) if wt[i] != s[i]]
        e_raw = float(esm_raw[k]) if esm_raw is not None else 0.0
        if not edits:
            rows.append([e_raw, 1, 1, 1, 1, 1, 1]); continue
        gr = bl = hy = ss = bu = 0.0
        for i, a, b in edits:
            b_i = bur.get(i + 1, 0.0) if has_struct else 0.0
            sev = severity(a, b)
            gr += 1 - grantham_distance(a, b) / G_MAX
            bl += ((float(_BL[(a, b)]) + 4) / 15) if _HAVE_BL else 1.0
            hy += 1 - (b_i if has_struct else 0.5) * min(1.0, abs(_KD[a] - _KD[b]) / 9.0)
            ss += float(np.clip(1 - max(0.0, max(_PA[a], _PB[a]) - max(_PA[b], _PB[b])) / 1.7
                                - (0.6 if (b == "P" and a != "P") else 0.0), 0, 1))
            bu += (1 - b_i * sev) if has_struct else 1.0
        n = len(edits)
        rows.append([e_raw, gr / n, np.clip(bl / n, 0, 1), hy / n, ss / n,
                     s_aggregation(wt, s), bu / n])
    return np.array(rows, float), has_struct


def build(datadir="data", out="/tmp/pooled_leakfree.parquet"):
    frames = []
    for assay, (entry, klass) in PANEL.items():
        tsv = f"{datadir}/proteingym/{assay}.tsv"
        if not os.path.exists(tsv):
            continue
        seqs, y, fold, wt = load_assay(tsv)
        pdb = f"{datadir}/structures/{entry}.pdb"
        ca, plddt = ({}, [])
        if os.path.exists(pdb):
            ca, _ = load_af2_ca(pdb); plddt = load_af2_plddt(pdb)
        esm_path = f"{datadir}/esm/{assay}.npy"
        esm_raw = np.load(esm_path) if os.path.exists(esm_path) else None
        X, has_struct = compute_pointwise(seqs, wt, ca, plddt, esm_raw)
        nmut = np.array([sum(1 for i in range(min(len(wt), len(s))) if wt[i] != s[i])
                         for s in seqs])
        df = pd.DataFrame(X, columns=FEAT_COLS)
        df.insert(0, "assay", assay)
        df.insert(1, "protein", entry)
        df.insert(2, "klass", klass)
        df.insert(3, "has_struct", has_struct)
        df.insert(4, "nmut", nmut)
        df["y"] = y
        yb = (y > np.median(y)).astype(int)
        if len(set(yb)) < 2:
            yb = (y > y.mean()).astype(int)
        df["ybin"] = yb
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_parquet(out)
    return pooled


if __name__ == "__main__":
    p = build()
    print("built", p.shape, "->", "/tmp/pooled_leakfree.parquet")
