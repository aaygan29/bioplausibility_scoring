"""
ddg.py — load RaSP ddG predictions and serve per-variant lookups for V_fold.

RaSP (Blaabjerg et al., eLife 2023) outputs a saturation table per structure:
every position x every amino acid -> predicted ddG (kcal/mol). We index it so
fvs.v_fold() can look up the ddG for any single-mutant (and sum for multi).

RaSP output columns vary slightly by version; this reader is tolerant and looks
for: a position column, a wild-type aa, a variant aa, and a 'score_ml'/'ddG' col.
"""
from __future__ import annotations
import csv
from typing import Dict, Optional


def load_rasp_table(csv_path: str) -> Dict[tuple, float]:
    """Return {(pos_1based, wt_aa, var_aa): ddG}. Tolerant to column naming."""
    table: Dict[tuple, float] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in reader.fieldnames or []}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_pos = pick("pos", "position", "variant_pos", "resid")
        c_wt = pick("wildtype", "wt", "wt_aa", "aa_ref")
        c_var = pick("mutant", "variant", "mt", "aa_alt", "mut_aa")
        c_ddg = pick("score_ml", "ddg", "ddg_ml", "rasp_ddg", "score")
        if not all([c_pos, c_wt, c_var, c_ddg]):
            raise ValueError(f"could not map RaSP columns from {reader.fieldnames}")

        for row in reader:
            try:
                pos = int(float(row[c_pos]))
                key = (pos, row[c_wt].strip().upper(), row[c_var].strip().upper())
                table[key] = float(row[c_ddg])
            except (ValueError, KeyError):
                continue
    return table


def ddg_for_variant(wt: str, var: str, table: Dict[tuple, float]) -> Optional[float]:
    """Sum RaSP ddG over edited positions (1-based). None if any edit missing."""
    total = 0.0
    found = False
    for i, (a, b) in enumerate(zip(wt, var)):
        if a == b:
            continue
        key = (i + 1, a, b)
        if key not in table:
            return None
        total += table[key]
        found = True
    return total if found else 0.0
