"""
constraints_extra.py — additional biologically-grounded constraints and a
structural/functional analysis tool, to enrich the bioplausibility metric.

Each constraint is a function of (wild-type, variant, [structure]) returning a
score in [0,1] where HIGHER = more plausible / more tolerable. They invert to
Proto's 0=best convention the same way the core terms do (proto = 1 - score).

Grounded in the variant-effect literature:
- Solvent accessibility (burial) is the single most predictive structural feature;
  buried mutations are far more deleterious (Frazer/CG bioRxiv 2025; Pucci 2018).
- Evolutionary substitution tolerance: BLOSUM62 complements Grantham's
  physicochemistry (Henikoff and Henikoff, 1992).
- Hydrophobicity: Kyte and Doolittle (1982).
- Secondary-structure propensity: Chou and Fasman (1978).
- Aggregation propensity of destabilizing edits: TANGO (Fernandez-Escamilla 2004).
- Epistasis / coevolution (hook): EVmutation / Potts (Hopf et al., 2017).
- Position-specific conservation (hook): MSA frequency (ConSurf-style).
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np

from bps_p import grantham_distance, G_MAX
from fvs import severity

# ---- physicochemical scales ----
# Kyte-Doolittle hydropathy (1982): + hydrophobic, - hydrophilic
_KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
       "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
       "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5}
# Chou-Fasman (1978) helix (Pa) and sheet (Pb) propensities
_PA = {"E": 1.51, "M": 1.45, "A": 1.42, "L": 1.21, "K": 1.16, "F": 1.13, "Q": 1.11,
       "W": 1.08, "I": 1.08, "V": 1.06, "D": 1.01, "H": 1.00, "R": 0.98, "T": 0.83,
       "S": 0.77, "C": 0.70, "Y": 0.69, "N": 0.67, "P": 0.57, "G": 0.57}
_PB = {"V": 1.70, "I": 1.60, "Y": 1.47, "C": 1.19, "W": 1.37, "F": 1.38, "L": 1.30,
       "T": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89, "H": 0.87, "A": 0.83,
       "S": 0.75, "G": 0.75, "K": 0.74, "P": 0.55, "D": 0.54, "E": 0.37}

try:
    from Bio.Align import substitution_matrices
    _BL = substitution_matrices.load("BLOSUM62")
    _BL_MIN, _BL_MAX = -4.0, 11.0
    _HAVE_BLOSUM = True
except Exception:
    _HAVE_BLOSUM = False


def _edits(wt: str, var: str):
    return [(i, a, b) for i, (a, b) in enumerate(zip(wt, var)) if a != b]


# ---------------------------------------------------------------------------
# Burial / relative solvent accessibility (the top structural feature)
# ---------------------------------------------------------------------------
def contact_numbers(ca: Dict[int, np.ndarray], radius: float = 10.0) -> Dict[int, int]:
    """CA contact number per residue (1-based). High = buried core, low = surface.
    A cheap, DSSP-free proxy for (inverse) solvent accessibility."""
    keys = sorted(ca)
    coords = np.array([ca[k] for k in keys])
    out = {}
    for idx, k in enumerate(keys):
        d = np.linalg.norm(coords - coords[idx], axis=1)
        out[k] = int(np.sum(d < radius) - 1)  # exclude self
    return out


def burial_fraction(ca: Dict[int, np.ndarray], radius: float = 10.0,
                    n_cap: int = 24) -> Dict[int, float]:
    """Per-residue burial in [0,1]: 1 = fully buried, 0 = fully exposed."""
    cn = contact_numbers(ca, radius)
    return {k: min(1.0, v / n_cap) for k, v in cn.items()}


def s_burial(wt: str, var: str, ca: Dict[int, np.ndarray]) -> float:
    """Penalize disruptive edits in the buried core (severity scaled by burial).
    Buried + radical -> low; surface or gentle -> high. (Pucci 2018; CG 2025)."""
    e = _edits(wt, var)
    if not e or not ca:
        return 1.0
    bur = burial_fraction(ca)
    vals = []
    for i, a, b in e:
        b_i = bur.get(i + 1, 0.0)               # 1-based structure index
        vals.append(1.0 - b_i * severity(a, b)) # only buried AND severe is penalized
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# BLOSUM62 evolutionary substitution tolerance
# ---------------------------------------------------------------------------
def s_blosum(wt: str, var: str) -> float:
    """Average BLOSUM62 of the edits, mapped to [0,1] (higher = more tolerated).
    Complements Grantham: BLOSUM is evolutionary frequency, Grantham is chemistry."""
    e = _edits(wt, var)
    if not e:
        return 1.0
    if not _HAVE_BLOSUM:
        return 1.0
    vals = []
    for _, a, b in e:
        try:
            s = float(_BL[(a, b)])
        except Exception:
            s = 0.0
        vals.append((s - _BL_MIN) / (_BL_MAX - _BL_MIN))
    return float(np.clip(np.mean(vals), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hydropathy: burying a polar residue or exposing the core is disruptive
# ---------------------------------------------------------------------------
def s_hydropathy(wt: str, var: str, ca: Optional[Dict[int, np.ndarray]] = None) -> float:
    """Penalize large hydrophobicity swaps, weighted by burial when structure given.
    A buried hydrophobic -> polar swap is the classic destabilizer (Kyte-Doolittle)."""
    e = _edits(wt, var)
    if not e:
        return 1.0
    bur = burial_fraction(ca) if ca else {}
    vals = []
    for i, a, b in e:
        d = abs(_KD[a] - _KD[b]) / 9.0          # max KD span ~9
        w = bur.get(i + 1, 0.5) if ca else 0.5  # burial weights the penalty
        vals.append(1.0 - w * min(1.0, d))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Secondary-structure compatibility (helix/sheet breakers, Pro/Gly)
# ---------------------------------------------------------------------------
def s_secondary_structure(wt: str, var: str) -> float:
    """Penalize edits that sharply lower local helix/sheet propensity or insert a
    backbone-breaking Proline / over-flexible Glycine (Chou and Fasman, 1978)."""
    e = _edits(wt, var)
    if not e:
        return 1.0
    vals = []
    for _, a, b in e:
        best_a = max(_PA[a], _PB[a]); best_b = max(_PA[b], _PB[b])
        drop = max(0.0, best_a - best_b) / 1.7  # loss of structure propensity
        pen = drop
        if b == "P" and a != "P":               # Pro breaks helix/sheet
            pen = max(pen, 0.6)
        if b == "G" and a != "G":               # Gly adds flexibility
            pen = max(pen, 0.3)
        vals.append(1.0 - min(1.0, pen))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Aggregation propensity (destabilizing edits can nucleate beta-aggregation)
# ---------------------------------------------------------------------------
def s_aggregation(wt: str, var: str, window: int = 5) -> float:
    """Penalize edits that raise local hydrophobic beta-sheet propensity, a proxy
    for aggregation nucleation (TANGO; Fernandez-Escamilla et al., 2004)."""
    e = _edits(wt, var)
    if not e:
        return 1.0
    def agg(seq, c):
        lo, hi = max(0, c - window // 2), min(len(seq), c + window // 2 + 1)
        seg = seq[lo:hi]
        h = np.mean([max(0.0, _KD[x]) for x in seg]) / 4.5
        bsheet = np.mean([_PB[x] for x in seg]) / 1.7
        return h * bsheet
    vals = []
    for i, a, b in e:
        inc = max(0.0, agg(var, i) - agg(wt, i))  # increase in aggregation risk
        vals.append(1.0 - min(1.0, 3.0 * inc))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Hooks: position-specific conservation (MSA) and epistasis (Potts/EVmutation)
# ---------------------------------------------------------------------------
def load_msa_frequencies(fasta_path: str) -> List[Dict[str, float]]:
    """Per-column amino-acid frequency from an aligned FASTA/A2M (ConSurf-style)."""
    seqs, cur = [], []
    for line in open(fasta_path):
        line = line.rstrip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        return []
    L = len(seqs[0]); freqs = []
    for c in range(L):
        col = [s[c] for s in seqs if c < len(s) and s[c].isalpha()]
        n = len(col) or 1
        freqs.append({aa: col.count(aa) / n for aa in set(col)})
    return freqs


def s_conservation(wt: str, var: str,
                   msa_freqs: Optional[List[Dict[str, float]]]) -> float:
    """How often nature uses the mutant residue at that column. Higher = more
    plausible. Inert (1.0) without an MSA. (Hopf 2017; ConSurf.)"""
    e = _edits(wt, var)
    if not e or not msa_freqs:
        return 1.0
    vals = [msa_freqs[i].get(b, 0.0) if i < len(msa_freqs) else 0.5 for i, a, b in e]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Structural / functional analysis tool: explain WHY an edit is (in)tolerable
# ---------------------------------------------------------------------------
def analyze_edits(wt: str, var: str, ca: Optional[Dict[int, np.ndarray]] = None,
                  active_site: Optional[Dict[int, float]] = None) -> List[Dict]:
    """Per-edit diagnostic combining sequence and structure features."""
    active_site = active_site or {}
    bur = burial_fraction(ca) if ca else {}
    site_coords = [ca[r + 1] for r in active_site if ca and (r + 1) in ca] if ca else []
    rows = []
    for i, a, b in _edits(wt, var):
        d_site = None
        if site_coords and (i + 1) in ca:
            d_site = float(min(np.linalg.norm(ca[i + 1] - sc) for sc in site_coords))
        rows.append({
            "pos": i + 1, "wt": a, "mut": b,
            "grantham": round(grantham_distance(a, b), 1),
            "blosum62": (float(_BL[(a, b)]) if _HAVE_BLOSUM else None),
            "burial": round(bur.get(i + 1, 0.0), 2),
            "dKD_hydropathy": round(_KD[b] - _KD[a], 1),
            "helix_prop_change": round(_PA[b] - _PA[a], 2),
            "sheet_prop_change": round(_PB[b] - _PB[a], 2),
            "is_active_site": (i in active_site),
            "dist_to_active_site_A": (round(d_site, 1) if d_site is not None else None),
            "introduces_proline": (b == "P" and a != "P"),
            "breaks_disulfide": (a == "C" and b != "C"),
        })
    return rows


# all extra constraints, for convenient batch scoring
EXTRA_CONSTRAINTS = {
    "burial": lambda wt, var, ctx: s_burial(wt, var, ctx.get("ca")),
    "blosum62": lambda wt, var, ctx: s_blosum(wt, var),
    "hydropathy": lambda wt, var, ctx: s_hydropathy(wt, var, ctx.get("ca")),
    "secondary_structure": lambda wt, var, ctx: s_secondary_structure(wt, var),
    "aggregation": lambda wt, var, ctx: s_aggregation(wt, var),
    "conservation": lambda wt, var, ctx: s_conservation(wt, var, ctx.get("msa_freqs")),
}


if __name__ == "__main__":
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    ca = {i + 1: np.array([i * 3.8, 0.0, 0.0]) for i in range(len(wt))}
    cases = {
        "gentle surface (I5V)": wt[:4] + "V" + wt[5:],
        "buried polar swap (L?->D at 10)": wt[:10] + "D" + wt[11:],
        "proline in helix (A4P)": wt[:4] + "P" + wt[5:],
    }
    for name, var in cases.items():
        ctx = {"ca": ca, "msa_freqs": None}
        scores = {k: round(f(wt, var, ctx), 3) for k, f in EXTRA_CONSTRAINTS.items()}
        print(f"{name:34s} {scores}")
    print("\nper-edit analysis of the buried polar swap:")
    for r in analyze_edits(wt, cases["buried polar swap (L?->D at 10)"], ca):
        print(" ", r)
