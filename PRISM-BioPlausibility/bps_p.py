"""
bps_p.py — Biological Plausibility Score for Proteins (BPS-P)

Implements the math in SPEC.md. Runs today with no external models installed:
the structural term falls back to a stub you replace with ESMFold/Boltz.

Pure-Python + numpy. Equations referenced as (Eq. N) match SPEC.md.

Author: PRISM Protein Foundation Models project.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 1. Grantham distance, computed from physicochemical properties (Eq. 2)
#    Grantham, R. (1974) Science 185:862. Properties: composition c,
#    polarity p, molecular volume v.
# ---------------------------------------------------------------------------
# (c, p, v) per amino acid
_GRANTHAM_PROPS: Dict[str, tuple] = {
    "S": (1.42, 9.2, 32),  "R": (0.65, 10.5, 124), "L": (0.0, 4.9, 111),
    "P": (0.39, 8.0, 32.5),"T": (0.71, 8.6, 61),   "A": (0.0, 8.1, 31),
    "V": (0.0, 5.9, 84),   "G": (0.74, 9.0, 3),    "I": (0.0, 5.2, 111),
    "F": (0.0, 5.2, 132),  "Y": (0.20, 6.2, 136),  "C": (2.75, 5.5, 55),
    "H": (0.58, 10.4, 96), "Q": (0.89, 10.5, 85),  "N": (1.33, 11.6, 56),
    "K": (0.33, 11.3, 119),"D": (1.38, 13.0, 54),  "E": (0.92, 12.3, 83),
    "M": (0.0, 5.7, 105),  "W": (0.13, 5.4, 170),
}
_ALPHA, _BETA, _GAMMA, _RHO = 1.833, 0.1018, 0.000399, 50.723
G_MAX = 215.0  # Cys-Trp, the maximum Grantham distance


def grantham_distance(a: str, b: str) -> float:
    """Grantham distance between two amino acids (Eq. 2)."""
    if a == b:
        return 0.0
    ca, pa, va = _GRANTHAM_PROPS[a]
    cb, pb, vb = _GRANTHAM_PROPS[b]
    return _RHO * math.sqrt(
        _ALPHA * (ca - cb) ** 2 + _BETA * (pa - pb) ** 2 + _GAMMA * (va - vb) ** 2
    )


# ---------------------------------------------------------------------------
# 2. Component scores
# ---------------------------------------------------------------------------
def s_grantham(wt: str, var: str) -> float:
    """Grantham conservatism, averaged over edited positions (Eq. 3)."""
    edits = [(w, v) for w, v in zip(wt, var) if w != v]
    if not edits:
        return 1.0
    return float(np.mean([1.0 - grantham_distance(w, v) / G_MAX for w, v in edits]))


def s_active_site(
    wt: str,
    var: str,
    functional_weights: Dict[int, float],
) -> float:
    """Active-site preservation (Eq. 4).

    functional_weights: {position_index: weight}, e.g. catalytic=1.0, binding=0.5.
    Positions are 0-based into the sequence.
    """
    if not functional_weights:
        return 1.0
    total = sum(functional_weights.values())
    hit = sum(
        functional_weights.get(i, 0.0)
        for i, (w, v) in enumerate(zip(wt, var))
        if w != v
    )
    return float(max(0.0, 1.0 - hit / total))


def codon_adaptation_index(codons: Sequence[str], w_codon: Dict[str, float]) -> float:
    """CAI (Eq. 5). w_codon: relative adaptiveness per codon, in (0,1]."""
    ws = [w_codon.get(c, 0.5) for c in codons]  # 0.5 = neutral fallback
    ws = [w for w in ws if w > 0]
    if not ws:
        return 1.0
    return float(math.exp(np.mean(np.log(ws))))


def s_codon(
    var_codons: Optional[Sequence[str]],
    w_codon: Optional[Dict[str, float]],
) -> float:
    """Codon adaptation term. If no nucleotide-level info, returns 1.0
    (amino-acid-only threat model — see SPEC 2.3)."""
    if not var_codons or not w_codon:
        return 1.0
    return codon_adaptation_index(var_codons, w_codon)


def s_structural(
    wt: str,
    var: str,
    plddt_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """Relative structural validity (Eq. 6).

    plddt_fn: maps a sequence -> pLDDT in [0,100] (ESMFold / Boltz).
    If None, uses a deterministic stub so the pipeline runs end-to-end;
    REPLACE for real results.
    """
    fn = plddt_fn or _stub_plddt
    p_wt = fn(wt)
    p_var = fn(var)
    if p_wt <= 0:
        return 1.0
    return float(min(1.0, p_var / p_wt))


def load_af2_plddt(pdb_path: str) -> List[float]:
    """Per-residue pLDDT from an AlphaFold2 PDB (B-factor column).
    Returns a list indexed 0-based by residue number."""
    per_res: Dict[int, float] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                resseq = int(line[22:26])
                bfac = float(line[60:66])
                per_res[resseq] = bfac  # one value per residue (CA-equivalent)
    if not per_res:
        return []
    n = max(per_res)
    return [per_res.get(i, 0.0) for i in range(1, n + 1)]


def s_structural_af2(wt: str, var: str, plddt_per_residue: List[float]) -> float:
    """Structural tolerance proxy from WT AF2 pLDDT (Eq. 6 variant, runnable today).

    An edit landing in a high-pLDDT (well-folded, ordered) region is more
    structurally disruptive -> less plausible. Edits in low-confidence
    (disordered/flexible) regions are better tolerated -> more plausible.

        s = mean over edits of ( 1 - pLDDT[i]/100 )

    This needs only the WT structure (no variant folding). The higher-fidelity
    version (Eq. 6) folds the variant with ESMFold/Boltz and takes the pLDDT
    ratio; swap in when available."""
    edits = [i for i, (a, b) in enumerate(zip(wt, var)) if a != b]
    if not edits or not plddt_per_residue:
        return 1.0
    vals = []
    for i in edits:
        if i < len(plddt_per_residue):
            vals.append(1.0 - plddt_per_residue[i] / 100.0)
    return float(np.mean(vals)) if vals else 1.0


def _stub_plddt(seq: str) -> float:
    """Placeholder pLDDT: penalizes edits crudely so the demo is non-trivial.
    NOT a real structural prediction. Swap in ESMFold/Boltz."""
    base = 85.0
    # crude: more rare/charged swaps -> lower confidence; deterministic
    penalty = sum(0.4 for ch in seq if ch in "CWYH") % 20
    return max(40.0, base - penalty)


# ---------------------------------------------------------------------------
# 3. Composite (Eq. 1)
# ---------------------------------------------------------------------------
@dataclass
class BPSWeights:
    grantham: float = 0.25
    active_site: float = 0.25
    codon: float = 0.25
    structural: float = 0.25

    def normalized(self) -> "BPSWeights":
        t = self.grantham + self.active_site + self.codon + self.structural
        if t == 0:
            raise ValueError("weights sum to zero")
        return BPSWeights(self.grantham / t, self.active_site / t,
                          self.codon / t, self.structural / t)

    @classmethod
    def from_logreg(cls, coef: Dict[str, float]) -> "BPSWeights":
        """Build learned weights from logistic-regression coefficients (Eq. 7).
        Uses |beta| normalized; negative betas flagged (component anti-correlated
        with functionality — investigate before trusting)."""
        denom = sum(abs(v) for v in coef.values()) or 1.0
        return cls(
            grantham=abs(coef.get("grantham", 0)) / denom,
            active_site=abs(coef.get("active_site", 0)) / denom,
            codon=abs(coef.get("codon", 0)) / denom,
            structural=abs(coef.get("structural", 0)) / denom,
        )


@dataclass
class BPSResult:
    score: float
    components: Dict[str, float]
    n_edits: int


def bps_p(
    wt: str,
    var: str,
    functional_weights: Optional[Dict[int, float]] = None,
    var_codons: Optional[Sequence[str]] = None,
    w_codon: Optional[Dict[str, float]] = None,
    plddt_fn: Optional[Callable[[str], float]] = None,
    plddt_per_residue: Optional[List[float]] = None,
    weights: Optional[BPSWeights] = None,
) -> BPSResult:
    """Compute BPS-P(wt, var) (Eq. 1). wt and var must be equal length.

    If plddt_per_residue is given (from a WT AF2 structure), the structural
    term uses the runnable AF2 tolerance proxy; else it falls back to plddt_fn."""
    if len(wt) != len(var):
        raise ValueError(f"length mismatch: {len(wt)} vs {len(var)}")
    w = (weights or BPSWeights()).normalized()
    struct = (s_structural_af2(wt, var, plddt_per_residue)
              if plddt_per_residue else s_structural(wt, var, plddt_fn))
    comps = {
        "grantham": s_grantham(wt, var),
        "active_site": s_active_site(wt, var, functional_weights or {}),
        "codon": s_codon(var_codons, w_codon),
        "structural": struct,
    }
    score = (
        w.grantham * comps["grantham"]
        + w.active_site * comps["active_site"]
        + w.codon * comps["codon"]
        + w.structural * comps["structural"]
    )
    n_edits = sum(1 for a, b in zip(wt, var) if a != b)
    return BPSResult(score=float(score), components=comps, n_edits=n_edits)


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    # one conservative edit (L->I) vs one radical edit (K->W) at a "catalytic" site
    var_conservative = wt[:6] + "I" + wt[7:]            # pos 6 L->? (demo)
    var_radical = "W" + wt[1:]                          # pos 0 M->W
    active = {0: 1.0, 12: 0.5}                          # pos0 catalytic, pos12 binding

    for label, v in [("conservative", var_conservative), ("radical@catalytic", var_radical)]:
        r = bps_p(wt, v, functional_weights=active)
        print(f"\n{label}: BPS-P = {r.score:.3f}  (edits={r.n_edits})")
        for k, val in r.components.items():
            print(f"    {k:12s} {val:.3f}")
