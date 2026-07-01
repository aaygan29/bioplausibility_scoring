"""
fvs.py — Functional Viability Score (FVS)

Multiplicative soft-AND metric for "will the protein produced by an altered
genetic code be functional?" See FUNCTIONAL_VIABILITY.md for the math (Eq. N).

Distinct from bps_p.py (additive plausibility). Here function is a CONJUNCTION:
FVS = P_prod * V_fold * F_site * E^eta  (Eq. 1) — any factor ~0 kills the score.

Runs today: F_site (AF2 coords) + V_fold (pluggable ddG). RaSP hook included.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from bps_p import grantham_distance, G_MAX

RT = 0.593  # kcal/mol at 298 K

# physicochemical flags for catalysis-aware severity (Eq. 4 severity)
_CHARGE = {"D": -1, "E": -1, "K": +1, "R": +1, "H": +1}  # approx at phys pH
_HBOND = set("STYNQHKRDEW")  # donors/acceptors (coarse)
_VOL = {k: v[2] for k, v in {
    "S": (0, 0, 32), "R": (0, 0, 124), "L": (0, 0, 111), "P": (0, 0, 32.5),
    "T": (0, 0, 61), "A": (0, 0, 31), "V": (0, 0, 84), "G": (0, 0, 3),
    "I": (0, 0, 111), "F": (0, 0, 132), "Y": (0, 0, 136), "C": (0, 0, 55),
    "H": (0, 0, 96), "Q": (0, 0, 85), "N": (0, 0, 56), "K": (0, 0, 119),
    "D": (0, 0, 54), "E": (0, 0, 83), "M": (0, 0, 105), "W": (0, 0, 170),
}.items()}
_DV_MAX = 167.0


# ---------------------------------------------------------------------------
# AF2 structure parsing: CA coordinates + per-residue pLDDT
# ---------------------------------------------------------------------------
def load_af2_ca(pdb_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Return ({resseq: CA xyz}, {resseq: pLDDT}) from an AF2 PDB (1-based)."""
    ca: Dict[int, np.ndarray] = {}
    plddt: Dict[int, float] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = int(line[22:26])
                ca[res] = np.array([float(line[30:38]), float(line[38:46]),
                                    float(line[46:54])])
                plddt[res] = float(line[60:66])
    return ca, plddt


# ---------------------------------------------------------------------------
# Factor 1: P_prod — production integrity (Eq. before §2)
# ---------------------------------------------------------------------------
def p_prod_missense(active_site: set, var_len: int, wt_len: int) -> float:
    """Missense / synonymous full-length path: protein is made. (=1)."""
    return 1.0


def p_prod_truncation(stop_residue_k: int, wt_len: int, active_site: set,
                      tau: float = 0.95, s: float = 0.02) -> float:
    """Premature stop at residue k (1-based). Hard-fails if any functional
    residue lies past k. Otherwise sigmoid on retained fraction."""
    if any(r > stop_residue_k for r in active_site):
        return 0.0
    f = stop_residue_k / wt_len
    return float(1.0 / (1.0 + math.exp(-(f - tau) / s)))


def p_prod_frameshift(fs_residue_k: int, wt_len: int, active_site: set) -> float:
    """Frameshift at residue k: downstream scrambled. ~0 unless very C-terminal
    AND all functional residues precede k."""
    if any(r >= fs_residue_k for r in active_site):
        return 0.0
    return float(fs_residue_k / wt_len)  # rare C-terminal salvage


# ---------------------------------------------------------------------------
# Factor 2: V_fold — folded fraction from ddG (Eq. 2-3)
# ---------------------------------------------------------------------------
def v_fold(ddg: float, dg_wt: float = 7.0) -> float:
    """Equilibrium folded fraction after destabilization ddg (kcal/mol)."""
    dg_var = dg_wt - ddg
    return float(1.0 / (1.0 + math.exp(-dg_var / RT)))


def v_fold_reference_free(ddg: float, ddg_crit: float = 3.0, s_f: float = 1.5) -> float:
    """Fallback when WT stability unknown: sigmoid centered at a critical ddg."""
    return float(1.0 / (1.0 + math.exp((ddg - ddg_crit) / s_f)))


# ---------------------------------------------------------------------------
# Factor 3: F_site — functional-site integrity (Eq. 4)
# ---------------------------------------------------------------------------
def severity(a: str, b: str) -> float:
    """Catalysis-aware substitution severity in [0,1]."""
    if a == b:
        return 0.0
    g = grantham_distance(a, b) / G_MAX
    charge_flip = 1.0 if _CHARGE.get(a, 0) != _CHARGE.get(b, 0) else 0.0
    hbond_loss = 1.0 if (a in _HBOND and b not in _HBOND) else 0.0
    dvol = abs(_VOL[a] - _VOL[b]) / _DV_MAX
    s = 0.4 * g + 0.25 * charge_flip + 0.20 * hbond_loss + 0.15 * dvol
    return float(min(1.0, s))


def f_site(wt: str, var: str, ca: Dict[int, np.ndarray],
           active_site: Dict[int, float], lam: float = 9.0,
           kappa0: float = 0.05) -> float:
    """Functional-site integrity via spatial influence on active site (Eq. 4).

    active_site: {residue_index_0based: essentiality} (catalytic=1.0, binding=0.5).
    ca: {resseq_1based: xyz}. Returns product over edits of (1 - omega*kappa*s)."""
    edits = [i for i, (a, b) in enumerate(zip(wt, var)) if a != b]
    if not edits or not active_site:
        return 1.0
    site_coords = [ca[r + 1] for r in active_site if (r + 1) in ca]  # 0->1 based
    if not site_coords:
        return 1.0
    prod = 1.0
    for i in edits:
        xyz = ca.get(i + 1)
        if xyz is None:
            continue
        d = min(float(np.linalg.norm(xyz - sc)) for sc in site_coords)
        omega = math.exp(-d / lam)
        kappa = active_site.get(i, kappa0)  # if the edit IS a site residue
        s = severity(wt[i], var[i])
        prod *= (1.0 - omega * kappa * s)
    return float(max(0.0, prod))


# ---------------------------------------------------------------------------
# Combine (Eq. 1)
# ---------------------------------------------------------------------------
@dataclass
class FVSResult:
    score: float
    factors: Dict[str, float]


def fvs(
    wt: str,
    var: str,
    ca: Dict[int, np.ndarray],
    active_site: Dict[int, float],
    ddg: Optional[float] = None,
    dg_wt: float = 7.0,
    cai: float = 1.0,
    eta: float = 0.15,
    edit_kind: str = "missense",
    stop_k: Optional[int] = None,
) -> FVSResult:
    """Functional Viability Score (Eq. 1). edit_kind in
    {synonymous, missense, nonsense, frameshift, startloss}."""
    site_set = set(active_site)
    if edit_kind in ("synonymous", "missense"):
        pp = 1.0
    elif edit_kind == "nonsense":
        pp = p_prod_truncation(stop_k or len(var), len(wt), site_set)
    elif edit_kind == "frameshift":
        pp = p_prod_frameshift(stop_k or 1, len(wt), site_set)
    else:  # startloss / splice
        pp = 0.0

    vf = v_fold(ddg, dg_wt) if ddg is not None else v_fold_reference_free(0.0)
    fs = f_site(wt, var, ca, active_site)
    e = cai ** eta
    score = pp * vf * fs * e
    return FVSResult(score=float(score),
                     factors={"P_prod": pp, "V_fold": vf, "F_site": fs, "E": e})


# ---------------------------------------------------------------------------
# RaSP hook (run separately, then feed ddg per variant)
# ---------------------------------------------------------------------------
def rasp_ddg_stub(wt: str, var: str) -> float:
    """Placeholder. Replace with RaSP output (Blaabjerg et al. eLife 2023):
    run RaSP on the AF2 PDB -> per-position saturation ddG table -> look up.
    Stub returns a crude Grantham-scaled estimate so the pipeline runs."""
    edits = [(a, b) for a, b in zip(wt, var) if a != b]
    return sum(2.5 * grantham_distance(a, b) / G_MAX for a, b in edits)


if __name__ == "__main__":
    # tiny smoke test on a fake structure
    wt = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    var = "MKTAYIAKQRQISFVKSHFSRWLEERLGLIEVQ"  # Q->W near a "site"
    ca = {i + 1: np.array([i * 3.8, 0.0, 0.0]) for i in range(len(wt))}
    active = {20: 1.0}  # catalytic residue at index 20
    ddg = rasp_ddg_stub(wt, var)
    r = fvs(wt, var, ca, active, ddg=ddg)
    print(f"FVS = {r.score:.3f}  (ddG~{ddg:.2f})")
    for k, v in r.factors.items():
        print(f"  {k:8s} {v:.3f}")
