"""
proto_panel.py — evaluate ONE produced biomolecule against a PANEL of constraints.

This is the shape the experiment needs: a protein/genomic foundation model PRODUCES a
biomolecule, and we ask "will it actually be functional / structurally plausible?" —
not from one number but from several independent, biologically-grounded constraints,
each in Proto form (0.0 = best/plausible, 1.0 = worst/implausible). Proto's value is
exactly this: score one construct from many angles, then let an optimizer or a risk
gate combine them.

Each constraint is INDEPENDENT and reports its own applicability, so a de novo produced
protein (no wild-type) simply runs the reference-free constraints and skips the
reference-based ones. Nothing silently fails.

Constraints in the default panel
--------------------------------
  calibrated_plausibility   (referenced)     enhanced metric -> 1 - P(tolerated). LOPO-validated.
  functional_viability      (referenced)     multiplicative FVS: produced AND folds AND site intact.
  self_structure_confidence (reference-free) 1 - mean(pLDDT) of the PRODUCED structure.
  aggregation               (reference-free) 1 - aggregation-resistance of the sequence.
  naturalness_esm           (reference-free) squashed ESM2 pseudo-likelihood of the sequence.

Every entry returns (score in [0,1], metadata). `evaluate()` returns the full panel plus
a weighted aggregate you can feed to a Proto Optimizer or threshold in a risk gate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def _seq_str(seq) -> str:
    return getattr(seq, "sequence", seq)


@dataclass
class Constraint:
    """One Proto constraint: name, weight, and a fn(molecule, ctx) -> (score, meta)."""
    name: str
    fn: Callable
    weight: float = 1.0
    regime: str = "reference-free"   # or "referenced"


@dataclass
class ProtoPanel:
    """A weighted panel of constraints scoring one produced molecule."""
    constraints: List[Constraint] = field(default_factory=list)

    def evaluate(self, molecule, ctx: Optional[Dict] = None) -> Dict:
        ctx = ctx or {}
        seq = _seq_str(molecule)
        rows, total_w, total_s = [], 0.0, 0.0
        for c in self.constraints:
            try:
                s, meta = c.fn(seq, ctx)
                applicable = meta.get("applicable", True)
            except Exception as e:                      # missing data -> skip, don't crash
                s, meta, applicable = None, {"error": str(e)}, False
            rows.append(dict(name=c.name, regime=c.regime, weight=c.weight,
                             score=s, applicable=applicable, meta=meta))
            if applicable and s is not None:
                total_w += c.weight; total_s += c.weight * s
        aggregate = total_s / total_w if total_w else None
        return {"per_constraint": rows, "aggregate": aggregate,
                "n_applicable": int(sum(r["applicable"] and r["score"] is not None
                                        for r in rows))}


# ---------------------------------------------------------------------------
# Constraint builders (each returns a Constraint you can add to a panel)
# ---------------------------------------------------------------------------
def calibrated_plausibility_constraint(scorer, weight=1.0) -> Constraint:
    """Referenced: enhanced, calibrated metric -> 1 - P(tolerated)."""
    def fn(seq, ctx):
        r = scorer.score(seq)
        return 1.0 - r.p_tolerated, {"p_tolerated": r.p_tolerated,
                                     "applicable": r.applicable}
    return Constraint("calibrated_plausibility", fn, weight, "referenced")


def functional_viability_constraint(fvs_scorer, weight=1.0) -> Constraint:
    """Referenced: multiplicative FVS (produced AND folds AND active-site intact)."""
    def fn(seq, ctx):
        s = fvs_scorer.functionality(seq)
        return 1.0 - s, {"fvs": s, "applicable": True}
    return Constraint("functional_viability", fn, weight, "referenced")


def self_structure_confidence_constraint(weight=1.0) -> Constraint:
    """Reference-free: 1 - mean pLDDT of the PRODUCED structure.

    ctx must supply 'produced_plddt' (list/array of per-residue pLDDT for the produced
    fold, e.g. from ESMFold/AF2 on the produced sequence). This is variant-DEPENDENT
    structural evidence — the honest replacement for the removed position-only term."""
    def fn(seq, ctx):
        p = ctx.get("produced_plddt")
        if p is None:
            return None, {"applicable": False, "why": "no produced structure supplied"}
        m = float(np.mean(p)) / (100.0 if np.mean(p) > 1.5 else 1.0)
        return 1.0 - m, {"mean_plddt": m, "applicable": True}
    return Constraint("self_structure_confidence", fn, weight, "reference-free")


def aggregation_constraint(weight=0.5) -> Constraint:
    """Reference-free: penalize aggregation-prone produced sequences (Kyte-Doolittle
    hydrophobic-run heuristic; swap in TANGO/AGGRESCAN for production)."""
    _KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
           "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
           "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
    def fn(seq, ctx):
        h = np.array([_KD.get(a, 0.0) for a in seq])
        w = np.convolve(np.clip(h, 0, None), np.ones(6)/6, mode="valid")
        agg = float(np.mean(w > 1.5)) if len(w) else 0.0   # fraction hydrophobic windows
        return agg, {"hydrophobic_window_frac": agg, "applicable": True}
    return Constraint("aggregation", fn, weight, "reference-free")


def naturalness_esm_constraint(esm_ll_fn, weight=1.0) -> Constraint:
    """Reference-free: squashed ESM2 pseudo-log-likelihood of the produced sequence.
    esm_ll_fn(seq) -> mean per-residue log-likelihood (higher = more natural)."""
    def fn(seq, ctx):
        ll = esm_ll_fn(seq)
        s = 1.0 / (1.0 + np.exp(2.5 * (ll + 3.0)))   # squash: natural(ll~-2)->low
        return float(s), {"mean_ll": float(ll), "applicable": True}
    return Constraint("naturalness_esm", fn, weight, "reference-free")
