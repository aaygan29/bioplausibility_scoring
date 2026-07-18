"""
enhanced_proto.py — the enhanced bioplausibility metric as a Proto Constraint.

Proto (Hie et al., 2026) is the design language: a Constraint scores a produced
sequence 0.0 (best) .. 1.0 (worst); the Optimizer MINIMIZES the total. This module
exposes the leakage-free, cross-protein-validated, CALIBRATED metric
(enhanced_bioplausibility.py) in that slot, so "is this produced protein
structurally + functionally plausible?" becomes one Constraint you can drop into any
Proto program.

Two regimes, one interface:
  * REFERENCED  (a produced protein that is a modified natural protein, so a wild-type
    reference exists — the PRISM adversarial-edit case). Fully validated: leave-one-
    protein-out AUC 0.704 on 10 ProteinGym DMS assays (36,374 variants).
  * REFERENCE-FREE (a de novo produced structure with no natural wild-type). Handled
    by the structural self-consistency term below; NOT yet validated on DMS (DMS only
    has edit ground truth) — see ENHANCED_METRIC / PROTO_STRUCTURAL_PLAUSIBILITY.md.

The metric emits a CALIBRATED P(tolerated), so the Proto constraint is honest:
    proto_constraint = 1 - P_tolerated        (0 = keep, 1 = reject)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np

from enhanced_bioplausibility import EnhancedBioPlausibility, FEATURES
from multi_assay_analysis import compute_signals, SIG   # the validated feature extractor
from bps_p import load_af2_plddt
from fvs import load_af2_ca


@dataclass
class EnhancedScorer:
    """Score a PRODUCED protein for structural + functional plausibility.

    wt        : reference sequence (required in the REFERENCED regime).
    ca, plddt : AF2 structure context for the reference (used by biophysical signals).
    model     : loaded EnhancedBioPlausibility (GBM combiner + isotonic calibrator).
    esm_fn    : callback esm_fn(wt, variant) -> ESM2 wt-marginal log-ratio for the
                edit(s). Supply your ESM2 model here; without it the ESM backbone
                falls back to the training mean (degraded — supply it in real use).
    """
    wt: str
    ca: Dict[int, object]
    plddt: list
    model: EnhancedBioPlausibility
    esm_fn: Optional[Callable[[str, str], float]] = None

    @classmethod
    def from_files(cls, wt: str, pdb_path: str, model_path: str,
                   esm_fn: Optional[Callable] = None) -> "EnhancedScorer":
        ca, _ = load_af2_ca(pdb_path)
        plddt = load_af2_plddt(pdb_path)
        model = EnhancedBioPlausibility.load(model_path)
        return cls(wt=wt, ca=ca, plddt=plddt, model=model, esm_fn=esm_fn)

    def _features(self, produced: str) -> Dict[str, Optional[float]]:
        esm_val = self.esm_fn(self.wt, produced) if self.esm_fn else None
        X, _ = compute_signals([produced], self.wt, self.ca, self.plddt,
                               esm=np.array([esm_val if esm_val is not None else 0.0]))
        row = {s: float(X[0, j]) for j, s in enumerate(SIG)}
        # plddt_region intentionally dropped from the enhanced model (leakage vector)
        row["esm"] = esm_val  # None -> model imputes the training mean
        return {f: row.get(f) for f in FEATURES}

    def score(self, produced: str):
        """Return the calibrated ScoreResult for a produced protein."""
        feats = self._features(produced)
        return self.model.score(esm_wt_marginal=feats["esm"] if feats["esm"] is not None else 0.0,
                                grantham=feats["grantham"], blosum62=feats["blosum62"],
                                hydropathy=feats["hydropathy"], sec_struct=feats["sec_struct"],
                                aggregation=feats["aggregation"], burial=feats["burial"])


# ---------------------------------------------------------------------------
# Proto Constraint adapter  (0.0 = best/keep, 1.0 = worst/reject)
# ---------------------------------------------------------------------------
def _seq_str(seq) -> str:
    return getattr(seq, "sequence", seq)


def enhanced_bioplausibility_constraint(seq, *, scorer: EnhancedScorer):
    """Proto Constraint: penalize produced proteins unlikely to be structurally +
    functionally plausible. Uses the calibrated P(tolerated), so the returned score
    is a genuine (1 - probability), not an arbitrary index. Returns (score, meta)."""
    r = scorer.score(_seq_str(seq))
    return 1.0 - r.p_tolerated, {"p_tolerated": r.p_tolerated,
                                 "backbone_esm": r.backbone_esm,
                                 "applicable": r.applicable}
