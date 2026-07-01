"""
proto_bioplausibility.py — run our bioplausibility / functional-viability scoring
as Proto Constraints.

Proto (Hie et al., 2026) builds designs from Segments, Generators, Constraints,
and Optimizers. A Constraint scores a candidate from 0.0 (perfect) to 1.0 (worst),
and the Optimizer MINIMIZES the total. Our scores run the other way (0..1, higher
= more plausible / more functional), so the adapter INVERTS them:

    proto_constraint_score = 1 - our_score

This module gives three things:
  1. BioPlausibilityScorer — loads a protein's context once (structure, active
     sites, optional RaSP ddG) and scores any variant fast. Framework-agnostic.
  2. Proto constraint adapters — bioplausibility_constraint, functional_viability_
     constraint, robust_viability_constraint — drop straight into Proto's
     Constraint(function=...) slot.
  3. build_prism_program() — a full Proto program (protein Segment + point-mutation
     Generator + our Constraints + MCMC Optimizer) that searches for edits which
     are evasive AND plausible AND still functional (the PRISM "dangerous corner").

Run `python3 proto_bioplausibility.py` for a standalone self-test (no Proto needed).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Optional

from bps_p import bps_p, load_af2_plddt, BPSWeights
from fvs import fvs, load_af2_ca, rasp_ddg_stub
from constraints_extra import EXTRA_CONSTRAINTS, analyze_edits
try:
    from ddg import load_rasp_table, ddg_for_variant
    _HAVE_DDG = True
except Exception:
    _HAVE_DDG = False


# ---------------------------------------------------------------------------
# 1. The scoring kernel (no Proto dependency)
# ---------------------------------------------------------------------------
@dataclass
class BioPlausibilityScorer:
    """Holds one protein's reference context and scores its variants.

    wt          : wild-type amino-acid sequence (the reference).
    ca          : {resseq_1based: xyz} CA coordinates (from AF2).
    plddt       : per-residue pLDDT list (0-based), used by the structural term.
    active_site : {residue_index_0based: essentiality} (catalytic=1.0, binding=0.5).
    rasp_table  : optional {(pos,wt,mut): ddG} from RaSP; else a Grantham-scaled stub.
    weights     : BPSWeights for the additive plausibility score.
    lam         : evidence floor in the robust combination (see ROBUST_METRIC.md).
    """
    wt: str
    ca: Dict[int, "object"]
    plddt: list
    active_site: Dict[int, float]
    rasp_table: Optional[Dict] = None
    weights: Optional[BPSWeights] = None
    lam: float = 0.5
    msa_freqs: Optional[list] = None   # per-column MSA frequencies (optional)

    @classmethod
    def from_files(cls, wt: str, pdb_path: str,
                   active_sites_json: Optional[str] = None,
                   assay_key: Optional[str] = None,
                   rasp_csv: Optional[str] = None,
                   **kw) -> "BioPlausibilityScorer":
        ca, _ = load_af2_ca(pdb_path)
        plddt = load_af2_plddt(pdb_path)
        active = {}
        if active_sites_json:
            allsites = json.load(open(active_sites_json))
            for k, v in allsites.items():
                if k != "_README" and (assay_key is None or k in assay_key):
                    active = {int(i): float(w) for i, w in v.items()}
                    break
        rasp = load_rasp_table(rasp_csv) if (rasp_csv and _HAVE_DDG) else None
        return cls(wt=wt, ca=ca, plddt=plddt, active_site=active, rasp_table=rasp, **kw)

    # --- individual axes, each in [0,1], higher = better ---
    def plausibility(self, var: str) -> float:
        """Does the edit look like a natural mutation? (additive BPS-P)"""
        return bps_p(self.wt, var, functional_weights=self.active_site,
                     plddt_per_residue=self.plddt, weights=self.weights).score

    def _ddg(self, var: str) -> float:
        if self.rasp_table is not None:
            d = ddg_for_variant(self.wt, var, self.rasp_table)
            if d is not None:
                return d
        return rasp_ddg_stub(self.wt, var)

    def functionality(self, var: str, edit_kind: str = "missense",
                      stop_k: Optional[int] = None) -> float:
        """Will the protein still work? (multiplicative FVS: produced AND folds
        AND active-site intact)."""
        return fvs(self.wt, var, self.ca, self.active_site, ddg=self._ddg(var),
                   edit_kind=edit_kind, stop_k=stop_k).score

    def robust(self, var: str, **kw) -> float:
        """Combined viability-weighted plausibility (ROBUST_METRIC.md form):
        functionality gates, plausibility refines. In [0,1], higher = better."""
        g = self.functionality(var, **kw)          # gate (necessary)
        v = self.plausibility(var)                  # evidence (graded)
        return g * (self.lam + (1.0 - self.lam) * v)

    # --- extra biologically-grounded signals (constraints_extra.py) ---
    def _ctx(self) -> Dict:
        return {"ca": self.ca, "msa_freqs": self.msa_freqs}

    def extras(self, var: str) -> Dict[str, float]:
        """All extra constraints (burial, BLOSUM62, hydropathy, secondary
        structure, aggregation, conservation), each [0,1] higher = better."""
        ctx = self._ctx()
        return {k: f(self.wt, var, ctx) for k, f in EXTRA_CONSTRAINTS.items()}

    def extended_plausibility(self, var: str) -> float:
        """Richer plausibility: core BPS-P blended with the extra signals."""
        ex = self.extras(var)
        return float((self.plausibility(var) + sum(ex.values())) / (1 + len(ex)))

    def analyze(self, var: str) -> list:
        """Per-edit structural/functional diagnostic explaining WHY an edit scores
        as it does (position, burial, distance to active site, BLOSUM, propensities)."""
        return analyze_edits(self.wt, var, self.ca, self.active_site)

    def report(self, var: str, **kw) -> Dict[str, float]:
        r = {"plausibility": self.plausibility(var),
             "functionality": self.functionality(var, **kw),
             "robust": self.robust(var, **kw),
             "extended_plausibility": self.extended_plausibility(var)}
        r.update({f"extra_{k}": v for k, v in self.extras(var).items()})
        return r


# ---------------------------------------------------------------------------
# 2. Proto constraint adapters  (score 0.0 = best, 1.0 = worst)
# ---------------------------------------------------------------------------
def _seq_str(seq) -> str:
    """Proto passes a Sequence object (.sequence) or we accept a raw string."""
    return getattr(seq, "sequence", seq)


def bioplausibility_constraint(seq, *, scorer: BioPlausibilityScorer):
    """Proto Constraint: penalize edits that look UNnatural. Returns (score, meta)."""
    s = scorer.plausibility(_seq_str(seq))
    return 1.0 - s, {"bioplausibility": s}


def functional_viability_constraint(seq, *, scorer: BioPlausibilityScorer,
                                    edit_kind: str = "missense"):
    """Proto Constraint: penalize edits that BREAK the protein."""
    s = scorer.functionality(_seq_str(seq), edit_kind=edit_kind)
    return 1.0 - s, {"functional_viability": s}


def robust_viability_constraint(seq, *, scorer: BioPlausibilityScorer):
    """Proto Constraint: penalize edits that are implausible OR non-functional
    (the combined robust score)."""
    r = scorer.report(_seq_str(seq))
    return 1.0 - r["robust"], r


def make_extra_constraint(name: str):
    """Factory: turn any extra signal (burial, blosum62, hydropathy,
    secondary_structure, aggregation, conservation) into a Proto Constraint."""
    fn = EXTRA_CONSTRAINTS[name]
    def _c(seq, *, scorer: BioPlausibilityScorer):
        s = fn(scorer.wt, _seq_str(seq), scorer._ctx())
        return 1.0 - s, {name: s}
    _c.__name__ = f"{name}_constraint"
    return _c


def extended_plausibility_constraint(seq, *, scorer: BioPlausibilityScorer):
    """Proto Constraint: core plausibility blended with all extra biophysical signals."""
    s = scorer.extended_plausibility(_seq_str(seq))
    return 1.0 - s, {"extended_plausibility": s}


# ---------------------------------------------------------------------------
# 3. A full Proto program (the PRISM dangerous-corner search)
# ---------------------------------------------------------------------------
def build_prism_program(wt: str, scorer: BioPlausibilityScorer,
                        classifier_evasion_fn=None, num_steps: int = 500,
                        num_mutations: int = 6, num_results: int = 10):
    """Wire our Constraints into a Proto optimization that searches for protein
    edits that are evasive AND plausible AND still functional.

    classifier_evasion_fn(variant_str) -> [0,1] toxicity confidence of the target
    biosecurity classifier; lower means it evades. Pass None to search for plausible
    + functional variants only. Requires `proto_language` installed.
    """
    from proto_language.core import Segment, Construct, Constraint, Program
    from proto_language.optimizer import MCMCOptimizer, MCMCOptimizerConfig
    # protein point-mutation generator (analogue of RandomNucleotideGenerator)
    from proto_language.generator import RandomProteinGenerator, RandomProteinGeneratorConfig
    from proto_tools.transforms.masking import MaskingStrategy

    seg = Segment(length=len(wt), sequence_type="protein", sequence=wt)
    construct = Construct(segments=[seg])
    gen = RandomProteinGenerator(RandomProteinGeneratorConfig(
        masking_strategy=MaskingStrategy(num_mutations=num_mutations)))
    gen.assign(seg)

    constraints = [
        # stay biologically plausible (look natural)
        Constraint(inputs=[seg],
                   function=lambda s, **_: bioplausibility_constraint(s, scorer=scorer),
                   weight=1.0),
        # stay functional (produced AND folds AND active-site intact)
        Constraint(inputs=[seg],
                   function=lambda s, **_: functional_viability_constraint(s, scorer=scorer),
                   weight=1.0),
    ]
    if classifier_evasion_fn is not None:
        # evade the biosecurity classifier: minimize its toxicity confidence
        constraints.append(Constraint(
            inputs=[seg],
            function=lambda s, **_: (classifier_evasion_fn(_seq_str(s)),
                                     {"evasion": classifier_evasion_fn(_seq_str(s))}),
            weight=1.0))

    optimizer = MCMCOptimizer(constructs=[construct], generators=[gen],
                              constraints=constraints,
                              config=MCMCOptimizerConfig(num_steps=num_steps,
                                                         num_results=num_results))
    return Program(optimizers=[optimizer], num_results=num_results), construct


# ---------------------------------------------------------------------------
# 4. Standalone self-test (no Proto required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    PG = os.path.expanduser("~/Downloads/ProteinGym")
    pdb = f"{PG}/ProteinGym_AF2_structures/CCDB_ECOLI.pdb"
    sites = "data/active_sites.json"
    # CcdB wild type (101 aa) from the AF2 structure's sequence
    wt = ("MQFKVYTYKRESRYRLFVDVQSDIIDTPGRRMVIPLASARLLSDKVSRELYPVVHIGDESWRMM"
          "TTDMASVPVSVIGEEVADLSHRENDIKNAINLMFWGI")
    scorer = BioPlausibilityScorer.from_files(wt, pdb, active_sites_json=sites,
                                              assay_key="CCDB_ECOLI")
    cases = {
        "conservative far edit (V5I)": wt[:4] + "I" + wt[5:],
        "radical near C-term gyrase site (W99A)": wt[:98] + "A" + wt[99:],
    }
    for name, var in cases.items():
        r = scorer.report(var)
        proto_c = 1.0 - r["robust"]
        print(f"{name:42s} plaus={r['plausibility']:.3f} func={r['functionality']:.3f} "
              f"robust={r['robust']:.3f}  -> proto_constraint={proto_c:.3f}")
    print("\n(Proto constraint: 0.0 = best/keep, 1.0 = worst/reject. "
          "Lower means more plausible AND functional.)")
