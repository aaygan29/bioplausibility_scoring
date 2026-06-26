# The whole thing, in plain words

## The question we're answering
AI models are being proposed to screen DNA/protein sequences for danger (a
"biosecurity filter"). Two papers we build on showed these filters can be fooled:
- **Krishnan et al.** — change a few DNA letters and a genomic model misreads a
  dangerous sequence as safe.
- **SafeProtein (Fan et al.)** — similar trick works on protein models.

But fooling the *model* isn't the same as making a *real* threat. A clever edit
might slip past the filter yet produce a protein that is broken and harmless.
A separate paper (**Ikonomova**) proved exactly this: a sequence can look fine
on paper but fold into something that doesn't work.

So we need to answer two different questions about any edited sequence:
1. **Does it look like a natural, believable edit?**  -> "bioplausibility"
2. **Will the protein it produces actually function?** -> "functional viability"

These are NOT the same, and that gap is the whole point.

---

## What we built: two scores

### Score 1 — BPS-P (bioplausibility): "does this edit look natural?"
We combine four signals, each rated 0 to 1, and **average** them:
- **Grantham** — is the amino-acid swap chemically gentle or drastic?
- **Active-site** — did it avoid the protein's important residues?
- **Codon** — does the DNA still read like the organism's normal code?
- **Structure** — does the predicted fold stay confident?

Average is fine here because "plausible" is a soft, overall impression.

### Score 2 — FVS (functional viability): "will the protein work?"
Here averaging is WRONG. To work, a protein must pass *every* checkpoint:
it must be **made** correctly, AND **fold**, AND keep its **active site**.
Fail any one and it's dead — a great fold can't rescue a destroyed active site.

So instead of averaging, we **multiply** four factors (a logical AND):

```
FVS = (made correctly?) x (folds?) x (active site intact?) x (expressed?)
```

- **Made correctly (P_prod):** read the altered DNA codons. A letter swap that
  changes one amino acid is fine; an insertion that shifts the reading frame, or
  an early STOP that chops off the active site, scores ZERO — no working protein
  is made at all. (This is the "what does the genetic code actually produce" part.)
- **Folds (V_fold):** use real physics. A tool called **RaSP** predicts how much
  a mutation destabilizes the fold; we convert that into the fraction of molecules
  that stay folded using a standard thermodynamics equation.
- **Active site intact (F_site):** using the protein's 3D shape, we check whether
  edits land on or near the functional residues. Close + drastic = function lost.
- **Expressed (E):** a minor term for how much protein gets made.

Because it multiplies, one catastrophic failure correctly zeroes the whole score.
We proved this: an edit that deletes the active site gets FVS = 0.00, while an
average-based score would have called it "75% fine."

---

## How we check the scores are right
ProteinGym is a public database of real lab experiments where scientists mutated
proteins and measured whether they still worked. We found it contains real
**toxins** (CcdB, ParE) and a real **virus** (Coxsackie) with this data — so we
can test our scores against ground truth, then carry them over to SafeProtein.

We check three things:
1. **Does the score track reality?** (correlation with measured function)
2. **Can it tell working from broken?** (AUC, like a diagnostic test's accuracy)
3. **Which factor matters most?** (this also answers the project's research question
   about which biological constraint best limits an attacker)

Early result: adding the real structural signal lifted accuracy from 0.53 (coin
flip) to 0.72 — already close to the big ESM2 model's 0.76, using simple,
explainable math instead of a black box.

---

## Where it stands (honest status)
- The math, code, and testing pipeline all work end-to-end on real toxin data.
- Two ingredients still need plugging in to get final numbers:
  1. **RaSP** ddG values (free; runs on the protein structures we already have)
     -> turns on the "folds?" factor.
  2. **Active-site residues** from UniProt -> turns on the "active site intact?" factor.
- Until those are in, FVS scores at chance — expected, because it's only as good
  as its inputs. Plugging them in is the next step.

---

## The end result (what this becomes)
A single, explainable tool that takes an edited DNA/protein sequence and returns:
- a **bioplausibility** score (does this look like a believable edit?), and
- a **functional-viability** score (will the resulting protein actually work?),

each built from transparent biological math rather than a black-box model — so a
screening system can flag the dangerous case: an edit that is *both* believable
*and* still produces a working protein. That combination is the real risk, and
no existing protein-model benchmark measures it.

## File map
- `SPEC.md` — bioplausibility (BPS-P) math
- `FUNCTIONAL_VIABILITY.md` — functional-viability (FVS) math
- `bps_p.py`, `fvs.py`, `ddg.py` — the scorers
- `validate.py`, `validate_fvs.py` — the testing harnesses
- `RASP_SETUP.md` — how to turn on the real folding factor
- `README.md` — data + overlap findings
