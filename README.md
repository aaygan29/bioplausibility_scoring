# bioplausibility_scoring

A scorer that separates two things a protein-safety filter must not conflate: whether an edited protein variant *looks* biologically plausible, and whether it would still *function*. Built to red-team protein safety filters, where "looks natural" alone is not the real dual-use signal.

## What it does

- Scores a protein variant on plausibility signals (evolutionary conservation, structural context, hydrophobicity/physicochemical change) and on a functional-viability signal.
- Combines signals with a cross-validated learned model and compares strategies (single signals, naive average, learned combiner).
- Ships a runnable reference scorer plus a validation harness (construct/internal/predictive validity and calibration).

**Result on the benchmark:** across 10 ProteinGym deep-mutational-scanning assays spanning 9 protein classes, the cross-validated learned combiner beats a naive-average baseline on all 10 assays (mean AUC around 0.72). The scorer is honest about scope and is hardened against cross-validation leakage.

## Data & grounding

- ProteinGym deep-mutational-scanning assays (Notin et al., 2023), via the genbio-ai/ProteinGYM-DMS mirror, as ground-truth fitness labels.
- Signals: ESM2-150M language-model scores (Lin et al., 2023), AlphaFold pLDDT structural context, Grantham/hydropathy physicochemical terms.
- Framing follows the SafeProtein red-teaming task and prior evidence that structural plausibility does not imply functional retention.

## License

MIT — see [LICENSE](LICENSE).
