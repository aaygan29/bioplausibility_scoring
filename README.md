# bioplausibility_scoring

A scorer that separates two things a protein-safety filter must not conflate: whether an edited protein variant *looks* biologically plausible, and whether it would still *function*. Built to red-team protein safety filters, where "looks natural" alone is not the real dual-use signal.

## What it does

- Scores a protein variant on plausibility signals (evolutionary conservation, structural context, hydrophobicity/physicochemical change) and on a functional-viability signal.
- Combines signals with a cross-validated learned model and compares strategies (single signals, naive average, learned combiner).
- Ships a runnable reference scorer plus a validation harness (construct/internal/predictive validity and calibration).

## Result on the benchmark

Across 10 ProteinGym deep-mutational-scanning assays (36,374 single variants, 9 protein classes), the leakage-corrected metric in [`enhanced/`](enhanced/) reaches **mean AUC 0.719, +0.059 over zero-shot ESM2** (0.660 to 0.719), with positive lift on **8 of 9 proteins**, stable across seeds (0.717 to 0.721).

The headline correction is honest cross-validation. The original strongest feature (`plddt_region`, a pure function of residue position) leaked under random-fold CV: it inflated GBM AUC to 0.807, but collapsed to **0.712 under leave-position-out** and **0.704 under leave-one-protein-out** once positions could no longer sit in both train and test. Dropping it and rebuilding around the ESM2 zero-shot substitution signal gives the honest 0.719. See [`enhanced/ENHANCED_METRIC.md`](enhanced/ENHANCED_METRIC.md), [`enhanced/ROBUSTNESS.md`](enhanced/ROBUSTNESS.md), and the metric / leakage / seed-stability CSVs and figures in [`enhanced/`](enhanced/).

## Data & grounding

- ProteinGym deep-mutational-scanning assays (Notin et al., 2023), via the genbio-ai/ProteinGYM-DMS mirror, as ground-truth fitness labels.
- Signals: ESM2 language-model zero-shot scores (Lin et al., 2023) as the backbone, plus Grantham/BLOSUM/hydropathy, secondary-structure, aggregation and burial terms in the refiner. `plddt_region` is deliberately excluded as leakage.
- Framing follows the SafeProtein red-teaming task and prior evidence that structural plausibility does not imply functional retention.

*Note: the `enhanced/*.joblib` files are pickled scikit-learn models; only load model files you trust.*

## License

MIT — see [LICENSE](LICENSE).
