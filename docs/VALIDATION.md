# Verification performed for this clean edition

**47 unit tests passed.** Tests include exact two-, three- and four-group
comparisons against SciPy's independent permutation implementation; exact paired
sign-flip comparisons; unequal group sizes; the grand sum (no accidental extra
squaring); zero-statistic ties; the +1 Monte Carlo correction; group-order and
response-scale invariance; subject aggregation and ID matching; missing-pair and
shared-independent-ID rejection; numeric cache safety; and ANOVA/t-test checks.

**24 regression cases passed against the uploaded poster script.** Both calibration
models, seeded null/alternative datasets, distance statistics, exact/Monte Carlo
p-values and allocation counts agree within floating-point tolerance. Holm
adjustment also matched Statsmodels in 40 separate randomized cross-checks.
Details: `refactor_regression_checks.json`.

Both complete poster suites were run in test mode, including the optional archive
validation model. All three real-data examples were run in both test and final
permutation-count modes. Final-settings examples are single dataset analyses,
**not** the R=5,000-per-scenario simulation benchmarks.

Serial and two-worker simulation results were identical. Cache-only redraws did
not change numeric checkpoint bytes. Truncated CSV and partial pickle simulations
resumed to the same completed results. Changed scientific settings were rejected
without overwriting checkpoints. Entry points were also tested from an external
working directory. The two calibration diagnostics and explicit legacy-pickle
plot were exercised. All source-input checksums remain unchanged after execution.

The seeded null sanity checks in `integration_checks.json` concern one simple
Gaussian population family; they are not proof of broad Type I error robustness.
All produced primary PDF figures were rendered for visual inspection.

## Re-run locally

```bash
python -m unittest discover -s tests -v
python analysis/poster_validation_suite.py --self-test
python analysis/poster_notebook_suite.py --self-test
```

Full R=5,000 simulation benchmarks, broad paired/multi-group power studies,
outlier contamination, unrestricted unequal variances, missing-data handling,
and every supported operating system have **not** been validated here. The code
was executed on Linux; plain Python entry points and spawn-based multiprocessing
are provided for portability, but no separate Windows/macOS runtime test was made.

The original CSV does not encode physical response units. The output axes retain
'source units' until these are supplied from experimental metadata.
