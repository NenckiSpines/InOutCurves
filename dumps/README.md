# Numerical results and caches

- `legacy/`: unchanged historical numeric pickles/CSVs, logically renamed.
- `notebook/<mode>/`: corrected numerical pickle caches and CSV exports.
- `validation/<mode>_<model>/`: raw per-experiment paired method results, calibration arrays, summary rates and a fingerprinted manifest.
- `examples/<example>/<mode>/`: subject-level comparison cache, omnibus result, null distribution and post-hoc tables.
- `calibration/<model>/`: descriptive calibration exports.

Reuse matching files on normal runs; `--regenerate` recomputes managed results.
Different numeric library versions can require regeneration even with identical
input data. Never run concurrent writers into the same cache folder. A test cache
is not relabelled as a final-size run.
