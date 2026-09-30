# Examples

1. `compare_same_subjects.py`: ID-matched PRE/POST data, condition swaps within subjects.
2. `compare_independent_subjects.py`: ANH/RES data, complete-curve between-subject permutations.
3. `compare_three_groups.py`: CTR/ANH/RES, L2_Grand sum of three squared distances, then pair-specific tests with Holm correction.

Run from the project root, e.g. `python examples/compare_three_groups.py`.
Default test B=199; `--mode final` requests B=9,999. Exact enumeration is used
when cheaper. Normal runs use compatible caches; `--regenerate` starts fresh.
All examples accept `--data`, `--groups`, `--seed`, `--permutations`, `--out`,
`--cache-dir`, `--response-label`, and `--formats png pdf svg`.

For the three-group example, pairwise outputs are computed even when the global
null is not rejected. By default the confirmatory decision is gated by the
omnibus result; the raw and Holm-adjusted values remain visible. No example is
selected or altered to force a statistically significant result.
