# Migration from the supplied archive

This is a clean sibling project, not an in-place modification. No original file
was deleted or overwritten. No credentials, notebook execution outputs, `.pyc`
files, draft paper assets, poster logos, or PowerPoint files are required here.

## Names and responsibilities

| Previously | Clean project |
|---|---|
| `animal_result.py`, `AnimalResult`, `animal_id` | `scripts/subject_result.py`, `SubjectResult`, `subject_id` |
| `animal_results.py`, `AnimalResults` | `scripts/subject_results.py`, `SubjectResults` |
| `group_by_animal()` | `group_by_subject()` |
| `animals_per_group`, `results_per_animal` | subject counts and `recordings_per_subject` |
| `group_result.py` and `group_results.py` | `scripts/group_results.py` |
| `utilities.py` distance and p-value helpers | `scripts/distances.py`, `scripts/permutation_tests.py` |
| `reference_tests.py`, `RefTests`, `SAME_SUBJECTS` | `scripts/reference_tests.py`; explicit `design='paired'/'independent'` |
| `simulation_batch.py` | `scripts/simulation.py` plus `scripts/poster_validation_workflow.py` |
| `rate_calc.py` | `scripts/metrics.py` |
| `draw_hist.py` | `analysis/poster_pvalue_distributions.py` and the shared notebook workflow |
| `get_distributions.py` | `python -m scripts.calibrate_variability` and shared calibration |
| `plotSigmoid.py` | `scripts/sigmoid_fit.py`, optional and not used for inference |
| `poster_figures.py` | `analysis/poster_validation_suite.py`, or individual logical entry points |
| `poster_notebook_figures.py` | `analysis/poster_notebook_suite.py`, or individual logical entry points |
| `dump/` mixed cached results | `dumps/legacy/` plus separate managed workflow directories |
| root-level/data-folder curves | logically named `input_data/*.csv`, same IDs/responses |

All active Python filenames, class names, arguments and variables use subject
terminology. Historical old names remain only in migration/provenance documents.
New CSVs use `GROUP,SUBJECT`. A header rename does not change the biological unit.

## Kept, changed and newly implemented

**Kept:** piecewise-linear squared distance; whole-curve reassignment; within-
subject averaging; two supplied simulation models and their numerical parameters;
both poster layouts/palettes; matched-dataset ANOVA comparison; numerical precision
experiment; default test/final controls; historical numerical results separately.

**Shared/refactored:** duplicated integration, p-value, calibration, loader and
metric implementations are now imported from common modules. Scripts do no
analysis merely because they are imported. Source CSVs are read strictly.

**Corrected:** ties and nonzero Monte Carlo p-values; broken global references in
the optional sigmoid helper; ambiguous accuracy labels; false implication that
three independent recordings create a subject-shared random effect; old cache
settings being mistaken for current settings. Corrected-poster statistical
calculations are preserved to numerical tolerance (see the regression report).

**New:** subject-ID matched paired randomization; independent G-group grand
statistic; pair-specific post-hoc nulls with Holm adjustment; automated tests;
runnable examples with result tables and figures. The old positional swapping
prototype was not used as proof of correct matching.

**Not carried forward as validated functionality:** disabled adaptive
randomization, disconnected alternate sigmoid simulation, automatic missing-data
interpolation, and the distribution-selection routine that depended on the
missing `ssqError.py`. The optional sigmoid fit and calibration diagnostics are
available, but no unsupported 'best distribution' result is asserted. No automatic
outlier deletion, unequal-variance guarantee, or paired/multi-group power study
is introduced silently.

## Defaults and cache compatibility

The attached notebook-poster script had `USE_PICKLE_FILES=False`. This clean
edition restores reuse-by-default and defaults to the small `test` preset;
`--regenerate` is the explicit recomputation option. Final expensive simulations
require an explicit initial regeneration. Compatible partial runs resume normally.

Caches are versioned and depend on the shared code, numeric-library versions,
source contents and scientific settings. Old caches cannot automatically become
new corrected tests or new three-group results. They remain in `dumps/legacy/`
and can be displayed explicitly, with their original p-values unchanged.

The renamed `ltp_templates_legacy_ids.csv` retains the archive's distinct POST IDs
for provenance only. Its response values duplicate those in `ltp_paired.csv`;
it is NOT evidence of an independent real-data design. All default LTP workflows
use `ltp_paired.csv`, preserving the real ID correspondence; independent simulated
groups are then generated explicitly.

The code license file from the source archive is included unchanged. Input-data
publication permissions are not inferred from the code license.
