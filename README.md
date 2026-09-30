# InOutCurves: subject-level whole-curve inference

Clean research-code edition of the supplied project. Python >= 3.10.

The project compares complete electrophysiological input-output curves using
integrated squared differences between group means and subject-level permutation
inference. It retains both poster-figure suites and the two supplied simulation
models, and adds ID-matched paired inference and an independent multi-group
omnibus/post-hoc workflow. The original archive has not been overwritten.

## Quick start

Extract this folder, open a terminal in it, and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/compare_same_subjects.py --regenerate
python examples/compare_independent_subjects.py --regenerate
python examples/compare_three_groups.py --regenerate
```

The examples default to **199 randomizations** for an execution check. For a
higher-precision analysis:

```bash
python examples/compare_same_subjects.py --mode final
python examples/compare_independent_subjects.py --mode final
python examples/compare_three_groups.py --mode final
```

`final` requests **9,999 randomizations**. Exact enumeration is used when it is
cheaper and within the safety limit. The filename/preset does not guarantee that
a dataset or inferential design is scientifically appropriate.

Run entry points from the extracted checkout. An optional editable install (`python -m pip install -e .`) keeps imports attached to these source/data folders; the ordinary wheel does not bundle the research datasets.

## Folder organization

```text
analysis/       Poster entry points, all named poster_<purpose>.py
scripts/        Shared data, statistics, simulations, cache and plotting modules
examples/       Paired, independent two-group and independent three-group examples
input_data/     Source curves with GROUP,SUBJECT headers; historical variability JSON
dumps/          Numerical pickles, raw simulation CSVs, summaries and run manifests
figures/        Generated PNG/PDF/SVG figures and captions
tests/          Self-contained unit tests and small exhaustive statistical checks
docs/           Scientific definitions, migration notes and verification evidence
```

No input response values or subject IDs were changed. Input files were renamed;
only the first two column headings and line-ending conventions were standardized.
See `docs/input_migration.json` for original-to-clean names and checksums.

## Three examples

| Entry point | Input and design | Permutation scheme |
|---|---|---|
| `examples/compare_same_subjects.py` | `ltp_paired.csv`, PRE/POST from the same IDs | Swap complete condition curves **within each subject** |
| `examples/compare_independent_subjects.py` | `stress_response_ca1.csv`, ANH/RES | Pool complete subject curves and reassign, preserving group sizes |
| `examples/compare_three_groups.py` | Same CA1 file, CTR/ANH/RES | Permute across all three groups, then pair-specific two-group tests and Holm correction |

Repeated recordings are averaged **within subject and condition** before any
comparison. Subject IDs are matched explicitly for paired analysis. Missing
pairs cause an error; they are not silently dropped. Independent analysis rejects
IDs shared across groups. Confirm what an ID represents in acquisition metadata.
Do not turn a paired experiment into an independent one just by changing IDs.

Each example writes its numerical cache and result tables to
`dumps/examples/<example>/<mode>/`, and figures/captions to the corresponding
`figures/examples/<example>/<mode>/`. Outputs include `results.json`,
`subject_curves.csv`, `group_summary.csv`, `null_distribution.csv`, and, for the
three-group example, `posthoc_tests.csv`.

## Three-group statistic and post-hoc tests

For piecewise-linear mean curves, let

```text
D_ij = L_ij^2 = integral (mean_i(x) - mean_j(x))^2 dx.
L2_Grand = D_12 + D_23 + D_13.
```

The implementation **sums the squared distances; it does not square them again**.
For more than three independent groups it uses the sum over all unordered pairs.
Every permutation recalculates all group means and the entire grand statistic.
Unequal group sizes are supported and preserved.

The omnibus p-value tests the global exchangeable-population null. Pairwise
follow-up uses the two-group statistic from the original project, through the
shared corrected engine. Each pair uses only its own subjects and its own null
permutations. All pairs are included in a **Holm correction**. Raw and adjusted
p-values are both saved. By default, `reject_after_gate` requires both a
significant omnibus test and a Holm-adjusted pairwise p-value <= alpha.
`reject_holm` is saved separately. All pairwise results are computed for
transparency even when the omnibus gate is closed. Use `--no-omnibus-gate` to
request ungated Holm decisions explicitly.

The supplied code had a two-group engine, not an implemented multiplicity-
controlled post-hoc family. The pairwise controller and correction are new here.

## Poster figures: logically named entry points

| Command file in `analysis/` | Figure(s) |
|---|---|
| `poster_curve_randomization.py` | Original/reassigned subject curves and means, 2x2 |
| `poster_permutation_precision.py` | Distance null distribution and repeated p-value estimates |
| `poster_simulation_design.py` | Experimental templates; alternative and null simulations |
| `poster_sample_size.py` | Power and false-positive rate versus subjects per group |
| `poster_pvalue_distributions.py` | Notebook-model p-value histograms under null/alternative |
| `poster_method_agreement.py` | Matched permutation/ANOVA p-values |
| `poster_simulated_recordings.py` | The additional original notebook simulation illustration |
| `poster_notebook_suite.py` | All four notebook-derived figures |
| `poster_validation_suite.py` | All three simulation-design/ANOVA-comparison figures |

Both suites retain the supplied layouts, palette, lettering and explanatory
captions. The key statistical kernels now live in shared modules instead of
being copied into each plotting script.

Small test runs:

```bash
python analysis/poster_notebook_suite.py --regenerate --mode test
python analysis/poster_validation_suite.py --regenerate --mode test
```

Final-size simulation runs (**not performed as part of this code delivery**):

```bash
python analysis/poster_notebook_suite.py --regenerate --mode final
python analysis/poster_validation_suite.py --regenerate --mode final --jobs 4
```

The notebook suite's final preset uses R=5,000 experiments per null/alternative
scenario, n=6 subjects/group, and B=9,999 requested permutations per test.
The validation suite uses the same R/B at n=5,6,7,8,9,10,12,14,16,18,20. Both
methods analyze the same artificial experiment. The ANOVA comparator remains
its **Group main effect**, not the Group-by-Stimulation interaction.

`n` (subjects), `R` (independent experiments), and `B` (within-test permutations)
are different quantities. Increasing B does not replace a sufficient R.

## Cache reuse, regeneration, and old results

Normal runs reuse matching managed caches. Change a figure label, format, or DPI
without regenerating data. Example:

```bash
python analysis/poster_notebook_suite.py --mode final --dpi 600
python analysis/poster_validation_suite.py --mode final --replot --dpi 600
python analysis/poster_method_agreement.py --mode final --replot
```

`--regenerate` restarts the selected managed analysis; it never overwrites the
source CSVs or `dumps/legacy/`. Matching interrupted simulations resume on a
normal run **without** `--regenerate`. Settings/data/code/version mismatches
produce an error rather than silently using stale numbers. Use a new
`--cache-dir` when retaining multiple configurations.

The validation workflow uses flushed raw-result CSV checkpoints; notebook and
example workflows use restricted numeric pickle caches. No arbitrary project
objects are unpickled. Use only trusted local cache files.

Historical caches are retained under `dumps/legacy/`, renamed logically. Their
unrecorded settings are **not** attributed to current defaults and their old zero
p-values are not repaired from p-value lists alone. To display them intentionally:

```bash
python analysis/poster_pvalue_distributions.py --legacy-pickles
```

That plot goes into a separate `figures/notebook/legacy/` folder and is explicitly
labelled as historical. It is not final validation. Newly regenerated exact/MC
p-values include ties; Monte Carlo uses `(1 + exceedances)/(B + 1)`.

## Two simulation models are intentionally separate

- **within-group** (validation default): calibrate each subject against its own
  condition mean, pool the within-condition gains and centered residuals, and
  generate one independent curve per artificial subject.
- **archive** (notebook workflow; optional in validation): use the supplied fixed
  shifted-exponential gain and stimulus-specific Gaussian SDs, and average three
  independently distorted recordings per artificial subject. This is the
  historical variance-reduction model, not a subject-shared random effect.

```bash
python analysis/poster_validation_suite.py --model archive --regenerate --mode test
python -m scripts.calibrate_variability --model within-group
```

Source PRE/POST measurements supply the simulation templates. The sample-size
benchmark still simulates **independent** groups; adding a paired-data example
does not turn that benchmark into a paired power analysis.

## Custom data and API

CSV format:

```csv
GROUP,SUBJECT,0,25,50,75
Control,S01,0.0,-0.1,-0.2,-0.3
Control,S02,0.0,-0.1,-0.3,-0.4
Treatment,S03,0.0,-0.2,-0.4,-0.6
Treatment,S04,0.0,-0.2,-0.5,-0.7
```

Repeated rows with the same GROUP and SUBJECT represent repeated recordings.
All curves must have finite responses on one strictly increasing common grid.
Missing values require an explicit preprocessing decision; no automatic
interpolation or outlier removal is applied.

```bash
python -m scripts.compare_groups --data /path/to/curves.csv --groups Control Treatment --design independent --mode final
```

```python
from scripts.subject_results import SubjectResults
from scripts.compare_groups import compare_dataset

data = SubjectResults.from_csv("input_data/stress_response_ca1.csv")
result = compare_dataset(data, ["CTR", "ANH", "RES"],
                         design="independent", permutations=9999, seed=42)
print(result["omnibus"]["pvalue"])
for pair in result["posthoc"]:
    print(pair["group_1"], pair["group_2"], pair["p_holm"])
```

## Interpretation and reproducibility

The test avoids a prescribed curve shape but is **not assumption-free**.
Independent-label exchangeability, or within-subject exchangeability for paired
swaps, must be appropriate to the design. In a before/after experiment this is an
assumption, not something guaranteed merely by matching IDs. Equal mean curves
with arbitrary unequal variances are not a general exact-permutation null.

The statistic measures differences between **group-mean functions**, not every
possible difference between distributions of curves. The grand sum is unweighted
across pairs, as requested, even when group sizes differ. A nonsignificant result
is not evidence of equivalence. Small numerical tests are implementation checks,
not proof of robustness to outliers, heterogeneous populations, or all designs.

The code is organized for copying into a GitLab repository. Preserve the original
`LICENSE` and check laboratory permission before publishing the included source
data. The archive contains no GitLab credentials or repository URL.

See `docs/METHODS.md`, `docs/MIGRATION.md`, and `docs/VALIDATION.md`.
