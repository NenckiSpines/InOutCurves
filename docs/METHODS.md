# Statistical definitions and design choices

## Unit of analysis

One curve contains all responses on a common stimulus grid. Raw recordings are
averaged within each `(GROUP, SUBJECT)` before computing group means, SEMs or test
statistics. Each retained subject contributes equally within its group. Repeated
recordings and stimulation levels do not increase the independent sample size.

The CSV identifier is treated as the experimental unit supplied by the researcher.
The program cannot determine whether an identifier actually represents a subject,
slice, cell, or another independent unit. That requires acquisition metadata.

## Squared functional distance

For group means f_i and f_j, define

    D_ij = L_ij^2 = integral [f_i(x) - f_j(x)]^2 dx.

For an interval of width h and endpoint differences d0,d1, the exact contribution
for piecewise-linear interpolation is

    h * (d0^2 + d0*d1 + d1^2) / 3.

This is NOT the trapezoidal rule applied to the squared endpoint differences.
There is no square root. Units are squared response units times stimulus units.
No rescaling or sign inversion is performed.

## Paired two-condition comparison

The two sets of subject IDs must agree exactly. After aggregation, both are sorted
by ID and aligned. Each randomization independently swaps the two complete
condition curves within each subject. Equivalently, it sign-flips each subject's
whole difference curve. The statistic is the squared distance between the two
condition means, or the squared L2 norm of the mean difference curve.

There are 2^n assignments for n matched subjects. Pairings themselves are never
randomized. The null requires the relevant within-subject label exchangeability;
a sign-flip interpretation requires symmetry of the difference-curve distribution.
Equality of means alone does not guarantee exactness for arbitrary distributions.
A pre/post intervention is not automatically randomized because the measurements
come from the same subject. No causal inference is created by this algorithm.

This release implements paired inference for exactly two conditions. It rejects
an unsupported paired multi-condition request rather than silently using an
independent-group analysis. Incomplete pairs are not silently discarded.

## Independent two- and multi-group comparison

For G independent groups, pool complete subject curves and redistribute them
among labelled groups while preserving every original group size. Unequal sizes
are permitted. Subject IDs shared across groups are rejected by the dataset API.
For a truly independent design with locally repeated ID numbers, the researcher
must first resolve them into globally unique original IDs using the metadata.

The unweighted grand statistic is

    L2_Grand = sum over i<j of D_ij.

For three groups, this is exactly

    L2_Grand = L_12^2 + L_23^2 + L_13^2.

It is the requested sum of pairwise squared distances, NOT a sum of fourth powers.
Every permutation recalculates all group means and the full sum. There are
N!/(n_1! ... n_G!) labelled group allocations. Exact enumeration considers one
canonical within-group ordering per allocation. Monte Carlo uses uniformly
sampled allocations with replacement across draws. No stimulus point is shuffled.

The exact independent null is exchangeability across groups (for example,
independent draws from one common distribution), not just equality of mean
functions with unrestricted unequal covariance structures. The grand statistic
is sensitive to mean-function differences, not every distributional difference.
It weights pairs equally even when their sizes differ, as requested.

## P-values and finite resolution

The statistic is nonnegative and nondirectional; use its upper tail without
multiplying by two. Numerical ties use a relative floating-point tolerance.

    Exact:       p = count(null statistic >= observed) / all allocations.
    Monte Carlo: p = (1 + count(null statistic >= observed)) / (B + 1).

Exact enumeration includes the original allocation. `exact=auto` enumerates when
the entire allocation space is no larger than both the requested B and the safety
limit. The hard default safety limit is 50,000. `exact=always` fails explicitly
when too large; `exact=never` forces Monte Carlo, useful for precision experiments.

With six subjects in each of two groups there are 924 labelled allocations.
Because swapping the two group labels leaves the statistic unchanged, the minimum
attainable exhaustive p-value is at least 2/924. Thus zero rejection at alpha=.001
in this exact design reflects finite resolution; it is not evidence of ideal
Type I error control at that threshold.

## Pairwise post-hoc inference

The original code supplied a two-group method but not a complete post-hoc family.
The new controller applies that shared two-group engine to every unordered pair.
It generates the null using ONLY the subjects in that pair, not a global
three-group permutation distribution. The latter would impose a stronger null
and could give invalid pairwise conclusions when a third group differs.

Holm adjustment sorts the m raw p-values and computes

    adjusted_(i) = min(1, max over j<=i [(m-j+1) * p_(j)]),

then restores the original pair order. All prespecified pairs enter the family,
not only pairs that looked promising. Valid pairwise p-values are necessary for
the usual family-wise error control.

The program calculates all pairs for transparent reporting. With the default
gate, a confirmatory pairwise rejection requires both the omnibus p-value and the
pair's Holm-adjusted p-value to be <= alpha. The raw p-value, adjusted p-value,
ungated Holm decision, gate status and gated decision are separate output fields.
The gate does not numerically alter the Holm p-values. No Holm adjustment of a
selected-only subset is performed.

## ANOVA comparators

For independent groups on a complete common stimulation grid, the mixed-design
ANOVA Group main effect equals a one-way F test on each subject's arithmetic mean
across stimulation points: the common number of repeated levels multiplies both
relevant sums of squares and cancels in the ratio. The paired two-condition
comparator is the Condition main effect, equal to the squared paired t statistic
on these subject means.

These are not interaction tests or joint tests of arbitrary curve differences.
Their residual/variance assumptions remain parametric. A difference in rejection
rates does not automatically mean one method is universally better.

## Simulations and historical results

Both supplied population models are retained in `scripts/simulation.py`.
`within-group` calibrates gains/residuals within conditions and generates one
curve per subject; `archive` keeps the historical gain/noise parameters and
averages three independently generated recordings per subject. The latter does
not simulate a shared subject random effect. Their sample-size results must not
be silently combined. Neither calibrates all residual covariance or outlier
mechanisms. Calibration uncertainty is not included in the conditional binomial
intervals for simulated power and false-positive rates.

The paired example is a real-data design extension; the supplied poster power
benchmark remains an independent-group simulation. The new multi-group example
is not a full three-group power/robustness benchmark.

## Sources

- SciPy project, `permutation_test` documentation: independent multi-sample and
  paired-label permutation definitions, allocation counts, inclusive tails.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- Statsmodels project, `multipletests` documentation: Holm step-down Bonferroni.
  https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
- Phipson B, Smyth GK (2010). Permutation P-values Should Never Be Zero.
  DOI: 10.2202/1544-6115.1585.
- Morris TP, White IR, Crowther MJ (2019). Using simulation studies to evaluate
  statistical methods. DOI: 10.1002/sim.8086.
- The supplied poster scripts and original source archive, identified by checksum
  in `source_files.json` and `input_migration.json`.
