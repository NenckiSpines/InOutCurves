# Poster figure entry points

The files `poster_<purpose>.py` are short wrappers for shared workflows under
`scripts/`; their names describe the figure rather than an old notebook cell.
Use `--help` for all settings. `poster_notebook_suite.py` creates four figures;
`poster_validation_suite.py` creates three. Individual wrappers select only the
requested figure, and reuse the same compatible suite cache.

For example:

    python analysis/poster_sample_size.py --regenerate --mode test
    python analysis/poster_method_agreement.py --mode test --replot

The second command uses the sample-size run's stored paired method p-values; it
does not generate different datasets for the method-agreement plot.
