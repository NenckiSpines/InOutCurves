# Historical results: retained, not silently corrected

The numeric file contents are unchanged from the original archive. Renames:

- p_values_without_differences.pkl -> p_values_null.pkl
- p_values_with_differences.pkl -> p_values_alternative.pkl
- rates_vs_animals_per_group.pkl -> sample_size_rates.pkl
- root-level p-value CSVs -> p_values_{null,alternative}_10000.csv
- data/p_values_{NO,DIFF}_17.csv -> method_comparison_{null,alternative}.csv

'Paired method p-values' means the two methods analyzed the same simulated
experiment; it does NOT establish a paired-subject biological design.

Their original generating settings/seeds are not fully recorded. The old scalar
p-value lists may contain zeros and cannot be fixed without the original null
statistics/allocation counts. They are excluded from new inference and included
in an explicitly historical plot only with `--legacy-pickles`.
