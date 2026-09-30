# Input data

These are the archived response values, with filenames and identifier headers
standardized. `GROUP,SUBJECT` is followed by the common numeric stimulus grid.
See `../docs/input_migration.json` for checksums and the exact mapping.

- `ltp_paired.csv`: PRE and POST, 24 matched IDs. Paired example and simulation templates.
- `stress_response_ca1.csv`: ANH, CTR, RES; respectively 6, 6, 5 unique IDs, with repeated recordings. Independent two-/three-group examples.
- `ltp_fepsp_amplitude.csv`, `ltp_fepsp_slope.csv`, `ltp_fiber_volley_amplitude.csv`: additional paired field-potential readouts from the archive, each with 8 IDs per condition.
- `ltp_templates_legacy_ids.csv`: archival re-identification of the same LTP responses. Not a new independent biological cohort; not used by default.
- `legacy_artificial_curves.csv`: old artificial data; generating settings/design are not established. Not used for a primary validation example.
- `archive_variability.json`: fixed historical shifted-exponential gain, stimulus-specific noise SDs, original stimulus grid and three-recording setting, transcribed from the supplied source.

Confirm biological units and identity mapping from recording metadata. Do not
interpret an ID count as an animal count merely because of its old column name.
No species, ages, treatment protocol, or response-unit values are inferred here.
