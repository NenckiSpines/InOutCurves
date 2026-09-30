#!/usr/bin/env python3
"""Stress-response dataset: two different sets of subjects; repeated slices are averaged within subject.

python examples/compare_independent_subjects.py                       # small execution test
python examples/compare_independent_subjects.py --mode final          # 9,999 requested permutations
python examples/compare_independent_subjects.py --regenerate          # recompute rather than reuse cache

The example preserves IDs and response values from the supplied data. Confirm
what each ID represents biologically before interpreting n as a biological count.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_groups import main
from scripts.paths import INPUT_DATA

if __name__=='__main__':
    try:
        main(default_data=INPUT_DATA/'stress_response_ca1.csv', default_groups=['ANH', 'RES'],
             default_design='independent', example_name='compare_independent_subjects')
    except (ValueError,FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
