#!/usr/bin/env python3
"""Classical LTP: the same subjects measured before and after LTP. IDs are matched explicitly.

python examples/compare_same_subjects.py                       # small execution test
python examples/compare_same_subjects.py --mode final          # 9,999 requested permutations
python examples/compare_same_subjects.py --regenerate          # recompute rather than reuse cache

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
        main(default_data=INPUT_DATA/'ltp_paired.csv', default_groups=['PRE', 'POST'],
             default_design='paired', example_name='compare_same_subjects')
    except (ValueError,FileNotFoundError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
