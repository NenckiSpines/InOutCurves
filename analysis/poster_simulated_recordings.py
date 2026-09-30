#!/usr/bin/env python3
"""Notebook illustrative experimental/null/alternative recordings.

Example: python analysis/poster_simulated_recordings.py --mode test
Use --mode final for publication settings; start new data with --regenerate.
Both absolute-path execution and python -m analysis.poster_simulated_recordings are supported.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.poster_notebook_workflow import main

if __name__ == '__main__':
    args=sys.argv[1:]
    if not any(a == '--figures' or a.startswith('--figures=') for a in args):
        args += ['--figures',*map(str,[3])]
    try:
        main(args)
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Matching checkpoints can be resumed without --regenerate.')
    except (ValueError,FileNotFoundError,PermissionError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
