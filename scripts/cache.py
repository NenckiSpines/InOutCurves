"""Atomic numeric caches; fingerprints prevent silent reuse of stale analyses.

Only primitive Python containers and numeric NumPy objects can be unpickled.
Use trusted local caches; the reader is not a general sandbox against resource
exhaustion. Never unpickle unknown files with the unrestricted pickle.load.
"""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
from typing import Any
import numpy as np

SCHEMA=2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_code_hash() -> str:
    """Shared-code changes invalidate managed numerical caches too."""
    root=Path(__file__).parent
    h=hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path,value: Any) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=jsonable,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def write_csv(path: Path,rows: list[dict],fields: list[str] | None=None) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields if fields is not None else (list(rows[0]) if rows else [])
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


class NumericUnpickler(pickle.Unpickler):
    def find_class(self,module: str,name: str) -> Any:
        if module=='numpy' and name in {'dtype','ndarray'}:return getattr(np,name)
        if module in {'numpy.core.multiarray','numpy._core.multiarray'} and name in {'scalar','_reconstruct'}:
            try:m=importlib.import_module('numpy._core.multiarray')
            except ImportError:m=importlib.import_module('numpy.core.multiarray')
            return getattr(m,name)
        raise pickle.UnpicklingError(f'Refused non-numeric pickle global: {module}.{name}')


def numeric_pickle(path: Path) -> Any:
    with Path(path).open('rb') as f:return NumericUnpickler(f).load()


def fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec,sort_keys=True,default=jsonable).encode()).hexdigest()


def save_cache(path: Path,spec: dict,data: dict,complete: bool=True) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=json.loads(json.dumps({'schema':SCHEMA,'spec':spec,'fingerprint':fingerprint(spec),
                                'complete':complete,'saved_utc':utc_now(),'data':data},
                               default=jsonable,allow_nan=False))
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        pickle.dump(value,f,protocol=4);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def read_cache(path: Path,spec: dict,regenerate: bool=False) -> dict | None:
    if regenerate or not Path(path).exists():return None
    value=numeric_pickle(path)
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA:
        raise ValueError(f'Unrecognized managed cache {path}. Use --regenerate or a new --cache-dir.')
    if value.get('fingerprint')!=fingerprint(spec):
        raise ValueError(f'Settings, source, or code mismatch in {path}; use --regenerate or a new --cache-dir.')
    return value
