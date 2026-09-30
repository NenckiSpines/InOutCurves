"""Strict CSV ingestion, subject aggregation, and explicit experimental designs.

CSV format: GROUP,SUBJECT,0,25,... . Duplicate (group,subject) rows are repeated
recordings, averaged before inference. Missing values are never silently filled.
Paired designs are matched by subject ID, not row order. Independent designs
reject subject IDs shared across selected groups. Do not relabel paired data to
make them appear independent.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from .subject_result import SubjectResult
from .distances import validate_grid


@dataclass
class SubjectResults:
    currents: np.ndarray
    results: list[SubjectResult]
    path: Path | None = None

    def __post_init__(self) -> None:
        self.currents = validate_grid(self.currents)
        if not self.results:
            raise ValueError('No recordings were supplied.')
        if any(len(r.responses) != len(self.currents) for r in self.results):
            raise ValueError('Every curve must use the complete common stimulus grid.')

    @classmethod
    def from_csv(cls, path: str | Path, excluded_group_ids: Sequence[str] = ()) -> 'SubjectResults':
        path = Path(path).resolve()
        results = []
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle)
            head = next(reader, [])
            if len(head) < 4 or [s.strip().upper() for s in head[:2]] != ['GROUP', 'SUBJECT']:
                raise ValueError(f'{path}: CSV must start with GROUP,SUBJECT, then numeric stimulus levels. '
                                 'Only rename the identifier HEADER in older input files; preserve IDs and measurements.')
            x = validate_grid([float(s) for s in head[2:]])
            for line, row in enumerate(reader, 2):
                if not row or not any(v.strip() for v in row):
                    continue
                if len(row) != len(head):
                    raise ValueError(f'{path}:{line}: incorrect number of fields.')
                group, subject = row[0].strip(), row[1].strip()
                if group in excluded_group_ids:
                    continue
                try:
                    result = SubjectResult(subject, [float(v) for v in row[2:]], group)
                except ValueError as exc:
                    raise ValueError(f'{path}:{line}: {exc}') from exc
                results.append(result)
        return cls(x, results, path)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['GROUP', 'SUBJECT', *self.currents.tolist()])
            for r in self.results:
                writer.writerow([r.group_id, r.subject_id, *r.responses.tolist()])

    def get_group_ids(self) -> list[str]:
        return sorted({r.group_id for r in self.results})

    def select(self, groups: Sequence[str]) -> 'SubjectResults':
        if len(set(groups)) != len(groups) or not groups:
            raise ValueError('Group labels must be unique and nonempty.')
        absent = set(groups) - set(self.get_group_ids())
        if absent:
            raise ValueError(f'Unknown group(s): {sorted(absent)}')
        return SubjectResults(self.currents, [r for r in self.results if r.group_id in groups], self.path)

    def group_by_subject(self) -> 'SubjectResults':
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        for r in self.results:
            grouped.setdefault((r.group_id, r.subject_id), []).append(r.responses)
        records = [SubjectResult(subject, np.mean(values, axis=0), group)
                   for (group, subject), values in sorted(grouped.items())]
        return SubjectResults(self.currents, records, self.path)

    def subject_arrays(self, groups: Sequence[str] | None = None
                       ) -> tuple[tuple[np.ndarray, ...], tuple[list[str], ...]]:
        labels = list(groups) if groups is not None else self.get_group_ids()
        averaged = self.select(labels).group_by_subject()
        arrays, ids = [], []
        for group in labels:
            block = sorted([r for r in averaged.results if r.group_id == group], key=lambda r: r.subject_id)
            if len(block) < 2:
                raise ValueError(f'Group {group!r} has fewer than two subjects.')
            arrays.append(np.stack([r.responses for r in block]))
            ids.append([r.subject_id for r in block])
        return tuple(arrays), tuple(ids)

    def analysis_arrays(self, groups: Sequence[str], design: str
                        ) -> tuple[tuple[np.ndarray, ...], tuple[list[str], ...]]:
        arrays, ids = self.subject_arrays(groups)
        if len(groups) < 2:
            raise ValueError('Comparison requires at least two groups.')
        if design == 'paired':
            if len(groups) != 2:
                raise ValueError('Paired inference currently supports exactly two conditions.')
            if set(ids[0]) != set(ids[1]):
                raise ValueError('Paired analysis requires the same subject IDs in both conditions; '
                                 f'only in first: {sorted(set(ids[0])-set(ids[1]))}; '
                                 f'only in second: {sorted(set(ids[1])-set(ids[0]))}. No pairs were silently dropped.')
            # Both blocks were sorted by ID, so they are now aligned, not merely same-sized.
            if ids[0] != ids[1]:
                raise AssertionError('Internal paired-ID alignment error.')
        elif design == 'independent':
            seen: set[str] = set()
            for group, block in zip(groups, ids):
                repeated = seen.intersection(block)
                if repeated:
                    raise ValueError(f'Independent groups share subject IDs {sorted(repeated)} in {group!r}. '
                                     'Use a paired design for repeated subjects. For genuinely distinct subjects, '
                                     'supply globally unique original IDs after checking acquisition metadata.')
                seen.update(block)
        else:
            raise ValueError('Design must be paired or independent.')
        return arrays, ids

    def randomize_groups(self, *, design: str = 'independent', groups: Sequence[str] | None = None,
                         rng: np.random.Generator | None = None) -> 'SubjectResults':
        """One subject-level reassignment for illustrations; never permute stimulus points."""
        labels = list(groups) if groups is not None else self.get_group_ids()
        arrays, ids = self.analysis_arrays(labels, design)
        rng = np.random.default_rng() if rng is None else rng
        records = []
        if design == 'paired':
            for i, subject in enumerate(ids[0]):
                order = rng.permutation(2)
                for g in range(2):
                    records.append(SubjectResult(subject, arrays[order[g]][i], labels[g]))
        else:
            pooled = [(sid, curve) for arr, block in zip(arrays, ids) for sid, curve in zip(block, arr)]
            order = rng.permutation(len(pooled))
            offset = 0
            for label, arr in zip(labels, arrays):
                for index in order[offset:offset+len(arr)]:
                    sid, curve = pooled[index]
                    records.append(SubjectResult(sid, curve, label))
                offset += len(arr)
        return SubjectResults(self.currents, records, self.path)

    def group_by_group(self, groups: Sequence[str] | None = None):
        from .group_results import GroupResults
        labels = list(groups) if groups is not None else self.get_group_ids()
        arrays, _ = self.subject_arrays(labels)
        return GroupResults(self.currents, tuple(labels), np.stack([a.mean(0) for a in arrays]),
                            np.stack([a.std(0, ddof=1)/np.sqrt(len(a)) for a in arrays]))
