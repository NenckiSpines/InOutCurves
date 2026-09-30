"""One recording from one subject in one condition; no analysis at import time."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SubjectResult:
    subject_id: str
    responses: np.ndarray
    group_id: str

    def __post_init__(self) -> None:
        subject, group = str(self.subject_id).strip(), str(self.group_id).strip()
        values = np.asarray(self.responses, dtype=float).copy()
        if not subject or not group:
            raise ValueError('Subject and group identifiers must be nonempty.')
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError('Responses must be a finite one-dimensional curve.')
        values.setflags(write=False)
        object.__setattr__(self, 'subject_id', subject)
        object.__setattr__(self, 'group_id', group)
        object.__setattr__(self, 'responses', values)

    def copy(self, group_id: str) -> 'SubjectResult':
        """Copy the complete curve with another group label, preserving subject ID."""
        return SubjectResult(self.subject_id, self.responses, group_id)
