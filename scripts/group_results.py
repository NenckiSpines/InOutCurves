"""Group-mean summaries, with both pairwise and grand squared distances."""
from dataclasses import dataclass
import numpy as np
from .distances import l2_squared, l2_grand


@dataclass
class GroupResults:
    currents: np.ndarray
    labels: tuple[str, ...]
    means: np.ndarray
    sem: np.ndarray

    def get_l2(self, first_index: int = 0, second_index: int = 1) -> float:
        """Return squared L2; retained conceptual role of the old pairwise accessor."""
        return l2_squared(self.means[first_index], self.means[second_index], self.currents)

    def get_l2_grand(self) -> float:
        return l2_grand(self.means, self.currents)
