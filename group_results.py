from dataclasses import dataclass

from constants import STIMULATION_CURRENT_LABEL, RESPONSE_LABEL, PLOT_LINE_WIDTH
from utilities import get_l2


@dataclass
class GroupResults:
    currents: list
    results: list
    errors: list
    group_colors: dict

    def plot(self, axes):
        for r, e in zip(self.results, self.errors):
            axes.errorbar(
                x=self.currents,
                y=r.responses,
                yerr=e,
                color=self.group_colors[r.group_id],
                label=r.group_id,
                marker=".",
                capsize=4,
                linewidth=PLOT_LINE_WIDTH,
            )
        axes.legend(title="Group")
        axes.set_xlabel(STIMULATION_CURRENT_LABEL)
        axes.set_ylabel(RESPONSE_LABEL)

    def get_l2(self, first_index=0, second_index=1):
        return get_l2(
            fy=self.results[first_index].responses,
            gy=self.results[second_index].responses,
            x=self.currents,
        )
