from dataclasses import dataclass

from matplotlib import pyplot

from utilities import get_l2


@dataclass
class GroupResults:
    currents: list
    results: list
    group_colors: dict

    def plot(self, title):
        pyplot.title(title)
        for r in self.results:
            pyplot.plot(self.currents, r.responses, label=r.group_id, color=self.group_colors[r.group_id])
        pyplot.legend()

    def get_l2(self, first_index=0, second_index=1):
        return get_l2(
            fy=self.results[first_index].responses,
            gy=self.results[second_index].responses,
            x=self.currents,
        )
