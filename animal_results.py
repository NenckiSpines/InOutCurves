from dataclasses import dataclass, field
import csv

from matplotlib import pyplot
from matplotlib.pyplot import colormaps
from numpy import mean, array
from numpy.random import randint, shuffle

from animal_result import AnimalResult
from group_result import GroupResult
from group_results import GroupResults


@dataclass
class AnimalResults:
    currents: list
    results: list
    group_ids: set = field(init=False)
    group_colors: dict = field(init=False)

    def __post_init__(self):
        self.group_ids = {r.group_id for r in self.results}
        color_map = colormaps["Set1"]
        self.group_colors = dict([(gi, color_map(i)) for (i, gi) in enumerate(self.group_ids)])

    @staticmethod
    def from_csv(path):
        with open(path) as file:
            reader = csv.reader(file)
            header = next(reader)
            currents = [float(string) for string in header[2:]]
            return AnimalResults(
                currents=currents,
                results=[AnimalResult.from_csv_row(csv_row, currents) for csv_row in reader],
            )

    def plot(self, title):
        pyplot.title(title)
        for i, r in enumerate(self.results):
            pyplot.plot(self.currents, r.responses, color=self.group_colors[r.group_id])

    def group_by_animal(self):
        animal_ids = {r.animal_id for r in self.results}
        results = []
        for ai in animal_ids:
            filtered_results = [r for r in self.results if r.animal_id == ai]
            filtered_responses = [fr.responses for fr in filtered_results]
            results.append(
                AnimalResult(
                    animal_id=ai,
                    responses=[mean(series) for series in zip(*[array(fr) for fr in filtered_responses])],
                    group_id=filtered_results[0].group_id,
                )
            )
        return AnimalResults(self.currents, results)

    def group_by_group(self):
        results = []
        for gi in self.group_ids:
            filtered_results = [r for r in self.results if r.group_id == gi]
            filtered_responses = [fr.responses for fr in filtered_results]
            results.append(
                GroupResult(
                    group_id=gi,
                    responses=[mean(series) for series in zip(*[array(fr) for fr in filtered_responses])],
                )
            )
        return GroupResults(self.currents, results, self.group_colors)

    def randomize_groups(self):
        group_occurrence_counts = []
        for i in range(len(self.group_ids) - 1):
            group_occurrence_counts.append(
                randint(1, len(self.results) - sum(group_occurrence_counts) - len(self.group_ids) + i + 2),
            )
        group_occurrence_counts.append(len(self.results) - sum(group_occurrence_counts))
        group_ids = []
        for gi, goc in zip(self.group_ids, group_occurrence_counts):
            group_ids.extend([gi for _ in range(goc)])
        shuffle(group_ids)
        return AnimalResults(currents=self.currents, results=[r.copy(gi) for r, gi in zip(self.results, group_ids)])

    def to_fake(self, group_count=2, animals_per_group=6, results_per_animal=3):
        real_responses = [r.responses for r in self.results]
        average_responses = [mean(series) for series in zip(*[array(r) for r in real_responses])]
        results = []
        animal_id = 0
        for group_index in range(group_count):
            for _ in range(animals_per_group):
                for _ in range(results_per_animal):
                    results.append(
                        AnimalResult(
                            animal_id=animal_id,
                            responses=average_responses,
                            group_id=f"{group_index}",
                        ).distort()
                    )
                animal_id += 1
        return AnimalResults(self.currents, results)
