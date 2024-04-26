import csv
from dataclasses import dataclass

from numpy import mean
from numpy.random import shuffle
from scipy.stats import sem

from animal_result import AnimalResult
from constants import STIMULATION_CURRENT_LABEL, RESPONSE_LABEL, PLOT_LINE_WIDTH, RANDOMIZED_GROUP_ID_PREFIX, \
    PLOT_COLORS
from group_result import GroupResult
from group_results import GroupResults


@dataclass
class AnimalResults:
    currents: list
    results: list

    @staticmethod
    def from_csv(path, excluded_group_ids=None):
        if excluded_group_ids is None:
            excluded_group_ids = []
        with open(path) as file:
            reader = csv.reader(file)
            header = next(reader)
            currents = [float(string) for string in header[2:]]
            return AnimalResults(
                currents=currents,
                results=[
                    ar
                    for ar in [AnimalResult.from_csv_row(csv_row, currents, excluded_group_ids) for csv_row in reader]
                    if ar is not None
                ],
            )

    def get_group_ids(self):
        return sorted({r.group_id for r in self.results})

    def get_group_colors(self):
        return {gi: color for gi, color in zip(self.get_group_ids(), PLOT_COLORS)}

    def plot(self, axes):
        for i, r in enumerate(self.results):
            axes.plot(self.currents, r.responses, color=self.get_group_colors()[r.group_id], linewidth=PLOT_LINE_WIDTH)
        axes.set_xlabel(STIMULATION_CURRENT_LABEL)
        axes.set_ylabel(RESPONSE_LABEL)

    def get_average_responses(self):
        return mean(a=[r.responses for r in self.results], axis=0)

    def group_by_animal(self):
        animal_ids = {r.animal_id for r in self.results}
        results = []
        for ai in animal_ids:
            group_ids = {r.group_id for r in self.results if r.animal_id == ai}
            for gi in group_ids:
                filtered_results = [r for r in self.results if r.animal_id == ai and r.group_id == gi]
                filtered_responses = [fr.responses for fr in filtered_results]
                results.append(AnimalResult(animal_id=ai, responses=mean(a=filtered_responses, axis=0), group_id=gi))
        return AnimalResults(self.currents, results)

    def group_by_group(self):
        results = []
        errors = []
        for gi in self.get_group_ids():
            filtered_results = [r for r in self.results if r.group_id == gi]
            filtered_responses = [fr.responses for fr in filtered_results]
            results.append(GroupResult(group_id=gi, responses=mean(a=filtered_responses, axis=0)))
            errors.append(sem(a=filtered_responses, axis=0))
        return GroupResults(self.currents, results, errors, self.get_group_colors())

    def randomize_groups(self, permute_vertically=False):
        if permute_vertically:
            results = []
            grouped_by_group = [[r for r in self.results if r.group_id == gi] for gi in self.get_group_ids()]
            for i in range(len(grouped_by_group[0])):
                column = [row[i] for row in grouped_by_group]
                group_ids = [f"{RANDOMIZED_GROUP_ID_PREFIX}{i}" for i in range(len(self.get_group_ids()))]
                shuffle(group_ids)
                results.extend([result.copy(gi) for result, gi in zip(column, group_ids)])
        else:
            old_to_new_group_ids = {gi: f"{RANDOMIZED_GROUP_ID_PREFIX}{i}" for i, gi in enumerate(self.get_group_ids())}
            group_ids = [old_to_new_group_ids[r.group_id] for r in self.results]
            shuffle(group_ids)
            results = [r.copy(gi) for r, gi in zip(self.results, group_ids)]
        return AnimalResults(self.currents, results)

    def to_fake(self, keep_differences=False, group_count=2, animals_per_group=6, results_per_animal=3):
        results = []
        animal_id = 0
        if keep_differences:
            for i, gi in enumerate(self.get_group_ids()):
                average_responses = mean(a=[r.responses for r in self.results if r.group_id == gi], axis=0)
                for _ in range(animals_per_group):
                    for _ in range(results_per_animal):
                        results.append(
                            AnimalResult(animal_id=animal_id, responses=average_responses, group_id=f"{i}").distort(),
                        )
                    animal_id += 1
        else:
            average_responses = self.get_average_responses()
            for i in range(group_count):
                for _ in range(animals_per_group):
                    for _ in range(results_per_animal):
                        results.append(
                            AnimalResult(animal_id=animal_id, responses=average_responses, group_id=f"{i}").distort(),
                        )
                    animal_id += 1
        return AnimalResults(self.currents, results)
