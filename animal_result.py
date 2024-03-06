from dataclasses import dataclass

from numpy.random import random_sample, rand
from scipy.interpolate import interp1d


@dataclass
class AnimalResult:
    animal_id: int
    responses: list
    group_id: str

    @staticmethod
    def from_csv_row(csv_row, currents, excluded_group_ids):
        if csv_row[0] in excluded_group_ids:
            return None
        responses = []
        missing_response_indices = []
        for i, response in enumerate(csv_row[2:]):
            try:
                responses.append(float(response))
            except ValueError:
                missing_response_indices.append(i)
        currents_with_responses = currents.copy()
        for mri in reversed(missing_response_indices):
            del currents_with_responses[mri]
        interpolated_responses = interp1d(currents_with_responses, responses)
        return AnimalResult(
            animal_id=csv_row[1],
            responses=[interpolated_responses(c) for c in currents],
            group_id=csv_row[0],
        )

    def distort(self):
        noise = 0.1 * random_sample(len(self.responses))
        factor = 2 * rand()
        return AnimalResult(
            animal_id=self.animal_id,
            responses=[factor * (r + noise[i]) for i, r in enumerate(self.responses)],
            group_id=self.group_id,
        )

    def copy(self, group_id):
        return AnimalResult(self.animal_id, self.responses, group_id)
