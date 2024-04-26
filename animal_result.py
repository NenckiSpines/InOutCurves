from dataclasses import dataclass

from numpy.random import normal
from scipy.interpolate import interp1d
from scipy.stats import expon

_distortion_standard_deviations = [
    0.054885048194197456,
    0.05676759556530963,
    0.10814783762667769,
    0.13004693391945915,
    0.124372052893709,
    0.09618222782118657,
    0.07518932352814438,
    0.04667043664556347,
    0.04105516959123446,
    0.045754366942977215,
    0.06974427826764401,
    0.08754087417925256,
    0.10359794124335046,
]


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
        factor = expon.rvs(loc=0.12757045614297235, scale=0.8724295438570279)
        return AnimalResult(
            animal_id=self.animal_id,
            responses=[
                factor * r + normal(scale=_distortion_standard_deviations[i])
                for i, r in enumerate(self.responses)
            ],
            group_id=self.group_id,
        )

    def copy(self, group_id):
        return AnimalResult(self.animal_id, self.responses, group_id)
