import typing

import numpy as np

from phyloframe.legacy._impl import TgsAllmdcGroup, tgs_make_allmdc_group

# (gain rate, selection) of a focal trait and two neutral traits
TRAITS = [(2e-3, -0.05), (1e-3, 0.0), (1e-3, 0.0)]
SHIFT = 15.0


def simulate_allmdc_groups(
    num_branches: typing.Sequence[int] = (40_000, 20_000),
    seed: int = 1,
) -> typing.List[TgsAllmdcGroup]:
    """Branches of a single-leaf group and a clade group, each gaining each
    trait independently under the allmdc model, with the clade group
    shifted by `SHIFT`."""
    rng = np.random.default_rng(seed)
    groups = []
    for j, num in enumerate(num_branches):
        lengths = rng.uniform(0.5, 60 if j == 0 else 30, num)
        shift = 0.0 if j == 0 else SHIFT
        gains = []
        for gain_rate, selection in TRAITS:
            p = (
                gain_rate * lengths
                if selection == 0
                else gain_rate * np.expm1(selection * lengths) / selection
            ) * np.exp(selection * shift)
            gains.append(np.flatnonzero(rng.random(num) < p))
        groups.append(
            tgs_make_allmdc_group(
                branch_lengths=lengths,
                origin_times=rng.uniform(0, 100, num),
                trait_units=[np.arange(num)] * len(TRAITS),
                trait_gains=gains,
                neutral_traits=[1, 2],
                trait_names=["focal", "n1", "n2"],
                calibrate=False,
                calibration_neighborhood_fractions=(1.0,),
                num_calibration_partitions=256,
            ),
        )
    return groups
