import typing

import numpy as np

from phyloframe.legacy._impl import TgsAllmdcBranches


def simulate_allmdc_branches(
    num_branches: int = 40_000, seed: int = 1, num_neutral: int = 2
) -> TgsAllmdcBranches:
    """Half leaves (survival time 0), half inner branches of clades of 2 to
    9 leaves, every branch
    starting without each trait and gaining it under the allmdc model.

    Trait 0 has twice the neutral gain rate and selection -0.05; the
    `num_neutral` traits after it are neutral.
    """
    rng = np.random.default_rng(seed)
    is_leaf = np.arange(num_branches) % 2 == 0
    clade_sizes = np.where(is_leaf, 1, rng.integers(2, 10, num_branches))
    lengths = rng.uniform(0.5, 60, num_branches)
    survival = np.where(is_leaf, 0.0, rng.uniform(0.5, 60, num_branches))
    trait_gains: typing.List[np.ndarray] = []
    for gain, selection in [(2e-3, -0.05)] + [(1e-3, 0.0)] * num_neutral:
        if selection:
            exposure = np.expm1(selection * lengths) / selection
        else:
            exposure = lengths
        p = gain * exposure * np.exp(selection * survival)
        trait_gains.append(np.flatnonzero(rng.random(num_branches) < p))
    return TgsAllmdcBranches(
        clade_sizes=clade_sizes,
        branch_lengths=lengths,
        survival_times=survival,
        origin_times=lengths + survival + rng.uniform(0, 10, num_branches),
        trait_units=[np.arange(num_branches)] * (1 + num_neutral),
        trait_gains=trait_gains,
    )
