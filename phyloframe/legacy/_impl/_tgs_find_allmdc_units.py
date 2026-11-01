import typing

import numpy as np

from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._TgsAllmdcUnits import TgsAllmdcUnits
from ._tgs_calc_clade_size_bounds import tgs_calc_clade_size_bounds
from ._tgs_find_trait_mdcs import tgs_find_trait_mdcs


def tgs_find_allmdc_units(
    *,
    ancestor_ids: np.ndarray,
    branch_lengths: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
    groups_per_doubling: int,
) -> TgsAllmdcUnits:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and no
    unifurcations. Groups informative branches (positive length, so
    excluding roots and nan) by the number of leaves they subtend: single
    leaves first, then log-scaled clade-size groups (see
    `tgs_calc_clade_size_bounds`) up to the largest clade.
    """
    is_informative = np.asarray(branch_lengths, dtype=np.float64) > 0
    informative = np.flatnonzero(is_informative)
    num_leaves = _alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids)[
        informative
    ]
    bounds = tgs_calc_clade_size_bounds(
        max_clade_size=int(num_leaves.max(initial=1)),
        groups_per_doubling=groups_per_doubling,
    )
    # group 0 holds single leaves; group j > 0, clades of bounds[j - 1] to
    # bounds[j] - 1 leaves
    unit_groups = np.searchsorted(bounds, num_leaves, side="right")
    group_bounds = [(1, 1)] + [
        (int(lower), int(upper) - 1)
        for lower, upper in zip(bounds[:-1], bounds[1:])
    ]
    num_groups = len(group_bounds)
    # position of each informative branch within its group
    positions = np.zeros(len(informative), dtype=np.int64)
    for j in range(num_groups):
        is_member = unit_groups == j
        positions[is_member] = np.arange(is_member.sum())

    trait_units = [[] for __ in range(num_groups)]
    trait_gains = [[] for __ in range(num_groups)]
    for trait_mask in trait_masks:
        units, gains = tgs_find_trait_mdcs(
            ancestor_ids=ancestor_ids,
            trait_mask=trait_mask,
            is_informative=is_informative,
        )
        for j in range(num_groups):
            trait_units[j].append(positions[units[unit_groups[units] == j]])
            trait_gains[j].append(positions[gains[unit_groups[gains] == j]])

    return TgsAllmdcUnits(
        group_nodes=[informative[unit_groups == j] for j in range(num_groups)],
        group_bounds=group_bounds,
        trait_units=trait_units,
        trait_gains=trait_gains,
    )
