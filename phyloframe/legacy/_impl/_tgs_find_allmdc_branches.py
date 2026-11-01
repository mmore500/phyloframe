import typing

import numpy as np

from .._alifestd_mark_max_descendant_origin_time_asexual import (
    _alifestd_mark_max_descendant_origin_time_asexual_fast_path,
)
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._TgsAllmdcBranches import TgsAllmdcBranches
from ._tgs_find_trait_mdcs import tgs_find_trait_mdcs


def tgs_find_allmdc_branches(
    *,
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    branch_lengths: np.ndarray,
    trait_masks: typing.Sequence[np.ndarray],
) -> TgsAllmdcBranches:
    """Implementation detail for `alifestd_tgs_allmdc_estimate_asexual`.

    Here, tgs stands for trait gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and no
    unifurcations. Returns its informative branches (positive length, so
    excluding roots and nan), each with its survival time: the time from
    its end node to the latest sampled leaf below it, during which a trait
    gained on the branch must persist to be seen at every leaf of its
    clade. See `tgs_find_trait_mdcs` for the per-trait branches.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    is_informative = np.asarray(branch_lengths, dtype=np.float64) > 0
    latest = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, origin_time
    )
    trait_units, trait_gains = [], []
    for trait_mask in trait_masks:
        units, gains = tgs_find_trait_mdcs(
            ancestor_ids=ancestor_ids,
            trait_mask=trait_mask,
            is_informative=is_informative,
        )
        trait_units.append(units)
        trait_gains.append(gains)
    return TgsAllmdcBranches(
        clade_sizes=_alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids)[
            is_informative
        ],
        branch_lengths=np.asarray(branch_lengths, dtype=np.float64)[
            is_informative
        ],
        survival_times=(latest - origin_time)[is_informative],
        origin_times=origin_time[is_informative],
        trait_units=trait_units,
        trait_gains=trait_gains,
    )
