import typing

import numpy as np
import polars as pl

from .._alifestd_mark_lineage_delta_asexual import (
    _alifestd_mark_lineage_delta_asexual_fast_path,
)
from .._alifestd_mark_max_descendant_origin_time_asexual import (
    _alifestd_mark_max_descendant_origin_time_asexual_fast_path,
)
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)


def tgs_calc_exposure_df(
    *,
    ancestor_ids: np.ndarray,
    origin_time: np.ndarray,
    trait_states: typing.Sequence[np.ndarray],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Takes a phylogeny with contiguous, topologically sorted ids and no
    unifurcations, and each trait's state (0 or 1) at every node. Returns
    one row per branch, i.e., per non-root node, with columns

    - `origin_time`, of the branch's end node;
    - `branch_length`;
    - `survival_duration`, from the end node to the latest leaf below it (0 for
      a leaf), for which a trait gained on the branch must persist to be
      seen at every leaf of its clade;
    - `clade_size`, the number of leaves below (1 for a leaf); and
    - per trait `i`, `trait{i}_starts_absent`, whether the branch started
      without the trait (i.e., its parent lacks it), so could gain it on the
      branch, and `trait{i}_gain`, whether it gained it.

    Raises ValueError if a non-root node's branch has zero, negative, or
    undefined (nan) length.
    """
    ancestor_ids = np.asarray(ancestor_ids, dtype=np.int64)
    origin_time = np.asarray(origin_time, dtype=np.float64)
    branch_length = _alifestd_mark_lineage_delta_asexual_fast_path(
        ancestor_ids, origin_time
    )
    keep = ancestor_ids != np.arange(len(ancestor_ids))  # excludes roots
    (bad_ids,) = np.nonzero(keep & ~(branch_length > 0))  # nan is bad
    if len(bad_ids):
        raise ValueError(
            "branch lengths must be positive, but non-root nodes (e.g., ids "
            f"{bad_ids[:5].tolist()}) have zero, negative, or undefined (nan) "
            "branch length; a parent with unknown (nan) origin_time, such "
            "as a root, leaves its children's branch lengths undefined",
        )
    latest = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, origin_time
    )
    columns = {
        "origin_time": origin_time[keep],
        "branch_length": branch_length[keep],
        "survival_duration": (latest - origin_time)[keep],
        "clade_size": _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids
        )[keep],
    }
    for i, states in enumerate(trait_states):
        deltas = _alifestd_mark_lineage_delta_asexual_fast_path(
            ancestor_ids, states
        )
        # a node's parent's state is its state minus its delta
        columns[f"trait{i}_starts_absent"] = (states - deltas == 0)[keep]
        columns[f"trait{i}_gain"] = (deltas == 1)[keep]
    return pl.DataFrame(columns)
