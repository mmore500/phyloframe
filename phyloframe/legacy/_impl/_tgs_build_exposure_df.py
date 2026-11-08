import numpy as np
import polars as pl
import polars.selectors as cs

from .._alifestd_mark_lineage_delta_asexual import (
    _alifestd_mark_lineage_delta_asexual_fast_path,
)
from .._alifestd_mark_max_descendant_origin_time_asexual import (
    _alifestd_mark_max_descendant_origin_time_asexual_fast_path,
)
from .._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)


def tgs_build_exposure_df(phylogeny_df: pl.DataFrame) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_pendantmdc_estimate_polars` and
    `alifestd_tgs_allmdc_estimate_polars`. Acronym tgs stands for trait
    gain/selection.

    Returns the exposure table of `phylogeny_df`, a phylogeny with
    contiguous, topologically sorted ids and no unifurcations, with one row
    per branch, i.e., per non-root node; see Input Schema and Output
    Schema.

    Raises ValueError if a non-root node's branch has zero, negative, or
    undefined (nan) length.

    Input Schema
    ------------
    'ancestor_id' : integer
        Parent's id (or own, for a root).
    'origin_time' : float
        Time of each node's origin.
    'trait{i}_state' : integer, per trait `i`
        Whether the node has trait `i` (0 or 1).

    Output Schema
    -------------
    'origin_time' : float
        Origin time of the branch's end node.
    'branch_length' : float
        Branch length.
    'survival_duration' : float
        Time from the end node to the latest leaf below it (0 for a leaf),
        for which a trait gained on the branch must persist to be seen at
        every leaf of its clade.
    'clade_size' : integer
        Number of leaves below the end node (1 for a leaf).
    'trait{i}_starts_absent' : boolean, per trait `i`
        Whether the branch started without the trait (i.e., its parent
        lacks it), so could gain it on the branch.
    'trait{i}_gain' : boolean, per trait `i`
        Whether the branch gained the trait.
    """
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    origin_time = phylogeny_df["origin_time"].to_numpy().astype(np.float64)
    branch_length = _alifestd_mark_lineage_delta_asexual_fast_path(
        ancestor_ids, origin_time
    )
    # a root has no parent, so no branch over which to gain a trait
    keep = ancestor_ids != np.arange(len(ancestor_ids))
    (bad_ids,) = np.nonzero(keep & ~(branch_length > 0))  # nan is bad
    if len(bad_ids):
        raise ValueError(
            "branch lengths must be positive, but non-root nodes (e.g., ids "
            f"{bad_ids[:5].tolist()}) have zero, negative, or undefined (nan) "
            "branch length; a parent with unknown (nan) origin_time, such "
            "as a root, leaves its children's branch lengths undefined",
        )
    latest_ot = _alifestd_mark_max_descendant_origin_time_asexual_fast_path(
        ancestor_ids, origin_time
    )
    columns = {
        "origin_time": origin_time[keep],
        "branch_length": branch_length[keep],
        "survival_duration": (latest_ot - origin_time)[keep],
        "clade_size": _alifestd_mark_num_leaves_asexual_fast_path(
            ancestor_ids
        )[keep],
    }
    num_traits = phylogeny_df.select(cs.matches(r"^trait\d+_state$")).width
    for i in range(num_traits):
        states = phylogeny_df[f"trait{i}_state"].to_numpy()
        deltas = _alifestd_mark_lineage_delta_asexual_fast_path(
            ancestor_ids, states
        )
        # a node's parent's state is its state minus its delta
        columns[f"trait{i}_starts_absent"] = (states - deltas == 0)[keep]
        columns[f"trait{i}_gain"] = (deltas == 1)[keep]
    return pl.DataFrame(columns)
