import numpy as np
import pandas as pd

from ._alifestd_find_leaf_ids import (
    _alifestd_find_leaf_ids_asexual_fast_path,
    alifestd_find_leaf_ids,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_has_multiple_roots import alifestd_has_multiple_roots
from ._alifestd_is_strictly_bifurcating_asexual import (
    alifestd_is_strictly_bifurcating_asexual,
)
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_leaves import alifestd_mark_leaves
from ._alifestd_mark_num_descendants_asexual import (
    _alifestd_mark_num_descendants_asexual_fast_path,
    alifestd_mark_num_descendants_asexual,
)
from ._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._alifestd_mark_num_preceding_leaves_asexual import (
    _alifestd_mark_num_preceding_leaves_asexual_fast_path,
    alifestd_mark_num_preceding_leaves_asexual,
)
from ._alifestd_mark_right_child_asexual import (
    _alifestd_mark_right_child_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col
from ._alifestd_unfurl_traversal_semiorder_asexual import (
    _alifestd_unfurl_traversal_semiorder_asexual_fast_path,
    _alifestd_unfurl_traversal_semiorder_asexual_slow_path,
)


def _alifestd_unfurl_traversal_inorder_asexual_fast_path(
    ancestor_ids: np.ndarray,
    num_leaves: np.ndarray,
    right_children: np.ndarray,
) -> np.ndarray:
    """Streamlined inorder traversal for contiguous, topologically sorted,
    strictly bifurcating trees.

    Calls JIT fast paths directly, avoiding redundant validation checks
    in intermediate functions.
    """
    # is_right_child: True when this node is the right child of its parent
    is_right_child = (
        right_children[ancestor_ids] == np.arange(len(ancestor_ids))
    ) & (ancestor_ids != np.arange(len(ancestor_ids)))

    num_preceding_leaves = (
        _alifestd_mark_num_preceding_leaves_asexual_fast_path(
            ancestor_ids, num_leaves, is_right_child
        )
    )
    num_descendants = _alifestd_mark_num_descendants_asexual_fast_path(
        ancestor_ids,
    )

    leaf_ids = _alifestd_find_leaf_ids_asexual_fast_path(ancestor_ids)
    sorted_leaf_ids = np.empty_like(leaf_ids)
    sorted_leaf_ids[num_preceding_leaves[leaf_ids]] = leaf_ids

    return _alifestd_unfurl_traversal_semiorder_asexual_fast_path(
        ancestor_ids, num_descendants, sorted_leaf_ids
    )


def alifestd_unfurl_traversal_inorder_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List `id` values in semiorder traversal order, with left children
    visited first.

    The provided dataframe must be asexual and strictly bifurcating.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if alifestd_has_multiple_roots(phylogeny_df):
        raise ValueError(
            "Phylogeny must have a single root for inorder traversal."
        )
    if not alifestd_is_strictly_bifurcating_asexual(phylogeny_df):
        raise ValueError(
            "Phylogeny must be strictly bifurcating for inorder traversal.",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(
        phylogeny_df
    ) and alifestd_is_topologically_sorted(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        num_leaves = (
            phylogeny_df["num_leaves"].to_numpy()
            if "num_leaves" in phylogeny_df.columns
            else _alifestd_mark_num_leaves_asexual_fast_path(ancestor_ids)
        )
        right_children = (
            phylogeny_df["right_child_id"].to_numpy()
            if "right_child_id" in phylogeny_df.columns
            else _alifestd_mark_right_child_asexual_fast_path(ancestor_ids)
        )
        return _alifestd_unfurl_traversal_inorder_asexual_fast_path(
            ancestor_ids,
            num_leaves,
            right_children,
        )

    if "num_preceding_leaves" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_num_preceding_leaves_asexual(
            phylogeny_df,
            mutate=True,
        )

    if (
        "ancestor_id" in phylogeny_df.columns
        and alifestd_has_contiguous_ids(
            phylogeny_df,
        )
        and alifestd_is_topologically_sorted(phylogeny_df)
    ):
        if "num_descendants" not in phylogeny_df.columns:
            phylogeny_df = alifestd_mark_num_descendants_asexual(
                phylogeny_df, mutate=True
            )
        leaf_positions = phylogeny_df["num_preceding_leaves"].to_numpy()

        leaf_ids = alifestd_find_leaf_ids(phylogeny_df)
        sorted_leaf_ids = np.empty_like(leaf_ids)
        sorted_leaf_ids[leaf_positions[leaf_ids]] = leaf_ids

        return _alifestd_unfurl_traversal_semiorder_asexual_fast_path(
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df["num_descendants"].to_numpy(),
            sorted_leaf_ids,
        )
    else:
        if "is_leaf" not in phylogeny_df.columns:
            phylogeny_df = alifestd_mark_leaves(phylogeny_df, mutate=True)

        sorted_phylogeny_df = phylogeny_df.sort_values("num_preceding_leaves")
        sorted_leaf_ids = sorted_phylogeny_df.loc[
            sorted_phylogeny_df["is_leaf"], "id"
        ]

        return _alifestd_unfurl_traversal_semiorder_asexual_slow_path(
            phylogeny_df,
            sorted_leaf_ids,
        )
