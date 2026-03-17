import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_node_depth_asexual import (
    _alifestd_calc_node_depth_asexual_contiguous,
    alifestd_mark_node_depth_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_unfurl_traversal_postorder_asexual_sibling_jit(
    ancestor_ids: np.ndarray,
    first_child_ids: np.ndarray,
    next_sibling_ids: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal using first-child/next-sibling pointers.

    Avoids computing node depths and lexsort by doing O(n) DFS directly.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    first_child_ids : np.ndarray
        Array where first_child_ids[i] is the smallest-id child of i,
        or i itself if i is a leaf.
    next_sibling_ids : np.ndarray
        Array where next_sibling_ids[i] is the next-highest id sharing
        the same parent, or i itself if no such sibling.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.
    """
    n = len(ancestor_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    result = np.empty(n, dtype=np.int64)
    result_pos = 0

    stack = np.empty(n, dtype=np.int64)
    stack_top = 0
    expanded = np.zeros(n, dtype=np.bool_)

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        stack[0] = root
        stack_top = 1

        while stack_top > 0:
            node = stack[stack_top - 1]
            first_child = first_child_ids[node]

            if not expanded[node] and first_child != node:
                expanded[node] = True
                sib = first_child
                while True:
                    stack[stack_top] = sib
                    stack_top += 1
                    nxt = next_sibling_ids[sib]
                    if nxt == sib:
                        break
                    sib = nxt
            else:
                stack_top -= 1
                result[result_pos] = node
                result_pos += 1

    return result


def _alifestd_unfurl_traversal_postorder_asexual_fast_path(
    ancestor_ids: np.ndarray,
    node_depths: np.ndarray,
) -> np.ndarray:
    """Return postorder traversal indices for contiguous, sorted phylogeny.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.

    Returns
    -------
    np.ndarray
        Index array giving postorder traversal order.
    """
    return np.lexsort((ancestor_ids, node_depths))[::-1]


def _alifestd_unfurl_traversal_postorder_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
) -> np.ndarray:
    """Implementation detail for `alifestd_unfurl_traversal_postorder_asexual`.

    Handles non-contiguous ids using pandas indexing.
    """
    phylogeny_df = alifestd_mark_node_depth_asexual(phylogeny_df, mutate=True)
    postorder_index = np.lexsort(
        (phylogeny_df["ancestor_id"], phylogeny_df["node_depth"]),
    )[::-1]
    id_loc = phylogeny_df.columns.get_loc("id")
    return phylogeny_df.iloc[postorder_index, id_loc].to_numpy()


def alifestd_unfurl_traversal_postorder_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List `id` values in postorder traversal order.

    The provided dataframe must be asexual.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(
        phylogeny_df,
    ) and alifestd_is_topologically_sorted(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        if (
            "first_child_id" in phylogeny_df.columns
            and "next_sibling_id" in phylogeny_df.columns
        ):
            return _alifestd_unfurl_traversal_postorder_asexual_sibling_jit(
                ancestor_ids.astype(np.int64),
                phylogeny_df["first_child_id"].to_numpy().astype(np.int64),
                phylogeny_df["next_sibling_id"].to_numpy().astype(np.int64),
            )
        if "node_depth" not in phylogeny_df.columns:
            node_depths = _alifestd_calc_node_depth_asexual_contiguous(
                ancestor_ids,
            )
        else:
            node_depths = phylogeny_df["node_depth"].to_numpy()
        return _alifestd_unfurl_traversal_postorder_asexual_fast_path(
            ancestor_ids,
            node_depths,
        )
    else:
        return _alifestd_unfurl_traversal_postorder_asexual_slow_path(
            phylogeny_df,
        )
