import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_num_descendants_asexual import (
    _alifestd_mark_num_descendants_asexual_fast_path,
)
from ._alifestd_topological_sort import alifestd_topological_sort
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_unfurl_traversal_preorder_asexual_sibling_jit(
    ancestor_ids: np.ndarray,
    first_child_ids: np.ndarray,
    next_sibling_ids: np.ndarray,
) -> np.ndarray:
    """Return DFS preorder traversal using first-child/next-sibling pointers.

    Avoids CSR construction by navigating the tree via left-child
    right-sibling representation.

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
        Index array giving DFS preorder traversal order.
    """
    n = len(ancestor_ids)
    dtype = ancestor_ids.dtype
    if n == 0:
        return np.empty(0, dtype=dtype)

    result = np.empty(n, dtype=dtype)
    result_pos = 0

    # Stack stores the *next* sibling to visit at each open level (cursor).
    # Cursor equal to its own id signals "no more siblings" — pop on entry.
    stack = np.empty(n, dtype=dtype)
    stack_top = 0

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        result[result_pos] = root
        result_pos += 1
        first_child = first_child_ids[root]
        if first_child != root:
            stack[0] = first_child
            stack_top = 1
            while stack_top > 0:
                node = stack[stack_top - 1]
                nxt = next_sibling_ids[node]
                if nxt == node:
                    stack_top -= 1  # no more siblings at this level
                else:
                    stack[stack_top - 1] = nxt
                result[result_pos] = node
                result_pos += 1
                fc = first_child_ids[node]
                if fc != node:
                    stack[stack_top] = fc
                    stack_top += 1

    return result


@jit(nopython=True)
def _alifestd_unfurl_traversal_preorder_asexual_jit(
    ancestor_ids: np.ndarray,
    num_descendants: np.ndarray,
) -> np.ndarray:
    """Return DFS preorder traversal indices for contiguous, sorted phylogeny.

    Uses subtree-size offsets to write each node directly to its final
    position in a single forward sweep, yielding contiguous subtrees.
    Siblings are visited in ascending id order (smallest-id child first).

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    num_descendants : np.ndarray
        Number of descendants (excluding self) for each node.

    Returns
    -------
    np.ndarray
        Index array giving DFS preorder traversal order.
    """
    n = len(ancestor_ids)
    dtype = ancestor_ids.dtype
    if n == 0:
        return np.empty(0, dtype=dtype)

    # Offset array uses the input id dtype: narrower dtypes (e.g., int32)
    # halve bandwidth on the random-access offset array vs int64, while
    # still safely indexing positions in [0, n).
    result = np.empty(n, dtype=dtype)
    offset = np.empty(n, dtype=dtype)
    root_pos = dtype.type(0)

    for node in range(n):
        ancestor = ancestor_ids[node]
        nd = num_descendants[node]
        if ancestor == node:
            start = root_pos
            root_pos += nd + 1
        else:
            start = offset[ancestor]
            offset[ancestor] = start + nd + 1
        result[start] = node
        offset[node] = start + 1

    return result


def _alifestd_unfurl_traversal_preorder_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
) -> np.ndarray:
    """Implementation detail for `alifestd_unfurl_traversal_preorder_asexual`.

    Handles non-contiguous ids using pandas indexing.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_topologically_sorted(phylogeny_df):
        phylogeny_df = alifestd_topological_sort(phylogeny_df, mutate=True)

    # Build children mapping using id values
    children_of = {}
    for _, row in phylogeny_df.iterrows():
        node_id = row["id"]
        ancestor_id = row["ancestor_id"]
        if node_id != ancestor_id:
            children_of.setdefault(ancestor_id, []).append(node_id)

    # Sort children by ascending id
    for parent in children_of:
        children_of[parent].sort()

    # Find roots (ancestor_id == id)
    roots = phylogeny_df.loc[
        phylogeny_df["id"] == phylogeny_df["ancestor_id"], "id"
    ].tolist()

    # Iterative DFS preorder
    result = []
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        result.append(node)
        for child in reversed(children_of.get(node, [])):
            stack.append(child)

    return np.array(result, dtype=phylogeny_df["id"].dtype)


def alifestd_unfurl_traversal_preorder_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List `id` values in DFS preorder traversal order.

    The provided dataframe must be asexual.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(
        phylogeny_df,
    ) and alifestd_is_topologically_sorted(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        if "num_descendants" in phylogeny_df.columns:
            num_descendants = phylogeny_df["num_descendants"].to_numpy()
        else:
            num_descendants = _alifestd_mark_num_descendants_asexual_fast_path(
                ancestor_ids,
            )
        return _alifestd_unfurl_traversal_preorder_asexual_jit(
            ancestor_ids,
            num_descendants,
        )
    else:
        return _alifestd_unfurl_traversal_preorder_asexual_slow_path(
            phylogeny_df,
        )
