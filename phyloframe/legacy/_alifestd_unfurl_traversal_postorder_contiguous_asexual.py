import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
    ancestor_ids: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal indices for contiguous, sorted phylogeny.

    Uses iterative depth-first search so that each subtree's nodes are
    contiguous in the result.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.
    """
    n = len(ancestor_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    ancestor_ids = ancestor_ids.astype(np.int64)

    # Count children per node
    child_count = np.zeros(n, dtype=np.int64)
    root_count = 0
    for i in range(n):
        if ancestor_ids[i] != i:
            child_count[ancestor_ids[i]] += 1
        else:
            root_count += 1

    # Build CSR-style children array
    child_start = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        child_start[i + 1] = child_start[i] + child_count[i]

    children_flat = np.empty(n, dtype=np.int64)
    insert_pos = child_start[:-1].copy()
    for i in range(n):
        p = ancestor_ids[i]
        if p != i:
            children_flat[insert_pos[p]] = i
            insert_pos[p] += 1

    # Collect roots and push in reverse order so first root is on top
    roots = np.empty(root_count, dtype=np.int64)
    ri = 0
    for i in range(n):
        if ancestor_ids[i] == i:
            roots[ri] = i
            ri += 1

    # Iterative DFS postorder traversal
    result = np.empty(n, dtype=np.int64)
    result_pos = 0

    stack = np.empty(n, dtype=np.int64)
    stack_top = 0
    expanded = np.zeros(n, dtype=np.bool_)

    for ri in range(root_count - 1, -1, -1):
        stack[stack_top] = roots[ri]
        stack_top += 1

    while stack_top > 0:
        node = stack[stack_top - 1]
        c_start = child_start[node]
        c_end = child_start[node + 1]

        if not expanded[node] and c_start < c_end:
            expanded[node] = True
            # Push children; ascending id order means highest-id on top
            for ci in range(c_start, c_end):
                stack[stack_top] = children_flat[ci]
                stack_top += 1
        else:
            stack_top -= 1
            result[result_pos] = node
            result_pos += 1

    return result


def alifestd_unfurl_traversal_postorder_contiguous_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List node indices in DFS postorder traversal order, with subtree
    contiguity.

    The provided dataframe must be asexual with contiguous ids and
    topologically sorted rows.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_has_contiguous_ids(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    if not alifestd_is_topologically_sorted(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
        phylogeny_df["ancestor_id"].to_numpy(),
    )
