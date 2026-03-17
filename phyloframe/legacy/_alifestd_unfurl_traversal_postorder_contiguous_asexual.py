import numpy as np
import pandas as pd

from .._auxlib._build_children_csr import build_children_csr
from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _alifestd_unfurl_traversal_postorder_contiguous_asexual_sibling_jit(
    ancestor_ids: np.ndarray,
    first_child_ids: np.ndarray,
    next_sibling_ids: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal using first-child/next-sibling pointers.

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
        Index array giving DFS postorder traversal order.
    """
    n = len(ancestor_ids)
    dtype = ancestor_ids.dtype
    if n == 0:
        return np.empty(0, dtype=dtype)

    result = np.empty(n, dtype=dtype)
    result_pos = 0

    stack = np.empty(n, dtype=dtype)
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
                # Push children; collect then push so first child on top
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


@jit(nopython=True)
def _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
    ancestor_ids: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal indices for contiguous, sorted phylogeny.

    Uses iterative depth-first search so that each subtree's nodes are
    contiguous in the result.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    num_children : np.ndarray
        Array of child counts per node.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.
    """
    n = len(ancestor_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    ancestor_ids = ancestor_ids.astype(np.int64)
    child_start, children_flat = build_children_csr(
        ancestor_ids,
        num_children.astype(np.int64),
    )

    # Iterative DFS postorder traversal
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

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    if (
        "first_child_id" in phylogeny_df.columns
        and "next_sibling_id" in phylogeny_df.columns
    ):
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_sibling_jit(
            ancestor_ids.astype(np.int64),
            phylogeny_df["first_child_id"].to_numpy().astype(np.int64),
            phylogeny_df["next_sibling_id"].to_numpy().astype(np.int64),
        )
    if "num_children" not in phylogeny_df.columns:
        num_children = _alifestd_mark_num_children_asexual_fast_path(
            ancestor_ids,
        )
    else:
        num_children = phylogeny_df["num_children"].to_numpy()
    return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
        ancestor_ids,
        num_children,
    )
