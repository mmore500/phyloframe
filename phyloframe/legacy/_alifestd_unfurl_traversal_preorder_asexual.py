import numpy as np
import pandas as pd

from .._auxlib._build_children_csr import build_children_csr
from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
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
    if n == 0:
        return np.empty(0, dtype=np.int64)

    result = np.empty(n, dtype=np.int64)
    result_pos = 0

    stack = np.empty(n, dtype=np.int64)
    stack_top = 0

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        stack[0] = root
        stack_top = 1

        while stack_top > 0:
            stack_top -= 1
            node = stack[stack_top]

            result[result_pos] = node
            result_pos += 1

            # Push siblings in reverse order (last sibling first onto stack)
            # so that first child ends up on top
            first_child = first_child_ids[node]
            if first_child != node:  # has children
                # Collect siblings to push in reverse
                sibs_top = 0
                sib = first_child
                while True:
                    stack[stack_top + sibs_top] = sib
                    sibs_top += 1
                    nxt = next_sibling_ids[sib]
                    if nxt == sib:
                        break
                    sib = nxt
                # Reverse so first child is on top of stack
                for i in range(sibs_top // 2):
                    j = sibs_top - 1 - i
                    tmp = stack[stack_top + i]
                    stack[stack_top + i] = stack[stack_top + j]
                    stack[stack_top + j] = tmp
                stack_top += sibs_top

    return result


@jit(nopython=True)
def _alifestd_unfurl_traversal_preorder_asexual_jit(
    ancestor_ids: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Return DFS preorder traversal indices for contiguous, sorted phylogeny.

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
        Index array giving DFS preorder traversal order.
    """
    n = len(ancestor_ids)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    ancestor_ids = ancestor_ids.astype(np.int64)
    child_start, children_flat = build_children_csr(
        ancestor_ids,
        num_children.astype(np.int64),
    )

    # Iterative DFS preorder traversal
    result = np.empty(n, dtype=np.int64)
    result_pos = 0

    stack = np.empty(n, dtype=np.int64)
    stack_top = 0

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        stack[0] = root
        stack_top = 1

        while stack_top > 0:
            stack_top -= 1
            node = stack[stack_top]

            # Emit node immediately (preorder)
            result[result_pos] = node
            result_pos += 1

            # Push children in reverse order so smallest id is on top
            c_start = child_start[node]
            c_end = child_start[node + 1]
            for ci in range(c_end - 1, c_start - 1, -1):
                stack[stack_top] = children_flat[ci]
                stack_top += 1

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
        if (
            "first_child_id" in phylogeny_df.columns
            and "next_sibling_id" in phylogeny_df.columns
        ):
            return _alifestd_unfurl_traversal_preorder_asexual_sibling_jit(
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
        return _alifestd_unfurl_traversal_preorder_asexual_jit(
            ancestor_ids,
            num_children,
        )
    else:
        return _alifestd_unfurl_traversal_preorder_asexual_slow_path(
            phylogeny_df,
        )
