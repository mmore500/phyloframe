import typing

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_csr_children_asexual import (
    _alifestd_mark_csr_children_asexual_fast_path,
)
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_mark_num_descendants_asexual import (
    _alifestd_mark_num_descendants_asexual_fast_path
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
    right-sibling representation.  Siblings are visited in descending id
    order (highest-id child processed first).

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
    csr_offsets: np.ndarray,
    csr_children: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal indices for contiguous, sorted phylogeny.

    Uses iterative depth-first search so that each subtree's nodes are
    contiguous in the result.  Siblings are visited in descending id order
    (highest-id child processed first).

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    csr_offsets : np.ndarray
        CSR offset array of length n.
    csr_children : np.ndarray
        Flat array of child ids, grouped by parent.
    num_children : np.ndarray
        Array of child counts per node.

    Returns
    -------
    np.ndarray
        Index array giving DFS postorder traversal order.
    """
    n = len(ancestor_ids)
    dtype = ancestor_ids.dtype
    if n == 0:
        return np.empty(0, dtype=dtype)

    # Iterative DFS postorder traversal
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
            c_start = csr_offsets[node]
            c_end = c_start + num_children[node]

            if not expanded[node] and c_start < c_end:
                expanded[node] = True
                # Push children; ascending id order means highest-id on top
                for ci in range(c_start, c_end):
                    stack[stack_top] = csr_children[ci]
                    stack_top += 1
            else:
                stack_top -= 1
                result[result_pos] = node
                result_pos += 1

    return result


@jit(nopython=True)
def _alifestd_unfurl_traversal_postorder_contiguous_asexual_asc_jit(
    ancestor_ids: np.ndarray,
    num_descendants: np.ndarray,
) -> np.ndarray:
    """Return DFS postorder traversal indices for contiguous, sorted phylogeny.

    Uses iterative depth-first search so that each subtree's nodes are
    contiguous in the result.  Siblings are visited in ascending id order
    (smallest-id child processed first).

    Parameters
    ----------
    ancestor_ids : np.ndarray
        Array of ancestor IDs, assumed contiguous (ids == row indices)
        and topologically sorted.
    csr_offsets : np.ndarray
        CSR offset array of length n.
    csr_children : np.ndarray
        Flat array of child ids, grouped by parent.
    num_children : np.ndarray
        Array of child counts per node.

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
    offset = np.empty(n, dtype=int)

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        result[root + num_descendants[root]] = root
        offset[root] = root

        for node in range(num_descendants[root]):
            node += root + 1
            ancestor = ancestor_ids[node]
            ancestor_offset = offset[ancestor]
            node_pos = ancestor_offset + num_descendants[node]
            result[node_pos] = node
            offset[ancestor] += num_descendants[node] + 1
            offset[node] = node_pos + 1

    return result


def alifestd_unfurl_traversal_postorder_contiguous_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    child_order: typing.Optional[typing.Literal["asc", "desc"]] = None,
) -> np.ndarray:
    """List node indices in DFS postorder traversal order, with subtree
    contiguity.

    The provided dataframe must be asexual with contiguous ids and
    topologically sorted rows.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Asexual phylogeny in alife standard format with contiguous ids
        and topologically sorted rows.
    mutate : bool, default False
        If True, allow modification of the input dataframe.
    child_order : {"asc", "desc", None}, default None
        Order in which siblings are visited when descending the tree.
        ``"asc"`` visits smallest-id child first, ``"desc"`` visits
        largest-id child first, and ``None`` uses an arbitrary
        (implementation-defined) order.
    """
    if child_order not in (None, "asc", "desc"):
        raise ValueError(
            f"child_order must be 'asc', 'desc', or None; got {child_order!r}",
        )

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

    if child_order != "desc":
        if "num_descendants" in phylogeny_df.columns:
            num_descendants = phylogeny_df["num_descendants"].to_numpy()
        else:
            num_descendants = _alifestd_mark_num_descendants_asexual_fast_path(
                ancestor_ids
            )
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_asc_jit(
            ancestor_ids,
            num_descendants,
        )
    elif (
        "first_child_id" in phylogeny_df.columns
        and "next_sibling_id" in phylogeny_df.columns
    ):
        assert child_order == "desc"
        first_child_ids = phylogeny_df["first_child_id"].to_numpy()
        next_sibling_ids = phylogeny_df["next_sibling_id"].to_numpy()
        return _alifestd_unfurl_traversal_postorder_contiguous_asexual_sibling_jit(
            ancestor_ids,
            first_child_ids,
            next_sibling_ids,
        )
    
    if "num_children" not in phylogeny_df.columns:
        num_children = _alifestd_mark_num_children_asexual_fast_path(
            ancestor_ids,
        )
    else:
        num_children = phylogeny_df["num_children"].to_numpy()
    if "csr_offsets" in phylogeny_df.columns:
        csr_offsets = phylogeny_df["csr_offsets"].to_numpy()
    else:
        csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
            ancestor_ids,
        )
    if "csr_children" in phylogeny_df.columns:
        csr_children = phylogeny_df["csr_children"].to_numpy()
    else:
        csr_children = _alifestd_mark_csr_children_asexual_fast_path(
            ancestor_ids,
            csr_offsets,
        )
    assert child_order == "desc"
    return _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
        ancestor_ids,
        csr_offsets,
        csr_children,
        num_children,
    )
