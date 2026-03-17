import typing

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_csr_children_asexual import (
    alifestd_mark_csr_children_asexual,
)
from ._alifestd_mark_csr_offsets_asexual import (
    alifestd_mark_csr_offsets_asexual,
)
from ._alifestd_mark_num_children_asexual import (
    alifestd_mark_num_children_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _calc_mrca_id_matrix_postorder_jit(
    ancestor_ids: np.ndarray,
    csr_offsets: np.ndarray,
    csr_children: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Compute MRCA id matrix via postorder traversal.

    Adapted from the distance matrix postorder approach: traverse the tree
    bottom-up, tracking descendant ids in flat arrays and filling in
    cross-subtree MRCA pairs at each internal node.

    Avoids the depth-driven O(n^2 * depth) algorithm, reducing time
    complexity to O(n^2).
    """
    n = len(ancestor_ids)
    result = -np.ones((n, n), dtype=np.int64)
    if n == 0:
        return result

    # Flat buffer tracking descendant ids for each subtree.
    # Each node appears exactly once; total entries == n.
    buf_ids = np.empty(n, dtype=np.int64)
    buf_pos = np.int64(0)

    # Per-node segment tracking: where this node's subtree taxa live
    # in the buffer.
    seg_start = np.empty(n, dtype=np.int64)
    seg_count = np.empty(n, dtype=np.int64)

    # Record where a node's descendant entries begin (set on expansion).
    subtree_buf_start = np.empty(n, dtype=np.int64)

    # Iterative DFS postorder traversal.
    stack = np.empty(n, dtype=np.int64)
    stack_top = np.int64(0)
    expanded = np.zeros(n, dtype=np.bool_)

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        stack[0] = root
        stack_top = 1

        while stack_top > 0:
            node = stack[stack_top - 1]
            cs = csr_offsets[node]
            ce = csr_offsets[node] + num_children[node]

            if not expanded[node] and cs < ce:
                expanded[node] = True
                subtree_buf_start[node] = buf_pos
                n_children = ce - cs
                stack[stack_top : stack_top + n_children] = csr_children[cs:ce]
                stack_top += n_children
            else:
                stack_top -= 1

                if cs == ce:
                    # Leaf node: single buffer entry.
                    seg_start[node] = buf_pos
                    seg_count[node] = 1
                    buf_ids[buf_pos] = node
                    result[node, node] = node
                    buf_pos += 1
                else:
                    # Internal node.
                    # 1) Cross-child pairwise MRCAs: MRCA of any two
                    #    nodes from different child subtrees is this node.
                    for ci1 in range(cs, ce - 1):
                        child1 = csr_children[ci1]
                        s1 = seg_start[child1]
                        e1 = s1 + seg_count[child1]
                        for ci2 in range(ci1 + 1, ce):
                            child2 = csr_children[ci2]
                            s2 = seg_start[child2]
                            e2 = s2 + seg_count[child2]
                            ids2 = buf_ids[s2:e2]
                            for k1 in range(s1, e1):
                                id1 = buf_ids[k1]
                                result[id1, ids2] = node
                                result[ids2, id1] = node

                    # 2) MRCA of this node with all descendants is
                    #    this node.
                    desc_start = subtree_buf_start[node]
                    desc_ids = buf_ids[desc_start:buf_pos]
                    result[node, desc_ids] = node
                    result[desc_ids, node] = node

                    result[node, node] = node

                    # 3) Add this node to the buffer and record its
                    #    combined segment.
                    buf_ids[buf_pos] = node
                    seg_start[node] = desc_start
                    seg_count[node] = (buf_pos - desc_start) + 1
                    buf_pos += 1

    return result


def _alifestd_calc_mrca_id_matrix_asexual_fast_path(
    ancestor_ids: np.ndarray,
    *,
    csr_offsets: np.ndarray,
    csr_children: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Shared implementation detail for
    `alifestd_calc_mrca_id_matrix_asexual` and
    `alifestd_calc_mrca_id_matrix_asexual_polars`.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        1-D int64 array of ancestor ids, indexed by organism id.
        Roots are self-referential (ancestor_ids[i] == i).
    csr_offsets : np.ndarray
        CSR offsets array.
    csr_children : np.ndarray
        Flat children array.
    num_children : np.ndarray
        Child counts array.

    Returns
    -------
    np.ndarray
        n x n int64 matrix of MRCA ids.  Entry [i, j] is -1 when
        organisms i and j share no common ancestor.
    """
    return _calc_mrca_id_matrix_postorder_jit(
        ancestor_ids,
        csr_offsets,
        csr_children,
        num_children,
    )


def alifestd_calc_mrca_id_matrix_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Calculate the Most Recent Common Ancestor (MRCA) taxon id for each pair
    of taxa.

    Taxa sharing no common ancestor will have MRCA id -1.

    Pass tqdm or equivalent as `progress_wrap` to display a progress bar.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    """

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if not alifestd_is_working_format_asexual(phylogeny_df, mutate=True):
        raise NotImplementedError(
            "current implementation requires phylogeny_df in working format",
        )

    assert np.all(
        phylogeny_df["id"].to_numpy() == np.arange(len(phylogeny_df))
    )

    if "num_children" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_num_children_asexual(
            phylogeny_df, mutate=True,
        )
    if "csr_offsets" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_csr_offsets_asexual(
            phylogeny_df, mutate=True,
        )
    if "csr_children" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_csr_children_asexual(
            phylogeny_df, mutate=True,
        )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    num_children = phylogeny_df["num_children"].to_numpy()
    csr_offsets = phylogeny_df["csr_offsets"].to_numpy().astype(np.int64)
    csr_children = phylogeny_df["csr_children"].to_numpy().astype(np.int64)

    return _alifestd_calc_mrca_id_matrix_asexual_fast_path(
        ancestor_ids,
        csr_offsets=csr_offsets,
        csr_children=csr_children,
        num_children=num_children,
    )
