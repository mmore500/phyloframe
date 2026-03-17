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
def _calc_distance_matrix_postorder_jit(
    ancestor_ids: np.ndarray,
    criterion_values: np.ndarray,
    csr_offsets: np.ndarray,
    csr_children: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Compute pairwise distance matrix via postorder traversal.

    Adapted from TreeSwift's distance_matrix approach: traverse the tree
    bottom-up, accumulating node-to-ancestor distances in flat arrays and
    filling in cross-subtree pairwise distances at each internal node.

    Avoids computing the full MRCA matrix, reducing time complexity from
    O(n^2 * depth) to O(n^2) and halving peak memory.
    """
    n = len(ancestor_ids)
    result = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return result

    # Edge lengths as criterion delta from parent; roots are
    # self-referential so their delta is naturally zero.
    edge_lengths = criterion_values - criterion_values[ancestor_ids]

    # Flat buffer tracking (node_id, distance_to_subtree_root) pairs.
    # Each node appears exactly once; total entries == n.
    buf_ids = np.empty(n, dtype=np.int64)
    buf_dists = np.empty(n, dtype=np.float64)
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
                    buf_dists[buf_pos] = 0.0
                    result[node, node] = 0.0
                    buf_pos += 1
                else:
                    # Internal node.
                    # 1) Add each child's edge length to its subtree
                    #    distances.
                    for ci in range(cs, ce):
                        child = csr_children[ci]
                        s = seg_start[child]
                        buf_dists[s : s + seg_count[child]] += edge_lengths[
                            child
                        ]

                    # 2) Cross-child pairwise distances.
                    for ci1 in range(cs, ce - 1):
                        child1 = csr_children[ci1]
                        s1 = seg_start[child1]
                        e1 = s1 + seg_count[child1]
                        for ci2 in range(ci1 + 1, ce):
                            child2 = csr_children[ci2]
                            s2 = seg_start[child2]
                            e2 = s2 + seg_count[child2]
                            ids2 = buf_ids[s2:e2]
                            dists2 = buf_dists[s2:e2]
                            for k1 in range(s1, e1):
                                id1 = buf_ids[k1]
                                cross = buf_dists[k1] + dists2
                                result[id1, ids2] = cross
                                result[ids2, id1] = cross

                    # 3) Distance from this node to all descendants.
                    desc_start = subtree_buf_start[node]
                    desc_ids = buf_ids[desc_start:buf_pos]
                    desc_dists = buf_dists[desc_start:buf_pos]
                    result[node, desc_ids] = desc_dists
                    result[desc_ids, node] = desc_dists

                    result[node, node] = 0.0

                    # 4) Add this node to the buffer and record its
                    #    combined segment.
                    buf_ids[buf_pos] = node
                    buf_dists[buf_pos] = 0.0
                    seg_start[node] = desc_start
                    seg_count[node] = (buf_pos - desc_start) + 1
                    buf_pos += 1

    return result


def _alifestd_calc_distance_matrix_asexual_fast_path(
    ancestor_ids: np.ndarray,
    criterion_values: np.ndarray,
    *,
    csr_offsets: np.ndarray,
    csr_children: np.ndarray,
    num_children: np.ndarray,
) -> np.ndarray:
    """Shared implementation detail for
    `alifestd_calc_distance_matrix_asexual` and
    `alifestd_calc_distance_matrix_asexual_polars`.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        1-D int64 array of ancestor ids, indexed by organism id.
        Roots are self-referential (ancestor_ids[i] == i).
    criterion_values : np.ndarray
        1-D float64 array of criterion values, indexed by organism id.
    csr_offsets : np.ndarray
        CSR offsets array.
    csr_children : np.ndarray
        Flat children array.
    num_children : np.ndarray
        Child counts array.

    Returns
    -------
    np.ndarray
        n x n float64 matrix of pairwise distances.  Entry [i, j] is NaN
        when organisms i and j share no common ancestor.
    """
    return _calc_distance_matrix_postorder_jit(
        ancestor_ids,
        criterion_values,
        csr_offsets,
        csr_children,
        num_children,
    )


def alifestd_calc_distance_matrix_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    criterion: str = "origin_time",
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Calculate pairwise distances between all taxa via their MRCAs.

    The distance between two taxa is computed as the sum of criterion
    differences between each taxon and their Most Recent Common Ancestor
    (MRCA):

        distance[i, j] = (criterion[i] - criterion[mrca])
                       + (criterion[j] - criterion[mrca])

    Taxa sharing no common ancestor will have distance NaN.

    Pass tqdm or equivalent as `progress_wrap` to display a progress bar.

    Input dataframe is not mutated by this operation unless `mutate` set True.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny in alife standard format.
    mutate : bool, default False
        If True, allows in-place modification of `phylogeny_df`.
    criterion : str, default "origin_time"
        Column name used to measure distance between taxa and their MRCA.
    progress_wrap : callable, optional
        Wrapper for progress display (e.g., tqdm).

    Returns
    -------
    np.ndarray
        n x n float64 matrix of pairwise distances.  Entry [i, j] is NaN
        when organisms i and j share no common ancestor.

    See Also
    --------
    alifestd_calc_mrca_id_matrix_asexual :
        Computes the MRCA id matrix used internally by this function.
    alifestd_find_pair_distance_asexual :
        Computes distance for a single pair of taxa.
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
            phylogeny_df,
            mutate=True,
        )
    if "csr_offsets" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_csr_offsets_asexual(
            phylogeny_df,
            mutate=True,
        )
    if "csr_children" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_csr_children_asexual(
            phylogeny_df,
            mutate=True,
        )

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    criterion_values = phylogeny_df[criterion].to_numpy().astype(np.float64)
    num_children = phylogeny_df["num_children"].to_numpy()
    csr_offsets = phylogeny_df["csr_offsets"].to_numpy().astype(np.int64)
    csr_children = phylogeny_df["csr_children"].to_numpy().astype(np.int64)

    return _alifestd_calc_distance_matrix_asexual_fast_path(
        ancestor_ids,
        criterion_values,
        csr_offsets=csr_offsets,
        csr_children=csr_children,
        num_children=num_children,
    )
