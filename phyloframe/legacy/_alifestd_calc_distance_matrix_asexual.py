import typing

import numpy as np
import pandas as pd

from .._auxlib._bit_length_numpy import bit_length_numpy
from .._auxlib._jit import jit
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


@jit(nopython=True)
def _build_euler_tour(ancestor_ids: np.ndarray) -> tuple:
    """Build an Euler tour of the forest defined by ancestor_ids.

    Returns (tour, tour_depth, first_occurrence) where:
    - tour: node ids in DFS visit order (length 2n - num_roots)
    - tour_depth: depth of each tour entry
    - first_occurrence: index of each node's first appearance in tour
    """
    n = len(ancestor_ids)

    # Build children adjacency in CSR format
    child_count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ancestor_ids[i] != i:
            child_count[ancestor_ids[i]] += 1

    child_start = np.empty(n + 1, dtype=np.int64)
    child_start[0] = 0
    for i in range(n):
        child_start[i + 1] = child_start[i] + child_count[i]

    children = np.empty(child_start[n], dtype=np.int64)
    fill = child_start[:n].copy()
    for i in range(n):
        if ancestor_ids[i] != i:
            p = ancestor_ids[i]
            children[fill[p]] = i
            fill[p] += 1

    # Euler tour via iterative DFS
    tour = np.empty(2 * n, dtype=np.int64)
    tour_depth = np.empty(2 * n, dtype=np.int64)
    first_occurrence = np.full(n, -1, dtype=np.int64)
    pos = 0

    stack_node = np.empty(n, dtype=np.int64)
    stack_ci = np.empty(n, dtype=np.int64)
    stack_depth = np.empty(n, dtype=np.int64)
    stack_top = 0

    for root in range(n):
        if ancestor_ids[root] != root:
            continue

        stack_node[0] = root
        stack_ci[0] = 0
        stack_depth[0] = 0
        stack_top = 1

        while stack_top > 0:
            idx = stack_top - 1
            node = stack_node[idx]
            ci = stack_ci[idx]
            depth = stack_depth[idx]

            if ci == 0:
                tour[pos] = node
                tour_depth[pos] = depth
                first_occurrence[node] = pos
                pos += 1

            start = child_start[node]
            count = child_count[node]

            if ci < count:
                stack_ci[idx] = ci + 1
                child = children[start + ci]
                stack_node[stack_top] = child
                stack_ci[stack_top] = 0
                stack_depth[stack_top] = depth + 1
                stack_top += 1
            else:
                stack_top -= 1
                if stack_top > 0:
                    tour[pos] = stack_node[stack_top - 1]
                    tour_depth[pos] = stack_depth[stack_top - 1]
                    pos += 1

    return tour[:pos], tour_depth[:pos], first_occurrence


@jit(nopython=True)
def _compute_root_ids(ancestor_ids: np.ndarray) -> np.ndarray:
    """Compute the root id for each node.

    Requires topologically sorted, contiguous ids.
    """
    n = len(ancestor_ids)
    root_ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        if ancestor_ids[i] == i:
            root_ids[i] = i
        else:
            root_ids[i] = root_ids[ancestor_ids[i]]
    return root_ids


def _build_sparse_table(tour_depth: np.ndarray) -> np.ndarray:
    """Build a sparse table for range minimum queries on tour_depth.

    Returns a 2-D array where sparse[k, i] is the index of the minimum
    value in tour_depth[i : i + 2**k].
    """
    m = len(tour_depth)
    if m == 0:
        return np.empty((0, 0), dtype=np.int64)

    K = m.bit_length()
    sparse = np.empty((K, m), dtype=np.int64)
    sparse[0] = np.arange(m, dtype=np.int64)

    for k in range(1, K):
        half = 1 << (k - 1)
        end = m - (1 << k) + 1
        if end <= 0:
            break
        left = sparse[k - 1, :end]
        right = sparse[k - 1, half : half + end]
        sparse[k, :end] = np.where(
            tour_depth[left] <= tour_depth[right], left, right
        )

    return sparse


def _alifestd_calc_distance_matrix_asexual_fast_path(
    ancestor_ids: np.ndarray,
    criterion_values: np.ndarray,
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Shared implementation detail for
    `alifestd_calc_distance_matrix_asexual` and
    `alifestd_calc_distance_matrix_asexual_polars`.

    Uses an Euler tour with a sparse table for O(1) LCA queries,
    giving O(n log n) preprocessing and O(n**2) total for all pairs.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        1-D int64 array of ancestor ids, indexed by organism id.
        Roots are self-referential (ancestor_ids[i] == i).
    criterion_values : np.ndarray
        1-D float64 array of criterion values, indexed by organism id.
    progress_wrap : callable, optional
        Wrapper for progress display (e.g., tqdm).

    Returns
    -------
    np.ndarray
        n x n float64 matrix of pairwise distances.  Entry [i, j] is NaN
        when organisms i and j share no common ancestor.
    """
    n = len(ancestor_ids)
    result = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return result
    np.fill_diagonal(result, 0.0)
    if n <= 1:
        return result

    # Preprocess: Euler tour + sparse table for O(1) LCA (O(n log n))
    tour, tour_depth, first_occ = _build_euler_tour(ancestor_ids)
    sparse = _build_sparse_table(tour_depth)
    root_ids = _compute_root_ids(ancestor_ids)

    # All upper-triangle pairs
    rows, cols = np.triu_indices(n, k=1)

    # Filter to same-tree pairs (cross-tree pairs stay NaN)
    same_tree = root_ids[rows] == root_ids[cols]
    valid_rows = rows[same_tree]
    valid_cols = cols[same_tree]

    if len(valid_rows) == 0:
        return result

    # Vectorized LCA via range minimum query on Euler tour depths
    left = np.minimum(first_occ[valid_rows], first_occ[valid_cols])
    right = np.maximum(first_occ[valid_rows], first_occ[valid_cols])
    length = right - left + 1
    k = bit_length_numpy(length) - 1
    half = np.int64(1) << k

    left_min = sparse[k, left]
    right_min = sparse[k, right - half + 1]
    lca_pos = np.where(
        tour_depth[left_min] <= tour_depth[right_min],
        left_min,
        right_min,
    )
    mrca_ids = tour[lca_pos]

    # Vectorized distance computation
    distances = (
        criterion_values[valid_rows]
        + criterion_values[valid_cols]
        - 2.0 * criterion_values[mrca_ids]
    )
    result[valid_rows, valid_cols] = distances
    result[valid_cols, valid_rows] = distances

    return result


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
        Computes the MRCA id matrix for all taxon pairs.
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

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    criterion_values = phylogeny_df[criterion].to_numpy().astype(np.float64)
    assert np.all(
        phylogeny_df["id"].to_numpy() == np.arange(len(phylogeny_df))
    )

    return _alifestd_calc_distance_matrix_asexual_fast_path(
        ancestor_ids, criterion_values, progress_wrap
    )
