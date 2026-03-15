import typing

import numpy as np
import pandas as pd

from ._alifestd_calc_mrca_id_matrix_asexual import (
    _alifestd_calc_mrca_id_matrix_asexual_fast_path,
)
from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_node_depth_asexual import alifestd_mark_node_depth_asexual
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_calc_distance_matrix_asexual_fast_path(
    ancestor_ids: np.ndarray,
    node_depths: np.ndarray,
    criterion_values: np.ndarray,
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Shared implementation detail for
    `alifestd_calc_distance_matrix_asexual` and
    `alifestd_calc_distance_matrix_asexual_polars`.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        1-D int64 array of ancestor ids, indexed by organism id.
        Roots are self-referential (ancestor_ids[i] == i).
    node_depths : np.ndarray
        1-D int64 array of node depths, indexed by organism id.
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
    mrca_matrix = _alifestd_calc_mrca_id_matrix_asexual_fast_path(
        ancestor_ids, node_depths, progress_wrap
    )

    n = len(ancestor_ids)
    result = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0:
        return result

    valid_mask = mrca_matrix != -1
    safe_mrca = np.where(valid_mask, mrca_matrix, 0)

    mrca_criterion = criterion_values[safe_mrca]
    row_criterion = criterion_values[:, None]
    col_criterion = criterion_values[None, :]

    distances = row_criterion + col_criterion - 2.0 * mrca_criterion
    result[valid_mask] = distances[valid_mask]

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

    phylogeny_df = alifestd_mark_node_depth_asexual(phylogeny_df, mutate=True)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    node_depths = phylogeny_df["node_depth"].to_numpy().astype(np.int64)
    criterion_values = phylogeny_df[criterion].to_numpy().astype(np.float64)
    assert np.all(
        phylogeny_df["id"].to_numpy() == np.arange(len(phylogeny_df))
    )

    return _alifestd_calc_distance_matrix_asexual_fast_path(
        ancestor_ids, node_depths, criterion_values, progress_wrap
    )
