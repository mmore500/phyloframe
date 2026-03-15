import typing

import numpy as np
import pandas as pd

from ._alifestd_is_working_format_asexual import (
    alifestd_is_working_format_asexual,
)
from ._alifestd_mark_node_depth_asexual import alifestd_mark_node_depth_asexual
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_calc_mrca_id_matrix_asexual_fast_path(
    ancestor_ids: np.ndarray,
    node_depths: np.ndarray,
    progress_wrap: typing.Callable = lambda x: x,
) -> np.ndarray:
    """Shared implementation detail for
    `alifestd_calc_mrca_id_matrix_asexual` and
    `alifestd_calc_mrca_id_matrix_asexual_polars`.

    Parameters
    ----------
    ancestor_ids : np.ndarray
        1-D int64 array of ancestor ids, indexed by organism id.
        Roots are self-referential (ancestor_ids[i] == i).
    node_depths : np.ndarray
        1-D int64 array of node depths, indexed by organism id.
    progress_wrap : callable, optional
        Wrapper for progress display (e.g., tqdm).

    Returns
    -------
    np.ndarray
        n x n int64 matrix of MRCA ids.  Entry [i, j] is -1 when
        organisms i and j share no common ancestor.
    """
    n = len(ancestor_ids)
    result = -np.ones((n, n), dtype=np.int64)
    if n == 0:
        return result

    max_depth = int(node_depths.max())
    cur_positions = np.arange(n, dtype=np.int64)

    for depth in progress_wrap(reversed(range(max_depth + 1))):
        depth_mask = node_depths[cur_positions] == depth

        ansatz = -np.ones_like(result)

        ansatz[:, depth_mask] = cur_positions[depth_mask]
        ansatz[depth_mask, :] = cur_positions[depth_mask, None]
        ansatz[~depth_mask, :] = -1
        ansatz[:, ~depth_mask] = -1

        ansatz[ansatz != ansatz.T] = -1

        result = np.maximum(result, ansatz)
        cur_positions[depth_mask] = ancestor_ids[cur_positions[depth_mask]]

    return result


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

    phylogeny_df = alifestd_mark_node_depth_asexual(phylogeny_df, mutate=True)

    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy().astype(np.int64)
    node_depths = phylogeny_df["node_depth"].to_numpy().astype(np.int64)
    assert np.all(
        phylogeny_df["id"].to_numpy() == np.arange(len(phylogeny_df))
    )

    return _alifestd_calc_mrca_id_matrix_asexual_fast_path(
        ancestor_ids, node_depths, progress_wrap
    )
