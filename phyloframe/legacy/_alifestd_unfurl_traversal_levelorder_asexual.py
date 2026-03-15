import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_mark_node_depth_asexual import (
    _alifestd_calc_node_depth_asexual_contiguous,
    alifestd_mark_node_depth_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _alifestd_unfurl_traversal_levelorder_asexual_fast_path(
    node_depths: np.ndarray,
) -> np.ndarray:
    """Return levelorder traversal indices for contiguous, sorted phylogeny.

    Parameters
    ----------
    node_depths : np.ndarray
        Array of node depths.

    Returns
    -------
    np.ndarray
        Index array giving levelorder (BFS) traversal order.
    """
    return np.argsort(node_depths, kind="stable")


def _alifestd_unfurl_traversal_levelorder_asexual_slow_path(
    phylogeny_df: pd.DataFrame,
) -> np.ndarray:
    """Implementation detail for
    `alifestd_unfurl_traversal_levelorder_asexual`.

    Handles non-contiguous ids using pandas indexing.
    """
    phylogeny_df = alifestd_mark_node_depth_asexual(phylogeny_df, mutate=True)
    levelorder_index = np.argsort(
        phylogeny_df["node_depth"].to_numpy(),
        kind="stable",
    )
    id_loc = phylogeny_df.columns.get_loc("id")
    return phylogeny_df.iloc[levelorder_index, id_loc].to_numpy()


def alifestd_unfurl_traversal_levelorder_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List `id` values in levelorder (BFS) traversal order.

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
        if "node_depth" not in phylogeny_df.columns:
            node_depths = _alifestd_calc_node_depth_asexual_contiguous(
                ancestor_ids,
            )
        else:
            node_depths = phylogeny_df["node_depth"].to_numpy()
        return _alifestd_unfurl_traversal_levelorder_asexual_fast_path(
            node_depths,
        )
    else:
        return _alifestd_unfurl_traversal_levelorder_asexual_slow_path(
            phylogeny_df,
        )
