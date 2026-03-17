import numpy as np
import pandas as pd

from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_is_topologically_sorted import alifestd_is_topologically_sorted
from ._alifestd_topological_sort import (
    _topological_sort_fast_path,
    alifestd_topological_sort,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def alifestd_unfurl_traversal_topological_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
) -> np.ndarray:
    """List `id` values in topological traversal order.

    Parents are visited before children. If the dataframe is already
    topologically sorted, the existing id order is returned directly.
    Otherwise, a topological ordering is computed.

    The provided dataframe must be asexual.

    Input dataframe is not mutated by this operation unless `mutate` set True.
    If mutate set True, operation does not occur in place; still use return
    value to get transformed phylogeny dataframe.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_is_topologically_sorted(phylogeny_df):
        return phylogeny_df["id"].to_numpy()

    if alifestd_has_contiguous_ids(phylogeny_df):
        ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
        order = _topological_sort_fast_path(ancestor_ids)
        id_loc = phylogeny_df.columns.get_loc("id")
        return phylogeny_df.iloc[order, id_loc].to_numpy()

    sorted_df = alifestd_topological_sort(phylogeny_df, mutate=True)
    return sorted_df["id"].to_numpy()
