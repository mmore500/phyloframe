import numpy as np
import pandas as pd

from ._alifestd_mark_leaves import alifestd_mark_leaves


def alifestd_is_ultrametric(
    phylogeny_df: pd.DataFrame,
    *,
    atol: float = 0.0,
) -> bool:
    """Do all tips share the same `origin_time` (within ``atol``)?

    Tests the peak-to-peak (``ptp``) range of ``origin_time`` among tips
    against ``atol``. Returns ``True`` for empty phylogenies. Raises
    ``ValueError`` if any tip's ``origin_time`` is null/NaN.

    Input dataframe is not mutated by this operation.
    """
    if "origin_time" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_is_ultrametric requires 'origin_time' column",
        )

    if phylogeny_df.empty:
        return True

    if "is_leaf" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_leaves(phylogeny_df)

    leaf_origin_times = phylogeny_df.loc[
        phylogeny_df["is_leaf"].to_numpy(), "origin_time"
    ].to_numpy()
    if pd.isna(leaf_origin_times).any():
        raise ValueError(
            "alifestd_is_ultrametric: tip 'origin_time' contains null/NaN",
        )
    return bool(np.ptp(leaf_origin_times) <= atol)
