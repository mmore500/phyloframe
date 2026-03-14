import numpy as np
import polars as pl


def alifestd_find_root_ids_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """What ids have an empty `ancestor_list`?"""
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError("ancestor_id column required")

    return (
        phylogeny_df.lazy()
        .filter(pl.col("id") == pl.col("ancestor_id"))
        .select("id")
        .collect()
        .to_series()
        .to_numpy()
    )
