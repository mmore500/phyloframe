import numpy as np
import polars as pl


def alifestd_find_root_ids_polars(
    phylogeny_df: pl.DataFrame,
) -> np.ndarray:
    """What ids have an empty `ancestor_list`?"""
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    return (
        phylogeny_df.lazy()
        .filter(pl.col("id") == pl.col("ancestor_id"))
        .select("id")
        .collect()
        .to_series()
        .to_numpy()
    )
