import os

import polars as pl


def alifestd_has_contiguous_ids_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Do organisms ids' correspond to their row number?"""
    if os.environ.get(
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS",
    ):
        return True

    return (
        phylogeny_df.lazy()
        .select((pl.col("id") == pl.int_range(0, pl.len())).all())
        .collect()
        .item()
    )
