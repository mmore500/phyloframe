import polars as pl

from ._alifestd_mark_origin_time_delta_polars import (
    alifestd_mark_origin_time_delta_polars,
)


def alifestd_sum_origin_time_deltas_polars(
    phylogeny_df: pl.DataFrame,
) -> float:
    """Sum origin_time_delta values."""
    if "origin_time_delta" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_mark_origin_time_delta_polars(phylogeny_df)

    return (
        phylogeny_df.lazy()
        .select(pl.col("origin_time_delta").sum())
        .collect()
        .item()
    )
