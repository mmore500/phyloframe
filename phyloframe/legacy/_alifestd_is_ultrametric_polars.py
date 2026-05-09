import polars as pl

from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars


def alifestd_is_ultrametric_polars(
    phylogeny_df: pl.DataFrame,
    *,
    atol: float = 0.0,
) -> bool:
    """Do all tips share the same `origin_time` (within ``atol``)?

    Tests the peak-to-peak (``ptp``) range of ``origin_time`` among tips
    against ``atol``. Returns ``True`` for empty phylogenies. Raises
    ``ValueError`` if any tip's ``origin_time`` is null/NaN. Must
    represent an asexual phylogeny (when ``is_leaf`` is not already
    present).
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "origin_time" not in schema_names:
        raise ValueError(
            "alifestd_is_ultrametric_polars requires 'origin_time' column",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return True

    if "is_leaf" not in schema_names:
        phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)

    leaf_stats = (
        phylogeny_df.lazy()
        .filter(pl.col("is_leaf"))
        .select(
            pl.col("origin_time").max() - pl.col("origin_time").min(),
            pl.col("origin_time").is_null().any().alias("any_null"),
            pl.col("origin_time").is_nan().any().alias("any_nan"),
        )
        .collect()
        .row(0)
    )
    leaf_ot, any_null, any_nan = leaf_stats
    if any_null or any_nan:
        raise ValueError(
            "alifestd_is_ultrametric_polars: "
            "tip 'origin_time' contains null/NaN",
        )
    return bool(leaf_ot <= atol)
