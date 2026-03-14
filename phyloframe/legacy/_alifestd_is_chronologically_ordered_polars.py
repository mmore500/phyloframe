import polars as pl

from ._alifestd_find_chronological_inconsistency_polars import (
    alifestd_find_chronological_inconsistency_polars,
)


def alifestd_is_chronologically_ordered_polars(
    phylogeny_df: pl.DataFrame,
) -> bool:
    """Check if all taxa have origin times at or after their ancestor's
    origin time.
    """
    return (
        alifestd_find_chronological_inconsistency_polars(phylogeny_df) is None
    )
