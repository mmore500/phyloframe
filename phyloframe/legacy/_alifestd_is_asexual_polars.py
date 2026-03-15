import polars as pl

from ._alifestd_is_sexual_polars import alifestd_is_sexual_polars


def alifestd_is_asexual_polars(phylogeny_df: pl.DataFrame) -> bool:
    """Do all organisms in the phylogeny have one or no immediate ancestor?"""
    return (
        "ancestor_id" in phylogeny_df.lazy().collect_schema().names()
        or not alifestd_is_sexual_polars(phylogeny_df)
    )
