import typing

import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_find_chronological_inconsistency_polars(
    phylogeny_df: pl.DataFrame,
) -> typing.Optional[int]:
    """Return the id of a taxon with origin time preceding its parent's, if
    any are present.
    """

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names or "origin_time" not in schema_names:
        return None

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return None

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "alifestd_find_chronological_inconsistency_polars requires "
            "contiguous ids",
        )

    # Use polars gather to look up ancestor origin times
    result = (
        phylogeny_df.lazy()
        .with_columns(
            ancestor_origin_time=pl.col("origin_time").gather(
                pl.col("ancestor_id"),
            ),
        )
        .filter(pl.col("ancestor_origin_time") > pl.col("origin_time"))
        .select("id")
        .limit(1)
        .collect()
    )

    if result.is_empty():
        return None
    return int(result.item())
