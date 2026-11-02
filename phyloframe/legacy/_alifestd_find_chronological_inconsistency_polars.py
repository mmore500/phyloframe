from numbers import Number
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
    *,
    epsilon: typing.Optional[Number] = None,
) -> typing.Optional[int]:
    """Return the id of a taxon with origin time preceding its parent's
    origin time plus `epsilon`, if any are present.

    Root taxa are never reported.

    See Also
    --------
    alifestd_find_chronological_inconsistency :
        Pandas-based implementation.
    alifestd_coerce_chronological_consistency_polars :
        Resolves origin time inversions by adjusting origin times.
    alifestd_graft_chronological_consistency_polars :
        Resolves origin time inversions by grafting onto nearest
        eligible ancestor node.
    alifestd_splice_chronological_consistency_polars :
        Resolves origin time inversions by grafting onto a node spliced
        into nearest eligible ancestor edge.
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
    ancestor_origin_time = pl.col("origin_time").gather(pl.col("ancestor_id"))
    if epsilon is not None:
        ancestor_origin_time += epsilon

    result = (
        phylogeny_df.lazy()
        .filter(
            (pl.col("id") != pl.col("ancestor_id"))
            & (ancestor_origin_time > pl.col("origin_time")),
        )
        .select("id")
        .limit(1)
        .collect()
    )

    if result.is_empty():
        return None
    return int(result.item())
