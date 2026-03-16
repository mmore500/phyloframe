import logging
import typing

import polars as pl

from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)


def alifestd_aggregate_phylogenies_polars(
    phylogeny_dfs: typing.List[pl.DataFrame],
) -> pl.DataFrame:
    """Concatenate independent phylogenies, reassigning organism ids to
    prevent collisions.

    Assumes asexual phylogenies with contiguous ids, topologically sorted,
    and with an ``ancestor_id`` column (not ``ancestor_list``).

    See Also
    --------
    alifestd_aggregate_phylogenies :
        Pandas-based implementation.
    """
    aggregate_least_available_id = 0

    res = []
    for phylogeny_df in phylogeny_dfs:
        phylogeny_df = phylogeny_df.lazy().collect()

        schema_names = phylogeny_df.collect_schema().names()

        if "ancestor_list" in schema_names:
            raise NotImplementedError(
                "ancestor_list column not supported in polars implementation",
            )

        if "ancestor_id" not in schema_names:
            raise NotImplementedError(
                "ancestor_id column required in polars implementation",
            )

        if phylogeny_df.is_empty():
            res.append(phylogeny_df)
            continue

        if not alifestd_has_contiguous_ids_polars(phylogeny_df):
            raise NotImplementedError(
                "non-contiguous ids not supported in polars implementation",
            )

        if not alifestd_is_topologically_sorted_polars(phylogeny_df):
            raise NotImplementedError(
                "non-topologically-sorted data not supported "
                "in polars implementation",
            )

        logging.info(
            "- alifestd_aggregate_phylogenies_polars: "
            "shifting ids for phylogeny...",
        )
        if aggregate_least_available_id:
            phylogeny_df = phylogeny_df.with_columns(
                id=pl.col("id") + aggregate_least_available_id,
                ancestor_id=pl.col("ancestor_id")
                + aggregate_least_available_id,
            )

        cur_max_id = (
            phylogeny_df.lazy().select(pl.col("id").max()).collect().item()
        )
        aggregate_least_available_id = cur_max_id + 1

        res.append(phylogeny_df)

    if not res:
        return pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        )

    logging.info(
        "- alifestd_aggregate_phylogenies_polars: concatenating results...",
    )
    return pl.concat(res, how="vertical")
