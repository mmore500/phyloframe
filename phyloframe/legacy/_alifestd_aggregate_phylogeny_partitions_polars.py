import logging
import typing

import polars as pl

from ._alifestd_aggregate_phylogenies_polars import (
    alifestd_aggregate_phylogenies_polars,
)
from ._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from ._alifestd_make_empty_polars import alifestd_make_empty_polars

_partition_key_prefix = "__phyloframe_partition_key_"


def alifestd_aggregate_phylogeny_partitions_polars(
    phylogeny_df: typing.Union[pl.DataFrame, pl.LazyFrame],
    partitioned_by: typing.Union[
        str,
        pl.Expr,
        typing.Sequence[typing.Union[str, pl.Expr]],
    ],
) -> pl.DataFrame:
    """Split a dataframe into partitions by the values of one or more columns
    or expressions, then concatenate the partitions as independent
    phylogenies, reassigning organism ids to prevent collisions.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition. Ids
    within each partition need not be contiguous.

    Assumes asexual phylogenies, topologically sorted, and with an
    ``ancestor_id`` column (not ``ancestor_list``).

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.
    partitioned_by : str, polars.Expr, or sequence of str and polars.Expr
        Column name(s) or expression(s) to split the dataframe over.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions :
        Pandas-based implementation.
    alifestd_aggregate_phylogenies_polars :
        Aggregates a list of separate phylogeny dataframes.
    """
    if isinstance(partitioned_by, (str, pl.Expr)):
        partitioned_by = [partitioned_by]
    else:
        partitioned_by = list(partitioned_by)
    if not partitioned_by:
        raise ValueError("partitioned_by must specify at least one column")

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
        return phylogeny_df

    key_names = [
        f"{_partition_key_prefix}{i}" for i in range(len(partitioned_by))
    ]
    keyed_df = phylogeny_df.with_columns(
        [
            (pl.col(key) if isinstance(key, str) else key).alias(name)
            for key, name in zip(partitioned_by, key_names)
        ],
    )

    logging.info(
        "- alifestd_aggregate_phylogeny_partitions_polars: " "partitioning...",
    )
    partitions = [
        alifestd_assign_contiguous_ids_polars(partition.drop(key_names))
        for partition in keyed_df.partition_by(
            key_names,
            maintain_order=True,
            include_key=True,
        )
    ]

    if not partitions:
        return alifestd_make_empty_polars()

    return alifestd_aggregate_phylogenies_polars(partitions)
