import typing

import polars as pl

from .._auxlib._preserve_id_dtypes_polars import preserve_id_dtypes_polars


@preserve_id_dtypes_polars
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
    i.e., no organism should have an ancestor in a different partition.

    Each partition's ids are shifted by a fixed offset, so resulting ids are
    unique, and are contiguous if ids within every partition are contiguous.
    Result is topologically sorted if input is topologically sorted.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with an ``ancestor_id`` column
        (not ``ancestor_list``).
    partitioned_by : str, polars.Expr, or sequence of str and polars.Expr
        Column name(s) or expression(s) to split the dataframe over.

    Returns
    -------
    polars.DataFrame
        The aggregated phylogenies, with unique organism ids.

    See Also
    --------
    alifestd_aggregate_phylogeny_partitions :
        Pandas-based implementation.
    alifestd_aggregate_phylogenies_polars :
        Aggregates a list of separate phylogeny dataframes.
    """
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_list" in schema_names:
        raise NotImplementedError(
            "ancestor_list column not supported in polars implementation",
        )
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "ancestor_id column required in polars implementation",
        )

    if isinstance(partitioned_by, (str, pl.Expr)):
        partitioned_by = [partitioned_by]
    keys = [pl.col(k) if isinstance(k, str) else k for k in partitioned_by]

    offset = pl.col("_offset").first().over("_part")
    return (
        phylogeny_df.lazy()
        .with_row_index("_row")
        # label partitions by first row, ordering them by first appearance
        .with_columns(_part=pl.col("_row").min().over(keys))
        .sort("_part", "_row")
        # id span claimed by each partition, recorded on its first row
        .with_columns(
            _span=pl.when(pl.col("_row") == pl.col("_part"))
            .then(pl.col("id").max().over("_part") + 1)
            .otherwise(0),
        )
        .with_columns(_offset=pl.col("_span").cum_sum() - pl.col("_span"))
        .with_columns(
            id=pl.col("id") + offset,
            ancestor_id=pl.col("ancestor_id") + offset,
        )
        .select(schema_names)
        .collect()
    )
