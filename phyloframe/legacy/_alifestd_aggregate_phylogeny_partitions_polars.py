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
) -> typing.Union[pl.DataFrame, pl.LazyFrame]:
    """Split a dataframe into partitions by the values of one or more columns
    or expressions, then aggregate the partitions as independent phylogenies,
    reassigning organism ids to prevent collisions.

    Equivalent to splitting `phylogeny_df` over `partitioned_by` and calling
    ``alifestd_aggregate_phylogenies_polars``, but performed as a single
    query plan without collecting intermediate results. A LazyFrame input
    yields a LazyFrame (nothing is collected); a DataFrame input yields a
    DataFrame.

    Partitions are ordered by first appearance of their key in
    `phylogeny_df`, and rows retain their relative order within each
    partition. Each partition is expected to be a self-contained phylogeny,
    i.e., no organism should have an ancestor in a different partition.

    Ids within each partition are shifted by a fixed offset, so resulting ids
    are unique, and are contiguous if the ids within every partition are
    contiguous (i.e., each id equals its row number within its partition).

    Assumes asexual phylogenies with an ``ancestor_id`` column (not
    ``ancestor_list``). Topological sorting is not required; the result is
    topologically sorted if and only if the input is.

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
    keys = [pl.col(k) if isinstance(k, str) else k for k in partitioned_by]

    is_eager = isinstance(phylogeny_df, pl.DataFrame)
    phylogeny_lf = phylogeny_df.lazy()
    schema_names = phylogeny_lf.collect_schema().names()

    if "ancestor_list" in schema_names:
        raise NotImplementedError(
            "ancestor_list column not supported in polars implementation",
        )
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "ancestor_id column required in polars implementation",
        )

    result = (
        phylogeny_lf.with_row_index("__row")
        # first row of each partition orders partitions by first appearance
        .with_columns(pl.col("__row").min().over(keys).alias("__part"))
        .sort("__part", "__row")
        # id span claimed by each partition, recorded on its first row
        .with_columns(
            pl.when(pl.col("__row") == pl.col("__part"))
            .then(pl.col("id").max().over("__part") + 1)
            .otherwise(0)
            .alias("__span"),
        )
        .with_columns(
            (pl.col("__span").cum_sum() - pl.col("__span")).alias("__offset"),
        )
        # shift each partition's ids past the ids of all prior partitions
        .with_columns(
            pl.col("id") + pl.col("__offset").first().over("__part"),
            pl.col("ancestor_id") + pl.col("__offset").first().over("__part"),
        )
        .select(schema_names)
    )

    return result.collect() if is_eager else result
